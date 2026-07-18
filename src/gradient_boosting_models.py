"""Gradient boosting experiments for PM2.5 one-hour-ahead forecasting."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DATETIME_COL, DEFAULT_LOCAL_DATA_DIR, STATION_COL
from src.data_io import read_competition_files
from src.features import add_boosting_interaction_features
from src.metrics import mae, rmse
from src.preprocessing import CausalPreprocessor, sort_raw
from src.preprocessing_window_baselines import feature_columns, station_metrics
from src.windows import add_window_summary_features, build_tabular_windows, chronological_cutoff


ID_COLUMNS = {"window_end_datetime", "target_datetime", "target_pm25"}
CATEGORICAL_COLUMNS = [STATION_COL, "wd_lag_1"]


def optional_imports() -> dict[str, object | None]:
    """Import optional boosting libraries without making the module fail."""
    modules: dict[str, object | None] = {}
    for name in ["lightgbm", "xgboost", "catboost"]:
        try:
            modules[name] = __import__(name)
        except Exception:
            modules[name] = None
    return modules


def build_feature_frame(train_raw: pd.DataFrame, train_fraction: float) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Create leakage-aware 24-hour windows with engineered boosting features."""
    cutoff = chronological_cutoff(train_raw, train_fraction=train_fraction)
    sorted_raw = sort_raw(train_raw)
    train_period = sorted_raw[sorted_raw[DATETIME_COL] <= cutoff].copy()

    preprocessor = CausalPreprocessor.fit(train_period)
    imputed = preprocessor.transform(train_raw)
    windows = build_tabular_windows(imputed, train_raw, window_size=24)
    windows = add_window_summary_features(windows, window_size=24)
    windows = add_boosting_interaction_features(windows)
    return windows, cutoff


def split_windows(windows: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split windows by the window-end timestamp."""
    train_mask = pd.to_datetime(windows["window_end_datetime"]) <= cutoff
    train_df = windows.loc[train_mask].copy()
    valid_df = windows.loc[~train_mask].copy()
    if train_df.empty or valid_df.empty:
        raise ValueError("Chronological split produced empty train or validation windows.")
    return train_df, valid_df


def model_features(windows: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return model feature columns."""
    numeric_cols, categorical_cols = feature_columns(windows)
    return numeric_cols, categorical_cols


def sklearn_preprocessor(numeric_cols: list[str], categorical_cols: list[str], scale: bool = False) -> ColumnTransformer:
    """Build sklearn preprocessing for models that need one-hot encoded categoricals."""
    numeric_transformer = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def evaluate(name: str, y_true: pd.Series, y_pred: np.ndarray, fit_seconds: float, train_rows: int, notes: str) -> dict:
    """Create a result row."""
    return {
        "model": name,
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "fit_seconds": round(fit_seconds, 3),
        "train_windows": train_rows,
        "notes": notes,
    }


def latest_training_subset(train_df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Use the latest chronological rows when limiting training size."""
    if max_rows <= 0 or len(train_df) <= max_rows:
        return train_df
    return train_df.sort_values("window_end_datetime").tail(max_rows).copy()


def add_baseline_predictions(valid_df: pd.DataFrame, y_valid: pd.Series) -> tuple[list[dict], pd.DataFrame]:
    """Add persistence and Ridge-comparison baseline placeholders."""
    predictions = pd.DataFrame(
        {
            "station": valid_df[STATION_COL].to_numpy(),
            "window_end_datetime": valid_df["window_end_datetime"].to_numpy(),
            "target_datetime": valid_df["target_datetime"].to_numpy(),
            "y_true": y_valid.to_numpy(),
        }
    )
    rows = []
    baseline_map = {
        "persistence_lag_1": "PM2.5_lag_1",
        "rolling_mean_3h": "pm25_mean_3h",
    }
    for model_name, col in baseline_map.items():
        pred = valid_df[col].astype(float).to_numpy()
        predictions[model_name] = pred
        rows.append(evaluate(model_name, y_valid, pred, 0.0, 0, f"Direct baseline from {col}"))
    return rows, predictions


def fit_ridge(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[dict, np.ndarray]:
    """Fit Ridge as a fast linear reference model."""
    y_train = train_df["target_pm25"].astype(float)
    y_valid = valid_df["target_pm25"].astype(float)
    features = numeric_cols + categorical_cols
    pipe = Pipeline(
        [
            ("preprocess", sklearn_preprocessor(numeric_cols, categorical_cols, scale=True)),
            ("model", Ridge(alpha=10.0, random_state=42)),
        ]
    )
    start = time.perf_counter()
    pipe.fit(train_df[features], y_train)
    pred = pipe.predict(valid_df[features])
    return evaluate("ridge_alpha_10", y_valid, pred, time.perf_counter() - start, len(train_df), "Linear reference model"), pred


def fit_lightgbm(
    lgb,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    random_state: int,
) -> tuple[list[dict], dict[str, np.ndarray], pd.DataFrame]:
    """Fit a compact LightGBM tuning grid."""
    y_train = train_df["target_pm25"].astype(float)
    y_valid = valid_df["target_pm25"].astype(float)
    features = numeric_cols + categorical_cols
    X_train = train_df[features].copy()
    X_valid = valid_df[features].copy()
    for col in categorical_cols:
        X_train[col] = X_train[col].astype("category")
        X_valid[col] = X_valid[col].astype("category")

    configs = [
        {
            "label": "lightgbm_depth8_lr03",
            "params": {
                "n_estimators": 900,
                "learning_rate": 0.03,
                "num_leaves": 63,
                "max_depth": 8,
                "min_child_samples": 50,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_lambda": 2.0,
                "objective": "regression",
                "random_state": random_state,
                "n_jobs": -1,
                "verbosity": -1,
            },
        },
        {
            "label": "lightgbm_depth10_lr02",
            "params": {
                "n_estimators": 1200,
                "learning_rate": 0.02,
                "num_leaves": 127,
                "max_depth": 10,
                "min_child_samples": 80,
                "subsample": 0.9,
                "colsample_bytree": 0.8,
                "reg_lambda": 4.0,
                "objective": "regression",
                "random_state": random_state,
                "n_jobs": -1,
                "verbosity": -1,
            },
        },
    ]

    rows = []
    preds = {}
    importances = []
    for config in configs:
        model = lgb.LGBMRegressor(**config["params"])
        start = time.perf_counter()
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(80, verbose=False)],
            categorical_feature=categorical_cols,
        )
        pred = model.predict(X_valid, num_iteration=getattr(model, "best_iteration_", None))
        rows.append(evaluate(config["label"], y_valid, pred, time.perf_counter() - start, len(train_df), json.dumps(config["params"])))
        preds[config["label"]] = pred
        importances.append(
            pd.DataFrame(
                {
                    "model": config["label"],
                    "feature": model.feature_name_,
                    "importance": model.feature_importances_,
                }
            )
        )
    return rows, preds, pd.concat(importances, ignore_index=True)


def fit_xgboost(
    xgb,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    random_state: int,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Fit XGBoost with one-hot categoricals."""
    y_train = train_df["target_pm25"].astype(float)
    y_valid = valid_df["target_pm25"].astype(float)
    features = numeric_cols + categorical_cols
    configs = [
        {
            "label": "xgboost_hist_depth6",
            "params": {
                "n_estimators": 800,
                "learning_rate": 0.035,
                "max_depth": 6,
                "min_child_weight": 8,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_lambda": 3.0,
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "random_state": random_state,
                "n_jobs": -1,
            },
        }
    ]

    rows = []
    preds = {}
    for config in configs:
        pipe = Pipeline(
            [
                ("preprocess", sklearn_preprocessor(numeric_cols, categorical_cols, scale=False)),
                ("model", xgb.XGBRegressor(**config["params"])),
            ]
        )
        start = time.perf_counter()
        pipe.fit(train_df[features], y_train)
        pred = pipe.predict(valid_df[features])
        rows.append(evaluate(config["label"], y_valid, pred, time.perf_counter() - start, len(train_df), json.dumps(config["params"])))
        preds[config["label"]] = pred
    return rows, preds


def fit_catboost(
    cb,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    random_state: int,
) -> tuple[list[dict], dict[str, np.ndarray], pd.DataFrame]:
    """Fit CatBoost using native categorical handling."""
    y_train = train_df["target_pm25"].astype(float)
    y_valid = valid_df["target_pm25"].astype(float)
    features = numeric_cols + categorical_cols
    X_train = train_df[features].copy()
    X_valid = valid_df[features].copy()
    for col in categorical_cols:
        X_train[col] = X_train[col].astype(str)
        X_valid[col] = X_valid[col].astype(str)

    configs = [
        {
            "label": "catboost_depth8_lr03",
            "params": {
                "iterations": 900,
                "learning_rate": 0.03,
                "depth": 8,
                "l2_leaf_reg": 6.0,
                "loss_function": "RMSE",
                "eval_metric": "RMSE",
                "random_seed": random_state,
                "verbose": False,
                "allow_writing_files": False,
                "early_stopping_rounds": 80,
                "thread_count": -1,
            },
        }
    ]

    rows = []
    preds = {}
    importances = []
    for config in configs:
        model = cb.CatBoostRegressor(**config["params"])
        start = time.perf_counter()
        model.fit(
            X_train,
            y_train,
            eval_set=(X_valid, y_valid),
            cat_features=categorical_cols,
            use_best_model=True,
        )
        pred = model.predict(X_valid)
        rows.append(evaluate(config["label"], y_valid, pred, time.perf_counter() - start, len(train_df), json.dumps(config["params"])))
        preds[config["label"]] = pred
        importances.append(
            pd.DataFrame(
                {
                    "model": config["label"],
                    "feature": features,
                    "importance": model.get_feature_importance(),
                }
            )
        )
    return rows, preds, pd.concat(importances, ignore_index=True)


def weighted_ensemble(predictions: pd.DataFrame, model_cols: list[str], y_valid: pd.Series) -> tuple[dict, np.ndarray]:
    """Grid-search a simple non-negative ensemble over the strongest models."""
    candidates = [col for col in model_cols if col not in {"rolling_mean_3h"}]
    candidates = sorted(candidates, key=lambda col: rmse(y_valid, predictions[col]))[:4]
    if len(candidates) < 2:
        pred = predictions[candidates[0]].to_numpy()
        return evaluate("weighted_ensemble", y_valid, pred, 0.0, 0, f"Single candidate fallback: {candidates}"), pred

    weight_values = np.linspace(0.0, 1.0, 11)
    best_score = float("inf")
    best_weights = None
    best_pred = None
    values = predictions[candidates].to_numpy(dtype=float)

    def gen_weights(n: int, remaining: float = 1.0):
        if n == 1:
            yield [remaining]
        else:
            for w in weight_values:
                if w <= remaining + 1e-12:
                    for rest in gen_weights(n - 1, remaining - w):
                        yield [w] + rest

    for weights in gen_weights(len(candidates)):
        if abs(sum(weights) - 1.0) > 1e-9:
            continue
        pred = values @ np.asarray(weights)
        score = rmse(y_valid, pred)
        if score < best_score:
            best_score = score
            best_weights = weights
            best_pred = pred

    notes = json.dumps(dict(zip(candidates, best_weights)))
    return evaluate("weighted_ensemble", y_valid, best_pred, 0.0, 0, notes), best_pred


def run(
    data_dir: str | Path,
    output_dir: str | Path,
    train_fraction: float,
    max_boost_train_rows: int,
    random_state: int,
) -> None:
    """Run gradient boosting experiments and write reports."""
    output_dir = Path(output_dir)
    report_dir = output_dir / "reports" / "gradient_boosting"
    report_dir.mkdir(parents=True, exist_ok=True)

    modules = optional_imports()
    files = read_competition_files(data_dir)
    train_raw = files.get("train_raw.csv")
    if train_raw is None:
        raise FileNotFoundError("train_raw.csv is required for gradient boosting experiments.")

    windows, cutoff = build_feature_frame(train_raw, train_fraction=train_fraction)
    train_df, valid_df = split_windows(windows, cutoff)
    boost_train_df = latest_training_subset(train_df, max_boost_train_rows)
    numeric_cols, categorical_cols = model_features(windows)
    y_valid = valid_df["target_pm25"].astype(float)

    results, predictions = add_baseline_predictions(valid_df, y_valid)
    ridge_result, ridge_pred = fit_ridge(train_df, valid_df, numeric_cols, categorical_cols)
    results.append(ridge_result)
    predictions["ridge_alpha_10"] = ridge_pred

    importance_frames = []

    if modules["lightgbm"] is not None:
        lgb_rows, lgb_preds, lgb_importance = fit_lightgbm(
            modules["lightgbm"], boost_train_df, valid_df, numeric_cols, categorical_cols, random_state
        )
        results.extend(lgb_rows)
        for name, pred in lgb_preds.items():
            predictions[name] = pred
        importance_frames.append(lgb_importance)

    if modules["xgboost"] is not None:
        xgb_rows, xgb_preds = fit_xgboost(
            modules["xgboost"], boost_train_df, valid_df, numeric_cols, categorical_cols, random_state
        )
        results.extend(xgb_rows)
        for name, pred in xgb_preds.items():
            predictions[name] = pred

    if modules["catboost"] is not None:
        cb_rows, cb_preds, cb_importance = fit_catboost(
            modules["catboost"], boost_train_df, valid_df, numeric_cols, categorical_cols, random_state
        )
        results.extend(cb_rows)
        for name, pred in cb_preds.items():
            predictions[name] = pred
        importance_frames.append(cb_importance)

    model_cols = [
        col
        for col in predictions.columns
        if col not in {"station", "window_end_datetime", "target_datetime", "y_true"}
    ]
    ensemble_result, ensemble_pred = weighted_ensemble(predictions, model_cols, y_valid)
    predictions["weighted_ensemble"] = ensemble_pred
    results.append(ensemble_result)
    model_cols.append("weighted_ensemble")

    results_df = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    per_station = station_metrics(predictions, model_cols)
    summary = pd.DataFrame(
        [
            {
                "train_fraction": train_fraction,
                "cutoff": cutoff,
                "raw_rows": len(train_raw),
                "window_rows": len(windows),
                "train_windows": len(train_df),
                "boost_train_windows": len(boost_train_df),
                "validation_windows": len(valid_df),
                "numeric_feature_columns": len(numeric_cols),
                "categorical_feature_columns": len(categorical_cols),
                "available_libraries": json.dumps({name: module is not None for name, module in modules.items()}),
                "random_state": random_state,
            }
        ]
    )

    summary.to_csv(report_dir / "experiment_summary.csv", index=False)
    results_df.to_csv(report_dir / "model_results.csv", index=False)
    per_station.to_csv(report_dir / "station_model_results.csv", index=False)
    predictions.to_csv(report_dir / "validation_predictions.csv", index=False)
    if importance_frames:
        feature_importance = pd.concat(importance_frames, ignore_index=True)
        feature_importance.sort_values(["model", "importance"], ascending=[True, False]).to_csv(
            report_dir / "feature_importance.csv", index=False
        )

    print("Gradient boosting experiments complete.")
    print(summary.to_string(index=False))
    print(results_df.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_LOCAL_DATA_DIR), help="Directory containing train_raw.csv.")
    parser.add_argument("--output-dir", default=".", help="Directory where reports are written.")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Chronological training fraction.")
    parser.add_argument(
        "--max-boost-train-rows",
        type=int,
        default=180_000,
        help="Latest chronological rows used for boosting models. Use 0 for all training windows.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_fraction=args.train_fraction,
        max_boost_train_rows=args.max_boost_train_rows,
        random_state=args.random_state,
    )

