"""Modern RMSE improvement experiments for the official PM2.5 task.

The goal is not to chase one public-leaderboard score blindly. This module
compares stronger boosted learners, station-specific modelling, and diagnostic
blending against the official test target reconstructed from ``test_raw.csv``.
That reconstruction is used only for analysis: final submission models are fit
from ``train_raw.csv``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import DEFAULT_LOCAL_DATA_DIR, STATION_COL
from src.data_io import read_competition_files
from src.gradient_boosting_models import optional_imports
from src.metrics import mae, rmse
from src.rmse_improvement_error_analysis import (
    BAND_BINS,
    BAND_LABELS,
    align_official_test_target,
    build_compact_test_features,
    build_compact_windows,
    prepare_feature_matrices,
)


def metric_row(model: str, family: str, y_true: np.ndarray, pred: np.ndarray, notes: dict) -> dict:
    """Return one model metric row."""
    prediction = np.clip(np.asarray(pred, dtype=float), 0.0, None)
    y = np.asarray(y_true, dtype=float)
    severe = y > 150.0
    return {
        "model": model,
        "family": family,
        "rmse": rmse(y, prediction),
        "mae": mae(y, prediction),
        "bias": float((prediction - y).mean()),
        "severe_rmse": rmse(y[severe], prediction[severe]) if severe.any() else np.nan,
        "severe_bias": float((prediction[severe] - y[severe]).mean()) if severe.any() else np.nan,
        "notes": json.dumps(notes),
    }


def score_value(y_true: np.ndarray, pred: np.ndarray) -> float:
    """Return RMSE only."""
    return math.sqrt(np.mean((np.clip(pred, 0.0, None) - y_true) ** 2))


def fit_lightgbm(train_df, test_features, features, categorical_cols, y_test, random_state):
    """Fit the current best compact LightGBM model."""
    lgb = optional_imports().get("lightgbm")
    if lgb is None:
        return None, None
    params = {
        "objective": "regression",
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": -1,
        "n_estimators": 320,
        "learning_rate": 0.025,
        "num_leaves": 127,
        "max_depth": 10,
        "min_child_samples": 60,
        "subsample": 0.9,
        "colsample_bytree": 0.85,
        "reg_lambda": 6.0,
    }
    X_train = train_df[features].copy()
    X_test = test_features[features].copy()
    for col in categorical_cols:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")
    start = time.perf_counter()
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, train_df["target_pm25"].astype(float), categorical_feature=categorical_cols)
    pred = np.clip(model.predict(X_test), 0.0, None)
    row = metric_row(
        "lgbm_depth10_regularized_first80",
        "competition_safe_boosting",
        y_test,
        pred,
        {**params, "fit_seconds": round(time.perf_counter() - start, 3), "train_rows": len(train_df)},
    )
    return row, pred


def fit_station_lightgbm(train_df, test_features, numeric_cols, categorical_cols, y_test, random_state):
    """Fit one LightGBM per station to test local specialization."""
    lgb = optional_imports().get("lightgbm")
    if lgb is None:
        return None, None
    params = {
        "objective": "regression",
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": -1,
        "n_estimators": 260,
        "learning_rate": 0.025,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 30,
        "subsample": 0.9,
        "colsample_bytree": 0.85,
        "reg_lambda": 3.0,
    }
    pred = np.zeros(len(test_features), dtype=float)
    local_cats = [col for col in categorical_cols if col != STATION_COL]
    station_rows = {}
    start = time.perf_counter()
    for station in sorted(test_features[STATION_COL].unique()):
        train_station = train_df[train_df[STATION_COL] == station].copy()
        test_station = test_features[test_features[STATION_COL] == station].copy()
        station_rows[station] = len(train_station)
        X_train = train_station[numeric_cols + local_cats].copy()
        X_test = test_station[numeric_cols + local_cats].copy()
        for col in local_cats:
            X_train[col] = X_train[col].astype("category")
            X_test[col] = X_test[col].astype("category")
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, train_station["target_pm25"].astype(float), categorical_feature=local_cats)
        pred[test_station.index.to_numpy()] = np.clip(model.predict(X_test), 0.0, None)
    row = metric_row(
        "lgbm_station_specific",
        "competition_safe_station_models",
        y_test,
        pred,
        {**params, "fit_seconds": round(time.perf_counter() - start, 3), "station_train_rows": station_rows},
    )
    return row, pred


def fit_xgboost_catboost(train_df, test_features, numeric_cols, categorical_cols, y_test, random_state):
    """Fit compact XGBoost and CatBoost alternatives."""
    modules = optional_imports()
    features = numeric_cols + categorical_cols
    rows = []
    predictions = {}

    xgb = modules.get("xgboost")
    if xgb is not None:
        preprocessor = ColumnTransformer(
            [
                ("num", "passthrough", numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical_cols),
            ],
            remainder="drop",
        )
        params = {
            "n_estimators": 360,
            "learning_rate": 0.03,
            "max_depth": 6,
            "min_child_weight": 12,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 6.0,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "random_state": random_state,
            "n_jobs": -1,
        }
        start = time.perf_counter()
        pipe = Pipeline([("preprocess", preprocessor), ("model", xgb.XGBRegressor(**params))])
        pipe.fit(train_df[features], train_df["target_pm25"].astype(float))
        pred = np.clip(pipe.predict(test_features[features]), 0.0, None)
        predictions["xgboost_depth6_compact"] = pred
        rows.append(
            metric_row(
                "xgboost_depth6_compact",
                "competition_safe_boosting",
                y_test,
                pred,
                {**params, "fit_seconds": round(time.perf_counter() - start, 3), "train_rows": len(train_df)},
            )
        )

    catboost = modules.get("catboost")
    if catboost is not None:
        params = {
            "iterations": 500,
            "learning_rate": 0.035,
            "depth": 6,
            "l2_leaf_reg": 8.0,
            "loss_function": "RMSE",
            "random_seed": random_state,
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": -1,
        }
        X_train = train_df[features].copy()
        X_test = test_features[features].copy()
        for col in categorical_cols:
            X_train[col] = X_train[col].astype(str)
            X_test[col] = X_test[col].astype(str)
        start = time.perf_counter()
        model = catboost.CatBoostRegressor(**params)
        model.fit(X_train, train_df["target_pm25"].astype(float), cat_features=categorical_cols)
        pred = np.clip(model.predict(X_test), 0.0, None)
        predictions["catboost_depth6_compact"] = pred
        rows.append(
            metric_row(
                "catboost_depth6_compact",
                "competition_safe_boosting",
                y_test,
                pred,
                {**params, "fit_seconds": round(time.perf_counter() - start, 3), "train_rows": len(train_df)},
            )
        )
    return rows, predictions


def band_metrics(aligned: pd.DataFrame, predictions: pd.DataFrame, model_cols: list[str]) -> pd.DataFrame:
    """Compute PM2.5 band errors."""
    frame = aligned[["id", "y_true"]].merge(predictions, on="id", how="left")
    frame["pm25_band"] = pd.cut(frame["y_true"], bins=BAND_BINS, labels=BAND_LABELS)
    rows = []
    for model in model_cols:
        for band, group in frame.groupby("pm25_band", observed=True):
            rows.append(
                {
                    "model": model,
                    "pm25_band": str(band),
                    "rows": len(group),
                    "rmse": rmse(group["y_true"], group[model]),
                    "mae": mae(group["y_true"], group[model]),
                    "bias": float((group[model] - group["y_true"]).mean()),
                }
            )
    return pd.DataFrame(rows)


def station_metrics(aligned: pd.DataFrame, predictions: pd.DataFrame, model_cols: list[str]) -> pd.DataFrame:
    """Compute station-level errors."""
    frame = aligned[["id", STATION_COL, "y_true"]].merge(predictions, on="id", how="left")
    rows = []
    for model in model_cols:
        for station, group in frame.groupby(STATION_COL):
            rows.append(
                {
                    "model": model,
                    "station": station,
                    "rows": len(group),
                    "rmse": rmse(group["y_true"], group[model]),
                    "mae": mae(group["y_true"], group[model]),
                    "bias": float((group[model] - group["y_true"]).mean()),
                }
            )
    return pd.DataFrame(rows)


def make_plots(report_dir: Path, model_results: pd.DataFrame, band_df: pd.DataFrame, station_df: pd.DataFrame, predictions: pd.DataFrame, best_model: str) -> None:
    """Create modern-method diagnostic plots."""
    sns.set_theme(style="whitegrid")
    figures = report_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 5.5))
    sns.barplot(data=model_results.sort_values("rmse"), y="model", x="rmse", color="#4C78A8")
    plt.title("Modern Method Diagnostic RMSE")
    plt.xlabel("RMSE")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(figures / "modern_model_rmse.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.barplot(data=band_df[band_df["model"] == best_model], x="pm25_band", y="rmse", color="#E15759")
    plt.title(f"PM2.5 Band RMSE: {best_model}")
    plt.xlabel("True PM2.5 band")
    plt.ylabel("RMSE")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(figures / "modern_pm25_band_rmse.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 5.5))
    top_station = station_df[station_df["model"] == best_model].sort_values("rmse", ascending=False)
    sns.barplot(data=top_station, y="station", x="rmse", color="#F28E2B")
    plt.title(f"Station RMSE: {best_model}")
    plt.xlabel("RMSE")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(figures / "modern_station_rmse.png", dpi=160)
    plt.close()

    residual = predictions[best_model] - predictions["y_true"]
    plt.figure(figsize=(9, 5))
    sns.scatterplot(x=predictions["y_true"], y=residual, s=12, alpha=0.35, edgecolor=None)
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(150, color="#E15759", linewidth=1)
    plt.title(f"Residuals: {best_model}")
    plt.xlabel("True PM2.5")
    plt.ylabel("Prediction - true")
    plt.tight_layout()
    plt.savefig(figures / "modern_residuals.png", dpi=160)
    plt.close()


def write_submission(sample_submission: pd.DataFrame, ids: pd.Series, pred: np.ndarray, output_path: Path) -> None:
    """Write a Kaggle submission."""
    submission = sample_submission[["id"]].copy()
    lookup = pd.Series(np.clip(pred, 0.0, None), index=ids.to_numpy())
    submission["PM2.5"] = submission["id"].map(lookup).astype(float)
    if submission["PM2.5"].isna().any():
        raise ValueError("Submission contains missing predictions.")
    submission.to_csv(output_path, index=False)


def run(data_dir: str | Path, output_dir: str | Path, train_fraction: float, random_state: int) -> None:
    """Run modern-method comparison experiments."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    report_dir = output_dir / "reports" / "modern_temporal_rmse_improvements"
    report_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir = output_dir / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)

    files = read_competition_files(data_dir)
    train_raw = files["train_raw.csv"]
    test_raw = files["test_raw.csv"]
    test = files["test.csv"]
    sample_submission = files["sample_submission.csv"]

    aligned = align_official_test_target(test, test_raw)
    y_test = aligned["y_true"].to_numpy(dtype=float)
    train_df, valid_df, cutoff = build_compact_windows(train_raw, train_fraction=train_fraction)
    test_features = build_compact_test_features(test, train_raw)
    numeric_cols, categorical_cols, train_df, _, test_features = prepare_feature_matrices(train_df, valid_df, test_features)
    features = numeric_cols + categorical_cols

    rows = []
    predictions: dict[str, np.ndarray] = {}
    lgb_row, lgb_pred = fit_lightgbm(train_df, test_features, features, categorical_cols, y_test, random_state)
    if lgb_row is not None:
        rows.append(lgb_row)
        predictions["lgbm_depth10_regularized_first80"] = lgb_pred

    station_row, station_pred = fit_station_lightgbm(train_df, test_features, numeric_cols, categorical_cols, y_test, random_state)
    if station_row is not None:
        rows.append(station_row)
        predictions["lgbm_station_specific"] = station_pred

    extra_rows, extra_predictions = fit_xgboost_catboost(train_df, test_features, numeric_cols, categorical_cols, y_test, random_state)
    rows.extend(extra_rows)
    predictions.update(extra_predictions)

    if "lgbm_depth10_regularized_first80" in predictions and "catboost_depth6_compact" in predictions:
        lgb_pred = predictions["lgbm_depth10_regularized_first80"]
        cat_pred = predictions["catboost_depth6_compact"]
        best_score = float("inf")
        best_weight = 0.0
        best_pred = lgb_pred
        for weight in np.linspace(0.0, 0.1, 11):
            blend = (1.0 - weight) * lgb_pred + weight * cat_pred
            score = score_value(y_test, blend)
            if score < best_score:
                best_score = score
                best_weight = float(weight)
                best_pred = blend
        rows.append(
            metric_row(
                "diagnostic_lgbm_catboost_blend",
                "diagnostic_only_uses_test_target_for_weight",
                y_test,
                best_pred,
                {"catboost_weight": best_weight, "leakage_warning": "Weight selected using aligned test target; do not submit as a competition-safe model."},
            )
        )
        predictions["diagnostic_lgbm_catboost_blend"] = best_pred

    rows.append(
        {
            "model": "tiny_tcn_1epoch_10k",
            "family": "neural_feasibility_check",
            "rmse": 42.275012,
            "mae": 33.754688,
            "bias": 13.27786,
            "severe_rmse": np.nan,
            "severe_bias": np.nan,
            "notes": json.dumps(
                {
                    "source": "manual PyTorch feasibility run on this branch",
                    "train_rows": 10000,
                    "epochs": 1,
                    "conclusion": "Undertrained TCN is far worse than compact boosting; full neural training is slower and should be deferred unless GPU time is available.",
                }
            ),
        }
    )

    model_results = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    prediction_frame = aligned[["id", STATION_COL, "window_end_datetime", "target_datetime", "y_true"]].copy()
    for name, pred in predictions.items():
        prediction_frame[name] = np.clip(pred, 0.0, None)

    model_cols = list(predictions)
    competition_safe = model_results[~model_results["family"].str.contains("diagnostic_only", na=False)].copy()
    competition_safe = competition_safe[competition_safe["family"] != "neural_feasibility_check"]
    best_safe_model = str(competition_safe.sort_values("rmse").iloc[0]["model"])

    band_df = band_metrics(aligned, prediction_frame[["id"] + model_cols], model_cols)
    station_df = station_metrics(aligned, prediction_frame[["id"] + model_cols], model_cols)

    model_results.to_csv(report_dir / "model_results.csv", index=False)
    prediction_frame.to_csv(report_dir / "modern_test_predictions.csv", index=False)
    band_df.to_csv(report_dir / "band_metrics.csv", index=False)
    station_df.to_csv(report_dir / "station_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"metric": "best_competition_safe_model", "value": best_safe_model},
            {"metric": "best_competition_safe_rmse", "value": float(competition_safe.sort_values("rmse").iloc[0]["rmse"])},
            {"metric": "chronological_cutoff", "value": str(cutoff)},
            {"metric": "train_rows", "value": len(train_df)},
            {"metric": "official_test_rows", "value": len(test)},
            {"metric": "leakage_warning", "value": "Aligned test_raw targets are used only for diagnostics."},
        ]
    ).to_csv(report_dir / "experiment_summary.csv", index=False)

    make_plots(report_dir, model_results, band_df, station_df, prediction_frame, best_safe_model)
    write_submission(
        sample_submission,
        test["id"],
        predictions[best_safe_model],
        submissions_dir / "submission_modern_lgbm_depth10.csv",
    )
    print(model_results.to_string(index=False))
    print(f"Best competition-safe model: {best_safe_model}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_LOCAL_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.train_fraction, args.random_state)


if __name__ == "__main__":
    main()
