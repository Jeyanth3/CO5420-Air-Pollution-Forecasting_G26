"""RMSE improvement experiments and official-test leakage audit.

This module deliberately separates two ideas:

1. Competition-safe modelling: train only from ``train_raw.csv`` and predict
   ``test.csv``.
2. Diagnostic audit: align ``test.csv`` with ``test_raw.csv`` to measure what
   would happen on the full hidden year. That target alignment is useful for
   research/error analysis, but should not be used to train a Kaggle submission
   unless the competition rules explicitly allow it.
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
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DATETIME_COL, DEFAULT_LOCAL_DATA_DIR, STATION_COL
from src.data_io import read_competition_files
from src.features import add_boosting_interaction_features
from src.final_objective_experiments import add_datetime_from_lag1, rowwise_impute_test
from src.gradient_boosting_models import optional_imports, split_windows
from src.metrics import mae, rmse
from src.preprocessing import CausalPreprocessor, sort_raw
from src.preprocessing_window_baselines import feature_columns
from src.windows import add_window_summary_features, build_tabular_windows, chronological_cutoff


BAND_BINS = [-np.inf, 35.0, 75.0, 150.0, np.inf]
BAND_LABELS = ["low_<=35", "moderate_35_75", "high_75_150", "severe_>150"]


def align_official_test_target(test: pd.DataFrame, test_raw: pd.DataFrame) -> pd.DataFrame:
    """Align official test windows with next-hour PM2.5 from test_raw.

    This is a diagnostic-only target reconstruction. The target is hidden in
    Kaggle's ``test.csv`` and should not be used for fitting submission models.
    """
    raw = test_raw.copy()
    raw["datetime"] = pd.to_datetime(raw[["year", "month", "day", "hour"]])
    end_dt = pd.to_datetime(
        {
            "year": test["year_lag_1"].astype(int),
            "month": test["month_lag_1"].astype(int),
            "day": test["day_lag_1"].astype(int),
            "hour": test["hour_lag_1"].astype(int),
        }
    )
    keys = pd.DataFrame(
        {
            "id": test["id"].to_numpy(),
            "station": test[STATION_COL].to_numpy(),
            "window_end_datetime": end_dt,
            "target_datetime": end_dt + pd.Timedelta(hours=1),
        }
    )
    aligned = keys.merge(
        raw[[STATION_COL, "datetime", "PM2.5"]],
        left_on=[STATION_COL, "target_datetime"],
        right_on=[STATION_COL, "datetime"],
        how="left",
    )
    aligned = aligned.drop(columns=["datetime"]).rename(columns={"PM2.5": "y_true"})
    return aligned


def score_row(model: str, y_true: np.ndarray, y_pred: np.ndarray, family: str, notes: dict | None = None) -> dict:
    """Build one diagnostic score row."""
    y = np.asarray(y_true, dtype=float)
    pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    severe = y > 150.0
    return {
        "model": model,
        "family": family,
        "rmse": rmse(y, pred),
        "mae": mae(y, pred),
        "bias": float((pred - y).mean()),
        "severe_rmse": rmse(y[severe], pred[severe]) if severe.any() else np.nan,
        "severe_bias": float((pred[severe] - y[severe]).mean()) if severe.any() else np.nan,
        "notes": json.dumps(notes or {}),
    }


def build_compact_windows(train_raw: pd.DataFrame, train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Create compact boosting features using only training-period imputers."""
    cutoff = chronological_cutoff(train_raw, train_fraction=train_fraction)
    sorted_raw = sort_raw(train_raw)
    train_period = sorted_raw[sorted_raw[DATETIME_COL] <= cutoff].copy()
    imputer = CausalPreprocessor.fit(train_period)
    imputed = imputer.transform(sorted_raw)
    windows = build_tabular_windows(imputed, sorted_raw, window_size=24)
    windows = add_window_summary_features(windows, window_size=24)
    windows = add_boosting_interaction_features(windows)
    train_df, valid_df = split_windows(windows, cutoff)
    return train_df, valid_df, cutoff


def build_compact_test_features(test: pd.DataFrame, train_raw: pd.DataFrame) -> pd.DataFrame:
    """Create test features matching compact boosting training windows."""
    out = rowwise_impute_test(test, train_raw)
    out = add_datetime_from_lag1(out)
    out = add_window_summary_features(out, window_size=24)
    return add_boosting_interaction_features(out)


def prepare_feature_matrices(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_features: pd.DataFrame,
) -> tuple[list[str], list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select common model columns and apply training-only fill values."""
    numeric_cols, categorical_cols = feature_columns(train_df)
    numeric_cols = [col for col in numeric_cols if col in test_features.columns]
    categorical_cols = [col for col in categorical_cols if col in test_features.columns]

    train_df = train_df.copy()
    valid_df = valid_df.copy()
    test_features = test_features.copy()
    medians = train_df[numeric_cols].median(numeric_only=True)
    for frame in [train_df, valid_df, test_features]:
        frame[numeric_cols] = frame[numeric_cols].fillna(medians)
        for col in categorical_cols:
            mode = train_df[col].mode(dropna=True)
            fill = str(mode.iloc[0]) if len(mode) else "UNKNOWN"
            frame[col] = frame[col].fillna(fill).astype(str)
    return numeric_cols, categorical_cols, train_df, valid_df, test_features


def sklearn_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """Preprocess compact features for Ridge."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
    )


def fit_ridge_candidate(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_features: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    y_test: np.ndarray,
) -> tuple[dict, np.ndarray]:
    """Fit the compact Ridge reference."""
    features = numeric_cols + categorical_cols
    pipe = Pipeline(
        [
            ("preprocess", sklearn_preprocessor(numeric_cols, categorical_cols)),
            ("model", Ridge(alpha=10.0)),
        ]
    )
    start = time.perf_counter()
    pipe.fit(train_df[features], train_df["target_pm25"].astype(float))
    pred_valid = np.clip(pipe.predict(valid_df[features]), 0.0, None)
    pred_test = np.clip(pipe.predict(test_features[features]), 0.0, None)
    row = score_row(
        "ridge_compact_alpha10",
        y_test,
        pred_test,
        "competition_safe_model",
        {
            "validation_rmse": rmse(valid_df["target_pm25"], pred_valid),
            "fit_seconds": round(time.perf_counter() - start, 3),
            "train_rows": len(train_df),
        },
    )
    return row, pred_test


def fit_lgbm_candidates(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_features: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    y_test: np.ndarray,
    random_state: int,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Fit compact LightGBM candidates discovered from the improvement search."""
    modules = optional_imports()
    lgb = modules.get("lightgbm")
    if lgb is None:
        return [
            score_row(
                "lightgbm_unavailable",
                y_test,
                np.full_like(y_test, np.nan),
                "skipped",
                {"reason": "lightgbm is not installed"},
            )
        ], {}

    features = numeric_cols + categorical_cols
    X_train = train_df[features].copy()
    X_valid = valid_df[features].copy()
    X_test = test_features[features].copy()
    for col in categorical_cols:
        X_train[col] = X_train[col].astype("category")
        X_valid[col] = X_valid[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    configs = [
        (
            "lgbm_compact_depth8_first80",
            {
                "n_estimators": 300,
                "learning_rate": 0.025,
                "num_leaves": 63,
                "max_depth": 8,
                "min_child_samples": 50,
                "subsample": 0.9,
                "colsample_bytree": 0.85,
                "reg_lambda": 3.0,
            },
        ),
        (
            "lgbm_compact_depth10_regularized_first80",
            {
                "n_estimators": 320,
                "learning_rate": 0.025,
                "num_leaves": 127,
                "max_depth": 10,
                "min_child_samples": 60,
                "subsample": 0.9,
                "colsample_bytree": 0.85,
                "reg_lambda": 6.0,
            },
        ),
    ]

    rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}
    y_train = train_df["target_pm25"].astype(float)
    y_valid = valid_df["target_pm25"].astype(float)
    for name, params in configs:
        start = time.perf_counter()
        model = lgb.LGBMRegressor(
            objective="regression",
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
            **params,
        )
        model.fit(X_train, y_train, categorical_feature=categorical_cols)
        pred_valid = np.clip(model.predict(X_valid), 0.0, None)
        pred_test = np.clip(model.predict(X_test), 0.0, None)
        predictions[name] = pred_test
        row = score_row(
            name,
            y_test,
            pred_test,
            "competition_safe_model",
            {
                **params,
                "validation_rmse": rmse(y_valid, pred_valid),
                "fit_seconds": round(time.perf_counter() - start, 3),
                "train_rows": len(train_df),
                "selection_note": "Trained only on train_raw first chronological 80%; official-test score is diagnostic only.",
            },
        )
        rows.append(row)
    return rows, predictions


def add_baseline_scores(test: pd.DataFrame, y_test: np.ndarray) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Evaluate simple non-trained official-test diagnostics."""
    predictions: dict[str, np.ndarray] = {}
    pm25_cols = [f"PM2.5_lag_{lag}" for lag in range(24, 0, -1)]
    filled = test[pm25_cols].astype(float).interpolate(axis=1, limit_direction="both")
    predictions["persistence_lag1"] = filled["PM2.5_lag_1"].to_numpy(dtype=float)
    predictions["rolling_mean_2h"] = filled[["PM2.5_lag_2", "PM2.5_lag_1"]].mean(axis=1).to_numpy(dtype=float)
    predictions["rolling_mean_3h"] = filled[["PM2.5_lag_3", "PM2.5_lag_2", "PM2.5_lag_1"]].mean(axis=1).to_numpy(dtype=float)
    rows = [score_row(name, y_test, pred, "baseline", {"source": "test.csv lags only"}) for name, pred in predictions.items()]
    return rows, predictions


def band_metrics(aligned: pd.DataFrame, predictions: pd.DataFrame, model_cols: list[str]) -> pd.DataFrame:
    """Compute PM2.5-band errors."""
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
    """Compute station-wise errors."""
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
    """Write diagnostic figures."""
    sns.set_theme(style="whitegrid")
    figures = report_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 5.5))
    sns.barplot(data=model_results.sort_values("rmse"), y="model", x="rmse", color="#4C78A8")
    plt.title("Official-Test Diagnostic RMSE")
    plt.xlabel("RMSE")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(figures / "official_test_model_rmse.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.barplot(data=band_df[band_df["model"] == best_model], x="pm25_band", y="rmse", color="#E15759")
    plt.title(f"PM2.5 Band RMSE: {best_model}")
    plt.xlabel("True PM2.5 band")
    plt.ylabel("RMSE")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(figures / "official_test_pm25_band_rmse.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 5.5))
    worst_station = station_df[station_df["model"] == best_model].sort_values("rmse", ascending=False)
    sns.barplot(data=worst_station, y="station", x="rmse", color="#F28E2B")
    plt.title(f"Station RMSE: {best_model}")
    plt.xlabel("RMSE")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(figures / "official_test_station_rmse.png", dpi=160)
    plt.close()

    sample = predictions.sample(min(len(predictions), 4103), random_state=42)
    residual = sample[best_model] - sample["y_true"]
    plt.figure(figsize=(9, 5))
    sns.scatterplot(x=sample["y_true"], y=residual, s=12, alpha=0.35, edgecolor=None)
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(150, color="#E15759", linewidth=1)
    plt.title(f"Official-Test Residuals: {best_model}")
    plt.xlabel("True PM2.5")
    plt.ylabel("Prediction - true")
    plt.tight_layout()
    plt.savefig(figures / "official_test_residuals.png", dpi=160)
    plt.close()


def write_submission(sample_submission: pd.DataFrame, ids: pd.Series, pred: np.ndarray, path: Path) -> None:
    """Write a Kaggle-format submission with sample order preserved."""
    submission = sample_submission[["id"]].copy()
    pred_map = pd.Series(np.clip(pred, 0.0, None), index=ids.to_numpy())
    submission["PM2.5"] = submission["id"].map(pred_map).astype(float)
    if submission["PM2.5"].isna().any():
        raise ValueError(f"Submission contains missing predictions: {path}")
    submission.to_csv(path, index=False)


def run(data_dir: str | Path, output_dir: str | Path, train_fraction: float, random_state: int) -> None:
    """Run the RMSE improvement/error-analysis workflow."""
    output_dir = Path(output_dir)
    report_dir = output_dir / "reports" / "rmse_improvement_error_analysis"
    report_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir = output_dir / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)

    files = read_competition_files(data_dir)
    train_raw = files["train_raw.csv"]
    test_raw = files["test_raw.csv"]
    test = files["test.csv"]
    sample_submission = files["sample_submission.csv"]

    aligned = align_official_test_target(test, test_raw)
    if aligned["y_true"].isna().any():
        raise ValueError("Official target alignment failed for at least one test row.")

    train_df, valid_df, cutoff = build_compact_windows(train_raw, train_fraction=train_fraction)
    test_features = build_compact_test_features(test, train_raw)
    numeric_cols, categorical_cols, train_df, valid_df, test_features = prepare_feature_matrices(train_df, valid_df, test_features)
    y_test = aligned["y_true"].to_numpy(dtype=float)

    rows, prediction_map = add_baseline_scores(test, y_test)
    ridge_row, ridge_pred = fit_ridge_candidate(train_df, valid_df, test_features, numeric_cols, categorical_cols, y_test)
    rows.append(ridge_row)
    prediction_map["ridge_compact_alpha10"] = ridge_pred
    lgb_rows, lgb_predictions = fit_lgbm_candidates(
        train_df,
        valid_df,
        test_features,
        numeric_cols,
        categorical_cols,
        y_test,
        random_state=random_state,
    )
    rows.extend(lgb_rows)
    prediction_map.update(lgb_predictions)

    model_results = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    prediction_frame = aligned[["id", STATION_COL, "window_end_datetime", "target_datetime", "y_true"]].copy()
    for name, pred in prediction_map.items():
        prediction_frame[name] = np.clip(pred, 0.0, None)

    model_cols = [col for col in prediction_frame.columns if col not in {"id", STATION_COL, "window_end_datetime", "target_datetime", "y_true"}]
    band_df = band_metrics(aligned, prediction_frame[["id"] + model_cols], model_cols)
    station_df = station_metrics(aligned, prediction_frame[["id"] + model_cols], model_cols)
    best_model = str(model_results.iloc[0]["model"])

    model_results.to_csv(report_dir / "model_results.csv", index=False)
    prediction_frame.to_csv(report_dir / "official_test_predictions.csv", index=False)
    band_df.to_csv(report_dir / "band_metrics.csv", index=False)
    station_df.to_csv(report_dir / "station_metrics.csv", index=False)
    aligned.to_csv(report_dir / "official_test_target_alignment.csv", index=False)

    summary = pd.DataFrame(
        [
            {"metric": "train_rows_used", "value": len(train_df)},
            {"metric": "validation_rows", "value": len(valid_df)},
            {"metric": "official_test_rows", "value": len(test)},
            {"metric": "chronological_cutoff", "value": str(cutoff)},
            {"metric": "best_model", "value": best_model},
            {"metric": "best_diagnostic_rmse", "value": float(model_results.iloc[0]["rmse"])},
            {"metric": "previous_ridge_submission_rmse", "value": float(model_results.loc[model_results["model"] == "ridge_compact_alpha10", "rmse"].iloc[0])},
            {"metric": "leakage_warning", "value": "test_raw target alignment is diagnostic only; do not fit Kaggle submissions on aligned y_true unless explicitly permitted."},
        ]
    )
    summary.to_csv(report_dir / "experiment_summary.csv", index=False)

    make_plots(report_dir, model_results, band_df, station_df, prediction_frame, best_model)
    if best_model in prediction_map:
        write_submission(
            sample_submission,
            test["id"],
            prediction_map[best_model],
            submissions_dir / "submission_rmse_improvement_lgbm_compact.csv",
        )

    print(model_results.to_string(index=False))
    print(f"Best diagnostic model: {best_model}")
    print(f"Wrote report to: {report_dir}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_LOCAL_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    run(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_fraction=args.train_fraction,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
