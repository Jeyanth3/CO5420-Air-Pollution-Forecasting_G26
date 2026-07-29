"""Final legal candidates using the friend-feature LightGBM pipeline.

This script deliberately avoids ``test_raw.csv``. It uses the improved feature
set from ``Jeyanth_analysis`` and trains final candidates from legal
``train_raw.csv`` windows only. The purpose is to improve beyond the reproduced
friend submission by using more of the public training period and model
averaging, without selecting anything from hidden official-test targets.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATETIME_COL, DEFAULT_LOCAL_DATA_DIR
from src.data_io import read_competition_files
from src.features import add_boosting_interaction_features
from src.final_objective_experiments import add_datetime_from_lag1, rowwise_impute_test
from src.gradient_boosting_models import optional_imports
from src.metrics import mae, rmse
from src.preprocessing import CausalPreprocessor, sort_raw
from src.preprocessing_window_baselines import feature_columns
from src.rmse_improvement_error_analysis import (
    build_compact_test_features,
    build_compact_windows,
    prepare_feature_matrices,
)
from src.windows import add_window_summary_features, build_tabular_windows


def add_city_context_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add leak-free same-time city aggregate features.

    These features use only measurements available at the input-window end time
    across stations. They do not use target-hour PM2.5.
    """
    if "window_end_datetime" not in frame.columns or "PM2.5_lag_1" not in frame.columns:
        return frame

    out = frame.copy()
    dt = "window_end_datetime"
    out[dt] = pd.to_datetime(out[dt])
    grouped = out.groupby(dt, sort=False)
    out["city_station_count"] = grouped["PM2.5_lag_1"].transform("count").astype(float)

    aggregate_specs = {
        "PM2.5": [1, 2, 3, 6, 24],
        "PM10": [1],
        "NO2": [1],
        "CO": [1],
        "O3": [1],
        "WSPM": [1],
        "TEMP": [1],
        "PRES": [1],
    }
    for feature, lags in aggregate_specs.items():
        for lag in lags:
            col = f"{feature}_lag_{lag}"
            if col not in out.columns:
                continue
            safe_name = feature.lower().replace(".", "")
            prefix = f"city_{safe_name}_lag{lag}"
            values = out[col].astype(float)
            count = grouped[col].transform("count").astype(float)
            total = grouped[col].transform("sum").astype(float)
            city_mean = grouped[col].transform("mean").astype(float)
            out[f"{prefix}_mean"] = city_mean
            out[f"{prefix}_max"] = grouped[col].transform("max").astype(float)
            out[f"{prefix}_std"] = grouped[col].transform("std").fillna(0.0).astype(float)
            denominator = (count - 1.0).replace(0.0, np.nan)
            other_mean = ((total - values) / denominator).fillna(city_mean)
            out[f"{prefix}_other_mean"] = other_mean
            out[f"{prefix}_station_minus_city"] = values - city_mean
            out[f"{prefix}_station_minus_other"] = values - other_mean

    if {"city_pm25_lag1_mean", "city_pm25_lag2_mean"}.issubset(out.columns):
        out["city_pm25_mean_delta_1h"] = out["city_pm25_lag1_mean"] - out["city_pm25_lag2_mean"]
    if {"city_pm25_lag1_other_mean", "city_pm25_lag2_other_mean"}.issubset(out.columns):
        out["city_pm25_other_mean_delta_1h"] = out["city_pm25_lag1_other_mean"] - out["city_pm25_lag2_other_mean"]
    if {"city_pm25_lag1_max", "city_pm25_lag1_mean"}.issubset(out.columns):
        out["city_pm25_peak_spread_lag1"] = out["city_pm25_lag1_max"] - out["city_pm25_lag1_mean"]
    return out


def build_all_train_features(train_raw: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Build friend-feature matrices from the full legal training period."""
    sorted_raw = sort_raw(train_raw)
    imputer = CausalPreprocessor.fit(sorted_raw)
    imputed = imputer.transform(sorted_raw)
    windows = build_tabular_windows(imputed, sorted_raw, window_size=24)
    windows = add_window_summary_features(windows, window_size=24)
    train_df = add_boosting_interaction_features(windows)
    train_df = add_city_context_features(train_df)

    test_features = rowwise_impute_test(test, train_raw)
    test_features = add_datetime_from_lag1(test_features)
    test_features = add_window_summary_features(test_features, window_size=24)
    test_features = add_boosting_interaction_features(test_features)
    test_features = add_city_context_features(test_features)

    numeric_cols, categorical_cols = feature_columns(train_df)
    numeric_cols = [col for col in numeric_cols if col in test_features.columns]
    categorical_cols = [col for col in categorical_cols if col in test_features.columns]

    train_df = train_df.copy()
    test_features = test_features.copy()
    medians = train_df[numeric_cols].median(numeric_only=True)
    train_df[numeric_cols] = train_df[numeric_cols].fillna(medians)
    test_features[numeric_cols] = test_features[numeric_cols].fillna(medians)
    for col in categorical_cols:
        mode = train_df[col].mode(dropna=True)
        fill = str(mode.iloc[0]) if len(mode) else "UNKNOWN"
        train_df[col] = train_df[col].fillna(fill).astype(str)
        test_features[col] = test_features[col].fillna(fill).astype(str)

    return train_df, test_features, numeric_cols, categorical_cols


def fit_lgbm(lgb, name: str, train_df: pd.DataFrame, test_features: pd.DataFrame, features: list[str], categorical_cols: list[str], params: dict) -> tuple[str, np.ndarray, dict]:
    """Fit one LightGBM candidate and return test predictions."""
    X_train = train_df[features].copy()
    X_test = test_features[features].copy()
    for col in categorical_cols:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    start = time.perf_counter()
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, train_df["target_pm25"].astype(float), categorical_feature=categorical_cols)
    pred = np.clip(model.predict(X_test), 0.0, None)
    notes = {**params, "fit_seconds": round(time.perf_counter() - start, 3), "train_rows": len(train_df)}
    return name, pred, notes


def write_submission(sample_submission: pd.DataFrame, pred: np.ndarray, path: Path) -> None:
    """Write a Kaggle-format submission."""
    out = sample_submission[["id"]].copy()
    out["PM2.5"] = np.clip(np.asarray(pred, dtype=float), 0.0, None)
    if out.shape != (4103, 2):
        raise ValueError(f"Unexpected submission shape for {path}: {out.shape}")
    if out["PM2.5"].isna().any():
        raise ValueError(f"Missing predictions in {path}")
    out.to_csv(path, index=False)


def validation_reference(data_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Run a train-only validation table for context, without using test_raw."""
    files = read_competition_files(data_dir)
    train_raw = files["train_raw.csv"]
    rows = []
    lgb = optional_imports().get("lightgbm")
    if lgb is None:
        return pd.DataFrame()
    configs = {
        "depth8": {
            "objective": "regression",
            "n_estimators": 300,
            "learning_rate": 0.025,
            "num_leaves": 63,
            "max_depth": 8,
            "min_child_samples": 50,
            "subsample": 0.9,
            "colsample_bytree": 0.85,
            "reg_lambda": 3.0,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        },
        "depth10": {
            "objective": "regression",
            "n_estimators": 320,
            "learning_rate": 0.025,
            "num_leaves": 127,
            "max_depth": 10,
            "min_child_samples": 60,
            "subsample": 0.9,
            "colsample_bytree": 0.85,
            "reg_lambda": 6.0,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        },
    }
    for fraction in [0.75, 0.80, 0.85]:
        train_df, valid_df, cutoff = build_compact_windows(train_raw, train_fraction=fraction)
        test_features = build_compact_test_features(files["test.csv"], train_raw)
        train_df = add_city_context_features(train_df)
        valid_df = add_city_context_features(valid_df)
        test_features = add_city_context_features(test_features)
        numeric_cols, categorical_cols, train_df, valid_df, test_features = prepare_feature_matrices(train_df, valid_df, test_features)
        features = numeric_cols + categorical_cols
        X_train = train_df[features].copy()
        X_valid = valid_df[features].copy()
        for col in categorical_cols:
            X_train[col] = X_train[col].astype("category")
            X_valid[col] = X_valid[col].astype("category")
        y_train = train_df["target_pm25"].astype(float)
        y_valid = valid_df["target_pm25"].astype(float)
        for label, params in configs.items():
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train, categorical_feature=categorical_cols)
            pred = np.clip(model.predict(X_valid), 0.0, None)
            rows.append(
                {
                    "candidate": f"{label}_frac{fraction:.2f}",
                    "fraction": fraction,
                    "cutoff": str(cutoff),
                    "validation_rmse": rmse(y_valid, pred),
                    "validation_mae": mae(y_valid, pred),
                    "train_rows": len(train_df),
                    "valid_rows": len(valid_df),
                }
            )
    out = pd.DataFrame(rows).sort_values("validation_rmse").reset_index(drop=True)
    report_dir = output_dir / "reports" / "friend_feature_final_candidates"
    report_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(report_dir / "validation_reference.csv", index=False)
    return out


def run(data_dir: str | Path, output_dir: str | Path) -> None:
    """Train all-legal final candidates and write Kaggle CSVs."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    report_dir = output_dir / "reports" / "friend_feature_final_candidates"
    submissions_dir = output_dir / "submissions"
    report_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir.mkdir(parents=True, exist_ok=True)

    files = read_competition_files(data_dir)
    train_raw = files["train_raw.csv"]
    test = files["test.csv"]
    sample_submission = files["sample_submission.csv"]

    lgb = optional_imports().get("lightgbm")
    if lgb is None:
        raise ImportError("lightgbm is required.")

    train_df, test_features, numeric_cols, categorical_cols = build_all_train_features(train_raw, test)
    features = numeric_cols + categorical_cols

    base_depth10 = {
        "objective": "regression",
        "n_estimators": 320,
        "learning_rate": 0.025,
        "num_leaves": 127,
        "max_depth": 10,
        "min_child_samples": 60,
        "subsample": 0.9,
        "colsample_bytree": 0.85,
        "reg_lambda": 6.0,
        "n_jobs": -1,
        "verbosity": -1,
    }
    base_depth8 = {
        "objective": "regression",
        "n_estimators": 300,
        "learning_rate": 0.025,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 50,
        "subsample": 0.9,
        "colsample_bytree": 0.85,
        "reg_lambda": 3.0,
        "n_jobs": -1,
        "verbosity": -1,
    }

    specs = [
        ("friend_city_context_depth10_all_train", {**base_depth10, "random_state": 42}),
        ("friend_city_context_depth8_all_train", {**base_depth8, "random_state": 42}),
        ("friend_city_context_depth10_bag_seed22", {**base_depth10, "random_state": 22, "subsample_freq": 1}),
        ("friend_city_context_depth10_bag_seed77", {**base_depth10, "random_state": 77, "subsample_freq": 1}),
        ("friend_city_context_depth10_bag_seed2026", {**base_depth10, "random_state": 2026, "subsample_freq": 1}),
    ]

    predictions: dict[str, np.ndarray] = {}
    rows = []
    for name, params in specs:
        print(f"Training {name}...", flush=True)
        model_name, pred, notes = fit_lgbm(lgb, name, train_df, test_features, features, categorical_cols, params)
        predictions[model_name] = pred
        rows.append(
            {
                "candidate": model_name,
                "rows": len(pred),
                "mean": float(np.mean(pred)),
                "std": float(np.std(pred)),
                "min": float(np.min(pred)),
                "max": float(np.max(pred)),
                "notes": json.dumps(notes),
            }
        )
        write_submission(sample_submission, pred, submissions_dir / f"submission_{model_name}.csv")

    bag_cols = [
        "friend_city_context_depth10_all_train",
        "friend_city_context_depth10_bag_seed22",
        "friend_city_context_depth10_bag_seed77",
        "friend_city_context_depth10_bag_seed2026",
    ]
    predictions["friend_city_context_depth10_bag_mean_all_train"] = np.mean([predictions[col] for col in bag_cols], axis=0)
    predictions["friend_city_context_depth10_depth8_blend_all_train"] = (
        0.85 * predictions["friend_city_context_depth10_all_train"] + 0.15 * predictions["friend_city_context_depth8_all_train"]
    )

    previous_path = submissions_dir / "submission_friend_features_lgbm_depth10.csv"
    if previous_path.exists():
        previous = pd.read_csv(previous_path)["PM2.5"].to_numpy(dtype=float)
        predictions["friend_plus_city_context_70_30_blend"] = 0.70 * previous + 0.30 * predictions["friend_city_context_depth10_all_train"]

    for name in [
        "friend_city_context_depth10_bag_mean_all_train",
        "friend_city_context_depth10_depth8_blend_all_train",
        "friend_plus_city_context_70_30_blend",
    ]:
        if name not in predictions:
            continue
        pred = predictions[name]
        rows.append(
            {
                "candidate": name,
                "rows": len(pred),
                "mean": float(np.mean(pred)),
                "std": float(np.std(pred)),
                "min": float(np.min(pred)),
                "max": float(np.max(pred)),
                "notes": json.dumps({"source": "legal blend of train_raw-only model predictions"}),
            }
        )
        write_submission(sample_submission, pred, submissions_dir / f"submission_{name}.csv")

    summary = pd.DataFrame(rows)
    summary.to_csv(report_dir / "candidate_summary.csv", index=False)
    validation_reference(data_dir, output_dir)

    report = """# Friend-Feature Final Candidates

## Purpose

These candidates try to improve beyond the reproduced friend submission without
using hidden official-test labels. They reuse the improved feature set and train
final LightGBM models from legal `train_raw.csv` windows only.

## Primary Candidates

- `submission_friend_city_context_depth10_all_train.csv`
- `submission_friend_city_context_depth10_bag_mean_all_train.csv`
- `submission_friend_plus_city_context_70_30_blend.csv`

The first one is the direct attempt to improve the friend feature model with city context by using
all legal training windows. The bagged and 70/30 blend files are safer
robustness probes.
"""
    (report_dir / "friend_feature_final_candidates_learning.md").write_text(report)
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_LOCAL_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    run(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
