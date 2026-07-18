"""Preprocessing, window generation, and baseline model experiments."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DATETIME_COL, DEFAULT_LOCAL_DATA_DIR, STATION_COL
from src.data_io import read_competition_files
from src.metrics import mae, rmse
from src.preprocessing import CausalPreprocessor, sort_raw
from src.windows import add_window_summary_features, build_tabular_windows, chronological_cutoff


ID_COLUMNS = {"window_end_datetime", "target_datetime", "target_pm25"}
CATEGORICAL_COLUMNS = [STATION_COL, "wd_lag_1"]


def split_raw_by_cutoff(train_raw: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split raw rows using the chronological cutoff timestamp."""
    sorted_raw = sort_raw(train_raw)
    train_part = sorted_raw[sorted_raw[DATETIME_COL] <= cutoff].copy()
    future_part = sorted_raw[sorted_raw[DATETIME_COL] > cutoff].copy()
    return train_part, future_part


def feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric and categorical model feature columns."""
    categorical = [col for col in CATEGORICAL_COLUMNS if col in frame.columns]
    excluded = ID_COLUMNS | set(categorical)
    numeric = [
        col for col in frame.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(frame[col])
    ]
    return numeric, categorical


def make_model_preprocessor(numeric_cols: list[str], categorical_cols: list[str], scale_numeric: bool) -> ColumnTransformer:
    """Build a sklearn preprocessing transformer for tabular models."""
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def evaluate_predictions(name: str, y_true: pd.Series, y_pred: np.ndarray, extra: dict | None = None) -> dict:
    """Return a standard metrics row."""
    row = {
        "model": name,
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
    }
    if extra:
        row.update(extra)
    return row


def train_and_evaluate_models(
    windows: pd.DataFrame,
    cutoff: pd.Timestamp,
    max_tree_train_rows: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train baselines/classical models and return metrics plus predictions."""
    train_mask = pd.to_datetime(windows["window_end_datetime"]) <= cutoff
    train_df = windows.loc[train_mask].copy()
    valid_df = windows.loc[~train_mask].copy()

    if train_df.empty or valid_df.empty:
        raise ValueError("Chronological split produced empty train or validation windows.")

    y_train = train_df["target_pm25"].astype(float)
    y_valid = valid_df["target_pm25"].astype(float)

    results = []
    predictions = pd.DataFrame(
        {
            "station": valid_df[STATION_COL].to_numpy(),
            "window_end_datetime": valid_df["window_end_datetime"].to_numpy(),
            "target_datetime": valid_df["target_datetime"].to_numpy(),
            "y_true": y_valid.to_numpy(),
        }
    )

    baseline_specs = {
        "persistence_lag_1": "PM2.5_lag_1",
        "rolling_mean_3h": "pm25_mean_3h",
        "rolling_mean_6h": "pm25_mean_6h",
        "rolling_mean_12h": "pm25_mean_12h",
        "rolling_mean_24h": "pm25_mean_24h",
    }

    for name, column in baseline_specs.items():
        pred = valid_df[column].astype(float).to_numpy()
        predictions[name] = pred
        results.append(
            evaluate_predictions(
                name,
                y_valid,
                pred,
                {
                    "train_windows": len(train_df),
                    "validation_windows": len(valid_df),
                    "fit_seconds": 0.0,
                    "notes": f"Direct baseline from {column}",
                },
            )
        )

    numeric_cols, categorical_cols = feature_columns(windows)
    X_train = train_df[numeric_cols + categorical_cols]
    X_valid = valid_df[numeric_cols + categorical_cols]

    model_specs = [
        (
            "ridge_alpha_10",
            Ridge(alpha=10.0, random_state=random_state),
            True,
            len(train_df),
            "All training windows; scaled numeric features",
        ),
        (
            "random_forest",
            RandomForestRegressor(
                n_estimators=120,
                max_depth=22,
                min_samples_leaf=3,
                max_features="sqrt",
                n_jobs=-1,
                random_state=random_state,
                bootstrap=True,
                max_samples=0.75,
            ),
            False,
            max_tree_train_rows,
            "Recent chronological subset for feasible tree training",
        ),
        (
            "extra_trees",
            ExtraTreesRegressor(
                n_estimators=160,
                max_depth=None,
                min_samples_leaf=2,
                max_features="sqrt",
                n_jobs=-1,
                random_state=random_state,
                bootstrap=True,
                max_samples=0.75,
            ),
            False,
            max_tree_train_rows,
            "Recent chronological subset for feasible tree training",
        ),
    ]

    for name, estimator, scale_numeric, row_limit, notes in model_specs:
        if row_limit and len(train_df) > row_limit:
            train_subset = train_df.tail(row_limit)
            y_subset = train_subset["target_pm25"].astype(float)
            X_subset = train_subset[numeric_cols + categorical_cols]
        else:
            X_subset = X_train
            y_subset = y_train

        pipeline = Pipeline(
            steps=[
                ("preprocess", make_model_preprocessor(numeric_cols, categorical_cols, scale_numeric)),
                ("model", estimator),
            ]
        )
        start = time.perf_counter()
        pipeline.fit(X_subset, y_subset)
        fit_seconds = time.perf_counter() - start
        pred = pipeline.predict(X_valid)
        predictions[name] = pred
        results.append(
            evaluate_predictions(
                name,
                y_valid,
                pred,
                {
                    "train_windows": len(X_subset),
                    "validation_windows": len(valid_df),
                    "fit_seconds": round(fit_seconds, 3),
                    "notes": notes,
                },
            )
        )

    results_df = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    return results_df, predictions


def station_metrics(predictions: pd.DataFrame, model_columns: list[str]) -> pd.DataFrame:
    """Compute station-wise RMSE and MAE for each model."""
    rows = []
    for model in model_columns:
        for station, group in predictions.groupby("station"):
            rows.append(
                {
                    "model": model,
                    "station": station,
                    "rows": len(group),
                    "rmse": rmse(group["y_true"], group[model]),
                    "mae": mae(group["y_true"], group[model]),
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "rmse"]).reset_index(drop=True)


def run(
    data_dir: str | Path,
    output_dir: str | Path,
    train_fraction: float,
    max_tree_train_rows: int,
    random_state: int,
) -> None:
    """Execute preprocessing/window baseline experiments."""
    output_dir = Path(output_dir)
    report_dir = output_dir / "reports" / "preprocessing_window_baselines"
    report_dir.mkdir(parents=True, exist_ok=True)

    files = read_competition_files(data_dir)
    train_raw = files.get("train_raw.csv")
    if train_raw is None:
        raise FileNotFoundError("train_raw.csv is required for preprocessing/window baseline experiments.")

    cutoff = chronological_cutoff(train_raw, train_fraction=train_fraction)
    train_period, _ = split_raw_by_cutoff(train_raw, cutoff)

    preprocessor = CausalPreprocessor.fit(train_period)
    imputed = preprocessor.transform(train_raw)
    windows = build_tabular_windows(imputed, train_raw, window_size=24)
    windows = add_window_summary_features(windows, window_size=24)

    results, predictions = train_and_evaluate_models(
        windows=windows,
        cutoff=cutoff,
        max_tree_train_rows=max_tree_train_rows,
        random_state=random_state,
    )
    model_cols = [col for col in predictions.columns if col not in {"station", "window_end_datetime", "target_datetime", "y_true"}]
    per_station = station_metrics(predictions, model_cols)

    summary = pd.DataFrame(
        [
            {
                "train_fraction": train_fraction,
                "cutoff": cutoff,
                "raw_rows": len(train_raw),
                "window_rows": len(windows),
                "train_windows": int((pd.to_datetime(windows["window_end_datetime"]) <= cutoff).sum()),
                "validation_windows": int((pd.to_datetime(windows["window_end_datetime"]) > cutoff).sum()),
                "numeric_feature_columns": len(feature_columns(windows)[0]),
                "categorical_feature_columns": len(feature_columns(windows)[1]),
                "max_tree_train_rows": max_tree_train_rows,
                "random_state": random_state,
            }
        ]
    )

    summary.to_csv(report_dir / "window_split_summary.csv", index=False)
    results.to_csv(report_dir / "model_results.csv", index=False)
    per_station.to_csv(report_dir / "station_model_results.csv", index=False)
    predictions.to_csv(report_dir / "validation_predictions.csv", index=False)

    print("Preprocessing/window baseline experiments complete.")
    print(summary.to_string(index=False))
    print(results.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_LOCAL_DATA_DIR), help="Directory containing train_raw.csv.")
    parser.add_argument("--output-dir", default=".", help="Directory where reports are written.")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Chronological train fraction.")
    parser.add_argument(
        "--max-tree-train-rows",
        type=int,
        default=160_000,
        help="Use the latest N training windows for tree models to keep runtime practical.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_fraction=args.train_fraction,
        max_tree_train_rows=args.max_tree_train_rows,
        random_state=args.random_state,
    )
