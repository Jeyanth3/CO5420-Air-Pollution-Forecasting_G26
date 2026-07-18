"""Day 1 EDA, alignment validation, and persistence baseline pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.baselines import persistence_from_raw, persistence_from_test
from src.data_io import read_competition_files
from src.metrics import mae, rmse
from src.preprocessing import stationwise_time_impute
from src.windows import make_one_hour_ahead_frame, validate_raw_alignment


def build_eda_tables(train_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create compact EDA summary tables."""
    eda = pd.DataFrame(
        {
            "metric": [
                "rows",
                "columns",
                "stations",
                "year_min",
                "year_max",
                "pm25_missing",
                "pm25_mean",
                "pm25_median",
                "pm25_std",
            ],
            "value": [
                len(train_raw),
                len(train_raw.columns),
                train_raw["station"].nunique() if "station" in train_raw.columns else None,
                train_raw["year"].min() if "year" in train_raw.columns else None,
                train_raw["year"].max() if "year" in train_raw.columns else None,
                train_raw["PM2.5"].isna().sum() if "PM2.5" in train_raw.columns else None,
                train_raw["PM2.5"].mean() if "PM2.5" in train_raw.columns else None,
                train_raw["PM2.5"].median() if "PM2.5" in train_raw.columns else None,
                train_raw["PM2.5"].std() if "PM2.5" in train_raw.columns else None,
            ],
        }
    )

    missing = (
        train_raw.isna()
        .sum()
        .rename("missing_count")
        .to_frame()
        .assign(missing_fraction=lambda x: x["missing_count"] / len(train_raw))
        .reset_index(names="column")
        .sort_values("missing_count", ascending=False)
    )
    return eda, missing


def chronological_holdout_metrics(aligned: pd.DataFrame, validation_fraction: float = 0.2) -> pd.DataFrame:
    """Evaluate persistence on the last chronological block of aligned rows."""
    if aligned.empty:
        raise ValueError("No aligned one-hour-ahead rows are available for validation.")

    ordered = aligned.sort_values("datetime").reset_index(drop=True)
    split_idx = int(len(ordered) * (1 - validation_fraction))
    valid = ordered.iloc[split_idx:].copy()

    y_true = valid["target_next_pm25"].astype(float)
    y_pred = persistence_from_raw(valid)

    return pd.DataFrame(
        [
            {
                "model": "persistence_latest_pm25",
                "validation_rows": len(valid),
                "rmse": rmse(y_true, y_pred),
                "mae": mae(y_true, y_pred),
                "validation_fraction": validation_fraction,
            }
        ]
    )


def create_persistence_submission(test_df: pd.DataFrame, sample_submission: pd.DataFrame) -> pd.DataFrame:
    """Create a Kaggle-formatted persistence submission."""
    submission = sample_submission[["id"]].copy()
    submission["PM2.5"] = persistence_from_test(test_df).fillna(persistence_from_test(test_df).median())
    return submission


def run(data_dir: str | Path | None, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    reports_dir = output_dir / "reports" / "day1"
    submissions_dir = output_dir / "submissions"
    reports_dir.mkdir(parents=True, exist_ok=True)
    submissions_dir.mkdir(parents=True, exist_ok=True)

    files = read_competition_files(data_dir)
    train_raw = files.get("train_raw.csv")
    test_df = files.get("test.csv")
    sample_submission = files.get("sample_submission.csv")

    if train_raw is None:
        raise FileNotFoundError("train_raw.csv is required for Day 1 EDA and validation.")

    eda, missing = build_eda_tables(train_raw)
    eda.to_csv(reports_dir / "eda_summary.csv", index=False)
    missing.to_csv(reports_dir / "missing_values.csv", index=False)

    alignment = pd.DataFrame([validate_raw_alignment(train_raw)])
    alignment.to_csv(reports_dir / "alignment_validation.csv", index=False)

    imputed_train = stationwise_time_impute(train_raw)
    aligned = make_one_hour_ahead_frame(imputed_train)
    metrics = chronological_holdout_metrics(aligned)
    metrics.to_csv(reports_dir / "persistence_validation_metrics.csv", index=False)

    if test_df is not None and sample_submission is not None:
        submission = create_persistence_submission(test_df, sample_submission)
        submission.to_csv(submissions_dir / "submission_persistence.csv", index=False)

    print("Day 1 pipeline complete.")
    print(metrics.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None, help="Directory containing Kaggle CSV files.")
    parser.add_argument("--output-dir", default=".", help="Directory where reports/submissions are written.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.data_dir, args.output_dir)

