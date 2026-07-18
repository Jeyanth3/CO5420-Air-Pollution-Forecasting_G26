"""Window and target-alignment utilities."""

from __future__ import annotations

import pandas as pd

from src.config import DATETIME_COL, STATION_COL, TARGET
from src.preprocessing import sort_raw


def make_one_hour_ahead_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Create aligned rows where current PM2.5 predicts next-hour PM2.5."""
    out = sort_raw(df)
    out["target_next_pm25"] = out.groupby(STATION_COL)[TARGET].shift(-1)
    out["next_datetime"] = out.groupby(STATION_COL)[DATETIME_COL].shift(-1)
    out["hours_to_target"] = (out["next_datetime"] - out[DATETIME_COL]).dt.total_seconds() / 3600.0
    aligned = out[(out["hours_to_target"] == 1.0) & out["target_next_pm25"].notna()].copy()
    return aligned.reset_index(drop=True)


def validate_raw_alignment(df: pd.DataFrame) -> dict[str, float | int]:
    """Return basic alignment diagnostics for one-hour-ahead forecasting."""
    sorted_df = sort_raw(df)
    candidate_rows = len(sorted_df) - sorted_df[STATION_COL].nunique()
    aligned = make_one_hour_ahead_frame(sorted_df)
    return {
        "raw_rows": int(len(sorted_df)),
        "stations": int(sorted_df[STATION_COL].nunique()),
        "candidate_one_step_rows": int(candidate_rows),
        "valid_one_hour_rows": int(len(aligned)),
        "dropped_nonconsecutive_or_missing_target_rows": int(candidate_rows - len(aligned)),
    }

