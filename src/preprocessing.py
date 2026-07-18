"""Preprocessing helpers for raw hourly air-quality data."""

from __future__ import annotations

import pandas as pd

from src.config import DATETIME_COL, RAW_NUMERIC_FEATURES, RAW_TIME_COLS, STATION_COL


def add_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Create a timestamp column from raw year/month/day/hour columns."""
    missing = [col for col in RAW_TIME_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing raw time columns: {missing}")

    out = df.copy()
    out[DATETIME_COL] = pd.to_datetime(out[RAW_TIME_COLS])
    return out


def sort_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Sort raw rows by station and timestamp."""
    out = add_datetime(df) if DATETIME_COL not in df.columns else df.copy()
    return out.sort_values([STATION_COL, DATETIME_COL]).reset_index(drop=True)


def stationwise_time_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Simple Day 1 imputation: station ffill/bfill with global median fallback."""
    out = sort_raw(df)
    numeric_cols = [col for col in RAW_NUMERIC_FEATURES if col in out.columns]

    out[numeric_cols] = out.groupby(STATION_COL, group_keys=False)[numeric_cols].ffill()
    out[numeric_cols] = out.groupby(STATION_COL, group_keys=False)[numeric_cols].bfill()

    medians = out[numeric_cols].median(numeric_only=True)
    out[numeric_cols] = out[numeric_cols].fillna(medians)

    if "wd" in out.columns:
        out["wd"] = out.groupby(STATION_COL, group_keys=False)["wd"].ffill()
        out["wd"] = out.groupby(STATION_COL, group_keys=False)["wd"].bfill()
        mode = out["wd"].mode(dropna=True)
        out["wd"] = out["wd"].fillna(mode.iloc[0] if len(mode) else "UNKNOWN")

    return out

