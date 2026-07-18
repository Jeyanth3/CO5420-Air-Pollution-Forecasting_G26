"""Baseline forecasting methods."""

from __future__ import annotations

import pandas as pd

from src.config import TARGET


def persistence_from_raw(aligned_raw: pd.DataFrame) -> pd.Series:
    """Predict next-hour PM2.5 as the latest observed PM2.5."""
    return aligned_raw[TARGET].astype(float)


def persistence_from_test(test_df: pd.DataFrame) -> pd.Series:
    """Predict Kaggle test rows using the most recent PM2.5 lag."""
    possible_cols = ["PM2.5_lag_1", "PM2.5_lag_01"]
    for col in possible_cols:
        if col in test_df.columns:
            return test_df[col].astype(float)

    lag_cols = [
        col for col in test_df.columns
        if col.startswith("PM2.5") and col.endswith("_lag_1")
    ]
    if not lag_cols:
        raise ValueError("Could not find PM2.5_lag_1 in test.csv.")
    return test_df[lag_cols[0]].astype(float)

