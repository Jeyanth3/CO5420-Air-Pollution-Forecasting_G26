"""Feature-engineering placeholders for Day 2+ work."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_cyclic_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclic encodings for hour and month when those columns exist."""
    out = df.copy()
    if "hour" in out.columns:
        out["sin_hour"] = np.sin(2 * np.pi * out["hour"] / 24.0)
        out["cos_hour"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    if "month" in out.columns:
        out["sin_month"] = np.sin(2 * np.pi * out["month"] / 12.0)
        out["cos_month"] = np.cos(2 * np.pi * out["month"] / 12.0)
    return out

