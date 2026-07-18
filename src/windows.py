"""Window and target-alignment utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DATETIME_COL, RAW_NUMERIC_FEATURES, STATION_COL, TARGET
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


def chronological_cutoff(df: pd.DataFrame, train_fraction: float = 0.8) -> pd.Timestamp:
    """Choose a chronological cutoff using unique timestamps."""
    sorted_df = sort_raw(df)
    unique_times = sorted_df[DATETIME_COL].drop_duplicates().sort_values().reset_index(drop=True)
    if len(unique_times) < 2:
        raise ValueError("Need at least two unique timestamps for a chronological split.")
    cutoff_index = max(0, min(len(unique_times) - 2, int(len(unique_times) * train_fraction) - 1))
    return pd.Timestamp(unique_times.iloc[cutoff_index])


def build_tabular_windows(
    imputed_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    window_size: int = 24,
    numeric_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Build flattened 24-hour windows with a raw one-hour-ahead PM2.5 target.

    The feature columns come from `imputed_df`; the validation target comes from
    the original raw PM2.5 values so we never evaluate against imputed labels.
    """
    if window_size < 2:
        raise ValueError("window_size must be at least 2.")

    numeric_cols = numeric_cols or [col for col in RAW_NUMERIC_FEATURES if col in imputed_df.columns]
    imputed = sort_raw(imputed_df)
    raw = sort_raw(raw_df)

    if len(imputed) != len(raw):
        raise ValueError("imputed_df and raw_df must have the same number of rows.")

    lag_feature_names = [
        f"{col}_lag_{lag}"
        for lag in range(window_size, 0, -1)
        for col in numeric_cols
    ]

    frames = []
    for station, station_features in imputed.groupby(STATION_COL, sort=False):
        station_raw = raw.loc[station_features.index]
        station_features = station_features.reset_index(drop=True)
        station_raw = station_raw.reset_index(drop=True)

        n_rows = len(station_features)
        if n_rows <= window_size:
            continue

        feature_values = station_features[numeric_cols].to_numpy(dtype=float)
        windows = np.lib.stride_tricks.sliding_window_view(
            feature_values,
            window_shape=window_size,
            axis=0,
        )
        windows = np.moveaxis(windows, -1, 1)
        windows = windows[: n_rows - window_size].reshape(n_rows - window_size, -1)

        target_positions = np.arange(window_size, n_rows)
        end_positions = target_positions - 1

        datetimes = station_features[DATETIME_COL]
        diffs = datetimes.diff().dt.total_seconds().div(3600).eq(1.0).fillna(False)
        continuity = diffs.astype(int).rolling(window=window_size, min_periods=window_size).sum()
        valid_continuity = continuity.iloc[target_positions].to_numpy() == window_size

        targets = station_raw[TARGET].iloc[target_positions].to_numpy(dtype=float)
        target_is_observed = ~pd.isna(targets)
        valid_mask = valid_continuity & target_is_observed

        if not valid_mask.any():
            continue

        frame = pd.DataFrame(windows[valid_mask], columns=lag_feature_names)
        frame[STATION_COL] = station
        if "wd" in station_features.columns:
            frame["wd_lag_1"] = station_features["wd"].iloc[end_positions].to_numpy()[valid_mask]

        end_dt = station_features[DATETIME_COL].iloc[end_positions].to_numpy()[valid_mask]
        target_dt = station_raw[DATETIME_COL].iloc[target_positions].to_numpy()[valid_mask]
        frame["window_end_datetime"] = end_dt
        frame["target_datetime"] = target_dt
        frame["target_pm25"] = targets[valid_mask]
        frames.append(frame)

    if not frames:
        raise ValueError("No valid windows were created.")

    return pd.concat(frames, ignore_index=True)


def add_window_summary_features(windows: pd.DataFrame, window_size: int = 24) -> pd.DataFrame:
    """Add compact lag-window summaries used by classical ML baselines."""
    out = windows.copy()

    pm25_cols = [f"{TARGET}_lag_{lag}" for lag in range(window_size, 0, -1)]
    available_pm25 = [col for col in pm25_cols if col in out.columns]
    if len(available_pm25) == window_size:
        latest = f"{TARGET}_lag_1"
        earliest = f"{TARGET}_lag_{window_size}"
        out["pm25_latest"] = out[latest]
        for k in [3, 6, 12, 24]:
            cols = [f"{TARGET}_lag_{lag}" for lag in range(k, 0, -1)]
            out[f"pm25_mean_{k}h"] = out[cols].mean(axis=1)
        out["pm25_std_24h"] = out[available_pm25].std(axis=1)
        out["pm25_min_24h"] = out[available_pm25].min(axis=1)
        out["pm25_max_24h"] = out[available_pm25].max(axis=1)
        out["pm25_trend_24h"] = out[latest] - out[earliest]

    for col in ["TEMP", "PRES", "DEWP", "RAIN", "WSPM", "PM10", "NO2", "CO", "O3"]:
        lag_cols = [f"{col}_lag_{lag}" for lag in range(window_size, 0, -1)]
        lag_cols = [lag_col for lag_col in lag_cols if lag_col in out.columns]
        if len(lag_cols) == window_size:
            out[f"{col.lower()}_latest"] = out[f"{col}_lag_1"]
            out[f"{col.lower()}_mean_24h"] = out[lag_cols].mean(axis=1)
            out[f"{col.lower()}_trend_24h"] = out[f"{col}_lag_1"] - out[f"{col}_lag_{window_size}"]

    end_dt = pd.to_datetime(out["window_end_datetime"])
    out["hour"] = end_dt.dt.hour
    out["month"] = end_dt.dt.month
    out["dayofweek"] = end_dt.dt.dayofweek
    out["is_weekend"] = out["dayofweek"].isin([5, 6]).astype(int)
    out["sin_hour"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["cos_hour"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["sin_month"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["cos_month"] = np.cos(2 * np.pi * out["month"] / 12.0)

    return out
