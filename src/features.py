"""Feature-engineering helpers for modeling work."""

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


def add_boosting_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add compact interactions that gradient boosting can exploit."""
    out = df.copy()

    if {"PM2.5_lag_1", "PM10_lag_1"}.issubset(out.columns):
        out["pm25_pm10_ratio_lag1"] = out["PM2.5_lag_1"] / (out["PM10_lag_1"].abs() + 1.0)

    if {"PM2.5_lag_1", "WSPM_lag_1"}.issubset(out.columns):
        out["pm25_low_wind_lag1"] = out["PM2.5_lag_1"] / (out["WSPM_lag_1"].abs() + 1.0)
        out["ventilation_proxy_lag1"] = out["PM2.5_lag_1"] * out["WSPM_lag_1"]

    if {"PM2.5_lag_1", "RAIN_lag_1"}.issubset(out.columns):
        out["pm25_rain_interaction_lag1"] = out["PM2.5_lag_1"] * out["RAIN_lag_1"]
        out["has_rain_lag1"] = (out["RAIN_lag_1"] > 0).astype(int)

    if {"TEMP_lag_1", "DEWP_lag_1"}.issubset(out.columns):
        out["temp_dewp_spread_lag1"] = out["TEMP_lag_1"] - out["DEWP_lag_1"]

    if {"PRES_lag_1", "PRES_lag_24"}.issubset(out.columns):
        out["pressure_change_24h"] = out["PRES_lag_1"] - out["PRES_lag_24"]

    if {"pm25_mean_3h", "pm25_mean_24h"}.issubset(out.columns):
        out["pm25_short_vs_daily"] = out["pm25_mean_3h"] - out["pm25_mean_24h"]
        out["pm25_short_daily_ratio"] = out["pm25_mean_3h"] / (out["pm25_mean_24h"].abs() + 1.0)

    if {"PM2.5_lag_1", "PM2.5_lag_2", "PM2.5_lag_3"}.issubset(out.columns):
        out["pm25_acceleration_3h"] = out["PM2.5_lag_1"] - 2 * out["PM2.5_lag_2"] + out["PM2.5_lag_3"]

    return out
