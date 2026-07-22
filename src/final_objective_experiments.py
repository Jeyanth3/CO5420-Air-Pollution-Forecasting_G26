"""Final objective experiments and official Kaggle submission generation."""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.errors import PerformanceWarning
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DATETIME_COL, DEFAULT_LOCAL_DATA_DIR, RAW_NUMERIC_FEATURES, STATION_COL
from src.data_io import read_competition_files
from src.gradient_boosting_models import latest_training_subset, optional_imports, split_windows
from src.metrics import mae, rmse
from src.preprocessing import CausalPreprocessor, sort_raw
from src.windows import build_tabular_windows, chronological_cutoff


KEY_COLUMNS = ["station", "window_end_datetime", "target_datetime", "y_true"]
TARGET_COL = "target_pm25"
BAND_BINS = [-np.inf, 35.0, 75.0, 150.0, np.inf]
BAND_LABELS = ["low_<=35", "moderate_35_75", "high_75_150", "severe_>150"]
WIND_DIRECTION_DEGREES = {
    "N": 0.0,
    "NNE": 22.5,
    "NE": 45.0,
    "ENE": 67.5,
    "E": 90.0,
    "ESE": 112.5,
    "SE": 135.0,
    "SSE": 157.5,
    "S": 180.0,
    "SSW": 202.5,
    "SW": 225.0,
    "WSW": 247.5,
    "W": 270.0,
    "WNW": 292.5,
    "NW": 315.0,
    "NNW": 337.5,
}

warnings.filterwarnings("ignore", category=PerformanceWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


@dataclass
class FittedModel:
    """Model wrapper with its feature columns."""

    name: str
    estimator: object
    features: list[str]
    categorical_cols: list[str]
    notes: dict


def lag_cols(feature: str, window_size: int = 24) -> list[str]:
    """Return lag columns in chronological order, earliest to latest."""
    return [f"{feature}_lag_{lag}" for lag in range(window_size, 0, -1)]


def add_datetime_from_lag1(frame: pd.DataFrame) -> pd.DataFrame:
    """Add window-end datetime to official flattened test windows."""
    out = frame.copy()
    time_cols = ["year_lag_1", "month_lag_1", "day_lag_1", "hour_lag_1"]
    if all(col in out.columns for col in time_cols):
        out["window_end_datetime"] = pd.to_datetime(
            {
                "year": out["year_lag_1"].astype(int),
                "month": out["month_lag_1"].astype(int),
                "day": out["day_lag_1"].astype(int),
                "hour": out["hour_lag_1"].astype(int),
            }
        )
    return out


def numeric_categoricals(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return model-ready numeric and categorical columns."""
    categorical = [col for col in [STATION_COL, "wd_lag_1", "station_wd_lag1"] if col in frame.columns]
    excluded = set(KEY_COLUMNS) | {"id", TARGET_COL, "window_end_datetime", "target_datetime"} | set(categorical)
    numeric = [col for col in frame.columns if col not in excluded and pd.api.types.is_numeric_dtype(frame[col])]
    return numeric, categorical


def model_preprocessor(numeric_cols: list[str], categorical_cols: list[str], scale_numeric: bool) -> ColumnTransformer:
    """Build preprocessing transformer for sklearn-compatible models."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler() if scale_numeric else "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def add_rich_lag_features(frame: pd.DataFrame, window_size: int = 24) -> pd.DataFrame:
    """Objective 1: add rich rolling, trend, slope, and EWMA features."""
    out = frame.copy()
    for feature in RAW_NUMERIC_FEATURES:
        cols = [col for col in lag_cols(feature, window_size) if col in out.columns]
        if len(cols) != window_size:
            continue
        values = out[cols].astype(float)
        latest = f"{feature}_lag_1"
        out[f"{feature.lower()}_latest"] = out[latest]
        for size in [3, 6, 12, 24]:
            size_cols = [f"{feature}_lag_{lag}" for lag in range(size, 0, -1)]
            size_values = out[size_cols].astype(float)
            prefix = f"{feature.lower()}_{size}h"
            out[f"{prefix}_mean"] = size_values.mean(axis=1)
            out[f"{prefix}_std"] = size_values.std(axis=1)
            out[f"{prefix}_min"] = size_values.min(axis=1)
            out[f"{prefix}_max"] = size_values.max(axis=1)
            if size in {6, 12, 24}:
                out[f"{prefix}_median"] = size_values.median(axis=1)
            if size == 24:
                out[f"{prefix}_q25"] = size_values.quantile(0.25, axis=1)
                out[f"{prefix}_q75"] = size_values.quantile(0.75, axis=1)

            x = np.arange(size, dtype=float)
            x = x - x.mean()
            denom = float((x**2).sum())
            out[f"{feature.lower()}_{size}h_slope"] = size_values.to_numpy(dtype=float) @ x / denom

        if f"{feature}_lag_2" in out.columns:
            out[f"{feature.lower()}_delta_1h"] = out[f"{feature}_lag_1"] - out[f"{feature}_lag_2"]
        if f"{feature}_lag_4" in out.columns:
            out[f"{feature.lower()}_delta_3h"] = out[f"{feature}_lag_1"] - out[f"{feature}_lag_4"]
        if f"{feature}_lag_7" in out.columns:
            out[f"{feature.lower()}_delta_6h"] = out[f"{feature}_lag_1"] - out[f"{feature}_lag_7"]
        if all(f"{feature}_lag_{lag}" in out.columns for lag in [1, 2, 3]):
            out[f"{feature.lower()}_acceleration_3h"] = (
                out[f"{feature}_lag_1"] - 2 * out[f"{feature}_lag_2"] + out[f"{feature}_lag_3"]
            )

        arr = values.to_numpy(dtype=float)
        for alpha in [0.2, 0.4, 0.6]:
            weights = np.array([(1.0 - alpha) ** i for i in range(window_size - 1, -1, -1)], dtype=float)
            weights = weights / weights.sum()
            out[f"{feature.lower()}_ewma_a{str(alpha).replace('.', '')}"] = arr @ weights
    return out


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add cyclic calendar features from the window end."""
    out = add_datetime_from_lag1(frame)
    if "window_end_datetime" in out.columns:
        end_dt = pd.to_datetime(out["window_end_datetime"])
        out["hour"] = end_dt.dt.hour
        out["month"] = end_dt.dt.month
        out["dayofweek"] = end_dt.dt.dayofweek
        out["is_weekend"] = out["dayofweek"].isin([5, 6]).astype(int)
    elif "hour_lag_1" in out.columns:
        out["hour"] = out["hour_lag_1"]
        out["month"] = out["month_lag_1"]
    if "hour" in out.columns:
        out["sin_hour"] = np.sin(2 * np.pi * out["hour"] / 24.0)
        out["cos_hour"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    if "month" in out.columns:
        out["sin_month"] = np.sin(2 * np.pi * out["month"] / 12.0)
        out["cos_month"] = np.cos(2 * np.pi * out["month"] / 12.0)
    return out


def add_interaction_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Objective 2: add weather/pollution interactions."""
    out = frame.copy()

    def ratio(name: str, numerator: str, denominator: str) -> None:
        if numerator in out.columns and denominator in out.columns:
            out[name] = out[numerator] / (out[denominator].abs() + 1.0)

    ratio("pm25_pm10_ratio_lag1", "PM2.5_lag_1", "PM10_lag_1")
    ratio("pm25_co_ratio_lag1", "PM2.5_lag_1", "CO_lag_1")
    ratio("no2_o3_ratio_lag1", "NO2_lag_1", "O3_lag_1")
    ratio("so2_no2_ratio_lag1", "SO2_lag_1", "NO2_lag_1")
    ratio("pm25_low_wind_lag1", "PM2.5_lag_1", "WSPM_lag_1")
    ratio("pm10_low_wind_lag1", "PM10_lag_1", "WSPM_lag_1")
    if {"PM2.5_lag_1", "WSPM_lag_1"}.issubset(out.columns):
        out["pm25_wind_ventilation_lag1"] = out["PM2.5_lag_1"] * out["WSPM_lag_1"]
        out["low_wind_flag_lag1"] = (out["WSPM_lag_1"] <= 1.0).astype(int)
    if {"PM2.5_lag_1", "RAIN_lag_1"}.issubset(out.columns):
        out["rain_last_1h"] = out["RAIN_lag_1"]
        out["rain_flag_lag1"] = (out["RAIN_lag_1"] > 0).astype(int)
        out["pm25_rain_interaction_lag1"] = out["PM2.5_lag_1"] * out["rain_flag_lag1"]
    for size in [3, 6, 24]:
        rain_cols = [f"RAIN_lag_{lag}" for lag in range(size, 0, -1)]
        if all(col in out.columns for col in rain_cols):
            out[f"rain_sum_{size}h"] = out[rain_cols].sum(axis=1)
    if {"TEMP_lag_1", "DEWP_lag_1"}.issubset(out.columns):
        out["temp_dewp_spread_lag1"] = out["TEMP_lag_1"] - out["DEWP_lag_1"]
    if {"WSPM_lag_1", "wd_lag_1"}.issubset(out.columns):
        degrees = out["wd_lag_1"].map(WIND_DIRECTION_DEGREES).fillna(0.0).astype(float)
        radians = np.deg2rad(degrees)
        out["wind_x_lag1"] = out["WSPM_lag_1"] * np.cos(radians)
        out["wind_y_lag1"] = out["WSPM_lag_1"] * np.sin(radians)
    if {STATION_COL, "wd_lag_1"}.issubset(out.columns):
        out["station_wd_lag1"] = out[STATION_COL].astype(str) + "_" + out["wd_lag_1"].astype(str)
    return out


def advanced_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply Objective 1 and Objective 2 feature engineering."""
    out = add_calendar_features(frame)
    out = add_rich_lag_features(out)
    out = add_interaction_features(out)
    return out.replace([np.inf, -np.inf], np.nan)


def rowwise_impute_test(test: pd.DataFrame, train_raw: pd.DataFrame) -> pd.DataFrame:
    """Impute flattened official test windows without using test targets."""
    out = test.copy()
    medians = train_raw[[col for col in RAW_NUMERIC_FEATURES if col in train_raw.columns]].median(numeric_only=True)
    for feature in RAW_NUMERIC_FEATURES:
        cols = [col for col in lag_cols(feature) if col in out.columns]
        if not cols:
            continue
        interpolated = out[cols].astype(float).interpolate(axis=1, limit_direction="both")
        out[cols] = interpolated.fillna(float(medians.get(feature, 0.0)))
    wd_cols = [f"wd_lag_{lag}" for lag in range(24, 0, -1) if f"wd_lag_{lag}" in out.columns]
    if wd_cols:
        mode = train_raw["wd"].mode(dropna=True)
        fallback = str(mode.iloc[0]) if len(mode) else "UNKNOWN"
        out[wd_cols] = out[wd_cols].ffill(axis=1).bfill(axis=1).fillna(fallback)
    return out


def build_validation_windows(train_raw: pd.DataFrame, train_fraction: float) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Create validation windows with rich features."""
    cutoff = chronological_cutoff(train_raw, train_fraction=train_fraction)
    sorted_raw = sort_raw(train_raw)
    train_period = sorted_raw[sorted_raw[DATETIME_COL] <= cutoff].copy()
    imputer = CausalPreprocessor.fit(train_period)
    imputed = imputer.transform(train_raw)
    windows = build_tabular_windows(imputed, train_raw, window_size=24)
    return advanced_feature_frame(windows), cutoff


def build_full_training_windows(train_raw: pd.DataFrame) -> pd.DataFrame:
    """Build windows for final models using all public training rows."""
    sorted_raw = sort_raw(train_raw)
    imputer = CausalPreprocessor.fit(sorted_raw)
    imputed = imputer.transform(sorted_raw)
    windows = build_tabular_windows(imputed, sorted_raw, window_size=24)
    return advanced_feature_frame(windows)


def fit_predict_sklearn(
    name: str,
    estimator,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    scale_numeric: bool,
    sample_weight: np.ndarray | None = None,
) -> tuple[FittedModel, np.ndarray, dict]:
    """Fit sklearn-compatible estimator and return predictions/metrics."""
    features = numeric_cols + categorical_cols
    pipe = Pipeline(
        [
            ("preprocess", model_preprocessor(numeric_cols, categorical_cols, scale_numeric=scale_numeric)),
            ("model", estimator),
        ]
    )
    fit_kwargs = {"model__sample_weight": sample_weight} if sample_weight is not None else {}
    start = time.perf_counter()
    pipe.fit(train_df[features], train_df[TARGET_COL].astype(float), **fit_kwargs)
    fit_seconds = time.perf_counter() - start
    pred = np.clip(pipe.predict(valid_df[features]), 0.0, None)
    row = metric_row(name, valid_df[TARGET_COL], pred, "validation", {"fit_seconds": round(fit_seconds, 3)})
    return FittedModel(name, pipe, features, categorical_cols, row["notes"]), pred, row


def fit_predict_lightgbm(
    lgb,
    name: str,
    params: dict,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    random_state: int,
) -> tuple[FittedModel, np.ndarray, dict] | None:
    """Fit LightGBM if installed."""
    if lgb is None:
        return None
    features = numeric_cols + categorical_cols
    X_train = train_df[features].copy()
    X_valid = valid_df[features].copy()
    for col in categorical_cols:
        X_train[col] = X_train[col].astype("category")
        X_valid[col] = X_valid[col].astype("category")
    full_params = {
        "objective": "regression",
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": -1,
        **params,
    }
    model = lgb.LGBMRegressor(**full_params)
    start = time.perf_counter()
    model.fit(
        X_train,
        train_df[TARGET_COL].astype(float),
        eval_set=[(X_valid, valid_df[TARGET_COL].astype(float))],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(80, verbose=False)],
        categorical_feature=categorical_cols,
    )
    pred = np.clip(model.predict(X_valid, num_iteration=getattr(model, "best_iteration_", None)), 0.0, None)
    notes = {"fit_seconds": round(time.perf_counter() - start, 3), **full_params, "best_iteration": getattr(model, "best_iteration_", None)}
    row = metric_row(name, valid_df[TARGET_COL], pred, "validation", notes)
    return FittedModel(name, model, features, categorical_cols, notes), pred, row


def fit_predict_xgboost(
    xgb,
    name: str,
    params: dict,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    random_state: int,
) -> tuple[FittedModel, np.ndarray, dict] | None:
    """Fit XGBoost if installed."""
    if xgb is None:
        return None
    full_params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": random_state,
        "n_jobs": -1,
        **params,
    }
    return fit_predict_sklearn(
        name,
        xgb.XGBRegressor(**full_params),
        train_df,
        valid_df,
        numeric_cols,
        categorical_cols,
        scale_numeric=False,
    )


def fit_predict_catboost(
    cb,
    name: str,
    params: dict,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    random_state: int,
) -> tuple[FittedModel, np.ndarray, dict] | None:
    """Fit CatBoost if installed."""
    if cb is None:
        return None
    features = numeric_cols + categorical_cols
    X_train = train_df[features].copy()
    X_valid = valid_df[features].copy()
    for col in categorical_cols:
        X_train[col] = X_train[col].astype(str)
        X_valid[col] = X_valid[col].astype(str)
    full_params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": random_state,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": -1,
        **params,
    }
    model = cb.CatBoostRegressor(**full_params)
    start = time.perf_counter()
    model.fit(
        X_train,
        train_df[TARGET_COL].astype(float),
        eval_set=(X_valid, valid_df[TARGET_COL].astype(float)),
        cat_features=categorical_cols,
        use_best_model=True,
    )
    pred = np.clip(model.predict(X_valid), 0.0, None)
    notes = {"fit_seconds": round(time.perf_counter() - start, 3), **full_params}
    row = metric_row(name, valid_df[TARGET_COL], pred, "validation", notes)
    return FittedModel(name, model, features, categorical_cols, notes), pred, row


def metric_row(model: str, y_true: pd.Series | np.ndarray, y_pred: np.ndarray, scope: str, notes: dict | str) -> dict:
    """Build one metrics row."""
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    severe_mask = y > 150.0
    return {
        "model": model,
        "scope": scope,
        "rmse": rmse(y, pred),
        "mae": mae(y, pred),
        "severe_rmse": rmse(y[severe_mask], pred[severe_mask]) if severe_mask.any() else np.nan,
        "severe_bias": float((pred[severe_mask] - y[severe_mask]).mean()) if severe_mask.any() else np.nan,
        "notes": json.dumps(notes) if isinstance(notes, dict) else notes,
    }


def learn_blend(predictions: pd.DataFrame, candidate_cols: list[str]) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Grid-search non-negative blend weights on validation predictions."""
    ranked = sorted(candidate_cols, key=lambda col: rmse(predictions["y_true"], predictions[col]))
    selected = ranked[: min(5, len(ranked))]
    grid = np.linspace(0.0, 1.0, 11)
    values = predictions[selected].to_numpy(dtype=float)
    y = predictions["y_true"].to_numpy(dtype=float)
    best_score = float("inf")
    best_weights = np.ones(len(selected)) / len(selected)
    best_pred = values @ best_weights

    def recurse(n: int, remaining: float):
        if n == 1:
            yield [remaining]
            return
        for w in grid:
            if w <= remaining + 1e-12:
                for rest in recurse(n - 1, remaining - w):
                    yield [w] + rest

    for weights in recurse(len(selected), 1.0):
        if abs(sum(weights) - 1.0) > 1e-8:
            continue
        pred = values @ np.asarray(weights)
        score = rmse(y, pred)
        if score < best_score:
            best_score = score
            best_weights = np.asarray(weights)
            best_pred = pred
    return best_weights, selected, best_pred


def band_metrics(predictions: pd.DataFrame, model_cols: list[str]) -> pd.DataFrame:
    """Compute error metrics per PM2.5 band."""
    rows = []
    frame = predictions.copy()
    frame["pm25_band"] = pd.cut(frame["y_true"], bins=BAND_BINS, labels=BAND_LABELS)
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
    return pd.DataFrame(rows).sort_values(["model", "pm25_band"]).reset_index(drop=True)


def station_metrics(predictions: pd.DataFrame, model_cols: list[str]) -> pd.DataFrame:
    """Compute station-wise metrics."""
    rows = []
    for model in model_cols:
        for station, group in predictions.groupby("station"):
            rows.append({"model": model, "station": station, "rows": len(group), "rmse": rmse(group["y_true"], group[model]), "mae": mae(group["y_true"], group[model])})
    return pd.DataFrame(rows).sort_values(["model", "rmse"]).reset_index(drop=True)


def rolling_origin_validation(windows: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> pd.DataFrame:
    """Objective 6: rolling-origin validation with fast Ridge reference."""
    folds = [
        ("fold_1", "2014-09-01", "2014-12-31 23:00:00"),
        ("fold_2", "2015-01-01", "2015-04-30 23:00:00"),
        ("fold_3", "2015-05-01", "2015-08-31 23:00:00"),
        ("fold_4", "2015-09-01", "2016-02-29 23:00:00"),
    ]
    rows = []
    focused_numeric = [
        col
        for col in numeric_cols
        if col.startswith("PM2.5_lag_")
        or col.startswith("PM10_lag_")
        or col.startswith("NO2_lag_")
        or col.startswith("pm2.5_")
        or col.startswith("pm25_")
        or col in {"hour", "month", "sin_hour", "cos_hour", "sin_month", "cos_month", "is_weekend"}
    ][:180]
    features = focused_numeric + categorical_cols
    dated = windows.copy()
    dated["target_datetime"] = pd.to_datetime(dated["target_datetime"])
    for fold, start, end in folds:
        val_start = pd.Timestamp(start)
        val_end = pd.Timestamp(end)
        train_df = dated[dated["target_datetime"] < val_start].copy()
        valid_df = dated[(dated["target_datetime"] >= val_start) & (dated["target_datetime"] <= val_end)].copy()
        if train_df.empty or valid_df.empty:
            continue
        pipe = Pipeline(
            [
                ("preprocess", model_preprocessor(focused_numeric, categorical_cols, scale_numeric=True)),
                ("model", Ridge(alpha=10.0, random_state=42)),
            ]
        )
        pipe.fit(train_df[features], train_df[TARGET_COL].astype(float))
        pred = np.clip(pipe.predict(valid_df[features]), 0.0, None)
        rows.append(
            {
                "fold": fold,
                "train_rows": len(train_df),
                "validation_rows": len(valid_df),
                "validation_start": val_start,
                "validation_end": val_end,
                "rmse": rmse(valid_df[TARGET_COL], pred),
                "mae": mae(valid_df[TARGET_COL], pred),
            }
        )
    return pd.DataFrame(rows)


def objective_status(
    model_results: pd.DataFrame,
    rolling_results: pd.DataFrame,
    blend_cols: list[str],
    blend_weights: np.ndarray,
) -> pd.DataFrame:
    """Summarize Objective 1-6 outcomes."""
    best = model_results.sort_values("rmse").iloc[0]
    clean_weights = {col: float(np.round(weight, 3)) for col, weight in zip(blend_cols, blend_weights)}
    rows = [
        {"objective": 1, "name": "Rich lag and rolling features", "status": "implemented", "evidence": "Added rolling mean/std/min/max/median/quantiles, trends, slopes, deltas, acceleration, and EWMA features."},
        {"objective": 2, "name": "Weather and pollution interactions", "status": "implemented", "evidence": "Added ratios, low-wind/ventilation, rain, humidity proxy, wind vector, and station-wind categorical interactions."},
        {"objective": 3, "name": "Stronger boosting models", "status": "tested", "evidence": f"Best validation model is {best['model']} with RMSE {best['rmse']:.4f}."},
        {"objective": 4, "name": "Blend multiple model families", "status": "tested", "evidence": f"Blend selected {clean_weights}."},
        {"objective": 5, "name": "Temporal neural networks", "status": "referenced", "evidence": "LSTM, GRU, CNN-LSTM, and TCN were implemented earlier; CNN-LSTM did not beat boosting/Ridge, so it is kept as analysis evidence rather than final submission core."},
        {"objective": 6, "name": "Rolling-origin validation", "status": "implemented", "evidence": f"Ran {len(rolling_results)} chronological folds; mean RMSE {rolling_results['rmse'].mean():.4f}." if not rolling_results.empty else "No folds available."},
    ]
    return pd.DataFrame(rows)


def predict_with_model(fitted: FittedModel, frame: pd.DataFrame) -> np.ndarray:
    """Predict with a fitted model wrapper."""
    X = frame[fitted.features].copy()
    for col in fitted.categorical_cols:
        if col in X.columns and fitted.name.startswith(("lightgbm",)):
            X[col] = X[col].astype("category")
        elif col in X.columns and fitted.name.startswith("catboost"):
            X[col] = X[col].astype(str)
    if fitted.name.startswith("lightgbm"):
        pred = fitted.estimator.predict(X, num_iteration=getattr(fitted.estimator, "best_iteration_", None))
    else:
        pred = fitted.estimator.predict(X)
    return np.clip(np.asarray(pred, dtype=float), 0.0, None)


def make_submission_files(
    train_raw: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
    output_dir: Path,
    validation_weights: dict[str, float],
    max_boost_train_rows: int,
    random_state: int,
) -> pd.DataFrame:
    """Train final models and write official test submissions."""
    submissions_dir = output_dir / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    train_windows = build_full_training_windows(train_raw)
    test_features = advanced_feature_frame(rowwise_impute_test(test, train_raw))
    numeric_cols, categorical_cols = numeric_categoricals(train_windows)
    numeric_cols = [col for col in numeric_cols if col in test_features.columns]
    categorical_cols = [col for col in categorical_cols if col in test_features.columns]
    train_windows = train_windows.dropna(subset=numeric_cols + categorical_cols + [TARGET_COL]).copy()
    for col in numeric_cols:
        fill_value = float(train_windows[col].median())
        train_windows[col] = train_windows[col].fillna(fill_value)
        test_features[col] = test_features[col].fillna(fill_value)
    for col in categorical_cols:
        mode = train_windows[col].mode(dropna=True)
        fill_value = str(mode.iloc[0]) if len(mode) else "UNKNOWN"
        train_windows[col] = train_windows[col].fillna(fill_value).astype(str)
        test_features[col] = test_features[col].fillna(fill_value).astype(str)

    fitted_models: dict[str, FittedModel] = {}
    ridge_model, _, _ = fit_predict_sklearn(
        "ridge_alpha_10",
        Ridge(alpha=10.0, random_state=random_state),
        train_windows,
        train_windows.tail(min(len(train_windows), 1000)),
        numeric_cols,
        categorical_cols,
        scale_numeric=True,
    )
    fitted_models["ridge_alpha_10"] = ridge_model

    modules = optional_imports()
    lgb = modules.get("lightgbm")
    if lgb is not None:
        boost_train = latest_training_subset(train_windows, max_boost_train_rows)
        lgb_model = lgb.LGBMRegressor(
            n_estimators=160,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=8,
            min_child_samples=80,
            subsample=0.85,
            colsample_bytree=0.75,
            reg_lambda=4.0,
            objective="regression",
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
        features = numeric_cols + categorical_cols
        X_train = boost_train[features].copy()
        for col in categorical_cols:
            X_train[col] = X_train[col].astype("category")
        lgb_model.fit(X_train, boost_train[TARGET_COL].astype(float), categorical_feature=categorical_cols)
        fitted_models["lightgbm_final"] = FittedModel("lightgbm_final", lgb_model, features, categorical_cols, {"train_rows": len(boost_train)})

    test_predictions = pd.DataFrame({"id": test["id"].to_numpy()})
    for name, fitted in fitted_models.items():
        test_predictions[name] = predict_with_model(fitted, test_features)

    weights = validation_weights.copy()
    if "lightgbm_final" not in test_predictions.columns:
        weights = {"ridge_alpha_10": 1.0}
    total_weight = sum(weight for model, weight in weights.items() if model in test_predictions.columns)
    if total_weight <= 0:
        weights = {"ridge_alpha_10": 1.0}
        total_weight = 1.0
    weighted_pred = np.zeros(len(test_predictions), dtype=float)
    for model, weight in weights.items():
        if model in test_predictions.columns:
            weighted_pred += (weight / total_weight) * test_predictions[model].to_numpy(dtype=float)

    base_submission = sample_submission[["id"]].copy()
    base_submission["PM2.5"] = np.clip(weighted_pred, 0.0, None)
    calibrated_submission = base_submission.copy()
    calibrated_submission["PM2.5"] = np.clip(
        calibrated_submission["PM2.5"] + 0.025 * np.maximum(calibrated_submission["PM2.5"] - 150.0, 0.0),
        0.0,
        None,
    )

    base_submission.to_csv(submissions_dir / "submission_final_weighted_ensemble.csv", index=False)
    calibrated_submission.to_csv(submissions_dir / "submission_final_weighted_ensemble_calibrated.csv", index=False)

    prediction_summary = pd.DataFrame(
        [
            {
                "submission": "submission_final_weighted_ensemble.csv",
                "rows": len(base_submission),
                "mean_prediction": base_submission["PM2.5"].mean(),
                "min_prediction": base_submission["PM2.5"].min(),
                "max_prediction": base_submission["PM2.5"].max(),
            },
            {
                "submission": "submission_final_weighted_ensemble_calibrated.csv",
                "rows": len(calibrated_submission),
                "mean_prediction": calibrated_submission["PM2.5"].mean(),
                "min_prediction": calibrated_submission["PM2.5"].min(),
                "max_prediction": calibrated_submission["PM2.5"].max(),
            },
        ]
    )
    test_predictions["weighted_ensemble_submission"] = base_submission["PM2.5"]
    test_predictions["weighted_ensemble_calibrated_submission"] = calibrated_submission["PM2.5"]
    return prediction_summary, test_predictions


def make_plots(report_dir: Path, model_results: pd.DataFrame, station_df: pd.DataFrame, band_df: pd.DataFrame, rolling_df: pd.DataFrame, predictions: pd.DataFrame, best_model: str) -> None:
    """Generate final analysis figures."""
    sns.set_theme(style="whitegrid")
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 6))
    sns.barplot(data=model_results.head(12), y="model", x="rmse", color="#4C78A8")
    plt.title("Objective Experiments: Validation RMSE")
    plt.xlabel("RMSE")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(figures_dir / "objective_model_rmse.png", dpi=160)
    plt.close()

    best_station = station_df[station_df["model"] == best_model].sort_values("rmse", ascending=False)
    plt.figure(figsize=(11, 6))
    sns.barplot(data=best_station, y="station", x="rmse", color="#F28E2B")
    plt.title(f"Station-Wise RMSE: {best_model}")
    plt.xlabel("RMSE")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(figures_dir / "objective_station_rmse.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.barplot(data=band_df[band_df["model"] == best_model], x="pm25_band", y="rmse", color="#E15759")
    plt.title(f"PM2.5 Band RMSE: {best_model}")
    plt.xlabel("True PM2.5 band")
    plt.ylabel("RMSE")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(figures_dir / "objective_pm25_band_rmse.png", dpi=160)
    plt.close()

    if not rolling_df.empty:
        plt.figure(figsize=(8.5, 5))
        sns.lineplot(data=rolling_df, x="fold", y="rmse", marker="o")
        plt.title("Rolling-Origin Ridge RMSE")
        plt.xlabel("Fold")
        plt.ylabel("RMSE")
        plt.tight_layout()
        plt.savefig(figures_dir / "objective_rolling_origin_rmse.png", dpi=160)
        plt.close()

    sample = predictions.sample(min(len(predictions), 8000), random_state=42)
    residual = sample[best_model] - sample["y_true"]
    plt.figure(figsize=(9, 5))
    sns.scatterplot(x=sample["y_true"], y=residual, s=8, alpha=0.25, edgecolor=None)
    plt.axhline(0, color="black", linewidth=1)
    plt.axvline(150, color="#E15759", linewidth=1)
    plt.title(f"Residuals: {best_model}")
    plt.xlabel("True PM2.5")
    plt.ylabel("Prediction - true")
    plt.tight_layout()
    plt.savefig(figures_dir / "objective_residuals.png", dpi=160)
    plt.close()


def run(
    data_dir: str | Path,
    output_dir: str | Path,
    train_fraction: float,
    max_boost_train_rows: int,
    random_state: int,
) -> None:
    """Run Objective 1-6 experiments and create official submissions."""
    output_dir = Path(output_dir)
    report_dir = output_dir / "reports" / "final_objective_experiments"
    report_dir.mkdir(parents=True, exist_ok=True)
    files = read_competition_files(data_dir)
    train_raw = files.get("train_raw.csv")
    test = files.get("test.csv")
    sample_submission = files.get("sample_submission.csv")
    if train_raw is None or test is None or sample_submission is None:
        raise FileNotFoundError("train_raw.csv, test.csv, and sample_submission.csv are required.")

    windows, cutoff = build_validation_windows(train_raw, train_fraction=train_fraction)
    train_df, valid_df = split_windows(windows, cutoff)
    practical_boost_rows = min(max_boost_train_rows, 20_000)
    boost_train_df = latest_training_subset(train_df, practical_boost_rows)
    numeric_cols, categorical_cols = numeric_categoricals(windows)
    for col in numeric_cols:
        fill_value = float(train_df[col].median())
        train_df[col] = train_df[col].fillna(fill_value)
        valid_df[col] = valid_df[col].fillna(fill_value)
        boost_train_df[col] = boost_train_df[col].fillna(fill_value)
    for col in categorical_cols:
        mode = train_df[col].mode(dropna=True)
        fill_value = str(mode.iloc[0]) if len(mode) else "UNKNOWN"
        train_df[col] = train_df[col].fillna(fill_value).astype(str)
        valid_df[col] = valid_df[col].fillna(fill_value).astype(str)
        boost_train_df[col] = boost_train_df[col].fillna(fill_value).astype(str)

    predictions = pd.DataFrame(
        {
            "station": valid_df[STATION_COL].to_numpy(),
            "window_end_datetime": valid_df["window_end_datetime"].to_numpy(),
            "target_datetime": valid_df["target_datetime"].to_numpy(),
            "y_true": valid_df[TARGET_COL].to_numpy(dtype=float),
        }
    )
    model_rows = []
    fitted_models = []

    specs = [
        ("ridge_alpha_10", Ridge(alpha=10.0, random_state=random_state), True, train_df),
    ]
    for name, estimator, scale, frame in specs:
        print(f"Training {name} on {len(frame)} rows...", flush=True)
        fitted, pred, row = fit_predict_sklearn(name, estimator, frame, valid_df, numeric_cols, categorical_cols, scale_numeric=scale)
        fitted_models.append(fitted)
        predictions[name] = pred
        model_rows.append(row)

    model_rows.append(
        {
            "model": "elasticnet_alpha005_l105",
            "scope": "validation",
            "rmse": np.nan,
            "mae": np.nan,
            "severe_rmse": np.nan,
            "severe_bias": np.nan,
            "notes": json.dumps(
                {
                    "status": "skipped_in_final_objective_run",
                    "reason": "ElasticNet was too slow on the expanded Objective 1-2 dense feature matrix; Ridge remains the stronger practical linear baseline.",
                }
            ),
        }
    )

    modules = optional_imports()
    lgb_configs = [
        ("lightgbm_tuned_compact_recent", {"n_estimators": 180, "learning_rate": 0.05, "num_leaves": 63, "max_depth": 8, "min_child_samples": 80, "subsample": 0.85, "colsample_bytree": 0.75, "reg_lambda": 4.0}),
    ]
    for name, params in lgb_configs:
        print(f"Training {name} on {len(boost_train_df)} rows...", flush=True)
        result = fit_predict_lightgbm(modules.get("lightgbm"), name, params, boost_train_df, valid_df, numeric_cols, categorical_cols, random_state)
        if result is not None:
            fitted, pred, row = result
            fitted_models.append(fitted)
            predictions[name] = pred
            model_rows.append(row)

    if modules.get("xgboost") is not None:
        model_rows.append(
            {
                "model": "xgboost_tuned_hist",
                "scope": "validation",
                "rmse": np.nan,
                "mae": np.nan,
                "severe_rmse": np.nan,
                "severe_bias": np.nan,
                "notes": json.dumps(
                    {
                        "status": "skipped_in_final_objective_run",
                        "reason": "XGBoost was tested earlier; expanded Objective 1-2 feature matrix is too slow for the final reproducible notebook run.",
                    }
                ),
            }
        )

    if modules.get("catboost") is not None:
        model_rows.append(
            {
                "model": "catboost_tuned_depth8",
                "scope": "validation",
                "rmse": np.nan,
                "mae": np.nan,
                "severe_rmse": np.nan,
                "severe_bias": np.nan,
                "notes": json.dumps(
                    {
                        "status": "skipped_in_final_objective_run",
                        "reason": "CatBoost was already tested earlier and is too slow on the expanded Objective 1-2 feature set for the final interactive run.",
                    }
                ),
            }
        )

    model_cols = [col for col in predictions.columns if col not in KEY_COLUMNS]
    weights, selected_cols, blend_pred = learn_blend(predictions, model_cols)
    predictions["objective_weighted_blend"] = np.clip(blend_pred, 0.0, None)
    model_rows.append(metric_row("objective_weighted_blend", predictions["y_true"], predictions["objective_weighted_blend"], "validation", {"weights": dict(zip(selected_cols, np.round(weights, 6)))}))
    predictions["objective_weighted_blend_calibrated"] = np.clip(
        predictions["objective_weighted_blend"] + 0.025 * np.maximum(predictions["objective_weighted_blend"] - 150.0, 0.0),
        0.0,
        None,
    )
    model_rows.append(metric_row("objective_weighted_blend_calibrated", predictions["y_true"], predictions["objective_weighted_blend_calibrated"], "validation", {"calibration": "prediction + 0.025*max(prediction-150,0)"}))

    model_results = pd.DataFrame(model_rows).sort_values("rmse").reset_index(drop=True)
    final_model_cols = [col for col in predictions.columns if col not in KEY_COLUMNS]
    station_df = station_metrics(predictions, final_model_cols)
    band_df = band_metrics(predictions, final_model_cols)
    print("Running focused rolling-origin validation...", flush=True)
    rolling_df = rolling_origin_validation(windows, numeric_cols, categorical_cols)
    objective_df = objective_status(model_results, rolling_df, selected_cols, weights)
    best_model = model_results.iloc[0]["model"]

    validation_weights: dict[str, float] = {}
    for model_name, weight in zip(selected_cols, weights):
        if model_name == "ridge_alpha_10":
            validation_weights["ridge_alpha_10"] = float(weight)
        elif model_name == "lightgbm_tuned_compact_recent":
            validation_weights["lightgbm_final"] = float(weight)
    if not validation_weights:
        validation_weights = {"ridge_alpha_10": 1.0}
    submission_summary, test_predictions = make_submission_files(
        train_raw=train_raw,
        test=test,
        sample_submission=sample_submission,
        output_dir=output_dir,
        validation_weights=validation_weights,
        max_boost_train_rows=practical_boost_rows,
        random_state=random_state,
    )

    summary = pd.DataFrame(
        [
            {
                "official_train_rows": len(train_raw),
                "official_test_rows": len(test),
                "sample_submission_rows": len(sample_submission),
                "validation_windows": len(valid_df),
                "feature_columns": len(numeric_cols) + len(categorical_cols),
                "best_validation_model": best_model,
                "best_validation_rmse": model_results.iloc[0]["rmse"],
                "best_validation_mae": model_results.iloc[0]["mae"],
                "best_validation_severe_rmse": model_results.iloc[0]["severe_rmse"],
                "public_leaderboard_fraction": 0.30,
                "private_leaderboard_fraction": 0.70,
                "blend_weights": json.dumps(dict(zip(selected_cols, np.round(weights, 6)))),
            }
        ]
    )

    summary.to_csv(report_dir / "experiment_summary.csv", index=False)
    model_results.to_csv(report_dir / "model_results.csv", index=False)
    station_df.to_csv(report_dir / "station_metrics.csv", index=False)
    band_df.to_csv(report_dir / "band_metrics.csv", index=False)
    rolling_df.to_csv(report_dir / "rolling_origin_results.csv", index=False)
    objective_df.to_csv(report_dir / "objective_status.csv", index=False)
    predictions.to_csv(report_dir / "validation_predictions.csv", index=False)
    submission_summary.to_csv(report_dir / "submission_summary.csv", index=False)
    test_predictions.to_csv(report_dir / "test_predictions.csv", index=False)
    make_plots(report_dir, model_results, station_df, band_df, rolling_df, predictions, best_model)

    print("Final Objective 1-6 experiments complete.")
    print(summary.to_string(index=False))
    print("\nTop validation models:")
    print(model_results.head(12).to_string(index=False))
    print("\nRolling-origin results:")
    print(rolling_df.to_string(index=False))
    print("\nSubmission summary:")
    print(submission_summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_LOCAL_DATA_DIR), help="Directory containing official Kaggle files.")
    parser.add_argument("--output-dir", default=".", help="Directory where reports and submissions are written.")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--max-boost-train-rows", type=int, default=180_000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.data_dir, args.output_dir, args.train_fraction, args.max_boost_train_rows, args.random_state)
