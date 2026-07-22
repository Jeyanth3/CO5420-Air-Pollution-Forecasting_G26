"""Severe-pollution correction experiments for PM2.5 forecasting."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler

from src.config import DEFAULT_LOCAL_DATA_DIR
from src.data_io import read_competition_files
from src.gradient_boosting_models import (
    build_feature_frame,
    latest_training_subset,
    optional_imports,
    split_windows,
)
from src.metrics import mae, rmse
from src.preprocessing_window_baselines import feature_columns


KEY_COLUMNS = ["station", "window_end_datetime", "target_datetime", "y_true"]
BAND_BINS = [-np.inf, 35.0, 75.0, 150.0, np.inf]
BAND_LABELS = ["low_<=35", "moderate_35_75", "high_75_150", "severe_>150"]


def make_preprocessor(numeric_cols: list[str], categorical_cols: list[str], scale: bool = True) -> ColumnTransformer:
    """Build sklearn feature preprocessing."""
    numeric_transformer = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def severe_weights(y: pd.Series | np.ndarray, high_weight: float = 1.0, severe_weight: float = 4.0) -> np.ndarray:
    """Create sample weights that emphasize high and severe PM2.5 rows."""
    values = np.asarray(y, dtype=float)
    weights = np.ones_like(values, dtype=float)
    weights += high_weight * (values >= 75.0)
    weights += severe_weight * (values >= 150.0)
    return weights


def score_row(
    model: str,
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    family: str,
    notes: dict | str,
    evaluation_scope: str = "full_validation",
) -> dict:
    """Create one result row."""
    pred = np.clip(np.asarray(y_pred, dtype=float), 0.0, None)
    return {
        "model": model,
        "family": family,
        "evaluation_scope": evaluation_scope,
        "rmse": rmse(y_true, pred),
        "mae": mae(y_true, pred),
        "severe_rmse": band_metric(y_true, pred, "severe_>150", "rmse"),
        "severe_mae": band_metric(y_true, pred, "severe_>150", "mae"),
        "severe_bias": band_bias(y_true, pred, "severe_>150"),
        "notes": json.dumps(notes) if isinstance(notes, dict) else notes,
    }


def band_frame(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Return y/pred rows with PM2.5 bands."""
    out = pd.DataFrame({"y_true": np.asarray(y_true, dtype=float), "prediction": np.asarray(y_pred, dtype=float)})
    out["pm25_band"] = pd.cut(out["y_true"], bins=BAND_BINS, labels=BAND_LABELS)
    return out


def band_metric(y_true: pd.Series | np.ndarray, y_pred: np.ndarray, band: str, metric: str) -> float:
    """Return RMSE or MAE inside one true-PM2.5 band."""
    frame = band_frame(y_true, y_pred)
    group = frame[frame["pm25_band"].astype(str) == band]
    if group.empty:
        return float("nan")
    if metric == "rmse":
        return rmse(group["y_true"], group["prediction"])
    if metric == "mae":
        return mae(group["y_true"], group["prediction"])
    raise ValueError(f"Unknown metric: {metric}")


def band_bias(y_true: pd.Series | np.ndarray, y_pred: np.ndarray, band: str) -> float:
    """Return mean prediction bias inside one true-PM2.5 band."""
    frame = band_frame(y_true, y_pred)
    group = frame[frame["pm25_band"].astype(str) == band]
    if group.empty:
        return float("nan")
    return float((group["prediction"] - group["y_true"]).mean())


def all_band_metrics(predictions: pd.DataFrame, model_cols: list[str]) -> pd.DataFrame:
    """Compute error metrics per pollution band for selected model columns."""
    rows = []
    for model in model_cols:
        frame = band_frame(predictions["y_true"], predictions[model])
        for band, group in frame.groupby("pm25_band", observed=True):
            rows.append(
                {
                    "model": model,
                    "pm25_band": str(band),
                    "rows": len(group),
                    "rmse": rmse(group["y_true"], group["prediction"]),
                    "mae": mae(group["y_true"], group["prediction"]),
                    "bias": float((group["prediction"] - group["y_true"]).mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "pm25_band"]).reset_index(drop=True)


def fit_ridge_variant(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    name: str,
    transform: str,
    sample_weight: np.ndarray | None,
) -> tuple[dict, np.ndarray]:
    """Fit a Ridge variant with optional target transform and sample weights."""
    features = numeric_cols + categorical_cols
    y_train = train_df["target_pm25"].astype(float).to_numpy()
    y_valid = valid_df["target_pm25"].astype(float).to_numpy()
    pipeline = Pipeline(
        [
            ("preprocess", make_preprocessor(numeric_cols, categorical_cols, scale=True)),
            ("model", Ridge(alpha=10.0, random_state=42)),
        ]
    )

    transformer = None
    if transform == "raw":
        target = y_train
    elif transform == "log1p":
        target = np.log1p(y_train)
    elif transform == "boxcox":
        transformer = PowerTransformer(method="box-cox", standardize=True)
        target = transformer.fit_transform((y_train + 1.0).reshape(-1, 1)).ravel()
    else:
        raise ValueError(f"Unknown transform: {transform}")

    start = time.perf_counter()
    fit_kwargs = {"model__sample_weight": sample_weight} if sample_weight is not None else {}
    pipeline.fit(train_df[features], target, **fit_kwargs)
    pred_transformed = pipeline.predict(valid_df[features])

    if transform == "raw":
        pred = pred_transformed
    elif transform == "log1p":
        pred = np.expm1(pred_transformed)
    else:
        pred = transformer.inverse_transform(pred_transformed.reshape(-1, 1)).ravel() - 1.0

    notes = {
        "transform": transform,
        "sample_weight": sample_weight is not None,
        "fit_seconds": round(time.perf_counter() - start, 3),
        "train_rows": len(train_df),
    }
    return score_row(name, y_valid, pred, "ridge_retrain", notes), np.clip(pred, 0.0, None)


def fit_lightgbm_variant(
    lgb,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    name: str,
    transform: str,
    sample_weight: np.ndarray | None,
    random_state: int,
) -> tuple[dict, np.ndarray] | None:
    """Fit a LightGBM severe-pollution variant when LightGBM is available."""
    if lgb is None:
        return None

    features = numeric_cols + categorical_cols
    X_train = train_df[features].copy()
    X_valid = valid_df[features].copy()
    for col in categorical_cols:
        X_train[col] = X_train[col].astype("category")
        X_valid[col] = X_valid[col].astype("category")

    y_train = train_df["target_pm25"].astype(float).to_numpy()
    y_valid = valid_df["target_pm25"].astype(float).to_numpy()
    transformer = None
    if transform == "raw":
        target_train = y_train
        target_valid = y_valid
    elif transform == "log1p":
        target_train = np.log1p(y_train)
        target_valid = np.log1p(y_valid)
    elif transform == "boxcox":
        transformer = PowerTransformer(method="box-cox", standardize=True)
        target_train = transformer.fit_transform((y_train + 1.0).reshape(-1, 1)).ravel()
        target_valid = transformer.transform((y_valid + 1.0).reshape(-1, 1)).ravel()
    else:
        raise ValueError(f"Unknown transform: {transform}")

    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.025,
        num_leaves=95,
        max_depth=9,
        min_child_samples=70,
        subsample=0.88,
        colsample_bytree=0.82,
        reg_lambda=3.0,
        objective="regression",
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )
    start = time.perf_counter()
    model.fit(
        X_train,
        target_train,
        sample_weight=sample_weight,
        eval_set=[(X_valid, target_valid)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(70, verbose=False)],
        categorical_feature=categorical_cols,
    )
    pred_transformed = model.predict(X_valid, num_iteration=getattr(model, "best_iteration_", None))
    if transform == "raw":
        pred = pred_transformed
    elif transform == "log1p":
        pred = np.expm1(pred_transformed)
    else:
        pred = transformer.inverse_transform(pred_transformed.reshape(-1, 1)).ravel() - 1.0

    notes = {
        "transform": transform,
        "sample_weight": sample_weight is not None,
        "fit_seconds": round(time.perf_counter() - start, 3),
        "train_rows": len(train_df),
        "best_iteration": getattr(model, "best_iteration_", None),
    }
    return score_row(name, y_valid, pred, "lightgbm_retrain", notes), np.clip(pred, 0.0, None)


def load_saved_gradient_predictions(output_dir: Path) -> pd.DataFrame | None:
    """Load the previous best validation predictions if available."""
    path = output_dir / "reports" / "gradient_boosting" / "validation_predictions.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    frame["window_end_datetime"] = pd.to_datetime(frame["window_end_datetime"])
    frame["target_datetime"] = pd.to_datetime(frame["target_datetime"])
    return frame


def late_validation_mask(predictions: pd.DataFrame, train_fraction: float) -> pd.Series:
    """Split validation predictions chronologically for calibration experiments."""
    unique_times = predictions["target_datetime"].drop_duplicates().sort_values().reset_index(drop=True)
    split_index = max(1, min(len(unique_times) - 1, int(len(unique_times) * train_fraction)))
    split_time = unique_times.iloc[split_index - 1]
    return predictions["target_datetime"] <= split_time


def correction_features(base_pred: np.ndarray, prediction_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Create calibration features from saved model predictions."""
    out = pd.DataFrame(
        {
            "base_pred": base_pred,
            "base_pred_sq": base_pred**2,
            "base_gt_75": (base_pred >= 75.0).astype(float),
            "base_gt_100": (base_pred >= 100.0).astype(float),
            "base_gt_125": (base_pred >= 125.0).astype(float),
            "base_gt_150": (base_pred >= 150.0).astype(float),
            "excess_75": np.maximum(base_pred - 75.0, 0.0),
            "excess_100": np.maximum(base_pred - 100.0, 0.0),
            "excess_125": np.maximum(base_pred - 125.0, 0.0),
            "excess_150": np.maximum(base_pred - 150.0, 0.0),
        }
    )
    if prediction_frame is not None:
        for col in prediction_frame.columns:
            if col not in KEY_COLUMNS and pd.api.types.is_numeric_dtype(prediction_frame[col]):
                out[f"model_{col}"] = prediction_frame[col].astype(float).to_numpy()
    return out


def select_additive_correction(
    calibration: pd.DataFrame,
    base_col: str,
    objective: str,
) -> tuple[dict, np.ndarray]:
    """Search a simple threshold/offset/slope correction on calibration rows."""
    y = calibration["y_true"].to_numpy(dtype=float)
    base = calibration[base_col].to_numpy(dtype=float)
    best_score = float("inf")
    best_params = None
    best_pred = None
    for threshold in [60.0, 75.0, 90.0, 100.0, 115.0, 130.0, 150.0]:
        excess = np.maximum(base - threshold, 0.0)
        mask = (base >= threshold).astype(float)
        for offset in np.linspace(0.0, 28.0, 15):
            for slope in np.linspace(0.0, 0.35, 15):
                pred = np.clip(base + offset * mask + slope * excess, 0.0, None)
                if objective == "overall_rmse":
                    score = rmse(y, pred)
                elif objective == "severe_rmse":
                    score = band_metric(y, pred, "severe_>150", "rmse")
                elif objective == "balanced":
                    score = rmse(y, pred) + 0.2 * abs(band_bias(y, pred, "severe_>150"))
                else:
                    raise ValueError(f"Unknown correction objective: {objective}")
                if np.isfinite(score) and score < best_score:
                    best_score = score
                    best_params = {"threshold": threshold, "offset": float(offset), "slope": float(slope), "objective": objective}
                    best_pred = pred
    return best_params, best_pred


def apply_additive_correction(base: np.ndarray, params: dict) -> np.ndarray:
    """Apply a selected threshold correction."""
    threshold = params["threshold"]
    return np.clip(
        base + params["offset"] * (base >= threshold).astype(float) + params["slope"] * np.maximum(base - threshold, 0.0),
        0.0,
        None,
    )


def run_calibration_experiments(saved: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate calibration/residual correction on saved best-model predictions."""
    if "weighted_ensemble" not in saved.columns:
        raise ValueError("Expected weighted_ensemble in gradient_boosting validation predictions.")

    base_col = "weighted_ensemble"
    calibration_mask = late_validation_mask(saved, train_fraction=0.5)
    calib = saved.loc[calibration_mask].copy()
    holdout = saved.loc[~calibration_mask].copy()
    y_holdout = holdout["y_true"].to_numpy(dtype=float)

    prediction_out = saved[KEY_COLUMNS + [base_col]].copy()
    rows = [
        score_row(
            "reference_weighted_ensemble_holdout",
            y_holdout,
            holdout[base_col].to_numpy(dtype=float),
            "saved_prediction_calibration",
            {"evaluation": "late_validation_holdout"},
            evaluation_scope="late_validation_holdout",
        )
    ]

    for objective in ["overall_rmse", "balanced", "severe_rmse"]:
        params, _ = select_additive_correction(calib, base_col, objective)
        corrected_full = apply_additive_correction(saved[base_col].to_numpy(dtype=float), params)
        corrected_holdout = apply_additive_correction(holdout[base_col].to_numpy(dtype=float), params)
        name = f"additive_threshold_{objective}"
        prediction_out[name] = corrected_full
        rows.append(
            score_row(
                name,
                y_holdout,
                corrected_holdout,
                "saved_prediction_calibration",
                {"evaluation": "late_validation_holdout", **params},
                evaluation_scope="late_validation_holdout",
            )
        )

    x_calib = correction_features(calib[base_col].to_numpy(dtype=float), calib)
    x_holdout = correction_features(holdout[base_col].to_numpy(dtype=float), holdout)
    x_full = correction_features(saved[base_col].to_numpy(dtype=float), saved)
    y_resid_calib = calib["y_true"].to_numpy(dtype=float) - calib[base_col].to_numpy(dtype=float)
    weights = severe_weights(calib["y_true"], high_weight=0.5, severe_weight=4.0)
    residual_model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=25.0, random_state=42)),
        ]
    )
    residual_model.fit(x_calib, y_resid_calib, ridge__sample_weight=weights)
    full_pred = np.clip(saved[base_col].to_numpy(dtype=float) + residual_model.predict(x_full), 0.0, None)
    holdout_pred = np.clip(holdout[base_col].to_numpy(dtype=float) + residual_model.predict(x_holdout), 0.0, None)
    prediction_out["residual_ridge_severe_weighted"] = full_pred
    rows.append(
        score_row(
            "residual_ridge_severe_weighted",
            y_holdout,
            holdout_pred,
            "saved_prediction_calibration",
            {"evaluation": "late_validation_holdout", "base_model": base_col, "sample_weight": "high=0.5,severe=4.0"},
            evaluation_scope="late_validation_holdout",
        )
    )

    target_model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=25.0, random_state=42)),
        ]
    )
    target_model.fit(x_calib, calib["y_true"].to_numpy(dtype=float), ridge__sample_weight=weights)
    full_pred = np.clip(target_model.predict(x_full), 0.0, None)
    holdout_pred = np.clip(target_model.predict(x_holdout), 0.0, None)
    prediction_out["target_ridge_severe_weighted"] = full_pred
    rows.append(
        score_row(
            "target_ridge_severe_weighted",
            y_holdout,
            holdout_pred,
            "saved_prediction_calibration",
            {"evaluation": "late_validation_holdout", "inputs": "all saved model predictions"},
            evaluation_scope="late_validation_holdout",
        )
    )

    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True), prediction_out


def run_retraining_experiments(
    train_raw: pd.DataFrame,
    train_fraction: float,
    max_boost_train_rows: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run severe-focused retraining variants on the original train/validation split."""
    windows, cutoff = build_feature_frame(train_raw, train_fraction=train_fraction)
    train_df, valid_df = split_windows(windows, cutoff)
    numeric_cols, categorical_cols = feature_columns(windows)
    y_valid = valid_df["target_pm25"].astype(float).to_numpy()

    prediction_out = pd.DataFrame(
        {
            "station": valid_df["station"].to_numpy(),
            "window_end_datetime": valid_df["window_end_datetime"].to_numpy(),
            "target_datetime": valid_df["target_datetime"].to_numpy(),
            "y_true": y_valid,
        }
    )
    rows = []

    ridge_specs = [
        ("ridge_reference_raw", "raw", None),
        ("ridge_weighted_high1_severe4", "raw", severe_weights(train_df["target_pm25"], high_weight=1.0, severe_weight=4.0)),
        ("ridge_weighted_high2_severe8", "raw", severe_weights(train_df["target_pm25"], high_weight=2.0, severe_weight=8.0)),
        ("ridge_log1p", "log1p", None),
        ("ridge_boxcox", "boxcox", None),
        ("ridge_boxcox_weighted_high1_severe4", "boxcox", severe_weights(train_df["target_pm25"], high_weight=1.0, severe_weight=4.0)),
    ]
    for name, transform, weights in ridge_specs:
        row, pred = fit_ridge_variant(train_df, valid_df, numeric_cols, categorical_cols, name, transform, weights)
        rows.append(row)
        prediction_out[name] = pred

    modules = optional_imports()
    lgb = modules.get("lightgbm")
    if lgb is not None:
        boost_train_df = latest_training_subset(train_df, max_boost_train_rows)
        boost_weight_mild = severe_weights(boost_train_df["target_pm25"], high_weight=0.75, severe_weight=3.0)
        boost_weight_strong = severe_weights(boost_train_df["target_pm25"], high_weight=1.5, severe_weight=6.0)
        lgb_specs = [
            ("lightgbm_reference_raw", "raw", None),
            ("lightgbm_weighted_high075_severe3", "raw", boost_weight_mild),
            ("lightgbm_weighted_high15_severe6", "raw", boost_weight_strong),
            ("lightgbm_log1p_weighted_high075_severe3", "log1p", boost_weight_mild),
            ("lightgbm_boxcox_weighted_high075_severe3", "boxcox", boost_weight_mild),
        ]
        for name, transform, weights in lgb_specs:
            result = fit_lightgbm_variant(
                lgb,
                boost_train_df,
                valid_df,
                numeric_cols,
                categorical_cols,
                name,
                transform,
                weights,
                random_state,
            )
            if result is None:
                continue
            row, pred = result
            rows.append(row)
            prediction_out[name] = pred

    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True), prediction_out


def make_plots(report_dir: Path, results: pd.DataFrame, band_metrics_df: pd.DataFrame, predictions: pd.DataFrame, best_model: str) -> None:
    """Save severe-pollution correction figures."""
    sns.set_theme(style="whitegrid")
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    full_results = results[results["evaluation_scope"] == "full_validation"].copy()
    if full_results.empty:
        full_results = results.copy()

    plt.figure(figsize=(11, 6))
    sns.barplot(data=full_results.sort_values("rmse").head(12), y="model", x="rmse", hue="family", dodge=False)
    plt.title("Severe-Pollution Fix Experiments: Full-Validation RMSE")
    plt.xlabel("RMSE")
    plt.ylabel("")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(figures_dir / "overall_rmse_severe_fix_candidates.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 6))
    ordered = full_results.sort_values("severe_rmse").head(12)
    sns.barplot(data=ordered, y="model", x="severe_rmse", hue="family", dodge=False)
    plt.title("Severe-Band RMSE Candidate Comparison")
    plt.xlabel("RMSE for true PM2.5 > 150")
    plt.ylabel("")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(figures_dir / "severe_band_rmse_candidates.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8.5, 5.5))
    sns.scatterplot(data=full_results, x="rmse", y="severe_rmse", hue="family", s=85)
    plt.title("Full-Validation RMSE vs Severe-Band RMSE")
    plt.xlabel("Overall RMSE")
    plt.ylabel("Severe-band RMSE")
    plt.tight_layout()
    plt.savefig(figures_dir / "overall_vs_severe_rmse_tradeoff.png", dpi=160)
    plt.close()

    if best_model in predictions.columns:
        sample = predictions.sample(min(len(predictions), 8000), random_state=42)
        residual = sample[best_model] - sample["y_true"]
        plt.figure(figsize=(9, 5.5))
        sns.scatterplot(x=sample["y_true"], y=residual, s=8, alpha=0.25, edgecolor=None)
        plt.axhline(0, color="black", linewidth=1)
        plt.axvline(150, color="#E15759", linewidth=1)
        plt.title(f"Residuals For Selected Candidate: {best_model}")
        plt.xlabel("True PM2.5")
        plt.ylabel("Prediction - true")
        plt.tight_layout()
        plt.savefig(figures_dir / "selected_candidate_residuals.png", dpi=160)
        plt.close()

    severe_rows = band_metrics_df[band_metrics_df["pm25_band"] == "severe_>150"]
    plt.figure(figsize=(11, 6))
    sns.barplot(data=severe_rows.sort_values("bias").head(14), y="model", x="bias", color="#E15759")
    plt.axvline(0, color="black", linewidth=1)
    plt.title("Severe-Band Bias")
    plt.xlabel("Prediction - true")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(figures_dir / "severe_band_bias.png", dpi=160)
    plt.close()


def run(
    data_dir: str | Path,
    output_dir: str | Path,
    train_fraction: float,
    max_boost_train_rows: int,
    random_state: int,
) -> None:
    """Run severe-pollution correction experiments and write reports."""
    output_dir = Path(output_dir)
    report_dir = output_dir / "reports" / "severe_pollution_correction"
    report_dir.mkdir(parents=True, exist_ok=True)

    files = read_competition_files(data_dir)
    train_raw = files.get("train_raw.csv")
    if train_raw is None:
        raise FileNotFoundError("train_raw.csv is required for severe-pollution correction experiments.")

    retrain_results, retrain_predictions = run_retraining_experiments(
        train_raw=train_raw,
        train_fraction=train_fraction,
        max_boost_train_rows=max_boost_train_rows,
        random_state=random_state,
    )

    saved = load_saved_gradient_predictions(output_dir)
    saved_full_results = pd.DataFrame()
    calibration_results = pd.DataFrame()
    calibration_predictions = pd.DataFrame()
    if saved is not None:
        saved_reference_rows = []
        for col in ["weighted_ensemble", "ridge_alpha_10", "lightgbm_depth10_lr02", "catboost_depth8_lr03"]:
            if col in saved.columns:
                saved_reference_rows.append(
                    score_row(
                        f"saved_{col}",
                        saved["y_true"],
                        saved[col].to_numpy(dtype=float),
                        "saved_full_validation_reference",
                        {"source": "reports/gradient_boosting/validation_predictions.csv"},
                        evaluation_scope="full_validation",
                    )
                )
        saved_full_results = pd.DataFrame(saved_reference_rows)
        calibration_results, calibration_predictions = run_calibration_experiments(saved)

    all_results = pd.concat(
        [retrain_results, saved_full_results, calibration_results],
        ignore_index=True,
    ).sort_values(["evaluation_scope", "rmse"]).reset_index(drop=True)

    retrain_model_cols = [col for col in retrain_predictions.columns if col not in KEY_COLUMNS]
    band_frames = [all_band_metrics(retrain_predictions, retrain_model_cols)]
    if not calibration_predictions.empty:
        calibration_model_cols = [col for col in calibration_predictions.columns if col not in KEY_COLUMNS]
        band_frames.append(all_band_metrics(calibration_predictions, calibration_model_cols))
    band_metrics_df = pd.concat(band_frames, ignore_index=True)

    full_results = all_results[all_results["evaluation_scope"] == "full_validation"].sort_values("rmse").reset_index(drop=True)
    severe_full_results = all_results[all_results["evaluation_scope"] == "full_validation"].sort_values("severe_rmse").reset_index(drop=True)
    holdout_results = all_results[all_results["evaluation_scope"] == "late_validation_holdout"].sort_values("rmse").reset_index(drop=True)
    best_overall = full_results.iloc[0]["model"]
    best_severe = severe_full_results.iloc[0]["model"]
    summary = pd.DataFrame(
        [
            {
                "best_full_validation_model": best_overall,
                "best_full_validation_rmse": full_results.iloc[0]["rmse"],
                "best_full_validation_mae": full_results.iloc[0]["mae"],
                "best_full_validation_model_severe_rmse": full_results.iloc[0]["severe_rmse"],
                "best_full_validation_severe_bias": full_results.iloc[0]["severe_bias"],
                "best_full_validation_severe_model": best_severe,
                "best_full_validation_severe_model_rmse": severe_full_results.iloc[0]["severe_rmse"],
                "best_late_holdout_calibration_model": holdout_results.iloc[0]["model"] if not holdout_results.empty else None,
                "best_late_holdout_calibration_rmse": holdout_results.iloc[0]["rmse"] if not holdout_results.empty else None,
                "max_boost_train_rows": max_boost_train_rows,
                "train_fraction": train_fraction,
            }
        ]
    )

    all_predictions = retrain_predictions.copy()
    if saved is not None:
        saved_cols = [col for col in saved.columns if col not in KEY_COLUMNS]
        aligned_saved = retrain_predictions[KEY_COLUMNS].merge(
            saved[KEY_COLUMNS + saved_cols],
            on=KEY_COLUMNS,
            how="left",
            validate="one_to_one",
        )
        for col in saved_cols:
            all_predictions[f"saved_{col}"] = aligned_saved[col]
    if not calibration_predictions.empty:
        calibration_cols = [col for col in calibration_predictions.columns if col not in KEY_COLUMNS]
        aligned = retrain_predictions[KEY_COLUMNS].merge(
            calibration_predictions[KEY_COLUMNS + calibration_cols],
            on=KEY_COLUMNS,
            how="left",
            validate="one_to_one",
        )
        for col in calibration_cols:
            all_predictions[col] = aligned[col]

    summary.to_csv(report_dir / "experiment_summary.csv", index=False)
    all_results.to_csv(report_dir / "model_results.csv", index=False)
    band_metrics_df.to_csv(report_dir / "band_metrics.csv", index=False)
    all_predictions.to_csv(report_dir / "validation_predictions.csv", index=False)

    make_plots(report_dir, all_results, band_metrics_df, all_predictions, best_overall)

    print("Severe-pollution correction experiments complete.")
    print(summary.to_string(index=False))
    print("\nTop full-validation candidates:")
    print(full_results.head(12).to_string(index=False))
    print("\nBest full-validation severe-band candidates:")
    print(severe_full_results.head(12).to_string(index=False))
    if not holdout_results.empty:
        print("\nTop late-validation calibration candidates:")
        print(holdout_results.head(8).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_LOCAL_DATA_DIR), help="Directory containing train_raw.csv.")
    parser.add_argument("--output-dir", default=".", help="Directory where reports are written.")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Chronological training fraction.")
    parser.add_argument(
        "--max-boost-train-rows",
        type=int,
        default=180_000,
        help="Latest chronological rows used for LightGBM correction variants. Use 0 for all training windows.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_fraction=args.train_fraction,
        max_boost_train_rows=args.max_boost_train_rows,
        random_state=args.random_state,
    )
