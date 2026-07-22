"""Ensemble, ablation, and error-analysis workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DATETIME_COL, DEFAULT_LOCAL_DATA_DIR, RAW_NUMERIC_FEATURES, STATION_COL, TARGET
from src.data_io import read_competition_files
from src.features import add_boosting_interaction_features
from src.metrics import mae, rmse
from src.preprocessing import CausalPreprocessor, sort_raw
from src.preprocessing_window_baselines import feature_columns, make_model_preprocessor, station_metrics
from src.windows import add_window_summary_features, build_tabular_windows, chronological_cutoff


KEY_COLUMNS = ["station", "window_end_datetime", "target_datetime", "y_true"]
NON_MODEL_COLUMNS = set(KEY_COLUMNS)
POLLUTANT_FEATURES = {TARGET, "PM10", "SO2", "NO2", "CO", "O3"}
WEATHER_FEATURES = {"TEMP", "PRES", "DEWP", "RAIN", "WSPM"}
CALENDAR_FEATURES = {"hour", "month", "dayofweek", "is_weekend", "sin_hour", "cos_hour", "sin_month", "cos_month"}
ALLOWED_MODEL_PREFIXES = ("boost_", "neural_", "baseline_")


def read_prediction_table(path: Path, prefix: str) -> pd.DataFrame | None:
    """Read a validation prediction table and prefix model columns."""
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    missing = [col for col in KEY_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    keep = frame[KEY_COLUMNS].copy()
    for col in frame.columns:
        if col not in NON_MODEL_COLUMNS:
            keep[f"{prefix}_{col}"] = frame[col].astype(float)

    keep["window_end_datetime"] = pd.to_datetime(keep["window_end_datetime"])
    keep["target_datetime"] = pd.to_datetime(keep["target_datetime"])
    return keep


def merge_prediction_tables(report_root: Path) -> pd.DataFrame:
    """Merge saved predictions from baseline, boosting, and neural experiments."""
    specs = [
        ("baseline", report_root / "preprocessing_window_baselines" / "validation_predictions.csv"),
        ("boost", report_root / "gradient_boosting" / "validation_predictions.csv"),
        ("neural", report_root / "temporal_neural_models" / "validation_predictions.csv"),
    ]
    tables = [table for prefix, path in specs if (table := read_prediction_table(path, prefix)) is not None]
    if not tables:
        raise FileNotFoundError("No validation prediction tables found. Run earlier experiment pipelines first.")

    merged = tables[0]
    for table in tables[1:]:
        overlap_models = [col for col in table.columns if col not in NON_MODEL_COLUMNS and col in merged.columns]
        if overlap_models:
            raise ValueError(f"Duplicate model prediction columns after prefixing: {overlap_models}")
        merged = merged.merge(table, on=KEY_COLUMNS, how="inner", validate="one_to_one")

    if merged.empty:
        raise ValueError("Prediction tables did not share a common validation index.")
    return merged.sort_values(["target_datetime", "station"]).reset_index(drop=True)


def model_columns(predictions: pd.DataFrame) -> list[str]:
    """Return prediction columns."""
    return [col for col in predictions.columns if col not in NON_MODEL_COLUMNS and col.startswith(ALLOWED_MODEL_PREFIXES)]


def score_predictions(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> tuple[float, float]:
    """Return RMSE and MAE for one prediction vector."""
    return rmse(y_true, y_pred), mae(y_true, y_pred)


def individual_model_metrics(predictions: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Compute full-validation metrics for all available models."""
    rows = []
    for col in columns:
        row_rmse, row_mae = score_predictions(predictions["y_true"], predictions[col])
        rows.append({"model": col, "rmse": row_rmse, "mae": row_mae, "rows": len(predictions)})
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def blend_split_mask(predictions: pd.DataFrame, blend_train_fraction: float) -> pd.Series:
    """Chronologically split validation rows into blend-train and blend-evaluation periods."""
    unique_times = predictions["target_datetime"].drop_duplicates().sort_values().reset_index(drop=True)
    split_index = max(1, min(len(unique_times) - 1, int(len(unique_times) * blend_train_fraction)))
    split_time = unique_times.iloc[split_index - 1]
    return predictions["target_datetime"] <= split_time


def fit_positive_blend(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit non-negative linear blend weights and normalize them."""
    model = LinearRegression(positive=True, fit_intercept=False)
    model.fit(X, y)
    weights = np.clip(model.coef_.astype(float), 0.0, None)
    if weights.sum() <= 0:
        weights = np.ones(X.shape[1], dtype=float)
    return weights / weights.sum()


def build_ensembles(predictions: pd.DataFrame, columns: list[str], blend_train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create simple and learned ensembles, evaluated on the later validation block."""
    individual = individual_model_metrics(predictions, columns)
    ranked_cols = individual["model"].tolist()
    top_cols = ranked_cols[: min(5, len(ranked_cols))]

    blend_train_mask = blend_split_mask(predictions, blend_train_fraction)
    blend_eval = predictions.loc[~blend_train_mask].copy()
    if blend_eval.empty:
        raise ValueError("Blend evaluation split is empty. Lower --blend-train-fraction.")

    ensemble_predictions = predictions[KEY_COLUMNS].copy()
    rows = []

    for k in [2, 3, 5]:
        selected = ranked_cols[: min(k, len(ranked_cols))]
        if len(selected) < 2:
            continue
        name = f"ensemble_mean_top_{len(selected)}"
        ensemble_predictions[name] = predictions[selected].mean(axis=1)
        eval_pred = ensemble_predictions.loc[~blend_train_mask, name]
        row_rmse, row_mae = score_predictions(blend_eval["y_true"], eval_pred)
        rows.append(
            {
                "model": name,
                "rmse": row_rmse,
                "mae": row_mae,
                "rows": len(blend_eval),
                "evaluation_window": "late_validation_holdout",
                "weights": json.dumps({col: round(1.0 / len(selected), 6) for col in selected}),
            }
        )

    inv_cols = top_cols
    inv_scores = individual.set_index("model").loc[inv_cols, "rmse"].to_numpy()
    inv_weights = 1.0 / np.maximum(inv_scores, 1e-9)
    inv_weights = inv_weights / inv_weights.sum()
    ensemble_predictions["ensemble_inverse_rmse_top_5"] = predictions[inv_cols].to_numpy() @ inv_weights
    eval_pred = ensemble_predictions.loc[~blend_train_mask, "ensemble_inverse_rmse_top_5"]
    row_rmse, row_mae = score_predictions(blend_eval["y_true"], eval_pred)
    rows.append(
        {
            "model": "ensemble_inverse_rmse_top_5",
            "rmse": row_rmse,
            "mae": row_mae,
            "rows": len(blend_eval),
            "evaluation_window": "late_validation_holdout",
            "weights": json.dumps(dict(zip(inv_cols, np.round(inv_weights, 6)))),
        }
    )

    X_blend_train = predictions.loc[blend_train_mask, top_cols].to_numpy(dtype=float)
    y_blend_train = predictions.loc[blend_train_mask, "y_true"].to_numpy(dtype=float)
    learned_weights = fit_positive_blend(X_blend_train, y_blend_train)
    ensemble_predictions["ensemble_positive_linear_top_5"] = predictions[top_cols].to_numpy(dtype=float) @ learned_weights
    eval_pred = ensemble_predictions.loc[~blend_train_mask, "ensemble_positive_linear_top_5"]
    row_rmse, row_mae = score_predictions(blend_eval["y_true"], eval_pred)
    rows.append(
        {
            "model": "ensemble_positive_linear_top_5",
            "rmse": row_rmse,
            "mae": row_mae,
            "rows": len(blend_eval),
            "evaluation_window": "late_validation_holdout",
            "weights": json.dumps(dict(zip(top_cols, np.round(learned_weights, 6)))),
        }
    )

    for col in top_cols:
        row_rmse, row_mae = score_predictions(blend_eval["y_true"], blend_eval[col])
        rows.append(
            {
                "model": f"reference_{col}",
                "rmse": row_rmse,
                "mae": row_mae,
                "rows": len(blend_eval),
                "evaluation_window": "late_validation_holdout",
                "weights": json.dumps({col: 1.0}),
            }
        )

    result = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    return result, ensemble_predictions


def median_impute(raw: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Impute every numeric feature from training medians, then wind direction mode."""
    out = sort_raw(raw)
    train = out[out[DATETIME_COL] <= cutoff]
    numeric_cols = [col for col in RAW_NUMERIC_FEATURES if col in out.columns]
    out[numeric_cols] = out[numeric_cols].fillna(train[numeric_cols].median(numeric_only=True))
    if "wd" in out.columns:
        mode = train["wd"].mode(dropna=True)
        out["wd"] = out["wd"].fillna(str(mode.iloc[0]) if len(mode) else "UNKNOWN")
    return out


def window_interpolated_windows(raw: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Build raw windows, then interpolate missing lags inside each 24-hour input window.

    This avoids using values after the prediction window. A missing lag may be
    filled from other observed lag positions in the same 24-hour input, then
    remaining gaps fall back to training-period medians.
    """
    sorted_raw = sort_raw(raw)
    train = sorted_raw[sorted_raw[DATETIME_COL] <= cutoff]
    numeric_cols = [col for col in RAW_NUMERIC_FEATURES if col in sorted_raw.columns]
    medians = train[numeric_cols].median(numeric_only=True)

    windows = build_tabular_windows(sorted_raw, sorted_raw, window_size=24)
    for feature in numeric_cols:
        lag_cols = [f"{feature}_lag_{lag}" for lag in range(24, 0, -1)]
        lag_cols = [col for col in lag_cols if col in windows.columns]
        if not lag_cols:
            continue
        interpolated = windows[lag_cols].astype(float).interpolate(axis=1, limit_direction="both")
        windows[lag_cols] = interpolated.fillna(float(medians.get(feature, 0.0)))

    if "wd_lag_1" in windows.columns:
        mode = train["wd"].mode(dropna=True) if "wd" in train.columns else pd.Series(dtype=object)
        windows["wd_lag_1"] = windows["wd_lag_1"].fillna(str(mode.iloc[0]) if len(mode) else "UNKNOWN")

    windows = add_window_summary_features(windows, window_size=24)
    return add_boosting_interaction_features(windows)


def build_windows_for_imputation(raw: pd.DataFrame, cutoff: pd.Timestamp, method: str) -> pd.DataFrame:
    """Build feature windows under a selected imputation strategy."""
    if method == "causal_ffill":
        imputed = CausalPreprocessor.fit(raw[raw[DATETIME_COL] <= cutoff]).transform(raw)
    elif method == "median":
        imputed = median_impute(raw, cutoff)
    elif method == "window_interpolation":
        return window_interpolated_windows(raw, cutoff)
    else:
        raise ValueError(f"Unknown imputation method: {method}")
    windows = build_tabular_windows(imputed, raw, window_size=24)
    windows = add_window_summary_features(windows, window_size=24)
    return add_boosting_interaction_features(windows)


def filter_feature_group(numeric_cols: list[str], categorical_cols: list[str], feature_group: str) -> tuple[list[str], list[str]]:
    """Select columns for feature-group ablations."""
    if feature_group == "all_features":
        return numeric_cols, categorical_cols

    pollutant_prefixes = ("pm25", "pm2.5", "pm10", "so2", "no2", "co", "o3")
    weather_prefixes = ("temp", "pres", "dewp", "rain", "wspm", "wind", "ventilation")

    def belongs_to(prefixes: tuple[str, ...], col: str) -> bool:
        normalized = col.lower().replace(".", "")
        return any(normalized == prefix.replace(".", "") or normalized.startswith(f"{prefix.replace('.', '')}_") for prefix in prefixes)

    selected_numeric = []
    for col in numeric_cols:
        base = col.split("_lag_")[0]
        is_pollutant = base in POLLUTANT_FEATURES or belongs_to(pollutant_prefixes, col)
        is_weather = base in WEATHER_FEATURES or belongs_to(weather_prefixes, col)
        is_calendar = col in CALENDAR_FEATURES

        if feature_group == "pollution_only" and is_pollutant:
            selected_numeric.append(col)
        elif feature_group == "pollution_weather" and (is_pollutant or is_weather):
            selected_numeric.append(col)
        elif feature_group == "pollution_weather_calendar" and (is_pollutant or is_weather or is_calendar):
            selected_numeric.append(col)

    selected_categorical = []
    if feature_group == "all_features":
        selected_categorical = categorical_cols
    return selected_numeric, selected_categorical


def fit_ridge_ablation(train_df: pd.DataFrame, valid_df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> np.ndarray:
    """Fit a fast Ridge model for ablation comparisons."""
    features = numeric_cols + categorical_cols
    if not features:
        raise ValueError("Ablation produced no feature columns.")
    pipeline = Pipeline(
        [
            ("preprocess", make_model_preprocessor(numeric_cols, categorical_cols, scale_numeric=True)),
            ("model", Ridge(alpha=10.0, random_state=42)),
        ]
    )
    pipeline.fit(train_df[features], train_df["target_pm25"].astype(float))
    return pipeline.predict(valid_df[features])


def run_ablation_studies(train_raw: pd.DataFrame, train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run imputation and feature-group ablations with a fixed Ridge estimator."""
    raw = sort_raw(train_raw)
    cutoff = chronological_cutoff(raw, train_fraction=train_fraction)
    rows = []
    prediction_frames = []

    for imputation in ["causal_ffill", "window_interpolation", "median"]:
        windows = build_windows_for_imputation(raw, cutoff, imputation)
        train_mask = pd.to_datetime(windows["window_end_datetime"]) <= cutoff
        train_df = windows.loc[train_mask].copy()
        valid_df = windows.loc[~train_mask].copy()
        y_valid = valid_df["target_pm25"].astype(float)
        numeric_cols, categorical_cols = feature_columns(windows)

        feature_groups = ["all_features"] if imputation != "causal_ffill" else [
            "pollution_only",
            "pollution_weather",
            "pollution_weather_calendar",
            "all_features",
        ]
        for feature_group in feature_groups:
            selected_numeric, selected_categorical = filter_feature_group(numeric_cols, categorical_cols, feature_group)
            pred = fit_ridge_ablation(train_df, valid_df, selected_numeric, selected_categorical)
            row_rmse, row_mae = score_predictions(y_valid, pred)
            model_name = f"ridge_{imputation}_{feature_group}"
            rows.append(
                {
                    "model": model_name,
                    "imputation": imputation,
                    "feature_group": feature_group,
                    "rmse": row_rmse,
                    "mae": row_mae,
                    "train_windows": len(train_df),
                    "validation_windows": len(valid_df),
                    "numeric_features": len(selected_numeric),
                    "categorical_features": len(selected_categorical),
                }
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "station": valid_df[STATION_COL].to_numpy(),
                        "window_end_datetime": valid_df["window_end_datetime"].to_numpy(),
                        "target_datetime": valid_df["target_datetime"].to_numpy(),
                        "y_true": y_valid.to_numpy(),
                        "model": model_name,
                        "prediction": pred,
                    }
                )
            )

    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True), pd.concat(prediction_frames, ignore_index=True)


def pollution_band_metrics(predictions: pd.DataFrame, selected_models: list[str]) -> pd.DataFrame:
    """Compute error metrics by PM2.5 concentration band."""
    bins = [-np.inf, 35.0, 75.0, 150.0, np.inf]
    labels = ["low_<=35", "moderate_35_75", "high_75_150", "severe_>150"]
    banded = predictions.copy()
    banded["pm25_band"] = pd.cut(banded["y_true"], bins=bins, labels=labels)
    rows = []
    for model in selected_models:
        for band, group in banded.groupby("pm25_band", observed=True):
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


def residual_summary(predictions: pd.DataFrame, selected_models: list[str]) -> pd.DataFrame:
    """Summarize bias and residual spread for selected models."""
    rows = []
    for model in selected_models:
        residual = predictions[model] - predictions["y_true"]
        rows.append(
            {
                "model": model,
                "bias_mean": float(residual.mean()),
                "residual_std": float(residual.std()),
                "p05": float(residual.quantile(0.05)),
                "p50": float(residual.quantile(0.50)),
                "p95": float(residual.quantile(0.95)),
                "underprediction_rate": float((residual < 0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("residual_std").reset_index(drop=True)


def make_plots(
    report_dir: Path,
    model_metrics: pd.DataFrame,
    ensemble_results: pd.DataFrame,
    station_results: pd.DataFrame,
    band_results: pd.DataFrame,
    ablation_results: pd.DataFrame,
    predictions: pd.DataFrame,
    best_model: str,
) -> None:
    """Save analysis plots for the report."""
    sns.set_theme(style="whitegrid")
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    top_models = model_metrics.head(10)
    sns.barplot(data=top_models, y="model", x="rmse", color="#4C78A8")
    plt.title("Top Individual Model RMSE")
    plt.xlabel("RMSE")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(figures_dir / "top_model_rmse.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 4.8))
    sns.barplot(data=ensemble_results.head(8), y="model", x="rmse", color="#59A14F")
    plt.title("Late-Validation Ensemble Comparison")
    plt.xlabel("RMSE")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(figures_dir / "ensemble_rmse.png", dpi=160)
    plt.close()

    station_subset = station_results[station_results["model"].isin([best_model])].sort_values("rmse", ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=station_subset, y="station", x="rmse", color="#F28E2B")
    plt.title(f"Station-Wise RMSE: {best_model}")
    plt.xlabel("RMSE")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(figures_dir / "station_rmse_best_model.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    selected_band = band_results[band_results["model"] == best_model]
    sns.barplot(data=selected_band, x="pm25_band", y="rmse", color="#E15759")
    plt.title(f"PM2.5 Band RMSE: {best_model}")
    plt.xlabel("True PM2.5 band")
    plt.ylabel("RMSE")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(figures_dir / "pm25_band_rmse_best_model.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=ablation_results, y="model", x="rmse", color="#B07AA1")
    plt.title("Ridge Ablation RMSE")
    plt.xlabel("RMSE")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(figures_dir / "ablation_rmse.png", dpi=160)
    plt.close()

    sample = predictions.sample(min(len(predictions), 8000), random_state=42)
    residual = sample[best_model] - sample["y_true"]
    plt.figure(figsize=(9, 5))
    sns.scatterplot(x=sample["y_true"], y=residual, s=8, alpha=0.25, edgecolor=None)
    plt.axhline(0, color="black", linewidth=1)
    plt.title(f"Residuals vs True PM2.5: {best_model}")
    plt.xlabel("True PM2.5")
    plt.ylabel("Prediction - true")
    plt.tight_layout()
    plt.savefig(figures_dir / "residuals_best_model.png", dpi=160)
    plt.close()


def run(
    data_dir: str | Path,
    output_dir: str | Path,
    train_fraction: float,
    blend_train_fraction: float,
) -> None:
    """Run the full ensemble, ablation, and error-analysis workflow."""
    output_dir = Path(output_dir)
    report_root = output_dir / "reports"
    report_dir = report_root / "ensemble_ablation_error_analysis"
    report_dir.mkdir(parents=True, exist_ok=True)

    predictions = merge_prediction_tables(report_root)
    cols = model_columns(predictions)
    if len(cols) < 2:
        raise ValueError("Need at least two model prediction columns for ensembling.")

    model_metrics = individual_model_metrics(predictions, cols)
    ensemble_results, ensemble_predictions = build_ensembles(predictions, cols, blend_train_fraction)
    combined = predictions.merge(ensemble_predictions.drop(columns=KEY_COLUMNS), left_index=True, right_index=True)

    ensemble_cols = [col for col in ensemble_predictions.columns if col not in NON_MODEL_COLUMNS]
    selected_models = model_metrics.head(4)["model"].tolist() + ensemble_results.head(2)["model"].tolist()
    selected_models = list(dict.fromkeys([model for model in selected_models if model in combined.columns]))
    best_model = ensemble_results.iloc[0]["model"] if ensemble_results.iloc[0]["model"] in combined.columns else model_metrics.iloc[0]["model"]

    station_results = station_metrics(combined, selected_models)
    band_results = pollution_band_metrics(combined, selected_models)
    residual_results = residual_summary(combined, selected_models)

    files = read_competition_files(data_dir)
    train_raw = files.get("train_raw.csv")
    if train_raw is None:
        raise FileNotFoundError("train_raw.csv is required for ablation studies.")
    ablation_results, ablation_predictions = run_ablation_studies(train_raw, train_fraction)

    summary = pd.DataFrame(
        [
            {
                "prediction_rows": len(predictions),
                "available_prediction_models": len(cols),
                "ensemble_prediction_columns": len(ensemble_cols),
                "best_individual_model": model_metrics.iloc[0]["model"],
                "best_individual_rmse": model_metrics.iloc[0]["rmse"],
                "best_late_validation_ensemble": ensemble_results.iloc[0]["model"],
                "best_late_validation_ensemble_rmse": ensemble_results.iloc[0]["rmse"],
                "best_ablation": ablation_results.iloc[0]["model"],
                "best_ablation_rmse": ablation_results.iloc[0]["rmse"],
                "train_fraction": train_fraction,
                "blend_train_fraction": blend_train_fraction,
            }
        ]
    )

    summary.to_csv(report_dir / "analysis_summary.csv", index=False)
    model_metrics.to_csv(report_dir / "individual_model_metrics.csv", index=False)
    ensemble_results.to_csv(report_dir / "ensemble_results.csv", index=False)
    station_results.to_csv(report_dir / "station_error_metrics.csv", index=False)
    band_results.to_csv(report_dir / "pm25_band_error_metrics.csv", index=False)
    residual_results.to_csv(report_dir / "residual_summary.csv", index=False)
    ablation_results.to_csv(report_dir / "ablation_results.csv", index=False)
    ablation_predictions.to_csv(report_dir / "ablation_predictions.csv", index=False)
    combined.to_csv(report_dir / "combined_validation_predictions.csv", index=False)

    make_plots(
        report_dir=report_dir,
        model_metrics=model_metrics,
        ensemble_results=ensemble_results,
        station_results=station_results,
        band_results=band_results,
        ablation_results=ablation_results,
        predictions=combined,
        best_model=best_model,
    )

    print("Ensemble, ablation, and error analysis complete.")
    print(summary.to_string(index=False))
    print("\nTop individual models:")
    print(model_metrics.head(8).to_string(index=False))
    print("\nLate-validation ensemble comparison:")
    print(ensemble_results.head(8).to_string(index=False))
    print("\nAblation results:")
    print(ablation_results.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_LOCAL_DATA_DIR), help="Directory containing train_raw.csv.")
    parser.add_argument("--output-dir", default=".", help="Directory where reports are written.")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Chronological train fraction for ablations.")
    parser.add_argument(
        "--blend-train-fraction",
        type=float,
        default=0.5,
        help="Fraction of validation period used to learn blend weights; later rows evaluate the blend.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_fraction=args.train_fraction,
        blend_train_fraction=args.blend_train_fraction,
    )
