"""Fast optimized ensemble to beat RMSE 13.54 on Kaggle.

Strategy:
- Use city-context features (already the best feature set)
- Run Optuna tuning (50 trials) on LightGBM with city context, TimeSeriesSplit CV
- Train final models on ALL train_raw data with tuned params + bagging
- Generate a strong blended submission CSV
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.config import DEFAULT_LOCAL_DATA_DIR
from src.data_io import read_competition_files
from src.features import add_boosting_interaction_features
from src.final_objective_experiments import add_datetime_from_lag1, rowwise_impute_test
from src.friend_feature_final_candidates import add_city_context_features, build_all_train_features
from src.gradient_boosting_models import optional_imports
from src.metrics import rmse
from src.preprocessing import CausalPreprocessor, sort_raw
from src.preprocessing_window_baselines import feature_columns
from src.rmse_improvement_error_analysis import (
    align_official_test_target,
    build_compact_test_features,
    build_compact_windows,
    prepare_feature_matrices,
)
from src.windows import add_window_summary_features, build_tabular_windows

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


def run_optuna_lgbm(train_df: pd.DataFrame, numeric_cols: list, categorical_cols: list, n_trials: int = 50) -> dict:
    """Run Optuna to find best LightGBM hyperparameters using TimeSeriesSplit."""
    lgb = optional_imports().get("lightgbm")
    if lgb is None:
        raise ImportError("lightgbm required")
    
    features = numeric_cols + categorical_cols
    X = train_df[features].copy()
    y = train_df["target_pm25"].astype(float)
    for col in categorical_cols:
        X[col] = X[col].astype("category")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "n_jobs": -1,
            "random_state": 42,
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 63, 255),
            "max_depth": trial.suggest_int("max_depth", 6, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 15.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        }
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        for tr_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr, y_tr, categorical_feature=categorical_cols)
            pred = np.clip(model.predict(X_val), 0.0, None)
            scores.append(rmse(y_val, pred))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    best["objective"] = "regression"
    best["n_jobs"] = -1
    best["verbosity"] = -1
    best["random_state"] = 42
    print(f"  Optuna best CV RMSE: {study.best_value:.4f}")
    print(f"  Best params: {best}")
    return best


def fit_lgbm_full(lgb, name: str, train_df: pd.DataFrame, test_df: pd.DataFrame, features: list, categorical_cols: list, params: dict) -> np.ndarray:
    X_train = train_df[features].copy()
    X_test = test_df[features].copy()
    for col in categorical_cols:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")
    t = time.perf_counter()
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, train_df["target_pm25"].astype(float), categorical_feature=categorical_cols)
    pred = np.clip(model.predict(X_test), 0.0, None)
    print(f"  {name}: {time.perf_counter()-t:.1f}s, mean_pred={pred.mean():.2f}")
    return pred


def fit_xgb_full(xgb, name: str, train_df: pd.DataFrame, test_df: pd.DataFrame, features: list, categorical_cols: list) -> np.ndarray:
    X_train = train_df[features].copy()
    X_test = test_df[features].copy()
    for col in categorical_cols:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")
    t = time.perf_counter()
    model = xgb.XGBRegressor(
        n_estimators=400, learning_rate=0.02, max_depth=8,
        subsample=0.85, colsample_bytree=0.8, min_child_weight=5,
        enable_categorical=True, random_state=42, tree_method="hist",
        device="cpu",
    )
    model.fit(X_train, train_df["target_pm25"].astype(float))
    pred = np.clip(model.predict(X_test), 0.0, None)
    print(f"  {name}: {time.perf_counter()-t:.1f}s, mean_pred={pred.mean():.2f}")
    return pred


def fit_catboost_full(cb, name: str, train_df: pd.DataFrame, test_df: pd.DataFrame, features: list, categorical_cols: list) -> np.ndarray:
    X_train = train_df[features].copy()
    X_test = test_df[features].copy()
    for col in categorical_cols:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)
    t = time.perf_counter()
    model = cb.CatBoostRegressor(
        iterations=400, learning_rate=0.03, depth=8,
        random_seed=42, verbose=False, task_type="CPU",
    )
    model.fit(X_train, train_df["target_pm25"].astype(float), cat_features=categorical_cols)
    pred = np.clip(model.predict(X_test), 0.0, None)
    print(f"  {name}: {time.perf_counter()-t:.1f}s, mean_pred={pred.mean():.2f}")
    return pred


def bagged_lgbm(lgb, n_bags: int, params: dict, train_df: pd.DataFrame, test_df: pd.DataFrame, features: list, categorical_cols: list) -> np.ndarray:
    """Train n_bags LightGBM models with different seeds and average."""
    all_preds = []
    X_test = test_df[features].copy()
    for col in categorical_cols:
        X_test[col] = X_test[col].astype("category")
    for i in range(n_bags):
        seed_params = {**params, "random_state": 42 + i * 13}
        X_train = train_df[features].copy()
        for col in categorical_cols:
            X_train[col] = X_train[col].astype("category")
        model = lgb.LGBMRegressor(**seed_params)
        model.fit(X_train, train_df["target_pm25"].astype(float), categorical_feature=categorical_cols)
        all_preds.append(np.clip(model.predict(X_test), 0.0, None))
    return np.mean(all_preds, axis=0)


def run(data_dir: Path, output_dir: Path, n_optuna_trials: int = 50, n_bags: int = 5) -> None:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    submissions_dir = output_dir / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)
    report_dir = output_dir / "reports" / "optimized_ensemble"
    report_dir.mkdir(parents=True, exist_ok=True)

    imports = optional_imports()
    lgb = imports.get("lightgbm")
    xgb = imports.get("xgboost")
    cb = imports.get("catboost")
    if lgb is None:
        raise ImportError("lightgbm required")

    print("Loading data...")
    files = read_competition_files(data_dir)
    train_raw = files["train_raw.csv"]
    test = files["test.csv"]
    test_raw = files.get("test_raw.csv")
    sample_submission = files["sample_submission.csv"]

    # Align ground truth for evaluation
    has_target = test_raw is not None
    if has_target:
        aligned = align_official_test_target(test, test_raw)
        y_true = aligned["y_true"].to_numpy(dtype=float)
        aligned_ids = aligned["id"].to_numpy()
    
    print("\nBuilding features (full training set)...")
    t0 = time.perf_counter()
    train_df, test_features, numeric_cols, categorical_cols = build_all_train_features(train_raw, test)
    features = numeric_cols + categorical_cols
    print(f"  Features built: {len(features)} cols, {len(train_df)} train rows in {time.perf_counter()-t0:.1f}s")

    print(f"\nRunning Optuna LightGBM tuning ({n_optuna_trials} trials)...")
    t1 = time.perf_counter()
    best_lgbm_params = run_optuna_lgbm(train_df, numeric_cols, categorical_cols, n_trials=n_optuna_trials)
    print(f"  Tuning done in {time.perf_counter()-t1:.1f}s")

    # Save best params
    with open(report_dir / "best_lgbm_params_optimized.json", "w") as f:
        json.dump(best_lgbm_params, f, indent=2)

    print("\nTraining final models on ALL training data...")
    predictions = {}

    print("  [1/5] LightGBM depth10 (baseline)...")
    base_params = {
        "objective": "regression", "n_estimators": 320, "learning_rate": 0.025,
        "num_leaves": 127, "max_depth": 10, "min_child_samples": 60,
        "subsample": 0.9, "colsample_bytree": 0.85, "reg_lambda": 6.0,
        "n_jobs": -1, "verbosity": -1, "random_state": 42,
    }
    predictions["lgbm_depth10"] = fit_lgbm_full(lgb, "lgbm_depth10", train_df, test_features, features, categorical_cols, base_params)

    print(f"  [2/5] LightGBM Optuna-tuned (bagged x{n_bags})...")
    predictions["lgbm_optuna_bagged"] = bagged_lgbm(lgb, n_bags, best_lgbm_params, train_df, test_features, features, categorical_cols)
    print(f"    mean_pred={predictions['lgbm_optuna_bagged'].mean():.2f}")

    if xgb:
        print("  [3/5] XGBoost...")
        predictions["xgboost"] = fit_xgb_full(xgb, "xgboost", train_df, test_features, features, categorical_cols)
    
    if cb:
        print("  [4/5] CatBoost...")
        predictions["catboost"] = fit_catboost_full(cb, "catboost", train_df, test_features, features, categorical_cols)

    # Final blend weights
    print("  [5/5] Building optimized blend...")
    blend_parts = []
    blend_weights = []

    # Tuned LightGBM bagged gets highest weight
    blend_parts.append(predictions["lgbm_optuna_bagged"])
    blend_weights.append(0.50)

    # Baseline LightGBM depth10
    blend_parts.append(predictions["lgbm_depth10"])
    blend_weights.append(0.20)

    if "xgboost" in predictions:
        blend_parts.append(predictions["xgboost"])
        blend_weights.append(0.15)

    if "catboost" in predictions:
        blend_parts.append(predictions["catboost"])
        blend_weights.append(0.15)

    # Normalize weights
    total = sum(blend_weights)
    blend_weights = [w / total for w in blend_weights]
    
    final_blend = np.zeros(len(blend_parts[0]))
    for pred, w in zip(blend_parts, blend_weights):
        final_blend += w * pred
    final_blend = np.clip(final_blend, 0.0, None)
    predictions["optimized_blend"] = final_blend
    print(f"    blend mean_pred={final_blend.mean():.2f}")

    # Evaluate locally if test_raw available
    results = []
    if has_target:
        sub_temp = pd.DataFrame({"id": test["id"].to_numpy(), "pred": final_blend})
        merged = aligned[["id", "y_true"]].merge(sub_temp, on="id", how="left")
        local_rmse = rmse(merged["y_true"].to_numpy(), merged["pred"].to_numpy())
        print(f"\n*** LOCAL DIAGNOSTIC RMSE (optimized_blend): {local_rmse:.4f} ***")
        for name, pred_arr in predictions.items():
            sub_t = pd.DataFrame({"id": test["id"].to_numpy(), "pred": pred_arr})
            m = aligned[["id", "y_true"]].merge(sub_t, on="id", how="left")
            r = rmse(m["y_true"].to_numpy(), m["pred"].to_numpy())
            results.append({"model": name, "local_rmse": r})
        results_df = pd.DataFrame(results).sort_values("local_rmse")
        print("\nAll model local RMSEs:")
        print(results_df.to_string(index=False))
        results_df.to_csv(report_dir / "model_comparison.csv", index=False)

    # Write submission files
    print("\nWriting submission CSVs...")
    for name, pred_arr in predictions.items():
        out = sample_submission[["id"]].copy()
        out["PM2.5"] = np.clip(pred_arr, 0.0, None)
        path = submissions_dir / f"submission_{name}.csv"
        out.to_csv(path, index=False)
        print(f"  Wrote: {path.name}")
    
    best_path = submissions_dir / "submission_BEST_optimized_blend.csv"
    out = sample_submission[["id"]].copy()
    out["PM2.5"] = np.clip(final_blend, 0.0, None)
    out.to_csv(best_path, index=False)
    print(f"\n>>> BEST SUBMISSION: {best_path} <<<")
    return best_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_LOCAL_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-bags", type=int, default=5)
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.n_trials, args.n_bags)
