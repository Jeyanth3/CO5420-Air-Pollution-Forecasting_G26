"""Optuna hyperparameter tuning for LightGBM."""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from src.config import DEFAULT_LOCAL_DATA_DIR
from src.data_io import read_competition_files
from src.rmse_improvement_error_analysis import (
    build_compact_test_features,
    build_compact_windows,
    prepare_feature_matrices,
)


def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, categorical_cols: list[str]) -> float:
    """Optuna objective function for LightGBM with TimeSeriesSplit."""
    param = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_state": 42,
        "n_jobs": -1,
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 256),
        "max_depth": trial.suggest_int("max_depth", 6, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }

    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    
    for train_index, valid_index in tscv.split(X):
        X_tr, X_val = X.iloc[train_index], X.iloc[valid_index]
        y_tr, y_val = y.iloc[train_index], y.iloc[valid_index]
        
        model = lgb.LGBMRegressor(**param)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            categorical_feature=categorical_cols,
        )

        preds = np.clip(model.predict(X_val), 0.0, None)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        scores.append(rmse)
        
    return float(np.mean(scores))


def run(data_dir: Path, output_dir: Path, n_trials: int, train_fraction: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data for tuning...")
    files = read_competition_files(data_dir)
    train_raw = files["train_raw.csv"]
    test = files["test.csv"]

    # Use a large train_fraction to use the entire dataset for cross validation
    train_df, valid_df, _ = build_compact_windows(train_raw, train_fraction=0.999)
    test_features = build_compact_test_features(test, train_raw)
    numeric_cols, categorical_cols, train_df, valid_df, _ = prepare_feature_matrices(train_df, valid_df, test_features)

    # Combine back for TimeSeriesSplit
    all_train = pd.concat([train_df, valid_df], ignore_index=True)
    
    features = numeric_cols + categorical_cols
    X = all_train[features].copy()
    y = all_train["target_pm25"].astype(float)

    for col in categorical_cols:
        X[col] = X[col].astype("category")

    print(f"Starting Optuna tuning with {n_trials} trials (GPU-accelerated, 3-fold TSCV)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X, y, categorical_cols), n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (RMSE): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    best_params_path = report_dir / "best_lgbm_params.json"
    with open(best_params_path, "w") as f:
        json.dump(trial.params, f, indent=4)
    print(f"Saved best parameters to {best_params_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Tuning for LightGBM")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_LOCAL_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    args = parser.parse_args()

    run(args.data_dir, args.output_dir, args.n_trials, args.train_fraction)
