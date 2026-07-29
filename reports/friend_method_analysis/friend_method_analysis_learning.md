# Friend Method Analysis

## What changed in GitHub

The latest merged branch from `Jeyanth_analysis` improved the tabular boosting pipeline rather than replacing it with a neural model.

Main changes:

- Added Optuna-style LightGBM hyperparameter tuning support in `src/optuna_tuning.py`.
- Added tuned LightGBM parameters in `reports/best_lgbm_params.json`.
- Added extra temporal and physical features:
  - wind-vector components from wind speed and direction
  - PM2.5 short-term differences
  - pollutant ratios such as O3/NO2 and SO2/NO2
  - temperature acceleration
  - wind-speed squared
  - PM2.5 standard deviation over 6h and 12h
  - TEMP and WSPM 24h standard deviation
- Added XGBoost, CatBoost, simple averaging, and Ridge stacking inside the RMSE-improvement workflow.

## What we were lacking

Our earlier strongest submission was a good hand-tuned LightGBM, but it was too dependent on the original compact feature set. The missing piece was not only "more models"; it was better causal feature representation inside the same 24-hour window.

The added wind, volatility, ratio, and short-term change features let LightGBM see rapid pollution dynamics more directly. That is why the same depth-10 LightGBM configuration improved after the feature update.

## Reproduced result

I reran the latest workflow locally with CPU-safe XGBoost/CatBoost settings. The strongest reproduced candidate was:

```text
lgbm_compact_depth10_regularized_first80
```

Diagnostic official-year score, used only for analysis:

```text
RMSE: 15.408241
MAE: 8.937097
Severe RMSE: 29.071941
Severe bias: -5.684800
```

This improved over our previous diagnostic reference around `15.4825`.

## Candidate submissions

Primary next submission:

```text
submissions/submission_friend_features_lgbm_depth10.csv
```

Validation-selected probe:

```text
submissions/submission_friend_features_validation_selected.csv
```

Other generated candidates:

```text
submissions/submission_optuna_tuned.csv
submissions/submission_meta_ensemble.csv
submissions/submission_simple_average.csv
```

## Recommendation

Submit `submission_friend_features_lgbm_depth10.csv` first. It is the cleanest continuation of the already strong model, but with better legal features.

Use `submission_friend_features_validation_selected.csv` only as a second probe. It won the chronological validation-only cutoff sweep, but this competition has already shown that chronological validation and public leaderboard scores do not always move together.

Do not lead with the stacking candidate. It overfit validation by assigning high weight to Ridge and performed worse on the official-year diagnostic.
