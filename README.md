# CO5420 Air Pollution Forecasting

Forecast next-hour PM2.5 from the previous 24 hours of pollution, weather,
calendar, and station data.

This repository contains the Group 26 workflow for the Kaggle competition
**CO5420 Air Pollution Forecasting Using Temporal NNs**. The official metric is
RMSE. We also report MAE, station-wise error, and severe-pollution error because
they explain where the model is useful and where it still fails.

## Problem

Each Kaggle row is one flattened 24-hour window from a Beijing air-quality
station.

```text
24-hour input window -> PM2.5 one hour later
```

The source is the Beijing Multi-Site Air Quality Dataset. It contains 12
stations from March 2013 to February 2017. The competition training file covers
the first three years. The final prediction file contains 4,103 windows.

## Repository Layout

```text
docs/        project roadmap and planning notes
notebooks/   reproducible notebook entry points
reports/     experiment notes, tables, and figures
src/         reusable preprocessing, modelling, and analysis code
submissions/ selected Kaggle-format candidate submissions
```

Raw data and generated training artifacts are not committed. Put Kaggle files in
`data/raw/` when running locally.

Expected files:

```text
train_raw.csv
test.csv
sample_submission.csv
```

`test_raw.csv` is used only for extended analysis. It must not be used to train
or tune a Kaggle submission because it can reveal hidden targets by alignment.

## Method Progression

We built the project in stages. Each stage had a clear reason.

| Stage | Main idea | Best local / diagnostic result |
|---|---|---:|
| Persistence baseline | Predict the latest observed PM2.5 | RMSE 21.4431 |
| Classical baselines | Rolling means, Ridge, Random Forest, ExtraTrees | Ridge RMSE 20.1097 |
| Boosting models | LightGBM, XGBoost, CatBoost with lag features | Ensemble RMSE 19.5651 |
| Temporal neural models | LSTM, GRU, CNN-LSTM | CNN-LSTM RMSE 21.4933 |
| Severe-pollution correction | Target high PM2.5 underprediction | Severe-band RMSE improved to 37.5473 |
| Official-file audit | Compact LightGBM on official files | Diagnostic RMSE 15.4825 |
| Team feature update | Wind vectors, ratios, volatility, tuned LightGBM | Diagnostic RMSE 15.4082 |
| Final candidates | City-context LightGBM and robustness blends | See `submissions/` |

The main lesson was simple: for this one-hour forecast, strong tabular boosting
with carefully designed lag, rolling, weather, and station features outperformed
the early neural models.

## Algorithms Used

- Persistence and rolling mean baselines.
- Ridge regression on flattened windows.
- Random Forest and ExtraTrees.
- LightGBM with chronological validation.
- XGBoost and CatBoost as diversity checks.
- Weighted averaging and Ridge-style stacking.
- LSTM, GRU, and CNN-LSTM temporal neural networks.
- Severe-pollution calibration and residual analysis.
- City-context features using same-window station aggregates.

## Current Final Candidates

The selected tracked submission files are:

```text
submissions/submission_friend_city_context_depth10_all_train.csv
submissions/submission_friend_city_context_depth10_bag_mean_all_train.csv
submissions/submission_friend_plus_city_context_70_30_blend.csv
```

Recommended order:

1. `submission_friend_city_context_depth10_all_train.csv`
2. `submission_friend_city_context_depth10_bag_mean_all_train.csv`
3. `submission_friend_plus_city_context_70_30_blend.csv`

These candidates are trained from legal `train_raw.csv` windows. They do not use
official hidden targets for fitting.

## Run Locally

Create an environment and install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Place the competition CSV files in:

```text
data/raw/
```

Run the main pipelines:

```bash
python3 -m src.day1_pipeline --data-dir data/raw --output-dir .
python3 -m src.preprocessing_window_baselines --data-dir data/raw --output-dir .
python3 -m src.gradient_boosting_models --data-dir data/raw --output-dir .
python3 -m src.temporal_neural_models --data-dir data/raw --output-dir .
python3 -m src.ensemble_ablation_error_analysis --data-dir data/raw --output-dir .
python3 -m src.severe_pollution_correction --data-dir data/raw --output-dir .
python3 -m src.final_objective_experiments --data-dir data/raw --output-dir .
python3 -m src.rmse_improvement_error_analysis --data-dir data/raw --output-dir .
python3 -m src.friend_feature_final_candidates --data-dir data/raw --output-dir .
```

If using the original 12-station UCI/Kaggle archive, first reconstruct the raw
competition-style files:

```bash
python3 -m src.prepare_data_from_source \
  --source-dir "/path/to/archive-2" \
  --output-dir data/raw \
  --include-test-raw
```

## Notebook Order

```text
01_day1_persistence_baseline.ipynb
02_preprocessing_window_baselines.ipynb
03_gradient_boosting_feature_engineering.ipynb
04_temporal_neural_models.ipynb
05_ensemble_ablation_error_analysis.ipynb
06_severe_pollution_correction.ipynb
07_final_objective_experiments.ipynb
08_rmse_improvement_error_analysis.ipynb
```

## Validation Rules

- Use chronological splits only.
- Fit imputers, encoders, scalers, and target transforms on the training fold.
- Do not use `test_raw.csv` targets for submission training or selection.
- Use the public leaderboard as feedback, not as the only model-selection rule.
- Check severe-pollution and station-wise error before choosing a final file.

## Reports

Start with:

```text
docs/roadmap_report.md
reports/README.md
reports/friend_method_analysis/friend_method_analysis_learning.md
reports/friend_feature_final_candidates/friend_feature_final_candidates_learning.md
```

The other report folders keep the experiment history and figures.

## Team

Group 26:

- E/22/176 - A.L. Jeyanth
- E/22/271 - P. Pathimilan
- E/22/051 - K. Bhaveenthan
- E/22/385 - S. Sulaksan
- E/22/227 - P. Manojh
