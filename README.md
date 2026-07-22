# CO5420: Air Pollution Forecasting Using Temporal Neural Networks

> A reproducible forecasting pipeline predicting next-hour PM2.5 concentrations using 24-hour weather, pollution, and calendar data.

## About The Project

This repository contains the machine learning workflow for the **CO5420: Artificial Neural Networks and Deep Learning** course project. The objective is to forecast short-term PM2.5 levels using data from the Beijing Multi-Site Air Quality Dataset. The project explores both traditional machine learning baselines and temporal deep learning models to determine the most effective approach for time-series regression.

The Kaggle task is to predict the PM2.5 concentration one hour after a 24-hour input window from one Beijing monitoring station. Kaggle evaluates submissions with RMSE. Our local workflow also reports MAE for interpretation.

## Key Features

- **Data Preprocessing:** 24-hour window generation, missing-value imputation, wind-direction encoding, and feature scaling.
- **Baseline Models:** Persistence, rolling mean, Ridge Regression, and Random Forest.
- **Advanced Models:** Gradient Boosting with XGBoost/LightGBM/CatBoost and temporal neural networks such as LSTM and GRU.
- **Extended Analysis:** Ensemble weighting, imputation/feature-group ablations, station-wise errors, PM2.5-band errors, feature importance, and AQI classification.

## Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, TensorFlow/Keras or PyTorch for later neural models
- **Environment:** Jupyter Notebook / Kaggle

## Dataset

- **Name:** Beijing Multi-Site Air Quality Dataset
- **Source:** UCI Machine Learning Repository and Kaggle competition files
- **Target:** Next-hour PM2.5 concentration
- **Inputs:** Pollutants, meteorological data, station name, and calendar information

The raw data files are not committed to Git. Put them in `data/raw/` locally, or run the notebook in Kaggle where the files are mounted under `/kaggle/input/...`.

Expected competition files:

```text
train_raw.csv
test.csv
sample_submission.csv
```

Optional for extended work:

```text
test_raw.csv
```

If you downloaded the original 12-station Beijing source dataset from Kaggle or UCI, prepare the competition-style raw files with:

```bash
python3 -m src.prepare_data_from_source \
  --source-dir "/Users/bhaveenthankajanikanth/Downloads/archive-2" \
  --output-dir data/raw \
  --include-test-raw
```

## Day 1 Status

Day 1 creates the project foundation:

- Research roadmap added in `docs/roadmap_report.md`.
- Source-code structure added under `src/`.
- Kaggle-ready notebook added under `notebooks/`.
- Persistence baseline pipeline added.
- Day 1 learning report added in `reports/day1/day1_work_learning.md`.

The local persistence baseline from the reconstructed `train_raw.csv` produced:

```text
RMSE: 21.4431
MAE:  9.9529
```

The preprocessing/window baseline validation currently gives:

```text
Best model: Ridge Regression
RMSE: 20.1097
MAE:  10.0946
```

The current best tabular validation result is:

```text
Best model: Weighted ensemble
RMSE: 19.5651
MAE:  9.6886
```

The current temporal neural-model validation result is:

```text
Best neural model: CNN-LSTM
RMSE: 21.4933
MAE:  11.7208
```

The current analysis stage compares saved model predictions, tests ensemble blends, runs imputation/feature ablations, and creates report figures:

```text
Analysis module: src.ensemble_ablation_error_analysis
Report folder:   reports/ensemble_ablation_error_analysis/
```

Current ensemble/error-analysis finding:

```text
Best full-validation individual model: boost_weighted_ensemble
Full-validation RMSE: 19.5651
Full-validation MAE:  9.6886

Best late-validation ensemble candidate: reference_boost_weighted_ensemble
Late-validation RMSE: 24.7594
Late-validation MAE:  11.9328

Best Ridge ablation: window_interpolation + all_features
RMSE: 19.7279
MAE:  9.9664
```

The severe-pollution correction stage targets the biggest remaining error mode: underprediction when true PM2.5 is above 150.

```text
Analysis module: src.severe_pollution_correction
Report folder:   reports/severe_pollution_correction/
```

Current severe-pollution correction finding:

```text
Best full-validation RMSE model: saved_weighted_ensemble
RMSE: 19.5531
MAE:  9.6707
Severe-band RMSE: 38.8770
Severe-band bias: -9.0084

Best severe-band RMSE model: ridge_weighted_high2_severe8
Overall RMSE: 20.5573
Severe-band RMSE: 37.5473
Severe-band bias: -3.8752

Best late-holdout calibration: additive_threshold_overall_rmse
Late-holdout RMSE: 24.6262 vs 24.7516 baseline
```

The final official-data stage runs Objective 1-6 from the future research plan and creates official Kaggle submission candidates:

```text
Analysis module: src.final_objective_experiments
Report folder:   reports/final_objective_experiments/
Submissions:     submissions/submission_final_weighted_ensemble*.csv
```

The public leaderboard uses about 30% of `test.csv`; final ranking uses the other 70%, so local chronological validation remains the main model-selection signal.

Current final objective finding:

```text
Best objective-stage validation model: objective_weighted_blend_calibrated
Validation RMSE: 19.7454
Validation MAE:  9.8912
Severe-band RMSE: 38.9453

Learned blend weights:
ridge_alpha_10: 1.0
lightgbm_tuned_compact_recent: 0.0

Generated submissions:
submissions/submission_final_weighted_ensemble.csv
submissions/submission_final_weighted_ensemble_calibrated.csv
```

## Quick Start

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the Day 1 persistence baseline locally after placing Kaggle files in `data/raw/`:

```bash
python3 -m src.day1_pipeline --data-dir data/raw --output-dir .
```

Run the preprocessing/window/model baseline experiments:

```bash
python3 -m src.preprocessing_window_baselines --data-dir data/raw --output-dir .
```

Run the gradient boosting and feature-engineering experiments:

```bash
python3 -m src.gradient_boosting_models --data-dir data/raw --output-dir . --max-boost-train-rows 0
```

Run the temporal neural model experiments:

```bash
python3 -m src.temporal_neural_models --data-dir data/raw --output-dir .
```

Run ensemble, ablation, and error analysis after the earlier validation prediction files exist:

```bash
python3 -m src.ensemble_ablation_error_analysis --data-dir data/raw --output-dir .
```

Run severe-pollution correction experiments:

```bash
python3 -m src.severe_pollution_correction --data-dir data/raw --output-dir .
```

Run final Objective 1-6 experiments and create official submission candidates:

```bash
python3 -m src.final_objective_experiments --data-dir data/raw --output-dir .
```

The pipeline writes:

```text
reports/day1/eda_summary.csv
reports/day1/missing_values.csv
reports/day1/persistence_validation_metrics.csv
submissions/submission_persistence.csv
```

## Kaggle Notebook

Use:

```text
notebooks/01_day1_persistence_baseline.ipynb
notebooks/02_preprocessing_window_baselines.ipynb
notebooks/03_gradient_boosting_feature_engineering.ipynb
notebooks/04_temporal_neural_models.ipynb
notebooks/05_ensemble_ablation_error_analysis.ipynb
notebooks/06_severe_pollution_correction.ipynb
notebooks/07_final_objective_experiments.ipynb
```

The Day 1 notebook is designed to run in Kaggle and locate the competition input directory automatically. The later notebooks run local experiment pipelines and display the saved result tables.

## Team: Group_26

- E/22/176 - A.L.JEYANTH
- E/22/271 - P.PATHIMILAN
- E/22/051 - K.BHAVEENTHAN
- E/22/385 - S.SULAKSAN
- E/22/227 - P.MANOJH
