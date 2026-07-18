# Day 1 Work Learning

## What We Did

Day 1 was about building the foundation before trying advanced neural networks. In a forecasting competition, the biggest early danger is not choosing a weak model. The biggest danger is building the data pipeline incorrectly and getting a validation score that looks good but is false.

We connected the local project to the GitHub repository and created the basic project structure:

- `README.md` explains how to run the project.
- `requirements.txt` records the Python libraries needed.
- `src/` contains reusable code.
- `notebooks/` contains a Kaggle-ready notebook.
- `reports/day1/` stores the Day 1 explanation and generated diagnostics.
- `submissions/` will store generated Kaggle submission files.

We also added a Day 1 pipeline that can:

1. Load the Kaggle files.
2. Summarize the training data.
3. Count missing values.
4. Validate one-hour-ahead target alignment.
5. Impute missing values using a simple station-wise time-series method.
6. Evaluate a persistence baseline.
7. Create a valid Kaggle submission using the most recent PM2.5 value in each test window.

## Why We Started With A Baseline

The simplest strong baseline for this competition is persistence:

```text
next hour PM2.5 = latest observed PM2.5
```

This is simple, but it is not weak. PM2.5 usually changes smoothly from one hour to the next, so the latest observed value can be a very competitive first prediction. Every later model must beat this baseline locally before we trust it.

This baseline gives us three benefits:

1. It checks that the submission format is correct.
2. It gives a minimum RMSE score that future models must improve.
3. It helps detect mistakes in window alignment, because a broken target alignment often makes persistence look strangely bad or strangely good.

## What Method We Are Following

We are following a leakage-free chronological forecasting workflow.

The training rows are sorted by station and time. For each station, the model uses current and previous values to predict the next hour. We do not randomly shuffle time-series rows for validation, because random splitting can put future patterns into training and make the local score unrealistic.

The workflow is:

```text
load raw data
create datetime
sort by station and time
check missing values
impute within each station
validate one-hour target alignment
evaluate persistence baseline on a chronological holdout
create Kaggle submission from test.csv
```

The first imputation method is simple:

1. Forward-fill values within each station.
2. Backward-fill remaining values within each station.
3. Fill any remaining numeric missing values with training medians.
4. Fill wind direction using nearby station values and a mode fallback.

This method is not assumed to be the final best method. It is the Day 1 baseline imputation method. Later we will compare it with median imputation, linear interpolation, and missingness indicators.

## How We Validate Window Alignment

For each station, the target must be exactly one hour after the input time. The code checks this by creating:

```text
target_next_pm25 = PM2.5 shifted by -1 within each station
next_datetime = datetime shifted by -1 within each station
hours_to_target = next_datetime - datetime
```

Only rows where `hours_to_target == 1` are valid one-hour-ahead rows. This prevents accidental station-boundary leakage or gaps in the time series.

For the Kaggle `test.csv`, every row is already a flattened 24-hour window. The persistence submission uses:

```text
PM2.5_lag_1
```

This is the most recent PM2.5 value in the 24-hour window.

## Local Data Source Update

We found the original Beijing Multi-Site Air Quality source files in:

```text
/Users/bhaveenthankajanikanth/Downloads/archive-2
```

This folder contains the 12 PRSA station CSV files from 2013-03-01 to 2017-02-28. That is the same public source dataset used by the competition.

We added a reproducible preparation script:

```text
src/prepare_data_from_source.py
```

This script concatenates the 12 station files, sorts by station and time, and creates:

```text
data/raw/train_raw.csv
data/raw/test_raw.csv
```

The `train_raw.csv` file covers 2013-03-01 00:00 through 2016-02-29 23:00, which matches the competition training period.

## What We Could Not Do Locally Yet

The original source data is now available locally. However, the competition-specific `test.csv` and `sample_submission.csv` were not found locally, and the `kaggle` command-line tool was not installed on this machine. Because of that, we could not actually create or submit the official Kaggle prediction file from the terminal today.

Instead, we made the workflow ready for Kaggle:

- If the notebook runs inside Kaggle, it automatically searches `/kaggle/input/co-5420-air-pollution-forecasting-using-temporal-n-ns/`.
- If the files are downloaded locally later, the same pipeline can run from `data/raw/`.
- The output submission will be written as `submissions/submission_persistence.csv`.

## Why This Matters For The Final Project

This Day 1 work creates the controlled experiment setup. Once this is correct, every advanced model can be compared fairly:

- Ridge regression
- Random Forest
- ExtraTrees
- XGBoost
- LightGBM
- CatBoost
- LSTM
- GRU
- CNN-LSTM
- TCN
- Ensemble models

The main lesson is that a strong Kaggle result starts with a correct validation pipeline. If the baseline, imputation, alignment, and submission format are correct, then model improvements become meaningful instead of accidental.
