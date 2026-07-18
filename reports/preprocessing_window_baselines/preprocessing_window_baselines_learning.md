# Preprocessing Window Baselines Learning

## What We Built

This work moved the project from a simple one-step baseline into a real supervised-learning forecasting setup.

We implemented:

- Leakage-aware preprocessing.
- A 24-hour window generator.
- Rolling mean baselines.
- Ridge Regression.
- Random Forest.
- ExtraTrees.
- Local chronological RMSE/MAE reporting.
- Station-wise error reporting.

The main script is:

```text
src/preprocessing_window_baselines.py
```

The main window-building utilities are in:

```text
src/windows.py
```

## Why This Step Matters

The competition test file contains flattened 24-hour windows. That means our training data must be converted into the same learning format:

```text
previous 24 hours -> next-hour PM2.5
```

If this alignment is wrong, every later model will be wrong even if the code runs. So this stage focuses on making window construction explicit, testable, and reproducible.

## Preprocessing Method

We use chronological validation. The first 80 percent of timestamps are training, and the final 20 percent are validation.

Preprocessing follows a leakage-aware rule:

```text
Fit fallback values on the training period only.
Apply causal forward-fill within each station.
Use training medians/modes only when forward-fill cannot fill a value.
```

This matters because validation data must simulate the future. We should not use future validation values to clean earlier validation rows.

## Window Generator

For each station separately, the code builds:

```text
X_t = [x_{t-23}, x_{t-22}, ..., x_t]
y_t = PM2.5_{t+1}
```

The generator checks that all rows in the window and target are exactly one hour apart. It also drops rows where the real raw target PM2.5 is missing. Feature values are imputed, but validation labels are not imputed.

This gives us fair evaluation: the model is judged only against observed PM2.5 values.

## Features

Each 24-hour window is flattened into tabular features:

```text
PM2.5_lag_24 ... PM2.5_lag_1
PM10_lag_24 ... PM10_lag_1
...
WSPM_lag_24 ... WSPM_lag_1
```

We also add summary features:

- Latest PM2.5.
- PM2.5 rolling means over 3, 6, 12, and 24 hours.
- PM2.5 24-hour standard deviation, minimum, maximum, and trend.
- Weather latest values, means, and trends.
- Hour, month, weekend flag, and cyclic time encodings.
- Station and latest wind direction as categorical variables.

## Models

### Persistence

Predicts:

```text
next PM2.5 = PM2.5_lag_1
```

This remains the minimum model to beat.

### Rolling Mean

Predicts the average of the last 3, 6, 12, or 24 PM2.5 values. These models test whether smoothing recent pollution improves over the latest value.

### Ridge Regression

Ridge uses the flattened lag features and engineered summaries. Numeric features are standardized, and station/wind direction are one-hot encoded.

Ridge is important because it is fast, stable, and reveals whether the lag features contain useful linear signal.

### Random Forest And ExtraTrees

These are nonlinear tree ensembles. They can capture interactions such as:

```text
high PM2.5 + low wind speed + no rain -> PM2.5 likely remains high
```

Tree models were trained on a recent chronological subset of the training windows to keep runtime practical while still using realistic recent patterns.

## What To Learn From The Results

The RMSE/MAE table tells us which model is strongest locally.

The station-wise table tells us whether a model is generally strong or only good for some stations.

This matters because a model with a good average RMSE may still fail badly at one station. Later, when we train boosting and neural networks, we should compare not only total RMSE but also station-wise behavior.

## Actual Local Results

The preprocessing/window baseline experiment used this chronological split:

```text
cutoff: 2015-07-25 18:00:00
raw rows: 315,648
valid 24-hour windows: 308,988
training windows: 247,162
validation windows: 61,826
numeric feature columns: 308
categorical feature columns: 2
```

The local validation results were:

| Rank | Model | RMSE | MAE | Notes |
|---:|---|---:|---:|---|
| 1 | Ridge Regression | 20.1097 | 10.0946 | Best baseline model in this stage; uses scaled flattened windows and engineered features. |
| 2 | Persistence | 21.6663 | 10.1598 | Very strong one-hour baseline using `PM2.5_lag_1`. |
| 3 | Random Forest | 25.0510 | 11.0457 | Nonlinear model, trained on recent 160,000 training windows. |
| 4 | ExtraTrees | 27.9341 | 12.9475 | More randomized tree ensemble, weaker than Ridge here. |
| 5 | Rolling mean 3h | 30.6516 | 14.8907 | Smoothing already hurts one-hour forecasting. |
| 6 | Rolling mean 6h | 40.1195 | 20.3134 | Too much smoothing. |
| 7 | Rolling mean 12h | 51.3000 | 27.3377 | Much weaker. |
| 8 | Rolling mean 24h | 61.9499 | 35.3263 | Daily average loses short-term signal. |

The most important lesson is that the latest PM2.5 value is extremely informative. Ridge improves on persistence because it can combine the latest PM2.5 with short-term trend, other pollutants, weather, station, wind direction, and calendar effects.

The tree models did not beat Ridge in this first setup. That does not mean tree-based models are bad for this task. It means Random Forest and ExtraTrees are not the strongest tree strategy for this high-dimensional lag table. The next stronger direction is gradient boosting with LightGBM, XGBoost, and CatBoost, because boosting usually performs better than bagged trees on tabular Kaggle-style data.

## Output Files

Running the script creates:

```text
reports/preprocessing_window_baselines/window_split_summary.csv
reports/preprocessing_window_baselines/model_results.csv
reports/preprocessing_window_baselines/station_model_results.csv
reports/preprocessing_window_baselines/validation_predictions.csv
```

The CSV files are generated experiment artifacts and are ignored by Git. The explanation file is committed.

## Command

```bash
python3 -m src.preprocessing_window_baselines --data-dir data/raw --output-dir .
```
