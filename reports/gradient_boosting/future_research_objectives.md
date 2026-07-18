# Future Research Objectives And Experiment Plan

## Purpose

This document records the next research goals for the PM2.5 forecasting project. The goal is not to randomly try models. The goal is to build a controlled forecasting system where each new method is tested against the same leakage-free validation protocol and judged by RMSE, MAE, station-wise behavior, and edge-case performance.

Current best local result:

```text
Persistence RMSE:       21.6663
Ridge RMSE:             19.7469
Weighted ensemble RMSE: 19.5651
```

The current direction is correct: the project has moved from baselines to engineered tabular models and an ensemble. The next work should deepen the feature set, improve validation reliability, add temporal neural models, and build explicit error analysis.

## Objective 1: Strengthen Lag And Rolling Feature Engineering

### Motivation

Forecasting competition research shows that gradient boosted trees, rolling statistics, exogenous variables, and ensembles are often among the strongest methods for tabular forecasting problems. The M5 Accuracy Competition reported that top methods heavily used machine-learning models, especially LightGBM, with explanatory variables and model combinations. Kaggle forecasting competition reviews also identify gradient boosted decision trees, neural networks, validation strategy, and external/exogenous information as major success factors.

### Planned Experiments

Add richer lag-window features:

```text
PM2.5 lags: 1-24h
PM10, SO2, NO2, CO, O3 lags: 1-24h
TEMP, PRES, DEWP, RAIN, WSPM lags: 1-24h
```

Add rolling statistics:

```text
mean: 3h, 6h, 12h, 24h
std: 3h, 6h, 12h, 24h
min/max: 3h, 6h, 12h, 24h
median: 6h, 12h, 24h
quantiles: q25, q75 over 24h
```

Add trend and dynamics:

```text
delta_1h = lag_1 - lag_2
delta_3h = lag_1 - lag_4
delta_6h = lag_1 - lag_7
slope_6h
slope_12h
slope_24h
acceleration_3h = lag_1 - 2 * lag_2 + lag_3
```

Add exponential smoothing:

```text
PM2.5 EWMA alpha = 0.2
PM2.5 EWMA alpha = 0.4
PM2.5 EWMA alpha = 0.6
pollutant EWMA features
weather EWMA features
```

### Success Criteria

The feature set is accepted only if it improves local chronological RMSE or improves edge-case performance without damaging overall RMSE.

## Objective 2: Improve Weather And Pollution Interaction Features

### Motivation

PM2.5 forecasting research on Beijing data reports that historical PM2.5, other pollutant concentrations, meteorological variables, wind speed, and selected correlated variables are strong predictors. Air-pollution behavior is not purely autoregressive. Weather affects accumulation, dispersion, and washout.

### Planned Features

Pollution ratios:

```text
PM2.5 / PM10
PM2.5 / CO
NO2 / O3
SO2 / NO2
```

Dispersion and ventilation:

```text
PM2.5 / (WSPM + 1)
PM10 / (WSPM + 1)
PM2.5 * WSPM
low_wind_flag
```

Rain effects:

```text
rain_last_1h
rain_sum_3h
rain_sum_6h
rain_sum_24h
PM2.5 * rain_flag
```

Humidity proxy:

```text
TEMP - DEWP
DEWP trend
TEMP trend
pressure trend
```

Wind direction:

```text
one-hot wind direction
wind_x = WSPM * cos(direction)
wind_y = WSPM * sin(direction)
station x wind-direction interaction
```

### Success Criteria

These features should help especially during sharp PM2.5 changes, low-wind periods, rainy periods, and high-pollution episodes.

## Objective 3: Build Stronger Boosting Models

### Motivation

The first gradient-boosting run showed that Ridge still has the best single-model RMSE, while LightGBM and CatBoost improve MAE. This means the boosting models capture typical-hour behavior but still produce larger errors that hurt RMSE. The next step is targeted tuning.

### Planned Models

LightGBM:

```text
num_leaves: 31, 63, 127, 255
max_depth: 6, 8, 10, -1
learning_rate: 0.01, 0.02, 0.03, 0.05
min_child_samples: 20, 50, 80, 120
feature_fraction: 0.7, 0.8, 0.9
bagging_fraction: 0.7, 0.8, 0.9
lambda_l1/lambda_l2 regularization
```

XGBoost:

```text
max_depth: 4, 6, 8
min_child_weight: 3, 6, 10
eta: 0.02, 0.03, 0.05
subsample: 0.75, 0.85, 0.95
colsample_bytree: 0.75, 0.85, 0.95
reg_alpha/reg_lambda
```

CatBoost:

```text
depth: 6, 8, 10
learning_rate: 0.02, 0.03, 0.05
l2_leaf_reg: 3, 6, 10
iterations with early stopping
native categorical station and wind features
```

### Success Criteria

A tuned boosting model should either:

```text
beat Ridge on RMSE
or improve MAE/station-wise robustness enough to improve an ensemble
```

## Objective 4: Blend Multiple Model Families

### Motivation

Forecasting competition results repeatedly show that ensembles and cross-learning models tend to outperform a single local model. In this project, Ridge and boosting models already show different error behavior: Ridge has stronger RMSE, while LightGBM/CatBoost have competitive MAE.

### Planned Ensemble Candidates

Candidate models:

```text
Persistence baseline
Ridge
ElasticNet
LightGBM
XGBoost
CatBoost
LSTM
GRU
CNN-LSTM
TCN
```

Blend methods:

```text
simple average
manual weighted average
non-negative least squares
Ridge stacking on validation predictions
station-specific blend weights
pollution-level-specific blend weights
```

### Validation Rule

Ensemble weights must be learned only from chronological validation predictions. They must not be tuned directly on Kaggle public leaderboard feedback.

### Success Criteria

The final ensemble should improve:

```text
overall RMSE
overall MAE
high-pollution RMSE
station-wise stability
```

## Objective 5: Add Temporal Neural Networks

### Motivation

Research on Beijing PM2.5 forecasting reports that LSTM and CNN-LSTM are strong for one-hour forecasting when appropriate inputs are selected. Neural models may capture short sequential patterns that flattened Ridge/boosting models miss.

### Planned Neural Models

Sequence input shape:

```text
(samples, window_size, features)
```

Models:

```text
LSTM
GRU
1D CNN
CNN-LSTM
Temporal Convolutional Network
attention over lag hours
```

Window sizes:

```text
6h
12h
24h
48h
72h
```

Feature groups:

```text
pollution only
pollution + weather
pollution + weather + calendar
all features + station
```

Training controls:

```text
StandardScaler fitted on training only
early stopping
dropout
weight decay
fixed random seed
chronological validation
```

### Success Criteria

Neural models do not need to beat all tabular models alone. They are useful if:

```text
they beat persistence
they improve ensemble diversity
they reduce errors during nonlinear transition periods
```

## Objective 6: Use Rolling-Origin Validation

### Motivation

A single last-20-percent validation split may be sensitive to one seasonal period. Research on forecasting validation emphasizes that time-series validation should respect chronology. Rolling-origin validation gives a more reliable picture of generalization.

### Planned Folds

Example:

```text
Fold 1: train 2013-03 to 2014-08, validate 2014-09 to 2014-12
Fold 2: train 2013-03 to 2014-12, validate 2015-01 to 2015-04
Fold 3: train 2013-03 to 2015-04, validate 2015-05 to 2015-08
Fold 4: train 2013-03 to 2015-08, validate 2015-09 to 2016-02
```

Metrics:

```text
fold RMSE
fold MAE
mean RMSE
standard deviation of RMSE
station-wise RMSE per fold
```

### Success Criteria

Prefer models with stable fold performance over models that only win one validation period.

## Objective 7: Build Error Analysis

### Motivation

Improving the average RMSE is not enough. We need to know when the model fails. PM2.5 forecasting errors are likely larger during sudden pollution spikes, rapid weather changes, rain events, and station-specific local behavior.

### Error Slices

Analyze errors by:

```text
station
hour of day
month/season
wind direction
wind speed bucket
rain/no-rain
PM2.5 target bucket
high-pollution episodes
rapid-rise episodes
rapid-drop episodes
missing-value-heavy windows
```

Target buckets:

```text
0-35
35-75
75-150
150-250
250+
```

Error metrics:

```text
RMSE
MAE
bias = mean(prediction - target)
underprediction rate
overprediction rate
95th percentile absolute error
```

### Fix Strategy

If errors are high during spikes:

```text
add acceleration/trend features
add high-pollution specialized model
try target transformation
increase weight of spike windows during training
```

If errors are station-specific:

```text
add station-specific models
add station-specific ensemble weights
add station x hour and station x wind features
```

If errors are rain/wind-specific:

```text
add weather interaction features
add rain-window features
add wind-vector features
```

## Objective 8: Prepare Kaggle Submission Pipeline

### Motivation

The official Kaggle submission requires `test.csv`, not `test_raw.csv`. The file contains flattened 24-hour windows with IDs. The output must contain exactly:

```text
id,PM2.5
```

### Planned Submission Models

When official `test.csv` and `sample_submission.csv` are available, create submissions for:

```text
persistence
ridge
best LightGBM
best CatBoost
best weighted ensemble
best neural ensemble
final blended model
```

### Submission Discipline

Do not submit every experiment. Use local validation first. Submit only meaningful candidates:

```text
baseline submission
first strong tabular submission
boosting/ensemble submission
neural/ensemble submission
final selected submission
```

## Final Research Direction

The project should continue in this order:

```text
1. Improve lag/rolling/EWMA/interactions.
2. Tune LightGBM/XGBoost/CatBoost.
3. Build rolling-origin validation.
4. Add error-analysis dashboards/tables.
5. Train LSTM/GRU/CNN-LSTM/TCN.
6. Blend Ridge + boosting + neural models.
7. Generate final Kaggle submission once official test.csv is available.
```

The guiding principle is:

```text
Every new method must beat a baseline, improve an ensemble, or explain an error mode.
```

## Sources

- Makridakis, Spiliotis, and Assimakopoulos, "M5 accuracy competition: Results, findings, and conclusions", International Journal of Forecasting, 2022. https://doi.org/10.1016/j.ijforecast.2021.11.013
- Bojer and Meldgaard, "Kaggle forecasting competitions: An overlooked learning opportunity", International Journal of Forecasting, 2021. https://doi.org/10.1016/j.ijforecast.2020.07.007
- Januschowski et al., "Forecasting with trees", International Journal of Forecasting, 2022. https://doi.org/10.1016/j.ijforecast.2021.10.004
- Yang et al., "PM2.5 concentrations forecasting in Beijing through deep learning with different inputs, model structures and forecast time", Atmospheric Pollution Research, 2021. https://doi.org/10.1016/j.apr.2021.101168

