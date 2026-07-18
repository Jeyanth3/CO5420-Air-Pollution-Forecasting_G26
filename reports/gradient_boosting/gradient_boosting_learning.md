# Gradient Boosting And Feature Engineering Learning

## What We Built

This stage adds stronger tabular modelling on top of the preprocessing/window baseline work.

We implemented:

- LightGBM.
- XGBoost.
- CatBoost.
- Ridge reference model with the expanded feature set.
- A simple validation-weighted ensemble.
- Extra interaction features for pollution, weather, ventilation, rainfall, and short-term trend.
- Feature-importance export.
- Station-wise error export.

The main script is:

```text
src/gradient_boosting_models.py
```

The feature helpers are in:

```text
src/features.py
```

## Why Gradient Boosting Is The Right Next Step

The data is a flattened 24-hour lag table with pollutants, weather, station, wind direction, and calendar features. Gradient boosting is well suited for this structure because it can learn nonlinear threshold-style interactions, for example:

```text
high latest PM2.5 + low wind speed + no rain -> pollution likely persists
```

Boosting models are also strong Kaggle models because they usually handle tabular feature interactions better than plain Random Forests.

## Feature Engineering Added

The previous window features already contained:

- 24 lag values for each numeric variable.
- PM2.5 rolling means.
- PM2.5 trend, standard deviation, minimum, and maximum.
- Weather means and trends.
- Station and latest wind direction.
- Cyclic hour/month encodings.

This stage added interaction features:

- `pm25_pm10_ratio_lag1`
- `pm25_low_wind_lag1`
- `ventilation_proxy_lag1`
- `pm25_rain_interaction_lag1`
- `has_rain_lag1`
- `temp_dewp_spread_lag1`
- `pressure_change_24h`
- `pm25_short_vs_daily`
- `pm25_short_daily_ratio`
- `pm25_acceleration_3h`

These features help the model represent short-term pollution buildup, dispersion, rainfall washout, and humidity/temperature conditions.

## Experiment Setup

The experiment used the same chronological validation principle:

```text
training period: earlier timestamps
validation period: later timestamps
```

The split summary was:

```text
cutoff: 2015-07-25 18:00:00
raw rows: 315,648
valid 24-hour windows: 308,988
training windows: 247,162
validation windows: 61,826
numeric feature columns: 318
categorical feature columns: 2
```

All available training windows were used for the final boosting run.

## Local Validation Results

| Rank | Model | RMSE | MAE | Interpretation |
|---:|---|---:|---:|---|
| 1 | Weighted ensemble | 19.5651 | 9.6886 | Best RMSE. Combines Ridge with LightGBM. |
| 2 | Ridge Regression | 19.7469 | 9.9710 | Best single RMSE model. Strong linear signal from lag features. |
| 3 | LightGBM depth10 | 21.2307 | 9.6029 | Best single MAE model. Good median-sized errors but worse large-error RMSE. |
| 4 | CatBoost depth8 | 21.2462 | 9.7701 | Similar to LightGBM, uses native categorical handling. |
| 5 | LightGBM depth8 | 21.6204 | 9.6933 | Near persistence on RMSE, better MAE. |
| 6 | Persistence | 21.6663 | 10.1598 | Strong baseline using latest PM2.5. |
| 7 | XGBoost depth6 | 23.2893 | 10.0287 | Weaker in this initial configuration. |
| 8 | Rolling mean 3h | 30.6516 | 14.8907 | Too much smoothing for one-hour prediction. |

The weighted ensemble is the current best local model:

```text
RMSE: 19.5651
MAE:  9.6886
```

Compared with the previous persistence baseline:

```text
Persistence RMSE: 21.6663
Ensemble RMSE:    19.5651
Improvement:       2.1012 RMSE
```

## Interpretation

Ridge performs very well because one-hour-ahead PM2.5 has strong linear persistence. The latest PM2.5 value and nearby pollutant lags carry most of the predictive signal.

LightGBM and CatBoost are useful even though their RMSE is worse than Ridge. Their MAE is lower than Ridge, which means they reduce typical absolute error but still make some larger errors that hurt RMSE. This difference is exactly why the ensemble helps: Ridge controls large-error behavior, while LightGBM improves ordinary-hour behavior.

XGBoost was weaker in this first configuration. It should not be discarded permanently, but it needs additional tuning before it can be trusted as a final candidate.

## Feature Importance

The strongest CatBoost features were:

```text
PM2.5_lag_1
pm25_latest
dewp_trend_24h
pm10_latest
PM10_lag_1
pm25_low_wind_lag1
PM2.5_lag_2
pm25_acceleration_3h
temp_dewp_spread_lag1
pm25_mean_3h
```

This confirms the expected physical story:

- Latest PM2.5 dominates one-hour prediction.
- PM10 is useful because coarse and fine particulates often move together.
- Dew point, temperature-dewpoint spread, and wind-speed interaction help describe pollutant dispersion and atmospheric conditions.
- Short-term acceleration captures whether pollution is rising or falling.

## Output Files

Running the script creates:

```text
reports/gradient_boosting/experiment_summary.csv
reports/gradient_boosting/model_results.csv
reports/gradient_boosting/station_model_results.csv
reports/gradient_boosting/validation_predictions.csv
reports/gradient_boosting/feature_importance.csv
```

The CSV files are generated experiment artifacts and are ignored by Git. The learning report is committed.

## Command

```bash
python3 -m src.gradient_boosting_models --data-dir data/raw --output-dir . --max-boost-train-rows 0
```

