# Ensemble, Ablation, and Error Analysis Learning Report

## Purpose

This stage turns the earlier model experiments into evidence for final model selection. Up to this point we had separate validation results for persistence/classical baselines, gradient boosting, and temporal neural networks. Here we compare them together, test whether blending improves the late validation period, and identify where errors are concentrated.

## What We Implemented

The new pipeline is `src/ensemble_ablation_error_analysis.py`.

It does four jobs:

1. Loads saved validation predictions from:
   - `reports/preprocessing_window_baselines/validation_predictions.csv`
   - `reports/gradient_boosting/validation_predictions.csv`
   - `reports/temporal_neural_models/validation_predictions.csv`
2. Builds ensemble candidates:
   - equal average of the top 2, 3, and 5 models
   - inverse-RMSE weighted top-5 blend
   - non-negative linear top-5 blend
3. Runs Ridge-based ablations:
   - pollution only
   - pollution + weather
   - pollution + weather + calendar
   - all features
   - causal forward-fill, within-window interpolation, and median imputation
4. Generates analysis tables and plots:
   - individual model metrics
   - ensemble metrics
   - station-wise RMSE/MAE
   - PM2.5 concentration-band errors
   - residual summary
   - ablation comparison plots

## Method

The ensemble is evaluated carefully. Instead of learning blend weights and reporting the same period, the validation block is split chronologically. The first half of validation is used to estimate non-negative blend weights, and the second half is used to evaluate the blend. This is stricter than simply optimizing weights on the whole validation set.

The ablations use Ridge regression because it is fast, deterministic, and gives a stable signal about whether a feature family or imputation strategy helps. Boosting is still the main leaderboard model, but Ridge is better for broad repeated ablation tests because it can run quickly across several controlled settings.

Within-window interpolation is used instead of whole-station interpolation. That matters because whole-station interpolation can use values after the prediction window. Within-window interpolation only uses observed lag values already present inside the 24-hour input window, then falls back to training-period medians for any remaining gaps.

## Results

The saved validation predictions contain 61,826 aligned validation windows and 19 model prediction columns from earlier stages.

### Best Individual Models On Full Validation

| Model | RMSE | MAE |
|---|---:|---:|
| `boost_weighted_ensemble` | 19.5651 | 9.6886 |
| `boost_ridge_alpha_10` | 19.7469 | 9.9710 |
| `baseline_ridge_alpha_10` | 20.1097 | 10.0946 |
| `boost_lightgbm_depth10_lr02` | 21.2307 | 9.6029 |
| `boost_catboost_depth8_lr03` | 21.2462 | 9.7701 |
| `neural_cnn_lstm_64` | 21.4933 | 11.7208 |

The strongest overall model remains the weighted boosting ensemble from the gradient-boosting stage.

### Ensemble Result On Late Validation Holdout

The validation period was split chronologically. Blend weights were learned or selected using the first half, then evaluated on the later half.

| Candidate | RMSE | MAE | Decision |
|---|---:|---:|---|
| `reference_boost_weighted_ensemble` | 24.7594 | 11.9328 | Keep as reference winner |
| `ensemble_mean_top_2` | 24.7781 | 12.0453 | Slightly worse |
| `ensemble_mean_top_3` | 24.8622 | 12.0009 | Worse |
| `ensemble_inverse_rmse_top_5` | 25.1008 | 11.8006 | Lower MAE, worse RMSE |
| `ensemble_positive_linear_top_5` | 27.2372 | 12.1285 | Overweighted weaker late-period models |

The ensemble experiment is valuable because it tells us not to force a larger blend. The best late-validation RMSE is still the previous boosting weighted ensemble. For Kaggle RMSE, the final candidate should remain boosting-first unless a later model beats it locally.

### Ablation Results

| Ablation | RMSE | MAE | Meaning |
|---|---:|---:|---|
| `ridge_window_interpolation_all_features` | 19.7279 | 9.9664 | Best Ridge ablation |
| `ridge_causal_ffill_all_features` | 19.7469 | 9.9710 | Very close to interpolation |
| `ridge_causal_ffill_pollution_weather_calendar` | 19.8017 | 10.0257 | Calendar helps only modestly without categorical station/wind |
| `ridge_causal_ffill_pollution_weather` | 19.8021 | 10.0263 | Weather improves over pollution-only |
| `ridge_causal_ffill_pollution_only` | 20.1608 | 9.8957 | Strong, but not enough by itself |
| `ridge_median_all_features` | 20.8440 | 10.3055 | Median-only imputation is clearly weaker |

The ablation result says the task is strongly autoregressive, but weather, calendar, station, wind, and careful imputation still add useful RMSE improvements.

### Error Analysis Findings

For the best model, `boost_weighted_ensemble`, station-wise RMSE ranges from about 17.43 at Huairou to 22.16 at Wanshouxigong. The hardest station in this validation split is Wanshouxigong, followed by Aotizhongxin and Shunyi.

By PM2.5 band, the biggest RMSE problem is severe pollution:

| Band | Rows | RMSE | MAE | Bias |
|---|---:|---:|---:|---:|
| low `<=35` | 28,927 | 10.0438 | 5.5188 | 3.1316 |
| moderate `35-75` | 12,288 | 13.8240 | 8.4202 | 2.0115 |
| high `75-150` | 11,151 | 19.1799 | 11.4702 | 0.0624 |
| severe `>150` | 9,460 | 38.8770 | 21.9866 | -9.0084 |

The negative severe-band bias means the model tends to underpredict very high PM2.5 events. That is important because Kaggle uses RMSE, and RMSE punishes these large misses heavily.

## How To Run

```bash
python3 -m src.ensemble_ablation_error_analysis --data-dir data/raw --output-dir .
```

The matching notebook is:

```text
notebooks/05_ensemble_ablation_error_analysis.ipynb
```

## Outputs

Generated tables are written to:

```text
reports/ensemble_ablation_error_analysis/
```

Important files:

- `analysis_summary.csv`
- `individual_model_metrics.csv`
- `ensemble_results.csv`
- `station_error_metrics.csv`
- `pm25_band_error_metrics.csv`
- `residual_summary.csv`
- `ablation_results.csv`

Generated figures are written to:

```text
reports/ensemble_ablation_error_analysis/figures/
```

## Interpretation Guide

The best individual model tells us which previous modeling family is strongest overall. The late-validation ensemble comparison tells us whether blending improves robustness on unseen later data. Station-wise error identifies monitoring stations where the model struggles. PM2.5-band error shows whether the model is especially weak during high-pollution events, which matters because RMSE penalizes large misses heavily.

The ablation table should be read as a controlled diagnostic:

- If pollution-only is close to all-features, the one-hour task is dominated by autoregressive pollutant memory.
- If weather improves RMSE, meteorological variables are useful for short-term transport/dispersion.
- If calendar features improve RMSE, seasonal and daily cycles matter.
- If within-window interpolation beats causal forward-fill, missing-value smoothness inside the observed lag window is helpful.
- If median imputation performs worse, station-local temporal continuity is important.

## Next Fixes

Use this analysis to choose the final Kaggle model:

1. Keep the strongest boosting model as the main candidate.
2. Add only ensemble members that improve the late-validation holdout.
3. Investigate the worst stations and severe-pollution band before final submission.
4. If the ensemble does not beat the best boosting model, submit the best boosting model rather than forcing an ensemble.
5. Try severe-pollution fixes next: log/Box-Cox target transform, high-pollution sample weighting, and a residual correction model for events above 150.
