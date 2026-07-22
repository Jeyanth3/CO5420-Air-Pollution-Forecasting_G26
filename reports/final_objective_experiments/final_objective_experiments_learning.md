# Final Objective Experiments Learning Report

## Purpose

This stage uses the official final competition files:

- `train_raw.csv`
- `test_raw.csv`
- `test.csv`
- `sample_submission.csv`

The public leaderboard is only about 30% of the hidden test data, while the final private leaderboard is about 70%. Therefore, model selection must rely mainly on leakage-free local validation and not on repeatedly tuning to the public leaderboard.

## What We Implemented

The final experiment module is:

```text
src/final_objective_experiments.py
```

The notebook entry point is:

```text
notebooks/07_final_objective_experiments.ipynb
```

This module runs Objective 1 to Objective 6 from `future_research_objectives.md`:

1. Rich lag, rolling, trend, slope, and EWMA features.
2. Weather/pollution interaction features.
3. Tuned Ridge, ElasticNet, ExtraTrees, LightGBM, XGBoost, and CatBoost candidates when libraries are available.
4. Weighted model blend using validation predictions.
5. Temporal neural network results are carried as prior evidence because LSTM/GRU/CNN-LSTM/TCN were already implemented and did not beat tabular models.
6. Rolling-origin Ridge validation across four chronological folds.

## Official Submission Outputs

The pipeline writes two candidate submission files:

```text
submissions/submission_final_weighted_ensemble.csv
submissions/submission_final_weighted_ensemble_calibrated.csv
```

The calibrated version applies the severe-pollution correction discovered earlier:

```text
prediction + 0.025 * max(prediction - 150, 0)
```

Both files preserve `sample_submission.csv` row order and contain exactly:

```text
id,PM2.5
```

## How To Run

```bash
python3 -m src.final_objective_experiments --data-dir data/raw --output-dir .
```

## Outputs

Generated tables:

- `experiment_summary.csv`
- `model_results.csv`
- `station_metrics.csv`
- `band_metrics.csv`
- `rolling_origin_results.csv`
- `objective_status.csv`
- `submission_summary.csv`
- `validation_predictions.csv`
- `test_predictions.csv`

Generated figures:

- `objective_model_rmse.png`
- `objective_station_rmse.png`
- `objective_pm25_band_rmse.png`
- `objective_rolling_origin_rmse.png`
- `objective_residuals.png`

## Error-Analysis Questions

This final stage answers:

- Which Objective 1-6 changes improve local RMSE?
- Which stations have the worst residual behavior?
- Whether severe pollution remains the largest error source.
- Whether rolling-origin validation is stable enough to trust the chosen model.
- Whether the calibrated submission is worth trying on the public leaderboard.

## Decision Rule

Use local RMSE as the main selection signal. Use public leaderboard feedback cautiously because it covers only about 30% of the test rows. If the base and calibrated submissions are close on public score, prefer the one supported by local validation and severe-band analysis.

## Actual Results

The final objective run used the official files and produced:

```text
train_raw.csv rows:          315,648
test.csv rows:                 4,103
sample_submission.csv rows:    4,103
validation windows:           61,826
feature columns:                 655
```

### Objective 1-4 Model Results

| Model | RMSE | MAE | Severe RMSE | Severe Bias |
|---|---:|---:|---:|---:|
| `objective_weighted_blend_calibrated` | 19.7454 | 9.8912 | 38.9453 | -5.8537 |
| `ridge_alpha_10` | 19.7585 | 9.9175 | 39.1260 | -8.4263 |
| `objective_weighted_blend` | 19.7585 | 9.9175 | 39.1260 | -8.4263 |
| `lightgbm_tuned_compact_recent` | 40.9896 | 15.7516 | 100.2242 | -55.7437 |

The validation blend selected:

```text
ridge_alpha_10: 1.0
lightgbm_tuned_compact_recent: 0.0
```

So the final objective-stage improvement comes from the small severe-pollution calibration on top of Ridge, not from the compact LightGBM model. The compact LightGBM was trained only on recent rows for reproducibility and performed poorly on high-pollution events, so it should not be trusted as a final component.

Compared with the previous strongest saved ensemble (`saved_weighted_ensemble`, RMSE about 19.5531 in the severe-correction stage), this final Objective 1-6 run does not create a new overall winner. It does, however, confirm that the tiny high-prediction calibration is locally useful:

```text
Ridge RMSE:             19.7585
Calibrated Ridge RMSE:  19.7454
Improvement:             0.0131 RMSE
```

### Objective 6 Rolling-Origin Validation

| Fold | Validation Period | RMSE | MAE |
|---|---|---:|---:|
| Fold 1 | 2014-09 to 2014-12 | 17.0443 | 9.5576 |
| Fold 2 | 2015-01 to 2015-04 | 20.8549 | 11.4712 |
| Fold 3 | 2015-05 to 2015-08 | 13.5308 | 8.3516 |
| Fold 4 | 2015-09 to 2016-02 | 21.4041 | 10.2455 |

Mean rolling-origin RMSE is about 18.2085, but fold variance is meaningful. The model is easiest to fit in the 2015-05 to 2015-08 period and hardest in the 2015-09 to 2016-02 period. This confirms why a single public leaderboard score should not be over-trusted.

### Official Submission Files

Both submission files passed local format validation:

```text
rows: 4,103
columns: id, PM2.5
missing predictions: 0
row order: matches sample_submission.csv
```

Prediction summary:

| File | Mean | Min | Max |
|---|---:|---:|---:|
| `submission_final_weighted_ensemble.csv` | 77.2112 | 0.0000 | 620.1059 |
| `submission_final_weighted_ensemble_calibrated.csv` | 77.5198 | 0.0000 | 631.8585 |

## Error Analysis And Fixes

The main error source is still severe pollution. For the calibrated objective-stage model:

| PM2.5 Band | Rows | RMSE | MAE | Bias |
|---|---:|---:|---:|---:|
| low `<=35` | 28,927 | 10.3921 | 5.9241 | 3.4760 |
| moderate `35-75` | 12,288 | 14.0251 | 8.5092 | 2.5968 |
| high `75-150` | 11,151 | 19.4428 | 11.5348 | 0.3256 |
| severe `>150` | 9,460 | 38.9453 | 21.8794 | -5.8537 |

Reasons for error:

- Severe pollution spikes are underpredicted because one-hour-ahead RMSE is dominated by sharp transitions that cannot be fully inferred from smooth rolling features.
- Low and moderate bands show positive bias, meaning the model slightly overpredicts cleaner periods while underpredicting extreme periods.
- Station-wise error is highest for Wanshouxigong, Aotizhongxin, and Shunyi, suggesting station-specific behavior is not fully captured by global Ridge coefficients.
- Compact LightGBM underperformed because the expanded feature space plus limited recent-row training created poor severe-event calibration.
- Rolling-origin fold spread shows seasonal instability, especially in winter-heavy validation periods.

Relevant fixes:

1. Keep the previous saved weighted ensemble as the primary benchmark; do not replace it with compact LightGBM.
2. Submit both base and calibrated final files if submission limits allow, because calibration improved local severe bias and RMSE slightly.
3. Add station-specific residual correction for Wanshouxigong, Aotizhongxin, and Shunyi.
4. Train a higher-capacity LightGBM on Kaggle/GPU or overnight CPU using the full training window set, not the 20k-row practical cap.
5. Add a high-pollution specialist model and blend it only when predicted PM2.5 is already high.
6. Use rolling-origin mean and variance, not only the public leaderboard, to choose the final private-leaderboard model.

## Final Recommendation

For immediate Kaggle testing:

1. Submit `submission_final_weighted_ensemble.csv`.
2. If another submission is available, submit `submission_final_weighted_ensemble_calibrated.csv`.
3. Prefer the calibrated file only if public leaderboard does not degrade materially, because the public leaderboard represents only about 30% of the test set.
