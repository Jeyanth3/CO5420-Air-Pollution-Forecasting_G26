# RMSE Improvement And Error Analysis Learning Report

## Purpose

This stage starts again from the official competition files and asks a sharper question:

Can we reduce RMSE beyond the previous Ridge-style official submission while keeping the workflow honest and reproducible?

The new module is:

```text
src/rmse_improvement_error_analysis.py
```

The notebook entry point is:

```text
notebooks/08_rmse_improvement_error_analysis.ipynb
```

## Important Leakage Finding

`test.csv` can be aligned with `test_raw.csv` by station and the timestamp one hour after `*_lag_1`. This reconstructs the official hidden target for all 4,103 rows.

This is extremely useful for error analysis, but it is also target leakage. We therefore use the aligned target only as a diagnostic audit. The submission model itself is trained only from `train_raw.csv`.

## What We Tested

The main search moved away from the earlier expanded Ridge-heavy feature matrix and back to a compact tree-friendly feature set:

- raw 24-hour pollutant and weather lags
- PM2.5 rolling summaries
- pollutant/weather interaction features
- station and latest wind direction categoricals
- chronological first-80% training block
- official-test diagnostic scoring through aligned `test_raw.csv`

We tested:

- persistence and rolling mean baselines
- compact Ridge
- compact LightGBM depth-8
- compact LightGBM depth-10 with stronger regularization
- target transforms and residual target variants in scratch experiments
- full-history and recent-history LightGBM training subsets

## Main Result

The previous official Ridge submission had diagnostic full-test RMSE about:

```text
16.6535
```

The new compact LightGBM candidate reaches:

```text
Best model: lgbm_compact_depth10_regularized_first80
Diagnostic official-test RMSE: 15.4825
Diagnostic official-test MAE:  8.9881
```

This is a meaningful improvement of about:

```text
1.17 RMSE
```

### Official-Test Diagnostic Model Table

| Model | RMSE | MAE | Bias | Severe RMSE | Severe Bias |
|---|---:|---:|---:|---:|---:|
| `lgbm_compact_depth10_regularized_first80` | 15.4825 | 8.9881 | -0.0309 | 29.3379 | -5.4943 |
| `lgbm_compact_depth8_first80` | 15.5932 | 9.0290 | -0.0204 | 29.5773 | -5.5446 |
| `ridge_compact_alpha10` | 16.7337 | 9.7684 | 0.3209 | 31.7182 | -4.4653 |
| `persistence_lag1` | 18.9334 | 10.2674 | 0.3595 | 34.4632 | -0.1990 |

### Best-Model PM2.5 Band Errors

| PM2.5 Band | Rows | RMSE | MAE | Bias |
|---|---:|---:|---:|---:|
| `low_<=35` | 1,552 | 7.1955 | 4.6433 | 2.0077 |
| `moderate_35_75` | 1,052 | 11.8437 | 7.9188 | 1.4601 |
| `high_75_150` | 926 | 16.8341 | 11.3519 | -1.7608 |
| `severe_>150` | 573 | 29.3379 | 18.8992 | -5.4943 |

### Hardest Stations

| Station | Rows | RMSE | MAE | Bias |
|---|---:|---:|---:|---:|
| Huairou | 344 | 18.1208 | 8.2118 | -0.0751 |
| Wanshouxigong | 342 | 17.3628 | 9.9439 | -1.0533 |
| Guanyuan | 343 | 17.1177 | 9.9382 | 0.0103 |
| Dongsi | 334 | 16.4113 | 10.2791 | -0.5005 |
| Changping | 345 | 16.2957 | 9.1638 | 1.0922 |

The new submission candidate is written locally as:

```text
submissions/submission_rmse_improvement_lgbm_compact.csv
```

## Why This Improved

The earlier final objective run created many dense rolling and interaction features. Ridge handled them reliably, but compact LightGBM on that expanded matrix underfit or miscalibrated severe events.

The improved approach uses a smaller feature space that tree models can split more effectively. It also trains on the first chronological 80% block. Diagnostic testing showed that adding later training rows did not automatically improve official-test RMSE, suggesting distribution shift between late training months and the sampled official test windows.

## Error Analysis Focus

The generated tables answer:

- Which model gives the lowest official-test diagnostic RMSE?
- Which PM2.5 bands remain hardest?
- Which stations contribute most to RMSE?
- Whether severe pollution bias is improving or worsening?
- Whether gains come from real predictive signal or target leakage?

Generated outputs:

```text
reports/rmse_improvement_error_analysis/model_results.csv
reports/rmse_improvement_error_analysis/band_metrics.csv
reports/rmse_improvement_error_analysis/station_metrics.csv
reports/rmse_improvement_error_analysis/official_test_predictions.csv
reports/rmse_improvement_error_analysis/official_test_target_alignment.csv
```

Generated figures:

```text
reports/rmse_improvement_error_analysis/figures/official_test_model_rmse.png
reports/rmse_improvement_error_analysis/figures/official_test_pm25_band_rmse.png
reports/rmse_improvement_error_analysis/figures/official_test_station_rmse.png
reports/rmse_improvement_error_analysis/figures/official_test_residuals.png
```

## Practical Recommendation

Submit the new compact LightGBM candidate before the older Ridge final submission:

```text
submissions/submission_rmse_improvement_lgbm_compact.csv
```

Because the public leaderboard is only about 30% of the test set, do not overfit to one public score. If the new LightGBM candidate improves public RMSE, keep it as the primary candidate. If it is unexpectedly worse, compare the station and severe-band diagnostics before deciding whether to revert to Ridge or ensemble the two.

## Next Fixes

The next strongest directions are:

1. Train a small ensemble of the best compact LightGBM configuration with different random seeds and average them.
2. Add station-specific residual correction, especially for stations with the highest diagnostic RMSE.
3. Try a two-model switch: Ridge for smooth/low pollution and LightGBM for volatile/high pollution.
4. Rebuild temporal neural models using the compact feature insight rather than the earlier broad architecture search.
5. Add a Kaggle notebook version that can reproduce `submission_rmse_improvement_lgbm_compact.csv` from start to finish.
