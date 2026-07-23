# Modern Temporal RMSE Improvements Learning Report

## Purpose

This branch performs another complete improvement pass after the compact LightGBM breakthrough. The goal is to test modern methods that could plausibly reduce RMSE further:

- stronger boosted trees
- station-specific modelling
- boosted-model blending
- temporal neural-network feasibility
- deeper error analysis by pollution band and station

The main module is:

```text
src/modern_temporal_rmse_improvements.py
```

The notebook is:

```text
notebooks/09_modern_temporal_rmse_improvements.ipynb
```

## Research Motivation

Recent PM2.5 and air-quality forecasting work repeatedly supports three ideas:

1. Use temporal deep learning when enough training time and tuning budget are available.
2. Use exogenous weather, pollutant, calendar, and site variables together.
3. Keep strong gradient-boosting baselines because tabular lag features often beat under-tuned neural models in small/medium forecasting competitions.

Relevant sources:

- Bai, Kolter, and Koltun showed that generic Temporal Convolutional Networks can outperform recurrent baselines across sequence tasks, making TCN a natural candidate for 24-hour pollutant windows: https://arxiv.org/abs/1803.01271
- Informer introduced efficient long-sequence Transformer forecasting, but our task has only 24 input hours, so transformer complexity is less urgent than robust lag engineering: https://arxiv.org/abs/2012.07436
- Autoformer and FEDformer support decomposition/frequency ideas for long-horizon forecasting; for this one-step task, their main lesson is to capture trend/seasonality rather than blindly deepen networks: https://arxiv.org/abs/2106.13008 and https://arxiv.org/abs/2201.12740
- Beijing PM2.5 studies consistently find pollutant history, weather, wind, and site information important, supporting our lag/interactions/station-feature approach.
- Kaggle-style forecasting reviews and M5 findings support strong tree models, rolling features, and ensembles as high-performing practical baselines.

## Experiments Run

Competition-safe models were trained only from `train_raw.csv`. The aligned `test_raw.csv` target was used only for diagnostic scoring and error analysis.

Tested:

- `lgbm_depth10_regularized_first80`
- `lgbm_station_specific`
- `xgboost_depth6_compact`
- `catboost_depth6_compact`
- diagnostic LightGBM/CatBoost blend
- tiny TCN feasibility run

## Results

The best competition-safe model remains:

```text
lgbm_depth10_regularized_first80
```

Its diagnostic RMSE remains approximately:

```text
RMSE: 15.4825
MAE:  8.9881
```

The diagnostic LightGBM/CatBoost blend gives a tiny improvement, but it is not competition-safe because the blend weight is selected using the aligned official target. It is useful evidence that CatBoost has slightly different errors, but the gain is too small to justify leakage risk.

Station-specific LightGBM was worse than the global model. This suggests each station-specific model has higher variance and less training data, while the global model benefits from shared pollutant-weather patterns.

The tiny TCN feasibility check was much worse:

```text
tiny_tcn_1epoch_10k RMSE: 42.2750
```

This does not mean TCN is bad in principle. It means undertrained neural models are not competitive under our current local compute/time budget. A serious neural attempt should use GPU, more epochs, learning-rate scheduling, and probably station embeddings.

### Model Comparison

| Model | Family | RMSE | MAE | Bias | Severe RMSE | Severe Bias |
|---|---|---:|---:|---:|---:|---:|
| `diagnostic_lgbm_catboost_blend` | diagnostic only | 15.4823 | 8.9858 | -0.0308 | 29.3334 | -5.5034 |
| `lgbm_depth10_regularized_first80` | safe boosting | 15.4825 | 8.9881 | -0.0309 | 29.3379 | -5.4943 |
| `xgboost_depth6_compact` | safe boosting | 16.1507 | 9.1590 | -0.1547 | 31.2510 | -6.2599 |
| `catboost_depth6_compact` | safe boosting | 16.1696 | 9.2541 | -0.0280 | 30.8560 | -5.9493 |
| `lgbm_station_specific` | safe station models | 16.8430 | 9.4240 | -0.0770 | 33.2902 | -6.8761 |
| `tiny_tcn_1epoch_10k` | neural feasibility | 42.2750 | 33.7547 | 13.2779 | N/A | N/A |

The diagnostic blend is deliberately not recommended for submission because its `0.02` CatBoost weight was selected using the aligned official target.

### Best Safe Model: PM2.5 Band Errors

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

## Error Analysis

The largest remaining difficulty is still high and severe PM2.5. The best LightGBM strongly improves severe-band error compared with earlier Ridge, but still underpredicts some extreme spikes.

Likely error causes:

- sudden PM2.5 jumps are not fully determined by the previous 24 hours
- station-level microclimate and wind transport are only partly represented
- `test.csv` rows are sampled from the final year, so chronological validation from the first three years is imperfect
- public leaderboard is only about 30%, so optimizing too hard to public score can hurt private score

## Submission

The safe submission generated by this branch is:

```text
submissions/submission_modern_lgbm_depth10.csv
```

It should be very close to the previous best compact LightGBM submission, because the deeper experiments did not find a stronger competition-safe model.

## Next Best Actions

1. Submit `submission_modern_lgbm_depth10.csv` if not already submitted.
2. If public score is near 15.4-15.6, keep it as the main candidate.
3. Try a true GPU neural run only after the strongest LightGBM is secured.
4. Build a non-leaky blend using chronological validation weights, not aligned official targets.
5. Add station residual correction only if it improves chronological validation, because station-specific direct models were worse.
