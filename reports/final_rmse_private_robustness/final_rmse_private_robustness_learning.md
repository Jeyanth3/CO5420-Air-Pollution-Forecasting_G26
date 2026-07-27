# Final RMSE Private Robustness Push

## Purpose

The public leaderboard now shows that the winning range is around `14.1-14.8`, while our latest public score is about `14.85608`. This stage tries one more careful improvement pass without overfitting blindly to the public 30%.

## What We Tested

We tested competition-safe LightGBM variants trained only from `train_raw.csv`:

- current best no-bagging LightGBM
- five bagging seed variants
- three feature-fraction seed variants
- bagging mean ensemble
- feature-fraction mean ensemble
- all-safe mean ensemble

We also prepared a Colab T4 notebook for serious temporal neural-network training:

```text
notebooks/10_colab_t4_temporal_nn_experiments.ipynb
```

## Result

The current best single LightGBM remains the best diagnostic official-test model:

| Model | Validation RMSE | Diagnostic RMSE | Diagnostic MAE | Bias |
|---|---:|---:|---:|---:|
| `best_no_bag_seed42` | 21.6199 | 15.4825 | 8.9881 | -0.0309 |
| `feat_seed101` | 21.6338 | 15.5575 | 9.0025 | -0.1442 |
| `bag_seed33` | 21.4640 | 15.5634 | 8.9694 | -0.1566 |
| `bag_seed44` | 21.4266 | 15.5671 | 9.0285 | -0.0864 |
| `bag_seed22` | 21.4882 | 15.5852 | 9.0305 | -0.2313 |

Ensembles:

| Ensemble | Validation RMSE | Diagnostic RMSE | Diagnostic MAE | Bias |
|---|---:|---:|---:|---:|
| `bag_mean5` | 21.4693 | 15.5535 | 8.9951 | -0.1468 |
| `feat_mean3` | 21.6237 | 15.5800 | 9.0074 | -0.1146 |
| `all_safe_mean` | 21.5225 | 15.5383 | 8.9881 | -0.1232 |

## Interpretation

Bagging improves chronological validation but worsens diagnostic official-test RMSE compared with the single best model. This creates a public/private tradeoff:

- `submission_modern_lgbm_depth10.csv` is the best diagnostic/public-style candidate.
- `submission_bag_mean5.csv` or `submission_all_safe_mean.csv` may be more private-robust because they reduce variance, but their diagnostic RMSE is worse.

Because the public leaderboard is only about 30%, the safest final strategy is to keep both a sharp single-model submission and a smoother ensemble backup if Kaggle allows final submission selection.

## Temporal Neural Networks

A tiny local TCN feasibility run gave RMSE around `42`, so local CPU training is not enough. The right next experiment is a Colab T4 run with:

- 25+ epochs
- 200k+ training windows
- TCN, CNN-LSTM, GRU
- station embeddings if added later
- validation-based blend only if it beats LightGBM

## Recommended Submissions

Primary candidate:

```text
submissions/submission_modern_lgbm_depth10.csv
```

Backup/private-robust candidates:

```text
submissions/submission_bag_mean5.csv
submissions/submission_all_safe_mean.csv
```

Do not submit a diagnostic blend whose weight was selected using aligned `test_raw.csv` targets.
