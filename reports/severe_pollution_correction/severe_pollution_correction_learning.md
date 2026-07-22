# Severe-Pollution Correction Learning Report

## Purpose

The previous error analysis showed that the strongest model underpredicts severe pollution events where true PM2.5 is above 150. This stage tests targeted fixes for that weakness while protecting the main Kaggle metric, RMSE.

## What We Implemented

The new pipeline is:

```text
src/severe_pollution_correction.py
```

The notebook entry point is:

```text
notebooks/06_severe_pollution_correction.ipynb
```

The experiment families are:

- Raw Ridge reference model.
- Sample-weighted Ridge models that give more weight to PM2.5 >= 75 and PM2.5 >= 150.
- Log-target Ridge using `log1p(PM2.5)`.
- Box-Cox target Ridge using a fitted training-target transformer.
- Weighted Box-Cox Ridge.
- LightGBM reference and severe-weighted LightGBM variants.
- Log/Box-Cox LightGBM variants.
- Calibration corrections on the saved best `weighted_ensemble` predictions.
- Residual correction model using saved validation predictions as calibration features.

## Method

The retraining experiments keep the same chronological train/validation split used by the earlier modeling stages. Sample weights are fitted only on the training labels.

The calibration experiments are stricter. They split the existing validation period chronologically into:

- earlier validation calibration block
- later validation holdout block

Correction parameters are learned on the earlier block and evaluated on the later block. This avoids selecting a correction using the same rows used for reporting.

## How To Run

```bash
python3 -m src.severe_pollution_correction --data-dir data/raw --output-dir .
```

## Outputs

Generated outputs are written to:

```text
reports/severe_pollution_correction/
```

Important generated tables:

- `experiment_summary.csv`
- `model_results.csv`
- `band_metrics.csv`
- `validation_predictions.csv`

Generated figures:

- `overall_rmse_severe_fix_candidates.png`
- `severe_band_rmse_candidates.png`
- `overall_vs_severe_rmse_tradeoff.png`
- `selected_candidate_residuals.png`
- `severe_band_bias.png`

## Reading The Results

This stage should be judged with two questions:

1. Did the candidate improve overall RMSE?
2. Did the candidate reduce severe-band RMSE or severe-band negative bias without damaging the overall score?

Because Kaggle ranks by RMSE, an aggressive severe-event correction is useful only if the overall RMSE does not become worse.

## Results

### Full-Validation RMSE

The best full-validation model is still the saved weighted ensemble from the previous boosting stage.

| Model | Overall RMSE | MAE | Severe RMSE | Severe Bias |
|---|---:|---:|---:|---:|
| `saved_weighted_ensemble` | 19.5531 | 9.6707 | 38.8770 | -9.0084 |
| `ridge_reference_raw` | 19.7223 | 9.9355 | 38.7950 | -8.3959 |
| `ridge_weighted_high1_severe4` | 20.1406 | 10.8165 | 37.7038 | -4.4168 |
| `ridge_weighted_high2_severe8` | 20.5573 | 11.5055 | 37.5473 | -3.8752 |
| `lightgbm_weighted_high075_severe3` | 21.4328 | 10.0917 | 43.7426 | -9.2695 |

Interpretation: sample-weighted Ridge does reduce the severe-band underprediction, but it makes the overall RMSE worse. Since Kaggle uses overall RMSE, the weighted severe model should not replace the weighted ensemble as the main submission model.

### Severe-Band Tradeoff

The strongest severe-band model is:

```text
ridge_weighted_high2_severe8
Overall RMSE: 20.5573
Severe-band RMSE: 37.5473
Severe-band bias: -3.8752
```

Compared with the saved weighted ensemble:

```text
saved_weighted_ensemble
Overall RMSE: 19.5531
Severe-band RMSE: 38.8770
Severe-band bias: -9.0084
```

This means the aggressive weighted Ridge model improves severe-event behavior, but the cost to normal/moderate predictions is too high for the final RMSE objective.

### Target Transform Results

The target transforms did not help here:

- `log1p` Ridge became unstable after inverse transformation.
- Box-Cox Ridge and weighted Box-Cox Ridge also produced much worse RMSE.
- Log/Box-Cox LightGBM did not improve severe-band RMSE.

The lesson is that target transformation is not automatically helpful for one-hour PM2.5 forecasting. In this validation setup, it reduced calibration quality for high concentrations.

### Calibration Results

The best late-validation calibration is:

```text
additive_threshold_overall_rmse
Late-holdout RMSE: 24.6262
Reference late-holdout RMSE: 24.7516
Threshold: 150.0
Offset: 0.0
Slope: 0.025
```

This correction adds a very small upward adjustment only when the base prediction is above 150:

```text
corrected = prediction + 0.025 * max(prediction - 150, 0)
```

This is the only correction that improved late-holdout RMSE. It also reduced severe-band negative bias from about -9.85 to -7.01 on the late holdout.

## Final Decision From This Stage

Do not replace the best model with a heavily severe-weighted model. The best full-validation RMSE candidate remains:

```text
saved_weighted_ensemble
```

For the final Kaggle submission stage, keep two candidate outputs:

1. Base weighted ensemble.
2. Base weighted ensemble plus the tiny high-prediction calibration:

```text
prediction + 0.025 * max(prediction - 150, 0)
```

Then decide using the final reproducible notebook and, if available, cautious Kaggle public leaderboard feedback.
