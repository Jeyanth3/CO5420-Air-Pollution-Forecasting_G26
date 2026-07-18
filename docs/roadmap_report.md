# CO5420 Air Pollution Forecasting - Research Roadmap

## 1. Competition Objective

The Kaggle task is a one-step-ahead regression problem: use the previous 24 hourly observations from one Beijing monitoring station to predict the next hour's PM2.5 concentration. The official ranking metric is RMSE, and the project report should also include MAE.

The source data is the Beijing Multi-Site Air Quality Dataset: 420,768 hourly observations from 12 stations between 2013-03-01 and 2017-02-28, with pollutant, meteorological, calendar, and station fields. Missing values are retained and must be handled without leakage.

## 2. Project Principles

1. Use chronological validation only. Random splits are not acceptable for model selection because they leak future behavior.
2. Fit imputers, encoders, scalers, and target transforms only on training folds.
3. Always beat simple baselines before trusting complex neural models.
4. Keep a reproducible Kaggle notebook that runs end to end: data loading, preprocessing, window construction, model training, prediction, and submission creation.
5. Select the final model using local validation first, then use the public leaderboard only as a sanity check.

## 3. Repository Structure

```text
CO5420-Air-Pollution-Forecasting_G26/
  README.md
  requirements.txt
  notebooks/
    00_eda.ipynb
    01_baselines.ipynb
    02_tabular_models.ipynb
    03_temporal_nn.ipynb
    04_ensemble_submission.ipynb
  src/
    config.py
    data_io.py
    preprocessing.py
    windows.py
    features.py
    metrics.py
    models/
      baselines.py
      tabular.py
      neural.py
      ensemble.py
  experiments/
    results.csv
    model_cards/
  reports/
    figures/
    final_report.md
  submissions/
    submission_baseline.csv
    submission_final.csv
```

## 4. Data Pipeline

### 4.1 Raw Data Checks

- Verify row counts, station names, date coverage, duplicate timestamps, missing values, and target distribution.
- Build `datetime` from `year`, `month`, `day`, and `hour`.
- Sort by `station, datetime`.
- Check each station has continuous hourly records.

### 4.2 Imputation Experiments

Methods to compare:

- Median imputation fitted on training only.
- Station-wise forward fill, then backward fill, then training median fallback.
- Linear interpolation within each station, then station/training median fallback.
- Missingness indicators for pollutant and weather columns.

Research justification: recent air-quality imputation work shows that imputation choice can materially affect downstream forecasting, so it should be treated as an experiment rather than a cleanup detail.

### 4.3 Feature Encoding

- Encode `wd` as categorical one-hot for tabular models.
- Also test wind vector encoding:
  - `wind_x = WSPM * cos(theta)`
  - `wind_y = WSPM * sin(theta)`
- Encode hour/month cyclically:
  - `sin_hour`, `cos_hour`
  - `sin_month`, `cos_month`
- Encode station as one-hot or CatBoost categorical.
- Add weekend/weekday and season if useful.

## 5. Window Construction

For every station independently:

```text
X_t = [x_{t-23}, x_{t-22}, ..., x_t]
y_t = PM2.5_{t+1}
```

Important validation checks:

- The latest input PM2.5 must be at lag 1.
- The target must be exactly one hour after lag 1.
- No window may cross station boundaries.
- Windows containing missing raw values should be imputed by a fitted training pipeline, not dropped blindly.

## 6. Validation Strategy

Use at least two local validation modes:

1. Holdout split: train on earlier windows, validate on the final chronological block of `train_raw.csv`.
2. Rolling-origin validation: train on earlier time blocks and validate on later blocks to confirm the model is stable across seasons.

Track:

- RMSE: Kaggle metric.
- MAE: project-analysis metric.
- Station-wise RMSE: detects models that only work for easier stations.
- High-pollution RMSE: errors above 75 or 150 micrograms/m3 matter for public-health usefulness.

## 7. Model Roadmap

### Stage 1 - Baselines

Purpose: establish a trustworthy floor.

- Persistence: predict latest PM2.5.
- Rolling means: last 3h, 6h, 12h, 24h.
- Station-hour climatology: median PM2.5 by station, month, hour.
- Hybrid baseline: weighted average of persistence and short rolling mean.

Expected result: persistence and short rolling means should be hard to beat for one-hour-ahead forecasting.

### Stage 2 - Linear and Classical Models

- Ridge and ElasticNet on flattened lag windows plus engineered summaries.
- Random Forest and ExtraTrees as nonlinear baselines.

Purpose: detect whether engineered lags, weather summaries, and station/time encodings add real signal.

### Stage 3 - Gradient Boosting

Primary Kaggle candidates:

- LightGBM: efficient for high-dimensional flattened lag features.
- XGBoost: robust sparse-aware boosting baseline.
- CatBoost: strong handling of categorical station/wind features and ordered boosting ideas.

Feature set:

- All flattened lags from `test.csv`-compatible columns.
- Latest values for each pollutant/weather variable.
- PM2.5 rolling mean/std/min/max over 3h, 6h, 12h, 24h.
- PM2.5 trend: lag 1 minus lag 24.
- Pollutant ratios and interactions, for example PM2.5/PM10.
- Weather summaries: wind speed mean/max, rain sum, pressure trend, dew-point spread.
- Calendar and station features.

### Stage 4 - Temporal Neural Networks

Required project-family models:

- LSTM: `(24, F) -> LSTM -> dense -> PM2.5`.
- GRU: faster recurrent alternative to LSTM.
- CNN-LSTM: local temporal convolution over 24 hours followed by recurrent layer.
- TCN: causal dilated convolutions for short-window temporal structure.

Training controls:

- Standardize continuous features using training fold only.
- Early stopping on validation RMSE.
- Small models first to avoid overfitting.
- Fixed random seeds.
- Save out-of-fold predictions for ensembling.

### Stage 5 - Research-Inspired Extensions

Use these only after baselines/boosting/neural foundations are correct:

- Attention over 24 hourly steps to learn which lag hours matter most.
- CNN-BiLSTM-style feature extractor inspired by air-quality hybrid models.
- Patch/Transformer model if compute allows, although 24 hours is short and may not justify heavy architecture.
- Multi-station graph model only if `test_raw.csv` and station geography/relationships are useful for the extended task; Kaggle `test.csv` rows are flattened single-window rows, so graph models may be more valuable for report experiments than leaderboard submissions.

## 8. Final Ensemble Strategy

Build model diversity deliberately:

- Best persistence/rolling baseline.
- Best LightGBM/XGBoost/CatBoost model.
- Best LSTM/GRU or CNN-LSTM model.
- Optional Ridge model if it gives different errors.

Use validation predictions to learn ensemble weights:

- Simple weighted average.
- Non-negative least squares.
- Ridge stacking on out-of-fold predictions.

Avoid overfitting:

- Do not tune weights directly on Kaggle public leaderboard.
- Confirm ensemble improves both RMSE and MAE locally.
- Check station-wise degradation before final submission.

## 9. Extended Analysis Plan

1. Window-size study: 6h, 12h, 24h, 48h, 72h using train/test raw data.
2. Feature ablation:
   - pollution only
   - pollution + weather
   - pollution + weather + calendar
   - all features + station
3. Imputation ablation:
   - median
   - station ffill/bfill
   - interpolation
   - missingness indicators
4. Feature importance:
   - gain/split importance for boosting
   - permutation importance
   - SHAP if compute allows
5. AQI-style classification:
   - convert PM2.5 regression output into warning categories
   - evaluate accuracy, macro-F1, and confusion matrix

## 10. Execution Timeline

### Day 1

- Connect local repo to GitHub.
- Add README, requirements, source structure, and notebook skeleton.
- Download Kaggle files into the Kaggle notebook environment.
- Run EDA and validate row/window alignment.
- Submit persistence baseline.

### Days 2-3

- Implement preprocessing and window generator.
- Run persistence, rolling mean, Ridge, Random Forest, ExtraTrees.
- Record local RMSE/MAE.

### Days 4-5

- Implement feature engineering.
- Train LightGBM, XGBoost, CatBoost.
- Tune using chronological validation.
- Make first serious Kaggle submission.

### Days 6-7

- Implement LSTM and GRU.
- Add CNN-LSTM or TCN if LSTM/GRU are stable.
- Compare neural models against boosting.

### Days 8-9

- Build ensemble.
- Run ablations for imputation, feature groups, and station-wise errors.
- Generate plots and tables.

### Final Days

- Freeze final model and seed.
- Re-run notebook from start to finish.
- Create final `submission.csv`.
- Write final report: methodology, research support, experiments, results, limitations, AI-use statement.

## 11. Research Basis

- UCI Beijing Multi-Site Air Quality Dataset: confirms 12 stations, hourly pollutant and meteorological data, 2013-03-01 to 2017-02-28, 420,768 instances, and retained missing values.
- Zhang et al. (2017): supports the Beijing PM2.5/environmental motivation and the dataset's public-health relevance.
- Du et al. (2021): supports hybrid CNN + BiLSTM architectures for multivariate air-quality forecasting.
- Li et al. (2022): supports attention/CNN/BiLSTM combinations for multi-site Beijing PM2.5 prediction.
- Hua et al. (2024): supports treating imputation as a controlled experiment.
- Breiman (2001), Chen and Guestrin (2016), Ke et al. (2017), and Prokhorenkova et al. (2018): justify Random Forest, XGBoost, LightGBM, and CatBoost as strong tabular baselines.
- Hochreiter and Schmidhuber (1997), Cho et al. (2014), Lim et al. (2021), and Nie et al. (2023): justify LSTM, GRU, attention/Transformer, and PatchTST-style time-series methods.
- Recent GNN/Transformer PM2.5 work suggests graph-based spatiotemporal models are promising when station relationships and wind-driven transport can be modeled, but this should be secondary for the Kaggle flattened 24-hour test format.

## 12. Source Links

- Kaggle competition page: https://www.kaggle.com/competitions/co-5420-air-pollution-forecasting-using-temporal-n-ns/overview
- UCI dataset: https://archive-beta.ics.uci.edu/dataset/501/beijing%2Bmulti%2Bsite%2Bair%2Bquality%2Bdata
- Du et al., Deep Air Quality Forecasting Using Hybrid Deep Learning Framework: https://doi.org/10.1109/tkde.2019.2954510
- Li et al., CNN-BiLSTM with CBAM for Beijing PM2.5: https://www.mdpi.com/2073-4433/13/10/1719
- Hua et al., impact of imputation on air-quality prediction: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0306303
- XGBoost paper: https://doi.org/10.1145/2939672.2939785
- Random Forest paper: https://doi.org/10.1023/A:1010933404324
- LightGBM paper: https://papers.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html
- CatBoost paper: https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html
- PatchTST paper: https://doi.org/10.48550/arXiv.2211.14730
- Temporal Fusion Transformer paper: https://www.sciencedirect.com/science/article/pii/S0169207021000637
- Recent STGCN air-pollution forecasting paper: https://doi.org/10.1016/j.techfore.2024.123684
- Recent dynamic geographical GNN PM2.5 paper: https://doi.org/10.1016/j.envsoft.2025.106351
