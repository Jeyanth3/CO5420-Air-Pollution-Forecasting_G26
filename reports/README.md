# Reports Index

This folder keeps the experiment record. The README gives the short path through
the work; the individual reports contain the details.

## Recommended Reading Order

1. `docs/roadmap_report.md` - original research plan and timeline.
2. `day1/day1_work_learning.md` - repository setup, EDA, and persistence.
3. `preprocessing_window_baselines/preprocessing_window_baselines_learning.md` -
   imputation, window building, and classical baselines.
4. `gradient_boosting/gradient_boosting_learning.md` - feature engineering and
   first strong boosting models.
5. `temporal_neural_models/temporal_neural_models_learning.md` - LSTM, GRU, and
   CNN-LSTM experiments.
6. `ensemble_ablation_error_analysis/ensemble_ablation_error_analysis_learning.md`
   - model blending, ablations, and error plots.
7. `severe_pollution_correction/severe_pollution_correction_learning.md` -
   high-PM2.5 bias analysis.
8. `rmse_improvement_error_analysis/rmse_improvement_error_analysis_learning.md`
   - official-file audit and compact LightGBM improvement.
9. `friend_method_analysis/friend_method_analysis_learning.md` - analysis of the
   merged team feature update.
10. `friend_feature_final_candidates/friend_feature_final_candidates_learning.md`
    - final legal candidate generation.

## Main Takeaways

- Persistence is a strong baseline for one-hour-ahead PM2.5, but it leaves large
  errors during sudden changes.
- Ridge improves the baseline, but it cannot capture nonlinear pollutant and
  weather interactions well enough.
- LightGBM became the strongest family after adding rolling, ratio, volatility,
  wind-vector, and station/context features.
- Early LSTM/GRU/CNN-LSTM models were useful for the project requirement, but
  they did not beat the best tabular boosting models.
- Severe pollution remains the hardest region. Most high-PM2.5 errors come from
  underprediction.
- `test_raw.csv` is valuable for analysis, but it can expose official targets.
  It is not used for legal submission training or selection.

## Final Candidate Folder

Tracked Kaggle-format candidates live in `submissions/`. Generated submissions
are ignored by default, so only selected final files should be force-added when
needed.
