# Friend-Feature Final Candidates

## Purpose

These candidates try to improve beyond the reproduced friend submission without
using hidden official-test labels. They reuse the improved feature set and train
final LightGBM models from legal `train_raw.csv` windows only.

The latest pass adds leak-free same-time city context inspired by the attached
research plan's multi-station-context section.

City-context features include:

- same-timestamp city PM2.5 mean, max and standard deviation;
- other-station PM2.5 mean;
- station-minus-city and station-minus-other pollution contrast;
- city PM2.5 short-term change;
- same-time aggregate PM10, NO2, CO, O3, WSPM, TEMP and PRES signals.

These features use only information available at the end of the 24-hour input
window. They do not use `test_raw.csv` target labels.

## Validation Evidence

Adding city context produced a large chronological-validation improvement:

| Candidate | Validation RMSE | Validation MAE |
|---|---:|---:|
| `depth10_frac0.75` with city context | 18.3091 | 8.6849 |
| `depth8_frac0.75` with city context | 18.3912 | 8.7409 |
| previous non-city `depth10_frac0.75` | 20.2175 | 9.2745 |

This suggests that multi-station context is useful for the competition windows.
It helps the model recognise city-wide pollution movement instead of treating
each station as isolated.

## Primary Candidates

- `submission_friend_city_context_depth10_all_train.csv`
- `submission_friend_city_context_depth10_bag_mean_all_train.csv`
- `submission_friend_plus_city_context_70_30_blend.csv`

## Submission Order

Use this order for Kaggle probing:

1. `submissions/submission_friend_city_context_depth10_all_train.csv`
   - Strongest current better-than-friend attempt.
   - Uses city-context features and 308,988 legal training windows.

2. `submissions/submission_friend_plus_city_context_70_30_blend.csv`
   - Safer candidate if the full city-context model moves too far.
   - Keeps 70% of the reproduced friend-feature model and adds 30% of the city-context all-train model.

3. `submissions/submission_friend_city_context_depth10_bag_mean_all_train.csv`
   - Variance-reduced all-train LightGBM bagging candidate.

All files preserve the Kaggle format:

```text
id,PM2.5
4103 rows
0 missing predictions
```

## What This Improves

The friend branch improved feature quality. This pass improves the final
training strategy and feature context by using all legal training windows and
adding same-time multi-station aggregate features. It still avoids fitting or
selecting from `test_raw.csv` targets.
