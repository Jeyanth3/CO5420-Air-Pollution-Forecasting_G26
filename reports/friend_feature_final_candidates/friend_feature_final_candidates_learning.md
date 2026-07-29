# Friend-Feature Final Candidates

## Purpose

These candidates try to improve beyond the reproduced friend submission without
using hidden official-test labels. They reuse the improved feature set and train
final LightGBM models from legal `train_raw.csv` windows only.

## Primary Candidates

- `submission_friend_city_context_depth10_all_train.csv`
- `submission_friend_city_context_depth10_bag_mean_all_train.csv`
- `submission_friend_plus_city_context_70_30_blend.csv`

The first one is the direct attempt to improve the friend feature model with city context by using
all legal training windows. The bagged and 70/30 blend files are safer
robustness probes.
