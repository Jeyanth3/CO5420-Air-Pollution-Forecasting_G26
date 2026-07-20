# Temporal Neural Models Learning

## Purpose

This stage implements sequence models for the 24-hour PM2.5 forecasting task and compares them with the current tabular boosting result.

The implemented neural models are:

- LSTM
- GRU
- CNN-LSTM
- TCN

The default experiment runs LSTM, GRU, and CNN-LSTM. TCN is implemented as an optional model because it is slower on CPU for this dataset.

## Research Motivation

LSTM is the standard recurrent architecture for learning temporal dependencies and controlling vanishing gradients. GRU is a lighter recurrent alternative. CNN-LSTM combines local temporal feature extraction from 1D convolutions with recurrent sequence modeling. TCN uses causal dilated convolutions and is a strong alternative to recurrent models for sequence tasks.

For Beijing PM2.5 forecasting, prior work reports that LSTM and CNN-LSTM are strong one-hour forecasting models when past PM2.5, correlated pollutants, and meteorological variables are used as inputs.

## Data Design

The input tensor is:

```text
samples x 24 hours x features
```

The target is:

```text
PM2.5 one hour after the 24-hour window
```

The feature set is deliberately compact for CPU training:

- PM2.5
- PM10
- SO2
- NO2
- CO
- O3
- TEMP
- PRES
- DEWP
- RAIN
- WSPM
- cyclic hour
- cyclic month
- station code
- wind-direction sine/cosine

Preprocessing is leakage-free:

```text
fit imputation fallback values on training period only
fit StandardScaler on training period only
create station-isolated 24-hour windows
drop windows with missing raw target PM2.5
```

## Model Comparison Target

The neural models are compared against:

```text
Ridge reference RMSE:             19.7469
Boosting weighted ensemble RMSE: 19.5651
```

This is important because beating persistence is not enough anymore. A neural model should either beat these tabular models or contribute useful diversity to a later ensemble.

## Actual Local Results

The CPU-feasible neural experiment used:

```text
cutoff: 2015-07-25 18:00:00
window size: 24 hours
training windows available: 247,162
training windows used by neural models: 80,000 latest chronological windows
validation windows: 61,826
feature count: 18
device: CPU
epochs: 6
batch size: 1024
target standardization: enabled
```

Validation results:

| Rank | Model | RMSE | MAE | Interpretation |
|---:|---|---:|---:|---|
| 1 | Boosting weighted ensemble reference | 19.5651 | 9.6886 | Current best local tabular model. |
| 2 | Ridge reference | 19.7469 | 9.9710 | Strong linear lag-feature reference. |
| 3 | CNN-LSTM | 21.4933 | 11.7208 | Best neural model in this run; convolution helps before recurrent modeling. |
| 4 | GRU | 21.6718 | 11.9504 | Similar to persistence-level RMSE, faster than CNN-LSTM. |
| 5 | LSTM | 22.0004 | 12.2264 | Stable but weaker in this CPU-constrained run. |

The neural models are stable and train successfully, but they do not yet beat the tabular models. This is not a failure. It tells us that this one-hour task is strongly driven by short-term persistence and engineered lag features. CNN-LSTM is the most promising neural architecture so far.

## Interpretation And Next Fixes

The likely reasons neural models trail the tabular ensemble are:

- The sequence feature set is intentionally compact.
- Training used only the latest 80,000 training windows for CPU feasibility.
- The target is one-hour ahead, where simple lag structure is very strong.
- Neural models may need more epochs, larger training windows, and target/log transformations.

Next improvements:

- Train on all 247,162 training windows in Kaggle or overnight.
- Add a target transform experiment: raw target vs standardized target vs `log1p(PM2.5)`.
- Add richer neural sequence features, including rolling statistics as extra channels.
- Tune hidden size, dropout, learning rate, and weight decay.
- Try the optional TCN run when compute time is available.
- Use neural validation predictions as ensemble candidates if they add diversity.

## Command

```bash
python3 -m src.temporal_neural_models --data-dir data/raw --output-dir .
```

Optional TCN run:

```bash
python3 -m src.temporal_neural_models --data-dir data/raw --output-dir . --models lstm,gru,cnn_lstm,tcn
```

## Sources

- Hochreiter and Schmidhuber, "Long Short-Term Memory", Neural Computation, 1997. https://doi.org/10.1162/neco.1997.9.8.1735
- Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation", EMNLP, 2014. https://doi.org/10.3115/v1/D14-1179
- Bai, Kolter, and Koltun, "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling", arXiv, 2018. https://arxiv.org/abs/1803.01271
- Yang et al., "PM2.5 concentrations forecasting in Beijing through deep learning with different inputs, model structures and forecast time", Atmospheric Pollution Research, 2021. https://doi.org/10.1016/j.apr.2021.101168
