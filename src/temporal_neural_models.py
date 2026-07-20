"""Temporal neural models for one-hour-ahead PM2.5 forecasting."""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import DATETIME_COL, DEFAULT_LOCAL_DATA_DIR, RAW_NUMERIC_FEATURES, STATION_COL, TARGET
from src.data_io import read_competition_files
from src.metrics import mae, rmse
from src.preprocessing import CausalPreprocessor, sort_raw
from src.windows import chronological_cutoff


WIND_DIRECTION_DEGREES = {
    "N": 0.0,
    "NNE": 22.5,
    "NE": 45.0,
    "ENE": 67.5,
    "E": 90.0,
    "ESE": 112.5,
    "SE": 135.0,
    "SSE": 157.5,
    "S": 180.0,
    "SSW": 202.5,
    "SW": 225.0,
    "WSW": 247.5,
    "W": 270.0,
    "WNW": 292.5,
    "NW": 315.0,
    "NNW": 337.5,
}


@dataclass
class SequenceData:
    """Container for temporal model arrays and metadata."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_valid: np.ndarray
    y_valid: np.ndarray
    valid_meta: pd.DataFrame
    feature_names: list[str]
    cutoff: pd.Timestamp


def set_seed(seed: int) -> None:
    """Make model training as reproducible as practical."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


def add_sequence_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create compact continuous sequence features for neural models."""
    out = sort_raw(df)
    numeric_cols = [col for col in RAW_NUMERIC_FEATURES if col in out.columns]

    out["sin_hour"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["cos_hour"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["sin_month"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["cos_month"] = np.cos(2 * np.pi * out["month"] / 12.0)

    station_codes = {station: idx for idx, station in enumerate(sorted(out[STATION_COL].unique()))}
    out["station_code"] = out[STATION_COL].map(station_codes).astype(float)

    if "wd" in out.columns:
        degrees = out["wd"].map(WIND_DIRECTION_DEGREES).fillna(0.0).astype(float)
        radians = np.deg2rad(degrees)
        out["wd_sin"] = np.sin(radians)
        out["wd_cos"] = np.cos(radians)
    else:
        out["wd_sin"] = 0.0
        out["wd_cos"] = 0.0

    feature_names = numeric_cols + [
        "sin_hour",
        "cos_hour",
        "sin_month",
        "cos_month",
        "station_code",
        "wd_sin",
        "wd_cos",
    ]
    return out, feature_names


def build_sequence_data(
    train_raw: pd.DataFrame,
    train_fraction: float,
    window_size: int,
) -> SequenceData:
    """Build scaled sequence arrays using a leakage-free chronological split."""
    cutoff = chronological_cutoff(train_raw, train_fraction=train_fraction)
    raw_sorted = sort_raw(train_raw)
    train_period = raw_sorted[raw_sorted[DATETIME_COL] <= cutoff].copy()

    imputer = CausalPreprocessor.fit(train_period)
    imputed = imputer.transform(train_raw)
    features_df, feature_names = add_sequence_features(imputed)
    raw_sorted = sort_raw(train_raw)

    train_feature_rows = features_df[features_df[DATETIME_COL] <= cutoff]
    scaler = StandardScaler()
    scaler.fit(train_feature_rows[feature_names])
    features_df[feature_names] = scaler.transform(features_df[feature_names]).astype(np.float32)

    X_parts = []
    y_parts = []
    meta_parts = []
    split_parts = []

    for station, station_features in features_df.groupby(STATION_COL, sort=False):
        station_raw = raw_sorted.loc[station_features.index].reset_index(drop=True)
        station_features = station_features.reset_index(drop=True)
        n_rows = len(station_features)
        if n_rows <= window_size:
            continue

        values = station_features[feature_names].to_numpy(dtype=np.float32)
        windows = np.lib.stride_tricks.sliding_window_view(values, window_shape=window_size, axis=0)
        windows = np.moveaxis(windows, -1, 1)[: n_rows - window_size]

        target_positions = np.arange(window_size, n_rows)
        end_positions = target_positions - 1

        datetimes = station_features[DATETIME_COL]
        diffs = datetimes.diff().dt.total_seconds().div(3600).eq(1.0).fillna(False)
        continuity = diffs.astype(int).rolling(window=window_size, min_periods=window_size).sum()
        valid_continuity = continuity.iloc[target_positions].to_numpy() == window_size

        targets = station_raw[TARGET].iloc[target_positions].to_numpy(dtype=np.float32)
        target_observed = ~pd.isna(targets)
        valid_mask = valid_continuity & target_observed
        if not valid_mask.any():
            continue

        end_times = station_features[DATETIME_COL].iloc[end_positions].reset_index(drop=True)
        target_times = station_raw[DATETIME_COL].iloc[target_positions].reset_index(drop=True)
        is_train = end_times <= cutoff

        X_parts.append(np.ascontiguousarray(windows[valid_mask], dtype=np.float32))
        y_parts.append(targets[valid_mask].astype(np.float32))
        split_parts.append(is_train.to_numpy()[valid_mask])
        meta_parts.append(
            pd.DataFrame(
                {
                    "station": station,
                    "window_end_datetime": end_times.to_numpy()[valid_mask],
                    "target_datetime": target_times.to_numpy()[valid_mask],
                }
            )
        )

    if not X_parts:
        raise ValueError("No valid sequence windows were created.")

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    is_train_all = np.concatenate(split_parts, axis=0)
    meta = pd.concat(meta_parts, ignore_index=True)

    return SequenceData(
        X_train=X[is_train_all],
        y_train=y[is_train_all],
        X_valid=X[~is_train_all],
        y_valid=y[~is_train_all],
        valid_meta=meta.loc[~is_train_all].reset_index(drop=True),
        feature_names=feature_names,
        cutoff=cutoff,
    )


class LSTMRegressor(nn.Module):
    """LSTM sequence regressor."""

    def __init__(self, input_size: int, hidden_size: int = 64, dropout: float = 0.15):
        super().__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(x)
        return self.head(output[:, -1, :]).squeeze(-1)


class GRURegressor(nn.Module):
    """GRU sequence regressor."""

    def __init__(self, input_size: int, hidden_size: int = 64, dropout: float = 0.15):
        super().__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(x)
        return self.head(output[:, -1, :]).squeeze(-1)


class CNNLSTMRegressor(nn.Module):
    """1D CNN feature extractor followed by LSTM."""

    def __init__(self, input_size: int, hidden_size: int = 64, dropout: float = 0.15):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(48, 48, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.rnn = nn.LSTM(input_size=48, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv = self.conv(x.transpose(1, 2)).transpose(1, 2)
        output, _ = self.rnn(conv)
        return self.head(output[:, -1, :]).squeeze(-1)


class Chomp1d(nn.Module):
    """Remove right padding for causal convolution."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous() if self.chomp_size else x


class TemporalBlock(nn.Module):
    """Dilated causal convolution block."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TCNRegressor(nn.Module):
    """Compact Temporal Convolutional Network regressor."""

    def __init__(self, input_size: int, channels: int = 64, dropout: float = 0.15):
        super().__init__()
        self.input_projection = nn.Conv1d(input_size, channels, kernel_size=1)
        self.blocks = nn.Sequential(
            TemporalBlock(channels, kernel_size=3, dilation=1, dropout=dropout),
            TemporalBlock(channels, kernel_size=3, dilation=2, dropout=dropout),
            TemporalBlock(channels, kernel_size=3, dilation=4, dropout=dropout),
        )
        self.head = nn.Sequential(nn.Linear(channels, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.input_projection(x.transpose(1, 2))
        output = self.blocks(features)
        return self.head(output[:, :, -1]).squeeze(-1)


def make_loaders(
    data: SequenceData,
    batch_size: int,
    max_train_rows: int,
) -> tuple[DataLoader, DataLoader, np.ndarray, np.ndarray, float, float]:
    """Create PyTorch data loaders, optionally using the latest training rows."""
    if max_train_rows > 0 and len(data.X_train) > max_train_rows:
        X_train = data.X_train[-max_train_rows:]
        y_train = data.y_train[-max_train_rows:]
    else:
        X_train = data.X_train
        y_train = data.y_train

    target_mean = float(data.y_train.mean())
    target_std = float(data.y_train.std() + 1e-6)
    y_train_scaled = ((y_train - target_mean) / target_std).astype(np.float32)
    y_valid_scaled = ((data.y_valid - target_mean) / target_std).astype(np.float32)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train_scaled))
    valid_ds = TensorDataset(torch.from_numpy(data.X_valid), torch.from_numpy(y_valid_scaled))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size * 2, shuffle=False)
    return train_loader, valid_loader, X_train, y_train, target_mean, target_std


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Predict a full data loader."""
    model.eval()
    preds = []
    with torch.no_grad():
        for x_batch, _ in loader:
            preds.append(model(x_batch.to(device)).cpu().numpy())
    return np.concatenate(preds)


def train_model(
    name: str,
    model: nn.Module,
    data: SequenceData,
    device: torch.device,
    epochs: int,
    batch_size: int,
    max_train_rows: int,
    learning_rate: float,
    patience: int,
) -> tuple[dict, np.ndarray]:
    """Train one temporal neural model with early stopping on validation RMSE."""
    train_loader, valid_loader, X_train_used, _, target_mean, target_std = make_loaders(data, batch_size, max_train_rows)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_rmse = math.inf
    best_state = None
    best_pred = None
    wait = 0
    start = time.perf_counter()

    for _ in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_batch), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        val_pred_scaled = predict(model, valid_loader, device)
        val_pred = val_pred_scaled * target_std + target_mean
        val_rmse = rmse(data.y_valid, val_pred)
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_pred = val_pred
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    fit_seconds = time.perf_counter() - start
    row = {
        "model": name,
        "rmse": rmse(data.y_valid, best_pred),
        "mae": mae(data.y_valid, best_pred),
        "fit_seconds": round(fit_seconds, 3),
        "train_windows": len(X_train_used),
        "validation_windows": len(data.X_valid),
        "notes": f"target_standardized=true, epochs={epochs}, batch_size={batch_size}, max_train_rows={max_train_rows}",
    }
    return row, best_pred


def station_metrics(predictions: pd.DataFrame, model_columns: list[str]) -> pd.DataFrame:
    """Compute station-wise RMSE and MAE."""
    rows = []
    for model in model_columns:
        for station, group in predictions.groupby("station"):
            rows.append(
                {
                    "model": model,
                    "station": station,
                    "rows": len(group),
                    "rmse": rmse(group["y_true"], group[model]),
                    "mae": mae(group["y_true"], group[model]),
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "rmse"]).reset_index(drop=True)


def run(
    data_dir: str | Path,
    output_dir: str | Path,
    train_fraction: float,
    window_size: int,
    epochs: int,
    batch_size: int,
    max_train_rows: int,
    random_state: int,
    models: str,
) -> None:
    """Run temporal neural model experiments."""
    set_seed(random_state)
    output_dir = Path(output_dir)
    report_dir = output_dir / "reports" / "temporal_neural_models"
    report_dir.mkdir(parents=True, exist_ok=True)

    files = read_competition_files(data_dir)
    train_raw = files.get("train_raw.csv")
    if train_raw is None:
        raise FileNotFoundError("train_raw.csv is required for temporal neural model experiments.")

    data = build_sequence_data(train_raw=train_raw, train_fraction=train_fraction, window_size=window_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = data.X_train.shape[-1]

    requested = {name.strip() for name in models.split(",") if name.strip()}
    all_model_specs = {
        "lstm": ("lstm_64", LSTMRegressor(input_size=input_size, hidden_size=64)),
        "gru": ("gru_64", GRURegressor(input_size=input_size, hidden_size=64)),
        "cnn_lstm": ("cnn_lstm_64", CNNLSTMRegressor(input_size=input_size, hidden_size=64)),
        "tcn": ("tcn_64", TCNRegressor(input_size=input_size, channels=64)),
    }
    unknown = requested - set(all_model_specs)
    if unknown:
        raise ValueError(f"Unknown model names: {sorted(unknown)}")
    model_specs = [all_model_specs[name] for name in ["lstm", "gru", "cnn_lstm", "tcn"] if name in requested]

    results = []
    predictions = data.valid_meta.copy()
    predictions["y_true"] = data.y_valid

    comparison_rows = [
        {
            "model": "boosting_weighted_ensemble_reference",
            "rmse": 19.5651,
            "mae": 9.6886,
            "fit_seconds": 0.0,
            "train_windows": 247162,
            "validation_windows": len(data.X_valid),
            "notes": "Reference from gradient_boosting_models.py on the same chronological split",
        },
        {
            "model": "ridge_reference",
            "rmse": 19.7469,
            "mae": 9.9710,
            "fit_seconds": 0.0,
            "train_windows": 247162,
            "validation_windows": len(data.X_valid),
            "notes": "Reference from gradient_boosting_models.py on the same chronological split",
        },
    ]
    results.extend(comparison_rows)

    for name, model in model_specs:
        print(f"Training {name}...")
        row, pred = train_model(
            name=name,
            model=model,
            data=data,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            max_train_rows=max_train_rows,
            learning_rate=1e-3,
            patience=2,
        )
        results.append(row)
        predictions[name] = pred

    result_df = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    neural_cols = [name for name, _ in model_specs]
    per_station = station_metrics(predictions, neural_cols)
    summary = pd.DataFrame(
        [
            {
                "train_fraction": train_fraction,
                "cutoff": data.cutoff,
                "window_size": window_size,
                "train_windows": len(data.X_train),
                "validation_windows": len(data.X_valid),
                "feature_count": input_size,
                "features": ",".join(data.feature_names),
                "device": str(device),
                "epochs": epochs,
                "batch_size": batch_size,
                "max_train_rows": max_train_rows,
                "random_state": random_state,
                "requested_models": models,
            }
        ]
    )

    summary.to_csv(report_dir / "experiment_summary.csv", index=False)
    result_df.to_csv(report_dir / "model_results.csv", index=False)
    per_station.to_csv(report_dir / "station_model_results.csv", index=False)
    predictions.to_csv(report_dir / "validation_predictions.csv", index=False)

    print("Temporal neural model experiments complete.")
    print(summary.to_string(index=False))
    print(result_df.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_LOCAL_DATA_DIR), help="Directory containing train_raw.csv.")
    parser.add_argument("--output-dir", default=".", help="Directory where reports are written.")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--window-size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-train-rows", type=int, default=140_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--models",
        default="lstm,gru,cnn_lstm",
        help="Comma-separated subset from: lstm,gru,cnn_lstm,tcn. TCN is implemented but slower on CPU.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_fraction=args.train_fraction,
        window_size=args.window_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_train_rows=args.max_train_rows,
        random_state=args.random_state,
        models=args.models,
    )
