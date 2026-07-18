"""Data-loading helpers for local and Kaggle environments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import COMPETITION_SLUG


def find_data_dir(preferred: str | Path | None = None) -> Path:
    """Find the directory containing the competition CSV files."""
    candidates: list[Path] = []
    if preferred is not None:
        candidates.append(Path(preferred))

    candidates.extend(
        [
            Path("/kaggle/input") / COMPETITION_SLUG,
            Path("/kaggle/input"),
            Path("data/raw"),
            Path("../input") / COMPETITION_SLUG,
            Path("../input"),
        ]
    )

    for candidate in candidates:
        if (candidate / "train_raw.csv").exists() or (candidate / "test.csv").exists():
            return candidate

    checked = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find Kaggle CSV files. Place train_raw.csv, test.csv, and "
        f"sample_submission.csv in data/raw or run in Kaggle. Checked:\n{checked}"
    )


def read_competition_files(data_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Read available competition files from a data directory."""
    resolved = find_data_dir(data_dir)
    files = {}
    for name in ["train_raw.csv", "test.csv", "sample_submission.csv", "test_raw.csv"]:
        path = resolved / name
        if path.exists():
            files[name] = pd.read_csv(path)
    return files

