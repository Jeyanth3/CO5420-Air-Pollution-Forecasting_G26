"""Prepare competition-style raw files from the 12-station Beijing source CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import DATETIME_COL, RAW_TIME_COLS


TRAIN_END = pd.Timestamp("2016-02-29 23:00:00")
TEST_RAW_START = pd.Timestamp("2016-03-01 00:00:00")
TEST_RAW_END = pd.Timestamp("2017-02-28 23:00:00")


def read_station_files(source_dir: str | Path) -> pd.DataFrame:
    """Read and concatenate the 12 PRSA station CSV files."""
    source_dir = Path(source_dir)
    paths = sorted(source_dir.glob("PRSA_Data_*_20130301-20170228.csv"))
    if len(paths) != 12:
        raise FileNotFoundError(
            f"Expected 12 PRSA station CSV files in {source_dir}, found {len(paths)}."
        )

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frames.append(frame)

    raw = pd.concat(frames, ignore_index=True)
    raw[DATETIME_COL] = pd.to_datetime(raw[RAW_TIME_COLS])
    raw = raw.sort_values(["station", DATETIME_COL]).reset_index(drop=True)
    return raw


def prepare_train_raw(source_dir: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Create train_raw.csv covering 2013-03-01 through 2016-02-29."""
    raw = read_station_files(source_dir)
    train = raw[raw[DATETIME_COL] <= TRAIN_END].copy()
    train = train.drop(columns=[DATETIME_COL])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(output_path, index=False)
    return train


def prepare_test_raw(source_dir: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Create test_raw.csv covering 2016-03-01 through 2017-02-28 for extended work."""
    raw = read_station_files(source_dir)
    mask = (raw[DATETIME_COL] >= TEST_RAW_START) & (raw[DATETIME_COL] <= TEST_RAW_END)
    test_raw = raw[mask].copy().drop(columns=[DATETIME_COL])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    test_raw.to_csv(output_path, index=False)
    return test_raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, help="Folder containing the 12 PRSA source CSV files.")
    parser.add_argument("--output-dir", default="data/raw", help="Directory where train_raw.csv is written.")
    parser.add_argument("--include-test-raw", action="store_true", help="Also write test_raw.csv for extended work.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    train = prepare_train_raw(args.source_dir, output_dir / "train_raw.csv")
    print(f"Wrote {output_dir / 'train_raw.csv'} with shape {train.shape}")

    if args.include_test_raw:
        test_raw = prepare_test_raw(args.source_dir, output_dir / "test_raw.csv")
        print(f"Wrote {output_dir / 'test_raw.csv'} with shape {test_raw.shape}")


if __name__ == "__main__":
    main()

