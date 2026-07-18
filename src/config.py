"""Project constants."""

from pathlib import Path

COMPETITION_SLUG = "co-5420-air-pollution-forecasting-using-temporal-n-ns"
DEFAULT_LOCAL_DATA_DIR = Path("data/raw")
DEFAULT_OUTPUT_DIR = Path(".")

TARGET = "PM2.5"
STATION_COL = "station"
DATETIME_COL = "datetime"

RAW_TIME_COLS = ["year", "month", "day", "hour"]
RAW_NUMERIC_FEATURES = [
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM",
]
RAW_CATEGORICAL_FEATURES = ["wd", "station"]

