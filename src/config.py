"""Configuration constants and file integrity parameters."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_SPLIT_FILES = {
    "data.z01": "c324df53c88358a50aa23fd843b1e15af06e7a20b72d901a98d957c304a52b67",
    "data.z02": "34a30c8eea0fdfa77d58e15cd01c5593ea354ead4cc408a1b87364f6c46d4ed7",
    "data.zip": "1e06ad5f5e13f56f2a72bea304864d259e060d6ad95b3b030a4ad050d8df82d4",
}

RAW_CSV_FILENAME = "data.csv"
OUTPUT_PARQUET_FILENAME = "sgcc_electricity_theft.parquet"
OUTPUT_CLEAN_PARQUET_FILENAME = "sgcc_electricity_theft_preprocessed.parquet"

# Dataset Characteristics
EXPECTED_NUM_CUSTOMERS = 42372
EXPECTED_FRAUD_COUNT = 3615
EXPECTED_NORMAL_COUNT = 38757
EXPECTED_RAW_DATE_COLUMNS = 1034

START_DATE_STR = "2014-01-01"
END_DATE_STR = "2016-10-31"
MISSING_CALENDAR_DATE_STR = "2016-09-18"

# Weekly matrix dimensions for CNN intake
WEEKLY_CYCLE_DAYS = 7
NUM_WEEKS = 148  # 1035 / 7 ~ 147.85 weeks -> 148 weeks (padded/aligned)
