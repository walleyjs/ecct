"""Tests for schema parsing, date ordering, and data invariants."""

from pathlib import Path

import polars as pl

from src.ingestion import ingest_raw_csv_to_polars
from src.schema import (
    generate_full_calendar_range,
    validate_dataframe_invariants,
    validate_raw_date_columns,
)


def test_raw_date_columns_are_scrambled_lexicographically(sample_csv_path: Path) -> None:
    """Demonstrate the raw SGCC CSV date columns are lexicographically rather than chronologically ordered."""
    df_raw = pl.read_csv(sample_csv_path, n_rows=2)
    _, is_sorted = validate_raw_date_columns(df_raw.columns)
    assert not is_sorted, "Raw CSV columns were unexpectedly already sorted chronologically"


def test_full_calendar_range_continuity() -> None:
    """Verify that generate_full_calendar_range produces 1035 contiguous dates."""
    dates = generate_full_calendar_range()
    assert len(dates) == 1035
    assert str(dates[0]) == "2014-01-01"
    assert str(dates[-1]) == "2016-10-31"


def test_sample_dataframe_invariants(sample_csv_path: Path) -> None:
    """Verify that sample dataset respects schema invariants after ingestion."""
    df = ingest_raw_csv_to_polars(sample_csv_path, fill_missing_calendar_dates=True)
    metrics = validate_dataframe_invariants(df, is_sample=True)

    assert metrics["num_rows"] == 100
    assert metrics["unique_customers"] == 100
    assert "CONS_NO" in df.columns
    assert "FLAG" in df.columns
    # Check that 1035 dates + CONS_NO + FLAG = 1037 total columns
    assert df.width == 1037
