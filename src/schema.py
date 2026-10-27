"""Pandera schema specifications and invariant validation for the SGCC dataset."""

from __future__ import annotations

import datetime

import polars as pl

from src.config import (
    END_DATE_STR,
    EXPECTED_FRAUD_COUNT,
    EXPECTED_NORMAL_COUNT,
    EXPECTED_NUM_CUSTOMERS,
    START_DATE_STR,
)


def validate_raw_date_columns(columns: list[str]) -> tuple[list[datetime.date], bool]:
    """Parse raw date headers, test if they are strictly sorted, and return sorted dates."""
    date_cols = [c for c in columns if c not in ("CONS_NO", "FLAG")]
    parsed_dates = []
    for col in date_cols:
        try:
            d = datetime.datetime.strptime(col, "%Y/%m/%d").date()
        except ValueError:
            d = datetime.datetime.strptime(col, "%Y-%m-%d").date()
        parsed_dates.append(d)

    is_sorted = parsed_dates == sorted(parsed_dates)
    return parsed_dates, is_sorted


def generate_full_calendar_range(
    start_str: str = START_DATE_STR, end_str: str = END_DATE_STR
) -> list[datetime.date]:
    """Generate complete unbroken calendar sequence between start and end dates."""
    start = datetime.datetime.strptime(start_str, "%Y-%m-%d").date()
    end = datetime.datetime.strptime(end_str, "%Y-%m-%d").date()
    curr = start
    dates = []
    while curr <= end:
        dates.append(curr)
        curr += datetime.timedelta(days=1)
    return dates


def validate_dataframe_invariants(df: pl.DataFrame, is_sample: bool = False) -> dict:
    """Validate global invariants for SGCC electricity theft dataset."""
    metrics = {
        "num_rows": df.height,
        "num_cols": df.width,
        "unique_customers": df["CONS_NO"].n_unique(),
        "flag_distribution": {},
        "has_negative_values": False,
    }

    flag_counts = df["FLAG"].value_counts().to_dicts()
    for row in flag_counts:
        metrics["flag_distribution"][int(row["FLAG"])] = int(row["count"])

    if not is_sample:
        if metrics["num_rows"] != EXPECTED_NUM_CUSTOMERS:
            raise ValueError(f"Expected {EXPECTED_NUM_CUSTOMERS} rows, found {metrics['num_rows']}")
        if metrics["unique_customers"] != EXPECTED_NUM_CUSTOMERS:
            raise ValueError(
                f"Customer IDs are not unique: found {metrics['unique_customers']} unique out of {metrics['num_rows']}"
            )
        theft_count = metrics["flag_distribution"].get(1, 0)
        normal_count = metrics["flag_distribution"].get(0, 0)
        if theft_count != EXPECTED_FRAUD_COUNT or normal_count != EXPECTED_NORMAL_COUNT:
            raise ValueError(
                f"Class balance mismatch: expected {EXPECTED_FRAUD_COUNT} fraud and {EXPECTED_NORMAL_COUNT} normal, "
                f"got {theft_count} fraud and {normal_count} normal"
            )

    # Check for negative consumption values across numeric columns
    numeric_cols = [c for c in df.columns if c not in ("CONS_NO", "FLAG")]
    for col in numeric_cols:
        min_val = df[col].drop_nulls().min()
        if min_val is not None and min_val < 0:
            metrics["has_negative_values"] = True
            break

    return metrics
