"""Data ingestion, chronological alignment, and Parquet export pipeline using Polars."""

from __future__ import annotations

import datetime
from pathlib import Path

import polars as pl

from src.config import (
    BASE_DIR,
    DATA_DIR,
    MISSING_CALENDAR_DATE_STR,
    OUTPUT_PARQUET_FILENAME,
    RAW_CSV_FILENAME,
)
from src.schema import generate_full_calendar_range, validate_dataframe_invariants


def parse_and_sort_date_columns(columns: list[str]) -> list[tuple[str, str, datetime.date]]:
    """Map raw date column names to (raw_name, iso_name, parsed_date) and sort chronologically."""
    date_mappings = []
    for col in columns:
        if col in ("CONS_NO", "FLAG"):
            continue
        try:
            d = datetime.datetime.strptime(col, "%Y/%m/%d").date()
        except ValueError:
            d = datetime.datetime.strptime(col, "%Y-%m-%d").date()
        iso_str = d.strftime("%Y-%m-%d")
        date_mappings.append((col, iso_str, d))

    # Sort strictly by date object
    date_mappings.sort(key=lambda x: x[2])
    return date_mappings


def ingest_raw_csv_to_polars(
    csv_path: Path | str, fill_missing_calendar_dates: bool = True
) -> pl.DataFrame:
    """Read raw SGCC CSV, rename headers to ISO dates, sort chronologically, and cast types."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    # Read CSV schema first
    df = pl.read_csv(
        path,
        infer_schema_length=10000,
        null_values=["", "NA", "null", "NaN", "None"],
    )

    date_mappings = parse_and_sort_date_columns(df.columns)

    # Rename original date columns to ISO standard YYYY-MM-DD
    rename_dict = {raw_name: iso_name for raw_name, iso_name, _ in date_mappings}
    df = df.rename(rename_dict)

    # Cast FLAG to Int8 and CONS_NO to String
    df = df.with_columns(
        [
            pl.col("CONS_NO").cast(pl.String),
            pl.col("FLAG").cast(pl.Int8),
        ]
    )

    # Ensure all date columns are Float32
    sorted_iso_cols = [iso_name for _, iso_name, _ in date_mappings]
    cast_exprs = [pl.col(c).cast(pl.Float32) for c in sorted_iso_cols]
    df = df.with_columns(cast_exprs)

    # Reorder columns: CONS_NO, FLAG, then sorted chronological date columns
    ordered_cols = ["CONS_NO", "FLAG"] + sorted_iso_cols

    # Handle missing calendar date (2016-09-18)
    if fill_missing_calendar_dates:
        full_dates = generate_full_calendar_range()
        full_iso_strs = [d.strftime("%Y-%m-%d") for d in full_dates]
        if MISSING_CALENDAR_DATE_STR not in df.columns:
            # Add missing date as null column, then position it chronologically
            df = df.with_columns(pl.lit(None, dtype=pl.Float32).alias(MISSING_CALENDAR_DATE_STR))
            ordered_cols = ["CONS_NO", "FLAG"] + full_iso_strs

    df = df.select(ordered_cols)
    return df


def convert_csv_to_parquet(
    csv_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    filename: str = OUTPUT_PARQUET_FILENAME,
    is_sample: bool = False,
) -> Path:
    """Run full ingestion pipeline and write compressed Parquet dataset."""
    base = Path(BASE_DIR)
    target_csv = Path(csv_path) if csv_path else base / RAW_CSV_FILENAME
    target_out_dir = Path(output_dir) if output_dir else DATA_DIR
    target_out_dir.mkdir(parents=True, exist_ok=True)

    target_parquet = target_out_dir / filename

    print(f"Reading raw CSV from {target_csv}...")
    df = ingest_raw_csv_to_polars(target_csv)

    print("Validating dataset invariants...")
    metrics = validate_dataframe_invariants(df, is_sample=is_sample)
    print(f"Validated {metrics['num_rows']} customers, {metrics['num_cols']} columns.")
    print(f"Class distribution: {metrics['flag_distribution']}")

    print(f"Writing Parquet file with zstd compression to {target_parquet}...")
    df.write_parquet(target_parquet, compression="zstd", compression_level=7)

    raw_mb = target_csv.stat().st_size / (1024 * 1024)
    parquet_mb = target_parquet.stat().st_size / (1024 * 1024)
    print(
        f"Conversion complete! Original: {raw_mb:.1f} MB -> Parquet: {parquet_mb:.1f} MB "
        f"({(1 - parquet_mb / raw_mb) * 100:.1f}% reduction)"
    )

    return target_parquet
