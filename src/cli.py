"""Command-line interface for SGCC Electricity Theft Detection pipeline."""

from __future__ import annotations

import argparse
import sys

from src.baseline import train_and_evaluate_baseline
from src.config import DATA_DIR, OUTPUT_PARQUET_FILENAME
from src.ingestion import convert_csv_to_parquet
from src.integrity import ensure_raw_csv_extracted, verify_split_checksums
from src.preprocess import preprocess_consumption_matrix


def cmd_verify(args: argparse.Namespace) -> None:
    """Verify cryptographic checksums of raw split zips."""
    print("Verifying cryptographic SHA-256 digests for raw multipart split volumes...")
    results = verify_split_checksums()
    all_ok = True
    for filename, is_valid in results.items():
        status = "PASSED [OK]" if is_valid else "FAILED [MISMATCH/MISSING]"
        print(f"  - {filename:12s} : {status}")
        if not is_valid:
            all_ok = False

    if not all_ok:
        print("\n[ERROR] One or more archives failed verification.")
        sys.exit(1)

    print("\n[SUCCESS] All multipart archives verified successfully.")


def cmd_process(args: argparse.Namespace) -> None:
    """Extract, sort chronologically, validate, and convert to Parquet."""
    print("Ensuring raw CSV is unpacked...")
    csv_path = ensure_raw_csv_extracted()

    print("Running ingestion pipeline...")
    output_parquet = convert_csv_to_parquet(
        csv_path=csv_path,
        output_dir=DATA_DIR,
        filename=OUTPUT_PARQUET_FILENAME,
    )
    print(f"\n[SUCCESS] Parquet dataset generated at: {output_parquet}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Preprocess data and run reference baseline model."""
    import polars as pl

    parquet_path = DATA_DIR / OUTPUT_PARQUET_FILENAME
    if not parquet_path.exists():
        print(f"Parquet dataset not found at {parquet_path}. Running ingestion first...")
        csv_path = ensure_raw_csv_extracted()
        convert_csv_to_parquet(csv_path=csv_path)

    print(f"Loading dataset from {parquet_path}...")
    df = pl.read_parquet(parquet_path)

    if args.sample_size and args.sample_size < df.height:
        print(f"Subsampling to {args.sample_size} meters for fast benchmarking...")
        df = df.sample(n=args.sample_size, seed=42)

    print("Running 3-sigma outlier clipping and min-max scaling...")
    X, y, _ = preprocess_consumption_matrix(df)

    print(f"Feature matrix shape: {X.shape}, Target shape: {y.shape}")
    print(f"Fraud count: {y.sum()} / {len(y)} ({y.sum() / len(y) * 100:.2f}%)")

    train_and_evaluate_baseline(X, y, model_type=args.model)


def main() -> None:
    """Entry point for sgcc-theft CLI."""
    parser = argparse.ArgumentParser(
        prog="sgcc-theft",
        description="SGCC Electricity Theft Detection Pipeline CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: verify
    subparsers.add_parser("verify", help="Verify raw multipart zip SHA-256 checksums")

    # Subcommand: process
    subparsers.add_parser(
        "process", help="Convert raw CSV to chronologically sorted, compressed Parquet"
    )

    # Subcommand: benchmark
    parser_bench = subparsers.add_parser(
        "benchmark", help="Train and evaluate baseline classification models"
    )
    parser_bench.add_argument(
        "--model",
        choices=["gradient_boosting", "random_forest"],
        default="gradient_boosting",
        help="Baseline model architecture",
    )
    parser_bench.add_argument(
        "--sample-size",
        type=int,
        default=2000,
        help="Number of customer meters to evaluate for rapid benchmarking",
    )

    args = parser.parse_args()
    if args.command == "verify":
        cmd_verify(args)
    elif args.command == "process":
        cmd_process(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)


if __name__ == "__main__":
    main()
