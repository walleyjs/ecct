"""Pytest fixtures and configuration."""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_CSV_PATH = FIXTURES_DIR / "sample_100_meters.csv"


@pytest.fixture
def sample_csv_path() -> Path:
    """Return path to 100-meter sample CSV."""
    assert SAMPLE_CSV_PATH.exists(), f"Sample fixture not found at {SAMPLE_CSV_PATH}"
    return SAMPLE_CSV_PATH
