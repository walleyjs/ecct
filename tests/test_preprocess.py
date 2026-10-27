"""Tests for 3-sigma outlier clipping, imputation, scaling, and 2D weekly reshaping."""

from pathlib import Path

import numpy as np

from src.ingestion import ingest_raw_csv_to_polars
from src.preprocess import (
    apply_three_sigma_clipping,
    interpolate_missing_values,
    min_max_scale_series,
    preprocess_consumption_matrix,
    transform_1d_to_2d_weekly,
)


def test_three_sigma_clipping() -> None:
    """Ensure outliers exceeding 3 standard deviations are clipped."""
    # Data of length 50 with extreme outlier at index 5 (N must be >= 10 for 3-sigma detection)
    data = np.full(50, 10.0, dtype=np.float64)
    data[5] = 500.0
    clipped = apply_three_sigma_clipping(data)

    assert clipped[5] < 500.0
    assert np.all(clipped >= 0)


def test_interpolate_missing_values() -> None:
    """Ensure internal NaN values are linearly interpolated."""
    data = np.array([10.0, np.nan, 30.0, np.nan, 50.0], dtype=np.float64)
    imputed = interpolate_missing_values(data)

    assert not np.any(np.isnan(imputed))
    assert np.isclose(imputed[1], 20.0)
    assert np.isclose(imputed[3], 40.0)


def test_min_max_scaling() -> None:
    """Ensure series is bounded in [0, 1]."""
    data = np.array([5.0, 15.0, 25.0], dtype=np.float64)
    scaled = min_max_scale_series(data)

    assert np.isclose(scaled[0], 0.0)
    assert np.isclose(scaled[1], 0.5)
    assert np.isclose(scaled[2], 1.0)
    assert np.all((scaled >= 0.0) & (scaled <= 1.0))


def test_transform_1d_to_2d_weekly() -> None:
    """Verify reshaping 1D daily consumption into 2D weekly periodicity (Weeks x 7)."""
    # 15 days should reshape into ceil(15 / 7) = 3 weeks -> shape (3, 7)
    data = np.arange(15, dtype=np.float32)
    matrix_2d = transform_1d_to_2d_weekly(data, days_per_week=7)

    assert matrix_2d.shape == (3, 7)
    # Check that original 15 days are preserved in order
    assert np.array_equal(matrix_2d.flatten()[:15], data)


def test_full_sample_preprocessing(sample_csv_path: Path) -> None:
    """Test full matrix preprocessing pipeline on the 100-meter fixture."""
    df = ingest_raw_csv_to_polars(sample_csv_path)
    X, y, customer_ids = preprocess_consumption_matrix(df)

    assert X.shape == (100, 1035)
    assert y.shape == (100,)
    assert len(customer_ids) == 100
    assert not np.any(np.isnan(X))
    assert np.all(X >= 0.0)
    assert np.all(X <= 1.0)
