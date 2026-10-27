"""Smart-grid preprocessing pipeline following Zheng et al. (IEEE TII 2018).

Methods implemented:
1. 3-Sigma Rule Outlier Clipping
2. Missing Value Linear Interpolation
3. Per-Customer Min-Max Scaling [0, 1]
4. 2D Weekly Periodicity Transformation (Weeks x 7 Days) for CNN intake
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl


def apply_three_sigma_clipping(arr: np.ndarray) -> np.ndarray:
    """Clip values exceeding mu + 3 * sigma for a single customer time series."""
    valid_mask = ~np.isnan(arr)
    if not np.any(valid_mask):
        return arr

    valid_vals = arr[valid_mask]
    mean = np.mean(valid_vals)
    std = np.std(valid_vals)

    upper_bound = mean + 3.0 * std
    # Clamp negative values to 0 and values above 3-sigma threshold to upper_bound
    clipped = np.copy(arr)
    clipped = np.where(clipped < 0, 0.0, clipped)
    clipped = np.where(clipped > upper_bound, upper_bound, clipped)
    return clipped


def interpolate_missing_values(arr: np.ndarray) -> np.ndarray:
    """Interpolate missing readings linearly using surrounding time-steps."""
    if not np.any(np.isnan(arr)):
        return arr

    n = len(arr)
    indices = np.arange(n)
    valid_mask = ~np.isnan(arr)

    # Edge case: entire series is NaN
    if not np.any(valid_mask):
        return np.zeros_like(arr)

    # If only one valid point, fill everything with that point
    if np.sum(valid_mask) == 1:
        return np.full_like(arr, arr[valid_mask][0])

    # Standard linear 1D interpolation
    interpolated = np.interp(indices, indices[valid_mask], arr[valid_mask])
    return interpolated


def min_max_scale_series(arr: np.ndarray) -> np.ndarray:
    """Scale a 1D time series to [0, 1] per customer."""
    min_val = np.min(arr)
    max_val = np.max(arr)

    range_val = max_val - min_val
    if range_val == 0.0 or np.isnan(range_val):
        return np.zeros_like(arr)

    return (arr - min_val) / range_val


def transform_1d_to_2d_weekly(time_series: np.ndarray, days_per_week: int = 7) -> np.ndarray:
    """Reshape a 1D customer daily sequence into a 2D matrix (Weeks x Days).

    As described in Section III.B of Zheng et al. (IEEE TII 2018), smart meter consumption
    exhibits strong 7-day periodicity. Reshaping to (W, 7) allows 2D convolutional filters
    to capture both day-of-week and week-over-week temporal dependencies.
    """
    total_days = len(time_series)
    num_weeks = math.ceil(total_days / days_per_week)
    padded_length = num_weeks * days_per_week

    # Pad end with the last observed value
    if padded_length > total_days:
        padding = np.full(padded_length - total_days, time_series[-1])
        padded = np.concatenate([time_series, padding])
    else:
        padded = time_series

    return padded.reshape((num_weeks, days_per_week))


def preprocess_consumption_matrix(
    df: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Execute complete 3-step preprocessing pipeline on Polars DataFrame.

    Returns:
        X_scaled: np.ndarray of shape (num_customers, num_days) in range [0, 1]
        y: np.ndarray of shape (num_customers,) binary labels {0, 1}
        customer_ids: list of customer ID strings
    """
    customer_ids = df["CONS_NO"].to_list()
    y = df["FLAG"].to_numpy().astype(np.int8)

    date_cols = [c for c in df.columns if c not in ("CONS_NO", "FLAG")]
    raw_matrix = df.select(date_cols).to_numpy()

    num_customers, num_days = raw_matrix.shape
    X_processed = np.empty((num_customers, num_days), dtype=np.float32)

    for i in range(num_customers):
        row = raw_matrix[i].astype(np.float64)
        clipped = apply_three_sigma_clipping(row)
        imputed = interpolate_missing_values(clipped)
        scaled = min_max_scale_series(imputed)
        X_processed[i] = scaled.astype(np.float32)

    return X_processed, y, customer_ids
