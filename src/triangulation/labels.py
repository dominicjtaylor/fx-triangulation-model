"""
Binary label construction for the triangle gap closure classifier.

Label definition:
    y(t) = 1 if |residual[t+k]| ≤ 0.5 × |residual[t]| for any k ∈ [1, H]
    y(t) = 0 otherwise

In plain English: did the triangle gap close at least 50% of its current size
within the next H bars (wall-clock: 10 minutes at 10s resolution)?

Important: labels are forward-looking by design. To prevent data leakage
across train/val/test splits, apply a buffer of ≥ LABEL_10MIN_BARS bars at
each split boundary before using these labels for training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from triangulation.features import LABEL_10MIN_BARS

# Buffer to apply at split boundaries — must be ≥ label horizon
SPLIT_BUFFER_BARS = LABEL_10MIN_BARS  # 60 bars = 10 min wall-clock


def compute_binary_label(
    residual: pd.Series,
    horizon_bars: int = LABEL_10MIN_BARS,
) -> pd.Series:
    """Compute the binary gap-closure label at each bar.

    For each bar t: label = 1 if the residual's absolute value drops to ≤ 50%
    of its value at t within the next `horizon_bars` bars.

    Args:
        residual:     Raw triangle residual at 10s resolution.
        horizon_bars: Look-ahead window. Default 60 = 10 min wall-clock.

    Returns:
        Integer Series (0/1) named 'label'. The last `horizon_bars - 1` bars
        will have a truncated look-ahead (fewer future bars available) but are
        not NaN — they will have lower label quality near the series end.
    """
    abs_r = residual.abs()

    # Forward rolling minimum over [t+1, ..., t+horizon_bars]:
    #   1. Reverse the series.
    #   2. Compute rolling min (window = horizon_bars).
    #      At reversed position i this gives min over original positions
    #      [n-1-i, ..., n-1-i-(horizon_bars-1)], i.e. the next horizon_bars
    #      values going forward in time.
    #   3. Reverse back and shift(-1) to exclude t itself and include t+H.
    abs_fwd_min = (
        abs_r.iloc[::-1]
        .rolling(horizon_bars, min_periods=1)
        .min()
        .iloc[::-1]
        .shift(-1)
    )

    label = (abs_fwd_min <= abs_r * 0.5).astype(int)
    label.name = "label"
    return label


def add_labels(
    features: pd.DataFrame,
    horizon_bars: int = LABEL_10MIN_BARS,
) -> pd.DataFrame:
    """Attach the binary label column to a feature DataFrame in-place.

    Args:
        features:     Feature DataFrame containing a 'residual' column.
        horizon_bars: Label horizon in 10s bars.

    Returns:
        Same DataFrame with a 'label' column appended.
    """
    features = features.copy()
    features["label"] = compute_binary_label(features["residual"], horizon_bars)
    return features


def split_by_date(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
    buffer_bars: int = SPLIT_BUFFER_BARS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological date-boundary split with label-leakage buffer.

    Args:
        df:           Full feature+label DataFrame with DatetimeIndex.
        train_end:    Inclusive end date for training, e.g. "2024-12-31".
        val_end:      Inclusive end date for validation, e.g. "2025-06-30".
        buffer_bars:  Number of bars to drop at each boundary to prevent
                      forward-label leakage. Default = LABEL_10MIN_BARS (60).

    Returns:
        (train, val, test) DataFrames. Buffer bars are dropped from the END
        of each split (not the start) to avoid contamination.
    """
    train_mask = df.index <= train_end
    val_mask   = (df.index >  train_end) & (df.index <= val_end)
    test_mask  = df.index >  val_end

    train = df[train_mask].iloc[:-buffer_bars]   # drop last H bars
    val   = df[val_mask].iloc[:-buffer_bars]
    test  = df[test_mask]                         # test: no trailing buffer needed

    return train, val, test
