"""
Regression target construction for the triangle z-score forecasting model.

Target definition:
    z_future_H(t) = zscore(t + H)

In plain English: what is the EWMA z-score of the triangle residual H bars from now?

A predicted z_future close to zero means the model expects the residual to mean-revert;
a large |z_future| means the model expects the gap to persist or widen.

The entry gate replaces P(closure) > 0.65 with:
    |z_current - z_future_predicted| > move_threshold   (e.g. 1.0 z-score units)

Important: targets are forward-looking by design. Apply a buffer of ≥ SPLIT_BUFFER_BARS bars
at each split boundary before using these targets for training.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Horizon constants (wall-clock duration → bar count at 10s resolution)
# ---------------------------------------------------------------------------
HORIZON_30_BARS  =  30   #  5 min wall-clock =  30 × 10s bars
HORIZON_60_BARS  =  60   # 10 min wall-clock =  60 × 10s bars  ← primary
HORIZON_180_BARS = 180   # 30 min wall-clock = 180 × 10s bars

# Buffer to apply at split boundaries — must be ≥ longest horizon used for training
SPLIT_BUFFER_BARS = HORIZON_60_BARS   # 60 bars = 10 min wall-clock


def compute_future_zscore_targets(
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Attach z_future_30, z_future_60, z_future_180 as regression targets.

    Each column is the zscore shifted forward by the corresponding horizon.
    The last HORIZON_180_BARS rows will be NaN for the longest target.

    Args:
        features: Feature DataFrame containing a 'zscore' column.

    Returns:
        Copy of features with three new columns appended:
            z_future_30   — zscore 30 bars ahead  (5 min wall-clock)
            z_future_60   — zscore 60 bars ahead  (10 min wall-clock)  ← primary
            z_future_180  — zscore 180 bars ahead (30 min wall-clock)
    """
    features = features.copy()
    features["z_future_30"]  = features["zscore"].shift(-HORIZON_30_BARS)
    features["z_future_60"]  = features["zscore"].shift(-HORIZON_60_BARS)
    features["z_future_180"] = features["zscore"].shift(-HORIZON_180_BARS)
    return features


def split_by_date(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
    buffer_bars: int = SPLIT_BUFFER_BARS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological date-boundary split with forward-target leakage buffer.

    Args:
        df:           Full feature+target DataFrame with DatetimeIndex.
        train_end:    Inclusive end date for training, e.g. "2024-12-31".
        val_end:      Inclusive end date for validation, e.g. "2025-06-30".
        buffer_bars:  Number of bars to drop at each boundary to prevent
                      forward-target leakage. Default = SPLIT_BUFFER_BARS (60).

    Returns:
        (train, val, test) DataFrames. Buffer bars are dropped from the END
        of each split (not the start) to avoid contamination.
    """
    train_mask = df.index <= train_end
    val_mask   = (df.index >  train_end) & (df.index <= val_end)
    test_mask  = df.index >  val_end

    train = df[train_mask].iloc[:-buffer_bars]   # drop last buffer_bars
    val   = df[val_mask].iloc[:-buffer_bars]
    test  = df[test_mask]                         # test: no trailing buffer needed

    return train, val, test
