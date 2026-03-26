"""
Multi-scale feature engineering on the native 10-second residual/zscore.

All inputs are expected at 10s bar resolution. Window sizes are expressed in
bars with the equivalent wall-clock duration noted in comments.

Label horizon reference (wall-clock → bar count at 10s resolution):
    5 min  →   30 bars   (LABEL_5MIN_BARS)
   10 min  →   60 bars   (LABEL_10MIN_BARS)  ← primary label horizon
   30 min  →  180 bars   (LABEL_30MIN_BARS)

Memory note: building the full feature frame for ~4.5M 10s bars across three
pairs requires approximately 800 MB of working RAM.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from triangulation.analysis import ou_halflife

# ---------------------------------------------------------------------------
# Label horizon constants (wall-clock duration → bar count at 10s resolution)
# ---------------------------------------------------------------------------
LABEL_5MIN_BARS  =  30   #  5 min wall-clock =  30 × 10s bars
LABEL_10MIN_BARS =  60   # 10 min wall-clock =  60 × 10s bars  ← primary
LABEL_30MIN_BARS = 180   # 30 min wall-clock = 180 × 10s bars

# ---------------------------------------------------------------------------
# Window sizes for multi-scale features (bars at 10s resolution)
# ---------------------------------------------------------------------------
_W_1M  =    6   #  1 min  =   6 × 10s
_W_5M  =   30   #  5 min  =  30 × 10s
_W_30M =  180   # 30 min  = 180 × 10s
_W_4H  = 1440   #  4 hr   = 1440 × 10s
_W_1H  =  360   #  1 hr   =  360 × 10s  (for per-pair vol)
_W_12H = 4320   # 12 hr   = 4320 × 10s  (for spread normalisation)


# ---------------------------------------------------------------------------
# Residual volatility (multi-scale)
# ---------------------------------------------------------------------------

def compute_multi_scale_rv(residual: pd.Series) -> pd.DataFrame:
    """Rolling realised volatility (std) of the residual at four timescales.

    Returns DataFrame with columns:
        rv_residual_1m   — std over   6 bars (  1 min)
        rv_residual_5m   — std over  30 bars (  5 min)
        rv_residual_30m  — std over 180 bars ( 30 min)
        rv_residual_4h   — std over 1440 bars (  4 hr)
    """
    return pd.DataFrame(
        {
            "rv_residual_1m":  residual.rolling(_W_1M,  min_periods=1).std(),
            "rv_residual_5m":  residual.rolling(_W_5M,  min_periods=1).std(),
            "rv_residual_30m": residual.rolling(_W_30M, min_periods=1).std(),
            "rv_residual_4h":  residual.rolling(_W_4H,  min_periods=1).std(),
        },
        index=residual.index,
    )


# ---------------------------------------------------------------------------
# EWMA mean (multi-scale)
# ---------------------------------------------------------------------------

def compute_multi_scale_ewma_mean(residual: pd.Series) -> pd.DataFrame:
    """EWMA mean of the residual at four timescales (slow drift trackers).

    Returns DataFrame with columns:
        ewma_mean_1m   — span=6    (  1 min half-life ≈ 0.5 min)
        ewma_mean_5m   — span=30   (  5 min half-life ≈ 2.5 min)
        ewma_mean_30m  — span=180  ( 30 min half-life ≈ 15 min)
        ewma_mean_4h   — span=1440 (  4 hr  half-life ≈ 2 hr)
    """
    return pd.DataFrame(
        {
            "ewma_mean_1m":  residual.ewm(span=_W_1M,  min_periods=1).mean(),
            "ewma_mean_5m":  residual.ewm(span=_W_5M,  min_periods=1).mean(),
            "ewma_mean_30m": residual.ewm(span=_W_30M, min_periods=1).mean(),
            "ewma_mean_4h":  residual.ewm(span=_W_4H,  min_periods=1).mean(),
        },
        index=residual.index,
    )


# ---------------------------------------------------------------------------
# Z-score derivatives
# ---------------------------------------------------------------------------

def compute_dz(zscore: pd.Series) -> pd.DataFrame:
    """Z-score velocity and acceleration at native 10s resolution.

    Returns DataFrame with columns:
        dz_10s   — first difference  (velocity, noisy but carries microstructure info)
        d2z_10s  — second difference (acceleration)
    """
    dz  = zscore.diff(1)
    d2z = dz.diff(1)
    return pd.DataFrame({"dz_10s": dz, "d2z_10s": d2z}, index=zscore.index)


# ---------------------------------------------------------------------------
# Per-pair realised vol + vol ratio
# ---------------------------------------------------------------------------

def compute_pair_vol(
    eurusd: pd.Series,
    audusd: pd.Series,
    euraud: pd.Series,
) -> pd.DataFrame:
    """Realised vol of individual pairs and vol leadership ratio at 1-hour window.

    Args:
        eurusd/audusd/euraud: Close price series (already aligned to same index).

    Returns DataFrame with columns:
        rv_eurusd_1h   — std of log returns over 360 bars (1 hr)
        rv_audusd_1h   — std of log returns over 360 bars (1 hr)
        rv_euraud_1h   — std of log returns over 360 bars (1 hr)
        rv_ratio_eu_aud — rv_eurusd_1h / rv_euraud_1h (vol leadership)
    """
    lr_eurusd = np.log(eurusd).diff(1)
    lr_audusd = np.log(audusd).diff(1)
    lr_euraud = np.log(euraud).diff(1)

    rv_eu  = lr_eurusd.rolling(_W_1H, min_periods=1).std()
    rv_au  = lr_audusd.rolling(_W_1H, min_periods=1).std()
    rv_ead = lr_euraud.rolling(_W_1H, min_periods=1).std()

    return pd.DataFrame(
        {
            "rv_eurusd_1h":    rv_eu,
            "rv_audusd_1h":    rv_au,
            "rv_euraud_1h":    rv_ead,
            "rv_ratio_eu_aud": rv_eu / rv_ead.clip(lower=1e-12),
        },
        index=eurusd.index,
    )


# ---------------------------------------------------------------------------
# Session / time-of-day features
# ---------------------------------------------------------------------------

def compute_session_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Cyclical time-of-day encoding and London/NY overlap flag.

    Returns DataFrame with columns:
        hour_sin       — sin(2π × minute_of_day / 1440)
        hour_cos       — cos(2π × minute_of_day / 1440)
        is_london_ny   — 1 if 12:00–16:00 UTC (London/NY overlap), else 0
    """
    # minute_of_day in UTC
    minute_of_day = index.hour * 60 + index.minute
    angle = 2 * np.pi * minute_of_day / 1440

    is_overlap = ((index.hour >= 12) & (index.hour < 16)).astype(int)

    return pd.DataFrame(
        {
            "hour_sin":     np.sin(angle),
            "hour_cos":     np.cos(angle),
            "is_london_ny": is_overlap,
        },
        index=index,
    )


# ---------------------------------------------------------------------------
# Spread features (requires bid/ask)
# ---------------------------------------------------------------------------

def compute_spread_features(
    bid: pd.Series,
    ask: pd.Series,
) -> pd.DataFrame:
    """EUR/AUD bid-ask spread and normalised spread.

    Args:
        bid: EUR/AUD bid price at bar close.
        ask: EUR/AUD ask price at bar close.

    Returns DataFrame with columns:
        spread_euraud       — raw bid-ask spread (ask - bid)
        spread_euraud_norm  — spread / 12-hour EWMA of spread
                              (> 1 = wider than usual, < 1 = tighter)
    """
    spread = ask - bid
    spread_norm = spread / spread.ewm(span=_W_12H, min_periods=1).mean().clip(lower=1e-10)
    return pd.DataFrame(
        {"spread_euraud": spread, "spread_euraud_norm": spread_norm},
        index=bid.index,
    )


# ---------------------------------------------------------------------------
# Interaction features
# ---------------------------------------------------------------------------

def compute_interactions(
    zscore: pd.Series,
    rv_residual_5m: pd.Series,
    spread_norm: pd.Series | None = None,
) -> pd.DataFrame:
    """Multiplicative interaction terms.

    Returns DataFrame with columns:
        z_x_rv      — zscore × rv_residual_5m  (gap size × vol regime)
        z_x_spread  — zscore × spread_norm     (gap penalised by execution cost)
                      Only included if spread_norm is provided.
    """
    cols: dict = {"z_x_rv": zscore * rv_residual_5m}
    if spread_norm is not None:
        cols["z_x_spread"] = zscore * spread_norm
    return pd.DataFrame(cols, index=zscore.index)


# ---------------------------------------------------------------------------
# Rolling OU half-life (expensive — call separately)
# ---------------------------------------------------------------------------

def compute_ou_halflife_feature(
    residual: pd.Series,
    window_bars: int = 360,
    min_periods: int = 180,
) -> pd.Series:
    """Rolling OU half-life estimated on the 1-min smoothed residual.

    The raw 10s residual is too noisy for reliable OU estimation. We first
    smooth it with EWMA span=6 (≈ 1-min equivalent), then estimate the
    Ornstein-Uhlenbeck AR(1) coefficient on a rolling window.

    Args:
        residual:    Raw 10s residual series.
        window_bars: Rolling window size in 10s bars. Default 360 = 1 hour.
        min_periods: Minimum bars before returning a non-NaN estimate.

    Returns:
        Series `ou_halflife_min` — rolling half-life in minutes.
        NaN for the first `min_periods` bars.
    """
    residual_smooth = residual.ewm(span=6, min_periods=1).mean()

    halflife_vals = np.full(len(residual_smooth), np.nan)
    for end in range(min_periods, len(residual_smooth)):
        start = max(0, end - window_bars)
        result = ou_halflife(residual_smooth.iloc[start:end])
        hl = result["halflife_bars"]
        halflife_vals[end] = hl * 10 / 60 if np.isfinite(hl) else np.nan

    return pd.Series(halflife_vals, index=residual.index, name="ou_halflife_min")


# ---------------------------------------------------------------------------
# Main convenience function
# ---------------------------------------------------------------------------

def build_feature_frame(signal_frame: pd.DataFrame) -> pd.DataFrame:
    """Build the full feature frame from a signal frame at 10s resolution.

    Args:
        signal_frame: Output of `build_signal_frame()`. Must contain columns
                      'residual', 'zscore', 'eurusd', 'audusd', 'euraud'.
                      Optionally 'euraud_bid', 'euraud_ask' for spread features.

    Returns:
        DataFrame (~22 columns) ready for model training.
        Does NOT include `ou_halflife_min` (call separately — it is O(n²)).

    Columns produced:
        residual, zscore,
        rv_residual_1m/5m/30m/4h,
        ewma_mean_1m/5m/30m/4h,
        dz_10s, d2z_10s,
        rv_eurusd_1h, rv_audusd_1h, rv_euraud_1h, rv_ratio_eu_aud,
        hour_sin, hour_cos, is_london_ny,
        spread_euraud, spread_euraud_norm,  ← only if bid/ask present
        z_x_rv, z_x_spread                 ← z_x_spread only if bid/ask present
    """
    residual = signal_frame["residual"]
    zscore   = signal_frame["zscore"]

    parts = [
        signal_frame[["residual", "zscore"]],
        compute_multi_scale_rv(residual),
        compute_multi_scale_ewma_mean(residual),
        compute_dz(zscore),
        compute_pair_vol(
            signal_frame["eurusd"],
            signal_frame["audusd"],
            signal_frame["euraud"],
        ),
        compute_session_features(signal_frame.index),
    ]

    spread_norm = None
    if "euraud_bid" in signal_frame.columns and "euraud_ask" in signal_frame.columns:
        spread_df  = compute_spread_features(signal_frame["euraud_bid"], signal_frame["euraud_ask"])
        spread_norm = spread_df["spread_euraud_norm"]
        parts.append(spread_df)

    parts.append(
        compute_interactions(
            zscore,
            rv_residual_5m=parts[1]["rv_residual_5m"],  # parts[1] is the RV frame
            spread_norm=spread_norm,
        )
    )

    return pd.concat(parts, axis=1)
