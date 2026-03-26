"""
Triangle residual computation for the EUR/USD/AUD triangle.

All series are at native 10-second bar resolution.

No-arbitrage identity:
    EUR/AUD_implied = EUR/USD ÷ AUD/USD

Raw residual (in log space):
    Δ(t) = ln(EURAUD) - ln(EURUSD) + ln(AUDUSD)
         ≡ 0 in a perfectly efficient market

Z-score (EWMA-normalised to handle slow structural drift):
    z(t) = (Δ(t) - μ_t) / σ_t

where μ_t and σ_t are exponentially-weighted mean and std. At 10s resolution,
halflife=360 bars corresponds to a 1-hour EWMA (360 × 10s = 3600s).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def align_pairs(
    eurusd: pd.DataFrame,
    audusd: pd.DataFrame,
    euraud: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join all three pairs on timestamp at 10s resolution.

    Returns a DataFrame with columns:
        eurusd, audusd, euraud  — close (mid) prices
        euraud_bid, euraud_ask  — bid/ask at bar close for the trading pair
                                  (present only if 'bid'/'ask' cols exist in euraud)
    """
    cols: dict = {
        "eurusd": eurusd["close"],
        "audusd": audusd["close"],
        "euraud": euraud["close"],
    }
    if "bid" in euraud.columns and "ask" in euraud.columns:
        cols["euraud_bid"] = euraud["bid"]
        cols["euraud_ask"] = euraud["ask"]

    df = pd.DataFrame(cols)
    df = df.dropna()
    return df


def compute_residual(aligned: pd.DataFrame) -> pd.Series:
    """Compute the log-space triangle residual at 10s resolution.

    Δ = ln(EURAUD) - ln(EURUSD) + ln(AUDUSD)

    Returns a Series named 'residual'.
    """
    residual = (
        np.log(aligned["euraud"])
        - np.log(aligned["eurusd"])
        + np.log(aligned["audusd"])
    )
    residual.name = "residual"
    return residual


def compute_zscore(
    residual: pd.Series,
    halflife: int = 360,
) -> pd.Series:
    """EWMA z-score of the 10s residual.

    Args:
        residual:  Raw residual series (Δ) at 10s resolution.
        halflife:  EWMA half-life in bars. Default 360 = 1 hour (360 × 10s).
                   Use longer values (e.g. 4320 = 12h) for macro-drift regimes.

    Returns:
        Series named 'zscore'.
    """
    ewma_mean = residual.ewm(halflife=halflife, min_periods=1).mean()
    ewma_std  = residual.ewm(halflife=halflife, min_periods=1).std()
    zscore = (residual - ewma_mean) / ewma_std.clip(lower=1e-10)
    zscore.name = "zscore"
    return zscore


def build_signal_frame(
    eurusd: pd.DataFrame,
    audusd: pd.DataFrame,
    euraud: pd.DataFrame,
    ewma_halflife: int = 360,
) -> pd.DataFrame:
    """Convenience wrapper: align → residual → z-score at 10s resolution.

    Args:
        eurusd/audusd/euraud: 10s OHLC DataFrames from load_pair().
        ewma_halflife: EWMA half-life in 10s bars. Default 360 = 1 hour.

    Returns:
        DataFrame with columns: eurusd, audusd, euraud, [euraud_bid, euraud_ask],
        residual, zscore. Bid/ask columns present only when loaded via load_pair().
    """
    aligned  = align_pairs(eurusd, audusd, euraud)
    residual = compute_residual(aligned)
    zscore   = compute_zscore(residual, halflife=ewma_halflife)

    out = aligned.copy()
    out["residual"] = residual
    out["zscore"]   = zscore
    return out
