"""
Week 1 statistical analysis:
  - Stationarity tests (ADF, KPSS)
  - Lag-1 autocorrelation
  - Ornstein-Uhlenbeck half-life estimation
  - Signal frequency and gap-size distribution by z-score threshold

OU process: dX = κ(θ - X)dt + σ dW
Discrete: X(t) = a + b·X(t-1) + ε
  half-life = -ln(2) / ln(b)  [in bars; multiply by bar_seconds for seconds]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss


# ---------------------------------------------------------------------------
# Stationarity
# ---------------------------------------------------------------------------

def adf_test(series: pd.Series, maxlag: int | None = None) -> dict:
    """Augmented Dickey-Fuller test. H0: unit root (non-stationary).
    Returns dict with stat, pvalue, lags, nobs, critical_values, is_stationary (5%).
    """
    result = adfuller(series.dropna(), maxlag=maxlag, autolag="AIC")
    return {
        "stat":            result[0],
        "pvalue":          result[1],
        "lags":            result[2],
        "nobs":            result[3],
        "critical_values": result[4],
        "is_stationary":   result[1] < 0.05,
    }


def kpss_test(series: pd.Series) -> dict:
    """KPSS test. H0: stationary (trend-stationary around constant).
    Returns dict with stat, pvalue, lags, critical_values, is_stationary (5%).
    """
    result = kpss(series.dropna(), regression="c", nlags="auto")
    return {
        "stat":            result[0],
        "pvalue":          result[1],
        "lags":            result[2],
        "critical_values": result[3],
        "is_stationary":   result[1] > 0.05,
    }


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------

def autocorr_at_lags(series: pd.Series, lags: list[int]) -> dict[int, float]:
    """Pearson autocorrelation at specified lags."""
    s = series.dropna()
    return {lag: float(s.autocorr(lag=lag)) for lag in lags}


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck half-life
# ---------------------------------------------------------------------------

def ou_halflife(series: pd.Series) -> dict:
    """Estimate OU half-life via OLS on the discrete-time AR(1) representation.

    Regresses ΔX_t = a + b·X_{t-1} + ε, where b = e^{-κΔt} - 1.
    Half-life = -ln(2) / ln(1 + b) bars.

    Input note: pass the 1-minute EWMA of the raw 10s residual
    (`residual.ewm(span=6).mean()`) rather than the raw 10s series.
    Running on raw 10s data produces noise-dominated estimates because the
    tick-level microstructure noise overwhelms the OU signal.

    Returns dict with halflife_bars, halflife_seconds (1 bar = 10s),
    halflife_minutes, kappa, theta (long-run mean), r_squared.
    """
    s = series.dropna().values
    x_lag = s[:-1]
    dx    = np.diff(s)

    # OLS: dx = a + b * x_lag
    slope, intercept, r, p, se = stats.linregress(x_lag, dx)

    b = slope  # = e^{-κΔt} - 1, so e^{-κΔt} = 1 + b
    if b >= 0 or (1 + b) <= 0:
        # No mean-reversion detected
        return {
            "halflife_bars":    np.inf,
            "halflife_seconds": np.inf,
            "kappa":            np.nan,
            "theta":            np.nan,
            "r_squared":        r**2,
        }

    halflife_bars = -np.log(2) / np.log(1 + b)
    kappa = -np.log(1 + b)      # mean-reversion speed per bar
    theta = -intercept / slope  # long-run mean

    return {
        "halflife_bars":    float(halflife_bars),
        "halflife_seconds": float(halflife_bars * 10),
        "halflife_minutes": float(halflife_bars * 10 / 60),
        "kappa":            float(kappa),
        "theta":            float(theta),
        "r_squared":        float(r**2),
    }


def ou_halflife_by_period(
    residual: pd.Series,
    period: str = "6ME",
) -> pd.DataFrame:
    """Compute OU half-life for rolling 6-month (or custom) periods.

    Args:
        residual: Smoothed residual series with DatetimeIndex. Should be the
                  1-min EWMA of the raw 10s residual (`residual.ewm(span=6).mean()`)
                  — see `ou_halflife` docstring for why.
        period:   Pandas period alias (e.g. '6ME' for 6-month end).

    Returns:
        DataFrame indexed by period start date.
    """
    rows = []
    for label, group in residual.groupby(pd.Grouper(freq=period)):
        if len(group) < 500:
            continue
        result = ou_halflife(group)
        result["period_start"] = label
        result["n_obs"] = len(group)
        rows.append(result)
    return pd.DataFrame(rows).set_index("period_start")


# ---------------------------------------------------------------------------
# Signal frequency and gap-size analysis
# ---------------------------------------------------------------------------

def signal_stats(
    zscore: pd.Series,
    thresholds: list[float] | None = None,
    bar_seconds: int = 10,
) -> pd.DataFrame:
    """Compute signal frequency and gap-size statistics for a range of z-score
    entry thresholds.

    A 'signal' is a bar where |z| first crosses the threshold (edge detection,
    not level). Gap size = residual × some scaling — we report |z| at entry.

    Args:
        zscore:      Z-score series with DatetimeIndex.
        thresholds:  List of |z| thresholds to evaluate.
        bar_seconds: Bar duration in seconds (for annualised frequency).

    Returns:
        DataFrame with columns: threshold, n_signals, signals_per_week,
        mean_abs_z_at_entry, median_abs_z_at_entry, pct_above_1std,
        pct_above_2std.
    """
    if thresholds is None:
        thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    total_bars = len(zscore.dropna())
    bars_per_week = (7 * 24 * 3600) / bar_seconds

    rows = []
    for thr in thresholds:
        above = zscore.abs() >= thr
        # Edge detection: first bar above threshold after being below
        crossings = above & (~above.shift(1).fillna(value=False).astype(bool))
        n = int(crossings.sum())
        rows.append({
            "threshold":           thr,
            "n_signals":           n,
            "signals_per_week":    round(n / (total_bars / bars_per_week), 2),
            "mean_abs_z":          round(float(zscore[crossings].abs().mean()), 3) if n > 0 else np.nan,
            "median_abs_z":        round(float(zscore[crossings].abs().median()), 3) if n > 0 else np.nan,
        })

    return pd.DataFrame(rows)


def residual_summary(residual: pd.Series) -> dict:
    """Basic descriptive stats for the residual."""
    s = residual.dropna()
    return {
        "n":         len(s),
        "mean":      float(s.mean()),
        "std":       float(s.std()),
        "min":       float(s.min()),
        "max":       float(s.max()),
        "skew":      float(s.skew()),
        "kurt":      float(s.kurtosis()),
        "autocorr1": float(s.autocorr(lag=1)),
    }
