"""
Visualisation module for the FX triangulation pipeline.

All functions:
  - Use dark_background matplotlib style
  - Annotate Liberation Day (2025-04-02) on any chart that spans that date
  - Save to outputs/plots/<name>.png at 150 DPI
  - Return the figure object

Usage:
    from triangulation.plots import plot_raw_pairs, plot_residual, ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for script execution
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from triangulation.analysis import ou_halflife, ou_halflife_by_period

plt.style.use("dark_background")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LD_DATE  = pd.Timestamp("2025-04-02", tz="UTC")   # Liberation Day
LD2_DATE = pd.Timestamp("2025-04-09", tz="UTC")   # AUD/USD 5-year low

SESSION_DEFS = {
    "Asian":      (0,   8),
    "European":   (8,  12),
    "London/NY":  (12, 16),
    "NY Afternoon": (16, 24),
}

# Muted colour palette for dark background
SESSION_COLORS = {
    "Asian":        "#4fc3f7",
    "European":     "#81c784",
    "London/NY":    "#ffb74d",
    "NY Afternoon": "#f48fb1",
}

HIGHLIGHT_COLOR = "#ffd54f"   # for |z| > 2 shading


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_ld_line(ax: plt.Axes, add_label: bool = True) -> None:
    """Add Liberation Day vertical line to an axes object."""
    xlim = ax.get_xlim()
    ld_num = matplotlib.dates.date2num(LD_DATE.to_pydatetime())
    if xlim[0] <= ld_num <= xlim[1]:
        ax.axvline(LD_DATE, color="red", linewidth=0.9, linestyle="--", alpha=0.75, zorder=5)
        if add_label:
            ymin, ymax = ax.get_ylim()
            ax.text(
                LD_DATE, ymax - (ymax - ymin) * 0.04,
                "Liberation Day", color="red", fontsize=7,
                rotation=90, va="top", ha="right", zorder=6,
            )


def _session_mask(index: pd.DatetimeIndex, session: str) -> np.ndarray:
    """Boolean mask for a session by UTC hour range."""
    lo, hi = SESSION_DEFS[session]
    return (index.hour >= lo) & (index.hour < hi)


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _rolling_lag1_autocorr(s: pd.Series, window: int) -> pd.Series:
    """Vectorised rolling lag-1 autocorrelation (fast, uses pandas C backend)."""
    s_lag = s.shift(1)
    cov = s.rolling(window).cov(s_lag)
    var = s.rolling(window).var()
    return cov / var.clip(lower=1e-12)


# ---------------------------------------------------------------------------
# 1. Raw pairs
# ---------------------------------------------------------------------------

def plot_raw_pairs(sig_frame: pd.DataFrame, path: Path) -> plt.Figure:
    """Three stacked panels: EURUSD / AUDUSD / EURAUD hourly mid price."""
    _ensure_dir(path)

    hourly = sig_frame[["eurusd", "audusd", "euraud"]].resample("1h").last()

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    pairs = [("eurusd", "EUR/USD"), ("audusd", "AUD/USD"), ("euraud", "EUR/AUD")]

    for ax, (col, label) in zip(axes, pairs):
        ax.plot(hourly.index, hourly[col], linewidth=0.6, color="#90caf9")
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.15)
        _add_ld_line(ax)

    axes[0].set_title("AUD-USD-EUR Triangle: Raw FX Pairs", fontsize=11)
    axes[-1].set_xlabel("Date (UTC)")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 2. Residual
# ---------------------------------------------------------------------------

def plot_residual(sig_frame: pd.DataFrame, path: Path) -> plt.Figure:
    """Triangle residual Δ(t) with EWMA overlay and |z|>2 highlight bands."""
    _ensure_dir(path)

    h_res = sig_frame["residual"].resample("1h").last().dropna()
    h_z   = sig_frame["zscore"].resample("1h").last().dropna()
    ewma  = h_res.ewm(span=12, min_periods=1).mean()   # ≈ 12-hour at hourly resolution
    idx   = h_res.index

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(idx, h_res, linewidth=0.4, color="#546e7a", alpha=0.6, label="Residual (hourly)")
    ax.plot(idx, ewma,  linewidth=0.9, color="#90caf9", label="EWMA (12h span)")
    ax.axhline(0, color="white", linewidth=0.5, linestyle=":")

    # Shade |z| > 2.0
    signal_bars = h_z.abs() >= 2.0
    ax.fill_between(idx, h_res.min(), h_res.max(),
                    where=signal_bars, color=HIGHLIGHT_COLOR, alpha=0.12, label="|z| ≥ 2.0")

    ax.set_ylabel("Residual Δ(t)")
    ax.set_title("Triangle Residual Δ(t) = ln(EURAUD) − ln(EURUSD) + ln(AUDUSD)", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.15)
    _add_ld_line(ax)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 3. Liberation Day zoom
# ---------------------------------------------------------------------------

def plot_liberation_day_zoom(sig_frame: pd.DataFrame, path: Path) -> plt.Figure:
    """Four-panel chart zoomed to 2025-03-17 → 2025-04-30 at hourly resolution."""
    _ensure_dir(path)

    zoom_start = pd.Timestamp("2025-03-17", tz="UTC")
    zoom_end   = pd.Timestamp("2025-04-30", tz="UTC")

    hourly = sig_frame.resample("1h").last()
    zoom   = hourly.loc[zoom_start:zoom_end]

    if len(zoom) == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Liberation Day window not in dataset",
                ha="center", va="center", transform=ax.transAxes)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        return fig

    cols   = [("eurusd", "EUR/USD"), ("audusd", "AUD/USD"),
              ("euraud", "EUR/AUD"), ("residual", "Residual Δ(t)")]
    colors = ["#90caf9", "#80cbc4", "#ce93d8", "#ffb74d"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for ax, (col, label), color in zip(axes, cols, colors):
        ax.plot(zoom.index, zoom[col], linewidth=0.7, color=color)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(alpha=0.15)
        # Liberation Day
        ax.axvline(LD_DATE, color="red", linewidth=1.0, linestyle="--", alpha=0.8, zorder=5)
        # AUD/USD 5-year low (2025-04-09)
        ax.axvline(LD2_DATE, color="orange", linewidth=0.8, linestyle=":", alpha=0.7, zorder=5)

    axes[0].set_title("Liberation Day Zoom: 2025-03-17 → 2025-04-30", fontsize=11)

    ymin0, ymax0 = axes[0].get_ylim()
    axes[0].text(LD_DATE, ymax0, "Liberation Day", color="red",
                 fontsize=7, rotation=90, va="top", ha="right")
    axes[0].text(LD2_DATE, ymax0, "AUD/USD 5yr low (0.5914)", color="orange",
                 fontsize=7, rotation=90, va="top", ha="right")

    axes[-1].set_xlabel("Date (UTC)")
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 4. OU diagnostics
# ---------------------------------------------------------------------------

def plot_ou_diagnostics(sig_frame: pd.DataFrame, path: Path) -> plt.Figure:
    """Rolling autocorrelation and monthly OU half-life over full history."""
    _ensure_dir(path)

    # Hourly downsample for efficiency
    h_res = sig_frame["residual"].resample("1h").last().dropna()
    h_smooth = h_res.ewm(span=3, min_periods=1).mean()   # 3-hour smooth at hourly res

    # Full-sample reference values
    full_autocorr = float(h_res.autocorr(lag=1))
    full_ou = ou_halflife(h_smooth.ewm(span=6, min_periods=1).mean())
    full_hl_min = full_ou["halflife_minutes"] if np.isfinite(full_ou["halflife_minutes"]) else np.nan

    # Rolling lag-1 autocorrelation (30-day window = 720 hourly bars)
    rolling_acorr = _rolling_lag1_autocorr(h_res, window=720)

    # Monthly OU half-life via ou_halflife_by_period
    ou_monthly = ou_halflife_by_period(h_smooth, period="1ME")
    ou_monthly = ou_monthly[np.isfinite(ou_monthly["halflife_minutes"])]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=False)

    # --- Top: rolling autocorrelation ---
    ax1.plot(rolling_acorr.index, rolling_acorr, linewidth=0.7, color="#90caf9",
             label="Lag-1 autocorr (30d window)")
    ax1.axhline(full_autocorr, color="#ffb74d", linewidth=0.8, linestyle="--",
                label=f"Full-sample: {full_autocorr:.3f}")
    ax1.set_ylabel("Lag-1 Autocorrelation")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.15)
    ax1.set_title("OU Mean-Reversion Diagnostics", fontsize=11)
    _add_ld_line(ax1)

    # --- Bottom: monthly OU half-life ---
    if len(ou_monthly) > 0:
        # Plot as a step function using period-start index
        hl_clipped = ou_monthly["halflife_minutes"].clip(upper=60)
        ax2.step(ou_monthly.index, hl_clipped, where="post",
                 linewidth=1.2, color="#81c784", label="OU half-life (monthly)")
        ax2.scatter(ou_monthly.index, hl_clipped, s=20, color="#81c784", zorder=5)
    if np.isfinite(full_hl_min):
        ax2.axhline(min(full_hl_min, 60), color="#ffb74d", linewidth=0.8, linestyle="--",
                    label=f"Full-sample: {full_hl_min:.1f} min")
    ax2.set_ylim(bottom=0, top=65)
    ax2.set_ylabel("OU Half-Life (minutes, capped at 60)")
    ax2.set_xlabel("Date (UTC)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.15)
    _add_ld_line(ax2)

    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 5. Signal distribution
# ---------------------------------------------------------------------------

def plot_signal_distribution(sig_frame: pd.DataFrame, path: Path) -> plt.Figure:
    """Z-score histogram by session + daily signal frequency time series."""
    _ensure_dir(path)

    z = sig_frame["zscore"].dropna()
    idx = z.index

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7))

    # --- Top: histogram by session ---
    bins = np.linspace(-6, 6, 80)
    for session, color in SESSION_COLORS.items():
        mask = _session_mask(idx, session)
        ax1.hist(z[mask].values, bins=bins, alpha=0.5, color=color,
                 label=session, density=True)
    ax1.axvline(2.0,  color="white", linewidth=0.8, linestyle="--", alpha=0.7)
    ax1.axvline(-2.0, color="white", linewidth=0.8, linestyle="--", alpha=0.7)
    ax1.set_xlabel("Z-Score")
    ax1.set_ylabel("Density")
    ax1.set_title("Z-Score Signal Distribution by Session", fontsize=11)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.15)

    # --- Bottom: daily rolling signal frequency ---
    signal_bars = (z.abs() >= 2.0).astype(float)
    daily_signals = signal_bars.resample("1D").sum()
    # bars per hour per day (divide by 24*360 bars/day at 10s, times 360 = bars/hour)
    bars_per_hour = 360
    daily_freq = (daily_signals / bars_per_hour)  # signals per hour of the day
    daily_freq_7d = daily_freq.rolling(7, min_periods=1).mean()

    ax2.fill_between(daily_freq.index, 0, daily_freq.values,
                     color="#546e7a", alpha=0.5, label="Daily signal count (|z|≥2)")
    ax2.plot(daily_freq_7d.index, daily_freq_7d.values,
             color="#90caf9", linewidth=1.0, label="7-day rolling mean")
    ax2.set_ylabel("Signal bars per hour")
    ax2.set_xlabel("Date (UTC)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.15)
    _add_ld_line(ax2)

    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 6. Model predictions
# ---------------------------------------------------------------------------

def plot_model_predictions(
    df_test: pd.DataFrame,
    y_pred: np.ndarray,
    path: Path,
) -> plt.Figure:
    """Scatter of predicted vs actual z_future_60 + time series of |predicted_move|."""
    _ensure_dir(path)

    y_actual = df_test["z_future_60"].values
    z_current = df_test["zscore"].values
    predicted_moves = np.abs(z_current - y_pred)
    actual_moves    = np.abs(z_current - y_actual)

    # Subsample scatter for performance
    n_scatter = min(50_000, len(y_pred))
    rng = np.random.default_rng(42)
    idx_sample = rng.choice(len(y_pred), size=n_scatter, replace=False)
    idx_sample.sort()

    # Session labels for colour
    session_ids = np.zeros(len(df_test), dtype=int)
    for i, session in enumerate(SESSION_DEFS):
        mask = _session_mask(df_test.index, session)
        session_ids[mask] = i
    session_colors_arr = [list(SESSION_COLORS.values())[s] for s in session_ids[idx_sample]]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))

    # --- Top: scatter predicted vs actual ---
    ax1.scatter(
        y_pred[idx_sample], y_actual[idx_sample],
        c=session_colors_arr, s=1, alpha=0.3, rasterized=True,
    )
    lim = max(abs(y_pred).max(), abs(y_actual).max()) * 1.05
    ax1.plot([-lim, lim], [-lim, lim], color="white", linewidth=0.7,
             linestyle="--", alpha=0.5, label="Perfect prediction")
    ax1.set_xlim(-lim, lim)
    ax1.set_ylim(-lim, lim)
    ax1.set_xlabel("Predicted z_future_60")
    ax1.set_ylabel("Actual z_future_60")
    ax1.set_title("Regression Model: Predicted vs Actual Future Z-Score", fontsize=11)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.15)

    # Session legend patches
    patches = [mpatches.Patch(color=c, label=s)
               for s, c in SESSION_COLORS.items()]
    ax1.legend(handles=patches + [
        matplotlib.lines.Line2D([0], [0], color="white", linewidth=0.7,
                                linestyle="--", label="Perfect prediction")
    ], fontsize=7)

    # --- Bottom: daily mean |predicted_move| vs actual ---
    pm_series = pd.Series(predicted_moves, index=df_test.index)
    am_series = pd.Series(actual_moves,    index=df_test.index)
    daily_pm = pm_series.resample("1D").mean()
    daily_am = am_series.resample("1D").mean()

    ax2.plot(daily_pm.index, daily_pm.values,
             color="#90caf9", linewidth=0.8, label="Mean |predicted move|")
    ax2.plot(daily_am.index, daily_am.values,
             color="#ffb74d", linewidth=0.8, alpha=0.7, label="Mean |actual move|")
    ax2.set_ylabel("|Move| (z-score units)")
    ax2.set_xlabel("Date (UTC)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.15)
    _add_ld_line(ax2)

    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 7. Equity curve
# ---------------------------------------------------------------------------

def plot_equity_curve(
    equity_series: pd.Series,
    trade_log: pd.DataFrame,
    split_dates: dict,
    stats_dict: dict,
    path: Path,
) -> plt.Figure:
    """Cumulative P&L + drawdown with train/val/test shading and stats annotation."""
    _ensure_dir(path)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Background shading for train / val / test periods
    full_start = equity_series.index[0]
    full_end   = equity_series.index[-1]
    train_end  = pd.Timestamp(split_dates.get("train_end", "2024-12-31"), tz="UTC")
    val_end    = pd.Timestamp(split_dates.get("val_end",   "2025-06-30"), tz="UTC")

    for ax in (ax1, ax2):
        ax.axvspan(full_start, train_end, alpha=0.07, color="#4fc3f7", label="Train")
        ax.axvspan(train_end,  val_end,   alpha=0.07, color="#81c784", label="Val")
        ax.axvspan(val_end,    full_end,  alpha=0.07, color="#ffb74d", label="Test")

    # Cumulative P&L
    ax1.plot(equity_series.index, equity_series.values,
             linewidth=0.9, color="#90caf9", label="Cumulative P&L")
    ax1.axhline(0, color="white", linewidth=0.4, linestyle=":")
    ax1.set_ylabel("Cumulative P&L (pips)")
    ax1.set_title("Simulated Equity Curve — Test Set (2025-07 → 2026-03)", fontsize=11)
    ax1.grid(alpha=0.15)
    _add_ld_line(ax1)

    # Stats annotation box
    sr    = stats_dict.get("sharpe",         float("nan"))
    mdd   = stats_dict.get("max_drawdown",   float("nan"))
    wr    = stats_dict.get("win_rate",        float("nan"))
    tpw   = stats_dict.get("trades_per_week", float("nan"))
    exits = stats_dict.get("exit_breakdown",  {})
    exit_str = "  ".join(f"{k} {v:.0%}" for k, v in exits.items())
    ann_text = (
        f"Sharpe: {sr:.2f}\n"
        f"Max DD: {mdd:.1f} pips\n"
        f"Win rate: {wr:.1%}\n"
        f"Trades/week: {tpw:.1f}\n"
        f"Exits: {exit_str}"
    )
    ax1.text(
        0.02, 0.97, ann_text,
        transform=ax1.transAxes, fontsize=8,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e", alpha=0.85, edgecolor="#555"),
    )

    # Legend patches for regions
    region_patches = [
        mpatches.Patch(color="#4fc3f7", alpha=0.4, label="Train"),
        mpatches.Patch(color="#81c784", alpha=0.4, label="Val"),
        mpatches.Patch(color="#ffb74d", alpha=0.4, label="Test"),
    ]
    ax1.legend(handles=region_patches, fontsize=7, loc="upper right")

    # Rolling drawdown
    running_max = equity_series.cummax()
    drawdown = equity_series - running_max
    ax2.fill_between(drawdown.index, drawdown.values, 0,
                     color="#ef5350", alpha=0.6, label="Drawdown")
    ax2.set_ylabel("Drawdown (pips)")
    ax2.set_xlabel("Date (UTC)")
    ax2.grid(alpha=0.15)
    _add_ld_line(ax2)

    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig
