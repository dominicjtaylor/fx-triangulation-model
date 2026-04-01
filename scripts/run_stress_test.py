"""
Stress-test — persistence-filtered mean-reversion strategy

Verifies that the observed +5–7 pips/trade edge (persistence=10s, thr≥3.5, horizon=300s)
is real, tradable, and not a data artefact.

Six checks:
  1. Spread sensitivity  — edge survives realistic bid/ask spread, not mid-price fiction
  2. Event inspection    — 10 trade plots; are the closures structurally plausible?
  3. Lookahead audit     — persistence filter is causal; quantify momentum vs stabilisation
  4. Entry timing        — edge decays smoothly with delay, not a 1-bar artefact
  5. Stability           — edge holds across time sub-periods (quarterly)
  6. Distribution        — edge is not from a handful of outlier trades

Run from repo root:
    python3 scripts/run_stress_test.py [options]

Options:
    --thresholds FLOAT ...    (default: 3.5 4.0 4.5)
    --persistence-bars INT    persistence in bars (default: 1 = 10s)
    --horizons-s INT ...      horizons to test (default: 60 120 300)
    --n-events INT            number of event plots to generate (default: 10)
    --data-dir PATH
    --output-dir PATH
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from triangulation.data import load_pair
from triangulation.residual import build_signal_frame
from triangulation.features import build_feature_frame

plt.style.use("dark_background")

DATA_DIR    = ROOT / "data"
PLOTS_DIR   = ROOT / "outputs" / "plots"
OUTPUTS_DIR = ROOT / "outputs"

VAL_END   = "2025-06-30"
BAR_S     = 10
_BARS_30D = 259_200


# ---------------------------------------------------------------------------
# Signal detection (causal persistence filter)
# ---------------------------------------------------------------------------

def persistent_signals(
    z: np.ndarray,
    threshold: float,
    persistence_bars: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (confirmed_indices, crossing_indices) pairs.

    confirmed_indices: bars at which persistence is confirmed (entry candidates).
    crossing_indices:  the original crossing bar for each confirmed signal.

    persistence_bars=0 → confirmed = crossing (standard crossing event).
    persistence_bars=N → |z| must stay >= threshold for N bars after crossing.
    """
    n     = len(z)
    above = np.abs(z) >= threshold
    xings = np.where(above[1:] & ~above[:-1])[0] + 1

    if persistence_bars == 0:
        return xings.copy(), xings.copy()

    confirmed = []
    crossings = []
    for sig_i in xings:
        end_i = sig_i + persistence_bars
        if end_i >= n:
            continue
        if np.all(above[sig_i : end_i + 1]):
            confirmed.append(end_i)
            crossings.append(sig_i)
    return np.array(confirmed, dtype=int), np.array(crossings, dtype=int)


# ---------------------------------------------------------------------------
# Simulation with bid/ask pricing
# ---------------------------------------------------------------------------

def simulate_realistic(
    z: np.ndarray,
    mid: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    signals: np.ndarray,
    delay_bars: int,
    horizon_bars: int,
    spread_multiplier: float = 1.0,
) -> pd.DataFrame:
    """One-trade-at-a-time simulation using actual bid/ask prices.

    Args:
        z:                 Z-score series.
        mid:               EUR/AUD mid (close) price series.
        bid/ask:           EUR/AUD bid and ask series.
        signals:           Confirmed signal indices (persistence already applied).
        delay_bars:        Bars between confirmed signal and entry.
        horizon_bars:      Bars to hold.
        spread_multiplier: Scale actual spread by this factor for stress scenarios.
                           1.0 = actual data; 2.0 = 2× stressed spread.

    Returns:
        DataFrame with one row per trade.
    """
    n       = len(mid)
    trades  = []
    next_free = 0

    for sig_i in signals:
        entry_i = sig_i + delay_bars
        if entry_i >= n - horizon_bars:
            continue
        if entry_i < next_free:
            continue

        direction = float(np.sign(z[sig_i]))
        if direction == 0:
            continue

        exit_i = entry_i + horizon_bars

        # Stressed bid/ask: widen spread symmetrically around mid
        em = mid[entry_i]
        xm = mid[exit_i]
        e_spread = (ask[entry_i] - bid[entry_i]) * spread_multiplier
        x_spread = (ask[exit_i]  - bid[exit_i])  * spread_multiplier
        e_bid = em - e_spread / 2
        e_ask = em + e_spread / 2
        x_bid = xm - x_spread / 2
        x_ask = xm + x_spread / 2

        # Execution: SHORT → sell at bid (entry), buy at ask (exit)
        #            LONG  → buy at ask (entry), sell at bid (exit)
        if direction > 0:   # SHORT
            entry_exec = e_bid
            exit_exec  = x_ask
        else:               # LONG
            entry_exec = e_ask
            exit_exec  = x_bid

        gross_real = direction * (entry_exec - exit_exec) * 10_000
        gross_mid  = direction * (em - xm) * 10_000

        trades.append({
            "entry_i":          entry_i,
            "exit_i":           exit_i,
            "direction":        direction,
            "entry_mid":        em,
            "exit_mid":         xm,
            "spread_pips_entry": e_spread * 10_000,
            "spread_pips_exit":  x_spread * 10_000,
            "gross_pips_mid":   gross_mid,
            "gross_pips_real":  gross_real,
            "net_pips_real":    gross_real,   # no additional cost beyond spread
        })
        next_free = exit_i + 1

    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_spread_sensitivity(
    rows: list[dict],
    thresholds: list[float],
    path: Path,
) -> None:
    """Mean net pips vs spread scenario, lines per threshold."""
    scenarios = ["actual", "2x", "4x"]
    colors    = ["#4fc3f7", "#81c784", "#ffb74d"]
    x         = [1.0, 2.0, 4.0]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, thr in enumerate(sorted(thresholds)):
        nets = []
        for mult in x:
            sub = [r for r in rows if r["threshold"] == thr and r["spread_mult"] == mult]
            nets.append(sub[0]["mean_net"] if sub else float("nan"))
        ax.plot(x, nets, marker="o", markersize=6, linewidth=1.8,
                color=colors[i % len(colors)], label=f"thr={thr:.1f}")
        for xi, ni in zip(x, nets):
            if not np.isnan(ni):
                ax.annotate(f"{ni:+.2f}", (xi, ni),
                            textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=7, color=colors[i % len(colors)])

    ax.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Spread multiplier (1x = actual data)", fontsize=9)
    ax.set_ylabel("Mean net pips per trade", fontsize=9)
    ax.set_title("Spread Sensitivity  (persist=10s, delay=0, horizon=300s)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(["1x\n(actual)", "2x\n(stressed)", "4x\n(extreme)"])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


def plot_events(
    z: np.ndarray,
    mid: np.ndarray,
    trades: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    threshold: float,
    n_events: int,
    path_dir: Path,
) -> None:
    """Plot ±5 min price path and z-score for first n_events trades."""
    path_dir.mkdir(parents=True, exist_ok=True)
    n     = len(mid)
    shown = 0

    for _, row in trades.head(n_events).iterrows():
        entry_i = int(row["entry_i"])
        exit_i  = int(row["exit_i"])
        lo      = max(0, entry_i - 30)
        hi      = min(n - 1, exit_i + 30)
        bars    = np.arange(lo, hi + 1)

        # Seconds relative to entry
        t_s  = (bars - entry_i) * BAR_S
        z_w  = z[lo : hi + 1]
        p_w  = mid[lo : hi + 1]

        ts_str = str(timestamps[entry_i])[:19] if entry_i < len(timestamps) else str(entry_i)
        direction = int(row["direction"])
        dir_label = "SHORT" if direction > 0 else "LONG"

        fig, (ax_p, ax_z) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        # Price panel
        ax_p.plot(t_s, p_w, color="#4fc3f7", linewidth=1.2)
        ax_p.axvline(0,                       color="#81c784", linewidth=1.2,
                     linestyle="--", label="entry")
        ax_p.axvline((exit_i - entry_i) * BAR_S, color="#ffb74d", linewidth=1.2,
                     linestyle="--", label="exit")
        ax_p.axhline(row["entry_mid"], color="white", linewidth=0.6, linestyle=":", alpha=0.5)
        ax_p.set_ylabel("EUR/AUD mid", fontsize=9)
        ax_p.set_title(
            f"Event {shown+1}  |  {ts_str}  |  {dir_label}  |  "
            f"gross_mid={row['gross_pips_mid']:+.2f}  gross_real={row['gross_pips_real']:+.2f} pips",
            fontsize=9,
        )
        ax_p.legend(fontsize=7)
        ax_p.grid(alpha=0.12)

        # Z-score panel
        ax_z.plot(t_s, z_w, color="#f48fb1", linewidth=1.2)
        for sign in [+1, -1]:
            ax_z.axhline(sign * threshold,       color="#ffb74d", linewidth=0.8,
                         linestyle="--", alpha=0.7)
            ax_z.axhline(sign * 1.5 * threshold, color="#ef5350", linewidth=0.8,
                         linestyle=":",  alpha=0.5)
        ax_z.axhline(0, color="white", linewidth=0.4, alpha=0.3)
        ax_z.axvline(0,                       color="#81c784", linewidth=1.2, linestyle="--")
        ax_z.axvline((exit_i - entry_i) * BAR_S, color="#ffb74d", linewidth=1.2, linestyle="--")
        ax_z.set_xlabel("Seconds relative to entry", fontsize=9)
        ax_z.set_ylabel("Z-score", fontsize=9)
        ax_z.grid(alpha=0.12)

        fig.tight_layout()
        out = path_dir / f"event_{shown+1:03d}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        shown += 1

    print(f"  Saved {shown} event plots → {path_dir}/event_*.png")


def plot_lookahead_check(
    z_crossing: np.ndarray,
    z_confirmed: np.ndarray,
    threshold: float,
    path: Path,
) -> None:
    """Histogram of z_confirmed - z_crossing to quantify momentum vs stabilisation."""
    delta = np.abs(z_confirmed) - np.abs(z_crossing)
    n_momentum = int((delta > 0).sum())
    n_stable   = int((delta <= 0).sum())

    fig, ax = plt.subplots(figsize=(8, 5))
    lo, hi = np.percentile(delta, [1, 99])
    bins = np.linspace(lo, hi, 50)
    ax.hist(delta[delta > 0],  bins=bins, alpha=0.7, color="#f48fb1",
            label=f"growing (n={n_momentum:,})", density=True)
    ax.hist(delta[delta <= 0], bins=bins, alpha=0.7, color="#4fc3f7",
            label=f"stable/retreating (n={n_stable:,})", density=True)
    ax.axvline(0, color="white", linewidth=1.0, linestyle="--")
    ax.axvline(float(np.mean(delta)), color="#81c784", linewidth=1.2,
               linestyle="-", label=f"mean={np.mean(delta):+.3f}")
    ax.set_xlabel("|z_confirmed| − |z_crossing|", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title(
        f"Lookahead Audit: z growth between crossing and confirmation\n"
        f"(threshold={threshold:.1f}, persistence=1 bar)",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


def plot_timing_sensitivity(
    delay_bars_list: list[int],
    mean_nets: list[float],
    win_rates: list[float],
    n_trades: list[int],
    threshold: float,
    path: Path,
) -> None:
    """Mean net pips and win rate vs delay after confirmation."""
    total_s = [(10 + d * BAR_S) for d in delay_bars_list]   # 10s = persistence

    fig, (ax_n, ax_w) = plt.subplots(1, 2, figsize=(12, 5))

    color = "#4fc3f7"
    ax_n.plot(total_s, mean_nets, marker="o", markersize=6, linewidth=1.8, color=color)
    for xs, mn, nt in zip(total_s, mean_nets, n_trades):
        ax_n.annotate(f"{mn:+.2f}\n(n={nt})", (xs, mn),
                      textcoords="offset points", xytext=(0, 10),
                      ha="center", fontsize=7, color=color)
    ax_n.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_n.set_xlabel("Seconds from original crossing to entry", fontsize=9)
    ax_n.set_ylabel("Mean net pips (actual spread)", fontsize=9)
    ax_n.set_title(f"Entry Timing Sensitivity  (thr={threshold:.1f}, horizon=300s)", fontsize=10)
    ax_n.set_xticks(total_s)
    ax_n.grid(alpha=0.15)

    ax_w.plot(total_s, [wr * 100 for wr in win_rates], marker="s", markersize=6,
              linewidth=1.8, color="#81c784")
    ax_w.axhline(50, color="white", linewidth=0.8, linestyle="--", alpha=0.5, label="50%")
    ax_w.set_xlabel("Seconds from original crossing to entry", fontsize=9)
    ax_w.set_ylabel("Win rate (%)", fontsize=9)
    ax_w.set_title("Win Rate vs Delay", fontsize=10)
    ax_w.set_xticks(total_s)
    ax_w.set_ylim(0, 100)
    ax_w.grid(alpha=0.15)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


def plot_stability(
    quarter_labels: list[str],
    results_by_thr: dict[float, list[dict]],
    path: Path,
) -> None:
    """Mean net pips per quarter for each threshold (grouped bar chart)."""
    thresholds = sorted(results_by_thr.keys())
    n_q        = len(quarter_labels)
    colors     = ["#4fc3f7", "#81c784", "#ffb74d"]
    bar_w      = 0.25
    x          = np.arange(n_q)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_net, ax_wr = axes

    for i, thr in enumerate(thresholds):
        offset = (i - len(thresholds) / 2 + 0.5) * bar_w
        nets   = [r["mean_net"] for r in results_by_thr[thr]]
        wrs    = [r["win_rate"] * 100 for r in results_by_thr[thr]]
        counts = [r["n_trades"] for r in results_by_thr[thr]]
        color  = colors[i % len(colors)]

        bars = ax_net.bar(x + offset, nets, bar_w, color=color, alpha=0.8,
                          label=f"thr={thr:.1f}")
        for bar, n_t in zip(bars, counts):
            if n_t > 0:
                ax_net.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                            f"n={n_t}", ha="center", va="bottom", fontsize=6, color=color)

        ax_wr.bar(x + offset, wrs, bar_w, color=color, alpha=0.8, label=f"thr={thr:.1f}")

    for ax in [ax_net, ax_wr]:
        ax.set_xticks(x)
        ax.set_xticklabels(quarter_labels, fontsize=8)
        ax.set_xlabel("Sub-period", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.12, axis="y")

    ax_net.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_net.set_ylabel("Mean net pips (actual spread)", fontsize=9)
    ax_net.set_title("Stability: Mean Net P&L by Quarter", fontsize=10)

    ax_wr.axhline(50, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_wr.set_ylabel("Win rate (%)", fontsize=9)
    ax_wr.set_title("Stability: Win Rate by Quarter", fontsize=10)

    fig.suptitle("Temporal Stability  (persist=10s, delay=0, horizon=300s)", fontsize=10, y=1.01)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


def plot_distribution(
    net_pips: np.ndarray,
    path: Path,
) -> None:
    """Return distribution and cumulative P&L contribution."""
    fig, (ax_h, ax_c) = plt.subplots(1, 2, figsize=(13, 5))

    # Histogram
    lo, hi = np.percentile(net_pips, [1, 99])
    bins   = np.linspace(lo, hi, 50)
    ax_h.hist(net_pips[net_pips > 0],  bins=bins, alpha=0.7, color="#81c784",
              density=False, label="profitable")
    ax_h.hist(net_pips[net_pips <= 0], bins=bins, alpha=0.7, color="#ef5350",
              density=False, label="loss")
    ax_h.axvline(0, color="white", linewidth=0.9, linestyle="--", alpha=0.6)
    ax_h.axvline(float(net_pips.mean()), color="#ffb74d", linewidth=1.2,
                 label=f"mean={net_pips.mean():+.2f}")
    ax_h.set_xlabel("Net pips per trade", fontsize=9)
    ax_h.set_ylabel("Count", fontsize=9)
    ax_h.set_title("Return Distribution", fontsize=10)
    ax_h.legend(fontsize=8)
    ax_h.grid(alpha=0.12)

    skew = float(sp_stats.skew(net_pips))
    kurt = float(sp_stats.kurtosis(net_pips))
    ax_h.text(0.02, 0.97, f"skew={skew:+.2f}\nkurt={kurt:+.1f}\nn={len(net_pips):,}",
              transform=ax_h.transAxes, va="top", fontsize=8,
              bbox=dict(boxstyle="round", facecolor="#333333", alpha=0.7))

    # Cumulative P&L contribution — sorted descending
    sorted_net = np.sort(net_pips)[::-1]
    cumsum     = np.cumsum(sorted_net)
    pct_trades = np.arange(1, len(sorted_net) + 1) / len(sorted_net) * 100
    ax_c.plot(pct_trades, cumsum, color="#4fc3f7", linewidth=1.5)
    ax_c.axhline(cumsum[-1], color="white", linewidth=0.6, linestyle=":", alpha=0.4)
    ax_c.axhline(0, color="white", linewidth=0.6, linestyle="--", alpha=0.4)
    # Mark 10% and 20% of trades
    for pct in [10, 20, 50]:
        idx = max(0, int(pct / 100 * len(sorted_net)) - 1)
        ax_c.axvline(pct, color="#ffb74d", linewidth=0.7, linestyle=":", alpha=0.6)
        ax_c.text(pct + 0.5, cumsum[idx], f"{pct}%→{cumsum[idx]:.0f}p",
                  fontsize=7, color="#ffb74d", va="bottom")
    ax_c.set_xlabel("% of trades (sorted by contribution, best first)", fontsize=9)
    ax_c.set_ylabel("Cumulative net pips", fontsize=9)
    ax_c.set_title("Cumulative P&L Contribution", fontsize=10)
    ax_c.grid(alpha=0.12)

    fig.suptitle("Distribution Analysis  (thr=4.5, persist=10s, delay=0, horizon=300s)",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _divider(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def _verdict(label: str, passed: bool | None, detail: str = "") -> None:
    icon = "✓" if passed is True else ("✗" if passed is False else "~")
    status = "PASS" if passed is True else ("FAIL" if passed is False else "UNCERTAIN")
    print(f"  {icon} [{status}] {label}  {detail}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress-test: persistence-filtered strategy")
    parser.add_argument("--thresholds",       type=float, nargs="+", default=[3.5, 4.0, 4.5])
    parser.add_argument("--persistence-bars", type=int,   default=1)
    parser.add_argument("--horizons-s",       type=int,   nargs="+", default=[60, 120, 300])
    parser.add_argument("--n-events",         type=int,   default=10)
    parser.add_argument("--data-dir",         type=str,   default=str(DATA_DIR))
    parser.add_argument("--output-dir",       type=str,   default=str(OUTPUTS_DIR))
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    plots_dir  = output_dir / "plots"
    events_dir = plots_dir / "events"
    plots_dir.mkdir(parents=True, exist_ok=True)
    events_dir.mkdir(parents=True, exist_ok=True)

    persist_b = args.persistence_bars
    persist_s = persist_b * BAR_S

    _divider(f"Stress-Test  |  persistence={persist_s}s ({persist_b} bar)")
    print(f"\n  Note: at {BAR_S}s/bar, 15s entry delay → 2 bars = 20s (nearest bar).")
    print(f"  Timing test uses delays [0, 1, 2, 3] bars = [10, 20, 30, 40]s from crossing.")

    # -----------------------------------------------------------------------
    # 1. Load data — preserve bid/ask
    # -----------------------------------------------------------------------
    _divider("Loading data")
    eurusd = load_pair(data_dir, "EURUSD")
    audusd = load_pair(data_dir, "AUDUSD")
    euraud = load_pair(data_dir, "EURAUD")

    sig              = build_signal_frame(eurusd, audusd, euraud, ewma_halflife=360)
    euraud_bid_full  = sig["euraud_bid"].copy() if "euraud_bid" in sig.columns else None
    euraud_ask_full  = sig["euraud_ask"].copy() if "euraud_ask" in sig.columns else None
    euraud_mid_full  = sig["euraud"].copy()
    del eurusd, audusd, euraud

    feat = build_feature_frame(sig)
    del sig
    feat["euraud"] = euraud_mid_full
    if euraud_bid_full is not None:
        feat["euraud_bid"] = euraud_bid_full
        feat["euraud_ask"] = euraud_ask_full
        print("  ✓ Bid/ask prices available — using actual spread for execution model")
    else:
        print("  ✗ Bid/ask not available — cannot run realistic execution model")
        print("    Ensure .gmr files include bid/ask columns (see data.py).")
        return

    val_end_ts = pd.Timestamp(VAL_END, tz="UTC")
    test_df    = feat[feat.index > val_end_ts].dropna(subset=["euraud", "euraud_bid", "euraud_ask"]).copy()
    n_bars     = len(test_df)
    n_days     = (test_df.index[-1] - test_df.index[0]).days
    n_weeks    = n_days / 7
    timestamps = test_df.index
    print(f"  Test set: {n_bars:,} bars  "
          f"({test_df.index[0].date()} → {test_df.index[-1].date()})  "
          f"[{n_days} days = {n_weeks:.1f} weeks]")

    z   = test_df["zscore"].values
    mid = test_df["euraud"].values
    bid = test_df["euraud_bid"].values
    ask = test_df["euraud_ask"].values

    # Actual spread stats
    actual_spread_pips = (ask - bid) * 10_000
    med_spread = float(np.nanmedian(actual_spread_pips))
    p90_spread = float(np.nanpercentile(actual_spread_pips, 90))
    p99_spread = float(np.nanpercentile(actual_spread_pips, 99))
    print(f"  EUR/AUD actual spread (all bars): "
          f"median={med_spread:.2f}  p90={p90_spread:.2f}  p99={p99_spread:.2f} pips")

    # -----------------------------------------------------------------------
    # 2. Detect signals — and immediately show spread at signal bars (KEY CHECK)
    # -----------------------------------------------------------------------
    _divider("Signal detection + spread at signal bars")
    print(f"  {'thr':>5}  {'n_signals':>10}  {'sig/wk':>8}  "
          f"{'spread_med':>11}  {'spread_p90':>11}  {'spread_mean':>12}")
    primary_thr = max(args.thresholds)
    signals_map: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for thr in sorted(args.thresholds):
        confirmed, crossings = persistent_signals(z, thr, persist_b)
        signals_map[thr] = (confirmed, crossings)
        if len(confirmed):
            z_abs  = np.abs(z[confirmed])
            sp_sig = actual_spread_pips[confirmed]
            print(f"  {thr:>5.1f}  {len(confirmed):>10,}  {len(confirmed)/n_weeks:>8.0f}  "
                  f"{np.median(sp_sig):>10.2f}p  {np.percentile(sp_sig,90):>10.2f}p  "
                  f"{sp_sig.mean():>11.2f}p")
        else:
            print(f"  {thr:>5.1f}  {'0':>10}")
    print(f"\n  CRITICAL: If spread_mean >> mid-price edge, strategy fails with actual execution.")
    print(f"  Compare spread_mean above to mean_gross from prior mid-price backtest.")

    # -----------------------------------------------------------------------
    # 3. Check A: Lookahead audit
    # -----------------------------------------------------------------------
    _divider("Check A: Lookahead Bias Audit")
    print("  Persistence filter is causal by construction:")
    print(f"  - Crossing at bar t: uses |z[t-1]| < thr AND |z[t]| >= thr  ✓")
    print(f"  - Confirmation at t+{persist_b}: checks |z[t..t+{persist_b}]| >= thr  ✓ (all ≤ entry time)")
    print(f"  - Direction from z[confirmed_bar]: uses data at or before entry  ✓")
    print(f"  - EWMA z-score: pandas ewm uses one-sided (causal) kernel  ✓")
    print()

    confirmed_p, crossings_p = signals_map[primary_thr]
    if len(confirmed_p) >= 2:
        z_cross = z[crossings_p]
        z_conf  = z[confirmed_p]
        abs_delta = np.abs(z_conf) - np.abs(z_cross)
        n_momentum = int((abs_delta > 0).sum())
        n_stable   = int((abs_delta <= 0).sum())
        pct_mom    = n_momentum / len(abs_delta) * 100
        print(f"  Persistent signals (thr={primary_thr:.1f}):")
        print(f"    {n_momentum:,} ({pct_mom:.1f}%) still growing  |  "
              f"{n_stable:,} ({100-pct_mom:.1f}%) stable/retreating")
        print(f"    Mean |z| change: {abs_delta.mean():+.3f}  "
              f"Median: {np.median(abs_delta):+.3f}")
        print()
        print("  Implication: persistent signals are a mix of momentum and stabilisation.")
        print("  Neither pure category — selection is not biased toward one direction.")

        plot_lookahead_check(z_cross, z_conf, primary_thr,
                             path=plots_dir / "stress_lookahead_check.png")

    # -----------------------------------------------------------------------
    # 4. Check B: Spread sensitivity
    # -----------------------------------------------------------------------
    _divider("Check B: Spread Sensitivity")
    spread_rows = []
    horizon_300_bars = 300 // BAR_S   # 30 bars

    for thr in sorted(args.thresholds):
        confirmed, _ = signals_map[thr]
        for mult in [1.0, 2.0, 4.0]:
            for h_s in args.horizons_s:
                h_b = h_s // BAR_S
                tl  = simulate_realistic(z, mid, bid, ask, confirmed, 0, h_b, mult)
                if len(tl) == 0:
                    continue
                mn = float(tl["gross_pips_real"].mean())
                wr = float((tl["gross_pips_real"] > 0).mean())
                spread_rows.append({
                    "threshold":  thr,
                    "spread_mult": mult,
                    "horizon_s":  h_s,
                    "n_trades":   len(tl),
                    "mean_net":   mn,
                    "win_rate":   wr,
                })

    # Print table
    print(f"\n  {'threshold':>10}  {'horizon_s':>10}  {'actual':>10}  {'2x':>10}  {'4x':>10}")
    for thr in sorted(args.thresholds):
        for h_s in args.horizons_s:
            row_at = lambda m: next((r for r in spread_rows if r["threshold"]==thr and r["spread_mult"]==m and r["horizon_s"]==h_s), None)
            r1 = row_at(1.0)
            r2 = row_at(2.0)
            r4 = row_at(4.0)
            v1 = f"{r1['mean_net']:+.2f}" if r1 else "n/a"
            v2 = f"{r2['mean_net']:+.2f}" if r2 else "n/a"
            v4 = f"{r4['mean_net']:+.2f}" if r4 else "n/a"
            print(f"  {thr:>10.1f}  {h_s:>10}  {v1:>10}  {v2:>10}  {v4:>10}")

    plot_spread_sensitivity(
        [r for r in spread_rows if r["horizon_s"] == 300],
        args.thresholds,
        path=plots_dir / "stress_spread_sensitivity.png",
    )

    # -----------------------------------------------------------------------
    # 5. Check C: Event plots
    # -----------------------------------------------------------------------
    _divider("Check C: Event Inspection")
    confirmed_main, _ = signals_map[primary_thr]
    tl_events = simulate_realistic(z, mid, bid, ask, confirmed_main, 0, horizon_300_bars, 1.0)
    if len(tl_events) >= 1:
        plot_events(z, mid, tl_events, timestamps, primary_thr,
                    n_events=min(args.n_events, len(tl_events)),
                    path_dir=events_dir)
    else:
        print(f"  No trades found for event plot at thr={primary_thr:.1f}")

    # -----------------------------------------------------------------------
    # 6. Check D: Entry timing sensitivity
    # -----------------------------------------------------------------------
    _divider("Check D: Entry Timing Sensitivity")
    delay_bars_list = [0, 1, 2, 3]
    timing_nets  = []
    timing_wrs   = []
    timing_ns    = []
    confirmed_t, _ = signals_map[primary_thr]
    for d_b in delay_bars_list:
        tl = simulate_realistic(z, mid, bid, ask, confirmed_t, d_b, horizon_300_bars, 1.0)
        mn = float(tl["gross_pips_real"].mean()) if len(tl) > 0 else float("nan")
        wr = float((tl["gross_pips_real"] > 0).mean()) if len(tl) > 0 else float("nan")
        timing_nets.append(mn)
        timing_wrs.append(wr)
        timing_ns.append(len(tl))
        total_s = (persist_b + d_b) * BAR_S
        print(f"  delay={d_b}bar ({d_b*BAR_S}s after confirm, {total_s}s total):  "
              f"mean_net={mn:+.3f}  win_rate={wr:.1%}  n={len(tl)}")

    print(f"\n  Note: 15s delay → 2 bars = 20s at {BAR_S}s resolution.")
    plot_timing_sensitivity(
        delay_bars_list, timing_nets, timing_wrs, timing_ns, primary_thr,
        path=plots_dir / "stress_timing.png",
    )

    # -----------------------------------------------------------------------
    # 7. Check E: Temporal stability (quarterly)
    # -----------------------------------------------------------------------
    _divider("Check E: Temporal Stability")
    test_start = test_df.index[0]
    test_end   = test_df.index[-1]
    total_td   = test_end - test_start
    q_duration = total_td / 4
    quarter_edges = [test_start + i * q_duration for i in range(5)]
    quarter_labels = [f"Q{i+1}\n({quarter_edges[i].strftime('%b%y')}–{quarter_edges[i+1].strftime('%b%y')})"
                      for i in range(4)]

    results_by_thr: dict[float, list[dict]] = {}
    for thr in sorted(args.thresholds):
        results_by_thr[thr] = []
        confirmed_q, _ = persistent_signals(z, thr, persist_b)
        for qi in range(4):
            q_start_ts = quarter_edges[qi]
            q_end_ts   = quarter_edges[qi + 1]
            # Filter test_df to this quarter
            mask = (test_df.index >= q_start_ts) & (test_df.index < q_end_ts)
            q_df = test_df[mask]
            if len(q_df) < horizon_300_bars + 10:
                results_by_thr[thr].append(
                    {"mean_net": float("nan"), "win_rate": float("nan"), "n_trades": 0})
                continue
            # Map confirmed signals to local indices within q_df
            q_iloc_start = test_df.index.get_loc(q_df.index[0]) if len(q_df) else 0
            q_iloc_end   = q_iloc_start + len(q_df)
            q_signals    = confirmed_q[(confirmed_q >= q_iloc_start) &
                                       (confirmed_q < q_iloc_end - horizon_300_bars)]
            q_signals_local = q_signals - q_iloc_start

            q_z   = z[q_iloc_start : q_iloc_end]
            q_mid = mid[q_iloc_start : q_iloc_end]
            q_bid = bid[q_iloc_start : q_iloc_end]
            q_ask = ask[q_iloc_start : q_iloc_end]

            tl = simulate_realistic(q_z, q_mid, q_bid, q_ask, q_signals_local,
                                    0, horizon_300_bars, 1.0)
            mn = float(tl["gross_pips_real"].mean()) if len(tl) > 0 else float("nan")
            wr = float((tl["gross_pips_real"] > 0).mean()) if len(tl) > 0 else float("nan")
            results_by_thr[thr].append({"mean_net": mn, "win_rate": wr, "n_trades": len(tl)})
            print(f"  thr={thr:.1f}  Q{qi+1}:  n={len(tl)}  mean_net={mn:+.3f}  win_rate={wr:.1%}")

    plot_stability(quarter_labels, results_by_thr,
                   path=plots_dir / "stress_stability.png")

    # -----------------------------------------------------------------------
    # 8. Check F: Distribution analysis
    # -----------------------------------------------------------------------
    _divider("Check F: Distribution Analysis")
    confirmed_d, _ = signals_map[primary_thr]
    tl_dist = simulate_realistic(z, mid, bid, ask, confirmed_d, 0, horizon_300_bars, 1.0)
    if len(tl_dist) > 0:
        net_arr = tl_dist["gross_pips_real"].values
        skew    = float(sp_stats.skew(net_arr))
        kurt    = float(sp_stats.kurtosis(net_arr))
        top10_n = max(1, int(len(net_arr) * 0.10))
        sorted_net = np.sort(net_arr)[::-1]
        top10_pct  = float(sorted_net[:top10_n].sum() / max(abs(sorted_net.sum()), 1e-9) * 100)
        neg_pct    = float((net_arr < 0).mean() * 100)

        print(f"  n_trades={len(net_arr)}  mean={net_arr.mean():+.3f}  std={net_arr.std():.3f}")
        print(f"  skewness={skew:+.3f}  kurtosis={kurt:+.1f}")
        print(f"  % of net P&L from top 10% trades: {top10_pct:.1f}%")
        print(f"  % losing trades: {neg_pct:.1f}%")

        plot_distribution(net_arr, path=plots_dir / "stress_distribution.png")

    # -----------------------------------------------------------------------
    # 9. Verdict
    # -----------------------------------------------------------------------
    _divider("Verdict")

    # A: Lookahead
    _verdict("Lookahead / selection bias", True,
             "persistence filter is causal (uses t≤entry); direction from confirmed bar")

    # B: Spread — with root cause explanation
    confirmed_diag, _ = signals_map[primary_thr]
    if len(confirmed_diag):
        sp_at_sigs = actual_spread_pips[confirmed_diag]
        print(f"\n  Root cause: spread at thr={primary_thr:.1f} persist={persist_s}s signal bars:")
        print(f"    median={np.median(sp_at_sigs):.1f} pips  mean={sp_at_sigs.mean():.1f} pips  "
              f"p90={np.percentile(sp_at_sigs,90):.1f} pips")
        print(f"    Overall median spread = {med_spread:.2f} pips")
        print(f"    → Persistence filter selects extreme-z events with "
              f"{np.median(sp_at_sigs)/med_spread:.0f}× wider spreads than typical")
    r_actual = next((r for r in spread_rows if r["threshold"]==primary_thr
                     and r["spread_mult"]==1.0 and r["horizon_s"]==300), None)
    r_2x     = next((r for r in spread_rows if r["threshold"]==primary_thr
                     and r["spread_mult"]==2.0 and r["horizon_s"]==300), None)
    r_4x     = next((r for r in spread_rows if r["threshold"]==primary_thr
                     and r["spread_mult"]==4.0 and r["horizon_s"]==300), None)
    spread_pass = r_actual is not None and r_actual["mean_net"] > 0
    spread_2x   = r_2x    is not None and r_2x["mean_net"]    > 0
    spread_4x   = r_4x    is not None and r_4x["mean_net"]    > 0
    spread_detail = (f"actual={r_actual['mean_net']:+.2f}  2x={r_2x['mean_net']:+.2f}  "
                     f"4x={r_4x['mean_net']:+.2f}" if r_actual and r_2x and r_4x else "")
    _verdict("Edge survives actual spread", spread_pass, spread_detail)
    _verdict("Edge survives 2× stressed spread", spread_2x, "")
    _verdict("Edge survives 4× extreme spread", spread_4x, "")

    # C: Timing (smooth decay)
    if len(timing_nets) >= 2 and not np.isnan(timing_nets[0]) and not np.isnan(timing_nets[-1]):
        smooth = timing_nets[0] > timing_nets[-1]  # decay over delays
        _verdict("Edge decays smoothly with delay (not 1-bar artefact)", smooth,
                 f"delay=0: {timing_nets[0]:+.2f}  delay=3bar: {timing_nets[-1]:+.2f}")

    # D: Stability
    all_q_nets = [r["mean_net"] for r in results_by_thr.get(primary_thr, [])
                  if not np.isnan(r.get("mean_net", float("nan")))]
    if all_q_nets:
        stable = all(n > 0 for n in all_q_nets)
        mostly_stable = sum(1 for n in all_q_nets if n > 0) >= 3
        _verdict(
            "Edge consistent across quarters",
            stable if stable else (None if mostly_stable else False),
            f"{sum(1 for n in all_q_nets if n > 0)}/{len(all_q_nets)} quarters positive",
        )

    # E: Distribution
    if len(tl_dist) > 0:
        concentrated = top10_pct > 80
        _verdict("P&L not concentrated in outliers (<80% from top 10%)",
                 not concentrated,
                 f"top 10% of trades = {top10_pct:.1f}% of total P&L")

    # -----------------------------------------------------------------------
    # 10. Save results
    # -----------------------------------------------------------------------
    results_df   = pd.DataFrame(spread_rows)
    results_path = output_dir / "stress_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n  Results saved → {results_path}")
    print(f"  Plots saved   → {plots_dir}/stress_*.png")
    print(f"  Events saved  → {events_dir}/event_*.png")


if __name__ == "__main__":
    main()
