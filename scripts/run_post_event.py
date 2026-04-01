"""
Post-Event Mean-Reversion Strategy

Tests whether extreme triangular dislocations (z-score threshold crossings) contain
predictive information about subsequent price behaviour once spreads normalise and
execution constraints are removed.

Strategy logic
--------------
1. Event detection: z-score onset (first crossing from below threshold |z| >= thr)
2. Entry: first bar at or after onset where spread <= spread_entry_max (1.5p)
   Skip event if no liquid bar within max_wait_bars (default 360 = 60 min)
3. Direction test: both mean-reversion and continuation (direction from z at onset)
4. Exit: time-based at each horizon {30s, 60s, 120s, 300s, 600s}
5. Costs: actual bid/ask execution (no fixed cost — spread captured in prices)

Hypothesis
----------
The extreme z-score at onset tells us something about the subsequent price move.
After the spread normalises, either:
  a) Mean-reversion: z returns toward 0 → short when z>0, long when z<0
  b) Continuation: z keeps moving → long when z>0, short when z<0

Usage
-----
python3 scripts/run_post_event.py
python3 scripts/run_post_event.py --thresholds 3.5 4.0 --max-wait 180
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR  = ROOT / "outputs"
PLOT_DIR = OUT_DIR / "plots"

sys.path.insert(0, str(ROOT / "src"))
from triangulation.data     import load_pair
from triangulation.residual import build_signal_frame

BAR_S = 10   # 10-second bars

# Spread thresholds
SPREAD_ILLIQ  = 2.0   # pips — onset marked as "illiquid" if spread > this
SPREAD_ENTRY  = 1.5   # pips — entry allowed only when spread <= this

# Horizons (bars and labels)
HORIZONS = {
    "30s":  3,
    "60s":  6,
    "120s": 12,
    "300s": 30,
    "600s": 60,
}


# ---------------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------------

def detect_events(
    z: np.ndarray,
    spread: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Return indices of z-score onset events.

    An onset is the first bar where |z| crosses above threshold from below.
    Only bars where a prior bar had |z| < threshold count as an onset.

    Args:
        z:          z-score array (10s resolution).
        spread:     spread_pips array, same length.
        threshold:  |z| threshold for event detection.

    Returns:
        1-D integer array of onset bar indices.
    """
    above    = np.abs(z) >= threshold
    # Onset: above[i] AND NOT above[i-1]  (first bar of each crossing)
    onset_mask           = np.zeros(len(z), dtype=bool)
    onset_mask[1:]       = above[1:] & ~above[:-1]
    return np.where(onset_mask)[0]


def find_entries(
    onset_indices: np.ndarray,
    spread: np.ndarray,
    max_wait_bars: int,
    spread_max: float = SPREAD_ENTRY,
) -> tuple[np.ndarray, np.ndarray]:
    """For each onset, find the first bar within max_wait_bars where spread <= spread_max.

    Args:
        onset_indices:  Array of onset bar indices.
        spread:         spread_pips array.
        max_wait_bars:  Maximum bars to wait for liquid entry.
        spread_max:     Spread threshold for liquid entry.

    Returns:
        entry_indices:  Array of entry bar indices (NaN replaced with -1).
        delays_bars:    Bars from onset to entry (0 = entry at onset bar itself).
    """
    n         = len(spread)
    entries   = np.full(len(onset_indices), -1, dtype=np.int64)
    delays    = np.full(len(onset_indices), -1, dtype=np.int64)

    for k, oi in enumerate(onset_indices):
        end = min(oi + max_wait_bars + 1, n)
        window = spread[oi:end]
        hits   = np.where(window <= spread_max)[0]
        if len(hits) > 0:
            off          = int(hits[0])
            entries[k]   = oi + off
            delays[k]    = off

    return entries, delays


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_post_event(
    z:          np.ndarray,
    mid:        np.ndarray,
    bid:        np.ndarray,
    ask:        np.ndarray,
    spread:     np.ndarray,
    timestamps: pd.DatetimeIndex,
    threshold:  float,
    horizon_bars: int,
    direction_mode: str,   # "reversion" or "continuation"
    max_wait_bars: int,
    spread_max: float = SPREAD_ENTRY,
) -> pd.DataFrame:
    """No-overlap post-event simulation.

    Detects events, finds liquid entry bar, executes at bid/ask, exits at horizon.

    Args:
        direction_mode:
          "reversion"   — trade opposite to z at onset (expect gap to close)
          "continuation"— trade in direction of z at onset (expect gap to widen)

    Returns:
        trade_log DataFrame with columns:
          ts_onset, ts_entry, delay_s, z_onset, z_entry, spread_entry,
          direction, gross, illiquid_onset
    """
    onset_idx = detect_events(z, spread, threshold)
    entry_idx, delay_bars = find_entries(onset_idx, spread, max_wait_bars, spread_max)

    n_bars  = len(z)
    trades  = []
    next_free = 0

    for k, oi in enumerate(onset_idx):
        ei = int(entry_idx[k])
        if ei < 0:
            continue                           # no liquid bar found in window
        if ei < next_free:
            continue                           # still in previous trade
        exit_i = ei + horizon_bars
        if exit_i >= n_bars:
            continue

        z_onset  = float(z[oi])
        z_entry  = float(z[ei])
        illiq    = bool(spread[oi] > SPREAD_ILLIQ)

        if direction_mode == "reversion":
            direction = -float(np.sign(z_onset))   # mean-revert: short when z>0
        else:
            direction = float(np.sign(z_onset))    # continuation: long when z>0

        if direction == 0:
            continue

        entry_exec = ask[ei] if direction > 0 else bid[ei]
        exit_exec  = bid[exit_i] if direction > 0 else ask[exit_i]
        gross      = direction * (exit_exec - entry_exec) * 10_000

        trades.append({
            "ts_onset":      timestamps[oi],
            "ts_entry":      timestamps[ei],
            "delay_s":       int(delay_bars[k]) * BAR_S,
            "z_onset":       z_onset,
            "z_entry":       z_entry,
            "spread_entry":  float(spread[ei]),
            "direction":     direction,
            "gross":         gross,
            "illiquid_onset": illiq,
        })
        next_free = exit_i + 1

    return pd.DataFrame(trades)


def run_sweep(
    z, mid, bid, ask, spread, timestamps,
    thresholds, max_wait_bars,
) -> list[dict]:
    """Full sweep: all (threshold × horizon × direction) combinations."""
    rows = []
    for thr in thresholds:
        for horizon_label, horizon_bars in HORIZONS.items():
            for dmode in ("reversion", "continuation"):
                tl = simulate_post_event(
                    z, mid, bid, ask, spread, timestamps,
                    thr, horizon_bars, dmode, max_wait_bars,
                )
                if len(tl) == 0:
                    rows.append({
                        "threshold": thr, "horizon": horizon_label,
                        "direction": dmode, "n_trades": 0,
                        "mean_net": float("nan"), "hit_rate": float("nan"),
                        "sharpe": float("nan"),
                        "pct_illiq_onset": float("nan"),
                        "trades": tl,
                    })
                    continue
                mean_g   = float(tl["gross"].mean())
                std_g    = float(tl["gross"].std() + 1e-10)
                n_weeks  = (timestamps[-1] - timestamps[0]).days / 7
                rows.append({
                    "threshold":       thr,
                    "horizon":         horizon_label,
                    "direction":       dmode,
                    "n_trades":        len(tl),
                    "trades_wk":       len(tl) / max(n_weeks, 1e-6),
                    "mean_net":        mean_g,
                    "hit_rate":        float((tl["gross"] > 0).mean()),
                    "sharpe":          mean_g / std_g,
                    "spread_med":      float(tl["spread_entry"].median()),
                    "pct_illiq_onset": float(tl["illiquid_onset"].mean()),
                    "trades":          tl,
                })
    return rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_pnl_vs_horizon(
    sweep_results: list[dict],
    thresholds: list[float],
    path: Path,
) -> None:
    """mean_net vs holding horizon for each threshold, comparing reversion/continuation."""
    n_thr = len(thresholds)
    fig, axes = plt.subplots(1, n_thr, figsize=(5 * n_thr, 5), sharey=True)
    if n_thr == 1:
        axes = [axes]

    horizon_labels = list(HORIZONS.keys())
    xs             = range(len(horizon_labels))

    for ax, thr in zip(axes, thresholds):
        for dmode, color in [("reversion", "#4fc3f7"), ("continuation", "#ffb74d")]:
            ys = []
            for hl in horizon_labels:
                r = next(
                    (r for r in sweep_results
                     if r["threshold"] == thr and r["horizon"] == hl
                     and r["direction"] == dmode),
                    None,
                )
                ys.append(r["mean_net"] if r and r["n_trades"] > 0 else float("nan"))
            ax.plot(xs, ys, marker="o", color=color, linewidth=1.8, label=dmode)

        ax.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(horizon_labels, rotation=30, fontsize=8)
        ax.set_title(f"z≥{thr}", fontsize=9)
        ax.set_xlabel("Holding horizon", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("Mean net pips (bid/ask)", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.12)

    fig.suptitle("Post-Event Strategy: P&L vs Horizon  (test set)", fontsize=10, y=1.01)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


def plot_mean_net_vs_threshold(
    sweep_results: list[dict],
    thresholds: list[float],
    path: Path,
) -> None:
    """mean_net vs z-threshold, one line per (horizon × direction)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    palette   = ["#4fc3f7", "#81c784", "#ffb74d", "#f48fb1", "#ce93d8"]

    for ax, dmode in zip(axes, ["reversion", "continuation"]):
        for i, (hl, _) in enumerate(HORIZONS.items()):
            ys = []
            for thr in thresholds:
                r = next(
                    (r for r in sweep_results
                     if r["threshold"] == thr and r["horizon"] == hl
                     and r["direction"] == dmode),
                    None,
                )
                ys.append(r["mean_net"] if r and r["n_trades"] > 0 else float("nan"))
            ax.plot(thresholds, ys, marker="o", color=palette[i % len(palette)],
                    linewidth=1.5, label=hl)

        ax.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlabel("|z| threshold", fontsize=9)
        ax.set_ylabel("Mean net pips", fontsize=9)
        ax.set_title(f"Direction: {dmode}", fontsize=9)
        ax.legend(fontsize=7, title="Horizon")
        ax.grid(alpha=0.12)

    fig.suptitle("Post-Event Strategy: mean_net vs Threshold", fontsize=10, y=1.01)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


def plot_liquidity_recovery(
    onset_indices: np.ndarray,
    spread: np.ndarray,
    timestamps: pd.DatetimeIndex,
    threshold: float,
    max_wait_bars: int,
    path: Path,
) -> None:
    """Histogram of liquidity recovery delay (time from onset to liquid entry)."""
    _, delays_bars = find_entries(onset_indices, spread, max_wait_bars, SPREAD_ENTRY)

    illiq_at_onset = spread[onset_indices] > SPREAD_ILLIQ
    liq_at_onset   = ~illiq_at_onset

    # For immediate entries (spread already OK at onset)
    imm_delays   = delays_bars[liq_at_onset & (delays_bars >= 0)] * BAR_S
    # For delayed entries (spread was illiquid at onset)
    delay_delays = delays_bars[illiq_at_onset & (delays_bars >= 0)] * BAR_S
    # Events that never recovered
    n_no_recovery = np.sum(delays_bars < 0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: immediate entries
    ax = axes[0]
    if len(imm_delays) > 0:
        bins = np.arange(0, min(imm_delays.max() + 20, 320), 10)
        ax.hist(imm_delays, bins=bins, color="#81c784", alpha=0.8, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Delay from onset to entry (seconds)", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title(
        f"Already-liquid onsets  (n={len(imm_delays):,})\n"
        f"spread≤{SPREAD_ENTRY}p at onset", fontsize=9,
    )
    ax.set_xlim(left=0)
    ax.grid(alpha=0.12)

    # Right: delayed entries (illiquid at onset)
    ax2 = axes[1]
    if len(delay_delays) > 0:
        bins2 = np.arange(0, min(delay_delays.max() + 30, 640), 30)
        ax2.hist(delay_delays, bins=bins2, color="#4fc3f7", alpha=0.8, edgecolor="black", linewidth=0.3)
    ax2.set_xlabel("Delay from onset to entry (seconds)", fontsize=9)
    ax2.set_ylabel("Count", fontsize=9)
    ax2.set_title(
        f"Illiquid-onset events  (n={len(delay_delays):,} recovered / {np.sum(illiq_at_onset):,} total)\n"
        f"{n_no_recovery:,} events no recovery within {max_wait_bars*BAR_S}s", fontsize=9,
    )
    ax2.set_xlim(left=0)
    ax2.grid(alpha=0.12)

    fig.suptitle(
        f"Liquidity Recovery Delay — z≥{threshold}  "
        f"(max wait={max_wait_bars*BAR_S}s)", fontsize=10, y=1.01,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


def plot_spread_around_events(
    onset_indices: np.ndarray,
    spread: np.ndarray,
    window_bars: int = 60,   # ±10 min at 10s
    path: Path = None,
) -> None:
    """Average spread in a symmetric window around event onsets."""
    n      = len(spread)
    # Subset to onsets with enough context
    valid  = onset_indices[
        (onset_indices >= window_bars) & (onset_indices + window_bars < n)
    ]

    if len(valid) == 0:
        print("  ⚠  Not enough events with context for spread-around-events plot")
        return

    # Stack windows
    stack = np.stack([spread[oi - window_bars : oi + window_bars + 1] for oi in valid])
    mean_sp  = stack.mean(axis=0)
    p25_sp   = np.percentile(stack, 25, axis=0)
    p75_sp   = np.percentile(stack, 75, axis=0)
    t_axis   = np.arange(-window_bars, window_bars + 1) * BAR_S   # seconds

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(t_axis, p25_sp, p75_sp, alpha=0.25, color="#4fc3f7", label="IQR")
    ax.plot(t_axis, mean_sp, color="#4fc3f7", linewidth=2, label="Mean spread")
    ax.axvline(0, color="#f48fb1", linewidth=1.5, linestyle="--", label="Event onset")
    ax.axhline(SPREAD_ENTRY, color="#81c784", linewidth=1.2, linestyle=":",
               label=f"Entry threshold ({SPREAD_ENTRY}p)")
    ax.axhline(SPREAD_ILLIQ, color="#ffb74d", linewidth=1.2, linestyle=":",
               label=f"Illiquid threshold ({SPREAD_ILLIQ}p)")
    ax.set_xlabel("Seconds relative to event onset", fontsize=9)
    ax.set_ylabel("Spread (pips)", fontsize=9)
    ax.set_title(
        f"Average spread around event onsets  (n={len(valid):,} events)", fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.12)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


def plot_return_distribution(
    reversion_trades: pd.DataFrame,
    continuation_trades: pd.DataFrame,
    threshold: float,
    horizon_label: str,
    path: Path,
) -> None:
    """Histogram of per-trade gross P&L for reversion vs continuation."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, tl, dmode, color in [
        (axes[0], reversion_trades,    "reversion",    "#4fc3f7"),
        (axes[1], continuation_trades, "continuation", "#ffb74d"),
    ]:
        if len(tl) == 0:
            ax.set_title(f"{dmode} — no trades")
            continue
        g     = tl["gross"].values
        bins  = np.linspace(np.percentile(g, 1), np.percentile(g, 99), 51)
        ax.hist(g, bins=bins, color=color, alpha=0.8, edgecolor="black", linewidth=0.2)
        ax.axvline(0,          color="white",   linewidth=1.0, linestyle="--")
        ax.axvline(g.mean(),   color="#f48fb1", linewidth=1.5, linestyle="-",
                   label=f"mean={g.mean():+.2f}p")
        ax.set_xlabel("Gross pips (bid/ask)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title(
            f"{dmode.capitalize()}  z≥{threshold}, h={horizon_label}\n"
            f"n={len(g):,}  hit={float((g>0).mean()):.1%}", fontsize=9,
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.12)

    fig.suptitle("Return Distribution — Post-Event Trades", fontsize=10, y=1.01)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[3.0, 3.5, 4.0])
    parser.add_argument("--max-wait",   type=int, default=360,
                        help="Max bars to wait for liquid entry (default 360 = 60 min at 10s)")
    parser.add_argument("--spread-entry", type=float, default=SPREAD_ENTRY)
    args = parser.parse_args()

    spread_entry = args.spread_entry

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Loading data (10s resolution)...")
    eurusd = load_pair(DATA_DIR, "EURUSD")
    audusd = load_pair(DATA_DIR, "AUDUSD")
    euraud = load_pair(DATA_DIR, "EURAUD")

    sig       = build_signal_frame(eurusd, audusd, euraud)
    z         = sig["zscore"].values
    mid       = sig["euraud"].values
    bid       = sig["euraud_bid"].values
    ask       = sig["euraud_ask"].values
    spread    = (sig["euraud_ask"] - sig["euraud_bid"]).values * 10_000
    timestamps = sig.index

    total_weeks = (timestamps[-1] - timestamps[0]).days / 7
    print(f"  Bars: {len(sig):,}  ({timestamps[0].date()} → {timestamps[-1].date()})")
    print(f"  Weeks: {total_weeks:.1f}")
    print(f"  Spread: median={np.median(spread):.3f}p  mean={spread.mean():.3f}p  "
          f"p90={np.percentile(spread, 90):.3f}p")

    # ------------------------------------------------------------------
    # 2. Event statistics
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("EVENT STATISTICS")
    print("=" * 60)
    for thr in sorted(set([3.0] + args.thresholds)):
        onsets   = detect_events(z, spread, thr)
        n_illiq  = int(np.sum(spread[onsets] > SPREAD_ILLIQ))
        n_liq    = int(np.sum(spread[onsets] <= spread_entry))
        _, delays = find_entries(onsets, spread, args.max_wait, spread_entry)
        n_recovered = int(np.sum(delays >= 0))
        rec_delays  = delays[delays >= 0] * BAR_S

        print(f"\n  z≥{thr}: {len(onsets):,} onsets ({len(onsets)/total_weeks:.1f}/wk)")
        print(f"    Already liquid (spread≤{spread_entry}p): {n_liq:,} "
              f"({100*n_liq/max(len(onsets),1):.1f}%)")
        print(f"    Illiquid at onset (spread>{SPREAD_ILLIQ}p): {n_illiq:,} "
              f"({100*n_illiq/max(len(onsets),1):.1f}%)")
        print(f"    Entries found within {args.max_wait*BAR_S}s: {n_recovered:,} "
              f"({100*n_recovered/max(len(onsets),1):.1f}%)")
        if len(rec_delays) > 0:
            # Split by whether onset was liquid or illiquid
            recovered_mask  = delays >= 0
            recovered_idx   = np.where(recovered_mask)[0]
            onset_liq_mask  = spread[onsets[recovered_idx]] <= spread_entry
            onset_illiq_mask = spread[onsets[recovered_idx]] > spread_entry

            imm_delays  = rec_delays[onset_liq_mask]
            del_delays  = rec_delays[onset_illiq_mask]

            if len(imm_delays) > 0:
                print(f"    Delay (immediately liquid): "
                      f"median={np.median(imm_delays):.0f}s  n={len(imm_delays):,}")
            if len(del_delays) > 0:
                print(f"    Delay (post-illiquid recover): "
                      f"median={np.median(del_delays):.0f}s  "
                      f"p90={np.percentile(del_delays, 90):.0f}s  "
                      f"n={len(del_delays):,}")

    # ------------------------------------------------------------------
    # 3. Run full sweep
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("RUNNING SIMULATIONS")
    print("=" * 60)
    print("  (threshold × horizon × direction — all combinations)")

    all_results = run_sweep(
        z, mid, bid, ask, spread, timestamps,
        args.thresholds, args.max_wait,
    )

    # ------------------------------------------------------------------
    # 4. Print results table
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("RESULTS TABLE (test set = full period)")
    print("=" * 60)
    hdr = (f"{'thr':>5} {'horizon':>7} {'direction':>14} "
           f"{'n_trades':>9} {'trades_wk':>10} {'mean_net':>10} "
           f"{'hit_rate':>9} {'sharpe':>8} {'spread_med':>11} {'%illiq':>7}")
    print(f"\n  {hdr}")
    print(f"  {'-'*len(hdr)}")
    for r in all_results:
        if r["n_trades"] == 0:
            continue
        print(
            f"  {r['threshold']:>5.1f} {r['horizon']:>7} {r['direction']:>14} "
            f"{r['n_trades']:>9,} {r['trades_wk']:>10.1f} {r['mean_net']:>10.3f} "
            f"{r['hit_rate']:>9.1%} {r['sharpe']:>8.3f} "
            f"{r['spread_med']:>10.3f}p {r['pct_illiq_onset']:>7.1%}"
        )

    # ------------------------------------------------------------------
    # 5. Best configs
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("BEST CONFIGURATIONS (by mean_net)")
    print("=" * 60)
    ranked = sorted(
        [r for r in all_results if r["n_trades"] > 10],
        key=lambda r: r["mean_net"], reverse=True,
    )
    print()
    for r in ranked[:5]:
        print(
            f"  thr={r['threshold']} h={r['horizon']} {r['direction']:14} "
            f"mean_net={r['mean_net']:+.3f}p  hit={r['hit_rate']:.1%}  "
            f"n={r['n_trades']:,}  {r['trades_wk']:.1f}/wk"
        )

    # ------------------------------------------------------------------
    # 6. Verdict
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    profitable = [r for r in all_results if r.get("mean_net", -np.inf) > 0 and r["n_trades"] > 10]
    if profitable:
        print(f"  ✓ {len(profitable)} profitable configurations found")
        for r in sorted(profitable, key=lambda r: r["mean_net"], reverse=True)[:3]:
            print(f"    thr={r['threshold']} h={r['horizon']} {r['direction']:14} "
                  f"→ {r['mean_net']:+.3f}p/trade  {r['trades_wk']:.1f}/wk")
    else:
        print("  ✗ No profitable configurations found")

    # Summarise reversion vs continuation
    for dmode in ("reversion", "continuation"):
        mean_nets = [r["mean_net"] for r in all_results
                     if r["direction"] == dmode and r["n_trades"] > 10]
        if mean_nets:
            print(f"  {dmode}: mean across configs = {np.mean(mean_nets):+.3f}p  "
                  f"(best={max(mean_nets):+.3f}p  worst={min(mean_nets):+.3f}p)")

    # ------------------------------------------------------------------
    # 7. Quarterly stability
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("QUARTERLY STABILITY (best reversion config)")
    print("=" * 60)
    rev_ranked = sorted(
        [r for r in all_results if r["direction"] == "reversion" and r["n_trades"] > 10],
        key=lambda r: r["mean_net"], reverse=True,
    )
    if rev_ranked:
        best = rev_ranked[0]
        tl   = best["trades"]
        if "ts_entry" in tl.columns and len(tl) > 0:
            tl["quarter"] = tl["ts_entry"].dt.to_period("Q")
            print(f"\n  Config: z≥{best['threshold']}  h={best['horizon']}  reversion")
            print(f"  {'Quarter':>10} {'n':>7} {'mean_net':>10} {'hit_rate':>10}")
            for q, grp in tl.groupby("quarter"):
                print(f"  {str(q):>10} {len(grp):>7,} "
                      f"{grp['gross'].mean():>10.3f}p "
                      f"{(grp['gross']>0).mean():>10.1%}")

    # ------------------------------------------------------------------
    # 8. Plots
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Generating plots...")
    print("=" * 60)

    # a) PnL vs horizon
    plot_pnl_vs_horizon(
        all_results, args.thresholds,
        PLOT_DIR / "post_event_pnl_vs_horizon.png",
    )

    # b) mean_net vs threshold
    plot_mean_net_vs_threshold(
        all_results, args.thresholds,
        PLOT_DIR / "post_event_mean_net_vs_threshold.png",
    )

    # c) Liquidity recovery delay (use reference threshold = median of thresholds)
    ref_thr   = args.thresholds[len(args.thresholds) // 2]
    ref_onset = detect_events(z, spread, ref_thr)
    plot_liquidity_recovery(
        ref_onset, spread, timestamps, ref_thr, args.max_wait,
        PLOT_DIR / "post_event_liquidity_recovery.png",
    )

    # d) Spread around events
    plot_spread_around_events(
        ref_onset, spread, window_bars=60,
        path=PLOT_DIR / "post_event_spread_around_events.png",
    )

    # e) Return distribution — best reversion config at 120s horizon
    if rev_ranked:
        best_thr = rev_ranked[0]["threshold"]
        rev_120  = next(
            (r for r in all_results
             if r["threshold"] == best_thr and r["horizon"] == "120s"
             and r["direction"] == "reversion"),
            None,
        )
        cont_120 = next(
            (r for r in all_results
             if r["threshold"] == best_thr and r["horizon"] == "120s"
             and r["direction"] == "continuation"),
            None,
        )
        if rev_120 and cont_120 and rev_120["n_trades"] > 0:
            plot_return_distribution(
                rev_120["trades"],
                cont_120["trades"] if cont_120 else pd.DataFrame(),
                best_thr, "120s",
                PLOT_DIR / "post_event_return_distribution.png",
            )

    # ------------------------------------------------------------------
    # 9. Save CSV
    # ------------------------------------------------------------------
    save_rows = [
        {k: v for k, v in r.items() if k != "trades"}
        for r in all_results if r["n_trades"] > 0
    ]
    if save_rows:
        out_csv = OUT_DIR / "post_event_results.csv"
        pd.DataFrame(save_rows).to_csv(out_csv, index=False)
        print(f"\n  Results saved → {out_csv}")


if __name__ == "__main__":
    main()
