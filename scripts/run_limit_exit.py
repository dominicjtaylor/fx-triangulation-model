"""
Limit-Order Exit Model

Tests whether replacing market exits with passive limit orders converts the
near-breakeven mean-reversion strategy into a genuinely profitable one.

Three exit models
-----------------
A) Market exit    : cross spread at T_horizon (bid for longs, ask for shorts)
B) Optimistic limit: always fill at mid[T_horizon] — no fill risk, upper bound
C) Realistic limit : post limit at mid[T_horizon]; scan forward up to max_wait_bars;
                     fill when bid[t] >= limit (long) or ask[t] <= limit (short);
                     forced market exit if not filled within window

The improvement from B over A quantifies the maximum value of a limit exit.
The gap between C and B quantifies fill risk / slippage from the optimistic bound.

Usage
-----
python3 scripts/run_limit_exit.py
python3 scripts/run_limit_exit.py --thresholds 4.0 4.5 --max-waits 60 120 300
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

BAR_S        = 10          # 10-second bars
HORIZON_BARS = 30          # 300 s = 5 min (best horizon from spread-bucket analysis)
MAX_WAIT_DEF = 360         # bars = 60 min max_wait for liquidity recovery at event onset
SPREAD_ILLIQ = 2.0         # pips — illiquid onset
SPREAD_ENTRY_CANDIDATES = [0.6, 0.8, 1.0]   # entry spread filter values to test
MAX_WAITS_EXIT = [6, 12, 30]                 # bars = 60s, 120s, 300s


# ---------------------------------------------------------------------------
# Event detection (same as run_post_event.py)
# ---------------------------------------------------------------------------

def detect_onsets(z: np.ndarray, threshold: float) -> np.ndarray:
    above      = np.abs(z) >= threshold
    onset_mask = np.zeros(len(z), dtype=bool)
    onset_mask[1:] = above[1:] & ~above[:-1]
    return np.where(onset_mask)[0]


# ---------------------------------------------------------------------------
# Simulation with three exit models
# ---------------------------------------------------------------------------

def simulate_all_exits(
    z: np.ndarray,
    mid: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    spread: np.ndarray,
    timestamps: pd.DatetimeIndex,
    threshold: float,
    horizon_bars: int,
    entry_spread_max: float,
    max_wait_exit_bars: int,
) -> pd.DataFrame:
    """No-overlap mean-reversion sim; records all three exit variants per trade.

    Returns one row per trade with columns for each exit model outcome, so all
    comparisons are on identical trade entry/exit samples (no selection bias).

    Columns
    -------
    ts_entry, ts_horizon, z_onset, direction, spread_entry,
    mid_entry, mid_horizon,
    gross_market,         # Model A: cross spread at T_horizon
    gross_optimistic,     # Model B: fill at mid[T_horizon] unconditionally
    gross_limit_{W}s,     # Model C per max_wait: fill if crossed, else forced
    fill_type_{W}s,       # "limit" or "forced"
    fill_delay_{W}s,      # bars from T_horizon to fill (0 = immediate next bar)
    """
    onsets    = detect_onsets(z, threshold)
    n         = len(z)
    trades    = []
    next_free = 0

    wait_labels = [w * BAR_S for w in [max_wait_exit_bars]]  # single for now
    # We'll compute all max_wait variants in one pass
    wait_variants = [6, 12, 30]   # always compute all three

    for oi in onsets:
        # --- Find liquid entry ---
        end  = min(oi + MAX_WAIT_DEF + 1, n)
        hits = np.where(spread[oi:end] <= entry_spread_max)[0]
        if len(hits) == 0:
            continue
        ei = oi + int(hits[0])

        if ei < next_free:
            continue
        if spread[ei] > entry_spread_max:
            continue

        horizon_i = ei + horizon_bars
        # Need enough bars for max_wait scan
        max_scan = max(wait_variants)
        if horizon_i + max_scan >= n:
            continue

        z_onset   = float(z[oi])
        direction = -float(np.sign(z_onset))   # reversion
        if direction == 0:
            continue

        # Entry execution
        entry_exec  = ask[ei] if direction > 0 else bid[ei]
        mid_entry   = float(mid[ei])
        mid_horizon = float(mid[horizon_i])
        limit_price = mid_horizon   # limit posted at mid of T_horizon bar

        # Model A: market exit at T_horizon
        market_exec    = bid[horizon_i] if direction > 0 else ask[horizon_i]
        gross_market   = direction * (market_exec - entry_exec) * 10_000

        # Model B: optimistic limit (always fill at mid[horizon])
        gross_optimistic = direction * (limit_price - entry_exec) * 10_000

        # Model C: realistic limit scan for each max_wait variant
        row = {
            "ts_entry":       timestamps[ei],
            "ts_horizon":     timestamps[horizon_i],
            "z_onset":        z_onset,
            "direction":      direction,
            "spread_entry":   float(spread[ei]),
            "mid_entry":      mid_entry,
            "mid_horizon":    mid_horizon,
            "gross_market":   gross_market,
            "gross_optimistic": gross_optimistic,
        }

        for mw in wait_variants:
            forced_exec  = bid[horizon_i + mw] if direction > 0 else ask[horizon_i + mw]
            fill_i       = -1
            # Scan from horizon_i + 1 to horizon_i + mw (inclusive)
            for t in range(horizon_i + 1, horizon_i + mw + 1):
                if direction > 0:  # LONG: limit SELL, fill when bid[t] >= limit
                    if bid[t] >= limit_price:
                        fill_i = t
                        break
                else:              # SHORT: limit BUY, fill when ask[t] <= limit
                    if ask[t] <= limit_price:
                        fill_i = t
                        break

            if fill_i >= 0:
                fill_exec = limit_price           # filled at limit (mid[horizon])
                fill_type = "limit"
                fill_delay = fill_i - horizon_i   # bars after horizon
            else:
                fill_exec  = forced_exec
                fill_type  = "forced"
                fill_delay = mw

            gross_limit = direction * (fill_exec - entry_exec) * 10_000
            label       = f"{mw * BAR_S}s"
            row[f"gross_limit_{label}"]  = gross_limit
            row[f"fill_type_{label}"]    = fill_type
            row[f"fill_delay_b_{label}"] = fill_delay
            row[f"hold_bars_{label}"]    = (
                fill_i - ei if fill_i >= 0 else horizon_i + mw - ei
            )

        trades.append(row)
        next_free = horizon_i + max(wait_variants) + 1

    tl = pd.DataFrame(trades)
    return tl


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def summarise(tl: pd.DataFrame, col: str, n_weeks: float) -> dict:
    g = tl[col].dropna()
    if len(g) == 0:
        return {}
    return {
        "n":        len(g),
        "trades_wk": len(g) / max(n_weeks, 1e-6),
        "mean_net": float(g.mean()),
        "std":      float(g.std()),
        "sharpe":   float(g.mean() / (g.std() + 1e-10)),
        "hit_rate": float((g > 0).mean()),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_model_comparison(
    results: list[dict],
    path: Path,
) -> None:
    """Bar chart comparing mean_net for market vs optimistic vs realistic exits."""
    # Group by (threshold, entry_spread_max)
    groups = sorted({(r["threshold"], r["entry_spread_max"]) for r in results})
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 5), sharey=True)
    if len(groups) == 1:
        axes = [axes]

    colors    = {"market": "#f48fb1", "optimistic": "#81c784",
                 "60s": "#4fc3f7", "120s": "#ffb74d", "300s": "#ce93d8"}
    mw_labels = ["60s", "120s", "300s"]

    for ax, (thr, sp_max) in zip(axes, groups):
        sub   = [r for r in results if r["threshold"] == thr and r["entry_spread_max"] == sp_max]
        model_names = ["market", "optimistic"] + mw_labels
        means = []
        for mn in model_names:
            vals = [r.get(f"mean_net_{mn}") for r in sub if f"mean_net_{mn}" in r]
            means.append(float(np.mean(vals)) if vals else float("nan"))

        xs = np.arange(len(model_names))
        bars = ax.bar(xs, means, color=[colors.get(m, "#888") for m in model_names],
                      alpha=0.85, edgecolor="black", linewidth=0.4)
        ax.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.6)
        for b, m in zip(bars, means):
            if not np.isnan(m):
                ax.text(b.get_x() + b.get_width() / 2, m + 0.01,
                        f"{m:+.3f}", ha="center", va="bottom" if m >= 0 else "top",
                        fontsize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels(model_names, rotation=20, fontsize=7)
        ax.set_title(f"z≥{thr}  |  spread≤{sp_max}p", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Mean net pips (bid/ask)", fontsize=8)
        ax.grid(alpha=0.1, axis="y")

    fig.suptitle("Exit Model Comparison: Market vs Limit  (horizon=300s)",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


def plot_return_distribution(
    tl: pd.DataFrame,
    threshold: float,
    entry_sp: float,
    path: Path,
) -> None:
    """Side-by-side histograms: market exit vs best realistic limit exit."""
    mw_labels = ["60s", "120s", "300s"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    model_cols = [("market", "gross_market")] + [
        (f"limit_{mw}", f"gross_limit_{mw}") for mw in mw_labels
    ]
    colors_map = {"market": "#f48fb1", "limit_60s": "#4fc3f7",
                  "limit_120s": "#ffb74d", "limit_300s": "#ce93d8"}

    for ax, (label, col) in zip(axes, model_cols):
        if col not in tl.columns:
            continue
        g    = tl[col].dropna().values
        lo   = float(np.percentile(g, 1))
        hi   = float(np.percentile(g, 99))
        bins = np.linspace(lo, hi, 61)
        ax.hist(g, bins=bins, color=colors_map.get(label, "#888"),
                alpha=0.85, edgecolor="black", linewidth=0.2)
        ax.axvline(0,       color="white", linewidth=0.8, linestyle="--")
        ax.axvline(g.mean(), color="#81c784", linewidth=1.5,
                   label=f"mean={g.mean():+.3f}p")
        hit = (g > 0).mean()
        ax.set_title(f"{label}\nhit={hit:.1%}  n={len(g):,}", fontsize=8)
        ax.set_xlabel("Gross pips", fontsize=7)
        if ax is axes[0]:
            ax.set_ylabel("Count", fontsize=7)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.1)

    fig.suptitle(f"Return Distributions — z≥{threshold}  spread≤{entry_sp}p  h=300s",
                 fontsize=9, y=1.01)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


def plot_fill_rate(
    fill_data: list[dict],   # list of {threshold, entry_sp, mw_label, fill_rate}
    path: Path,
) -> None:
    """Fill rate vs z-threshold, grouped by max_wait duration."""
    thresholds = sorted({d["threshold"] for d in fill_data})
    mw_labels  = ["60s", "120s", "300s"]
    colors     = {"60s": "#4fc3f7", "120s": "#ffb74d", "300s": "#ce93d8"}

    fig, axes = plt.subplots(1, len(SPREAD_ENTRY_CANDIDATES),
                             figsize=(5 * len(SPREAD_ENTRY_CANDIDATES), 5), sharey=True)
    if len(SPREAD_ENTRY_CANDIDATES) == 1:
        axes = [axes]

    for ax, sp in zip(axes, SPREAD_ENTRY_CANDIDATES):
        sub = [d for d in fill_data if d["entry_sp"] == sp]
        for mw in mw_labels:
            ys = [
                next((d["fill_rate"] for d in sub
                      if d["threshold"] == thr and d["mw_label"] == mw), float("nan"))
                for thr in thresholds
            ]
            ax.plot(thresholds, ys, marker="o", color=colors[mw],
                    linewidth=1.5, label=f"max_wait={mw}")
        ax.axhline(1.0, color="white", linewidth=0.7, linestyle="--", alpha=0.4)
        ax.set_xlabel("|z| threshold", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Limit fill rate", fontsize=9)
        ax.set_title(f"Entry spread≤{sp}p", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.12)

    fig.suptitle("Limit-Order Fill Rate vs z-Threshold  (limit at mid[T_horizon])",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


def plot_pnl_over_time(
    tl: pd.DataFrame,
    threshold: float,
    entry_sp: float,
    path: Path,
) -> None:
    """Cumulative P&L over time: market vs optimistic vs realistic exits."""
    mw_labels  = ["60s", "120s", "300s"]
    fig, ax    = plt.subplots(figsize=(13, 5))
    colors_map = {
        "gross_market":       "#f48fb1",
        "gross_optimistic":   "#81c784",
        "gross_limit_60s":    "#4fc3f7",
        "gross_limit_120s":   "#ffb74d",
        "gross_limit_300s":   "#ce93d8",
    }
    labels_map = {
        "gross_market":       "Market exit",
        "gross_optimistic":   "Optimistic limit (upper bound)",
        "gross_limit_60s":    "Realistic limit (max_wait=60s)",
        "gross_limit_120s":   "Realistic limit (max_wait=120s)",
        "gross_limit_300s":   "Realistic limit (max_wait=300s)",
    }

    tl_s = tl.sort_values("ts_entry")
    for col, color in colors_map.items():
        if col not in tl_s.columns:
            continue
        mn = float(tl_s[col].mean())
        ax.plot(tl_s["ts_entry"], tl_s[col].cumsum(),
                color=color, linewidth=1.4,
                label=f"{labels_map[col]}  μ={mn:+.3f}p")

    ax.axhline(0, color="white", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.set_xlabel("Date", fontsize=9)
    ax.set_ylabel("Cumulative net pips", fontsize=9)
    ax.set_title(f"P&L Over Time — z≥{threshold}  spread≤{entry_sp}p  h=300s",
                 fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.12)
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
                        default=[3.5, 4.0, 4.5])
    parser.add_argument("--max-waits",  nargs="+", type=int,
                        default=MAX_WAITS_EXIT,
                        help="Max wait bars for limit fill (default: 6 12 30 = 60/120/300s)")
    parser.add_argument("--entry-spreads", nargs="+", type=float,
                        default=SPREAD_ENTRY_CANDIDATES)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Loading data (10s resolution)...")
    eurusd = load_pair(DATA_DIR, "EURUSD")
    audusd = load_pair(DATA_DIR, "AUDUSD")
    euraud = load_pair(DATA_DIR, "EURAUD")

    sig        = build_signal_frame(eurusd, audusd, euraud)
    z          = sig["zscore"].values
    mid        = sig["euraud"].values
    bid        = sig["euraud_bid"].values
    ask        = sig["euraud_ask"].values
    spread     = (sig["euraud_ask"] - sig["euraud_bid"]).values * 10_000
    timestamps = sig.index

    n_weeks    = (timestamps[-1] - timestamps[0]).days / 7
    print(f"  {len(sig):,} bars  |  {timestamps[0].date()} → {timestamps[-1].date()}")
    print(f"  Spread: median={np.median(spread):.3f}p  mean={spread.mean():.3f}p")
    print(f"  Weeks: {n_weeks:.1f}")

    # ------------------------------------------------------------------
    # 2. Run simulations
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Running simulations...")
    print("=" * 60)

    all_tl:    dict = {}   # (thr, sp_max) -> trade_log
    all_stats: list = []
    fill_data: list = []
    comparison_results: list = []
    mw_labels = [f"{w * BAR_S}s" for w in [6, 12, 30]]

    for thr in args.thresholds:
        for sp_max in args.entry_spreads:
            tl = simulate_all_exits(
                z, mid, bid, ask, spread, timestamps,
                thr, HORIZON_BARS, sp_max, max(args.max_waits),
            )
            key = (thr, sp_max)
            all_tl[key] = tl
            nt = len(tl)
            print(f"  z≥{thr}  spread≤{sp_max}p: {nt:,} trades ({nt/n_weeks:.1f}/wk)")

            if nt == 0:
                continue

            # Fill rates
            for mw_label in mw_labels:
                ft_col = f"fill_type_{mw_label}"
                if ft_col in tl.columns:
                    fr = float((tl[ft_col] == "limit").mean())
                    fill_data.append({
                        "threshold": thr, "entry_sp": sp_max,
                        "mw_label":  mw_label, "fill_rate": fr,
                    })

            # Stats per model
            row = {"threshold": thr, "entry_spread_max": sp_max}
            stats_market = summarise(tl, "gross_market", n_weeks)
            row["mean_net_market"]     = stats_market.get("mean_net", np.nan)
            row["hit_rate_market"]     = stats_market.get("hit_rate", np.nan)
            row["trades_wk"]           = stats_market.get("trades_wk", np.nan)

            stats_opt = summarise(tl, "gross_optimistic", n_weeks)
            row["mean_net_optimistic"] = stats_opt.get("mean_net", np.nan)
            row["hit_rate_optimistic"] = stats_opt.get("hit_rate", np.nan)

            for mw_label in mw_labels:
                col = f"gross_limit_{mw_label}"
                ft  = f"fill_type_{mw_label}"
                if col in tl.columns:
                    stats_lim = summarise(tl, col, n_weeks)
                    row[f"mean_net_{mw_label}"]  = stats_lim.get("mean_net", np.nan)
                    row[f"hit_rate_{mw_label}"]  = stats_lim.get("hit_rate", np.nan)
                    if ft in tl.columns:
                        row[f"fill_rate_{mw_label}"]   = float((tl[ft] == "limit").mean())
                        row[f"forced_rate_{mw_label}"]  = float((tl[ft] == "forced").mean())
                        row[f"mean_delay_{mw_label}s"]  = float(
                            tl.loc[tl[ft] == "limit", f"fill_delay_b_{mw_label}"].mean()
                            * BAR_S
                        ) if (tl[ft] == "limit").any() else float("nan")

            all_stats.append(row)
            comparison_results.append(row)

    # ------------------------------------------------------------------
    # 3. Print results table
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("RESULTS TABLE")
    print("=" * 60)

    hdr = (f"{'thr':>5} {'sp':>5} {'model':>15} "
           f"{'n/wk':>7} {'mean_net':>10} {'hit_rate':>9} "
           f"{'fill_rate':>10} {'forced%':>8} {'delay_s':>8}")
    print(f"\n  {hdr}")
    print(f"  {'-'*len(hdr)}")

    for row in all_stats:
        thr    = row["threshold"]
        sp_max = row["entry_spread_max"]
        n_wk   = row["trades_wk"]

        for model, net_key, hit_key, fr_key, fo_key, delay_key in [
            ("market",     "mean_net_market",     "hit_rate_market",     None, None, None),
            ("optimistic", "mean_net_optimistic", "hit_rate_optimistic", None, None, None),
        ] + [
            (mw, f"mean_net_{mw}", f"hit_rate_{mw}",
             f"fill_rate_{mw}", f"forced_rate_{mw}", f"mean_delay_{mw}s")
            for mw in mw_labels
        ]:
            mn  = row.get(net_key, float("nan"))
            hit = row.get(hit_key, float("nan"))
            fr  = row.get(fr_key,  float("nan"))
            fo  = row.get(fo_key,  float("nan"))
            dl  = row.get(delay_key, float("nan"))
            profit_flag = " ✓" if not np.isnan(mn) and mn > 0 else ""
            fr_str  = f"{fr:.1%}" if not np.isnan(fr) else "   —  "
            fo_str  = f"{fo:.1%}" if not np.isnan(fo) else "   —  "
            dl_str  = f"{dl:.1f}s" if not np.isnan(dl) else "   —  "
            print(
                f"  {thr:>5.1f} {sp_max:>5.1f} {model:>15} "
                f"{n_wk:>7.1f} {mn:>10.3f}{profit_flag} {hit:>9.1%} "
                f"{fr_str:>10} {fo_str:>8} {dl_str:>8}"
            )
        print()

    # ------------------------------------------------------------------
    # 4. Verdict
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    profitable = [
        (row, model, row.get(f"mean_net_{model}", np.nan))
        for row in all_stats
        for model in ["optimistic"] + mw_labels
        if row.get(f"mean_net_{model}", -np.inf) > 0
    ]
    if profitable:
        print(f"\n  ✓ {len(profitable)} profitable model configurations found:")
        for row, model, mn in sorted(profitable, key=lambda x: x[2], reverse=True)[:8]:
            fr  = row.get(f"fill_rate_{model}",  float("nan"))
            fo  = row.get(f"forced_rate_{model}", float("nan"))
            nwk = row["trades_wk"]
            fr_s  = f"fill={fr:.0%}"   if not np.isnan(fr) else "fill=N/A"
            fo_s  = f"forced={fo:.0%}" if not np.isnan(fo) else ""
            print(
                f"  z≥{row['threshold']}  sp≤{row['entry_spread_max']}p  {model:15} "
                f"→ {mn:+.3f}p/trade  {nwk:.1f}/wk  {fr_s}  {fo_s}"
            )
    else:
        print("  ✗ No profitable model configurations found.")

    print("\n  Summary by exit model (mean across all thr × sp_max configs):")
    for model, col in [
        ("market",     "mean_net_market"),
        ("optimistic", "mean_net_optimistic"),
    ] + [(mw, f"mean_net_{mw}") for mw in mw_labels]:
        vals = [r.get(col, np.nan) for r in all_stats]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            print(f"  {model:>15}: avg={np.mean(vals):+.3f}p  "
                  f"best={max(vals):+.3f}p  worst={min(vals):+.3f}p")

    # ------------------------------------------------------------------
    # 5. Sensitivity analysis table
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("SENSITIVITY ANALYSIS")
    print("=" * 60)
    print("\n  z-threshold  |  entry spread ≤  |  market  optimistic   60s    120s    300s")
    print("  " + "-" * 75)
    for row in all_stats:
        thr    = row["threshold"]
        sp     = row["entry_spread_max"]
        vals   = [
            row.get("mean_net_market",     float("nan")),
            row.get("mean_net_optimistic", float("nan")),
        ] + [row.get(f"mean_net_{mw}", float("nan")) for mw in mw_labels]
        strs = [f"{v:+.3f}" if not np.isnan(v) else "  NaN " for v in vals]
        stars = ["✓" if v > 0 else " " for v in vals]
        combined = "  ".join(f"{s}{star}" for s, star in zip(strs, stars))
        print(f"  z≥{thr:<5} {sp:>3}p  |  {combined}")

    # ------------------------------------------------------------------
    # 6. Plots
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Generating plots...")
    print("=" * 60)

    # a) Model comparison bar chart
    plot_model_comparison(comparison_results, PLOT_DIR / "limit_exit_comparison.png")

    # b) Return distributions for best config (highest z, tightest spread)
    best_thr = max(args.thresholds)
    best_sp  = min(args.entry_spreads)
    tl_best  = all_tl.get((best_thr, best_sp))
    if tl_best is not None and len(tl_best) > 0:
        plot_return_distribution(
            tl_best, best_thr, best_sp,
            PLOT_DIR / f"limit_exit_returns_z{best_thr}_sp{best_sp}.png",
        )

    # c) Fill rate vs threshold
    if fill_data:
        plot_fill_rate(fill_data, PLOT_DIR / "limit_exit_fill_rate.png")

    # d) P&L over time (best config)
    if tl_best is not None and len(tl_best) > 0:
        plot_pnl_over_time(
            tl_best, best_thr, best_sp,
            PLOT_DIR / "limit_exit_pnl_over_time.png",
        )

    # ------------------------------------------------------------------
    # 7. Save CSV
    # ------------------------------------------------------------------
    summary_rows = [{k: v for k, v in r.items() if not isinstance(v, pd.DataFrame)}
                    for r in all_stats]
    if summary_rows:
        out_csv = OUT_DIR / "limit_exit_results.csv"
        pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
        print(f"\n  Results saved → {out_csv}")


if __name__ == "__main__":
    main()
