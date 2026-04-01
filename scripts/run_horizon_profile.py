"""
Horizon profiling — EUR/USD/AUD Triangle

Identifies the holding horizon where mean-reversion survives realistic costs and
execution delay. Uses simple z-score crossing events (not model predictions) to
profile the raw signal quality as a function of horizon.

Two analyses:

A) Profiling (unconstrained): for every crossing event, compute forward returns
   at ALL (delay, horizon) combinations. Shows the pure signal quality.

B) No-overlap simulation: one trade at a time. Shows realistic tradeable P&L.

Data frequency: 10-second bars (native resolution).
Second → bar mapping:
  Horizons [1,2,5,10,20,30,60,120]s → bars [1,1,1,1,2,3,6,12]
  Delays   [0,1,2,5]s               → bars [0,1,1,1]
  Sub-10s horizons (1,2,5s) all resolve to 1 bar = 10s at this resolution.

Run from repo root:
    python3 scripts/run_horizon_profile.py [options]

Options:
    --thresholds FLOAT ...    z-crossing thresholds (default: 2.0 2.5 3.0)
    --horizons-s INT ...      forward-look horizons in seconds
                              (default: 1 2 5 10 20 30 60 120)
    --delays-s INT ...        entry delays in seconds (default: 0 1 2 5)
    --costs FLOAT ...         round-trip costs in pips (default: 0.3 0.5)
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
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from triangulation.data import load_pair
from triangulation.residual import build_signal_frame
from triangulation.features import build_feature_frame
from triangulation.labels import compute_future_zscore_targets

plt.style.use("dark_background")

DATA_DIR    = ROOT / "data"
PLOTS_DIR   = ROOT / "outputs" / "plots"
OUTPUTS_DIR = ROOT / "outputs"

VAL_END   = "2025-06-30"
BAR_S     = 10        # native bar duration in seconds
_BARS_30D = 259_200   # 30 days at 10s/bar for vol baseline


# ---------------------------------------------------------------------------
# Signal detection
# ---------------------------------------------------------------------------

def crossing_events(z: np.ndarray, threshold: float) -> np.ndarray:
    """Return bar indices where |z| crosses ABOVE threshold from below.

    A crossing is defined as: |z[t-1]| < threshold AND |z[t]| >= threshold.
    Both positive (z→+threshold) and negative (z→-threshold) crossings are
    included. This prevents re-triggering on every bar while z remains elevated.
    """
    above = np.abs(z) >= threshold
    return np.where(above[1:] & ~above[:-1])[0] + 1


# ---------------------------------------------------------------------------
# Profiling (unconstrained)
# ---------------------------------------------------------------------------

def profile_returns(
    z: np.ndarray,
    prices: np.ndarray,
    rv1m: np.ndarray,
    signal_indices: np.ndarray,
    delay_bars: list[int],
    horizon_bars: list[int],
    horizon_s: list[int],
    threshold: float,
    vol_tertiles: tuple[float, float],
) -> pd.DataFrame:
    """Compute forward returns for every (signal, delay, horizon) combination.

    No overlap constraint is applied — all signals are profiled independently.
    This shows the raw signal quality before trading constraints.

    Args:
        z:             Z-score series.
        prices:        EUR/AUD price series (same index as z).
        rv1m:          1-min residual RV at each bar (for vol regime labelling).
        signal_indices: Bars where |z| crossed above threshold.
        delay_bars:    Unique delay values in bars.
        horizon_bars:  Unique horizon values in bars.
        horizon_s:     Corresponding horizon values in seconds (parallel list).
        threshold:     The threshold used to detect signal_indices.
        vol_tertiles:  (p33, p67) of rv1m over the sample, for regime labelling.

    Returns:
        DataFrame with columns:
            threshold, sig_i, sig_z, delay_bars, horizon_bars, horizon_s,
            gross_pips, vol_regime
    """
    n = len(prices)
    p33, p67 = vol_tertiles
    rows = []

    for sig_i in signal_indices:
        direction = float(np.sign(z[sig_i]))
        sig_rv    = rv1m[sig_i]
        if sig_rv < p33:
            vol_regime = "low"
        elif sig_rv < p67:
            vol_regime = "medium"
        else:
            vol_regime = "high"

        for d in delay_bars:
            entry_i = sig_i + d
            if entry_i >= n:
                continue
            entry_p = prices[entry_i]

            for h_b, h_s in zip(horizon_bars, horizon_s):
                exit_i = entry_i + h_b
                if exit_i >= n:
                    continue
                gross = direction * (entry_p - prices[exit_i]) * 10_000
                rows.append({
                    "threshold":   threshold,
                    "sig_i":       sig_i,
                    "sig_z":       float(z[sig_i]),
                    "delay_bars":  d,
                    "horizon_bars": h_b,
                    "horizon_s":   h_s,
                    "gross_pips":  gross,
                    "vol_regime":  vol_regime,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# No-overlap simulation
# ---------------------------------------------------------------------------

def simulate_no_overlap(
    z: np.ndarray,
    prices: np.ndarray,
    signal_indices: np.ndarray,
    delay_bars: int,
    horizon_bars: int,
) -> pd.DataFrame:
    """Simulate one-trade-at-a-time. Entry at sig_i + delay, exit at entry + horizon.

    Returns trade_log with gross_pips per trade.
    """
    n = len(prices)
    trades = []
    next_free = 0

    for sig_i in signal_indices:
        entry_i = sig_i + delay_bars
        if entry_i >= n - horizon_bars:
            continue
        if entry_i < next_free:
            continue

        direction = float(np.sign(z[sig_i]))
        exit_i    = entry_i + horizon_bars
        gross     = direction * (prices[entry_i] - prices[exit_i]) * 10_000
        trades.append({
            "entry_i":    entry_i,
            "exit_i":     exit_i,
            "direction":  direction,
            "gross_pips": gross,
        })
        next_free = exit_i + 1

    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_mean_return_vs_horizon(
    prof: pd.DataFrame,
    thresholds: list[float],
    cost: float,
    path: Path,
) -> None:
    """Mean gross and net return vs horizon — one line per threshold, delay=0."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#4fc3f7", "#81c784", "#ffb74d"]

    for ax, subtract_cost, title_suffix in zip(
        axes,
        [False, True],
        ["Gross (before costs)", f"Net (after {cost} pips)"],
    ):
        d0 = prof[prof["delay_bars"] == 0]
        for i, thr in enumerate(sorted(thresholds)):
            sub = d0[d0["threshold"] == thr]
            grp = sub.groupby("horizon_s")["gross_pips"].agg(["mean", "sem", "count"])
            grp = grp.sort_index()
            ys = grp["mean"] - (cost if subtract_cost else 0)
            ax.plot(grp.index, ys, marker="o", markersize=5, linewidth=1.8,
                    color=colors[i % len(colors)],
                    label=f"threshold={thr:.1f}  (n≈{int(grp['count'].mean()):,}/horizon)")
            ax.fill_between(
                grp.index,
                ys - grp["sem"],
                ys + grp["sem"],
                alpha=0.15, color=colors[i % len(colors)],
            )
        ax.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlabel("Horizon (seconds)", fontsize=9)
        ax.set_ylabel("Mean pips per trade", fontsize=9)
        ax.set_title(title_suffix, fontsize=10)
        ax.set_xscale("log")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.15)

    fig.suptitle("Mean Return vs Holding Horizon  (delay=0, profiling — no overlap constraint)",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


def plot_net_return_by_delay(
    prof: pd.DataFrame,
    delay_bars: list[int],
    cost: float,
    threshold: float,
    path: Path,
) -> None:
    """Net return vs horizon for each delay level — shows latency impact."""
    sub = prof[prof["threshold"] == threshold]
    colors = ["#4fc3f7", "#81c784", "#ffb74d", "#f48fb1"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, d in enumerate(sorted(delay_bars)):
        d_sub = sub[sub["delay_bars"] == d]
        grp = d_sub.groupby("horizon_s")["gross_pips"].agg(["mean", "sem"])
        grp = grp.sort_index()
        net = grp["mean"] - cost
        label = f"delay={d * BAR_S}s ({d} bar{'s' if d != 1 else ''})"
        ax.plot(grp.index, net, marker="o", markersize=5, linewidth=1.8,
                color=colors[i % len(colors)], label=label)
        ax.fill_between(grp.index, net - grp["sem"], net + grp["sem"],
                        alpha=0.15, color=colors[i % len(colors)])

    ax.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Horizon (seconds)", fontsize=9)
    ax.set_ylabel(f"Mean net pips (after {cost} pips cost)", fontsize=9)
    ax.set_title(
        f"Net Return vs Horizon by Entry Delay\n(threshold={threshold:.1f}, cost={cost})",
        fontsize=10,
    )
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


def plot_heatmap_delay_horizon(
    prof: pd.DataFrame,
    delay_bars: list[int],
    horizon_bars: list[int],
    horizon_s_vals: list[int],
    cost: float,
    threshold: float,
    path: Path,
) -> None:
    """Heatmap: delay × horizon → mean net pips. Profitable cells marked *."""
    sub   = prof[prof["threshold"] == threshold]
    h_map = dict(zip(horizon_bars, horizon_s_vals))
    d_lab = [f"{d * BAR_S}s" for d in sorted(delay_bars)]
    h_lab = [f"{h_map[h]}s\n({h}b)" for h in sorted(set(horizon_bars))]
    h_uniq = sorted(set(horizon_bars))

    grid = np.full((len(sorted(delay_bars)), len(h_uniq)), float("nan"))
    for ri, d in enumerate(sorted(delay_bars)):
        for ci, h in enumerate(h_uniq):
            sel = sub[(sub["delay_bars"] == d) & (sub["horizon_bars"] == h)]
            if len(sel):
                grid[ri, ci] = sel["gross_pips"].mean() - cost

    vmax = max(abs(np.nanmin(grid)), abs(np.nanmax(grid)), 0.01)

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=-vmax, vmax=vmax,
                   aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="mean net pips")

    ax.set_xticks(range(len(h_uniq)))
    ax.set_xticklabels(h_lab, fontsize=8)
    ax.set_yticks(range(len(sorted(delay_bars))))
    ax.set_yticklabels(d_lab, fontsize=8)
    ax.set_xlabel("Horizon (bars = 10s each)", fontsize=9)
    ax.set_ylabel("Entry delay", fontsize=9)
    ax.set_title(
        f"Mean Net P&L: Delay × Horizon  (threshold={threshold:.1f}, cost={cost})",
        fontsize=10,
    )

    for ri in range(grid.shape[0]):
        for ci in range(grid.shape[1]):
            v = grid[ri, ci]
            if not np.isnan(v):
                marker = "*" if v > 0 else ""
                color = "black" if abs(v) < vmax * 0.6 else "white"
                ax.text(ci, ri, f"{v:+.3f}{marker}", ha="center", va="center",
                        fontsize=8, color=color)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


def plot_return_distributions(
    prof: pd.DataFrame,
    horizon_bars_to_show: list[int],
    horizon_s_map: dict[int, int],
    threshold: float,
    cost: float,
    path: Path,
) -> None:
    """Histograms of gross_pips at selected horizons, delay=0."""
    sub = prof[(prof["threshold"] == threshold) & (prof["delay_bars"] == 0)]

    n_panels = len(horizon_bars_to_show)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.5 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    for ax, h_b in zip(axes, horizon_bars_to_show):
        sel   = sub[sub["horizon_bars"] == h_b]["gross_pips"].values
        h_s   = horizon_s_map.get(h_b, h_b * BAR_S)
        mn    = sel.mean()
        std   = sel.std()
        lo, hi = np.percentile(sel, [1, 99])
        bins = np.linspace(lo, hi, 40)
        ax.hist(sel, bins=bins, alpha=0.7, color="#4fc3f7", density=True)
        ax.axvline(0,    color="white",  linewidth=0.9, linestyle="--", alpha=0.5, label="zero")
        ax.axvline(cost, color="#ffb74d", linewidth=0.9, linestyle=":",  label=f"cost={cost}")
        ax.axvline(mn,   color="#81c784", linewidth=1.1, linestyle="-",  label=f"mean={mn:+.3f}")
        ax.set_title(f"{h_s}s ({h_b} bars)\nn={len(sel):,}", fontsize=9)
        ax.set_xlabel("gross pips", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("density", fontsize=8)
        ax.legend(fontsize=6)
        ax.grid(alpha=0.12)
        # Annotate win rate
        wr = float((sel > cost).mean())
        ax.text(0.97, 0.95, f"net>0: {wr:.1%}", transform=ax.transAxes,
                ha="right", va="top", fontsize=7)

    fig.suptitle(
        f"Return Distributions by Horizon  (threshold={threshold:.1f}, delay=0)",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


def plot_vol_regime(
    prof: pd.DataFrame,
    threshold: float,
    cost: float,
    path: Path,
) -> None:
    """Mean gross return vs horizon by vol regime, delay=0."""
    sub    = prof[(prof["threshold"] == threshold) & (prof["delay_bars"] == 0)]
    reg_colors = {"low": "#4fc3f7", "medium": "#81c784", "high": "#ffb74d"}

    fig, ax = plt.subplots(figsize=(9, 5))
    for regime, color in reg_colors.items():
        r_sub = sub[sub["vol_regime"] == regime]
        if len(r_sub) == 0:
            continue
        grp = r_sub.groupby("horizon_s")["gross_pips"].agg(["mean", "sem"]).sort_index()
        ax.plot(grp.index, grp["mean"], marker="o", markersize=4, linewidth=1.8,
                color=color, label=f"{regime} vol  (n≈{len(r_sub)//len(grp):.0f}/horizon)")
        ax.fill_between(grp.index, grp["mean"] - grp["sem"], grp["mean"] + grp["sem"],
                        alpha=0.15, color=color)

    ax.axhline(cost, color="white", linewidth=0.8, linestyle=":",
               alpha=0.6, label=f"cost={cost} pips")
    ax.axhline(0, color="white", linewidth=0.5, linestyle="--", alpha=0.3)
    ax.set_xlabel("Horizon (seconds)", fontsize=9)
    ax.set_ylabel("Mean gross pips", fontsize=9)
    ax.set_title(
        f"Mean Return by Volatility Regime  (threshold={threshold:.1f}, delay=0)",
        fontsize=10,
    )
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Horizon profiling for mean-reversion signal")
    parser.add_argument("--thresholds",  type=float, nargs="+", default=[2.0, 2.5, 3.0])
    parser.add_argument("--horizons-s",  type=int,   nargs="+",
                        default=[1, 2, 5, 10, 20, 30, 60, 120])
    parser.add_argument("--delays-s",    type=int,   nargs="+", default=[0, 1, 2, 5])
    parser.add_argument("--costs",       type=float, nargs="+", default=[0.3, 0.5])
    parser.add_argument("--data-dir",    type=str,   default=str(DATA_DIR))
    parser.add_argument("--output-dir",  type=str,   default=str(OUTPUTS_DIR))
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    plots_dir  = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 0. State data frequency and bar→second mapping
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("  Horizon Profiling — EUR/USD/AUD Triangle")
    print("=" * 60)
    print(f"\n  Data frequency: {BAR_S} seconds per bar (native resolution)")

    # Build unique bar values from requested seconds
    horizon_bars_raw = [max(1, round(h / BAR_S)) for h in args.horizons_s]
    delay_bars_raw   = [math.ceil(d / BAR_S)       for d in args.delays_s]

    # De-duplicate while preserving original seconds for labels
    seen_h:  dict[int, int] = {}   # bars → first matching seconds
    seen_d:  dict[int, int] = {}
    for h_b, h_s in zip(horizon_bars_raw, args.horizons_s):
        if h_b not in seen_h:
            seen_h[h_b] = h_s
    for d_b, d_s in zip(delay_bars_raw, args.delays_s):
        if d_b not in seen_d:
            seen_d[d_b] = d_s

    u_horizon_bars = sorted(seen_h.keys())   # [1, 2, 3, 6, 12]
    u_horizon_s    = [seen_h[b] for b in u_horizon_bars]
    u_delay_bars   = sorted(seen_d.keys())   # [0, 1]

    print("\n  Second → bar mapping:")
    print(f"  {'Horizon (s)':>12}  {'→ bars':>8}  {'→ effective (s)':>17}")
    for h_s in sorted(set(args.horizons_s)):
        h_b = max(1, round(h_s / BAR_S))
        note = f"  [same as {seen_h[h_b]}s]" if h_b in seen_h and seen_h[h_b] != h_s else ""
        print(f"  {h_s:>12}  {h_b:>8}  {h_b*BAR_S:>17}{note}")
    print(f"\n  Unique horizon bars: {u_horizon_bars}  = {[b*BAR_S for b in u_horizon_bars]}s")
    print(f"  Unique delay bars:   {u_delay_bars}  = {[b*BAR_S for b in u_delay_bars]}s")

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Loading data + rebuilding features")
    print("=" * 60)
    eurusd = load_pair(data_dir, "EURUSD")
    audusd = load_pair(data_dir, "AUDUSD")
    euraud = load_pair(data_dir, "EURAUD")

    sig  = build_signal_frame(eurusd, audusd, euraud, ewma_halflife=360)
    del eurusd, audusd, euraud
    euraud_prices_full = sig["euraud"].copy()
    feat = build_feature_frame(sig)
    del sig
    feat["euraud"] = euraud_prices_full
    print(f"  Full feature frame: {feat.shape}")

    # Compute vol baseline for rv regime labelling
    feat["rv_baseline_30d"] = (
        feat["rv_residual_1m"].rolling(_BARS_30D, min_periods=360).mean()
    )

    # -----------------------------------------------------------------------
    # 2. Filter to test set
    # -----------------------------------------------------------------------
    val_end_ts = pd.Timestamp(VAL_END, tz="UTC")
    test_df    = feat[feat.index > val_end_ts].copy()
    n_bars     = len(test_df)
    n_days     = (test_df.index[-1] - test_df.index[0]).days
    print(f"  Test set: {n_bars:,} bars  "
          f"({test_df.index[0].date()} → {test_df.index[-1].date()})  "
          f"[{n_days} days = {n_days/7:.1f} weeks]")

    z      = test_df["zscore"].values
    prices = test_df["euraud"].values
    rv1m   = test_df["rv_residual_1m"].values

    # Vol tertiles across test set
    p33, p67 = float(np.nanpercentile(rv1m, 33)), float(np.nanpercentile(rv1m, 67))
    vol_tertiles = (p33, p67)
    print(f"  Vol regime thresholds: low < {p33:.2e} ≤ medium < {p67:.2e} ≤ high")

    # -----------------------------------------------------------------------
    # 3. Detect crossing events per threshold
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Signal detection (z-score crossings)")
    print("=" * 60)
    signals_by_threshold: dict[float, np.ndarray] = {}
    for thr in sorted(args.thresholds):
        idx = crossing_events(z, thr)
        signals_by_threshold[thr] = idx
        z_at = np.abs(z[idx])
        print(f"  threshold={thr:.1f}:  {len(idx):,} crossings  "
              f"mean|z|={z_at.mean():.2f}  median|z|={np.median(z_at):.2f}  "
              f"({len(idx)/n_days*7:.1f}/week)")

    # -----------------------------------------------------------------------
    # 4. Profiling (unconstrained) — all thresholds
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Profiling (unconstrained forward returns)")
    print("=" * 60)
    all_profiles: list[pd.DataFrame] = []
    for thr in sorted(args.thresholds):
        sigs = signals_by_threshold[thr]
        df   = profile_returns(
            z, prices, rv1m, sigs,
            delay_bars=u_delay_bars,
            horizon_bars=u_horizon_bars,
            horizon_s=u_horizon_s,
            threshold=thr,
            vol_tertiles=vol_tertiles,
        )
        all_profiles.append(df)
        n_obs = len(df[df["delay_bars"] == 0]) // len(u_horizon_bars)
        print(f"  threshold={thr:.1f}:  {n_obs:,} signal events profiled  "
              f"({len(df):,} (signal, delay, horizon) combinations)")

    prof = pd.concat(all_profiles, ignore_index=True)

    # -----------------------------------------------------------------------
    # 5. Validation: check profiling results make sense
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Validation")
    print("=" * 60)
    for thr in sorted(args.thresholds):
        sub = prof[(prof["threshold"] == thr) &
                   (prof["delay_bars"] == 0) &
                   (prof["horizon_bars"] == 1)]
        if len(sub):
            mg = sub["gross_pips"].mean()
            # Theoretical: at 1 bar (10s), OU halflife ~9 bars → 1-exp(-1/9) ≈ 10.5% closure
            # At mean |z| * residual_std * 10000 pips gap size
            # Expected gross ≈ 0.1 * mean|z| * 0.45 pips ≈ 0.05-0.14 pips
            # Theoretical: at 1 bar (10s), OU halflife ~9 bars → 1-exp(-1/9) ≈ 10.5% closure.
            # At mean |z| ≈ 4, each z-unit ≈ 0.74 pips → expected ~0.31 pips.
            # Crossing events tend to have momentum (z still rising), so actual is
            # typically higher than the naive OU estimate.
            print(f"  threshold={thr:.1f}  delay=0  horizon=1bar(10s):  "
                  f"mean_gross={mg:+.4f} pips  n={len(sub):,}  "
                  f"(OU theory ≈ 0.3 pips; expect higher at crossing events)")
            t_stat, p_val = ttest_1samp(sub["gross_pips"].values, 0)
            print(f"    t-test vs 0: t={t_stat:.2f}  p={p_val:.4f}  "
                  f"{'✓ significant' if p_val < 0.05 else '✗ not significant at 5%'}")

    # -----------------------------------------------------------------------
    # 6. No-overlap simulation for primary cost (first in list)
    # -----------------------------------------------------------------------
    cost_primary = args.costs[0]
    print(f"\n  No-overlap simulation (cost={cost_primary} pips, primary cost assumption):")
    sim_rows = []
    for thr in sorted(args.thresholds):
        sigs = signals_by_threshold[thr]
        for d_b in u_delay_bars:
            for h_b, h_s_val in zip(u_horizon_bars, u_horizon_s):
                tl = simulate_no_overlap(z, prices, sigs, d_b, h_b)
                if len(tl) == 0:
                    continue
                mg   = float(tl["gross_pips"].mean())
                mnet = mg - cost_primary
                tpw  = len(tl) / max(n_days / 7, 1)
                sim_rows.append({
                    "threshold": thr, "delay_bars": d_b, "horizon_bars": h_b,
                    "horizon_s": h_s_val, "n_trades": len(tl),
                    "trades_wk": tpw, "mean_gross": mg, "mean_net": mnet,
                })

    sim_df = pd.DataFrame(sim_rows)
    # Print best combinations
    profitable_sim = sim_df[sim_df["mean_net"] > 0].sort_values("mean_net", ascending=False)
    if len(profitable_sim):
        print(f"  Profitable (no-overlap, cost={cost_primary}):")
        for _, r in profitable_sim.head(10).iterrows():
            print(f"    thr={r['threshold']:.1f}  delay={int(r['delay_bars'])*BAR_S}s  "
                  f"horizon={r['horizon_s']}s  trades/wk={r['trades_wk']:.0f}  "
                  f"mean_net={r['mean_net']:+.3f}")
    else:
        print(f"  No profitable no-overlap configurations at cost={cost_primary}")

    # -----------------------------------------------------------------------
    # 7. Plots
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Generating plots")
    print("=" * 60)

    # a) Mean return vs horizon (all thresholds, delay=0)
    plot_mean_return_vs_horizon(
        prof, args.thresholds, cost=cost_primary,
        path=plots_dir / "horizon_mean_return.png",
    )

    # b) Net return by delay (best threshold)
    best_thr = max(args.thresholds)
    plot_net_return_by_delay(
        prof, u_delay_bars, cost=cost_primary, threshold=best_thr,
        path=plots_dir / "horizon_net_return.png",
    )

    # c) Heatmap: delay × horizon
    h_s_map = dict(zip(u_horizon_bars, u_horizon_s))
    plot_heatmap_delay_horizon(
        prof, u_delay_bars, u_horizon_bars, u_horizon_s,
        cost=cost_primary, threshold=best_thr,
        path=plots_dir / "horizon_heatmap.png",
    )

    # d) Return distributions at key horizons
    show_h_bars = u_horizon_bars[:min(5, len(u_horizon_bars))]
    plot_return_distributions(
        prof, show_h_bars, h_s_map, threshold=best_thr, cost=cost_primary,
        path=plots_dir / "horizon_distributions.png",
    )

    # e) Vol regime analysis
    plot_vol_regime(
        prof, threshold=best_thr, cost=cost_primary,
        path=plots_dir / "horizon_vol_regime.png",
    )

    # -----------------------------------------------------------------------
    # 8. Analysis summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Analysis Summary")
    print("=" * 60)

    # Find optimal horizon (max mean_net at delay=0 in profiling)
    d0_prof = prof[(prof["delay_bars"] == 0) & (prof["threshold"] == best_thr)]
    grp = d0_prof.groupby("horizon_s")["gross_pips"].mean() - cost_primary
    opt_h_s   = int(grp.idxmax())
    opt_net   = float(grp.max())
    opt_h_b   = max(1, round(opt_h_s / BAR_S))

    print(f"\n  Optimal horizon (delay=0, threshold={best_thr:.1f}, cost={cost_primary}):")
    print(f"    {opt_h_s}s ({opt_h_b} bars)  →  mean_net = {opt_net:+.3f} pips")

    # Effect of 10s delay at optimal horizon
    d1_sub = prof[(prof["delay_bars"] == 1) & (prof["threshold"] == best_thr) &
                  (prof["horizon_s"] == opt_h_s)]
    if len(d1_sub):
        mg_d0 = d0_prof[d0_prof["horizon_s"] == opt_h_s]["gross_pips"].mean()
        mg_d1 = d1_sub["gross_pips"].mean()
        pct   = (mg_d1 - mg_d0) / max(abs(mg_d0), 1e-9) * 100
        print(f"\n  10s delay impact at optimal horizon:")
        print(f"    delay=0s  mean_gross={mg_d0:+.4f}  net={mg_d0-cost_primary:+.4f}")
        print(f"    delay=10s mean_gross={mg_d1:+.4f}  net={mg_d1-cost_primary:+.4f}")
        print(f"    Degradation: {pct:+.1f}% of gross edge lost per 10s delay")

    # Frequency at optimal horizon (no-overlap)
    sim_opt = sim_df[
        (sim_df["threshold"] == best_thr) &
        (sim_df["horizon_s"] == opt_h_s) &
        (sim_df["delay_bars"] == 0)
    ]
    if len(sim_opt):
        r = sim_opt.iloc[0]
        print(f"\n  No-overlap frequency at optimal horizon:")
        print(f"    {r['trades_wk']:.0f} trades/week  mean_net(no-overlap)={r['mean_net']:+.3f}")
        print(f"    Note: profiling n={len(d0_prof[d0_prof['horizon_s']==opt_h_s]):,}  "
              f"(incl. overlapping events)")

    # Deployability
    print("\n  Deployability assessment:")
    has_profitable_delay0 = len(sim_df[(sim_df["delay_bars"] == 0) & (sim_df["mean_net"] > 0)]) > 0
    delay1_in_sweep = 1 in u_delay_bars
    has_profitable_delay1 = (
        delay1_in_sweep and
        len(sim_df[(sim_df["delay_bars"] == 1) & (sim_df["mean_net"] > 0)]) > 0
    )

    if has_profitable_delay0 and has_profitable_delay1:
        print(f"    ✓ Edge survives 10s (1-bar) delay at some horizon/threshold combinations")
    elif has_profitable_delay0 and delay1_in_sweep:
        print(f"    ✗ Edge requires same-bar entry (sub-10s latency) — not deployable at 10s delay")
    elif has_profitable_delay0:
        print(f"    ✓ Edge exists at delay=0 (delay=10s not tested in this run)")
    else:
        print(f"    ✗ No profitable configuration found at any delay level")

    # Per-cost summary
    for cost in args.costs:
        d0_net = d0_prof.groupby("horizon_s")["gross_pips"].mean() - cost
        best = float(d0_net.max())
        best_h = int(d0_net.idxmax())
        print(f"\n  cost={cost}:  best mean_net={best:+.3f} at {best_h}s horizon (delay=0)")

    # Vol regime verdict
    print("\n  Vol regime edge concentration:")
    for regime in ["low", "medium", "high"]:
        sub = d0_prof[d0_prof["vol_regime"] == regime]
        if len(sub) == 0:
            continue
        best_h_gross = float(sub.groupby("horizon_s")["gross_pips"].mean().max())
        n_sig = len(sub) // len(u_horizon_bars)
        print(f"    {regime:>6} vol:  {n_sig:,} signals  "
              f"max_mean_gross={best_h_gross:+.4f}")

    # Save CSV
    sim_df.to_csv(output_dir / "horizon_profile_results.csv", index=False)
    prof.to_csv(output_dir / "horizon_profile_raw.csv", index=False)
    print(f"\n  Results saved → {output_dir}/horizon_profile_results.csv")
    print(f"  Raw profile  → {output_dir}/horizon_profile_raw.csv")
    print(f"  Plots saved  → {plots_dir}/horizon_*.png")


if __name__ == "__main__":
    main()
