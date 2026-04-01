"""
Robustness sweep — threshold × persistence × delay × horizon

Tests whether raising the z-score threshold and requiring signal persistence
produces larger per-trade edge that survives ≥1 bar (10s) of execution delay.

Two analyses:

A) Profiling (unconstrained): compute forward returns for every
   (threshold, persistence, delay, horizon) combination without an
   overlap constraint. Shows pure signal quality.

B) No-overlap simulation: one trade at a time. Shows realistic tradeable P&L.

Resolution note (10s/bar):
  persistence [1s, 2s, 5s] all → ceil(N/10) = 1 bar. Only 0-bar and 1-bar distinct.
  delays      [1s, 2s, 5s] all → ceil(N/10) = 1 bar. Only 0-bar and 1-bar distinct.
  Combined unique (persistence_bars, delay_bars): (0,0), (0,1), (1,0), (1,1)

Run from repo root:
    python3 scripts/run_robustness_sweep.py [options]

Options:
    --thresholds FLOAT ...    z-score crossing thresholds (default: 3.0 3.5 4.0 4.5)
    --persistence-s INT ...   min seconds signal must persist (default: 0 1 2 5)
    --delays-s INT ...        entry delay in seconds (default: 0 1 2 5)
    --horizons-s INT ...      forward-look horizons in seconds (default: 10 20 30 60 120 300)
    --costs FLOAT ...         round-trip costs in pips (default: 0.3 0.5 0.7)
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

from triangulation.data import load_pair
from triangulation.residual import build_signal_frame
from triangulation.features import build_feature_frame

plt.style.use("dark_background")

DATA_DIR    = ROOT / "data"
PLOTS_DIR   = ROOT / "outputs" / "plots"
OUTPUTS_DIR = ROOT / "outputs"

VAL_END   = "2025-06-30"
BAR_S     = 10        # native bar duration in seconds
_BARS_30D = 259_200   # 30 days at 10s/bar


# ---------------------------------------------------------------------------
# Signal detection
# ---------------------------------------------------------------------------

def persistent_signals(z: np.ndarray, threshold: float, persistence_bars: int) -> np.ndarray:
    """Return signal indices where |z| crosses above threshold and holds for persistence_bars.

    persistence_bars=0: first bar above threshold (standard crossing event).
    persistence_bars=N: crossing at sig_i AND |z| stays >= threshold through
                        sig_i + persistence_bars → returns sig_i + persistence_bars
                        as the confirmed signal index.

    Direction is determined at the returned (confirmed) signal index.
    """
    n     = len(z)
    above = np.abs(z) >= threshold
    # Crossings: above[t] and not above[t-1]
    crossings = np.where(above[1:] & ~above[:-1])[0] + 1

    if persistence_bars == 0:
        return crossings

    confirmed = []
    for sig_i in crossings:
        end_i = sig_i + persistence_bars
        if end_i >= n:
            continue
        if np.all(above[sig_i : end_i + 1]):
            confirmed.append(end_i)
    return np.array(confirmed, dtype=int)


# ---------------------------------------------------------------------------
# Profiling (unconstrained)
# ---------------------------------------------------------------------------

def profile_returns_ex(
    z: np.ndarray,
    prices: np.ndarray,
    signals: np.ndarray,
    delay_bars: list[int],
    horizon_bars: list[int],
    horizon_s: list[int],
    threshold: float,
    persistence_bars: int,
) -> pd.DataFrame:
    """Compute forward returns for every (signal, delay, horizon) — no overlap constraint.

    Returns DataFrame with columns:
        threshold, persistence_bars, sig_i, sig_z, delay_bars,
        horizon_bars, horizon_s, gross_pips
    """
    n    = len(prices)
    rows = []

    for sig_i in signals:
        direction = float(np.sign(z[sig_i]))
        if direction == 0:
            continue
        sig_z = float(z[sig_i])

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
                    "threshold":       threshold,
                    "persistence_bars": persistence_bars,
                    "sig_i":           sig_i,
                    "sig_z":           sig_z,
                    "delay_bars":      d,
                    "horizon_bars":    h_b,
                    "horizon_s":       h_s,
                    "gross_pips":      gross,
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# No-overlap simulation
# ---------------------------------------------------------------------------

def simulate_no_overlap_ex(
    z: np.ndarray,
    prices: np.ndarray,
    signals: np.ndarray,
    delay_bars: int,
    horizon_bars: int,
) -> pd.DataFrame:
    """One-trade-at-a-time simulation. Entry at signal + delay, exit at entry + horizon."""
    n       = len(prices)
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
        gross  = direction * (prices[entry_i] - prices[exit_i]) * 10_000
        trades.append({
            "entry_i":    entry_i,
            "exit_i":     exit_i,
            "direction":  direction,
            "gross_pips": gross,
        })
        next_free = exit_i + 1

    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(gross_pips: np.ndarray, cost: float, n_weeks: float) -> dict:
    """Summary metrics for a set of trades at a given cost assumption."""
    if len(gross_pips) == 0:
        return {
            "n_trades": 0, "trades_wk": 0.0,
            "mean_gross": float("nan"), "mean_net": float("nan"),
            "sharpe": float("nan"), "win_rate": float("nan"),
        }
    net       = gross_pips - cost
    mean_gross = float(np.mean(gross_pips))
    mean_net   = float(np.mean(net))
    std_net    = float(np.std(net))
    sharpe     = mean_net / (std_net + 1e-10)    # per-trade, not annualised
    win_rate   = float(np.mean(net > 0))
    trades_wk  = len(gross_pips) / max(n_weeks, 1e-6)
    return {
        "n_trades":   len(gross_pips),
        "trades_wk":  trades_wk,
        "mean_gross": mean_gross,
        "mean_net":   mean_net,
        "sharpe":     sharpe,
        "win_rate":   win_rate,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _pd_label(p_b: int, d_b: int) -> str:
    """Short human label for a (persistence_bars, delay_bars) pair."""
    p_s = p_b * BAR_S
    d_s = d_b * BAR_S
    if p_b == 0 and d_b == 0:
        return "persist=0s, delay=0s"
    if p_b == 0:
        return f"persist=0s, delay={d_s}s"
    return f"persist={p_s}s, delay={d_s}s"


def plot_net_vs_threshold(
    prof: pd.DataFrame,
    sim_df: pd.DataFrame,
    thresholds: list[float],
    pd_pairs: list[tuple[int, int]],
    fixed_horizon_s: int,
    cost: float,
    path: Path,
) -> None:
    """Mean net pips and trades/week vs threshold for each (persistence, delay) combo."""
    colors = ["#4fc3f7", "#81c784", "#ffb74d", "#f48fb1"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax_net, ax_freq = axes

    for i, (p_b, d_b) in enumerate(pd_pairs):
        label  = _pd_label(p_b, d_b)
        color  = colors[i % len(colors)]
        nets   = []
        freqs  = []

        for thr in thresholds:
            sub = prof[
                (prof["threshold"] == thr) &
                (prof["persistence_bars"] == p_b) &
                (prof["delay_bars"] == d_b) &
                (prof["horizon_s"] == fixed_horizon_s)
            ]
            mn = float(sub["gross_pips"].mean()) - cost if len(sub) else float("nan")
            nets.append(mn)

            sim_sub = sim_df[
                (sim_df["threshold"] == thr) &
                (sim_df["persistence_bars"] == p_b) &
                (sim_df["delay_bars"] == d_b) &
                (sim_df["horizon_s"] == fixed_horizon_s)
            ]
            tw = float(sim_sub["trades_wk"].iloc[0]) if len(sim_sub) else float("nan")
            freqs.append(tw)

        ax_net.plot(thresholds, nets, marker="o", markersize=5, linewidth=1.8,
                    color=color, label=label)
        ax_freq.plot(thresholds, freqs, marker="o", markersize=5, linewidth=1.8,
                     color=color, label=label)

    for ax in axes:
        ax.set_xlabel("Z-score threshold", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.15)
        ax.set_xticks(thresholds)

    ax_net.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_net.set_ylabel(f"Mean net pips (cost={cost})", fontsize=9)
    ax_net.set_title(f"Net Return vs Threshold  (horizon={fixed_horizon_s}s)", fontsize=10)

    ax_freq.set_ylabel("Trades per week (no-overlap)", fontsize=9)
    ax_freq.set_title(f"Trade Frequency vs Threshold  (horizon={fixed_horizon_s}s)", fontsize=10)
    ax_freq.set_yscale("log")

    fig.suptitle(f"Threshold Sensitivity  (cost={cost} pips)", fontsize=10, y=1.01)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


def plot_net_vs_delay(
    prof: pd.DataFrame,
    thresholds: list[float],
    u_delay_bars: list[int],
    u_persist_bars: list[int],
    fixed_horizon_s: int,
    cost: float,
    path: Path,
) -> None:
    """Mean net pips vs effective delay, for each threshold. Two panels: persistence=[0,1]."""
    colors = ["#4fc3f7", "#81c784", "#ffb74d", "#f48fb1"]
    n_panels = min(len(u_persist_bars), 2)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, p_b in zip(axes, u_persist_bars[:n_panels]):
        for i, thr in enumerate(thresholds):
            color = colors[i % len(colors)]
            nets  = []
            x_labels = []
            for d_b in u_delay_bars:
                sub = prof[
                    (prof["threshold"] == thr) &
                    (prof["persistence_bars"] == p_b) &
                    (prof["delay_bars"] == d_b) &
                    (prof["horizon_s"] == fixed_horizon_s)
                ]
                mn = float(sub["gross_pips"].mean()) - cost if len(sub) else float("nan")
                nets.append(mn)
                total_s = (p_b + d_b) * BAR_S
                x_labels.append(f"{total_s}s\n(p={p_b*BAR_S}s,d={d_b*BAR_S}s)")
            x = range(len(u_delay_bars))
            ax.plot(list(x), nets, marker="o", markersize=5, linewidth=1.8,
                    color=color, label=f"thr={thr:.1f}")

        ax.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xticks(range(len(u_delay_bars)))
        ax.set_xticklabels(x_labels, fontsize=7)
        ax.set_xlabel("Entry delay (total seconds from crossing)", fontsize=9)
        ax.set_ylabel(f"Mean net pips (cost={cost})", fontsize=9)
        ax.set_title(f"persistence={p_b * BAR_S}s", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.15)

    fig.suptitle(
        f"Net Return vs Delay  (horizon={fixed_horizon_s}s, cost={cost} pips)",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


def plot_freq_vs_threshold(
    sim_df: pd.DataFrame,
    thresholds: list[float],
    u_persist_bars: list[int],
    fixed_horizon_s: int,
    path: Path,
) -> None:
    """Trades/week vs threshold for each persistence level (delay=0)."""
    colors = ["#4fc3f7", "#81c784"]
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, p_b in enumerate(u_persist_bars):
        color = colors[i % len(colors)]
        freqs = []
        for thr in thresholds:
            sub = sim_df[
                (sim_df["threshold"] == thr) &
                (sim_df["persistence_bars"] == p_b) &
                (sim_df["delay_bars"] == 0) &
                (sim_df["horizon_s"] == fixed_horizon_s)
            ]
            tw = float(sub["trades_wk"].iloc[0]) if len(sub) else float("nan")
            freqs.append(tw)
        label = f"persistence={p_b * BAR_S}s"
        ax.plot(thresholds, freqs, marker="o", markersize=6, linewidth=1.8,
                color=color, label=label)
        for thr, tw in zip(thresholds, freqs):
            if not np.isnan(tw):
                ax.annotate(f"{tw:.0f}/wk", (thr, tw), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=7, color=color)

    ax.set_xlabel("Z-score threshold", fontsize=9)
    ax.set_ylabel("Trades per week (no-overlap)", fontsize=9)
    ax.set_title(
        f"Trade Frequency vs Threshold  (horizon={fixed_horizon_s}s, delay=0)",
        fontsize=10,
    )
    ax.set_xticks(thresholds)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


def plot_heatmap_grid(
    prof: pd.DataFrame,
    thresholds: list[float],
    u_persist_bars: list[int],
    u_delay_bars: list[int],
    u_horizon_s: list[int],
    costs_to_show: list[float],
    path: Path,
) -> None:
    """2×2 grid of heatmaps: rows=persistence, cols=cost.
    Each cell: (threshold × delay) → mean_net at optimal horizon."""
    n_rows = min(len(u_persist_bars), 2)
    n_cols = min(len(costs_to_show), 2)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    # Ensure 2D axes array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    thr_labels   = [f"z≥{t:.1f}" for t in thresholds]
    delay_labels = [f"d={d*BAR_S}s" for d in u_delay_bars]

    for ri, p_b in enumerate(u_persist_bars[:n_rows]):
        for ci, cost in enumerate(costs_to_show[:n_cols]):
            ax = axes[ri, ci]
            grid = np.full((len(thresholds), len(u_delay_bars)), float("nan"))

            for ti, thr in enumerate(thresholds):
                for di, d_b in enumerate(u_delay_bars):
                    sub = prof[
                        (prof["threshold"] == thr) &
                        (prof["persistence_bars"] == p_b) &
                        (prof["delay_bars"] == d_b)
                    ]
                    if len(sub) == 0:
                        continue
                    # Best horizon: max mean_net across horizons
                    best = float(sub.groupby("horizon_s")["gross_pips"].mean().max()) - cost
                    grid[ti, di] = best

            vmax = max(abs(np.nanmin(grid)), abs(np.nanmax(grid)), 0.01)
            im = ax.imshow(grid, cmap="RdYlGn", vmin=-vmax, vmax=vmax,
                           aspect="auto", interpolation="nearest")
            plt.colorbar(im, ax=ax, label="mean net pips")

            ax.set_xticks(range(len(u_delay_bars)))
            ax.set_xticklabels(delay_labels, fontsize=8)
            ax.set_yticks(range(len(thresholds)))
            ax.set_yticklabels(thr_labels, fontsize=8)
            ax.set_xlabel("Entry delay", fontsize=8)
            ax.set_ylabel("Threshold", fontsize=8)
            ax.set_title(
                f"persist={p_b*BAR_S}s | cost={cost} pips\n(best horizon)",
                fontsize=9,
            )

            for ti in range(grid.shape[0]):
                for di in range(grid.shape[1]):
                    v = grid[ti, di]
                    if not np.isnan(v):
                        marker = "*" if v > 0 else ""
                        txt_color = "black" if abs(v) < vmax * 0.6 else "white"
                        ax.text(di, ti, f"{v:+.3f}{marker}", ha="center", va="center",
                                fontsize=8, color=txt_color)

    fig.suptitle("(Threshold × Delay) → Mean Net Pips at Optimal Horizon", fontsize=11, y=1.01)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness sweep: threshold × persistence × delay × horizon")
    parser.add_argument("--thresholds",    type=float, nargs="+", default=[3.0, 3.5, 4.0, 4.5])
    parser.add_argument("--persistence-s", type=int,   nargs="+", default=[0, 1, 2, 5])
    parser.add_argument("--delays-s",      type=int,   nargs="+", default=[0, 1, 2, 5])
    parser.add_argument("--horizons-s",    type=int,   nargs="+", default=[10, 20, 30, 60, 120, 300])
    parser.add_argument("--costs",         type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--data-dir",      type=str,   default=str(DATA_DIR))
    parser.add_argument("--output-dir",    type=str,   default=str(OUTPUTS_DIR))
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    plots_dir  = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 0. Resolution mapping
    # -----------------------------------------------------------------------
    _divider("Robustness Sweep — EUR/USD/AUD Triangle")
    print(f"\n  Data frequency: {BAR_S} seconds per bar (native resolution)")
    print(f"\n  Resolution note: at {BAR_S}s/bar, sub-{BAR_S}s values collapse to 1 bar.")

    def _unique_bars(seconds_list: list[int], label: str) -> tuple[dict[int, int], list[int]]:
        """Return (bars→first_seconds, unique_bars_sorted). Print mapping."""
        print(f"\n  {label} second → bar mapping:")
        seen: dict[int, int] = {}
        for s in sorted(set(seconds_list)):
            b = math.ceil(s / BAR_S)
            note = f"  [same as {seen[b]}s]" if b in seen and seen[b] != s else ""
            print(f"    {s:>5}s → {b} bar(s) = {b*BAR_S}s{note}")
            if b not in seen:
                seen[b] = s
        return seen, sorted(seen.keys())

    p_map, u_persist_bars = _unique_bars(args.persistence_s, "Persistence")
    d_map, u_delay_bars   = _unique_bars(args.delays_s,      "Delay")
    h_map, u_horizon_bars = _unique_bars(args.horizons_s,    "Horizon")
    u_horizon_s           = [h_map[b] for b in u_horizon_bars]

    print(f"\n  Unique persistence bars: {u_persist_bars}  = {[b*BAR_S for b in u_persist_bars]}s")
    print(f"  Unique delay bars:       {u_delay_bars}  = {[b*BAR_S for b in u_delay_bars]}s")
    print(f"  Unique horizon bars:     {u_horizon_bars}  = {[b*BAR_S for b in u_horizon_bars]}s")
    print(f"\n  Unique (persistence, delay) pairs:")
    pd_pairs = [(p_b, d_b) for p_b in u_persist_bars for d_b in u_delay_bars]
    for p_b, d_b in pd_pairs:
        total_s = (p_b + d_b) * BAR_S
        print(f"    persist={p_b*BAR_S}s + delay={d_b*BAR_S}s → entry {total_s}s after crossing")

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    _divider("Loading data + rebuilding features")
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

    # -----------------------------------------------------------------------
    # 2. Filter to test set
    # -----------------------------------------------------------------------
    val_end_ts = pd.Timestamp(VAL_END, tz="UTC")
    test_df    = feat[feat.index > val_end_ts].copy()
    n_bars     = len(test_df)
    n_days     = (test_df.index[-1] - test_df.index[0]).days
    n_weeks    = n_days / 7
    print(f"  Test set: {n_bars:,} bars  "
          f"({test_df.index[0].date()} → {test_df.index[-1].date()})  "
          f"[{n_days} days = {n_weeks:.1f} weeks]")

    z      = test_df["zscore"].values
    prices = test_df["euraud"].values

    # -----------------------------------------------------------------------
    # 3. Detect signals per (threshold, persistence)
    # -----------------------------------------------------------------------
    _divider("Signal detection (threshold × persistence)")
    signals: dict[tuple[float, int], np.ndarray] = {}
    for thr in sorted(args.thresholds):
        for p_b in u_persist_bars:
            idx = persistent_signals(z, thr, p_b)
            signals[(thr, p_b)] = idx
            if len(idx):
                z_abs = np.abs(z[idx])
                print(f"  thr={thr:.1f}  persist={p_b*BAR_S}s:  "
                      f"{len(idx):,} signals  mean|z|={z_abs.mean():.2f}  "
                      f"median|z|={np.median(z_abs):.2f}  ({len(idx)/n_weeks:.0f}/wk)")
            else:
                print(f"  thr={thr:.1f}  persist={p_b*BAR_S}s:  0 signals")

    # -----------------------------------------------------------------------
    # 4. Profiling (unconstrained)
    # -----------------------------------------------------------------------
    _divider("Profiling (unconstrained forward returns)")
    all_profiles: list[pd.DataFrame] = []
    total = len(args.thresholds) * len(u_persist_bars)
    n_done = 0
    for thr in sorted(args.thresholds):
        for p_b in u_persist_bars:
            n_done += 1
            sigs = signals[(thr, p_b)]
            print(f"  [{n_done}/{total}] thr={thr:.1f}  persist={p_b*BAR_S}s  "
                  f"n_signals={len(sigs):,} ...", end=" ", flush=True)
            df = profile_returns_ex(
                z, prices, sigs,
                delay_bars=u_delay_bars,
                horizon_bars=u_horizon_bars,
                horizon_s=u_horizon_s,
                threshold=thr,
                persistence_bars=p_b,
            )
            all_profiles.append(df)
            print(f"rows={len(df):,}")

    prof = pd.concat(all_profiles, ignore_index=True) if all_profiles else pd.DataFrame()

    # -----------------------------------------------------------------------
    # 5. No-overlap simulation
    # -----------------------------------------------------------------------
    _divider("No-overlap simulation")
    cost_primary = args.costs[0]
    sim_rows: list[dict] = []
    total_sims = len(args.thresholds) * len(u_persist_bars) * len(u_delay_bars) * len(u_horizon_bars)
    n_sim = 0
    for thr in sorted(args.thresholds):
        for p_b in u_persist_bars:
            sigs = signals[(thr, p_b)]
            for d_b in u_delay_bars:
                for h_b, h_s in zip(u_horizon_bars, u_horizon_s):
                    n_sim += 1
                    tl = simulate_no_overlap_ex(z, prices, sigs, d_b, h_b)
                    for cost in args.costs:
                        m = compute_metrics(tl["gross_pips"].values if len(tl) else np.array([]), cost, n_weeks)
                        sim_rows.append({
                            "threshold":       thr,
                            "persistence_bars": p_b,
                            "persistence_s":   p_b * BAR_S,
                            "delay_bars":      d_b,
                            "delay_s":         d_b * BAR_S,
                            "horizon_bars":    h_b,
                            "horizon_s":       h_s,
                            "cost":            cost,
                            **m,
                        })

    sim_df = pd.DataFrame(sim_rows)

    # Print top profitable configs at primary cost
    print(f"\n  Profitable configs (no-overlap, cost={cost_primary}):")
    profitable = sim_df[(sim_df["cost"] == cost_primary) & (sim_df["mean_net"] > 0)] \
        .sort_values("mean_net", ascending=False)
    if len(profitable):
        for _, r in profitable.head(15).iterrows():
            print(f"    thr={r['threshold']:.1f}  persist={r['persistence_s']:.0f}s  "
                  f"delay={r['delay_s']:.0f}s  horizon={r['horizon_s']}s  "
                  f"trades/wk={r['trades_wk']:.0f}  "
                  f"mean_net={r['mean_net']:+.3f}  sharpe={r['sharpe']:+.3f}  "
                  f"win_rate={r['win_rate']:.1%}")
    else:
        print(f"    None at cost={cost_primary}")

    # -----------------------------------------------------------------------
    # 6. Plots
    # -----------------------------------------------------------------------
    _divider("Generating plots")

    # Fixed horizon for threshold/delay plots: 60s if available, else nearest
    fixed_h_s = 60 if 60 in u_horizon_s else u_horizon_s[min(3, len(u_horizon_s)-1)]

    # a) Net vs threshold (dual panel)
    plot_net_vs_threshold(
        prof, sim_df, sorted(args.thresholds), pd_pairs,
        fixed_horizon_s=fixed_h_s, cost=cost_primary,
        path=plots_dir / "robustness_net_vs_threshold.png",
    )

    # b) Net vs delay (two panels: persistence=[0, 1])
    plot_net_vs_delay(
        prof, sorted(args.thresholds), u_delay_bars, u_persist_bars,
        fixed_horizon_s=fixed_h_s, cost=cost_primary,
        path=plots_dir / "robustness_net_vs_delay.png",
    )

    # c) Frequency vs threshold
    plot_freq_vs_threshold(
        sim_df, sorted(args.thresholds), u_persist_bars,
        fixed_horizon_s=fixed_h_s,
        path=plots_dir / "robustness_freq_vs_threshold.png",
    )

    # d) Heatmap grid (2×2: persistence × cost)
    costs_for_heatmap = args.costs[:2]
    plot_heatmap_grid(
        prof, sorted(args.thresholds), u_persist_bars[:2], u_delay_bars,
        u_horizon_s, costs_for_heatmap,
        path=plots_dir / "robustness_heatmap.png",
    )

    # -----------------------------------------------------------------------
    # 7. Analysis summary
    # -----------------------------------------------------------------------
    _divider("Analysis Summary")

    # Per (persistence, delay) best config
    print("\n  Best (threshold, horizon) per (persistence, delay) at cost=0.3:")
    for p_b, d_b in pd_pairs:
        sub = sim_df[
            (sim_df["cost"] == 0.3) &
            (sim_df["persistence_bars"] == p_b) &
            (sim_df["delay_bars"] == d_b)
        ]
        if len(sub) == 0:
            continue
        best = sub.loc[sub["mean_net"].idxmax()]
        tag = "✓" if best["mean_net"] > 0 else "✗"
        print(f"    {tag} persist={p_b*BAR_S}s delay={d_b*BAR_S}s → "
              f"thr={best['threshold']:.1f} horizon={best['horizon_s']}s "
              f"mean_net={best['mean_net']:+.3f} trades/wk={best['trades_wk']:.0f}")

    # Deployability at delay≥1 bar
    delay1_configs = sim_df[
        (sim_df["delay_bars"] >= 1) &
        (sim_df["cost"] == 0.3) &
        (sim_df["mean_net"] > 0)
    ]
    print("\n  Deployability (delay ≥ 10s, cost=0.3):")
    if len(delay1_configs):
        best = delay1_configs.loc[delay1_configs["mean_net"].idxmax()]
        print(f"    ✓ Profitable at ≥10s delay: "
              f"thr={best['threshold']:.1f} persist={best['persistence_s']:.0f}s "
              f"delay={best['delay_s']:.0f}s horizon={best['horizon_s']}s "
              f"mean_net={best['mean_net']:+.3f}")
    else:
        print(f"    ✗ No profitable configuration at delay ≥ 10s (cost=0.3)")

    delay1_p1_configs = sim_df[
        (sim_df["delay_bars"] >= 1) &
        (sim_df["persistence_bars"] >= 1) &
        (sim_df["cost"] == 0.3) &
        (sim_df["mean_net"] > 0)
    ]
    print("\n  Deployability (persistence ≥ 10s + delay ≥ 10s = 20s total, cost=0.3):")
    if len(delay1_p1_configs):
        best = delay1_p1_configs.loc[delay1_p1_configs["mean_net"].idxmax()]
        print(f"    ✓ Profitable: thr={best['threshold']:.1f} "
              f"mean_net={best['mean_net']:+.3f}")
    else:
        print(f"    ✗ No profitable configuration at 20s total latency (cost=0.3)")

    # Frequency vs edge table
    print(f"\n  Frequency / edge trade-off at optimal horizon, delay=0, persist=0, cost=0.3:")
    print(f"  {'threshold':>10}  {'signals/wk':>12}  {'trades/wk':>12}  "
          f"{'mean_gross':>12}  {'mean_net(0.3)':>14}  {'win_rate':>9}")
    for thr in sorted(args.thresholds):
        sigs = signals[(thr, 0)]
        sig_wk = len(sigs) / n_weeks
        # Best no-overlap config at delay=0, persist=0, cost=0.3
        sub = sim_df[
            (sim_df["threshold"] == thr) &
            (sim_df["persistence_bars"] == 0) &
            (sim_df["delay_bars"] == 0) &
            (sim_df["cost"] == 0.3)
        ]
        if len(sub) == 0:
            continue
        best = sub.loc[sub["mean_net"].idxmax()]
        print(f"  {thr:>10.1f}  {sig_wk:>12.0f}  {best['trades_wk']:>12.0f}  "
              f"{best['mean_gross']:>+12.3f}  {best['mean_net']:>+14.3f}  "
              f"{best['win_rate']:>9.1%}")

    # Best deployable config across all params and costs
    print("\n  Best deployable configurations (mean_net > 0, any cost):")
    best_overall = sim_df[sim_df["mean_net"] > 0].sort_values("mean_net", ascending=False)
    if len(best_overall):
        for _, r in best_overall.head(5).iterrows():
            total_s = (r["persistence_bars"] + r["delay_bars"]) * BAR_S
            print(f"    thr={r['threshold']:.1f}  persist={r['persistence_s']:.0f}s  "
                  f"delay={r['delay_s']:.0f}s  total_lag={total_s}s  "
                  f"horizon={r['horizon_s']}s  cost={r['cost']}  "
                  f"mean_net={r['mean_net']:+.3f}  trades/wk={r['trades_wk']:.0f}")
    else:
        print("    None found")

    # -----------------------------------------------------------------------
    # 8. Save results
    # -----------------------------------------------------------------------
    results_path = output_dir / "robustness_results.csv"
    sim_df.to_csv(results_path, index=False)
    print(f"\n  Results saved → {results_path}")
    print(f"  Plots saved   → {plots_dir}/robustness_*.png")


if __name__ == "__main__":
    main()
