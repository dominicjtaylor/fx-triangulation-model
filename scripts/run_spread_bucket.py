"""
Spread-Bucketed Edge Analysis

Tests whether mean-reversion edge (mid-price directional value) holds constant or
improves in low-spread / high-liquidity regimes. Answers: does a tighter spread
filter unlock profitability, or does it merely reduce costs proportionally?

Post-event reversion strategy: same as run_post_event.py
  - Event: z-score onset at threshold
  - Entry: first bar with spread <= 1.5p (within 60 min)
  - Direction: reversion (-sign(z_onset))
  - Exit: time-based at fixed horizon

Spread buckets (at entry bar):
  A: < 0.6 pips
  B: 0.6 – 0.8 pips
  C: 0.8 – 1.0 pips
  D: > 1.0 pips

Usage:
  python3 scripts/run_spread_bucket.py
"""

from __future__ import annotations

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

BAR_S        = 10
SPREAD_ENTRY = 1.5    # pips — max spread allowed at entry
SPREAD_ILLIQ = 2.0    # pips — illiquid onset threshold
MAX_WAIT     = 360    # bars = 60 min

THRESHOLDS = [3.0, 3.5, 4.0, 4.5]
HORIZONS   = {"30s": 3, "120s": 12, "300s": 30}

BUCKETS = [
    ("<0.6p",    0.0, 0.6),
    ("0.6–0.8p", 0.6, 0.8),
    ("0.8–1.0p", 0.8, 1.0),
    (">1.0p",    1.0, 9.9),
]


# ---------------------------------------------------------------------------
# Core simulation (records mid_move in addition to gross)
# ---------------------------------------------------------------------------

def detect_onsets(z: np.ndarray, threshold: float) -> np.ndarray:
    above      = np.abs(z) >= threshold
    onset_mask = np.zeros(len(z), dtype=bool)
    onset_mask[1:] = above[1:] & ~above[:-1]
    return np.where(onset_mask)[0]


def simulate(
    z: np.ndarray,
    mid: np.ndarray,
    bid: np.ndarray,
    ask: np.ndarray,
    spread: np.ndarray,
    timestamps: pd.DatetimeIndex,
    threshold: float,
    horizon_bars: int,
) -> pd.DataFrame:
    """Reversion-only no-overlap simulation with mid_move recorded."""
    onsets    = detect_onsets(z, threshold)
    n         = len(z)
    trades    = []
    next_free = 0

    for oi in onsets:
        # Find first liquid bar from onset
        end    = min(oi + MAX_WAIT + 1, n)
        hits   = np.where(spread[oi:end] <= SPREAD_ENTRY)[0]
        if len(hits) == 0:
            continue
        ei = oi + int(hits[0])

        if ei < next_free:
            continue
        exit_i = ei + horizon_bars
        if exit_i >= n:
            continue

        z_onset   = float(z[oi])
        direction = -float(np.sign(z_onset))
        if direction == 0:
            continue

        entry_exec = ask[ei] if direction > 0 else bid[ei]
        exit_exec  = bid[exit_i] if direction > 0 else ask[exit_i]
        gross      = direction * (exit_exec - entry_exec) * 10_000
        mid_move   = direction * (mid[exit_i] - mid[ei]) * 10_000
        spread_e   = float(spread[ei])

        trades.append({
            "ts":        timestamps[ei],
            "z_onset":   z_onset,
            "spread_e":  spread_e,
            "gross":     gross,
            "mid_move":  mid_move,
        })
        next_free = exit_i + 1

    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------

def bucket_analysis(tl: pd.DataFrame) -> pd.DataFrame:
    """Compute per-bucket stats from a trade log."""
    rows = []
    for label, lo, hi in BUCKETS:
        mask = (tl["spread_e"] > lo) & (tl["spread_e"] <= hi)
        grp  = tl[mask]
        if len(grp) == 0:
            rows.append({
                "bucket": label, "n": 0,
                "mean_mid": float("nan"), "mean_net": float("nan"),
                "hit_rate": float("nan"),
            })
            continue
        rows.append({
            "bucket":   label,
            "n":        len(grp),
            "mean_mid": float(grp["mid_move"].mean()),
            "mean_net": float(grp["gross"].mean()),
            "hit_rate": float((grp["gross"] > 0).mean()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_bucket_analysis(
    results: dict,   # {(threshold, horizon_label): bucket_df}
    thresholds: list[float],
    horizon_label: str,
    path: Path,
) -> None:
    """mean_mid and mean_net vs spread bucket — one panel per threshold."""
    n_thr     = len(thresholds)
    fig, axes = plt.subplots(2, n_thr, figsize=(4.5 * n_thr, 9), sharey="row")
    if n_thr == 1:
        axes = axes.reshape(2, 1)

    bucket_labels = [b[0] for b in BUCKETS]
    xs            = np.arange(len(bucket_labels))

    for col, thr in enumerate(thresholds):
        key = (thr, horizon_label)
        bdf = results.get(key)
        if bdf is None:
            continue

        # Top row: mean_mid (pre-cost directional value)
        ax_top = axes[0, col]
        colors = ["#81c784" if v > 0 else "#f48fb1" for v in bdf["mean_mid"].fillna(0)]
        bars = ax_top.bar(xs, bdf["mean_mid"].fillna(0), color=colors, alpha=0.85,
                          edgecolor="black", linewidth=0.4)
        ax_top.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.6)
        # Annotate with n
        for i, (b, row) in enumerate(bdf.iterrows()):
            if row["n"] > 0:
                ax_top.text(i, 0.02, f"n={row['n']:,}", ha="center", va="bottom",
                            fontsize=6, color="white", rotation=90)
        ax_top.set_xticks(xs)
        ax_top.set_xticklabels(bucket_labels, rotation=20, fontsize=7)
        ax_top.set_title(f"z≥{thr}", fontsize=9)
        ax_top.set_ylabel("Mean mid-price move (pips)" if col == 0 else "", fontsize=8)
        ax_top.grid(alpha=0.1, axis="y")
        if col == 0:
            ax_top.set_title(f"z≥{thr}\nPRE-COST (mid price)", fontsize=9)

        # Bottom row: mean_net (after bid/ask)
        ax_bot = axes[1, col]
        colors_net = ["#81c784" if v > 0 else "#f48fb1"
                      for v in bdf["mean_net"].fillna(0)]
        ax_bot.bar(xs, bdf["mean_net"].fillna(0), color=colors_net, alpha=0.85,
                   edgecolor="black", linewidth=0.4)
        ax_bot.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.6)
        ax_bot.set_xticks(xs)
        ax_bot.set_xticklabels(bucket_labels, rotation=20, fontsize=7)
        ax_bot.set_ylabel("Mean net pips (bid/ask)" if col == 0 else "", fontsize=8)
        ax_bot.grid(alpha=0.1, axis="y")
        if col == 0:
            ax_bot.set_title("POST-COST (bid/ask)", fontsize=9)

    fig.suptitle(
        f"Spread-Bucketed Edge Analysis  |  Horizon={horizon_label}  |  Reversion",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


def plot_mid_vs_net_overlay(
    results: dict,
    thresholds: list[float],
    horizon_label: str,
    path: Path,
) -> None:
    """Overlay mean_mid and mean_net on same axes for each threshold."""
    fig, axes = plt.subplots(1, len(thresholds), figsize=(4.5 * len(thresholds), 5),
                             sharey=True)
    if len(thresholds) == 1:
        axes = [axes]

    bucket_labels = [b[0] for b in BUCKETS]
    xs            = np.arange(len(bucket_labels))
    width         = 0.35

    for ax, thr in zip(axes, thresholds):
        key = (thr, horizon_label)
        bdf = results.get(key)
        if bdf is None:
            continue

        mid_vals = bdf["mean_mid"].fillna(0).values
        net_vals = bdf["mean_net"].fillna(0).values

        ax.bar(xs - width / 2, mid_vals, width, label="Pre-cost (mid)",
               color="#4fc3f7", alpha=0.8, edgecolor="black", linewidth=0.3)
        ax.bar(xs + width / 2, net_vals, width, label="Post-cost (bid/ask)",
               color="#ffb74d", alpha=0.8, edgecolor="black", linewidth=0.3)
        ax.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels(bucket_labels, rotation=20, fontsize=7)
        ax.set_title(f"z≥{thr}", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Mean pips", fontsize=8)
            ax.legend(fontsize=7)
        ax.grid(alpha=0.12, axis="y")

        # Annotate gap (spread cost) for <0.6p bucket
        if not np.isnan(mid_vals[0]) and not np.isnan(net_vals[0]):
            gap = mid_vals[0] - net_vals[0]
            ax.annotate(
                f"spread\ncost\n{gap:.2f}p",
                xy=(xs[0] + width / 2, net_vals[0]),
                xytext=(xs[0] + 0.5, max(mid_vals[0], net_vals[0]) + 0.1),
                fontsize=6, color="#ce93d8",
                arrowprops=dict(arrowstyle="->", color="#ce93d8", lw=0.8),
            )

    fig.suptitle(
        f"Pre- vs Post-Cost Edge by Spread Bucket  |  Horizon={horizon_label}",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


def plot_profitability_heatmap(
    summary: pd.DataFrame,
    path: Path,
) -> None:
    """Heatmap of mean_net across (threshold × spread_bucket) for each horizon."""
    horizons = summary["horizon"].unique()
    n_h      = len(horizons)
    fig, axes = plt.subplots(1, n_h, figsize=(5 * n_h, 4))
    if n_h == 1:
        axes = [axes]

    bucket_labels = [b[0] for b in BUCKETS]

    for ax, h in zip(axes, horizons):
        sub  = summary[summary["horizon"] == h]
        grid = np.full((len(THRESHOLDS), len(BUCKETS)), np.nan)

        for i, thr in enumerate(THRESHOLDS):
            for j, (blabel, _, _) in enumerate(BUCKETS):
                row = sub[(sub["threshold"] == thr) & (sub["bucket"] == blabel)]
                if len(row) > 0:
                    grid[i, j] = row["mean_net"].values[0]

        vmax = max(abs(np.nanmin(grid)), abs(np.nanmax(grid)), 0.1)
        im   = ax.imshow(grid, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, label="Mean net pips")

        ax.set_xticks(range(len(BUCKETS)))
        ax.set_xticklabels(bucket_labels, rotation=20, fontsize=7)
        ax.set_yticks(range(len(THRESHOLDS)))
        ax.set_yticklabels([f"z≥{t}" for t in THRESHOLDS], fontsize=8)
        ax.set_title(f"Horizon={h}", fontsize=9)

        # Annotate cells with value + n
        for i in range(len(THRESHOLDS)):
            for j in range(len(BUCKETS)):
                if not np.isnan(grid[i, j]):
                    row = sub[(sub["threshold"] == THRESHOLDS[i]) &
                              (sub["bucket"] == bucket_labels[j])]
                    n   = int(row["n"].values[0]) if len(row) > 0 else 0
                    ax.text(j, i, f"{grid[i,j]:+.2f}\nn={n:,}",
                            ha="center", va="center", fontsize=5.5,
                            color="black")

    fig.suptitle("Mean Net Pips Heatmap  (threshold × spread bucket × horizon)",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # Load data
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

    total_weeks = (timestamps[-1] - timestamps[0]).days / 7
    print(f"  {len(sig):,} bars  |  {timestamps[0].date()} → {timestamps[-1].date()}")
    print(f"  Spread: median={np.median(spread):.3f}p  "
          f"p25={np.percentile(spread,25):.3f}p  "
          f"p75={np.percentile(spread,75):.3f}p")
    sp_buckets = [("<0.6p", spread < 0.6),
                  ("0.6-0.8p", (spread >= 0.6) & (spread < 0.8)),
                  ("0.8-1.0p", (spread >= 0.8) & (spread < 1.0)),
                  (">1.0p",    spread >= 1.0)]
    for label, mask in sp_buckets:
        pct = 100 * mask.sum() / len(spread)
        print(f"  {label}: {mask.sum():,} bars ({pct:.1f}%)")

    # ------------------------------------------------------------------
    # Run simulations
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Running simulations (reversion, all threshold × horizon)")
    print("=" * 60)

    results: dict = {}    # (threshold, horizon_label) -> bucket_df
    all_rows = []

    for thr in THRESHOLDS:
        for h_label, h_bars in HORIZONS.items():
            tl = simulate(z, mid, bid, ask, spread, timestamps, thr, h_bars)
            print(f"  z≥{thr}  h={h_label}: {len(tl):,} trades "
                  f"({len(tl)/total_weeks:.1f}/wk)")

            bdf = bucket_analysis(tl)
            bdf["threshold"] = thr
            bdf["horizon"]   = h_label
            results[(thr, h_label)] = bdf
            all_rows.append(bdf)

    summary = pd.concat(all_rows, ignore_index=True)

    # ------------------------------------------------------------------
    # Print tables
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("SPREAD-BUCKETED RESULTS")
    print("=" * 60)
    hdr = (f"{'thr':>5} {'horizon':>7} {'bucket':>11} "
           f"{'n':>7} {'mean_mid':>10} {'mean_net':>10} {'hit_rate':>9}")
    for h_label in HORIZONS:
        print(f"\n  --- Horizon: {h_label} ---")
        print(f"  {hdr}")
        print(f"  {'-'*len(hdr)}")
        for thr in THRESHOLDS:
            bdf = results[(thr, h_label)]
            for _, row in bdf.iterrows():
                if row["n"] == 0:
                    continue
                profit_flag = " ✓" if row["mean_net"] > 0 else ""
                print(
                    f"  {thr:>5.1f} {h_label:>7} {row['bucket']:>11} "
                    f"{int(row['n']):>7,} {row['mean_mid']:>10.3f} "
                    f"{row['mean_net']:>10.3f}{profit_flag} {row['hit_rate']:>9.1%}"
                )

    # ------------------------------------------------------------------
    # Key ratios
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("SIGNAL RATIO ANALYSIS (mid_move / spread_cost)")
    print("(ratio > 1.0 = profitable; shows whether edge scales with spread)")
    print("=" * 60)
    bucket_mid_spreads = {"<0.6p": 0.45, "0.6–0.8p": 0.70,
                          "0.8–1.0p": 0.90, ">1.0p": 1.25}
    for h_label in HORIZONS:
        print(f"\n  Horizon={h_label}:")
        for thr in THRESHOLDS:
            bdf   = results[(thr, h_label)]
            parts = []
            for _, row in bdf.iterrows():
                if row["n"] < 10:
                    continue
                approx_spread = bucket_mid_spreads.get(row["bucket"], 0.9)
                ratio = row["mean_mid"] / approx_spread if approx_spread > 0 else float("nan")
                parts.append(f"{row['bucket']}={ratio:.2f}")
            print(f"  z≥{thr}: " + "  ".join(parts))

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)

    profitable = summary[(summary["mean_net"] > 0) & (summary["n"] > 50)]
    if len(profitable) > 0:
        print(f"\n  ✓ {len(profitable)} profitable (threshold × horizon × bucket) configs:")
        for _, r in profitable.sort_values("mean_net", ascending=False).head(8).iterrows():
            print(f"    z≥{r['threshold']}  h={r['horizon']}  {r['bucket']:>11}  "
                  f"mean_net={r['mean_net']:+.3f}p  "
                  f"hit={r['hit_rate']:.1%}  n={int(r['n']):,}")
    else:
        print("  ✗ No profitable bucket found (n>50)")

    # Check if mid_move is monotonically improving with lower spread
    print("\n  Does mid_move improve at lower spreads?")
    for thr in THRESHOLDS:
        bdf  = results[(thr, "120s")]
        mids = bdf.set_index("bucket")["mean_mid"]
        a    = mids.get("<0.6p", np.nan)
        b    = mids.get("0.6–0.8p", np.nan)
        c    = mids.get("0.8–1.0p", np.nan)
        d    = mids.get(">1.0p", np.nan)
        trend = "decreasing ↓ (signal scales with cost)" if (
            not np.isnan(a) and not np.isnan(d) and a < d
        ) else "increasing ↑ (signal stronger at low spread)" if (
            not np.isnan(a) and not np.isnan(d) and a > d
        ) else "mixed"
        print(f"  z≥{thr}  h=120s:  "
              f"<0.6p={a:+.3f}  0.6-0.8p={b:+.3f}  "
              f"0.8-1.0p={c:+.3f}  >1.0p={d:+.3f}  → {trend}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Generating plots...")
    print("=" * 60)

    # Main bucketed bar charts for each horizon
    for h_label in HORIZONS:
        plot_bucket_analysis(
            results, THRESHOLDS, h_label,
            PLOT_DIR / f"spread_bucket_{h_label}.png",
        )
        plot_mid_vs_net_overlay(
            results, THRESHOLDS, h_label,
            PLOT_DIR / f"spread_bucket_overlay_{h_label}.png",
        )

    # Heatmap
    plot_profitability_heatmap(summary, PLOT_DIR / "spread_bucket_heatmap.png")

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    out_csv = OUT_DIR / "spread_bucket_results.csv"
    summary.to_csv(out_csv, index=False)
    print(f"\n  Results saved → {out_csv}")


if __name__ == "__main__":
    main()
