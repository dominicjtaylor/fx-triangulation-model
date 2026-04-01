"""
Execution realism stress-test — EUR/USD/AUD Triangle

Tests whether the observed gross edge survives realistic execution conditions:
  * Stochastic spread: each trade's cost ~ N(costs_pips, spread_std), clipped at 0
  * Slippage:          additional per-trade cost ~ N(0, slippage_std), mean-zero
  * Execution delay:   entry price shifts by `delay` bars

Design: simulate() is called once per delay value (with costs_pips=0) to get
gross_pips. Stochastic spread and slippage are applied post-hoc via numpy so the
expensive simulation runs only 3 times (one per delay), not 3×3×4=36 times.

Sweep:
  cost_pips    ∈ {0.3, 0.5, 0.8}
  slippage_std ∈ {0.0, 0.1, 0.2, 0.3}
  delay        ∈ {0, 1, 2}           bars (10s each = 0s, 10s, 20s)

Fixed: threshold=3.0, spread_std=0.15, seed=42

Run from repo root:
    python3 scripts/run_realism.py [options]
"""

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from triangulation.data import load_pair
from triangulation.residual import build_signal_frame
from triangulation.features import build_feature_frame
from triangulation.labels import compute_future_zscore_targets
from triangulation.backtest import simulate, daily_sharpe

plt.style.use("dark_background")

DATA_DIR    = ROOT / "data"
MODELS_DIR  = ROOT / "outputs" / "models"
PLOTS_DIR   = ROOT / "outputs" / "plots"
OUTPUTS_DIR = ROOT / "outputs"

VAL_END     = "2025-06-30"
ENTRY_Z_MIN = 1.5
HORIZON     = 60
_BARS_30D   = 259_200


def divider(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Post-hoc cost application
# ---------------------------------------------------------------------------

def apply_costs(
    trade_log_gross: pd.DataFrame,
    test_index: pd.DatetimeIndex,
    costs_pips: float,
    spread_std: float,
    slippage_std: float,
    rng: np.random.Generator,
) -> dict:
    """Apply stochastic spread + slippage to a gross_pips trade_log.

    spread  ~ max(0, N(costs_pips, spread_std)) per trade
    slippage ~ N(0, slippage_std) per trade  [mean-zero: adds variance, not bias]

    Returns dict: mean_net, std_net, n_trades, win_rate, net_pips_arr, sharpe
    """
    n = len(trade_log_gross)
    if n == 0:
        return {"mean_net": float("nan"), "std_net": float("nan"),
                "n_trades": 0, "win_rate": float("nan"),
                "net_pips_arr": np.array([]), "sharpe": float("nan")}

    gross    = trade_log_gross["gross_pips"].values
    spread   = np.maximum(0.0, rng.normal(costs_pips, spread_std, n))
    slippage = rng.normal(0.0, slippage_std, n) if slippage_std > 0 else np.zeros(n)
    net      = gross - spread - slippage

    # Build minimal trade_log preserving exit_time timezone for daily_sharpe
    tl = trade_log_gross[["exit_time"]].copy()
    tl["net_pips"] = net
    sharpe = daily_sharpe(tl, test_index)

    return {
        "mean_net":     float(net.mean()),
        "std_net":      float(net.std()),
        "n_trades":     n,
        "win_rate":     float((net > 0).mean()),
        "net_pips_arr": net,
        "sharpe":       sharpe,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_heatmap(
    results: list[dict],
    costs: list[float],
    slippages: list[float],
    path: Path,
) -> None:
    """Heatmap of mean_net vs (costs_pips, slippage_std) at delay=0."""
    delay0 = [r for r in results if r["delay"] == 0]

    mean_grid = np.full((len(costs), len(slippages)), float("nan"))
    std_grid  = np.full((len(costs), len(slippages)), float("nan"))
    for r in delay0:
        ci = costs.index(r["costs_pips"])
        si = slippages.index(r["slippage_std"])
        mean_grid[ci, si] = r["mean_net"]
        std_grid[ci, si]  = r["std_net"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for ax, grid, title_suffix in zip(
        axes,
        [mean_grid, std_grid],
        ["Mean Net P&L (pips/trade)", "Std Dev of Net P&L (pips/trade)"],
    ):
        vmax = max(abs(np.nanmin(grid)), abs(np.nanmax(grid)))
        vmin = -vmax if title_suffix.startswith("Mean") else 0

        cmap = "RdYlGn" if title_suffix.startswith("Mean") else "YlOrRd"
        im = ax.imshow(
            grid, cmap=cmap, vmin=vmin, vmax=vmax,
            aspect="auto", interpolation="nearest",
        )
        plt.colorbar(im, ax=ax, label="pips")

        ax.set_xticks(range(len(slippages)))
        ax.set_xticklabels([str(s) for s in slippages], fontsize=9)
        ax.set_yticks(range(len(costs)))
        ax.set_yticklabels([str(c) for c in costs], fontsize=9)
        ax.set_xlabel("slippage_std (pips)", fontsize=9)
        ax.set_ylabel("costs_pips", fontsize=9)
        ax.set_title(title_suffix, fontsize=10)

        for i in range(len(costs)):
            for j in range(len(slippages)):
                val = grid[i, j]
                if not np.isnan(val):
                    marker = "*" if (title_suffix.startswith("Mean") and val > 0) else ""
                    ax.text(j, i, f"{val:+.3f}{marker}", ha="center", va="center",
                            fontsize=8, color="black" if abs(val) < vmax * 0.6 else "white")

    fig.suptitle(
        "Execution Realism: Cost × Slippage (delay=0, threshold=3.0)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")


def plot_delay_lines(
    results: list[dict],
    costs: list[float],
    delays: list[int],
    slippage_ref: float,
    path: Path,
) -> None:
    """Line plot: mean_net vs delay (bars) for each cost level."""
    slip_results = [r for r in results if r["slippage_std"] == slippage_ref]

    colors = ["#4fc3f7", "#81c784", "#ffb74d", "#f48fb1"]
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, cost in enumerate(sorted(costs)):
        cost_rows = sorted(
            [r for r in slip_results if r["costs_pips"] == cost],
            key=lambda x: x["delay"],
        )
        xs = [r["delay"] * 10 for r in cost_rows]   # convert bars → seconds
        ys = [r["mean_net"] for r in cost_rows]
        ax.plot(xs, ys, marker="o", color=colors[i % len(colors)],
                linewidth=2, markersize=6, label=f"costs={cost} pips")

    ax.axhline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Execution delay (seconds)", fontsize=9)
    ax.set_ylabel("Mean net P&L (pips/trade)", fontsize=9)
    ax.set_title(
        f"Effect of Execution Delay on Mean Net P&L\n"
        f"(slippage_std={slippage_ref}, threshold=3.0)",
        fontsize=10,
    )
    ax.set_xticks([d * 10 for d in delays])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")


def plot_distribution(
    best_net: np.ndarray,
    real_net: np.ndarray,
    best_label: str,
    real_label: str,
    path: Path,
) -> None:
    """Overlapping histogram + KDE of per-trade net pips."""
    fig, ax = plt.subplots(figsize=(10, 5))

    all_vals = np.concatenate([best_net, real_net])
    lo, hi   = np.percentile(all_vals, [0.5, 99.5])
    bins     = np.linspace(lo, hi, 60)

    ax.hist(best_net, bins=bins, alpha=0.45, color="#4fc3f7",
            label=f"{best_label}  (mean={best_net.mean():+.3f})", density=True)
    ax.hist(real_net, bins=bins, alpha=0.45, color="#ffb74d",
            label=f"{real_label}  (mean={real_net.mean():+.3f})", density=True)

    # KDE overlays
    for arr, color in [(best_net, "#4fc3f7"), (real_net, "#ffb74d")]:
        if arr.std() > 0:
            kde  = gaussian_kde(arr, bw_method="scott")
            xs   = np.linspace(lo, hi, 300)
            ax.plot(xs, kde(xs), color=color, linewidth=1.5, alpha=0.8)

    ax.axvline(0, color="white", linewidth=1.0, linestyle="--", alpha=0.7, label="break-even")
    ax.set_xlabel("Net pips per trade", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title("Per-Trade P&L Distribution: Best-Case vs Realistic", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Execution realism stress-test")
    parser.add_argument("--threshold",    type=float, default=3.0)
    parser.add_argument("--horizon",      type=int,   default=60)
    parser.add_argument("--costs-pips",   type=float, nargs="+", default=[0.3, 0.5, 0.8])
    parser.add_argument("--slippage-std", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3])
    parser.add_argument("--delays",       type=int,   nargs="+", default=[0, 1, 2])
    parser.add_argument("--spread-std",   type=float, default=0.15)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--data-dir",     type=str,   default=str(DATA_DIR))
    parser.add_argument("--output-dir",   type=str,   default=str(OUTPUTS_DIR))
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    plots_dir  = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    costs     = sorted(args.costs_pips)
    slippages = sorted(args.slippage_std)
    delays    = sorted(args.delays)

    # -----------------------------------------------------------------------
    # 1. Load data + features (same preamble as run_sensitivity.py)
    # -----------------------------------------------------------------------
    divider("Loading data + rebuilding features")
    eurusd = load_pair(data_dir, "EURUSD")
    audusd = load_pair(data_dir, "AUDUSD")
    euraud = load_pair(data_dir, "EURAUD")

    sig  = build_signal_frame(eurusd, audusd, euraud, ewma_halflife=360)
    del eurusd, audusd, euraud
    euraud_prices_full = sig["euraud"].copy()
    feat = build_feature_frame(sig)
    del sig
    feat = compute_future_zscore_targets(feat)
    feat["euraud"] = euraud_prices_full
    print(f"Full feature frame: {feat.shape}")

    divider("Pre-computing vol-spike baseline")
    feat["rv_baseline_30d"] = feat["rv_residual_1m"].rolling(_BARS_30D, min_periods=360).mean()
    feat["vol_spike"]       = feat["rv_residual_1m"] > 2.5 * feat["rv_baseline_30d"]

    # -----------------------------------------------------------------------
    # 2. Load model
    # -----------------------------------------------------------------------
    model_path = MODELS_DIR / f"lgbm_regression_h{args.horizon}.pkl"
    if not model_path.exists():
        print(f"Error: model not found at {model_path}. Run run_training.py first.")
        sys.exit(1)
    with open(model_path, "rb") as f:
        artefact = pickle.load(f)
    model        = artefact["model"]
    feature_cols = artefact["feature_cols"]
    print(f"Loaded model: n_estimators={model.n_estimators}")

    val_end_ts = pd.Timestamp(VAL_END, tz="UTC")
    test_df    = feat[feat.index > val_end_ts].copy()
    test_df    = test_df.dropna(subset=feature_cols)
    print(f"Test set: {len(test_df):,} bars  ({test_df.index[0].date()} → {test_df.index[-1].date()})")

    # -----------------------------------------------------------------------
    # 3. Run simulate() once per delay (costs_pips=0 → pure gross P&L)
    # -----------------------------------------------------------------------
    divider("Running simulations (one per delay level)")
    trade_logs_by_delay: dict[int, pd.DataFrame] = {}
    for d in delays:
        tl, _ = simulate(
            test_df, model, feature_cols,
            move_threshold=args.threshold,
            entry_z_min=ENTRY_Z_MIN,
            horizon=args.horizon,
            costs_pips=0.0,
            delay=d,
        )
        trade_logs_by_delay[d] = tl
        mg = tl["gross_pips"].mean() if len(tl) else float("nan")
        print(f"  delay={d} bars ({d*10}s)  →  {len(tl):,} trades  mean_gross={mg:+.3f} pips")

    # -----------------------------------------------------------------------
    # 4. Validation: delay=0 with costs=0 → mean_gross must match known result
    # -----------------------------------------------------------------------
    divider("Validation")
    delay0_gross = trade_logs_by_delay[0]["gross_pips"].mean() if 0 in trade_logs_by_delay else float("nan")
    expected     = 0.54   # mean_gross at threshold=3.0 from sensitivity sweep
    tol          = 0.05
    ok = abs(delay0_gross - expected) < tol
    print(f"  delay=0 mean_gross: {delay0_gross:+.3f}  (expected ≈ {expected:+.2f})")
    print(f"  Validation: {'✓ PASS' if ok else f'✗ FAIL — expected {expected:.2f}±{tol:.2f}'}")
    if not ok:
        print("  WARNING: mean_gross deviates more than expected. Check threshold/data.")

    # -----------------------------------------------------------------------
    # 5. Apply stochastic costs post-hoc — sweep over all combinations
    # -----------------------------------------------------------------------
    divider("Applying stochastic costs across full grid")
    rng     = np.random.default_rng(args.seed)
    results = []
    n_total = len(delays) * len(costs) * len(slippages)
    n_done  = 0

    for d in delays:
        tl = trade_logs_by_delay[d]

        for cost in costs:
            for slip in slippages:
                n_done += 1
                m = apply_costs(
                    tl, test_df.index,
                    costs_pips=cost,
                    spread_std=args.spread_std,
                    slippage_std=slip,
                    rng=rng,
                )
                results.append({
                    "delay":       d,
                    "costs_pips":  cost,
                    "slippage_std": slip,
                    **{k: v for k, v in m.items() if k != "net_pips_arr"},
                })
                if n_done % 12 == 0 or n_done == n_total:
                    print(f"  [{n_done}/{n_total}] delay={d}  cost={cost:.1f}  slip={slip:.1f}  "
                          f"mean_net={m['mean_net']:+.3f}  std={m['std_net']:.3f}")

    # -----------------------------------------------------------------------
    # 6. Generate plots
    # -----------------------------------------------------------------------
    divider("Generating plots")

    # a) Heatmap
    plot_heatmap(results, costs, slippages, plots_dir / "realism_heatmap.png")

    # b) Delay line plot (reference slippage = 0.1, or first available > 0)
    slip_ref = next((s for s in slippages if s > 0), slippages[0])
    plot_delay_lines(results, costs, delays, slip_ref, plots_dir / "realism_delay.png")

    # c) Distribution: best-case (delay=0, cost=min, slip=0) vs realistic (delay=1, cost=mid, slip=0.2)
    best_cost  = costs[0]
    real_cost  = costs[1] if len(costs) > 1 else costs[0]
    real_slip  = 0.2 if 0.2 in slippages else slippages[-1]

    # Re-generate arrays for distribution plot with fixed seed
    rng2 = np.random.default_rng(args.seed)

    tl0 = trade_logs_by_delay.get(0, pd.DataFrame())
    if len(tl0):
        spread0 = np.maximum(0.0, rng2.normal(best_cost, args.spread_std, len(tl0)))
        best_net = tl0["gross_pips"].values - spread0   # slippage=0 for best-case

    tl1 = trade_logs_by_delay.get(1, trade_logs_by_delay.get(delays[1] if len(delays) > 1 else delays[0], pd.DataFrame()))
    if len(tl1):
        spread1   = np.maximum(0.0, rng2.normal(real_cost, args.spread_std, len(tl1)))
        slippage1 = rng2.normal(0.0, real_slip, len(tl1))
        real_net  = tl1["gross_pips"].values - spread1 - slippage1

    if len(tl0) and len(tl1):
        best_label = f"delay=0, cost={best_cost}, slip=0.0"
        real_label = f"delay=1, cost={real_cost}, slip={real_slip}"
        plot_distribution(best_net, real_net, best_label, real_label,
                          plots_dir / "realism_distribution.png")

    # -----------------------------------------------------------------------
    # 7. Analysis summary
    # -----------------------------------------------------------------------
    divider("Analysis Summary")

    profitable = [r for r in results if r["mean_net"] > 0]
    if profitable:
        print(f"\n  Profitable configurations: {len(profitable)} / {len(results)}")
        for r in sorted(profitable, key=lambda x: -x["mean_net"])[:5]:
            print(f"    delay={r['delay']}  cost={r['costs_pips']:.1f}  "
                  f"slip={r['slippage_std']:.1f}  →  mean_net={r['mean_net']:+.3f}  "
                  f"std={r['std_net']:.3f}  n={r['n_trades']:,}")
    else:
        print("  No profitable configurations found.")

    print("\n  Effect of delay on mean_gross (cost- and slippage-independent):")
    for d in delays:
        tl = trade_logs_by_delay[d]
        mg = tl["gross_pips"].mean() if len(tl) else float("nan")
        n  = len(tl)
        print(f"    delay={d} ({d*10}s)  mean_gross={mg:+.4f}  n={n:,}")

    print("\n  Effect of slippage on mean_net (delay=0, cost=0.3):")
    for slip in slippages:
        r = next((x for x in results if x["delay"] == 0 and x["costs_pips"] == 0.3 and x["slippage_std"] == slip), None)
        if r:
            print(f"    slippage_std={slip:.1f}  mean_net={r['mean_net']:+.4f}  "
                  f"std={r['std_net']:.3f}  (mean unchanged; std widens by slippage_std)")

    print("\n  Breakeven cost at delay=0 (slippage_std=0):")
    print(f"    mean_gross ≈ {delay0_gross:+.3f} pips  →  breakeven cost < {delay0_gross:.3f} pips")
    print(f"    At prime brokerage (0.3 pips): mean_net ≈ {delay0_gross - 0.3:+.3f} pips")
    print(f"    At LMAX (0.5 pips):            mean_net ≈ {delay0_gross - 0.5:+.3f} pips")

    # -----------------------------------------------------------------------
    # 8. Save results CSV
    # -----------------------------------------------------------------------
    results_df  = pd.DataFrame(results)
    results_path = output_dir / "realism_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved → {results_path}")
    print(f"Plots saved  → {plots_dir}/realism_*.png")


if __name__ == "__main__":
    main()
