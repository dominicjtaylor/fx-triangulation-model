"""
Sensitivity analysis — 2D sweep: move_threshold × costs_pips

Loads the model artefact and runs the simulation across a grid of
(horizon, move_threshold, costs_pips) combinations. Prints a 2D table
with thresholds as rows and cost assumptions as columns. Cells with
mean_net > 0 (breakeven or better) are marked with *.

For efficiency, simulate() is called once per (horizon, threshold) pair with
costs_pips=0 to get gross P&L. Metrics for each cost assumption are then
computed from the same trade_log without re-running the simulation.

Run from repo root:
    python3 scripts/run_sensitivity.py [options]

Options:
    --horizons INT [INT ...]          bars to hold (default: 60)
    --thresholds FLOAT [FLOAT ...]    min |predicted_move| to enter
                                      (default: 1.0 1.5 2.0 2.5 3.0)
    --costs-pips FLOAT [FLOAT ...]    round-trip costs to test in pips
                                      (default: 0.3 0.5 0.8 1.0 1.2 1.5)
    --kelly FLOAT                     Kelly fraction (default: 0.25)
    --data-dir PATH
    --output-dir PATH

The threshold=1.0, horizon=60, costs=1.2 cell should match run_backtest.py
mean_net output (-0.89 pips).
"""

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from triangulation.data import load_pair
from triangulation.residual import build_signal_frame
from triangulation.features import build_feature_frame
from triangulation.labels import compute_future_zscore_targets
from triangulation.backtest import simulate, daily_sharpe

DATA_DIR    = ROOT / "data"
MODELS_DIR  = ROOT / "outputs" / "models"
OUTPUTS_DIR = ROOT / "outputs"

VAL_END     = "2025-06-30"
ENTRY_Z_MIN = 1.5   # matches run_backtest.py default

# Cost labels for column headers (annotate known venue types)
_COST_LABELS = {
    0.3: "prime/T1",
    0.5: "LMAX",
    0.8: "retail+",
    1.0: "retail",
    1.2: "current",
    1.5: "wide",
}

# 30 days at 10s resolution
_BARS_30D = 259_200


def divider(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def _metrics(
    trade_log: pd.DataFrame,
    test_index: pd.DatetimeIndex,
    costs_pips: float,
) -> dict:
    """Compute summary stats for a given costs_pips applied to a trade_log
    that already has gross_pips populated.

    trade_log must have 'gross_pips', 'exit_time' columns.
    net_pips is recomputed here as gross_pips - costs_pips.
    """
    if len(trade_log) == 0:
        return {"n_trades": 0, "trades_wk": 0.0, "win_rate": float("nan"),
                "mean_gross": float("nan"), "mean_net": float("nan"), "sharpe": float("nan")}

    tl = trade_log.copy()
    tl["net_pips"] = tl["gross_pips"] - costs_pips

    test_days       = (test_index[-1] - test_index[0]).days
    trades_per_week = len(tl) / max(test_days / 7, 1)
    win_rate        = float((tl["net_pips"] > 0).mean())
    mean_gross      = float(tl["gross_pips"].mean())
    mean_net        = float(tl["net_pips"].mean())
    sharpe          = daily_sharpe(tl, test_index)

    return {
        "n_trades":   len(tl),
        "trades_wk":  trades_per_week,
        "win_rate":   win_rate,
        "mean_gross": mean_gross,
        "mean_net":   mean_net,
        "sharpe":     sharpe,
    }


def _print_2d_table(rows: list[dict], costs: list[float]) -> None:
    """Print a 2D table: rows = thresholds, columns = costs_pips.
    Cell value = mean_net pips. Cells with mean_net > 0 marked *.
    One table per horizon.
    """
    horizons = sorted(set(r["horizon"] for r in rows))
    col_w    = 9   # column cell width

    for h in horizons:
        h_rows = [r for r in rows if r["horizon"] == h]
        divider(f"Horizon: {h} bars ({h * 10 // 60} min)  — mean net pips per trade  (* = profitable)")

        # Header row: cost values
        cost_cols = "".join(
            f"  {_COST_LABELS.get(c, str(c)):>{col_w}}" for c in sorted(costs)
        )
        print(f"  {'threshold':>10}{cost_cols}")
        cost_vals = "".join(f"  {c:>{col_w}.1f}" for c in sorted(costs))
        print(f"  {'':>10}{cost_vals}")
        print("  " + "-" * (10 + (col_w + 2) * len(costs) + 2))

        thresholds = sorted(set(r["threshold"] for r in h_rows))
        for thr in thresholds:
            thr_rows = {r["costs_pips"]: r for r in h_rows if r["threshold"] == thr}
            cells = ""
            for c in sorted(costs):
                r = thr_rows.get(c)
                if r is None or np.isnan(r.get("mean_net", float("nan"))):
                    cells += f"  {'n/a':>{col_w}}"
                else:
                    mn = r["mean_net"]
                    marker = "*" if mn > 0 else " "
                    cells += f"  {f'{mn:+.2f}{marker}':>{col_w}}"
            print(f"  {thr:>10.1f}{cells}")

        # Also print trades/wk at costs=1.2 (reference) as a footer
        ref_cost = 1.2 if 1.2 in costs else sorted(costs)[-1]
        print(f"\n  trades/week at costs={ref_cost:.1f}:")
        for thr in thresholds:
            thr_rows = {r["costs_pips"]: r for r in h_rows if r["threshold"] == thr}
            r = thr_rows.get(ref_cost, {})
            tw = r.get("trades_wk", float("nan"))
            print(f"    threshold={thr:.1f}  →  {tw:.0f}/week" if not np.isnan(tw) else f"    threshold={thr:.1f}  →  n/a")


def main() -> None:
    parser = argparse.ArgumentParser(description="2D sensitivity: threshold × costs_pips")
    parser.add_argument("--horizons",   type=int,   nargs="+", default=[60])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[1.0, 1.5, 2.0, 2.5, 3.0])
    parser.add_argument("--costs-pips", type=float, nargs="+",
                        default=[0.3, 0.5, 0.8, 1.0, 1.2, 1.5])
    parser.add_argument("--kelly",      type=float, default=0.25)
    parser.add_argument("--data-dir",   type=str,   default=str(DATA_DIR))
    parser.add_argument("--output-dir", type=str,   default=str(OUTPUTS_DIR))
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load data and rebuild features (once — shared across all runs)
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

    # -----------------------------------------------------------------------
    # 2. Pre-compute vol-spike mask (needs full history for 30-day baseline)
    # -----------------------------------------------------------------------
    divider("Pre-computing vol-spike baseline")
    feat["rv_baseline_30d"] = feat["rv_residual_1m"].rolling(_BARS_30D, min_periods=360).mean()
    feat["vol_spike"]       = feat["rv_residual_1m"] > 2.5 * feat["rv_baseline_30d"]

    # -----------------------------------------------------------------------
    # 3. Sweep: one simulate() call per (horizon, threshold); apply each cost
    # -----------------------------------------------------------------------
    val_end_ts = pd.Timestamp(VAL_END, tz="UTC")
    costs      = sorted(args.costs_pips)
    rows       = []
    n_sims     = len(args.horizons) * len(args.thresholds)
    sim_num    = 0

    for horizon in sorted(args.horizons):
        model_path = MODELS_DIR / f"lgbm_regression_h{horizon}.pkl"
        if not model_path.exists():
            print(f"  [skip] No model for horizon={horizon} at {model_path}")
            continue
        with open(model_path, "rb") as f:
            artefact = pickle.load(f)
        model        = artefact["model"]
        feature_cols = artefact["feature_cols"]

        test_df = feat[feat.index > val_end_ts].copy()
        test_df = test_df.dropna(subset=feature_cols)

        for threshold in sorted(args.thresholds):
            sim_num += 1
            print(f"  [{sim_num}/{n_sims}] horizon={horizon}  threshold={threshold:.1f} ...",
                  end=" ", flush=True)

            # Run with costs=0 to get gross P&L; apply each cost assumption below
            trade_log, _ = simulate(
                test_df,
                model,
                feature_cols,
                move_threshold=threshold,
                entry_z_min=ENTRY_Z_MIN,
                horizon=horizon,
                kelly=args.kelly,
                base_size=100_000.0,
                costs_pips=0.0,
            )
            print(f"trades={len(trade_log):,}  mean_gross={trade_log['gross_pips'].mean():+.2f}" if len(trade_log) else "trades=0")

            for cost in costs:
                m = _metrics(trade_log, test_df.index, cost)
                rows.append({
                    "horizon":    horizon,
                    "threshold":  threshold,
                    "costs_pips": cost,
                    **m,
                })

    # -----------------------------------------------------------------------
    # 4. Print 2D table
    # -----------------------------------------------------------------------
    _print_2d_table(rows, costs)

    # -----------------------------------------------------------------------
    # 5. Save full results
    # -----------------------------------------------------------------------
    results_df   = pd.DataFrame(rows)
    results_path = output_dir / "sensitivity_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved → {results_path}")


if __name__ == "__main__":
    main()
