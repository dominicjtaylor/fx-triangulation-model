"""
Simulated trading backtest — Three-Leg, USD-Neutral Execution (Week 3 extension)

Same signal/entry logic and trained model as scripts/run_backtest.py, but
routes simultaneous orders across EUR/AUD, EUR/USD and AUD/USD sized so the
net EUR and AUD exposure is zero at entry, instead of trading EUR/AUD alone
(Approach A). See README "Planned: Three-Leg Execution" for the spec and
src/triangulation/backtest.py::simulate_3leg for the P&L derivation.

This isolates the actual triangulation arbitrage from the unhedged EUR/USD
and AUD/USD directional exposure that the single-leg backtest carries, at
the cost of paying three spreads instead of one. Run this after
run_backtest.py to compare directly: same model, two different execution
approaches.

Run from repo root:
    python3 scripts/run_backtest_3leg.py [options]

Options:
    --horizon INT           bars to hold (default: 60 = 10 min)
    --move-threshold FLOAT  min |predicted_move| to enter (default: 1.0)
    --entry-z-min FLOAT     min |zscore| at entry (default: 2.0 — wider than
                             single-leg's 1.5, to absorb three spreads)
    --kelly FLOAT           Kelly fraction for position sizing (default: 0.25)
    --base-size FLOAT       base position size in units (default: 100000)
    --costs-pips FLOAT      combined round-trip cost across all three legs,
                             in pips (default: 4.5 — vs 1.2 for single-leg)
"""

import argparse
import pickle
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from triangulation.data import load_pair
from triangulation.residual import build_signal_frame
from triangulation.features import build_feature_frame
from triangulation.labels import compute_future_zscore_targets
from triangulation.plots import plot_equity_curve
from triangulation.backtest import simulate_3leg, daily_sharpe

DATA_DIR    = ROOT / "data"
MODELS_DIR  = ROOT / "outputs" / "models"
PLOTS_DIR   = ROOT / "outputs" / "plots"
OUTPUTS_DIR = ROOT / "outputs"

TRAIN_END = "2024-12-31"
VAL_END   = "2025-06-30"

SPLIT_DATES = {"train_end": TRAIN_END, "val_end": VAL_END}

# 30 days at 10s resolution = 30 × 24 × 3600 / 10 = 259,200 bars
_BARS_30D = 259_200

# Profit target sweep — same grid as run_backtest.py for direct comparison.
_MIN_HOLD = 18   # bars = 3 minutes at 10s resolution (≈ OU half-life)
_SCENARIOS = [
    ("Original",  0.0, 0),
    ("T=1.5 pips", 1.5, _MIN_HOLD),
    ("T=2.0 pips", 2.0, _MIN_HOLD),
    ("T=2.5 pips", 2.5, _MIN_HOLD),
]


def divider(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def _metrics(tl: pd.DataFrame, eq: pd.Series, index: pd.DatetimeIndex, days: float) -> dict:
    arr = tl["net_pips"].values
    running_max = eq.cummax()
    return {
        "total_net":    float(arr.sum()),
        "win_rate":     float((arr > 0).mean()),
        "avg_hold_min": float(tl["bars_held"].mean()) * 10 / 60,
        "trades":       len(tl),
        "tpw":          len(tl) / max(days / 7, 1),
        "sharpe":       daily_sharpe(tl, index),
        "max_dd":       float((eq - running_max).min()),
        "exits":        tl["exit_reason"].value_counts(normalize=True).to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-leg USD-neutral backtest on test set")
    parser.add_argument("--horizon",        type=int,   default=60)
    parser.add_argument("--move-threshold", type=float, default=1.0)
    parser.add_argument("--entry-z-min",    type=float, default=2.0,
                        help="Minimum |zscore| at entry (default: 2.0). Wider than the "
                             "single-leg default (1.5) — the gap needs to clear three "
                             "spreads instead of one (README: ~6-8 pips vs ~5-7 pips).")
    parser.add_argument("--kelly",          type=float, default=0.25)
    parser.add_argument("--base-size",      type=float, default=100_000.0)
    parser.add_argument("--costs-pips",     type=float, default=4.5,
                        help="Combined round-trip cost across all three legs, in pips "
                             "(default: 4.5 — vs 1.2 for the single-leg EUR/AUD backtest).")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load model artefact (same trained model as the single-leg backtest —
    #    only execution/P&L changes, not the signal)
    # -----------------------------------------------------------------------
    model_path = MODELS_DIR / f"lgbm_regression_h{args.horizon}.pkl"
    if not model_path.exists():
        print(f"Error: model not found at {model_path}")
        print("Run scripts/run_training.py first.")
        sys.exit(1)

    with open(model_path, "rb") as f:
        artefact = pickle.load(f)
    model        = artefact["model"]
    feature_cols = artefact["feature_cols"]
    print(f"Loaded model from {model_path}  (n_estimators={model.n_estimators})")

    # -----------------------------------------------------------------------
    # 2. Load data and rebuild features — keep all three raw prices, not just
    #    EUR/AUD, since simulate_3leg() needs EUR/USD and AUD/USD for P&L.
    # -----------------------------------------------------------------------
    divider("Loading data + rebuilding features")
    eurusd = load_pair(DATA_DIR, "EURUSD")
    audusd = load_pair(DATA_DIR, "AUDUSD")
    euraud = load_pair(DATA_DIR, "EURAUD")

    sig  = build_signal_frame(eurusd, audusd, euraud, ewma_halflife=360)
    del eurusd, audusd, euraud
    eurusd_prices_full = sig["eurusd"].copy()
    audusd_prices_full = sig["audusd"].copy()
    euraud_prices_full = sig["euraud"].copy()
    feat = build_feature_frame(sig)
    del sig
    feat = compute_future_zscore_targets(feat)
    feat["euraud"] = euraud_prices_full   # re-attach raw prices (excluded from features)
    feat["eurusd"] = eurusd_prices_full
    feat["audusd"] = audusd_prices_full
    print(f"Full feature frame: {feat.shape}")

    # -----------------------------------------------------------------------
    # 3. Pre-compute vol-spike mask (needs full history for 30-day baseline)
    # -----------------------------------------------------------------------
    divider("Pre-computing vol-spike baseline (30-day rolling mean of rv_residual_1m)")
    feat["rv_baseline_30d"] = feat["rv_residual_1m"].rolling(_BARS_30D, min_periods=360).mean()
    feat["vol_spike"] = feat["rv_residual_1m"] > 2.5 * feat["rv_baseline_30d"]
    n_vol_spikes = int(feat["vol_spike"].sum())
    print(f"Vol-spike bars in full history: {n_vol_spikes:,}  ({n_vol_spikes/len(feat)*100:.2f}%)")

    # -----------------------------------------------------------------------
    # 4. Filter to test set
    # -----------------------------------------------------------------------
    val_end_ts = pd.Timestamp(VAL_END, tz="UTC")
    test_df = feat[feat.index > val_end_ts].copy()
    test_df = test_df.dropna(subset=feature_cols)
    print(f"Test set: {len(test_df):,} bars  ({test_df.index[0].date()} → {test_df.index[-1].date()})")

    # -----------------------------------------------------------------------
    # 5. Run all frac_target scenarios
    # -----------------------------------------------------------------------
    sim_kwargs = dict(
        move_threshold=args.move_threshold,
        entry_z_min=args.entry_z_min,
        horizon=args.horizon,
        kelly=args.kelly,
        base_size=args.base_size,
        costs_pips=args.costs_pips,
    )

    results: dict[str, tuple] = {}
    for label, pt_pips, mh_bars in _SCENARIOS:
        divider(f"Running simulation — {label}")
        tl, eq = simulate_3leg(
            test_df, model, feature_cols,
            profit_target_pips=pt_pips,
            min_hold_bars=mh_bars,
            **sim_kwargs,
        )
        results[label] = (tl, eq)
        print(f"  Trades executed: {len(tl):,}")

    # -----------------------------------------------------------------------
    # 6. Compute metrics for all scenarios
    # -----------------------------------------------------------------------
    test_days = (test_df.index[-1] - test_df.index[0]).days
    all_metrics: dict[str, dict] = {}
    for label, (tl, eq) in results.items():
        all_metrics[label] = _metrics(tl, eq, test_df.index, test_days)

    # -----------------------------------------------------------------------
    # 7. Comparison table across profit-target scenarios
    # -----------------------------------------------------------------------
    divider("Comparison: Original vs Profit Target exits (18-bar min hold) — Three-Leg")

    labels = list(all_metrics.keys())
    col_w  = 22
    col_v  = 13

    header = f"  {'Metric':<{col_w}}" + "".join(f"  {lb:>{col_v}}" for lb in labels)
    print(header)
    print(f"  {'-'*col_w}" + "".join(f"  {'-'*col_v}" for _ in labels))

    def row(label_str: str, vals: list, fmt: str = ".1f", pct: bool = False) -> None:
        suffix = "%" if pct else ""
        parts = [f"{v:{fmt}}{suffix}" for v in vals]
        line = f"  {label_str:<{col_w}}" + "".join(f"  {p:>{col_v}}" for p in parts)
        print(line)

    row("Total net P&L (pips)", [all_metrics[lb]["total_net"]     for lb in labels])
    row("Win rate",             [all_metrics[lb]["win_rate"]*100  for lb in labels], pct=True)
    row("Avg hold (min)",       [all_metrics[lb]["avg_hold_min"]  for lb in labels])
    row("Trades executed",      [all_metrics[lb]["trades"]        for lb in labels], fmt=".0f")
    row("Trades/week",          [all_metrics[lb]["tpw"]           for lb in labels])
    row("Daily Sharpe",         [all_metrics[lb]["sharpe"]        for lb in labels], fmt=".3f")
    row("Max drawdown (pips)",  [all_metrics[lb]["max_dd"]        for lb in labels])

    print()
    for lb in labels:
        exits = all_metrics[lb]["exits"]
        exits_str = "  ".join(f"{k} {v:.0%}" for k, v in exits.items())
        print(f"  Exits [{lb}]: {exits_str}")

    # -----------------------------------------------------------------------
    # 8. Pick primary scenario for downstream outputs (same convention as
    #    the single-leg script, for direct comparability)
    # -----------------------------------------------------------------------
    primary_label = "T=2.0 pips" if "T=2.0 pips" in results else "Original"
    trade_log, equity = results[primary_label]
    m_primary = all_metrics[primary_label]

    win_rate      = m_primary["win_rate"]
    total_net     = m_primary["total_net"]
    avg_hold_min  = m_primary["avg_hold_min"]
    trades_per_week = m_primary["tpw"]
    sharpe        = m_primary["sharpe"]
    max_drawdown  = m_primary["max_dd"]
    exit_counts   = m_primary["exits"]

    # -----------------------------------------------------------------------
    # 9. Liberation Day assertion (on primary scenario)
    # -----------------------------------------------------------------------
    divider(f"Liberation Day gate (2025-04-02 → 2025-04-09)  [{primary_label}]")
    ld_start = date(2025, 4, 2)
    ld_end   = date(2025, 4, 9)
    ld_trades = trade_log[
        (pd.to_datetime(trade_log["entry_time"]).dt.date >= ld_start) &
        (pd.to_datetime(trade_log["entry_time"]).dt.date <= ld_end)
    ]
    n_ld  = len(ld_trades)
    ld_pass = n_ld == 0
    print(f"  Liberation Day trades: {n_ld}")
    print(f"  Gate: {'✓ PASS' if ld_pass else f'✗ FAIL ({n_ld} trades entered during structural repricing)'}")

    # -----------------------------------------------------------------------
    # 10. Save primary trade log
    # -----------------------------------------------------------------------
    trade_log_path = OUTPUTS_DIR / "trade_log_test_3leg.csv"
    trade_log.to_csv(trade_log_path, index=False)
    print(f"\nTrade log saved → {trade_log_path}  ({len(trade_log):,} rows)  [{primary_label}]")

    # -----------------------------------------------------------------------
    # 11. Equity curve plot (primary scenario)
    # -----------------------------------------------------------------------
    divider(f"Generating equity curve plot  [{primary_label}]")
    stats_dict = {
        "sharpe":          sharpe,
        "max_drawdown":    max_drawdown,
        "win_rate":        win_rate,
        "trades_per_week": trades_per_week,
        "exit_breakdown":  exit_counts,
    }
    fig = plot_equity_curve(
        equity,
        trade_log,
        SPLIT_DATES,
        stats_dict,
        PLOTS_DIR / "equity_curve_3leg.png",
    )
    plt.close(fig)
    print(f"Plot saved → {PLOTS_DIR}/equity_curve_3leg.png")

    # -----------------------------------------------------------------------
    # 12. Direct comparison against the single-leg backtest, if available
    # -----------------------------------------------------------------------
    single_leg_path = OUTPUTS_DIR / "trade_log_test.csv"
    if single_leg_path.exists():
        divider("Direct comparison: single-leg (EUR/AUD only) vs three-leg (USD-neutral)")
        single_tl = pd.read_csv(single_leg_path, parse_dates=["entry_time", "exit_time"])
        single_sharpe = daily_sharpe(single_tl, test_df.index)
        single_total  = float(single_tl["net_pips"].sum())
        single_wr     = float((single_tl["net_pips"] > 0).mean())
        single_tpw    = len(single_tl) / max(test_days / 7, 1)

        print(f"  {'Metric':<26}{'Single-leg':>15}{'Three-leg':>15}")
        print(f"  {'-'*26}{'-'*15}{'-'*15}")
        print(f"  {'Total net P&L (pips)':<26}{single_total:>15.1f}{total_net:>15.1f}")
        print(f"  {'Win rate':<26}{single_wr*100:>14.1f}%{win_rate*100:>14.1f}%")
        print(f"  {'Trades/week':<26}{single_tpw:>15.1f}{trades_per_week:>15.1f}")
        print(f"  {'Daily Sharpe':<26}{single_sharpe:>15.3f}{sharpe:>15.3f}")
        print(f"\n  Note: three-leg pays ~{args.costs_pips:.1f} pips round-trip vs "
              f"~1.2 for single-leg, but strips out unhedged EUR/USD and AUD/USD "
              f"directional noise from the P&L (see backtest.py::simulate_3leg).")
    else:
        print(f"\n  (No single-leg trade log found at {single_leg_path} — "
              f"run scripts/run_backtest.py first for a side-by-side comparison.)")

    # -----------------------------------------------------------------------
    # 13. Final summary (primary scenario)
    # -----------------------------------------------------------------------
    divider(f"Test set performance — Three-Leg, {primary_label}  (2025-07-01 → 2026-03-01)")
    print(f"  Annualised Sharpe:      {sharpe:.2f}")
    print(f"  Max drawdown (pips):    {max_drawdown:.1f}")
    print(f"  Win rate:               {win_rate:.1%}")
    print(f"  Avg holding time:       {avg_hold_min:.1f} min")
    print(f"  Trades per week:        {trades_per_week:.1f}")
    exits_str = "  ".join(f"{k} {v:.0%}" for k, v in exit_counts.items())
    print(f"  Exit reasons:           {exits_str}")
    print(f"  Liberation Day trades:  {n_ld} {'✓' if ld_pass else '✗'}")


if __name__ == "__main__":
    main()
