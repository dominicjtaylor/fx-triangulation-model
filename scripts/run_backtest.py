"""
Simulated trading backtest — EUR/USD/AUD Triangle (Week 3)

Loads the saved model artefact from outputs/models/ and runs a deterministic
simulation on the test set (2025-07-01 → 2026-03-17).

Three deterministic exit conditions (checked in priority order):
  1. Vol-spike gate:     1-min RV > 2.5× 30-day rolling avg → exit + suppress 30 min
  2. Z-score reversal:  |z| grows to 1.5× |entry_z| (gap widened) → exit
  3. Time-based:        exit after `horizon` bars regardless

Entry: |predicted_move| > move_threshold, no open position, not suppressed.
Costs: 1.2 pips round-trip deducted per trade.

Run from repo root:
    python3 scripts/run_backtest.py [options]

Options:
    --horizon INT           bars to hold (default: 60 = 10 min)
    --move-threshold FLOAT  min |predicted_move| to enter (default: 1.0)
    --kelly FLOAT           Kelly fraction for position sizing (default: 0.25)
    --base-size FLOAT       base position size in units (default: 100000)
    --costs-pips FLOAT      round-trip cost per trade in pips (default: 1.2)
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

DATA_DIR   = ROOT / "data"
MODELS_DIR = ROOT / "outputs" / "models"
PLOTS_DIR  = ROOT / "outputs" / "plots"
OUTPUTS_DIR = ROOT / "outputs"

TRAIN_END = "2024-12-31"
VAL_END   = "2025-06-30"

SPLIT_DATES = {"train_end": TRAIN_END, "val_end": VAL_END}

# 30 days at 10s resolution = 30 × 24 × 3600 / 10 = 259,200 bars
_BARS_30D  = 259_200
# 30 minutes suppression = 30 × 60 / 10 = 180 bars
_BARS_30M  = 180


def divider(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def _build_equity_series(
    trade_log: pd.DataFrame,
    index: pd.DatetimeIndex,
) -> pd.Series:
    """Map trade net_pips to their exit bar, then cumsum to get equity curve."""
    pnl = pd.Series(0.0, index=index)
    for _, row in trade_log.iterrows():
        exit_ts = row["exit_time"]
        if exit_ts in pnl.index:
            pnl.loc[exit_ts] += row["net_pips"]
        else:
            # Snap to nearest available bar
            pos = pnl.index.searchsorted(exit_ts)
            if pos < len(pnl):
                pnl.iloc[pos] += row["net_pips"]
    return pnl.cumsum()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest regression model on test set")
    parser.add_argument("--horizon",        type=int,   default=60)
    parser.add_argument("--move-threshold", type=float, default=1.0)
    parser.add_argument("--entry-z-min",    type=float, default=1.5,
                        help="Minimum |zscore| at entry (default: 1.5). Prevents entering "
                             "at trivially-small z-scores where the 1.5x reversal stop "
                             "would fire at |z|<2.25 from normal z-score fluctuation.")
    parser.add_argument("--kelly",          type=float, default=0.25)
    parser.add_argument("--base-size",      type=float, default=100_000.0)
    parser.add_argument("--costs-pips",     type=float, default=1.2)
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load model artefact
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
    # 2. Load data and rebuild features
    # -----------------------------------------------------------------------
    divider("Loading data + rebuilding features")
    eurusd = load_pair(DATA_DIR, "EURUSD")
    audusd = load_pair(DATA_DIR, "AUDUSD")
    euraud = load_pair(DATA_DIR, "EURAUD")

    sig  = build_signal_frame(eurusd, audusd, euraud, ewma_halflife=360)
    del eurusd, audusd, euraud
    euraud_prices_full = sig["euraud"].copy()   # keep EUR/AUD price for P&L calculation
    feat = build_feature_frame(sig)
    del sig
    feat = compute_future_zscore_targets(feat)
    feat["euraud"] = euraud_prices_full         # re-attach EUR/AUD price (excluded from features)
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
    # 5. Batch model inference
    # -----------------------------------------------------------------------
    divider("Running batch inference on test set")
    X_test         = test_df[feature_cols].values
    z_current      = test_df["zscore"].values
    euraud_prices  = test_df["euraud"].values    # EUR/AUD spot price for P&L in pips
    vol_spike      = test_df["vol_spike"].values.astype(bool)
    timestamps     = test_df.index

    y_pred          = model.predict(X_test)
    predicted_moves = z_current - y_pred

    # Signal candidates:
    #   1. |predicted_move| > threshold (model sees a large expected z-score change)
    #   2. |z_current| >= entry_z_min (gap is large enough that the reversal stop won't fire trivially)
    #   3. sign(z_current) == sign(predicted_move) — mean-reversion only:
    #      - z > 0 and predicted_move > 0: model expects z to fall → SHORT EUR/AUD ✓
    #      - z < 0 and predicted_move < 0: model expects z to rise → LONG  EUR/AUD ✓
    #      Filters out cases where the model predicts z widening (momentum), which would
    #      create a direction inconsistent with the mean-reversion rationale.
    #   4. Not a vol-spike bar
    signal_mask = (
        (np.abs(predicted_moves) > args.move_threshold) &
        (np.abs(z_current) >= args.entry_z_min) &
        (np.sign(z_current) == np.sign(predicted_moves)) &
        (~vol_spike)
    )
    signal_indices = np.where(signal_mask)[0]
    print(f"Signal candidates: {len(signal_indices):,}  ({len(signal_indices)/len(test_df)*100:.1f}% of bars)")

    # -----------------------------------------------------------------------
    # 6. Simulation loop
    # -----------------------------------------------------------------------
    divider("Running simulation (one position at a time)")
    trades       = []
    next_free    = 0   # first bar index we can enter a new trade

    for sig_i in signal_indices:
        if sig_i < next_free:
            continue

        # Entry
        entry_i      = sig_i
        entry_z      = float(z_current[entry_i])
        entry_euraud = float(euraud_prices[entry_i])
        pm           = float(predicted_moves[entry_i])
        direction    = float(np.sign(entry_z))   # sign(z) = trade direction: +1 = SHORT, -1 = LONG
        pos_size     = args.base_size * abs(pm) * args.kelly

        # Vectorised exit detection over next `horizon` bars
        end_i       = min(entry_i + args.horizon, len(z_current) - 1)
        future_z    = z_current[entry_i + 1 : end_i + 1]
        future_vs   = vol_spike[entry_i + 1 : end_i + 1]
        future_len  = len(future_z)

        # Exit condition 1: vol spike
        vs_hits = np.where(future_vs)[0]
        vs_off  = int(vs_hits[0]) if len(vs_hits) > 0 else future_len

        # Exit condition 2: z reversal (|z| grew to > 1.5× |entry_z|, same sign = gap widened)
        rev_hits = np.where(
            (np.abs(future_z) > 1.5 * abs(entry_z)) &
            (np.sign(future_z) == np.sign(entry_z))
        )[0]
        rev_off = int(rev_hits[0]) if len(rev_hits) > 0 else future_len

        # Exit condition 3: time-based (last bar of horizon)
        time_off = future_len - 1

        # First exit wins
        exit_off    = min(vs_off, rev_off, time_off)
        if vs_off <= rev_off and vs_off < future_len:
            exit_reason = "vol_spike"
        elif rev_off < vs_off and rev_off < future_len:
            exit_reason = "reversal"
        else:
            exit_reason = "time"

        exit_i       = entry_i + 1 + exit_off
        exit_z       = float(z_current[exit_i])
        exit_euraud  = float(euraud_prices[exit_i])

        # P&L in EUR/AUD pips (Approach A: trade EUR/AUD leg only).
        # direction=+1 (SHORT): profit when EUR/AUD price falls (entry > exit).
        # direction=-1 (LONG):  profit when EUR/AUD price rises (exit > entry).
        # gross_pips = direction × (entry_euraud − exit_euraud) × 10000
        gross_pips = direction * (entry_euraud - exit_euraud) * 10_000
        net_pips   = gross_pips - args.costs_pips
        bars_held  = exit_i - entry_i

        trades.append({
            "entry_time":    timestamps[entry_i],
            "entry_z":       entry_z,
            "entry_euraud":  entry_euraud,
            "predicted_move": pm,
            "position_size":  pos_size,
            "exit_time":      timestamps[exit_i],
            "exit_z":         exit_z,
            "exit_euraud":   exit_euraud,
            "exit_reason":    exit_reason,
            "gross_pips":     gross_pips,
            "net_pips":       net_pips,
            "bars_held":      bars_held,
        })

        # Advance free cursor; vol spike adds 30-min suppression
        if exit_reason == "vol_spike":
            next_free = exit_i + 1 + _BARS_30M
        else:
            next_free = exit_i + 1

    trade_log = pd.DataFrame(trades)
    print(f"Trades executed: {len(trade_log):,}")

    if len(trade_log) == 0:
        print("No trades — check move_threshold or data range.")
        sys.exit(0)

    # -----------------------------------------------------------------------
    # 7. Performance metrics
    # -----------------------------------------------------------------------
    divider("Performance metrics")
    net_pips_arr = trade_log["net_pips"].values
    total_net    = float(net_pips_arr.sum())
    win_rate     = float((net_pips_arr > 0).mean())
    avg_hold_bars = float(trade_log["bars_held"].mean())
    avg_hold_min  = avg_hold_bars * 10 / 60

    # Annualised Sharpe on daily P&L
    equity = _build_equity_series(trade_log, test_df.index)
    pnl_at_exit = pd.Series(0.0, index=test_df.index)
    for _, row in trade_log.iterrows():
        ts = row["exit_time"]
        if ts in pnl_at_exit.index:
            pnl_at_exit.loc[ts] += row["net_pips"]
    daily_pnl   = pnl_at_exit.resample("1D").sum()
    daily_pnl   = daily_pnl[daily_pnl != 0]
    sharpe      = float(daily_pnl.mean() / (daily_pnl.std() + 1e-10) * np.sqrt(252))

    # Drawdown
    running_max  = equity.cummax()
    drawdown     = equity - running_max
    max_drawdown = float(drawdown.min())

    # Trades per week
    test_days    = (test_df.index[-1] - test_df.index[0]).days
    trades_per_week = len(trade_log) / max(test_days / 7, 1)

    # Exit breakdown
    exit_counts  = trade_log["exit_reason"].value_counts(normalize=True).to_dict()

    print(f"  Total net P&L:     {total_net:+.1f} pips")
    print(f"  Win rate:          {win_rate:.1%}")
    print(f"  Avg hold:          {avg_hold_min:.1f} min ({avg_hold_bars:.1f} bars)")
    print(f"  Trades/week:       {trades_per_week:.1f}")
    print(f"  Daily Sharpe:      {sharpe:.3f}")
    print(f"  Max drawdown:      {max_drawdown:.1f} pips")
    print(f"  Exit reasons:      "
          + "  ".join(f"{k}: {v:.0%}" for k, v in exit_counts.items()))

    # -----------------------------------------------------------------------
    # 8. Liberation Day assertion
    # -----------------------------------------------------------------------
    divider("Liberation Day gate (2025-04-02 → 2025-04-09)")
    ld_start = date(2025, 4, 2)
    ld_end   = date(2025, 4, 9)
    ld_trades = trade_log[
        (pd.to_datetime(trade_log["entry_time"]).dt.date >= ld_start) &
        (pd.to_datetime(trade_log["entry_time"]).dt.date <= ld_end)
    ]
    n_ld = len(ld_trades)
    ld_pass = n_ld == 0
    print(f"  Liberation Day trades: {n_ld}")
    print(f"  Gate: {'✓ PASS' if ld_pass else f'✗ FAIL ({n_ld} trades entered during structural repricing)'}")

    # -----------------------------------------------------------------------
    # 9. Save trade log
    # -----------------------------------------------------------------------
    trade_log_path = OUTPUTS_DIR / "trade_log_test.csv"
    trade_log.to_csv(trade_log_path, index=False)
    print(f"\nTrade log saved → {trade_log_path}  ({len(trade_log):,} rows)")

    # -----------------------------------------------------------------------
    # 10. Equity curve plot
    # -----------------------------------------------------------------------
    divider("Generating equity curve plot")
    stats_dict = {
        "sharpe":         sharpe,
        "max_drawdown":   max_drawdown,
        "win_rate":       win_rate,
        "trades_per_week": trades_per_week,
        "exit_breakdown": exit_counts,
    }
    fig = plot_equity_curve(
        equity,
        trade_log,
        SPLIT_DATES,
        stats_dict,
        PLOTS_DIR / "equity_curve.png",
    )
    plt.close(fig)
    print(f"Plot saved → {PLOTS_DIR}/equity_curve.png")

    # -----------------------------------------------------------------------
    # 11. Final summary
    # -----------------------------------------------------------------------
    divider("Test set performance (2025-07-01 → 2026-03-01)")
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
