"""
Simulation engine for the EUR/USD/AUD triangle strategy.

Extracted from scripts/run_backtest.py so that run_sensitivity.py can call
the same simulation loop with different parameters without code duplication.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# 30 minutes suppression window after a vol-spike exit (10s resolution: 30 × 60 / 10)
_BARS_30M = 180


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
            pos = pnl.index.searchsorted(exit_ts)
            if pos < len(pnl):
                pnl.iloc[pos] += row["net_pips"]
    return pnl.cumsum()


def daily_sharpe(
    trade_log: pd.DataFrame,
    index: pd.DatetimeIndex,
) -> float:
    """Annualised Sharpe ratio computed on daily P&L from the trade log.

    Args:
        trade_log: DataFrame with 'exit_time' and 'net_pips' columns.
        index:     DatetimeIndex of the test period (used for resampling).

    Returns:
        Annualised Sharpe, or NaN if fewer than 2 non-zero days.
    """
    pnl = pd.Series(0.0, index=index)
    for _, row in trade_log.iterrows():
        ts = row["exit_time"]
        if ts in pnl.index:
            pnl.loc[ts] += row["net_pips"]
        else:
            pos = pnl.index.searchsorted(ts)
            if pos < len(pnl):
                pnl.iloc[pos] += row["net_pips"]
    daily = pnl.resample("1D").sum()
    daily = daily[daily != 0]
    if len(daily) < 2:
        return float("nan")
    return float(daily.mean() / (daily.std() + 1e-10) * np.sqrt(252))


def simulate(
    test_df: pd.DataFrame,
    model,
    feature_cols: list[str],
    *,
    move_threshold: float = 1.0,
    entry_z_min: float = 1.5,
    horizon: int = 60,
    kelly: float = 0.25,
    base_size: float = 100_000.0,
    costs_pips: float = 1.2,
    delay: int = 0,
    min_hold_bars: int = 0,
    profit_target_pips: float = 0.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Run the deterministic simulation loop on a prepared test DataFrame.

    Args:
        test_df:             Test set DataFrame. Must contain feature columns
                             plus 'zscore', 'euraud', and 'vol_spike' columns.
        model:               Fitted LGBMRegressor.
        feature_cols:        Ordered list of feature column names.
        move_threshold:      Minimum |predicted_move| to enter a trade.
        entry_z_min:         Minimum |z_current| at entry. Prevents entering
                             when the 1.5× reversal stop would fire at trivially
                             small z.
        horizon:             Maximum bars to hold before time-based exit.
        kelly:               Kelly fraction for position sizing.
        base_size:           Base position size in units.
        costs_pips:          Round-trip execution cost per trade in pips.
        delay:               Execution delay in bars. Signal fires at sig_i;
                             order executes at sig_i + delay using the price at
                             that bar. Direction is locked in at sig_i.
                             Default 0 = no delay (original behaviour).
        min_hold_bars:       Minimum bars to hold before any profit-target exit
                             can fire. Lets the noise period (sub-OU-half-life)
                             pass before checking for a signal exit.
                             Default 0 = no minimum hold.
        profit_target_pips:  Gross pip profit target. When > 0, exit on the bar
                             AFTER gross P&L first reaches this level, but only
                             after min_hold_bars have elapsed. The time-based
                             exit (horizon) acts as a hard backstop.
                             When 0.0 (default), no signal exit — time is primary.

    Returns:
        trade_log:  DataFrame with one row per trade (columns: entry_time,
                    entry_z, entry_euraud, predicted_move, position_size,
                    exit_time, exit_z, exit_euraud, exit_reason,
                    gross_pips, net_pips, bars_held).
        equity:     pd.Series of cumulative net pips indexed by test_df.index.
                    Empty (all zeros) if no trades were executed.
    """
    X_test        = test_df[feature_cols].values
    z_current     = test_df["zscore"].values
    euraud_prices = test_df["euraud"].values
    vol_spike     = test_df["vol_spike"].values.astype(bool)
    timestamps    = test_df.index

    y_pred          = model.predict(X_test)
    predicted_moves = z_current - y_pred

    # Signal candidates:
    #   1. |predicted_move| > threshold
    #   2. |z_current| >= entry_z_min (gap large enough that reversal stop won't fire trivially)
    #   3. sign(z_current) == sign(predicted_move) — mean-reversion only
    #   4. Not a vol-spike bar
    signal_mask = (
        (np.abs(predicted_moves) > move_threshold) &
        (np.abs(z_current) >= entry_z_min) &
        (np.sign(z_current) == np.sign(predicted_moves)) &
        (~vol_spike)
    )
    signal_indices = np.where(signal_mask)[0]

    trades    = []
    next_free = 0

    for sig_i in signal_indices:
        # Actual execution bar: signal bar + delay
        entry_i = sig_i + delay
        if entry_i >= len(z_current) - 1:   # need at least 1 bar for exit
            continue
        if entry_i < next_free:              # previous position still open
            continue

        # CURRENT: Single-leg execution (Approach A)
        # Trades EUR/AUD only. Simple to implement and evaluate.
        # Carries unintended EUR and AUD directional exposure.
        #
        # PLANNED: Three-leg execution
        # Route simultaneous orders across EUR/AUD, EUR/USD, AUD/USD
        # in proportions that net USD exposure to zero.
        # Implement after single-leg version is validated end-to-end.
        # See README section "Planned: Three-Leg Execution" for full spec.

        # Entry — direction locked at signal bar; price/z at actual execution bar
        entry_z      = float(z_current[entry_i])
        entry_euraud = float(euraud_prices[entry_i])
        pm           = float(predicted_moves[sig_i])          # prediction from signal bar
        direction    = float(np.sign(z_current[sig_i]))       # direction from signal bar
        pos_size     = base_size * abs(pm) * kelly

        # Vectorised exit detection over next `horizon` bars
        end_i         = min(entry_i + horizon, len(z_current) - 1)
        future_z      = z_current[entry_i + 1 : end_i + 1]
        future_vs     = vol_spike[entry_i + 1 : end_i + 1]
        future_euraud = euraud_prices[entry_i + 1 : end_i + 1]
        future_len    = len(future_z)

        # Exit condition 1: vol spike
        vs_hits = np.where(future_vs)[0]
        vs_off  = int(vs_hits[0]) if len(vs_hits) > 0 else future_len

        # Exit condition 2: z reversal (|z| grew to > 1.5× |entry_z|, same sign = gap widened)
        rev_hits = np.where(
            (np.abs(future_z) > 1.5 * abs(entry_z)) &
            (np.sign(future_z) == np.sign(entry_z))
        )[0]
        rev_off = int(rev_hits[0]) if len(rev_hits) > 0 else future_len

        # Exit condition 3: price-based profit target with minimum hold
        # Only fires after min_hold_bars have elapsed (lets sub-OU-half-life
        # noise pass). Detected at bar T; exit executes at bar T+1 to avoid
        # look-ahead bias (we can only act after observing bar T's close).
        if profit_target_pips > 0.0:
            gross_at_bar = direction * (entry_euraud - future_euraud) * 10_000
            pt_cond = np.zeros(future_len, dtype=bool)
            if min_hold_bars < future_len:
                pt_cond[min_hold_bars:] = gross_at_bar[min_hold_bars:] >= profit_target_pips
            pt_hits = np.where(pt_cond)[0]
            if len(pt_hits) > 0:
                next_bar = int(pt_hits[0]) + 1
                pt_off = next_bar if next_bar < future_len else future_len
            else:
                pt_off = future_len
        else:
            pt_off = future_len

        # Exit condition 4: time-based backstop
        time_off = future_len - 1

        # First exit wins; priority: vol_spike > profit_target > reversal > time
        exit_off = min(vs_off, pt_off, rev_off, time_off)
        if exit_off == vs_off and vs_off < future_len:
            exit_reason = "vol_spike"
        elif exit_off == pt_off and pt_off < future_len:
            exit_reason = "profit_target"
        elif exit_off == rev_off and rev_off < future_len:
            exit_reason = "reversal"
        else:
            exit_reason = "time"

        exit_i      = entry_i + 1 + exit_off
        exit_z      = float(z_current[exit_i])
        exit_euraud = float(euraud_prices[exit_i])

        # P&L in EUR/AUD pips (Approach A: trade EUR/AUD leg only).
        # direction=+1 (SHORT): profit when EUR/AUD price falls (entry > exit).
        # direction=-1 (LONG):  profit when EUR/AUD price rises (exit > entry).
        gross_pips = direction * (entry_euraud - exit_euraud) * 10_000
        net_pips   = gross_pips - costs_pips
        bars_held  = exit_i - entry_i

        trades.append({
            "entry_time":     timestamps[entry_i],
            "entry_z":        entry_z,
            "entry_euraud":   entry_euraud,
            "predicted_move": pm,
            "position_size":  pos_size,
            "exit_time":      timestamps[exit_i],
            "exit_z":         exit_z,
            "exit_euraud":    exit_euraud,
            "exit_reason":    exit_reason,
            "gross_pips":     gross_pips,
            "net_pips":       net_pips,
            "bars_held":      bars_held,
        })

        if exit_reason == "vol_spike":
            next_free = exit_i + 1 + _BARS_30M
        else:
            next_free = exit_i + 1

    trade_log = pd.DataFrame(trades)
    if len(trade_log) == 0:
        return trade_log, pd.Series(0.0, index=test_df.index)

    equity = _build_equity_series(trade_log, test_df.index)
    return trade_log, equity
