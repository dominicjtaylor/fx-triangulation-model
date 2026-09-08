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

        # Single-leg execution (Approach A): trades EUR/AUD only, so it
        # carries unintended EUR and AUD directional exposure. See
        # `simulate_3leg()` below for the USD-neutral three-leg variant.

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


def simulate_3leg(
    test_df: pd.DataFrame,
    model,
    feature_cols: list[str],
    *,
    move_threshold: float = 1.0,
    entry_z_min: float = 2.0,
    horizon: int = 60,
    kelly: float = 0.25,
    base_size: float = 100_000.0,
    costs_pips: float = 4.5,
    delay: int = 0,
    min_hold_bars: int = 0,
    profit_target_pips: float = 0.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Three-leg, USD-neutral variant of `simulate()`.

    Instead of trading EUR/AUD alone (Approach A), this routes simultaneous
    orders across EUR/AUD, EUR/USD and AUD/USD sized so that the net EUR and
    AUD exposure is zero at entry (README "Planned: Three-Leg Execution"):

        z < 0 (EUR/AUD cheap):  BUY  EUR/AUD, SELL EUR/USD, BUY  AUD/USD
        z > 0 (EUR/AUD expensive): SELL EUR/AUD, BUY EUR/USD, SELL AUD/USD

    Leg notionals: N_eur on the EUR/AUD and EUR/USD legs, and
    N_aud = N_eur * entry_euraud on the AUD/USD leg (fixed at entry — the
    hedge is not rebalanced intra-trade, matching a real simultaneous-fill
    execution with no mid-trade adjustment).

    Working through the cash flows of opening all three legs and reversing
    them at exit (see README for the worked derivation) collapses to a
    single closed-form P&L that only depends on how much the raw triangle
    gap closed, independent of position size:

        g(t) = eurusd(t) - euraud(t) * audusd(t)      (≈ 0 in equilibrium)
        gross_pips = direction * (g_exit - g_entry) * 10_000

    This is the three-leg analogue of the single-leg
    `direction * (entry_euraud - exit_euraud) * 10_000` — same units, but
    immune to EUR/AUD or AUD/USD moving on their own between entry and exit.
    `costs_pips` is the combined round-trip cost across all three legs
    (wider than the single-leg cost — three spreads instead of one).

    Entry-signal and exit-condition logic (vol-spike / reversal / time /
    optional profit target) is otherwise identical to `simulate()` — only
    execution and P&L differ. See `simulate()` for full parameter docs.

    Args:
        test_df: Must contain feature columns plus 'zscore', 'euraud',
                  'eurusd', 'audusd', and 'vol_spike' columns.

    Returns:
        trade_log:  DataFrame with one row per trade (columns: entry_time,
                    entry_z, entry_euraud, entry_eurusd, entry_audusd,
                    predicted_move, position_size, exit_time, exit_z,
                    exit_euraud, exit_eurusd, exit_audusd, exit_reason,
                    gross_pips, net_pips, bars_held).
        equity:     pd.Series of cumulative net pips indexed by test_df.index.
    """
    X_test        = test_df[feature_cols].values
    z_current     = test_df["zscore"].values
    euraud_prices = test_df["euraud"].values
    eurusd_prices = test_df["eurusd"].values
    audusd_prices = test_df["audusd"].values
    vol_spike     = test_df["vol_spike"].values.astype(bool)
    timestamps    = test_df.index

    # Raw (non-log) triangle gap in USD-per-EUR terms. Proportional to the
    # log residual for small deviations; used directly here because the
    # 3-leg cash-flow derivation falls out cleanly in this form.
    g = eurusd_prices - euraud_prices * audusd_prices

    y_pred          = model.predict(X_test)
    predicted_moves = z_current - y_pred

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
        entry_i = sig_i + delay
        if entry_i >= len(z_current) - 1:
            continue
        if entry_i < next_free:
            continue

        entry_z      = float(z_current[entry_i])
        entry_euraud = float(euraud_prices[entry_i])
        entry_eurusd = float(eurusd_prices[entry_i])
        entry_audusd = float(audusd_prices[entry_i])
        entry_g      = float(g[entry_i])
        pm           = float(predicted_moves[sig_i])
        direction    = float(np.sign(z_current[sig_i]))
        pos_size     = base_size * abs(pm) * kelly

        end_i         = min(entry_i + horizon, len(z_current) - 1)
        future_z      = z_current[entry_i + 1 : end_i + 1]
        future_vs     = vol_spike[entry_i + 1 : end_i + 1]
        future_g      = g[entry_i + 1 : end_i + 1]
        future_len    = len(future_z)

        # Exit condition 1: vol spike
        vs_hits = np.where(future_vs)[0]
        vs_off  = int(vs_hits[0]) if len(vs_hits) > 0 else future_len

        # Exit condition 2: z reversal (gap widened rather than closed)
        rev_hits = np.where(
            (np.abs(future_z) > 1.5 * abs(entry_z)) &
            (np.sign(future_z) == np.sign(entry_z))
        )[0]
        rev_off = int(rev_hits[0]) if len(rev_hits) > 0 else future_len

        # Exit condition 3: profit target on the 3-leg gross P&L
        if profit_target_pips > 0.0:
            gross_at_bar = direction * (future_g - entry_g) * 10_000
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

        exit_off = min(vs_off, pt_off, rev_off, time_off)
        if exit_off == vs_off and vs_off < future_len:
            exit_reason = "vol_spike"
        elif exit_off == pt_off and pt_off < future_len:
            exit_reason = "profit_target"
        elif exit_off == rev_off and rev_off < future_len:
            exit_reason = "reversal"
        else:
            exit_reason = "time"

        exit_i       = entry_i + 1 + exit_off
        exit_z       = float(z_current[exit_i])
        exit_euraud  = float(euraud_prices[exit_i])
        exit_eurusd  = float(eurusd_prices[exit_i])
        exit_audusd  = float(audusd_prices[exit_i])
        exit_g       = float(g[exit_i])

        # Three-leg P&L: depends only on how much the gap g(t) closed,
        # independent of position size (see docstring derivation).
        gross_pips = direction * (exit_g - entry_g) * 10_000
        net_pips   = gross_pips - costs_pips
        bars_held  = exit_i - entry_i

        trades.append({
            "entry_time":     timestamps[entry_i],
            "entry_z":        entry_z,
            "entry_euraud":   entry_euraud,
            "entry_eurusd":   entry_eurusd,
            "entry_audusd":   entry_audusd,
            "predicted_move": pm,
            "position_size":  pos_size,
            "exit_time":      timestamps[exit_i],
            "exit_z":         exit_z,
            "exit_euraud":    exit_euraud,
            "exit_eurusd":    exit_eurusd,
            "exit_audusd":    exit_audusd,
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
