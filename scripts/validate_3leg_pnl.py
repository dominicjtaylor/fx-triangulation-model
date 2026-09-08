"""
Validation for the three-leg, USD-neutral P&L formula in `simulate_3leg()`.

Two checks, since there is no market data in this environment to run the
real backtest against:

  1. Ledger check — an explicit leg-by-leg cash-flow simulation (BUY/SELL
     EUR, AUD, USD at each of the three legs, entry and exit) is compared
     against the closed-form shortcut used in `simulate_3leg()`:

         g(t) = eurusd(t) - euraud(t) * audusd(t)
         gross_pips = direction * (g_exit - g_entry) * 10_000

     across several synthetic scenarios (gap closing, widening, fully
     closing; both signal directions). If the ledger and the formula
     disagree, the formula is wrong and `simulate_3leg()` cannot be trusted.

  2. Plumbing check — `simulate_3leg()` itself is run on a small synthetic
     10s-bar price path that manufactures a triangulation lag (EUR/USD
     jumps, EUR/AUD catches up gradually), using a dummy model. Confirms
     the signal/entry/exit machinery fires in the expected direction and
     that its own internal P&L matches the ledger calculation.

Saves a diagnostic plot to outputs/plots/threeleg_pnl_validation.png
showing the manufactured price paths, z-score, and g(t) with the
entry/exit markers from the trade the simulator actually took.

Run from repo root:
    python3 scripts/validate_3leg_pnl.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from triangulation.residual import compute_residual, compute_zscore
from triangulation.backtest import simulate_3leg

PLOTS_DIR = ROOT / "outputs" / "plots"


def divider(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 1. Ledger check: explicit cash-flow simulation vs closed-form formula
# ---------------------------------------------------------------------------

def ledger_pnl_usd(
    direction: float,
    n_eur: float,
    entry_euraud: float, entry_eurusd: float, entry_audusd: float,
    exit_euraud: float,  exit_eurusd: float,  exit_audusd: float,
) -> float:
    """Explicit leg-by-leg cash-flow ledger for the 3-leg trade.

    direction follows the existing single-leg convention: +1 = SHORT
    EUR/AUD (z > 0, expensive), -1 = LONG EUR/AUD (z < 0, cheap).
    Per README: dir_euraud = direction, dir_eurusd = -direction,
    dir_audusd = direction (all in the same +1=short/-1=long convention).

    Cash-flow convention for a leg with side `dir` (+1=short,-1=long),
    notional M in the base currency, at `price` (quote per base):
        ΔBase  = -dir * M
        ΔQuote = +dir * M * price
    """
    n_aud = n_eur * entry_euraud   # AUD notional fixed at entry, per README spec

    def leg(dir_, m, price):
        return -dir_ * m, dir_ * m * price

    # --- Open all three legs -------------------------------------------------
    d_euraud, d_eurusd, d_audusd = direction, -direction, direction

    eur1, aud1 = leg(d_euraud, n_eur, entry_euraud)   # EUR/AUD
    eur2, usd2 = leg(d_eurusd, n_eur, entry_eurusd)   # EUR/USD
    aud3, usd3 = leg(d_audusd, n_aud, entry_audusd)   # AUD/USD

    eur_bal = eur1 + eur2
    aud_bal = aud1 + aud3
    usd_bal = usd2 + usd3
    assert abs(eur_bal) < 1e-6, f"EUR did not net to zero at entry: {eur_bal}"
    assert abs(aud_bal) < 1e-6, f"AUD did not net to zero at entry: {aud_bal}"

    # --- Close all three legs (opposite side, exit prices, same notionals) --
    eur1c, aud1c = leg(-d_euraud, n_eur, exit_euraud)
    eur2c, usd2c = leg(-d_eurusd, n_eur, exit_eurusd)
    aud3c, usd3c = leg(-d_audusd, n_aud, exit_audusd)

    eur_bal += eur1c + eur2c
    aud_bal += aud1c + aud3c
    usd_bal += usd2c + usd3c
    assert abs(eur_bal) < 1e-6, f"EUR did not net to zero after close: {eur_bal}"

    # Leftover AUD balance (hedge notional was fixed at entry, so it drifts
    # from the exact offset once EUR/AUD has moved) is marked to market at
    # the exit AUD/USD rate to fold it into total USD P&L.
    total_usd_pnl = usd_bal + aud_bal * exit_audusd
    return total_usd_pnl


def formula_pnl_usd(
    direction: float, n_eur: float,
    entry_euraud: float, entry_eurusd: float, entry_audusd: float,
    exit_euraud: float,  exit_eurusd: float,  exit_audusd: float,
) -> float:
    g_entry = entry_eurusd - entry_euraud * entry_audusd
    g_exit  = exit_eurusd  - exit_euraud  * exit_audusd
    return direction * n_eur * (g_exit - g_entry)


def run_ledger_check() -> None:
    divider("Check 1: ledger cash-flow sim vs closed-form g(t) formula")

    n_eur = 100_000.0
    scenarios = [
        # label, direction, entry(euraud,eurusd,audusd), exit(euraud,eurusd,audusd)
        ("z>0 gap closes (short EUR/AUD, profit)",
         +1.0, (1.6700, 1.0850, 0.6500), (1.6692, 1.0850, 0.6500)),
        ("z<0 gap closes (long EUR/AUD, profit)",
         -1.0, (1.6685, 1.0850, 0.6500), (1.6692, 1.0850, 0.6500)),
        ("z>0 gap widens (short EUR/AUD, loss)",
         +1.0, (1.6700, 1.0850, 0.6500), (1.6710, 1.0850, 0.6500)),
        ("z<0, AUD/USD also moves mid-trade",
         -1.0, (1.6685, 1.0850, 0.6500), (1.6695, 1.0830, 0.6480)),
        ("gap fully closes to implied",
         +1.0, (1.6700, 1.0850, 0.6500), (1.0850 / 0.6500, 1.0850, 0.6500)),
    ]

    all_ok = True
    for label, direction, entry, exit_ in scenarios:
        ledger  = ledger_pnl_usd(direction, n_eur, *entry, *exit_)
        formula = formula_pnl_usd(direction, n_eur, *entry, *exit_)
        ok = np.isclose(ledger, formula, atol=1e-6)
        all_ok &= ok
        print(f"  [{'OK' if ok else 'MISMATCH'}] {label}")
        print(f"           ledger=${ledger:,.4f}   formula=${formula:,.4f}")

    if not all_ok:
        raise SystemExit("Ledger vs formula mismatch — do not trust simulate_3leg().")
    print("\n  All scenarios match — closed-form g(t) formula is a valid")
    print("  shortcut for the full 3-leg cash-flow ledger.")


# ---------------------------------------------------------------------------
# 2. Plumbing check: run simulate_3leg() on a manufactured lag event
# ---------------------------------------------------------------------------

class _ZeroModel:
    """Dummy model: predicts 0 always, so predicted_move == zscore exactly
    (sign condition in simulate_3leg is trivially satisfied on any |z| move)."""
    def predict(self, X):
        return np.zeros(len(X))


def build_synthetic_lag_event(n_bars: int = 300) -> pd.DataFrame:
    """Manufacture a triangulation lag: EUR/USD jumps at bar 20; EUR/AUD
    catches up to the new implied level gradually over the next 40 bars.
    AUD/USD stays flat throughout. 10s bar spacing to match real data."""
    idx = pd.date_range("2026-01-01", periods=n_bars, freq="10s", tz="UTC")

    eurusd = np.full(n_bars, 1.0800)
    eurusd[20:] = 1.0850   # instantaneous jump

    audusd = np.full(n_bars, 0.6500)   # no move — isolates the EUR/USD leg

    implied = eurusd / audusd
    euraud = implied.copy()
    # EUR/AUD lags: stays at the old implied level, then linearly catches
    # up to the new implied level over bars 20 -> 60.
    old_implied = 1.0800 / 0.6500
    catch_up = np.linspace(0.0, 1.0, 40)
    euraud[20:60] = old_implied + catch_up * (implied[20:60] - old_implied)
    euraud[:20] = old_implied

    df = pd.DataFrame({"eurusd": eurusd, "audusd": audusd, "euraud": euraud}, index=idx)
    residual = compute_residual(df)
    zscore = compute_zscore(residual, halflife=36)   # 6-min halflife, short enough for this toy series
    df["residual"] = residual
    df["zscore"] = zscore
    df["vol_spike"] = False
    df["dummy_feature"] = 0.0
    return df


def run_plumbing_check() -> None:
    divider("Check 2: simulate_3leg() plumbing on a manufactured lag event")

    df = build_synthetic_lag_event()
    trade_log, equity = simulate_3leg(
        df, _ZeroModel(), feature_cols=["dummy_feature"],
        move_threshold=0.1, entry_z_min=0.5, horizon=100,
        kelly=0.25, base_size=100_000.0, costs_pips=0.0,  # costs=0 to isolate gross P&L check
    )

    print(f"  Trades executed: {len(trade_log)}")
    if len(trade_log) == 0:
        raise SystemExit("Expected at least one trade on the manufactured lag event — got zero.")

    row = trade_log.iloc[0]
    print(f"  entry_time={row['entry_time']}  entry_z={row['entry_z']:.2f}")
    print(f"  exit_time={row['exit_time']}    exit_reason={row['exit_reason']}")
    print(f"  gross_pips={row['gross_pips']:.3f}")

    # Cross-check simulate_3leg's own P&L against the independent ledger fn.
    direction = -1.0 if row["entry_z"] < 0 else 1.0
    ledger = ledger_pnl_usd(
        direction, 100_000.0,
        row["entry_euraud"], row["entry_eurusd"], row["entry_audusd"],
        row["exit_euraud"], row["exit_eurusd"], row["exit_audusd"],
    )
    ledger_pips = ledger / (100_000.0 * 1e-4)
    ok = np.isclose(ledger_pips, row["gross_pips"], atol=1e-6)
    print(f"  Independent ledger check: {ledger_pips:.3f} pips  [{'OK' if ok else 'MISMATCH'}]")
    if not ok:
        raise SystemExit("simulate_3leg() internal P&L disagrees with the ledger check.")

    assert row["entry_z"] < 0, (
        f"Expected the manufactured event to signal z<0 (EUR/AUD cheap, "
        f"EUR/USD jumped up while EUR/AUD lagged); got entry_z={row['entry_z']:.2f}"
    )
    assert row["gross_pips"] > 0, (
        "Expected a profitable trade — EUR/AUD catches up to the new "
        "implied level within the horizon."
    )
    print("\n  Direction and sign of P&L match expectations for a")
    print("  liquidity-lag event: cheap EUR/AUD -> long the triangle -> profit")
    print("  as EUR/AUD catches up to implied.")

    # -----------------------------------------------------------------------
    # Diagnostic plot
    # -----------------------------------------------------------------------
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    g = df["eurusd"] - df["euraud"] * df["audusd"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax1.plot(df.index, df["euraud"], label="EUR/AUD actual", color="#90caf9")
    ax1.plot(df.index, df["eurusd"] / df["audusd"], label="EUR/AUD implied",
              color="#ffb74d", linestyle="--")
    ax1.set_ylabel("EUR/AUD")
    ax1.set_title("Manufactured triangulation lag — actual vs implied EUR/AUD")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.2)

    ax2.plot(df.index, df["zscore"], label="z-score", color="#81c784")
    ax2.plot(df.index, g * 10_000, label="g(t) x 10,000 (3-leg P&L driver)", color="#e57373")
    ax2.axvline(row["entry_time"], color="white", linestyle=":", linewidth=1, label="entry")
    ax2.axvline(row["exit_time"],  color="white", linestyle="--", linewidth=1, label="exit")
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax2.set_ylabel("z-score / g(t)")
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.2)

    fig.tight_layout()
    out_path = PLOTS_DIR / "threeleg_pnl_validation.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n  Diagnostic plot saved -> {out_path}")
    print("  Verifies: the manufactured EUR/USD jump opens a real EUR/AUD/")
    print("  implied gap, z-score and g(t) move together as expected, and the")
    print("  trade simulate_3leg() takes on this gap enters in the correct")
    print("  direction and closes profitably as EUR/AUD catches up.")


def main() -> None:
    run_ledger_check()
    run_plumbing_check()
    divider("All checks passed")


if __name__ == "__main__":
    main()
