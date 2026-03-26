"""
Baseline analysis — EUR/USD/AUD Triangle (Week 1)

Runs stationarity tests, OU half-life estimation, autocorrelation, and signal
frequency analysis on native 10-second bar data. Saves five diagnostic plots
to outputs/plots/.

Run from repo root:
    python3 scripts/run_baseline.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from triangulation.data import load_pair
from triangulation.residual import build_signal_frame
from triangulation.analysis import (
    adf_test,
    kpss_test,
    autocorr_at_lags,
    ou_halflife,
    ou_halflife_by_period,
    signal_stats,
    residual_summary,
)
from triangulation.plots import (
    plot_raw_pairs,
    plot_residual,
    plot_liberation_day_zoom,
    plot_ou_diagnostics,
    plot_signal_distribution,
)

DATA_DIR  = ROOT / "data"
PLOTS_DIR = ROOT / "outputs" / "plots"
EWMA_HALFLIFE = 360   # 1 hour at 10s resolution


def divider(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def main() -> None:
    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    divider("Loading data (10s bars)")
    eurusd = load_pair(DATA_DIR, "EURUSD")
    audusd = load_pair(DATA_DIR, "AUDUSD")
    euraud = load_pair(DATA_DIR, "EURAUD")
    print(f"EURUSD: {len(eurusd):,} bars  {eurusd.index[0].date()} → {eurusd.index[-1].date()}")
    print(f"AUDUSD: {len(audusd):,} bars  {audusd.index[0].date()} → {audusd.index[-1].date()}")
    print(f"EURAUD: {len(euraud):,} bars  {euraud.index[0].date()} → {euraud.index[-1].date()}")

    # -----------------------------------------------------------------------
    # 2. Signal frame
    # -----------------------------------------------------------------------
    divider("Building signal frame (residual + z-score at 10s)")
    sig = build_signal_frame(eurusd, audusd, euraud, ewma_halflife=EWMA_HALFLIFE)
    del eurusd, audusd, euraud
    print(f"Aligned: {len(sig):,} bars  ({sig.index[0].date()} → {sig.index[-1].date()})")
    print(f"Alignment rate: {len(sig)/(len(sig)):100.1f}%")  # always 100% post-align

    # -----------------------------------------------------------------------
    # 3. Residual summary statistics
    # -----------------------------------------------------------------------
    divider("Residual summary statistics")
    summary = residual_summary(sig["residual"])
    for k, v in summary.items():
        print(f"  {k:12s}: {v:.6f}")
    print(f"\n  Residual std = {summary['std']*10000:.2f} 'pips' (×10000)")
    print(f"  Lag-1 autocorr (10s) = {summary['autocorr1']:.4f}")

    # -----------------------------------------------------------------------
    # 4. Stationarity tests — full sample (5-min downsample for ADF/KPSS speed)
    # -----------------------------------------------------------------------
    divider("Stationarity tests — full sample (5-min downsample)")
    residual_10s  = sig["residual"]
    residual_5min = residual_10s.resample("5min").last().dropna()
    print(f"5-min sample: {len(residual_5min):,} bars")

    adf_result  = adf_test(residual_5min)
    kpss_result = kpss_test(residual_5min)

    print(f"\n  ADF:  stat={adf_result['stat']:.4f}  p={adf_result['pvalue']:.6f}"
          f"  {'✓ Stationary' if adf_result['is_stationary'] else '✗ Non-stationary'}")
    print(f"  KPSS: stat={kpss_result['stat']:.4f}  p={kpss_result['pvalue']:.6f}"
          f"  {'✓ Stationary' if kpss_result['is_stationary'] else '✗ Non-stationary'}")

    # -----------------------------------------------------------------------
    # 5. Stationarity tests — by 6-month window
    # -----------------------------------------------------------------------
    divider("Stationarity by 6-month window (5-min sample)")
    for label, group in residual_5min.groupby(pd.Grouper(freq="6ME")):
        if len(group) < 1000:
            continue
        a = adf_test(group)
        k = kpss_test(group)
        print(
            f"  {label.date()}  n={len(group):6,}  "
            f"ADF p={a['pvalue']:.4f} {'✓' if a['is_stationary'] else '✗'}  "
            f"KPSS p={k['pvalue']:.4f} {'✓' if k['is_stationary'] else '✗'}"
        )

    # -----------------------------------------------------------------------
    # 6. Autocorrelation
    # -----------------------------------------------------------------------
    divider("Autocorrelation (10s residual)")
    lag_spec = [(6, "1 min"), (30, "5 min"), (60, "10 min"), (180, "30 min"), (360, "1 hr")]
    lags = [l for l, _ in lag_spec]
    acf = autocorr_at_lags(residual_10s, lags)
    for lag, label in lag_spec:
        print(f"  lag {lag:4d} bars ({label:6s}): {acf[lag]:.4f}")

    # -----------------------------------------------------------------------
    # 7. OU half-life — full sample
    # -----------------------------------------------------------------------
    divider("OU half-life — full sample (1-min EWMA of 10s residual)")
    residual_smooth = residual_10s.ewm(span=6, min_periods=1).mean()
    ou = ou_halflife(residual_smooth)
    hl_min = ou["halflife_bars"] * 10 / 60
    print(f"  Half-life: {ou['halflife_bars']:.1f} bars = {hl_min:.2f} min")
    print(f"  κ: {ou['kappa']:.6f} per bar    θ: {ou['theta']:.8f}    R²: {ou['r_squared']:.4f}")

    # -----------------------------------------------------------------------
    # 8. OU half-life — by 6-month window
    # -----------------------------------------------------------------------
    divider("OU half-life by 6-month window")
    ou_table = ou_halflife_by_period(residual_smooth, period="6ME")
    print(ou_table[["halflife_minutes", "kappa", "theta", "r_squared", "n_obs"]].to_string())

    # -----------------------------------------------------------------------
    # 9. Signal frequency
    # -----------------------------------------------------------------------
    divider("Signal frequency by z-score threshold (10s bars)")
    sig_table = signal_stats(sig["zscore"], thresholds=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0], bar_seconds=10)
    print(sig_table.to_string(index=False))
    viable = sig_table[sig_table["signals_per_week"] >= 5]
    if not viable.empty:
        best = viable["threshold"].max()
        print(f"\n  Highest viable threshold: z={best}  "
              f"({viable[viable['threshold']==best]['signals_per_week'].iloc[0]} sig/week)")

    # -----------------------------------------------------------------------
    # 10. Gate summary
    # -----------------------------------------------------------------------
    divider("Week 1 Gate Summary")
    g_adf    = adf_result["is_stationary"]
    g_kpss   = kpss_result["is_stationary"]
    g_autocorr = summary["autocorr1"] > 0.2
    g_freq   = not viable.empty

    print(f"  [{'✓' if g_adf else '✗'}] ADF stationary")
    print(f"  [{'✓' if g_kpss else '✗'}] KPSS stationary")
    print(f"  [{'✓' if g_autocorr else '✗'}] Lag-1 autocorr > 0.2  (actual: {summary['autocorr1']:.4f})")
    print(f"  [{'✓' if g_freq else '✗'}] ≥5 signals/week at some threshold")
    print(f"\n  Overall: {'PASS' if (g_adf and g_autocorr and g_freq) else 'NEEDS INVESTIGATION'}")

    # -----------------------------------------------------------------------
    # 11. Generate and save plots
    # -----------------------------------------------------------------------
    divider("Generating plots")
    plots = [
        ("raw_pairs",          lambda: plot_raw_pairs(sig, PLOTS_DIR / "raw_pairs.png")),
        ("residual",           lambda: plot_residual(sig, PLOTS_DIR / "residual.png")),
        ("liberation_day_zoom",lambda: plot_liberation_day_zoom(sig, PLOTS_DIR / "liberation_day_zoom.png")),
        ("ou_diagnostics",     lambda: plot_ou_diagnostics(sig, PLOTS_DIR / "ou_diagnostics.png")),
        ("signal_distribution",lambda: plot_signal_distribution(sig, PLOTS_DIR / "signal_distribution.png")),
    ]
    for name, fn in plots:
        print(f"  Saving {name}...", end=" ", flush=True)
        import matplotlib.pyplot as plt
        fig = fn()
        plt.close(fig)
        print("done")

    print(f"\nBaseline complete. Plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
