"""
Week 1 Baseline Analysis — EUR/USD/AUD Triangle

All computation runs at native 10-second bar resolution.

Outputs:
  1. Basic residual statistics
  2. ADF / KPSS stationarity tests (full sample + by half-year)
  3. Lag autocorrelation at multiple lags (expressed in wall-clock time)
  4. OU half-life table by 6-month window (run on 1-min EWMA of residual)
  5. Signal frequency table across z-score thresholds

Run from repo root:
  python notebooks/week1_baseline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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

DATA_DIR = Path(__file__).parent.parent / "data"

# EWMA half-life for z-score normalisation: 360 bars × 10s = 3600s = 1 hour
EWMA_HALFLIFE = 360


def divider(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# 1. Load data (native 10s resolution — no resampling)
# ---------------------------------------------------------------------------
divider("Loading data (10s bars)")
eurusd = load_pair(DATA_DIR, "EURUSD")
audusd = load_pair(DATA_DIR, "AUDUSD")
euraud = load_pair(DATA_DIR, "EURAUD")
print(f"EURUSD: {len(eurusd):,} bars  {eurusd.index[0].date()} → {eurusd.index[-1].date()}")
print(f"AUDUSD: {len(audusd):,} bars  {audusd.index[0].date()} → {audusd.index[-1].date()}")
print(f"EURAUD: {len(euraud):,} bars  {euraud.index[0].date()} → {euraud.index[-1].date()}")

# ---------------------------------------------------------------------------
# 2. Build signal frame (aligned + residual + z-score at 10s resolution)
# ---------------------------------------------------------------------------
divider("Building residual (10s resolution)")
sig = build_signal_frame(eurusd, audusd, euraud, ewma_halflife=EWMA_HALFLIFE)
print(f"Aligned bars: {len(sig):,}  ({sig.index[0].date()} → {sig.index[-1].date()})")
print(f"Alignment rate: {len(sig)/len(eurusd)*100:.1f}% of EURUSD bars")

# ---------------------------------------------------------------------------
# 3. Residual summary stats
# ---------------------------------------------------------------------------
divider("Residual summary statistics")
summary = residual_summary(sig["residual"])
for k, v in summary.items():
    print(f"  {k:12s}: {v:.6f}")

print(f"\n  Residual std = {summary['std']*10000:.2f} 'pips' (×10000)")
print(f"  Lag-1 autocorr (10s) = {summary['autocorr1']:.4f}  (gate: > 0.2)")
gate_autocorr = summary['autocorr1'] > 0.2
print(f"  Gate PASS? {'✓ YES' if gate_autocorr else '✗ NO'}")

# ---------------------------------------------------------------------------
# 4. Stationarity tests — full sample
# ---------------------------------------------------------------------------
divider("Stationarity tests — full sample residual")
# ADF/KPSS are computationally expensive at scale. Resample to 5-min bars
# (every 30th 10s bar) for the test — retains >140k observations which gives
# ample statistical power.
residual_10s  = sig["residual"]
residual_5min = residual_10s.resample("5min").last().dropna()
print(f"10s residual: {len(residual_10s):,} bars  |  5-min sample for ADF/KPSS: {len(residual_5min):,} bars")

adf = adf_test(residual_5min)
print(f"\n  ADF test:")
print(f"    stat={adf['stat']:.4f}  p={adf['pvalue']:.6f}  lags={adf['lags']}")
for cv, val in adf['critical_values'].items():
    print(f"    critical {cv}: {val:.4f}")
print(f"    Stationary (5%)? {'✓ YES' if adf['is_stationary'] else '✗ NO'}")

kpss_r = kpss_test(residual_5min)
print(f"\n  KPSS test:")
print(f"    stat={kpss_r['stat']:.4f}  p={kpss_r['pvalue']:.6f}  lags={kpss_r['lags']}")
print(f"    Stationary (5%)? {'✓ YES' if kpss_r['is_stationary'] else '✗ NO'}")

# ---------------------------------------------------------------------------
# 5. Stationarity tests — by half-year
# ---------------------------------------------------------------------------
divider("Stationarity tests — by 6-month window (5-min sample)")
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

# ---------------------------------------------------------------------------
# 6. Autocorrelation at multiple lags (wall-clock expressed in bar counts)
# ---------------------------------------------------------------------------
divider("Autocorrelation (10s residual, lags in bars + wall-clock)")
# Lag counts in 10s bars with wall-clock equivalents
lag_spec = [
    (6,    "1 min"),
    (30,   "5 min"),
    (60,   "10 min"),
    (180,  "30 min"),
    (360,  "1 hr"),
]
lags = [l for l, _ in lag_spec]
acf = autocorr_at_lags(residual_10s, lags)
for lag, label in lag_spec:
    print(f"  lag {lag:4d} bars ({label:6s}): {acf[lag]:.4f}")

# ---------------------------------------------------------------------------
# 7. OU half-life — full sample (on 1-min EWMA of residual)
# ---------------------------------------------------------------------------
divider("OU half-life — full sample (1-min EWMA of 10s residual)")
# Smooth the raw 10s residual with EWMA span=6 (≈ 1-min equivalent) before
# estimating OU parameters — running on raw 10s produces noise-dominated estimates.
residual_smooth = residual_10s.ewm(span=6, min_periods=1).mean()
ou = ou_halflife(residual_smooth)
hl_sec = ou['halflife_bars'] * 10  # 1 bar = 10s
hl_min = hl_sec / 60
print(f"  Half-life: {ou['halflife_bars']:.1f} bars = {hl_sec:.0f}s = {hl_min:.2f} min")
print(f"  κ (mean-reversion speed): {ou['kappa']:.6f} per bar")
print(f"  θ (long-run mean):        {ou['theta']:.8f}")
print(f"  R²: {ou['r_squared']:.4f}")

# ---------------------------------------------------------------------------
# 8. OU half-life — by 6-month window (1-min EWMA of 10s residual)
# ---------------------------------------------------------------------------
divider("OU half-life by 6-month window (1-min EWMA of 10s residual)")
ou_table = ou_halflife_by_period(residual_smooth, period="6ME")
print(ou_table[["halflife_minutes", "kappa", "theta", "r_squared", "n_obs"]].to_string())

# ---------------------------------------------------------------------------
# 9. Signal frequency by z-score threshold (10s bars, bar_seconds=10)
# ---------------------------------------------------------------------------
divider("Signal frequency by z-score threshold (10s bars)")
sig_table = signal_stats(
    sig["zscore"],
    thresholds=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    bar_seconds=10,
)
print(sig_table.to_string(index=False))
print(f"\n  Gate: ≥5 signals/week.")
viable = sig_table[sig_table["signals_per_week"] >= 5]
if not viable.empty:
    best_thr = viable["threshold"].max()
    row = viable[viable["threshold"] == best_thr].iloc[0]
    print(f"  Highest viable threshold: z = {best_thr}  ({row['signals_per_week']} sig/week)")
else:
    print("  ✗ No threshold clears 5 signals/week gate.")

# ---------------------------------------------------------------------------
# 10. Summary / gate check
# ---------------------------------------------------------------------------
divider("Week 1 Gate Summary")
gate1_adf     = adf["is_stationary"]
gate1_kpss    = kpss_r["is_stationary"]
gate2_autocorr = gate_autocorr
gate3_freq    = not viable.empty

print(f"  [{'✓' if gate1_adf else '✗'}] ADF stationary (p < 0.05)")
print(f"  [{'✓' if gate1_kpss else '✗'}] KPSS stationary (p > 0.05)")
print(f"  [{'✓' if gate2_autocorr else '✗'}] Lag-1 autocorr > 0.2 (actual: {summary['autocorr1']:.4f})")
print(f"  [{'✓' if gate3_freq else '✗'}] ≥5 signals/week at some threshold")

all_pass = gate1_adf and gate2_autocorr and gate3_freq
print(f"\n  Overall: {'PASS — proceed to Week 2' if all_pass else 'NEEDS INVESTIGATION'}")
