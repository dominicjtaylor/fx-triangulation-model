# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Commands

```bash
# Run Week 1 baseline analysis (loads data, stationarity tests, OU half-life, signal freq)
python notebooks/week1_baseline.py
```

---

## Project Overview

An **ML-driven FX triangulation strategy** targeting the **EUR/USD/AUD triangle** (data available: EURUSD, AUDUSD, EURAUD). The strategy is **statistical triangulation** — not pure triangular arbitrage (which requires HFT co-location). The edge is probabilistic: the triangle residual exhibits mean-reversion tendencies driven by liquidity asymmetry and information propagation delays after macro events.

The research brief is in `skills/fx_triangulation_brief.md`. It is the authoritative source for this project — read it before making architectural decisions.

---

## Core Formula

```python
# Triangle residual — should be near zero in efficient markets
# EUR/AUD_implied = EUR/USD ÷ AUD/USD  →  ln(implied) = ln(EURUSD) - ln(AUDUSD)
Δ = ln(EURAUD_actual) - ln(EURUSD) + ln(AUDUSD)

# Z-score signal using time-varying EWMA mean/std (handles slow structural drift)
z_t = (Δ_t - μ_t) / σ_t
```

- `z_t > 0` → EUR/AUD expensive vs implied → **SELL EUR/AUD**
- `z_t < 0` → EUR/AUD cheap vs implied → **BUY EUR/AUD**

Entry gate: only trade when `P(closure) > θ` (e.g., 0.65), output of the ML model.

---

## Architecture

### What's Being Built

A **new triangulation ML model** built on the same infrastructure as an existing volatility forecasting model. Same data-loading conventions, feature engineering patterns, train/test methodology, and deployment pipeline — different target and input features.

### ML Model Target

**Target A (primary):** Binary classifier — does the gap close ≥50% within 10 minutes?
Output: `P(closure)` ∈ [0, 1]. Apply Platt scaling for calibration.

**Target B (extension):** Regression — what fraction of the gap closes? Drives position sizing directly.

### Feature Set (~20–30 features)

```python
# Core signal
z_t, dz_t, d2z_t, halflife_t (rolling 60-min OU estimation)

# Volatility regime
rv_eurusd_1h, rv_gbpusd_1h, rv_eurgbp_1h
rv_ratio_eu_gbp = rv_eurusd_1h / rv_gbpusd_1h  # vol leadership
rv_triangle     = std(residual, rolling 60 mins)

# Liquidity
spread_eurgbp_norm = current spread / 30-day average
spread_ratio       = spread_eurgbp / (spread_eurusd + spread_gbpusd)

# Session/event timing
hour_of_day (sin/cos encoded), is_london_ny_overlap
mins_to_event, mins_since_event  # from economic calendar

# Macro regime (daily, slow-moving)
ecb_boe_spread, rate_diff_trend (5-day change)
eurusd_5d_mom, gbpusd_5d_mom, eurgbp_5d_mom

# Interactions
z_x_rv = z_t * rv_triangle        # gap size × vol regime
z_x_spread = z_t * spread_norm    # gap size penalised by execution cost
```

### Train / Validate / Test Split

```
Training:    2020-01-01 → 2023-12-31   (4 years)
Validation:  2024-01-01 → 2024-06-30   (hyperparameter tuning)
Test:        2024-07-01 → 2025-12-31   (held out — includes 2025 tariff regime)
```

Walk-forward within training: 6-month rolling windows, 1-month OOS evaluation. Apply 10-bar buffer between folds to prevent label leakage.

### Evaluation Metrics

| Metric | Target |
|--------|--------|
| AUC-ROC | > 0.58 useful, > 0.65 strong |
| Precision at P > 0.65 | > 60% correct closure calls |
| Brier score | minimise (calibration quality) |
| Simulated Sharpe (test set, after costs) | > 0.8 annualised |

Prioritise calibration over raw AUC — position sizing depends on well-calibrated probabilities.

### Signal Pipeline

```
1. Compute z_t (triangle residual z-score with EWMA mean/std)
2. Assemble feature vector
3. Model inference → P(closure)
4. Entry gate: P(closure) > θ (e.g., 0.65)
5. Direction from sign of z_t
6. Position size = Base_size × (P(closure) - 0.5) / 0.5 × Kelly_fraction
   (Kelly fraction: 0.25–0.35 conservatively in early deployment)
7. Exit: time-based at 2× OU half-life; z-score reversal stop at 1.5× entry z;
   vol spike stop if EUR/GBP rv > 2× forecast mid-trade
```

### Regime Table

| Regime | Vol State | Policy State | Action |
|--------|-----------|--------------|--------|
| A | Low | Stable | Reduce size, tightest threshold |
| B | Medium | Stable | Full size, standard threshold — **sweet spot** |
| C | Medium | Diverging | Adjusted EWMA mean, medium size |
| D | High | Event-driven | Reduced size, fast exit, higher threshold |
| E | High | Regime-change | **No trade**, monitor only |

---

## Execution Design

**Preferred approach: Trade EUR/GBP leg only (Approach A).** One position, one spread, clean P&L attribution. The USD exposure is not hedged — the bet is simply that EUR/GBP actual converges to implied.

Full 3-leg hedge (Approach C) is only viable with simultaneous multi-leg execution on a Tier-1 ECN.

**Minimum signal threshold:** 3–5 pips gap to break even at retail ECN costs. Target 5–10+ pips for positive expectation. Round-trip execution cost: ~1.5–3.0 pips in normal conditions.

**Optimal execution window:** 2–5 minutes post-catalyst. First 30–60 seconds has widest spreads; gap is still open but spreads normalise in this window.

**Venues:** Tier-1 ECN (EBS/Refinitiv) for institutional; LMAX Exchange or Interactive Brokers for development/paper trading.

---

## Data Sources

- **HistData.com** — Free 1-minute OHLCV back to 2000. Sufficient for backtesting.
- **Dukascopy** — Free tick data with bid/ask for spread analysis. Register to access.
- **ForexFactory calendar** — Economic event timing and surprise magnitude.
- **FRED / ECB SDW** — Central bank rate series (free).
- **NY Fed Implied Vol** — 1-month ATM implied vol for major USD pairs (free, regime indicator).

Data quality requirement: all three pairs must be on the same bar clock. Timestamp offset > 1 second introduces spurious residual signal — this is the most common early-stage failure.

---

## Key Historical Events (Stress Tests for the Model)

| Event | Date | What Happened |
|-------|------|---------------|
| Liberation Day | Apr 2, 2025 | USD fell despite tariffs; EUR +1.5–2% vs USD, GBP +0.8–1.2%; EUR/GBP lagged implied by 15–45 mins — largest triangle gap of 2025. The model must **suppress** signals here (structural repricing, not mean-reversion). |
| BoE Hawkish Cut | Aug 7, 2025 | 5-4 vote, cautious guidance; GBP/USD jumped, EUR/USD flat; EUR/GBP actual lagged implied by ~15–20 pips for 10 mins — **clean triangulation event**, model should trade. |
| ECB December Pause | Dec 2025 | Hold surprise; EUR/USD spiked; EUR/GBP lagged implied by 10–15 pips for 5–10 mins — **tradeable** event. |

The test set (2024 H2–2025) spans these events. The model's handling of each is a qualitative validation check.

---

## 3-Week Implementation Plan

### Week 1 — Data and Baseline Statistics
- Synchronised 1-min data for all 3 pairs from single venue
- Compute residual series; ADF/KPSS stationarity tests
- OU half-life table by year (2020–2025)
- Signal frequency and gap size at various z-score thresholds

**Gate:** Residual stationary in ≥1 regime, lag-1 autocorrelation > 0.2, ≥5 signals/week at chosen threshold.

### Week 2 — Feature Engineering and ML Training
- Build 20–30 feature set; construct binary target with 10-bar buffer
- Train on existing pipeline architecture (XGBoost/LGBM or LSTM — match vol model)
- Calibrate with Platt scaling on validation set
- Evaluate on test set; specifically examine 2025 stress episodes

**Gate:** Test AUC-ROC > 0.57, calibration error < 0.05, no high-confidence signals during Liberation Day structural repricing.

### Week 3 — Simulated Trading
- Build deterministic signal-to-trade pipeline with all three exit conditions
- Backtest on test set in event-time; generate trade log with P&L attribution
- Sensitivity: 2× spread, varying P threshold (0.55–0.75), varying Kelly (0.15–0.50)
- Paper trading on live LMAX or IB demo feed

**Gate:** Backtest Sharpe > 0.7 after costs (> 0.4 at 2× spreads), max drawdown < 8%, live signal frequency within 20% of backtest.

---

## Data Format — `.gmr` Binary Files

Raw data lives in `data/tick10s-mid-{SYMBOL}-YYYY-MM.gmr`. Format (discovered empirically):

```
Header (9 bytes):
  [0:4]  version   big-endian int32 = 1
  [4:8]  row_count big-endian int32
  [8]    reserved  byte = 1

Row (56 bytes each):
  [0:8]   timestamp  big-endian int64 (milliseconds since Unix epoch)
  [8:16]  open       big-endian float64
  [16:24] high       big-endian float64
  [24:32] low        big-endian float64
  [32:40] close      big-endian float64
  [40:48] f4         big-endian float64 (unknown auxiliary)
  [48:56] f5         big-endian float64 (unknown auxiliary)
```

Loader is in `src/triangulation/data.py`. Always call `load_pair(..., resample='1min')` — the default — to avoid OOM with 4.5M raw 10s bars per pair. The numpy structured array must be cast to little-endian (`astype('<f8')`) before pandas operations.

## Week 1 Results (March 2024 – March 2026)

- **752K** aligned 1-min bars across all three pairs (99.3% alignment rate)
- Residual std ≈ 0.45 "pips" (×10000), mean ≈ 0 ✓
- **Strongly stationary**: ADF p≈0, KPSS p>0.05, both full-sample and most 6-month windows
- **Lag-1 autocorr = 0.629** (gate > 0.2: ✓)
- **OU half-life ≈ 1.5 min** overall (ranges 0.08–1.2 min by period) — very fast mean-reversion
- Signal frequency: 303/week at z=2.0, 63/week at z=3.0 (gate ≥5/week: ✓ at all thresholds)
- **Week 1 gate: ALL PASS → proceed to Week 2**

Notable: residual has extreme kurtosis (~4484) and negative skew — fat tails from news events. Position sizing must account for this.

## Week 2 Results

- **23 features** built at native 10s resolution (multi-scale RV, EWMA mean, per-pair vol, session, spread, interactions)
- **Binary label**: gap closes ≥50% within 60 bars (10 min wall-clock). Label rate: 94.4% overall, 98%+ at signal bars (|z| ≥ 2)
- **Splits** (adapted to available 2024-03→2026-03 data): train 2024-03→2024-12, val 2025-01→2025-06, test 2025-07→2026-03
- **Walk-forward**: 4 folds, mean AUC 0.980 (baseline residual+zscore only: 0.976), consistent +0.004 improvement from full feature set
- **Test set**: AUC-ROC = **0.980**, Brier = 0.0258 (vs naive 0.051), precision@P>0.65 = 0.980
- **Week 2 quantitative gates: ALL PASS**
- **Liberation Day (qualitative)**: model assigns P(closure) ≥ 0.65 to 355/355 signal bars — expected failure, no macro regime features to detect structural repricing. Week 3 must add a vol-spike suppression rule (Regime E).

Notable: `residual` is the single most important feature (1134 splits), followed by `ewma_mean_4h`, time-of-day features, and 4-hour RV. Spread (`spread_euraud_norm`) ranks 14th — meaningful but not dominant.

## Python Stack

`pandas`, `numpy`, `statsmodels` (ADF/KPSS), `scipy` (OU estimation), `lightgbm`, `scikit-learn` (Platt calibration). Python 3.9+ (uses `typing.Optional` not union syntax).
