# FX Triangulation Model

Statistical arbitrage on the EUR/USD/AUD triangle using a LightGBM probability classifier on native 10-second tick data.

---

## Strategy

The EUR/USD/AUD triangle satisfies the no-arbitrage identity:

```
EUR/AUD_implied = EUR/USD ÷ AUD/USD
```

In log space, the triangle **residual** should be zero in an efficient market:

```
Δ(t) = ln(EURAUD) - ln(EURUSD) + ln(AUDUSD)  ≈ 0
```

In practice, liquidity asymmetry and information propagation delays create temporary dislocations. The residual is **stationary and mean-reverting** (OU half-life ~7 min), making it exploitable — not as a pure arbitrage, but as a probabilistic mean-reversion trade on the EUR/AUD leg.

**Entry gate:** only trade when `P(gap closes ≥50% within 10 min) > 0.65`. Direction from sign of z-score.

---

## Results

| Metric | Value |
|--------|-------|
| Residual stationarity | ADF p≈0, KPSS p>0.1 ✓ |
| Lag-1 autocorrelation (10s) | 0.81 |
| OU half-life | ~7 min |
| Signal frequency @ z=2.0 | ~1,760/week |
| Test AUC-ROC | **0.980** |
| Brier improvement vs naive | **+0.025** |
| Precision @ P>0.65 | **98%** |

*Train: 2024-03→2024-12. Val: 2025-01→2025-06. Test: 2025-07→2026-03.*

**Known limitation:** Liberation Day (2025-04-02) structural repricing produces false high-confidence signals — no macro regime features in the current model. Suppressed in Week 3 via vol-spike gate (Regime E).

---

## Architecture

```
src/triangulation/
├── data.py       — GMR binary loader (proprietary 10s tick format, big-endian)
├── residual.py   — Triangle residual + EWMA z-score
├── features.py   — 23 multi-scale features (vol, EWMA mean, session, spread, interactions)
├── labels.py     — Binary label: gap closes ≥50% within 60 bars (10 min)
└── model.py      — LightGBM classifier + Platt scaling + walk-forward CV

notebooks/
├── week1_baseline.py  — Stationarity tests, OU estimation, signal frequency
└── week2_train.py     — Feature engineering, walk-forward CV, train/eval pipeline
```

### Feature set (23 features)

| Group | Features |
|-------|---------|
| Core signal | `residual`, `zscore`, `dz_10s`, `d2z_10s` |
| Residual vol | `rv_residual_1m/5m/30m/4h` |
| EWMA drift | `ewma_mean_1m/5m/30m/4h` |
| Pair vol | `rv_eurusd/audusd/euraud_1h`, `rv_ratio_eu_aud` |
| Session | `hour_sin`, `hour_cos`, `is_london_ny` |
| Spread | `spread_euraud`, `spread_euraud_norm` |
| Interactions | `z_x_rv`, `z_x_spread` |

### Model

LightGBM binary classifier (`objective='binary'`, `metric='auc'`) with early stopping on AUC. Platt scaling (logistic regression on raw scores) for probability calibration. Walk-forward cross-validation with expanding windows and label-leakage buffer at fold boundaries.

---

## Data

Proprietary `.gmr` binary files — 10-second OHLCV + bid/ask tick data. 25 months (2024-03→2026-03) across EURUSD, AUDUSD, EURAUD. ~4.5M raw 10s bars per pair; ~800 MB working RAM for the full feature frame.

---

## Status

| Week | Task | Status |
|------|------|--------|
| 1 | Statistical baseline (stationarity, OU, signal frequency) | ✓ Complete |
| 2 | Feature engineering + ML classifier | ✓ Complete |
| 3 | Signal-to-trade pipeline + backtest | In progress |

Week 3 will build the deterministic trade execution logic (3 exit conditions, Kelly sizing), backtest on the test set, and implement Liberation Day-style regime suppression via vol-spike detection.

---

## Setup

```bash
pip install -e .
python notebooks/week2_train.py
```

Requires Python ≥ 3.9, `lightgbm`, `pandas`, `numpy`, `statsmodels`, `scipy`, `scikit-learn`.
