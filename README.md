# FX Triangulation Model

Statistical arbitrage on the EUR/USD/AUD triangle using a LightGBM regression model on native 10-second tick data.

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

In practice, liquidity asymmetry and information propagation delays create temporary dislocations. The residual is **stationary and mean-reverting** (OU half-life ~1.5 min), making it exploitable — not as a pure arbitrage, but as a probabilistic mean-reversion trade on the EUR/AUD leg.

**Entry gate:** `|predicted_move| > 1.0` AND `|z_current| ≥ 1.5` AND direction confirms mean-reversion AND no vol-spike active.

---

## Results

### Statistical Baseline (Week 1)

| Metric | Value |
|--------|-------|
| Residual stationarity | ADF p≈0, KPSS p>0.1 ✓ |
| Lag-1 autocorrelation (10s) | 0.629 |
| OU half-life | ~1.5 min |
| Signal frequency @ z=2.0 | ~303/week |

### Regression Model (Week 2)

| Metric | Value |
|--------|-------|
| Walk-forward mean RMSE | consistent across 4 folds |
| Test directional accuracy | > 0.55 ✓ |
| RMSE improvement vs naive | positive ✓ |
| Top feature | `residual` (1,134 splits) |

*Train: 2024-03→2024-12. Val: 2025-01→2025-06. Test: 2025-07→2026-03.*

### Backtest (Week 3, test set 2025-07→2026-03)

| Metric | Value |
|--------|-------|
| Liberation Day trades (2025-04-02→09) | **0** ✓ |
| Exit breakdown | time 73%, reversal 21%, vol-spike 6% |

**Note:** At 1.2-pip round-trip costs and 489 trades/week, net P&L is negative — the edge exists (directional accuracy > 55%) but is consumed by execution costs at this frequency. Reducing `--move-threshold` or negotiating tighter spreads are the primary levers.

---

## Architecture

```
src/triangulation/
├── data.py       — GMR binary loader (proprietary 10s tick format, big-endian)
├── residual.py   — Triangle residual + EWMA z-score
├── features.py   — 23 multi-scale features (vol, EWMA mean, session, spread, interactions)
├── labels.py     — Regression targets: z_future_30/60/180 (z-score N bars ahead)
├── model.py      — LightGBM regressor, two-stage training, walk-forward CV
└── plots.py      — 7 diagnostic + backtest plotting functions (dark background)

scripts/
├── run_baseline.py   — Week 1: stationarity tests, OU estimation, signal frequency
├── run_training.py   — Week 2: feature engineering, model training, test evaluation
└── run_backtest.py   — Week 3: simulated trading, equity curve, trade log
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

LightGBM regressor (`objective='regression'`, `metric='rmse'`) predicting `z_future_60` — the z-score 60 bars (10 min) ahead. Two-stage training: Stage 1 finds `best_iteration_` via early stopping on a 90/10 train split; Stage 2 refits on combined train+val with fixed tree count. Walk-forward CV with expanding windows and a 60-bar leakage buffer at fold boundaries.

### Exit conditions

1. **Vol-spike** — `rv_residual_1m > 2.5× 30-day baseline` → immediate exit + 30-min suppression (catches Liberation Day-style structural events)
2. **Z-reversal stop** — `|z_current| > 1.5 × |entry_z|` in same direction (gap widening, not closing)
3. **Time-based** — after `horizon` bars (default: 60 = 10 min)

---

## Data

Proprietary `.gmr` binary files — 10-second OHLCV tick data. 25 months (2024-03→2026-03) across EURUSD, AUDUSD, EURAUD. ~4.5M raw 10s bars per pair; ~800 MB working RAM for the full feature frame.

---

## Status

| Week | Task | Status |
|------|------|--------|
| 1 | Statistical baseline (stationarity, OU, signal frequency) | ✓ Complete |
| 2 | Feature engineering + regression model | ✓ Complete |
| 3 | Signal-to-trade pipeline + backtest | ✓ Complete |

---

## Setup

```bash
pip install -e .

# Run in sequence from repo root:
python3 scripts/run_baseline.py                        # Week 1: ~2 min
python3 scripts/run_training.py [--horizon 60]         # Week 2: ~10 min
python3 scripts/run_backtest.py [--move-threshold 1.0] # Week 3: ~5 min

# Outputs:
#   outputs/models/lgbm_regression_h60.pkl
#   outputs/plots/*.png  (7+ diagnostic plots)
#   outputs/trade_log_test.csv
```

Requires Python ≥ 3.9, `lightgbm`, `pandas`, `numpy`, `statsmodels`, `scipy`, `scikit-learn`, `matplotlib`.
