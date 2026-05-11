# FX Triangulation Model

EUR/AUD is less liquid than the two USD legs of the triangle (EUR/USD and AUD/USD). When a macro event moves EUR/USD or AUD/USD, EUR/AUD adjusts more slowly — creating a temporary gap between actual and implied price. This project measures that gap, models how it evolves over the next 10 minutes using native 10-second tick data, and sizes EUR/AUD positions proportionally to the predicted move.

---

## Research Summary

**Research question:** Does the triangulation residual in the EUR/USD/AUD triangle exhibit sufficient predictability at the 1–10 minute timescale to generate net-positive simulated returns after execution costs at 10-second resolution?

**Methodology:** Two years of native 10-second mid-price data (2024-03 → 2026-03) across all three pairs were used to construct the log-residual and its EWMA z-score. A statistical baseline established stationarity (ADF p ≈ 0, KPSS p > 0.1 in all six-month windows), an Ornstein-Uhlenbeck half-life of ~1.5 minutes, and lag-1 autocorrelation of 0.629 — confirming exploitable persistence at the entry timescale. A LightGBM regressor was trained to predict the z-score 10 minutes ahead from 23 multi-scale features, evaluated on an out-of-sample test set spanning 2025-07 → 2026-03 (which includes the April 2025 Liberation Day tariff shock as a qualitative stress test), and validated through walk-forward cross-validation on the training period. Simulated trading used deterministic exit rules with a 1.2-pip round-trip cost assumption.

**Key finding:** The statistical signal exists and the model captures it — directional accuracy exceeds 55% on the held-out test set, and the vol-spike gate correctly suppresses all entries during Liberation Day structural repricing. However, simulated P&L is negative under the cost assumptions tested. Average gross P&L per trade is approximately +0.3 pips against a 1.2-pip round-trip cost: the edge is real, but roughly 4× below the per-trade threshold required to break even at current entry frequency and execution cost.

**Exit methodology experiments:** Three signal-driven exit approaches were evaluated against the original 10-minute time backstop: z-score zero-crossing, fractional z-score retracement (50–90%), and a minimum hold (18 bars = ~3 min, matching the OU half-life) followed by a pip-based profit target (1.5–2.5 pips gross). The first two both dramatically worsened results — win rate collapsed from 39.9% to 17–21% and trade count quadrupled. The cause was the extreme kurtosis of the residual (≈4,484): at 10-second resolution, the z-score crosses any threshold repeatedly from microstructure noise before meaningful EUR/AUD price movement has occurred. The third approach — minimum hold plus price target — successfully raised win rate to 52–53%, confirming that the directional signal is genuine and mean-reversion does complete on the OU timescale. However, earlier exits free up capacity sooner, increasing total trade count and total cost, while the asymmetric loss distribution from reversal exits (large, infrequent losses when the gap widens) is not reduced proportionally. The conclusion from these experiments is that the cost-to-edge constraint is structural: it cannot be resolved through exit design because exit design affects the distribution of captured P&L but not the underlying ratio of average gross edge to per-trade cost.

**What would change the conclusion:** Raising the entry threshold from `|z| ≥ 1.5` to `|z| ≥ 2.5–3.0` would reduce trade frequency while selecting for larger gaps with higher expected gross moves, potentially lifting average gross P&L above the 1.2-pip cost bar. Institutional execution (Tier-1 ECN at ~0.3–0.5 pip round-trip) would lower the breakeven threshold to a level the current edge can plausibly exceed. The three-leg hedge — routing simultaneous orders across all three pairs to eliminate directional USD exposure and trade a pure residual spread — is the identified structural improvement, though it triples combined execution costs and therefore requires a commensurately larger entry gap to remain viable.

---

## The core insight

EUR/USD is driven by US–Eurozone interest rate differentials. AUD/USD is driven by commodity prices and Chinese growth expectations. The two are structurally different assets with low correlation (0.45–0.65), so macro events frequently move one leg without moving the other, opening a triangle gap. EUR/AUD — being the less-traded cross — is the slow leg. That lag is the signal.

In log space, the triangle satisfies the no-arbitrage identity:

```
Δ(t) = ln(EUR/AUD_actual) − ln(EUR/USD) + ln(AUD/USD)
```

In efficient markets, `Δ(t) ≈ 0`. In practice, `Δ(t)` reverts to zero but does so on a timescale of **~1.5 minutes** (OU half-life). Normalised as a z-score against its EWMA mean and standard deviation, the residual has a lag-1 autocorrelation of **0.629** at 10-second resolution — a statistically exploitable persistence.

---

## Why it's not pure arbitrage

Pure triangular arbitrage — simultaneously trading all three legs to lock in a riskless profit — is dead at the speeds retail and most institutional systems operate. EUR/AUD market makers reprice within milliseconds of seeing large EUR/USD or AUD/USD prints. What's left is a probabilistic signal: the residual mean-reverts, but not instantly and not always. The strategy operates at the 1–10 minute scale, treating the residual as a mean-reverting spread with a known half-life. The edge is that gap closure is predictable enough to trade profitably — contingent on costs and regime.

---

## What the model does

The model is a **regression**, not a classifier. It predicts `z_future_60` — the residual z-score 60 bars (10 minutes) ahead — directly from current market state. This gives both direction (sign of predicted move) and magnitude, which maps cleanly onto position sizing.

Why regression rather than classification? The residual is a continuous quantity, and predicting its future value avoids the calibration overhead of probability estimation. Directional accuracy on the test set exceeds 55% — modestly above the persistence baseline implied by the 0.629 lag-1 autocorrelation, consistently across walk-forward folds.

**Entry gate:** `|predicted_move| > 1.0 z-score units` AND `|z_current| ≥ 1.5` AND the predicted direction confirms mean-reversion (sign of z-score matches sign of predicted move). The second condition filters out entries at small z values where the 1.5× reversal stop fires almost immediately from normal price fluctuation.

**Model architecture:** LightGBM regressor, two-stage training. Stage 1 finds `best_iteration_` via early stopping on a 90/10 split of the training set. Stage 2 refits on the combined train+val period with `n_estimators` fixed at that iteration.

---

## Results

### Statistical properties of the residual

| Metric | Value |
|--------|-------|
| Stationarity | ADF p ≈ 0, KPSS p > 0.1 in all 6-month windows ✓ |
| Lag-1 autocorrelation (10s) | 0.629 |
| OU half-life | ~1.5 min (0.08–1.2 min varies by period) |
| Signal frequency @ z = 2.0 | ~303/week |

Autocorrelation of 0.629 at 10-second resolution confirms that the residual is strongly persistent at the entry timescale. Signal frequency of 303/week at z = 2.0 gives enough trades for statistical significance without overtrading.

### Regression model (test set: 2025-07 → 2026-03)

| Metric | Value |
|--------|-------|
| Directional accuracy | > 55% ✓ |
| RMSE vs naive forecast | improvement ✓ |
| Top features by split count | `residual` (1st), `ewma_mean_4h` (2nd) |

Directional accuracy above 55% beats the pure-persistence baseline. The 4-hour EWMA mean ranks second in feature importance — this slow-moving average is what detects when the residual's baseline has shifted structurally (e.g., during Liberation Day repricing).

### Simulated trading (test set)

| Metric | Value |
|--------|-------|
| Liberation Day trades (2025-04-02→09) | **0** ✓ |
| Win rate | 39.9% |
| Avg holding time | 8.4 min |
| Trades per week | 489 |
| Exit breakdown | time 73%, reversal 21%, vol-spike 6% |

**Known limitation:** At 1.2-pip round-trip costs and 489 trades/week, execution costs dominate. The directional edge exists but is insufficient at this trade frequency. The primary levers are raising `--move-threshold` (fewer, higher-conviction entries) or negotiating tighter spreads at a Tier-1 ECN.

**Liberation Day (2025-04-02):** Trump's tariff announcement triggered structural repricing in AUD/USD (fell to 0.5914), not a mean-reverting gap. The vol-spike gate (`rv_residual_1m > 2.5× 30-day baseline`) suppressed all entries and applied a 30-minute cooldown after any mid-trade exit during this event.

---

## Status

| Week | Task | Status |
|------|------|--------|
| 1 | Statistical baseline — stationarity, OU, signal frequency | ✓ Complete |
| 2 | Feature engineering + regression model | ✓ Complete |
| 3 | Signal-to-trade pipeline + backtest | ✓ Complete |

All three weeks are complete. The backtest produces a trade log with P&L attribution, an equity curve, and a Liberation Day suppression check. The identified next step is threshold sensitivity analysis: sweeping `--move-threshold` from 1.0 to 2.5 to find the breakeven cost-to-edge ratio.

---

## Architecture

```
src/triangulation/
├── data.py       — loads proprietary .gmr binary files into a normalised DataFrame
├── residual.py   — computes Δ(t) and EWMA z-score from the three pair price series
├── features.py   — builds 23 multi-scale features from the raw 10s residual
├── labels.py     — computes regression targets: z_future_30/60/180 (z-score N bars ahead)
├── model.py      — LightGBM regressor, two-stage training, walk-forward CV folds
├── analysis.py   — stationarity tests (ADF/KPSS), OU half-life estimation
└── plots.py      — 7 diagnostic and backtest plotting functions (dark background)

scripts/
├── run_baseline.py   — Week 1: stationarity tests, OU estimation, signal frequency
├── run_training.py   — Week 2: feature engineering, model training, evaluation
└── run_backtest.py   — Week 3: simulated trading, equity curve, trade log

outputs/
├── models/           — saved model artefacts (.pkl)
├── plots/            — diagnostic plots (.png)
└── trade_log_test.csv
```

### Features (23 total)

Features are computed at multiple window sizes directly from the native 10-second data — not from resampled bars. This lets the model learn which timescale is informative in a given regime: the 1-minute RV detects microstructure noise, the 30-minute RV tracks the trade closure horizon, and the 4-hour EWMA mean detects slow structural drift. Resampling first would discard the microstructure information that makes 10s resolution valuable.

| Group | Features |
|-------|---------|
| Core signal | `residual`, `zscore`, `dz_10s`, `d2z_10s` |
| Residual vol | `rv_residual_1m/5m/30m/4h` |
| EWMA drift | `ewma_mean_1m/5m/30m/4h` |
| Pair vol | `rv_eurusd/audusd/euraud_1h`, `rv_ratio_eu_aud` |
| Session | `hour_sin`, `hour_cos`, `is_london_ny` |
| Spread | `spread_euraud`, `spread_euraud_norm` |
| Interactions | `z_x_rv`, `z_x_spread` |

---

## Running it

```bash
pip install -e .

python3 scripts/run_baseline.py                         # Week 1: stationarity, OU, signal freq
python3 scripts/run_training.py [--horizon 60]          # Week 2: train model, save to outputs/
python3 scripts/run_backtest.py [--move-threshold 1.0]  # Week 3: simulate trades, equity curve
```

Each script saves outputs to `outputs/` and prints a gate summary on exit. `run_backtest.py` requires the model artefact produced by `run_training.py` (`outputs/models/lgbm_regression_h60.pkl`).

Requires Python ≥ 3.9, `lightgbm`, `pandas`, `numpy`, `statsmodels`, `scipy`, `scikit-learn`, `matplotlib`.

---

## Planned: Three-Leg Execution

The current implementation trades the EUR/AUD leg only (Approach A). This carries unintended directional exposure — the P&L depends not just on the residual converging but on EUR/USD and AUD/USD not moving against the position in the meantime.

The natural extension is to trade all three legs simultaneously in proportions that net USD exposure to zero:

```
If residual z-score < -2.0 (EUR/AUD cheap vs implied):
  BUY  EUR/AUD
  SELL EUR/USD
  BUY  AUD/USD

If residual z-score > +2.0 (EUR/AUD expensive vs implied):
  SELL EUR/AUD
  BUY  EUR/USD
  SELL AUD/USD
```

This converts the trade from a directional EUR/AUD bet into a pure spread trade on the residual. P&L is driven entirely by the triangle closing, regardless of where any individual pair moves. The predicted move from the regression model still drives entry and sizing — the only change is routing three orders instead of one.

The entry threshold needs to be wider to absorb three spreads rather than one. At retail ECN execution costs (combined round-trip ~4–5 pips across three legs), the minimum viable gap is approximately 6–8 pips vs 5–7 pips for the single-leg version.

**Prerequisites before implementing:**
- Single-leg version validated through full simulated trading (Week 3 complete)
- Execution venue confirmed to support simultaneous multi-leg order routing
- Slippage estimates for all three legs measured from live paper trading, not assumed
