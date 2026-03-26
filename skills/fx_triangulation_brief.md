# FX Triangulation: AUD-USD-EUR Strategy Brief
### Building a Triangulation Strategy Using Your Existing ML Pipeline
**Date: March 2026 | Status: Working Document**

---

## Table of Contents

1. [FX Triangulation — Core Mechanics](#1-fx-triangulation--core-mechanics)
2. [AUD-USD-EUR Structural Dynamics](#2-aud-usd-eur-structural-dynamics)
3. [2025 Macro Context and Triangle Stress](#3-2025-macro-context-and-triangle-stress)
4. [Regimes and Exploitable Patterns](#4-regimes-and-exploitable-patterns)
5. [Making It Tradable](#5-making-it-tradable)
6. [Research and Data Pointers](#6-research-and-data-pointers)

---

## 1. FX Triangulation — Core Mechanics

### 1.1 The Cross-Rate Relationship

The AUD-USD-EUR triangle is defined by three exchange rates with a hard algebraic constraint:

```
EUR/AUD_implied = EUR/USD / AUD/USD
```

In market convention: EUR/USD is quoted as dollars per euro; AUD/USD as dollars per Australian dollar. Dividing cancels the USD leg and gives the implied rate of euros per Australian dollar. The directly quoted EUR/AUD rate on ECN platforms should, in an efficient market, equal this implied value at all times.

The *triangle residual* is:

```
Delta = ln(EUR/AUD_actual) - ln(EUR/USD / AUD/USD)
      = ln(EUR/AUD_actual) - ln(EUR/USD) + ln(AUD/USD)
```

When Delta != 0, the market is pricing EUR/AUD inconsistently with the two USD-leg quotes. The sign tells you the direction:

- **Delta > 0:** EUR/AUD actual is above implied → EUR/AUD is expensive relative to the USD legs → sell EUR/AUD
- **Delta < 0:** EUR/AUD actual is below implied → EUR/AUD is cheap → buy EUR/AUD

**Why the AUD-USD-EUR triangle is structurally different from EUR-GBP-USD:**
EUR and GBP are both European financial-centre currencies with broadly similar macro sensitivity profiles. AUD is a commodity and risk currency with strong China linkage and significant idiosyncratic volatility. The EUR/AUD correlation with EUR/USD is structurally lower (~0.45–0.65) than EUR/GBP's correlation with EUR/USD (~0.75–0.90). This lower co-movement means triangle gaps open more frequently and stay open longer — but it also means closure is less certain, because AUD-specific drivers (iron ore, China PMI, commodity flows) can move EUR/AUD away from the implied level for hours rather than minutes.

### 1.2 Triangular Arbitrage in Theory vs. Practice

**Pure triangular arbitrage** requires locking in all three legs simultaneously at quoted prices. In practice:

- **HFT compression:** Sub-millisecond monitoring by co-located market makers on EBS/Refinitiv eliminates pure arb gaps before non-HFT participants can react. By the time any unco-located system observes a gap, it is already closing.
- **The spread problem:** EUR/USD: 0.2–0.5 pips; AUD/USD: 0.3–0.8 pips; EUR/AUD: 1.0–2.5 pips in normal conditions. The combined round-trip cost of three legs is 2–5 pips. Pure arb discrepancies, when they appear, are typically 0.5–2 pips wide — wholly consumed by execution costs.
- **EUR/AUD is less liquid than EUR/GBP.** EUR/AUD daily turnover is approximately $25–40bn versus $80–100bn for EUR/GBP (BIS estimates). EUR/AUD has wider spreads, thinner books, and slower price discovery.
- **Commodity-driven AUD moves are sudden.** An iron ore flash-move or China PMI miss can shift AUD/USD by 30–50 pips in seconds. The EUR/AUD cross adjusts with a lag — but that lag is accompanied by extreme spread widening, making pure arb dangerous to execute mechanically.

**Conclusion for practice:** Pure triangular arbitrage in the AUD-USD-EUR triangle is not viable without co-located HFT infrastructure on Tier-1 ECNs. This is not the strategy.

### 1.3 Statistical/Probabilistic Triangulation — What We Actually Care About

The edge is that the triangle residual exhibits predictable mean-reversion exploitable at meaningful holding times. Evidence for this in the AUD triangle:

**1. EUR/AUD is a slow-adjusting cross.** AUD/USD is highly reactive to risk-off flows, commodity news, and China data — often moving sharply before EUR/AUD has time to adjust. EUR/USD moves on Fed/ECB news independently. The EUR/AUD cross inherits both shocks but adjusts later due to lower liquidity and fewer dedicated market makers.

**2. The residual autocorrelation is higher than in EUR-GBP-USD.** Because AUD has more idiosyncratic drivers that don't move EUR/USD, triangle gaps open more frequently and the EUR/AUD quote adjusts more slowly. At 1-minute resolution, expect positive autocorrelation of 0.4–0.7 at lag-1, decaying over 10–60 minutes depending on regime.

**3. Price discovery leadership is clearly structured.** AUD/USD leads during commodity and risk-sentiment events; EUR/USD leads during US and Eurozone data events; EUR/AUD follows both and is the last to adjust. This directional lag is tradeable.

**4. China shock asymmetry is a repeating pattern.** Negative China data hits AUD/USD far harder than EUR/USD. The implied EUR/AUD rises faster than actual EUR/AUD quotes adjust. This is a systematic, recurring source of triangle stress that does not exist in EUR-GBP-USD.

**Key mental model:** The AUD triangle is more volatile than the EUR/GBP triangle and gaps are larger, but the closure signal is noisier. The strategy's edge is in correctly identifying whether a gap is driven by a transient liquidity shock (close reliably, trade it) versus a fundamental re-rating of AUD's China/commodity outlook (does not close cleanly, do not trade it). This distinction is the central modelling challenge — and it is precisely what your ML pipeline is suited to learning.

---

## 2. AUD-USD-EUR Structural Dynamics

### 2.1 AUD/USD — What Drives It

AUD/USD is the world's fifth most-traded currency pair, representing approximately 5% of global FX turnover. It is the prototypical commodity and risk currency pair, and its drivers are qualitatively different from any EUR-cross:

**Commodity prices — the dominant structural driver:**
Australia is the world's largest exporter of iron ore and a major exporter of coal, LNG, and agricultural commodities. Iron ore price and AUD/USD have a documented correlation of approximately 0.76 over long periods. When iron ore rallies, AUD/USD reliably follows. This linkage has no analogue in EUR/USD or EUR/GBP.

**China demand — the meta-driver behind commodities:**
Approximately 35% of Australia's export revenue comes from China (primarily iron ore and coal for steel production). AUD is essentially a *proxy for Chinese economic activity* in FX markets. A positive China PMI print moves AUD/USD within seconds. A Chinese property sector stress event crushes AUD. This China-proxy characteristic is a key source of idiosyncratic AUD/USD volatility relative to EUR/USD.

**Risk sentiment:**
AUD/USD has a strong positive correlation with global risk appetite. When VIX spikes, AUD falls sharply — both because investors reduce commodity exposure and because the USD rally compounds the move. In the most acute risk-off episodes, AUD/USD can drop 2–4% in a single session. This risk-beta is structurally higher than GBP/USD's.

**RBA monetary policy:**
The Reserve Bank of Australia sets the cash rate. The RBA has historically been cautious relative to other central banks — slow to cut, slow to raise. The cash rate level relative to Fed Funds directly affects carry trade flows into/out of AUD.

**Terms of trade:**
Australia's terms of trade track commodity cycles closely. In commodity booms, ToT improves, AUD rises structurally. In busts, the reverse.

**The USD leg:** A significant component of AUD/USD variance is dollar-driven, not AUD-driven. USD moves propagate into AUD/USD and EUR/USD simultaneously — the shared USD beta is the mechanical basis of the triangle.

### 2.2 EUR/USD — What Drives It

EUR/USD is the world's most liquid FX pair (~24% of global turnover). Its primary drivers: US-Eurozone growth and rate differentials (2-year swap spread is the standard proxy); risk sentiment (EUR/USD has moderate negative VIX correlation in some regimes); Eurozone current account surplus (structural EUR bid); and ECB forward guidance.

**The key structural fact for the AUD triangle:** The correlation between EUR/USD and AUD/USD is moderate (~0.45–0.65), not high. EUR/USD is driven by US-Eurozone dynamics; AUD/USD is driven by commodities, China, and risk sentiment. Their frequent divergence is the primary source of triangle gaps. This is a structural advantage over EUR-GBP-USD (where EUR/USD and GBP/USD are correlated ~0.85 and gaps are smaller).

EUR/USD is the price discovery leader for USD risk. Fed decisions, US macro data, and US political events hit EUR/USD first. AUD/USD and EUR/AUD follow.

### 2.3 EUR/AUD — What Drives It

EUR/AUD is the cross with no direct USD exposure. Less frequently discussed in practitioner literature than EUR/GBP but with equally well-defined structural drivers:

**RBA vs. ECB policy differential — primary structural driver:**
When the RBA is relatively hawkish (higher rates, slower cutting) versus the ECB, AUD outperforms EUR and EUR/AUD falls. In 2025 this was the dominant trend driver in H2: ECB cut aggressively to 2.0% while RBA paused at 3.60%, leaving the 160bps rate differential supportive of AUD.

**China shock sensitivity — AUD-specific, not EUR:**
Negative China developments move AUD/USD but not EUR/USD. EUR/AUD therefore rises sharply on China shocks. This is a clean directional signal for EUR/AUD with no equivalent in EUR/GBP.

**EUR/AUD is a late and slow mover:**
EUR/AUD has fewer dedicated market makers than EUR/USD or AUD/USD. When both USD legs move, EUR/AUD quote adjusts with a lag of minutes rather than seconds. This structural slowness is the mechanical source of triangulation opportunity — and is more pronounced than in EUR/GBP due to lower liquidity.

**Risk-off amplification:**
In risk-off events, AUD falls (risk-off selling) while EUR is relatively stable. EUR/AUD rises in risk-off, creating large, rapid triangle gaps that are directionally clear but volatile to trade.

**Carry trade dynamics:**
When AUD carries a significant positive rate differential over EUR (as in 2025 with RBA 3.60% vs ECB 2.00%), EUR/AUD faces persistent selling from carry positioning. Carry unwinds during risk-off events create rapid EUR/AUD spikes — a recurring triangulation context.

### 2.4 When Does the Implied EUR/AUD Diverge from Actual?

**1. China data shocks (AUD-dominant, EUR-neutral):**
The purest and most reliable triangulation signal. China PMI miss, property headline, or iron ore demand concern hits AUD/USD immediately. EUR/USD is unmoved. The implied EUR/AUD spikes as AUD/USD falls, but actual EUR/AUD lags. Typical closure: 5–30 minutes. This signal type recurs approximately 8–12 times per month with meaningful gap size and has no analogue in EUR-GBP-USD.

**2. US macro shocks (USD-dominant, moves both legs):**
NFP, CPI, FOMC hit both EUR/USD and AUD/USD at different magnitudes (AUD's USD-beta differs from EUR's). Implied EUR/AUD shifts while actual lags. Typical closure: 2–15 minutes.

**3. RBA decisions and surprises (AUD-dominant, EUR-neutral):**
RBA decisions (8 per year) move AUD/USD. EUR/USD is unmoved. Implied EUR/AUD shifts while actual lags. Typical closure: 5–45 minutes depending on surprise magnitude.

**4. ECB decisions (EUR-dominant, AUD-neutral):**
ECB decisions move EUR/USD. AUD/USD unchanged. Implied EUR/AUD shifts; actual EUR/AUD lags. Typical closure: 5–20 minutes.

**5. Risk-off events (AUD-dominant via risk-beta):**
AUD/USD falls more sharply than EUR/USD during acute risk-off. EUR/AUD actual cannot adjust as fast as implied. Large gaps, but persistence is uncertain — if the risk-off event represents structural repricing (e.g., Liberation Day tariff shock), the gap may not mean-revert.

**6. Commodity flash moves:**
Iron ore futures (DCE) events during Asian session can shift AUD/USD suddenly when EUR/AUD liquidity is thin. Rapid triangle gaps, high closure uncertainty due to thin EUR/AUD market.

**7. Options expiries and WMR fixing:**
WMR 4pm London fix creates EUR/AUD hedging flows. Month-end rebalancing in AUD-denominated assets (large Australian pension sector) generates systematic AUD/USD pressure that EUR/AUD lags.

---

## 3. 2025 Macro Context and Triangle Stress

### 3.1 The AUD's Extreme Year

2025 was the most volatile year for AUD in recent memory outside of COVID. The currency traced a deep V-shape: a catastrophic sell-off in Q1–Q2 driven by China tariff fears, followed by a sharp recovery in H2 as USD broadly collapsed and the RBA turned hawkish.

**Full-year range: 0.5914 (April 9 low) to approximately 0.6620+ (December)**

The April 9 low of 0.5914 was a five-year low for AUD/USD, approaching levels last seen during the COVID panic of March 2020. The speed of the move — from approximately 0.6300 pre-Liberation Day to 0.5914 in one week — generated the most extreme triangulation events in years.

**H2 reversal:** As the USD broadly collapsed (DXY -9.6% for 2025), AUD recovered sharply. By December, AUD/USD was approaching 0.6600. The RBA's hawkish turn (inflation sticky at 3.2%, cutting cycle stalling at 3.60%) provided additional AUD support in H2.

**Impact on EUR/AUD:** EUR/AUD was pulled in competing directions throughout 2025. During Q1–Q2 it spiked sharply (EUR/USD rising + AUD/USD collapsing = EUR/AUD surging toward 1.73–1.75). By H2, with AUD recovering and ECB remaining dovish, EUR/AUD fell back toward 1.62–1.65, producing a 15-month low around 1.73 → 1.62 retracement.

### 3.2 Liberation Day (April 2–9, 2025) — The Defining Triangle Event

Trump's Liberation Day announcements introduced a tariff structure uniquely damaging for AUD via the China channel:
- Australia directly: 10% tariff
- China: 34% initially, escalating to 84%+ effective by April 9 after retaliation

**EUR/USD:** Rose approximately +1.5–2% as USD sold off. The EU's 20% tariff was harsh, but European fiscal response and the USD safe-haven collapse supported EUR.

**AUD/USD:** Initially moved with the broad USD selloff, but within 24–48 hours the market began pricing the China secondary effect — 84%+ tariffs on China meant severe damage to Chinese steel demand, iron ore imports, and Australian export revenue. AUD/USD collapsed from approximately 0.6300 to 0.5914 by April 9, a move of nearly 400 pips in one week. This was driven not by Australia's direct tariff (10%, manageable) but by China tariff contagion through the commodity/iron ore channel.

**EUR/AUD:** The critical triangulation event. As EUR/USD rose and AUD/USD fell simultaneously, the implied EUR/AUD (EUR/USD / AUD/USD) surged dramatically. The actual EUR/AUD quote lagged — but crucially, it lagged because it was *also fundamentally repricing*, not because of a temporary liquidity gap. EUR/AUD went from approximately 1.62 pre-event to above 1.73–1.75 within the week.

**The most important strategic lesson from Liberation Day:** This was a false positive for mean-reversion triangulation. The implied EUR/AUD surged because of genuine new information about AUD's fundamental value — Chinese demand for Australian commodities was structurally lower. A strategy that bought AUD/EUR (sold EUR/AUD) assuming closure would have lost badly as EUR/AUD continued to diverge for days.

This is the primary design constraint of the AUD triangle strategy: **you must distinguish China-driven structural gaps from liquidity-driven transient gaps.** This is harder than the equivalent task in EUR-GBP-USD and is the primary training target for the ML model's second classifier.

### 3.3 RBA Policy in 2025

The RBA's policy trajectory was the single most important domestic driver of the triangle's structural mean:

**February 2025 — First cut:** RBA lowered the cash rate by 25bps to **4.10%**. First cut after a prolonged hold. Well-telegraphed; AUD/USD weakened modestly on announcement. EUR/AUD rose slightly. Quick gap closure — no significant triangulation event.

**Subsequent cuts:** RBA delivered approximately **75bps** total over 2025, bringing the cash rate from 4.35% to **3.60%**. The pace was constrained by sticky services inflation (underlying CPI holding above 3%) and a resilient labour market.

**H2 2025 — Hawkish turn:** By late 2025, with inflation failing to fall as projected and AUD weakness adding imported inflation pressure, the RBA signalled a pause. Markets began pricing the possibility of a reversal. AUD/USD rallied sharply from its April lows. The RBA-ECB differential (3.60% vs 2.00% = 160bps in AUD's favour) supported AUD through carry demand.

**Triangle implication:** The RBA cutting cycle in H1 meant EUR/AUD had a directional upward drift. The hawkish turn in H2 caused EUR/AUD to fall. The triangle residual had two distinct sub-regime means in 2025. A single long-run mean estimated over the full year would be wrong in both halves. The Kalman filter approach to mean estimation (Section 5.2) is essential for this instrument.

### 3.4 ECB Policy in 2025 and Its AUD Triangle Impact

The ECB was the most aggressive cutter among G3 central banks in 2025, bringing the deposit rate from ~3% to **2.0%** by December with approximately 4–5 cuts.

For the AUD triangle, ECB cuts had a specific transmission: ECB cut → EUR/USD falls on initial announcement → implied EUR/AUD falls → if actual EUR/AUD is slow to adjust, triangle gap opens with EUR/AUD actual above implied → sell EUR/AUD signal. ECB cuts also contributed to USD strength initially (EUR selling), which hit AUD/USD via USD strength — partially offsetting.

The net 2025 effect: ECB's aggressive cutting was more than offset by AUD's commodity/China problems in H1. EUR/AUD rose substantially in H1 despite ECB cuts because AUD fell more than EUR.

**ECB December pause (December 2025):** ECB held at 2.0%, surprising a market positioned for one more cut. EUR/USD spiked. AUD/USD was relatively unchanged. Implied EUR/AUD jumped; actual EUR/AUD lagged by approximately 15–25 pips for 8–15 minutes. This was a cleaner triangulation event than Liberation Day — a transient liquidity-adjustment gap rather than fundamental repricing.

### 3.5 Notable 2025 Triangle Stress Episodes

**Episode 1 — Liberation Day structural break (April 2–9):**
Maximum EUR/AUD divergence in years. EUR/USD +2% vs AUD/USD -6% over the week. EUR/AUD actual consistently trailed implied by 30–80 pips throughout the week. **This is a false positive for mean-reversion — the gap reflected genuine AUD fundamental repricing, not temporary illiquidity.** This is the canonical "structural repricing" training example the ML classifier must learn.

**Episode 2 — China PMI misses (recurring, monthly throughout 2025):**
Throughout 2025, weak Caixin PMI prints (released first business day of each month at 01:45 UTC) created sharp AUD/USD dips with EUR/USD largely unchanged. These generated reliable implied EUR/AUD spikes that were mostly transient — the actual EUR/AUD adjusted to close within 15–45 minutes. These are the canonical good triangulation events in the AUD triangle: pure China data shocks where one leg moves and the other does not.

**Episode 3 — RBA February cut (February 2025):**
Well-telegraphed. AUD/USD fell ~0.4% on announcement; EUR/USD unmoved. EUR/AUD jumped ~70 pips in the actual quote with a 15–20 pip lag vs implied. Gap closed within 20 minutes. Clean triangulation event.

**Episode 4 — Iron ore flash crashes (Q2–Q3 2025):**
Iron ore futures on the DCE had multiple 4–6% single-day drops in Q2–Q3 2025 as Chinese steel demand disappointed. Each drove AUD/USD lower, opening EUR/AUD triangle gaps. The challenge: distinguishing transient iron ore volatility (gap closes) from sustained iron ore bear moves (structural AUD repricing, gap does not close). This is the recurring in-sample test of the Target B classifier.

**Episode 5 — ECB December pause (December 2025):**
Transient gap of 15–25 pips in EUR/AUD vs implied. Closed within 15 minutes. Clean, liquid-session, ECB-driven event. Reliable mean-reversion signal.

---

## 4. Regimes and Exploitable Patterns

### 4.1 Defining Regimes for the AUD Triangle

**Dimension 1: Volatility Regime**

Classify each of AUD/USD, EUR/USD, EUR/AUD into Low/Medium/High vol using 5-minute realised vol over a rolling 60-minute window:

- **Low vol:** The AUD triangle is rarely truly low-vol given AUD's commodity beta. Sub-pip gaps, not worth trading.
- **Medium vol:** Sweet spot. AUD/USD making 40–80 pip daily ranges. EUR/AUD gaps of 10–25 pips emerging and closing over 5–30 minutes.
- **High vol:** Episodes like Liberation Day, major China shock, or RBA surprise. Gaps are massive but persistence is uncertain.

**A key asymmetry vs EUR-GBP-USD:** In EUR-GBP-USD, high vol from any leg tended to mean a macro event — transient, tradeable with care. In the AUD triangle, high vol specifically in AUD/USD (not EUR/USD) correlates strongly with China/commodity structural repricing. The **source of vol** is more diagnostic than its level.

**Dimension 2: Liquidity Regime**

EUR/AUD liquidity is thinner than EUR/GBP. Key reference points:
- Asian session (00:00–07:00 UTC): Where commodity and China data lands. Highest chance of large gaps; worst liquidity for EUR/AUD; spread can be 3–5 pips. Selectively monitor but treat as elevated risk.
- European open (07:00–09:00 UTC): Liquidity improving rapidly. Overnight position adjustments create predictable mean-reversion patterns.
- London/NY overlap (12:00–17:00 UTC): Maximum EUR/AUD liquidity. Tightest spreads (~0.8–1.5 pips). Best execution window.
- Post-NY close: Thin. Avoid unless very strong signal.

**Dimension 3: Information Regime**

- **China-dominant (AUD/USD moves, EUR/USD quiet):** Purest triangulation signal. Closure reliability depends on whether the event is transient (PMI miss, temporary iron ore move) or structural (tariff escalation, property crisis).
- **USD-dominant (both legs move):** Common. Less clean directional signal — depends on differential AUD vs EUR USD-beta.
- **RBA-dominant (AUD/USD moves, EUR/USD quiet):** Clean, high reliability. Purely monetary event, not commodity/growth repricing.
- **ECB-dominant (EUR/USD moves, AUD/USD quiet):** Clean, high reliability. EUR/AUD lags EUR/USD in a well-defined way.
- **Commodity flash (AUD/USD moves fast):** Requires vol type assessment before trading.

**Dimension 4: China/Commodity Stress Regime (AUD-specific, no EUR-GBP-USD equivalent)**

- **Normal/supportive China:** Standard mean-reversion appropriate.
- **China stress (structural):** AUD is being fundamentally repriced lower. Mean-reversion assumptions do not hold. Shut down the strategy.
- **China stress (transient):** Short-lived commodity shock that bounces. Tradeable but with higher threshold and smaller size.

Classifying structural vs transient China stress: concurrent signals include iron ore futures move >4% with follow-through, VIX >20, AUD/USD magnitude >> EUR/USD magnitude. If all present, treat as structural.

### 4.2 Market Conditions Producing Exploitable Patterns

**High signal, high reliability:**
- Single RBA decision with a clear surprise (hawkish hold when cut was expected): clean AUD/USD move, EUR/USD unchanged.
- Single ECB decision surprise: EUR/USD moves, AUD/USD unchanged.
- Transient China PMI miss that bounces within 1–2 hours: the most frequent and reliable event in the AUD triangle.
- ECB/RBA governor speeches that move one pair only.

**High signal, lower reliability (require model confidence gate):**
- Iron ore flash moves that reverse within hours: check whether the commodity move is isolated or part of a broader China demand repricing.
- US macro data that moves EUR/USD and AUD/USD at different magnitudes.

**Low signal / avoid:**
- China tariff escalation headlines (structural repricing risk).
- Multi-event days.
- Thin Asian session without specific catalyst.
- Sustained iron ore downtrend (>5 consecutive days of iron ore decline).

### 4.3 Volatility Regimes and Triangulation Opportunities

**Low vol:** AUD triangle is rarely truly low-vol. Pass.

**Moderate vol — target regime:** The Goldilocks condition is approximately 0.7–1.5× the trailing 30-day average realised vol for each pair.

**High vol — source matters critically:**
- High vol in AUD/USD with EUR/USD quiet: China or commodity event. Use the China stress classifier before trading. If structural: do not trade. If transient: 50% normal size with tighter stop.
- High vol in EUR/USD with AUD/USD quieter: US or ECB event. More likely transient. Normal to reduced size.
- High vol in both simultaneously: broad USD event. Too noisy. Pass unless ML model shows high-confidence P(closure).

**The single most important vol rule for the AUD triangle:** High vol in AUD/USD is not automatically a trading opportunity. It is the primary indicator of structural China repricing — the category of gap that kills mean-reversion strategies. Your ML model's most valuable function is distinguishing high-AUD-vol-transient from high-AUD-vol-structural.

---

## 5. Making It Tradable

*This section is the operational core of the document.*

### 5.1 Execution Realities

**Choosing your execution venue:**

- **Tier-1 ECN (EBS/CME FX, Refinitiv FXall):** EUR/AUD spreads 0.5–1.0 pips in good conditions; AUD/USD 0.2–0.5 pips; EUR/USD 0.1–0.3 pips. Requires institutional relationships. Target venue for any real strategy.
- **Prime Broker FX:** Aggregated liquidity. EUR/AUD ~0.8–1.5 pips. Achievable for a systematic fund.
- **Retail ECN (LMAX, Interactive Brokers, Saxo):** EUR/AUD typically 1.5–3.0 pips in normal conditions, 4–8 pips in stressed conditions. Combined round-trip cost: 3–5 pips normally. Minimum viable gap for retail: **5–8 pips**, which constrains signal frequency significantly.

**EUR/AUD specific execution challenges:**

The best signals (China data in Asian session) occur when EUR/AUD is hardest to execute. European and London sessions have better EUR/AUD liquidity but fewer China-driven signals. A realistic strategy must choose between:
- Trading the best signals in the Asian session at worse execution (higher threshold, smaller size), or
- Trading in European/London sessions when execution is cleaner but signals are predominantly RBA and ECB driven.

The recommended approach: European/London session focus initially, adding Asian session capability once the pipeline is proven.

**The three-leg execution problem:**

**Approach A (recommended): Trade EUR/AUD only.** If the model signals EUR/AUD is cheap vs implied (Delta < 0), buy EUR/AUD. One position, one spread, clean P&L. This is the correct approach for first implementation.

**Approach B: Trade one USD leg.** Less clean; adds USD exposure.

**Approach C: Full 3-leg simultaneous.** Buy EUR/AUD + Sell EUR/USD + Buy AUD/USD (or reverse). Nets to zero USD exposure. Requires near-simultaneous ECN execution.

**Slippage estimation:**

For $5–10mn notional in EUR/AUD in London session:
- Quoted spread: 0.8–1.5 pips
- Market impact: 0.3–0.6 pips for $5–10mn
- Total round-trip: **2.5–4.5 pips** in normal conditions; 5–10 pips in moderate stress

Minimum signal threshold: **5–7 pips** for breakeven. Target: **8–15 pips** for positive expectation. These are higher than EUR-GBP-USD due to wider EUR/AUD spreads.

**Timing:**

- Triangle gaps peak immediately after the triggering event.
- The first 60–90 seconds: highest-risk for EUR/AUD execution (spreads widest).
- The **3–10 minute post-event window**: gap still partially open, liquidity normalising. Best entry timing.
- For China data events (01:00–03:00 UTC): no "better window" — trade then or not at all.

### 5.2 Statistical Modelling Architecture

**Step 1: Compute the Triangle Residual**

```python
# Using log mid-prices at 1-minute bars, from a single venue
ln_eurusd        = log(EUR/USD_mid)
ln_audusd        = log(AUD/USD_mid)
ln_euraud_actual = log(EUR/AUD_mid)

# Triangle residual (EUR/AUD_implied = EUR/USD / AUD/USD)
triangle_residual = ln_euraud_actual - (ln_eurusd - ln_audusd)
```

**Critical for AUD:** Ensure EUR/AUD comes from the same venue as the USD legs. EUR/AUD is less actively quoted and some venues show stale quotes in thin sessions. Validate by checking the cross-venue residual mean is stable over rolling windows.

**Step 2: Time-Varying Mean (More Important Here Than EUR-GBP-USD)**

The AUD triangle's residual mean shifts significantly across regimes. A Kalman filter is strongly preferred over a fixed mean. Use two speeds:

```
mu_fast_t = 0.2 * residual_t + 0.8 * mu_fast_{t-1}   [intraday signal normalisation]
mu_slow_t = 0.02 * residual_t + 0.98 * mu_slow_{t-1}  [regime mean tracking]

z_t = (residual_t - mu_fast_t) / sigma_t

Structural drift flag: if |mu_slow_t| > 2 * sigma_baseline -> flag regime shift
```

**Step 3: OU Half-Life Estimation**

The AUD triangle's OU half-life is materially longer than EUR-GBP-USD:
- Normal conditions (European/London session): 10–45 minutes
- China stress events: can extend to hours or not mean-revert
- RBA/ECB surprise events: 10–30 minutes

Use rolling 4-hour windows. Feed into position sizing (longer half-life = smaller position per unit of gap) and exit timing (time-based exit at 2× estimated half-life).

**Step 4: Cointegration Testing**

The AUD/USD-EUR/USD-EUR/AUD trio should be cointegrated by no-arbitrage. Run Johansen tests weekly. During sustained China stress (e.g., Liberation Day period), cointegration may break down. A failed cointegration test = no-trade regime indicator.

**Step 5: Regime Classification**

| Regime | AUD/USD Vol | EUR/USD Vol | China State | Action |
|--------|------------|-------------|-------------|--------|
| A | Low | Any | Benign | Pass — gap too small |
| B | Medium | Medium | Benign | Full size, standard threshold |
| C | Medium | Medium | Stress signal | Halve size, raise threshold |
| D | High (RBA/ECB event) | Low/Medium | Benign | Reduced size, fast exit |
| E | High (China structural) | Low/Medium | Stress | No trade |
| F | High | High | Any | No trade |

The China state classifier is the key addition vs EUR-GBP-USD. Build it as a separate binary classifier: is current AUD/USD vol driven by China structural repricing (class 1) or transient factors (class 0)? Features: iron ore futures change (%), VIX level, AUD/USD magnitude vs EUR/USD magnitude, time relative to recent China data release.

### 5.3 Applying Your Existing ML Pipeline to Triangulation

The goal is to build a triangulation-specific ML model using the same infrastructure: same data handling conventions, train/test/validation methodology, feature engineering patterns, evaluation framework, and deployment pipeline you already have.

#### What the Model Should Predict

**Target A — Probabilistic gap closure (recommended first model):**
Binary classifier: will the current triangle gap close by at least 50% within a fixed horizon (10 or 30 minutes)? Output: P(closure). Entry gate: trade only when P(closure) > 0.65.

**Target B — Structural vs. transient regime (required additional model, AUD-specific):**
Unlike EUR-GBP-USD, the AUD triangle requires an explicit classifier on gap type: P(structural_repricing) — the probability that the current gap is driven by China/commodity structural repricing rather than a transient liquidity/speed-of-adjustment gap. A trade proceeds only when P(closure) > 0.65 AND P(structural_repricing) < 0.3.

These two models can be trained as a joint multi-output classifier or as a two-stage pipeline where Target B gates access to Target A.

#### Feature Engineering

Features follow the same structure as your vol model, applied to AUD-specific series:

**Core signal features:**
```python
z_t      = (residual_t - ewma_mean_t) / ewma_std_t
dz_t     = z_t - z_{t-1}
d2z_t    = dz_t - dz_{t-1}
halflife = estimated OU half-life (rolling 60-min window)
```

**Volatility regime features:**
```python
rv_eurusd_1h = realised vol of EUR/USD (past 60 mins, 5-min bars)
rv_audusd_1h = realised vol of AUD/USD
rv_euraud_1h = realised vol of EUR/AUD
rv_ratio     = rv_audusd_1h / rv_eurusd_1h   # AUD vol dominance — key feature
rv_triangle  = std(residual, rolling 60 mins)
```

**China/Commodity features — AUD-specific additions:**
```python
iron_ore_chg_1h   = % change in iron ore futures (DCE) over past 60 mins
iron_ore_chg_24h  = % change in iron ore futures over past 24 hours
china_pmi_surprise = actual - forecast at most recent Caixin/NBS PMI release
vix_level          = current VIX level
```

**Spread and liquidity features:**
```python
spread_euraud_norm = current EUR/AUD spread / 30-day average spread
is_asian_session   = 1 if 00:00-07:00 UTC else 0
```

**Time and event features:**
```python
hour_of_day (sin/cos encoded)
mins_to_rba_decision
mins_since_rba_decision
mins_to_ecb_decision
mins_since_ecb_decision
mins_to_china_pmi        # Caixin: first business day of month, 01:45 UTC
mins_since_china_pmi
```

**Macro regime features (daily update):**
```python
rba_ecb_rate_differential = RBA cash rate - ECB deposit rate
rate_diff_trend           = 5-day change in differential
audusd_5d_mom             = 5-day log return of AUD/USD
eurusd_5d_mom             = 5-day log return of EUR/USD
euraud_5d_mom             = 5-day log return of EUR/AUD
iron_ore_5d_change        = % change in iron ore over 5 days
```

Total: approximately 25–35 features. The China/commodity features are the key additions vs the EUR-GBP-USD feature set.

#### Train / Test / Validation Methodology

Use the same walk-forward structure as your existing vol model:

```
Training set:     2020-01-01 to 2023-12-31
Validation set:   2024-01-01 to 2024-06-30   (hyperparameter tuning)
Test set:         2024-07-01 to 2025-12-31   (held out entirely)
```

The test set must include Liberation Day (April 2025) and the H2 AUD recovery. If the model does not correctly suppress signals during April–May 2025, it has not learned the most important regime in the dataset.

**Label construction for Target B (structural repricing):**
Define a gap as "structural" if, within 60 minutes of the gap opening, the triangle residual moved *further from zero* rather than toward it. Label as "transient" if it closed by 50%+ within 30 minutes. Exclude ambiguous cases from Target B training to keep labels clean.

**Walk-forward folds:** Same expanding/rolling window approach. 6-month training, 1-month out-of-sample, rolling. Apply 10-bar buffer between folds.

#### Model Architecture

Re-use your existing architecture. The two-target formulation can be:
- Two separate binary classifiers (simpler, faster)
- Single multi-output model with two sigmoid heads sharing a common feature encoder

The China stress classifier (Target B) may benefit from longer lookback windows (4-hour vs 30-minute) since structural repricing events build over hours. If your vol model uses a 30-minute lookback, consider a separate 4-hour lookback for Target B.

**Loss function:** Binary cross-entropy for each target. If joint training, weight Target B higher — a false negative on structural repricing (trading a structural gap as transient) is far more costly than a false positive.

**Calibration:** Apply Platt scaling separately for each target on the validation set.

#### Evaluation Metrics

| Metric | Target A (Gap Closure) | Target B (Structural Repricing) |
|--------|------------------------|----------------------------------|
| Primary | AUC-ROC > 0.58 | AUC-ROC > 0.65 |
| Precision/Recall at threshold | Precision > 60% at P > 0.65 | Recall > 70% at P > 0.3 |
| Key backtest metric | Simulated Sharpe on test set | Drawdown during April–May 2025 |
| Calibration | Brier score < 0.05 | Same |

Target B requires higher recall than precision — missing a structural repricing event is far more costly than missing a trade. Tune Target B's threshold toward high recall even at the cost of false positives.

#### Signal Generation Framework

```
Step 1:  Compute triangle residual z-score z_t
Step 2:  Assemble features at current bar
Step 3:  Run Target B -> P(structural_repricing)
Step 4:  Hard gate: if P(structural_repricing) > 0.3 -> NO TRADE
Step 5:  Run Target A -> P(closure)
Step 6:  Entry gate: proceed only if P(closure) > 0.65
Step 7:  Direction from sign of z_t:
           z_t > 0 -> EUR/AUD expensive vs implied -> SELL EUR/AUD
           z_t < 0 -> EUR/AUD cheap vs implied -> BUY EUR/AUD
Step 8:  Position size = Base_size x (P(closure) - 0.5) / 0.5 x Kelly_fraction
                       x (1 - P(structural_repricing)) / 0.7
Step 9:  Time-based exit at 2x OU half-life estimate
         Z-score reversal stop at 1.5x entry z
         Hard loss stop at 2x entry gap in pips
```

Kelly fraction: start at 0.25, scale up only after 3+ months of live or simulated performance consistent with backtest.

### 5.4 Risk Management Specific to AUD Triangulation

**Stop-loss design:**

- **Time-based exit:** If gap hasn't closed 50% within 2× OU half-life, exit. AUD half-lives are longer — a 30-minute half-life means exit by 60 minutes if not 50% closed.
- **Z-score reversal stop:** If gap widens to 1.5× entry z-score, exit. Structural repricing gaps frequently exhibit further widening after initial move.
- **Iron ore momentum stop (AUD-specific):** If iron ore futures move >2% in the direction that would widen the gap during a trade, exit immediately. No equivalent in EUR-GBP-USD. Catches structural moves that escaped the Target B classifier.
- **VIX threshold exit:** If VIX spikes above 25 during a trade, exit. AUD will continue to fall in broad risk-off, and EUR/AUD gaps in risk-off are often structural.

**Maximum drawdown per trade:**
Given wider EUR/AUD spreads, set per-trade drawdown limit at 10–15 pips (compared to 7–10 for EUR-GBP-USD).

**Session risk limits:**
- Asian session: max 1 open position, 50% normal size, iron ore momentum stop mandatory.
- European/London session: max 2 open positions, normal size.
- Post-NY close: no new positions.

**Correlation and macro overlay:**
Maintain a China stress overlay: if iron ore is in a sustained 5+ day downtrend, reduce exposure to 25% of normal across all regimes. Carry trade unwind overlay: when AUD/USD carry is being actively unwound (AUD falling with cross-asset risk-off), treat all AUD/USD moves as potentially structural and reduce size.

**Model risk:**
The EUR/AUD cross is less standardised across venues than EUR/GBP. Verify regularly that your EUR/AUD data source is quoting at market (not stale). Stale EUR/AUD quotes on some retail venues create phantom residuals that look like tradeable gaps but evaporate on execution. Test by checking whether your simulated gaps would have been fillable at the quoted price using Dukascopy tick data as the reference.

---

## 6. Research and Data Pointers

### 6.1 Key Academic Papers

**Triangular Arbitrage and FX Microstructure:**

- **Aiba et al. (2002)** — "Triangular arbitrage in foreign exchange markets" (*Physica A*). Foundational empirical work on triangle residuals. Framework applies to AUD triangle.

- **Fong, Valente & Fung (2010)** — "Covered interest arbitrage profits: the role of liquidity and credit risk" (*Journal of Banking & Finance*). How liquidity constraints affect arbitrage persistence — directly relevant to EUR/AUD's thinner market.

**AUD-Specific and Commodity Currency Research:**

- **Cashin, Cespedes & Sahay (2004)** — "Commodity Currencies and the Real Exchange Rate" (*Journal of Development Economics*). Foundational work on commodity currency dynamics; establishes the iron ore-AUD link.

- **Chen & Rogoff (2003)** — "Commodity Currencies" (*Journal of International Economics*). Documents the statistical relationship between commodity prices and AUD/USD. Quantifies predictive power of commodity prices for AUD.

- **RBA Research Discussion Papers** — Regular research on AUD dynamics including China linkages and commodity pass-through. Recent 2023–2024 papers on AUD post-COVID dynamics available at rba.gov.au.

**Statistical Arbitrage and Mean Reversion:**

- **Gatev, Goetzmann & Rouwenhorst (2006)** — "Pairs Trading: Performance of a Relative-Value Arbitrage Rule" (*RFS*). Foundational pairs-trading paper. The triangle residual is the 3-asset extension.

- **Elliott, van der Hoek & Malcolm (2005)** — "Pairs trading" (*Quantitative Finance*). OU process framework for statistical arbitrage.

**Regime Switching:**

- **Engel & Hamilton (1990)** — "Long swings in the dollar" (*AER*). Classic reference. AUD has even more pronounced regime-switching behaviour than USD.

- **Brunnermeier, Nagel & Pedersen (2008)** — "Carry Trades and Currency Crashes" (*NBER*). Directly relevant — AUD is the prototypical carry currency. Carry crash dynamics explain many AUD structural repricing events.

**ML for FX:**

- **Oxford Academic JFE (2024)** — "Volatility Forecasting with Machine Learning and Intraday Commonality." Cross-asset vol forecasting with ML; the feature commonality approach is applicable to the AUD triangle's multi-asset feature set.

- **Bucci (2020)** — "Realized Volatility Forecasting with Neural Networks" (*JFE*). The vol forecasting methodology your existing model likely draws from.

**2025-Specific:**

- **CEPR VoxEU (April 2025)** — "Tariffs, the dollar, and equities: High-frequency evidence from the Liberation Day announcement." Documents the USD's anomalous Liberation Day behaviour; explains why EUR/USD rose while AUD/USD ultimately crashed.

- **CSIS Analysis (April 2025)** — "China and the Impact of Liberation Day Tariffs." Quantifies Chinese economic impact and transmission to AUD via iron ore.

- **Australian Department of Industry (December 2025)** — "Resources and Energy Quarterly." Iron ore export revenue forecasts, commodity price trajectories for 2025–2026. Essential AUD structural context.

### 6.2 Practitioner Resources

- **RBA Commodity Price Index** — Monthly. Free at rba.gov.au/statistics. Leading indicator of AUD structural level.
- **BIS Quarterly Review** — FX market structure. BIS Triennial Survey covers AUD/USD and EUR/AUD turnover.
- **MacroMicro (en.macromicro.me)** — Free charts of AUD/USD vs iron ore prices, China PMI vs AUD/USD. Useful for regime monitoring.
- **Dalian Commodity Exchange (DCE) Iron Ore Futures** — Real-time iron ore pricing. Available via Bloomberg, Reuters, or free delay via commodity data aggregators. Essential for China stress regime classification.
- **Caixin PMI Release Calendar** — Published first business day of each month at 01:45 UTC. Single most reliable recurring signal event for the AUD triangle in the Asian session.

**Practitioner commentary:**
- **Spectra Markets (Brent Donnelly)** — Daily FX practitioner commentary. Regular AUD/commodity analysis, regime-aware.
- **AMP Capital / Shane Oliver** — Australian macro and AUD-specific analysis. Good for RBA regime context.
- **MacroAlf (Alfonso Peccatiello)** — Macro-driven FX including commodity currency dynamics.

### 6.3 Data Sources

**Tick and Intraday FX Data:**

- **Dukascopy** — Tick data (bid/ask) for AUD/USD, EUR/USD, and EUR/AUD. Free with registration. One of the only free sources with EUR/AUD tick data back to ~2010. Essential for spread analysis and validating EUR/AUD quotes are not stale in thin sessions.
- **HistData.com** — Free 1-minute OHLCV. Has EUR/AUD. Sufficient for strategy development. Does not include bid/ask.
- **Refinitiv/LSEG DataScope** — Institutional grade. Full bid/ask and venue-stamped tick data.
- **Bloomberg FX composite** — Best quality. Terminal required.

**Commodity Data (AUD-specific, essential):**

- **DCE Iron Ore Futures (front contract)** — Primary iron ore price driver. Available via Bloomberg, Refinitiv. Free delayed data via TradingEconomics.
- **RBA Commodity Price Index** — Monthly. Free at rba.gov.au/statistics/frequency/commodity-prices/.
- **Australia Resources and Energy Quarterly** — Quarterly. Free from industry.gov.au. Iron ore and coal price forecasts and retrospective data.

**Implied Volatility Data:**

- **NY Fed Implied Volatility** — 1-month ATM implied vol for major USD pairs including AUD/USD. Free. Useful regime indicator. https://www.newyorkfed.org/markets/impliedvolatility.html
- **CME Group FX Options** — Listed AUD/USD options. Public data. EUR/AUD options are OTC only.
- **Bloomberg/Refinitiv vol surfaces** — Full OTC vol surface for AUD/USD. EUR/AUD OTC vol is less actively traded but available on Bloomberg.

**Macro and Event Data:**

- **ForexFactory / Investing.com** — RBA, ECB, FOMC, Caixin PMI dates and forecasts.
- **FRED** — Policy rates, macro indicators. Free.
- **ABS (Australian Bureau of Statistics)** — CPI, employment, GDP. ausstats.abs.gov.au. Free.
- **CFTC COT** — Weekly AUD/USD positioning. Free. Crowded positioning = carry-crash risk indicator.

---

## Appendix: 3-Week Implementation Plan

A condensed plan from this brief to a running simulated trading setup. Each week has a gate — if not met, do not proceed.

---

### Week 1 — Data, Residual, and Baseline Statistics

**Goal:** Validate that the AUD triangle residual is exploitable in principle, and characterise the China/commodity regime structure.

**Day 1–2: Data pipeline**
- [ ] Pull synchronised 1-minute data for AUD/USD, EUR/USD, EUR/AUD from Dukascopy. Use the same timestamp normalisation and gap-handling conventions as your vol model.
- [ ] Verify timestamp alignment — all three pairs on the same bar clock. EUR/AUD on Dukascopy should be checked for stale quotes by confirming the cross-implied residual is not systematically biased vs zero.
- [ ] Compute the triangle residual: `ln(EUR/AUD) - ln(EUR/USD) + ln(AUD/USD)`.

**Day 2–3: Statistical characterisation**
- [ ] ADF and KPSS tests on the residual. Run on sub-periods: a pure China-stress period (April–May 2025) and a normal period (Q4 2024). Confirm non-stationarity during stress, stationarity in normal regime. This is the AUD-specific analogue of the stationarity check.
- [ ] Rolling autocorrelation at lags 1, 5, 10, 30 minutes. Compute separately for Asian session, European/London session, post-NY. Session breakdown is more important for AUD than EUR-GBP-USD.
- [ ] Build the regime timeline for 2025: tag each day as "normal/benign" or "China stress" using iron ore daily returns as a simple proxy (days when iron ore moved >2% = China stress flag). Compute residual autocorrelation and average half-life separately for each regime.

**Day 4–5: Signal frequency and cost analysis**
- [ ] Count signal frequency by session and threshold (|z_t| > 1.5, 2.0, 2.5). Split by regime.
- [ ] Confirm that the median gap at your chosen threshold exceeds EUR/AUD execution cost in each session (8–15 pips target in European/London; 12–20 pips in Asian session).
- [ ] Identify the five key 2025 episodes in your data (Liberation Day, monthly China PMI misses, RBA February cut, iron ore crashes, ECB December pause). Confirm they appear as anomalous residual spikes or structural drifts as expected.

**Week 1 Gate:** Residual is stationary in at least two non-stress sub-periods; autocorrelation at lag-1 > 0.3 in European/London session; at least 3–4 tradeable signals per week in the London session. If residual is non-stationary even in normal conditions, there is a data quality issue — investigate EUR/AUD stale quotes before proceeding.

---

### Week 2 — Feature Engineering and ML Model Training

**Goal:** Two trained, validated models — Target A (gap closure) and Target B (structural repricing) — using your existing pipeline.

**Day 1–2: Feature construction and labels**
- [ ] Build the full feature set from Section 5.3 using the same feature engineering conventions as your vol model. The iron ore change features require a commodity data source — use TradingEconomics API (free tier) or a static DCE iron ore daily series.
- [ ] Construct Target A labels (binary gap closure at 30 minutes, with 10-bar no-look-ahead buffer).
- [ ] Construct Target B labels (structural repricing): flag events as structural if the residual widened further than its entry level within 60 minutes. Confirm Liberation Day period is in the test set so it tests generalisation rather than memorisation.
- [ ] Apply train/val/test split: train through 2023, validate 2024 H1, test 2024 H2–2025.

**Day 2–4: Model training**
- [ ] Train Target B (structural repricing classifier) first — this is the gating model. Use your existing architecture with binary cross-entropy. Tune threshold for high recall (>70%) on the validation set. Confirm Liberation Day period in test set is correctly classified as structural.
- [ ] Train Target A (gap closure classifier). Same architecture and tuning procedure. Platt-scale both models on the validation set.
- [ ] Run feature importances. For Target A: residual z-score and velocity should dominate. For Target B: iron ore change, rv_ratio (AUD vol dominance), VIX level, and session indicator should dominate. If this pattern doesn't emerge, revisit feature construction.

**Day 4–5: Test set evaluation**
- [ ] Run both models on the test set. For Target B: check that all major 2025 stress episodes are correctly classified as structural — this is the most important qualitative validation in the entire project.
- [ ] Plot P(closure) and P(structural_repricing) alongside the triangle residual for April–May 2025. P(structural_repricing) should be consistently high throughout this period; the dual gate should be suppressing all trades.
- [ ] Compute simulated Sharpe applying only the signal gate (P(closure) > 0.65 AND P(structural_repricing) < 0.3) with flat position sizing, no cost model. This is the pre-cost theoretical Sharpe.

**Week 2 Gate:** Target B AUC > 0.63 and recall > 65% at threshold; Liberation Day period correctly suppressed in simulated P&L. Target A AUC > 0.56. Pre-cost theoretical Sharpe > 1.0 on test set. If Target B fails to catch Liberation Day events, the model cannot be safely deployed — this is the critical failure mode.

---

### Week 3 — Simulated Trading

**Goal:** Operational simulated trading with realistic costs, producing trade logs and P&L attribution, ending with live paper trading.

**Day 1–2: Trading engine**
- [ ] Build the full signal pipeline: residual computation → features → Target B inference (gate) → Target A inference (sizing) → direction → position sizing → entry. Wire using your existing ML inference infrastructure.
- [ ] Implement exits: time-based at 2× OU half-life; z-score reversal at 1.5×; iron ore momentum stop (>2% iron ore move against position); VIX spike exit (>25). All exits must be deterministic.
- [ ] Implement realistic costs: deduct current Dukascopy bid-ask spread at time of entry/exit plus 0.4 pip slippage per side.

**Day 2–3: Backtest on test period**
- [ ] Run full pipeline over test set (2024 H2–2025) in strict event-time order. Generate trade log: entry time, z-score, P(closure), P(structural_repricing), position size, exit time, exit reason, gross and net P&L.
- [ ] Compute: annualised Sharpe, max drawdown, win rate, average holding time, trades per week. Check Liberation Day attribution: strategy should be flat or near-flat during April 2–9, 2025. Any significant loss during this period means Target B failed in live conditions.
- [ ] Session breakdown: run attribution separately for Asian session vs European/London. If Asian session trades are a drag, restrict to European/London only.

**Day 3–4: Sensitivity analysis**
- [ ] Re-run with EUR/AUD spreads 2× normal. Sharpe should stay above 0.5 — if not, the strategy is too dependent on tight execution.
- [ ] Re-run with Target B threshold varied from 0.2 to 0.5 in steps of 0.05. Confirm the optimal threshold on the test set is consistent with validation-set calibration (within ±0.1).
- [ ] Re-run removing the iron ore momentum stop. Compare drawdown during commodity shock episodes — this quantifies how much the stop contributes to risk control.
- [ ] Kelly fraction sweep: 0.15 to 0.50. Confirm max drawdown scales approximately linearly. Identify acceptable Kelly level.

**Day 4–5: Live paper trading**
- [ ] Connect to live AUD/USD, EUR/USD, EUR/AUD quotes via LMAX demo or Interactive Brokers demo.
- [ ] Verify the live iron ore data feed is updating correctly and the China stress feature is being computed without error. Iron ore data integration is the most likely point of live pipeline failure.
- [ ] Run paper trading for the remainder of Week 3. Log every signal and hypothetical trade in real time. Do not adjust the model during this period.
- [ ] At end of Week 3: compare live signal frequency, session distribution, and P(closure) distribution to backtest expectations. Check especially whether the Asian session is generating spurious signals from stale EUR/AUD quotes — the most common live-vs-backtest divergence in the AUD triangle.

**Week 3 Gate:** Backtest Sharpe > 0.65 after costs; drawdown < 10% of notional; Liberation Day period loss < 1% of notional; live signal frequency within 25% of backtest expectation. If these are met, the strategy is ready for cautious live deployment at minimum notional with daily monitoring of iron ore data feed integrity.

---

*Brief compiled March 2026. Research conducted March 2026. AUD dynamics are highly regime-dependent — validate all structural assumptions against current iron ore, China PMI, and RBA policy conditions before deployment.*
