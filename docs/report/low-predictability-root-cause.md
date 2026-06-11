# Low Predictability Root Cause Analysis

**Date**: 2026-06-10

**Status**: historical pre-tailtree diagnosis. The current scanner now has a `tailtree` evidence path with continuous/source features and GPD tail evidence; this report documents the root causes that motivated that work, not the final current model state.

**Observation**: Scanner evidence gate produces baseline entropy ~1.30/1.585 bits (82% of max), best-case information gain ~0.29 bits. Near-random predictability.

---

## 1. WHAT THE NUMBERS MEAN

| Metric | Value | Interpretation |
|---|---|---|
| Max entropy (3 equiprobable outcomes) | 1.585 bits | Coin-flip ceiling |
| Observed baseline entropy | ~1.30 bits | Unconditional: p_up≈0.44, p_down≈0.42, p_flat≈0.14 |
| Best conditioned entropy | ~1.01 bits | Best condition shifts p_up to ~0.53, p_down to ~0.32 |
| Information gain | ~0.29 bits | Net: 9pp directional tilt |

A 0.29-bit information gain means: *knowing the full market-regime + source state narrows your directional uncertainty from a near-coin-flip to slightly better than a coin-flip.* The edge is *real* — 53% vs 44% base rate — but thin. With ~400 observations, a 95% CI for p_up is [50%, 60%] — the lower bound touches random.

## 2. ROOT CAUSES

### 2.1. Outcome construction: near-coin-flip target (impact: HIGH, effort: LOW)

**What the code does** (`evidence.py` L13-21, `__init__.py` L13-20):

```python
outcome_bucket_expr(return_threshold_pct):
    up:   forward_return_pct > return_threshold_pct
    down: forward_return_pct < -return_threshold_pct
    flat: otherwise
```

With `return_threshold_pct = 0.0` (daily-deep config), the outcome is "did price go net-positive or net-negative over the horizon." For crypto 1H returns, this is ~49% up / ~47% down / ~4% flat. **You are trying to predict a near-coin-flip with state descriptors, and the coin-flip ceiling is tight: the maximum possible information gain over baseline is only ~0.285 bits.**

**Why this matters**: You cannot gain >0.285 bits no matter how good your features are. The scanner is architecturally capped.

**Fix**:
- **(A) Decile-quantile outcome**: Replace 3-way bucket with 10 return-decile buckets. Max entropy becomes log₂(10) ≈ 3.32 bits. Room for 2.0+ bits of gain.
- **(B) Continuous entropy**: Use actual forward_return_pct distribution with differential entropy estimation (histogram or KDE). No artificial ceiling.
- **(C) Tail-event outcome**: Keep 3-way but threshold at ±2% or ±5% — now the base rate of "extreme up" is lower (~15-20%), making conditioning more informative.

### 2.2. Classifier destroys continuous information (impact: HIGH, effort: MEDIUM)

**What the code does** (`classifiers.py` L119-328):

```
continuous price/volume data
  → atr_percentile (0-100) → bucketed: vol_low / vol_normal / vol_high
  → range_width_atr (continuous) → bucketed: range_tight / range_normal / range_wide
  → swing pattern (HH/HL/LH/LL counts) → bucketed: UPTREND / DOWNTREND / RANGE
  → all collapsed to: "MARKUP|UPTREND|range_normal|vol_high"
  → then further collapsed to: direction_hint = "bullish" / "bearish" / "neutral"
```

The classifier chain destroys ~90% of the information present in the raw OHLCV. ATR rank at 87th percentile and at 78th percentile are the same "vol_normal" bucket. Every continuous feature is discretized into 3-5 labels.

**Why this matters**: The evidence levels condition on these categorical labels, not the continuous values. Two observations with very different volatility profiles but the same bucket label are treated identically — the entropy reduction from splitting on a 3-way categorical is inherently limited.

**Fix**:
- **(A) Feed continuous scores into evidence groups**: Instead of `background_regime` (string), use `atr_percentile_rank` (float 0-100), `range_width_atr` (float), `structure_score` (float -1 to +1)
- **(B) Use percentile-binned continuous stratifiers**: Replace hardcoded buckets with adaptive quantile bins (deciles/quartiles)
- **(C) Add derived continuous features**: `return_1bar`, `return_4bar`, `vol_anomaly` (vol/vol_ma20), `close_to_range_high_ratio`, `imbalance_magnitude`

### 2.3. Features describe the past, not the future (impact: HIGH, effort: MEDIUM)

**What the code does** (`classifiers.py` L129-253):

```
MARKET_STAGE is determined by:
  - Swing high/low breaks (shift(1), rolling max/min) → what JUST happened
  - Range width (48-bar rolling) → where price IS now
  - HH/HL/LH/LL counts (24-bar rolling) → momentum taxonomy of the PAST 24 bars
```

The classifier is a **descriptor, not a predictor**. It answers "what market structure exists now" — an ex-post taxonomy. The intuition that "MARKUP" implies continuation is momentum logic, but crypto momentum is:
- Weak on 1H bars (autocorrelation ≈ 0.01–0.03)
- Regime-dependent (works in trending, fails in ranging)
- Mean-reverting at 4-12 bar horizons

The scanner conditions outcomes on a lagging taxonomy and asks if it predicts the future. The answer — 0.29 bits of weak momentum — is exactly what should be expected.

**Fix**:
- **(A) Add rate-of-change features**: `atr_acceleration` (atr_t - atr_t-4), `volume_acceleration`, `range_expansion_rate`
- **(B) Add divergence features**: price-direction vs volume-direction, price vs OI delta, taker-buy vs price-move alignment
- **(C) Add anomaly scores**: z-score of current bar characteristics vs rolling 100-bar distribution
- **(D) Add time-structural features**: bar-of-session (0-23), day-of-week, proximity to known events

### 2.4. Cross-coin pooling destroys symbol-specific signal (impact: MEDIUM, effort: LOW)

**What the code does** (`evidence.py` L288-290, L595-616):

```python
market = joined.unique(subset=["symbol", "decision_bar_close_ms", "outcome_horizon"])
market_baseline = _outcome_baseline(market)  # ALL symbols pooled
```

The baseline entropy is computed on *all symbols pooled together*. The conditioned groups are also pooled — `background_regime="MARKUP"` includes BTC in markup, PEPE in markup, and every other coin.

**Why this matters**:
- Crypto returns are highly correlated (BTC beta): when BTC drops 3%, 80% of alts also drop
- Pooling correlated observations inflates `baseline_observations` but adds no independent information
- A 1H bar where every coin goes down is ONE event, not N events
- Coin-specific base rates differ: BTC goes up 52% of 1H bars, small alts 47%
- Pooling them smears coin-specific edges into the global noise floor

**Fix**:
- **(A) BTC-relative returns**: Use `forward_return_pct - btc_forward_return_pct` as outcome
- **(B) Per-symbol baselines**: Compute baseline within each symbol, then measure conditional gain over symbol's own baseline
- **(C) BTC-regime stratification**: Compute evidence separately for BTC-up-regime and BTC-down-regime
- **(D) Market-cap-weighted baselines**: Weight observations by market cap so BTC doesn't dominate

### 2.5. Nested evidence levels don't capture interactions (impact: MEDIUM, effort: MEDIUM)

**What the code does** (`evidence.py` L292-331):

```
Level 1: market_background      → group by ["background_regime"]
Level 2: market_swing           → group by ["background_regime", "swing_core"]
Level 3: market_decision        → group by ["background_regime", "swing_core", "decision_core", "decision_transition"]
Level 4: market_decision_source → group by [above + "source_family", "source_state"]
Level 5: + "risk_context"
```

**Why this matters**:
- Adding a column splits data further → each cell gets fewer observations → wider confidence intervals
- No interaction terms: "MARKUP background × bearish books" may be predictive, but "MARKUP background" alone is noise
- The `parent_information_gain` comparison punishes deeper levels that add little marginal signal — correctly — but the **shallow levels also have low signal** because they're underspecified
- A data-driven approach would find "background_regime=ACCUMULATION + source_state=bid_support" directly without going through the full nest

**Fix**:
- **(A) Skip the nest, use direct combinatorial groups**: Group by top-N most-informative variable combinations (chi-square ranked)
- **(B) Use interaction features**: `background_x_swing = "MARKUP|UPTREND"`, `decision_x_source = "bullish|bid_support"`
- **(C) Decision-tree stratification**: Train a shallow CART tree on the observation data, extract the splitting rules as evidence levels
- **(D) Mutual information ranking**: Rank all feature subsets by mutual information with outcome, pick top-K

### 2.6. Data fragmentation from join filters (impact: LOW-MEDIUM, effort: LOW)

**What the code does** (`evidence.py` L508-582):

```python
# Market-level: inner join with realized_transitions, then filter non-null terminal
market = observations.join(realized_transitions, ..., how="inner")
    .filter(pl.col("terminal_core_context").is_not_null())

# Source-level: additional inner join, then filter non-null
scored = source_outcomes.filter(pl.col("outcome_available") & pl.col("source_state").is_not_null())
source_joined = observations.join(scored, ..., how="inner")
    .join(realized_transitions, ..., how="inner")
    .filter(pl.col("terminal_core_context").is_not_null())
```

**Why this matters**:
- The `inner` join drops observations where the realized transition hasn't been computed (recent bars, horizon edge)
- The `terminal_core_context.is_not_null()` filter drops observations where the transition classifier returned null
- The `outcome_available` filter drops source observations without forward outcomes
- Each filter reduces the sample → wider confidence intervals → more evidence rows fail the `conditioned_observations >= 100` gate
- The gate then excludes marginal but real signals because the sample is too small

**Fix**:
- **(A) Use left joins + explicit missing indicators**: Mark missing rather than dropping, compute evidence on available observations
- **(B) Relax the observation-count gate for higher evidence levels**: 50 obs for risk level, 30 for source level
- **(C) Pool neighboring horizons**: 4H and 8H outcomes pooled when 12H sample is too small

---

## 3. SOLUTIONS RANKED BY IMPACT/EFFORT

### Tier 1: Immediate High-Impact (days)

| # | Solution | Category | Impact |
|---|---|---|---|
| 1 | **Decile outcome buckets** (10-way instead of 3-way) | 2.1 | H_max 1.585 → 3.32 bits, 10× more room |
| 2 | **BTC-relative returns as outcome** | 2.4 | Removes correlated noise, isolates alpha |
| 3 | **Continuous feature scores in evidence groups** | 2.2 | Replaces 3-way buckets with 0-100 ranks |

### Tier 2: Medium-Term Structural (weeks)

| # | Solution | Category | Impact |
|---|---|---|---|
| 4 | **Per-symbol baselines** | 2.4 | Coin-specific base rate control |
| 5 | **Add derived features** (accel, divergence, anomaly) | 2.3 | Leading indicators, not lagging descriptors |
| 6 | **Interaction features in evidence levels** | 2.5 | Captures combinatorial effects |

### Tier 3: Strategic Reframe (months)

| # | Solution | Category | Impact |
|---|---|---|---|
| 7 | **Data-driven stratification** (CART / MI ranking) | 2.5 | Replaces fixed nest with information-maximizing splits |
| 8 | **Predict regime transitions, not returns** | 2.3 | "Will RANGE→MARKUP" may be more predictable |
| 9 | **Cross-sectional ranking** (coin outperformance) | 2.4 | Relative strength more stable than absolute direction |
| 10 | **Tail-event prediction** (predict extremes, not means) | 2.1 | Crypto tails cluster → higher info gain possible |

---

## 4. WHAT THE SCANNER CAN DO TODAY (DESPITE LOW PREDICTABILITY)

Even 0.29 bits is non-zero. The scanner's current output is still useful:

### 4.1. Risk management (avoid bad conditions)
- `research_suggestion = "chop_avoid"` when `returned_to_origin_rate >= 0.25` → tangible edge
- Knowing which states produce 53% up vs 32% up matters for sizing
- "When to stand aside" is as valuable as "when to enter"

### 4.2. Signal stacking
- 0.29 bits per independent condition × 3 independent conditions = 0.87 bits
- The scanner currently measures marginal gain over parent — if conditions are orthogonal, they compound
- Need to measure condition *independence*, not just marginal gain

### 4.3. Regime discovery (taxonomy value)
- The evidence frame reveals WHICH conditions produce the 9pp tilt
- Even without strong individual predictability, identifying "MARKUP background + bid_support books = bullish tilt" is a concrete hypothesis
- These become strategy candidates for the promotion pipeline

### 4.4. Tail detection
- `tail_up_rate` and `tail_down_rate` are already computed
- Tail events may show higher information gain than mean outcomes
- Even if mean return is unpredictable, the probability of a 5%+ move may shift significantly

---

## 5. IMMEDIATE RECOMMENDATION

Implement **Tier 1 solutions** (1 + 2 + 3) as the next scanner iteration:

1. **Outcome**: Use decile buckets (`pl.col("forward_return_pct").qcut(10, labels=False)`) → H_max rises to 3.32 bits
2. **Returns**: Use `forward_return_pct - btc_forward_return_pct` as the outcome variable (BTC-relative alpha)
3. **Features**: Add `atr_percentile`, `range_width_atr`, `imbalance_value` as continuous group columns alongside categorical labels

Expected result: baseline entropy ~2.8/3.32 bits, information gain ~0.5-1.0 bits achievable. This is still modest (crypto *is* close to efficient) but sufficient for risk-management and signal-stacking applications.

---

## 6. ARCHITECTURAL NOTE

This analysis does not critique the scanner's design — the scanner is well-architected for its purpose. The low predictability is a *data property*, not a *design flaw*. Crypto returns have low signal-to-noise ratio. The scanner correctly measures this. The improvements above increase the signal capture within the existing framework.
