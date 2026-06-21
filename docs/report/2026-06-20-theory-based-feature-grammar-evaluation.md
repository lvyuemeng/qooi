# Theory-based scanner feature grammar evaluation

## User correction

The user does not want another list of plausible features.

The real requirement is:

```text
Find a generalized, robust extraction pattern / grammar.
It must have a theory base.
It must explain current class-state, transitions, and feature workflow.
It must avoid random feature additions.
```

So this report is not an implementation plan yet. It is the theory and code-evaluation layer that should gate future feature work.

## Current calculation workflow

### 1. Kline classifier state workflow: `src/qooi/scanner/state.py`

`KlineClassifier.classify()` calls `classify_states(frame, scale)` for each symbol/timeframe.

The classifier builds known-at-close symbolic state from OHLCV only:

```text
ATR_14
ATR percentile over 100 bars
5-bar swing highs/lows
48-bar range high/low
20-bar liquidity high/low
range_width_atr
trend structure from rolling swing counts
local liquidity event type
```

Then it projects each bar into categorical state:

```text
market_stage:
  warmup
  markup
  markdown
  transition
  accumulation
  distribution_or_reversal
  range
  trend_continuation
  wide_range
  unknown

structure_trend_state:
  uptrend
  downtrend
  range
  unknown

liquidity_event_type:
  breakout_acceptance_high
  breakout_acceptance_low
  failed_breakout_high
  failed_breakout_low
  none
```

Finally it forms a categorical `state_key`:

```text
market_stage | structure_trend_state | range_width_bucket | volatility_bucket
```

and a `context_event`:

```text
liquidity_event_type or none_in_accumulation/distribution/trend/compression
```

This is already a grammar, but it is a **symbolic regime grammar**, not a full numeric feature grammar.

### 2. Transition workflow: `src/qooi/scanner/transitions.py`

Transitions are computed from sequences of `state_key` and `contextual_event`.

For each symbol/timeframe:

```text
classified kline rows
-> prev_state
-> n-gram transition_path
-> future close-to-close return
-> future MFE/MAE path return
```

Then it aggregates by transition path:

```text
count
symbol_count
transition probability
recent probability
long probability
probability delta
p_up / p_down
average/median/quantile future return
forward min/max path quantiles
reward/risk
information bits
conditional information bits
```

This is a categorical empirical-state model. It asks:

```text
Does a symbolic state path change the distribution of future returns?
```

It is useful, but it has limits:

```text
- discrete buckets lose magnitude information
- buckets are local-window dependent
- path frequency can be sparse
- it mostly sees state sequences, not continuous microstructure pressure
```

### 3. Continuous feature workflow: `state.py` -> `tailrun/core.py`

`extract_continuous_features()` builds numeric known-at-close features.

Current implemented families:

```text
bar:
  return_1bar
  return_4bar
  return_24bar
  return_4bar_vol_scaled
  return_24bar_vol_scaled
  vol_anomaly
  close_to_range_high_ratio
  atr_percentile
  range_width_atr

source:
  funding_rate
  funding_rate_bps
  oi_delta
  oi_delta_pct
  taker_buy_sell_ratio
  taker_buy_pressure
  long_short_ratio
  long_short_log_ratio
  funding_age_ms
  oi_age_ms
  taker_age_ms
  lsr_age_ms
```

`tailrun/core.py` allowlists exactly these continuous columns plus five categorical columns:

```text
background_regime
swing_core
decision_core
decision_transition
decision_direction
```

So the current tailtree input is:

```text
categorical state grammar + compact numeric bar/source features
```

## Problem with my previous feature list

The previous mechanism list:

```text
compression -> expansion
momentum acceleration
crowding / forced positioning
liquidity sweep / breakout acceptance
symbol lifecycle
```

is market-intuitive, but it is not by itself a sufficient theory base.

It becomes theory-grounded only if each feature is derived from a small set of invariance principles.

Without those principles, the list can degrade into random feature accumulation.

## Robust extraction theory base

The robust grammar should be derived from four invariance principles.

### Principle 1: Known-at-close filtration

A feature is valid only if it belongs to the information set available at decision close:

```text
F_t = all bars/source snapshots with timestamp <= decision_bar_close_ms
```

Any feature using future bars or full-sample future labels is invalid.

Implication:

```text
rolling windows must use current/past rows only
source rows must be backward as-of joined by symbol
symbol prior rates must be past-only, not full-sample target encoding
```

Current code status:

```text
✅ classifier states use shifts/rolling past windows
✅ source features use backward as-of joins
✅ event labels stay in outcome/tailtree path
⚠️ symbol prior tail-rate is not implemented; if added, leakage risk is high
```

### Principle 2: Scale invariance / comparability

Altcoins differ by price level, volatility, liquidity, and contract scale. Raw values are often not comparable.

A robust feature should be dimensionless or locally normalized:

```text
ratio
percent change
z/rank/percentile
volatility-scaled return
bounded imbalance
log ratio
age/freshness
```

Current code status:

```text
✅ return_* are percent returns, not raw prices
✅ return_*_vol_scaled normalizes return by local 1H volatility
✅ funding_rate_bps makes funding scale explicit
✅ oi_delta_pct is better than raw oi_delta
✅ taker_buy_pressure bounds buy/sell imbalance to [-1, 1]
✅ long_short_log_ratio symmetrizes L/S ratio around 0
⚠️ raw oi_delta, raw taker_buy_sell_ratio, raw long_short_ratio are still kept
```

Keeping raw aliases is not automatically wrong for LightGBM, but the normalized version should be the primary conceptual feature.

### Principle 3: Multi-scale consistency, not arbitrary window expansion

A feature is robust when it measures the same phenomenon across nested horizons:

```text
short state
medium context
long baseline
```

For 1H bars, a minimal nested set is:

```text
1h / 4h / 24h / 7d / 30d
```

But this does not mean adding every window. The grammar is:

```text
short / baseline
or
short - baseline rate
or
short / baseline ratio
```

Current code status:

```text
✅ 1h/4h/24h returns exist
✅ 24h return is volatility-scaled by 7d return volatility
⚠️ 7d/30d range context is only partly present through 48-bar range position
❌ no explicit short-vs-long compression ratio
❌ no explicit 24h-vs-7d volume expansion ratio
```

The issue is not "missing features" in the abstract. The issue is incomplete nested-scale representation.

### Principle 4: State variables before mechanism names

A market mechanism should be decomposed into primitive state variables.

The primitive variables are:

```text
location: where is price inside recent range?
velocity: return over a window
acceleration: short velocity minus long velocity rate
dispersion: realized volatility / range width
compression: short dispersion vs long dispersion
participation: volume / turnover relative to baseline
crowding: OI/funding/LSR/taker imbalance
freshness: source age and availability
lifecycle: history depth and past-only base propensity
```

Mechanism labels are then derived from combinations:

```text
compression -> expansion = low compression + participation expansion + boundary location
momentum acceleration = positive/negative velocity + acceleration
forced positioning = crowding + adverse price movement
sweep/breakout = boundary location + categorical liquidity event
```

This is the key correction: **mechanism names are not the grammar**. Primitive state variables are the grammar.

## Generalized feature grammar

The general extractor should follow this template for each data family.

### Bar family grammar

Input:

```text
open, high, low, close, volume
```

Primitive transforms:

```text
level-free returns:
  r(w) = close / close.shift(w) - 1

realized dispersion:
  vol(w) = std(r(1), w)

range location:
  pos(w) = (close - rolling_low(w)) / (rolling_high(w) - rolling_low(w))

range compression:
  comp(short,long) = range(short) / range(long)

participation:
  vol_part(short,long) = mean(volume, short) / mean(volume, long)

acceleration:
  accel(short,long) = r(short) - r(long) * short / long

normalized velocity:
  norm_r(w,long) = r(w) / vol(long)
```

This is theory-based because every output is:

```text
known-at-close
scale-free
multi-scale
interpretable as a market state primitive
```

### Source family grammar

Input:

```text
funding, open_interest, taker volume, long_short ratio, books/trades where historically available
```

Primitive transforms:

```text
bounded imbalance:
  (buy - sell) / (buy + sell)

percent change:
  (x - x.shift(k)) / x.shift(k)

log ratio:
  log(ratio)

basis points:
  rate * 10000

source age:
  decision_time - source_time
```

Only add rolling z-score/rank when source history depth and cadence are reliable.

Current source cache/report shows some families are provider-bounded/current-ish; books/trades are not historical training coverage. Therefore broad source rolling features are not robust yet.

### State/transition grammar

State features should remain categorical summaries of bar structure:

```text
stage
trend state
range bucket
volatility bucket
liquidity event
```

Transition features should remain empirical path summaries:

```text
state n-gram -> future distribution shift
```

Do not duplicate transition logic as a pile of numeric flags. The categorical path grammar already handles local sweep/breakout acceptance.

Numeric features should complement it by providing continuous magnitude/context:

```text
range position
compression ratio
normalized return
participation ratio
crowding imbalance
```

## What is theoretically validated vs speculative

### Strong theory base: keep/add first

These are robust because they satisfy all four principles.

```text
return_4bar_vol_scaled
return_24bar_vol_scaled
range_position_long
range_compression_short_long
volume_participation_short_long
momentum_acceleration_short_long
taker_buy_pressure
long_short_log_ratio
oi_delta_pct
funding_rate_bps
source_age_ms
```

### Medium theory base: add only after source cadence is stable

```text
funding_z_7d
long_short_delta_24h
taker_pressure_z_24h
oi_delta_z_7d
```

Reason: they require reliable historical source cadence. If source history is shallow or provider-bounded, rolling z-scores can be fake precision.

### Leakage-risk / high caution

```text
symbol_prior_30pct_tail_rate
symbol_volatility_rank
listing_age_hours if inferred from available cache only
```

Reason: lifecycle/base propensity is theoretically important, but easy to encode with future/full-sample knowledge. Must be past-only per decision timestamp.

### Weak theory base for this model/goal

```text
sin/cos time-sampling banks
arbitrary Fourier encodings of bar index
large windows with no primitive relation
many duplicate distances to high/low plus range position
```

Reason: LightGBM threshold splits prefer interpretable monotone/ratio features. Harmonic embeddings are useful only for true cycles, not generic price-path representation.

## Current code gap analysis

### Good current pieces

```text
✅ known-at-close state construction
✅ local trend/range/liquidity categorical grammar
✅ transition path empirical distribution grammar
✅ source as-of materialization by symbol
✅ event-lift objective over all observations
✅ normalized source/bar aliases started
```

### Weak current pieces

```text
⚠️ state classifier uses hard thresholds: 48-bar range, 20-bar liquidity, ATR width <= 8
⚠️ continuous features are not generated from a declared grammar; they are manually listed
⚠️ bar numeric features still underrepresent location/compression/participation primitives
⚠️ source rolling primitives are avoided for good reason, but this leaves crowding temporal change thin
⚠️ transition module is separate from tailtree features; its information bits do not become model inputs
```

### Do not fix by adding a feature framework

A framework would be overkill. Ponytail solution is not a class hierarchy.

The smallest robust improvement is:

```text
write the grammar in docs/report
then add one compact Polars block in state.py that implements primitive bar transforms
```

No feature registry. No transform DSL. No plugin system.

## Revised ponytail feature plan

The next feature code should not be named after mechanisms.

It should be named after primitives:

```text
range_position_720
range_compression_48_720
realized_vol_ratio_24_168
volume_participation_24_168
return_12bar
return_48bar
momentum_accel_4_24
```

Add at most two interaction features:

```text
price_oi_pressure_24 = return_24bar_vol_scaled * oi_delta_pct
funding_stress_down_4 = funding_rate_bps * min(return_4bar, 0)
```

But interactions are second priority. Primitive variables first.

## Why this is more theory-based than the previous mechanism list

Previous form:

```text
compression -> expansion, momentum, crowding, sweep, lifecycle
```

This is an explanatory story.

Revised form:

```text
filtration + scale invariance + multi-scale consistency + primitive market variables
```

This is a generative grammar.

It tells us:

```text
which features are valid
which are redundant
which are leakage-risk
which require better source cadence
which are likely garbage for LightGBM
```

## Recommendation

Do not code the full previous mechanism list.

Use this implementation order:

1. **Bar primitive pass**

```text
range_position_720
range_compression_48_720
realized_vol_ratio_24_168
volume_participation_24_168
return_12bar
return_48bar
momentum_accel_4_24
```

2. **Only if primitive pass holds metrics**, add two source interactions:

```text
price_oi_pressure_24
funding_stress_down_4
```

3. **Only later**, design past-only lifecycle/base propensity.

```text
history_depth_days
past_symbol_tail_rate_30_h24
past_symbol_tail_rate_30_h48
```

This respects ponytail because it adds the fewest columns that complete the missing primitive grammar and uses the existing benchmark surface for acceptance/reversion.
