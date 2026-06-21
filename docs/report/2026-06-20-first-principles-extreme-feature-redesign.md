# First-principles extreme-event feature redesign

## User direction

Design features from first principles, not by adding a random pile of columns.

Constraints:

```text
- target: >=30% altcoin extreme event behavior
- model: LightGBM tail_event_lift
- no sin/cos/Fourier time-sampling bank
- ponytail: compact, mechanism-based, reversible
- diagnostics: reuse tailtree-selection-efficiency.csv only
```

## Current codespace state

The current feature surface has the event-lift objective and a small normalized mapping pass.

Current continuous training features:

```text
atr_percentile
range_width_atr
return_1bar
return_4bar
return_24bar
return_4bar_vol_scaled
return_24bar_vol_scaled
vol_anomaly
close_to_range_high_ratio
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

This means we have:

```text
✅ event-lift labels over all observations
✅ h24/h48 advanced horizons
✅ some normalized source/bar aliases
❌ no full compression -> expansion grammar
❌ no broader 7d/30d range geometry
❌ no momentum acceleration grammar beyond 1/4/24 bar returns
❌ no explicit crowding-delta grammar beyond one-step OI pct and bounded taker pressure
❌ no symbol lifecycle / prior tail propensity separation
```

The previous normalized pass was useful but incomplete. It was not a full feature redesign.

## First principles

The scanner should not ask "which columns can we add?"

It should ask:

```text
At a known decision close, what market mechanism can plausibly make a future >=30% path event more likely?
```

For altcoin extreme events, the reusable mechanisms are:

```text
1. compression -> expansion
2. momentum acceleration
3. crowding / forced positioning
4. liquidity sweep / breakout acceptance
5. symbol lifecycle / base propensity
```

Each mechanism should map to a small, consistent feature grammar:

```text
state level
short-window change
long-window context
normalized abnormality
```

Not every mechanism needs every field. Ponytail rule: keep only the smallest set that tests the mechanism.

## Feature grammar

### 1. Compression -> expansion

Theory:

```text
Large alt moves often start after local volatility/range contracts, then volume/price expands through a broader range boundary.
```

Feature intent:

```text
Is the coin quiet relative to its own recent history?
Is it near a broad range edge?
Is volume starting to expand?
```

Compact features:

```text
range_compression_48_720      = 48h high-low range / 30d high-low range
realized_vol_24_168_ratio     = 24h return vol / 7d return vol
volume_anomaly_24_168         = 24h volume mean / 7d volume mean
range_position_720            = close position inside 30d high-low range
```

Skip for now:

```text
distance_to_30d_high
_distance_to_30d_low
```

Reason: they are redundant with `range_position_720`. One position scalar is enough for LightGBM.

### 2. Momentum acceleration

Theory:

```text
Extreme events are not just high return. They often show convexity: short-window movement outruns longer-window movement.
```

Feature intent:

```text
Is momentum accelerating rather than merely trending?
```

Compact features:

```text
return_12bar
return_48bar
return_24bar_vol_scaled
momentum_accel_4_24 = return_4bar - return_24bar / 6
```

Keep existing:

```text
return_1bar
return_4bar
return_24bar
return_4bar_vol_scaled
```

Skip for now:

```text
return_72bar
```

Reason: advanced config now uses h24/h48 for runtime. h72 feature/horizon can wait until h72 earns its cost again.

### 3. Crowding / forced positioning

Theory:

```text
Forced flow comes from positioning imbalance: OI, funding, L/S ratio, and taker flow become stretched or move against price.
```

Feature intent:

```text
Is leverage/crowding building?
Is flow one-sided?
Is price disagreeing with positioning?
```

Compact features:

```text
oi_delta_pct                 existing one-step pct change
funding_rate_bps             existing normalized level
taker_buy_pressure           existing bounded flow imbalance
long_short_log_ratio         existing bounded crowding level
price_oi_pressure_24         = return_24bar_vol_scaled * oi_delta_pct
funding_stress_down_4        = funding_rate_bps * min(return_4bar, 0)
```

Skip for now:

```text
funding_z_7d
long_short_delta_24h
```

Reason: source history depth and cadence are uneven. Add source rolling z/deltas only after this compact interaction pass proves value. Current source features already suffer from freshness/capability edges; do not multiply source columns yet.

### 4. Liquidity sweep / breakout acceptance

Theory:

```text
A 30% event often begins near a prior range edge or after a sweep/reclaim. Current state machine has local transition labels, but continuous features need broader range context.
```

Feature intent:

```text
Is price near the 30d boundary?
Is it breaking out from compression?
```

Compact features:

```text
range_position_720
range_compression_48_720
```

Use existing categorical states:

```text
decision_transition
swing_core
background_regime
```

Skip for now:

```text
explicit failed_breakout_7d flags
explicit sweep_7d flags
```

Reason: current state machine already encodes local breakout/sweep categories. First add broad continuous context; do not duplicate state-machine labels into another handcrafted categorical system.

### 5. Symbol lifecycle / base propensity

Theory:

```text
New coins and low-depth high-volatility alts dominate >=30% tails. The model must distinguish baseline symbol propensity from current setup edge.
```

Feature intent:

```text
Is this symbol structurally prone to 30% events?
How much history exists?
Is the model selecting current setup or just selecting young/volatile coins?
```

Compact features:

```text
history_depth_days
symbol_tail_rate_30_h24
symbol_tail_rate_30_h48
```

Skip for now:

```text
listing_age_hours
symbol_volatility_rank
```

Reason: `history_depth_days` is the directly available lifecycle proxy from bars. Tail-rate features are more directly tied to the label. Volatility rank overlaps with return-vol scaling and can be added later if the model still over-selects known violent symbols.

## Target compact feature set

Add only these new columns beyond current state:

```text
# compression / range
range_compression_48_720
realized_vol_24_168_ratio
volume_anomaly_24_168
range_position_720

# acceleration
return_12bar
return_48bar
momentum_accel_4_24

# crowding interactions
price_oi_pressure_24
funding_stress_down_4

# lifecycle / base propensity
history_depth_days
symbol_tail_rate_30_h24
symbol_tail_rate_30_h48
```

That is 12 new columns.

Ponytail rationale:

```text
12 columns cover 5 mechanisms.
No duplicate distance/high/low columns.
No h72 until h72 pays for itself.
No source rolling z-score pile.
No sin/cos.
No new diagnostics.
```

## Implementation shape

### `state.py`

Extend `_kline_continuous_features()` with native Polars expressions:

```text
return_12bar
return_48bar
realized_vol_24_168_ratio
volume_anomaly_24_168
range_position_720
range_compression_48_720
momentum_accel_4_24
```

Extend source interaction after source joins are materialized:

```text
price_oi_pressure_24
funding_stress_down_4
```

Do not create a feature framework or class.

### `tailrun/core.py`

Add only the selected feature names to `_TAILTREE_CONTINUOUS_TRAIN_FEATURES`.

### Lifecycle features

Add lifecycle/base-propensity only if it can be done directly from existing outcome/history frames without another artifact family.

Preferred location:

```text
tailrun/core.py training preparation
```

Reason: `symbol_tail_rate_30_h24/h48` depends on the current threshold/horizon/outcome labels. It is target-family metadata, not raw market state. It should be computed inside the tailtree preparation path to avoid leaking scanner-global label assumptions into generic `state.py`.

Leakage rule:

```text
For a decision row, symbol prior tail rate must use only past rows before that decision timestamp.
```

Ponytail cut:

```text
If past-only symbol prior is not a small Polars expression, skip symbol_tail_rate_* for this pass and keep only history_depth_days.
```

No approximate full-sample target encoding. That would leak.

## Benchmark rule

Use the existing comparison only:

```text
tailtree-selection-efficiency.csv
report.md
profile/stages.csv
model metadata feature list
```

Compare against:

```text
data/output/potential/benchmarks/normalized-bounded-tailtree-selection-efficiency.csv
```

Keep if:

```text
best_lift / hpo_score improves or stays close
best_profit_proxy does not materially degrade
runtime remains within the bounded Optuna envelope
feature metadata shows only the designed columns were added
```

Revert if:

```text
feature_count grows but lift/utility degrade
```

## Apply order

1. Add bar mechanism features only.
2. Add two crowding interaction features.
3. Run checks.
4. Run bounded advanced benchmark.
5. Compare against normalized-bounded baseline.
6. Keep or revert.
7. Only after that consider lifecycle prior features, because they carry leakage risk.

## Current recommendation

Do not implement all five families at once.

Implement first:

```text
compression/range + acceleration + two crowding interactions
```

Defer:

```text
symbol_tail_rate_30_h24/h48
```

Reason: lifecycle tail-rate needs careful past-only encoding. It is theoretically important, but not worth adding incorrectly.
