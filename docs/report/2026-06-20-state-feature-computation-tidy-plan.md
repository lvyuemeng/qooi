# State/feature computation tidy plan

## Problem

The scanner now has overlapping concepts in three places:

```text
1. classify_states()          -> symbolic regime/range/vol/liquidity buckets
2. outcome.kline_path_rows()  -> categorical history/path projection from state_key
3. continuous_features_frame() -> numeric bar/source features for tailtree
```

The overlap is not fatal, but it is unclear:

```text
range compression appears as state bucket and numeric range_width_atr
volatility appears as ATR percentile bucket and numeric atr_percentile
liquidity breakout appears as categorical event, but range location is numeric
momentum exists as raw returns, but not as a declared primitive
```

The user wants a theory-based extraction grammar, not random features.

## Boundary rule

Use one boundary:

```text
classifier/outcome/transitions = symbolic categorical grammar
continuous_features            = numeric primitive grammar
tailtree                       = consumes both; does not invent features
```

## Keep classifier symbolic

`classify_states()` remains the owner of:

```text
market_stage
structure_trend_state
liquidity_event_type
state_key
context_event
direction_hint
quality_weight
```

It should not grow numeric model features.

Reason:

```text
Classifier state is for discrete regime/path evidence and reportability.
```

## Keep outcome path projection categorical

`outcome.kline_path_rows()` remains the owner of:

```text
regime_state
structure_state
core_context
transition_kind
state_age/event_age
compression_state/expansion_state categorical labels
transition_path
```

It should not grow numeric model features.

Reason:

```text
Outcome path rows define symbolic path context and future label surfaces.
```

## Make continuous features primitive numeric state

`continuous_features_frame()` should implement the numeric primitive grammar:

```text
velocity
acceleration
dispersion
compression
location
participation
crowding
freshness
```

This complements, not duplicates, the symbolic state grammar.

## Current safe code pass

Add only bar primitive features in `_kline_continuous_features()`:

```text
return_12bar
return_48bar
momentum_accel_4_24
realized_vol_ratio_24_168
volume_participation_24_168
range_position_720
range_compression_48_720
```

Definitions:

```text
return_12bar = close / close.shift(12) - 1
return_48bar = close / close.shift(48) - 1
momentum_accel_4_24 = return_4bar - return_24bar * 4/24
realized_vol_ratio_24_168 = rolling_std(return_1bar, 24) / rolling_std(return_1bar, 168)
volume_participation_24_168 = rolling_mean(volume, 24) / rolling_mean(volume, 168)
range_position_720 = (close - rolling_low_720) / (rolling_high_720 - rolling_low_720)
range_compression_48_720 = (rolling_high_48 - rolling_low_48) / (rolling_high_720 - rolling_low_720)
```

All are:

```text
known-at-close
scale-normalized
multi-scale
primitive rather than story-named
```

## What not to change in this pass

Do not remove existing categorical columns from tailtree allowlist:

```text
background_regime
swing_core
decision_core
decision_transition
decision_direction
```

Reason: this would be a model-input ablation, not a tidy. Keep the categorical grammar stable while numeric primitive grammar is completed.

Do not add lifecycle prior rates yet:

```text
symbol_tail_rate_30_h24
symbol_tail_rate_30_h48
```

Reason: past-only encoding is necessary to avoid leakage and deserves its own pass.

Do not add source rolling z-scores yet:

```text
funding_z_7d
long_short_delta_24h
```

Reason: source cadence/history depth is uneven; broad rolling source features can create fake precision.

## Code changes

1. Patch `_kline_continuous_features()` only.
2. Extend empty schema for new columns.
3. Extend `_TAILTREE_CONTINUOUS_TRAIN_FEATURES` allowlist.
4. Keep source feature functions unchanged.
5. Keep classifier and transition semantics unchanged.

## Verification

Run:

```bash
./.venv/Scripts/python.exe -m ruff format src/qooi/scanner/state.py src/qooi/scanner/tailrun/core.py
./.venv/Scripts/python.exe -m ruff check src tests scripts/scanner_backfill.py scripts/scanner_potential.py
./.venv/Scripts/python.exe -m ty check
./.venv/Scripts/python.exe -m pytest tests/ -q -m "not integration"
```

Slow scanner script benchmark is optional and can be deferred because the user explicitly noted script tests are slow earlier. If run later, compare against:

```text
data/output/potential/benchmarks/normalized-bounded-tailtree-selection-efficiency.csv
```

## Acceptance

Keep if static/tests pass and model feature metadata in the next bounded benchmark shows the primitive columns.

Revert if the bounded benchmark later worsens lift/utility materially.
