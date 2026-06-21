# Extreme-event ponytail feature implementation plan

## User direction

```text
do not use sin/cos time sampling
summarize above talking into detail plan along with ponytail skill reduction in evaluation of current codespace
then apply coding
record current benchmark of metric
run advanced tailtree config again
check improvement
```

## Current codespace evaluation

Current local worktree already moved the scanner target toward the extreme-event thesis:

```toml
# configs/potential-advanced-tailtree.toml
max_symbols = 160
threshold_pct = 30.0
outcome_horizon = [24]
max_trials = 5

# configs/potential-daily-tailtree.toml
threshold_pct = 30.0
```

Current feature contract:

```text
categorical:
  background_regime
  swing_core
  decision_core
  decision_transition
  decision_direction

continuous:
  atr_percentile
  range_width_atr
  return_1bar
  return_4bar
  return_24bar
  vol_anomaly
  close_to_range_high_ratio
  funding_rate
  oi_delta
  taker_buy_sell_ratio
  long_short_ratio
  funding_age_ms
  oi_age_ms
  taker_age_ms
  lsr_age_ms
```

Current gap against `>=30%` altcoin events:

```text
- 1H-only price memory is too short/coarse: 1h/4h/24h only.
- Current range geometry is only 48 bars.
- No realized volatility scale/compression feature.
- Source/crowding features are mostly levels or one-step deltas.
- Existing selection-efficiency artifact is enough for comparison; do not add new diagnostics.
```

## Ponytail reduction rule for this change

Do not build a feature framework.
Do not add sin/cos time encodings.
Do not add separate diagnostics.
Do not add new report sections.
Do not add a generic market-feature registry.

Use the existing path:

```text
state.py continuous features -> tailrun.core feature allowlist -> tailtree-selection-efficiency.csv
```

One compact feature block, one existing metric surface.

## Theory filter: robust information vs garbage

A feature is allowed only if it has a market mechanism and a threshold/scale meaning for LightGBM:

```text
price memory          -> multi-window returns
volatility scale      -> realized vol and compression ratio
range geometry        -> position in 7d/30d range
flow/crowding change  -> OI/taker/long-short deltas over 24h
```

Rejected:

```text
sin/cos time sampling
arbitrary Fourier features
high-dimensional embeddings
one-off symbol diagnostics
```

Reason: LightGBM splits thresholds. It benefits from interpretable rolling-window scalars, not dense vector geometry unless the cycle is real. We are not adding cyclic features in this pass.

## Feature block to add

### Kline features

Add to `_kline_continuous_features`:

```text
return_6bar
return_12bar
return_48bar
return_72bar
realized_vol_24bar
realized_vol_72bar
vol_compression_24_168
range_position_168bar
range_position_720bar
range_compression_48_720
volume_anomaly_72bar
```

Definitions:

```text
return_Nbar = close / close.shift(N) - 1, percent
realized_vol_Nbar = rolling std of 1h returns over N bars
vol_compression_24_168 = realized_vol_24bar / realized_vol_168bar
range_position_Nbar = (close - rolling_low_N) / (rolling_high_N - rolling_low_N)
range_compression_48_720 = range_48bar / range_720bar
volume_anomaly_72bar = volume / rolling_mean(volume, 72)
```

Why these are not garbage:

```text
returns: event acceleration / path memory
realized volatility: coin-scale normalization
compression: coiled state before expansion
range position: breakout / breakdown geometry
volume anomaly: participation expansion
```

### Source features

Add compact 24h changes where historical source rows exist:

```text
oi_change_24h_pct
taker_buy_sell_ratio_delta_24h
long_short_ratio_delta_24h
```

Funding stays level-only in this pass because funding cadence is coarser; a robust z-score needs enough historical rows and is not worth adding as a one-off now.

## Tailtree feature allowlist update

Add only the new columns to `_TAILTREE_CONTINUOUS_TRAIN_FEATURES`.

No change to objective in this pass. Objective redesign is larger; this pass isolates feature-quality improvement and compares metrics.

## Benchmark protocol

Use the existing advanced config and artifact:

```bash
python scripts/scanner_potential.py --config configs/potential-advanced-tailtree.toml
```

Baseline:

```text
current code + current local advanced config
```

Improved:

```text
same config + compact feature block
```

Compare only existing `tailtree-selection-efficiency.csv` columns:

```text
valid_tail_lift
selected_tail_count
selected_observation_count
selected_utility_mean
selected_utility_p90
profit_proxy_per_selected_obs
hpo_score
promotion_threshold_pass_int
```

Primary improvement criterion:

```text
best feasible hpo_score improves without support collapse
```

Secondary criteria:

```text
valid_tail_lift improves
selected_tail_count remains >= min_selected_tail_count
selected_observation_count remains >= min_selected_observation_count
utility p90 improves or does not materially degrade
```

## Implementation steps

1. Run baseline advanced config and record metric snapshot.
2. Patch `src/qooi/scanner/state.py` with the compact kline/source feature block.
3. Patch `src/qooi/scanner/tailrun/core.py` feature allowlist.
4. Run ruff/ty/tests.
5. Run advanced config again.
6. Compare baseline vs improved selection-efficiency metrics.
7. Keep the change only if the existing metric surface shows useful or neutral-with-clear-next-step behavior.

## Explicitly skipped

```text
sin/cos time encodings
new diagnostics files
new report sections
event-lift objective branch
multi-horizon config expansion
4H/1D config activation
```

Those are separate decisions. This pass tests whether a compact, mechanism-based feature block improves the current 30% h24 advanced target.
