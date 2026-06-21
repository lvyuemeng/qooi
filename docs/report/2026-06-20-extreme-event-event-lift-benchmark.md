# Extreme-event event-lift objective benchmark

## Change tested

Ponytail objective/horizon change only:

```text
- no new feature block
- no sin/cos/Fourier time sampling
- no new diagnostics/artifact family
- reused tailtree-selection-efficiency.csv
```

Code/config change:

```toml
[potential.bars]
timeframes = ["1H", "4H", "1D"]

[potential.evidence.tailtree]
threshold_pct = 30.0
outcome_horizon = [24, 48, 72]

[[potential.evidence.tailtree.profiles]]
objective = "tail_event_lift"
```

## Feature -> label process

### 1. Known-at-close features

`state.potential_observation_frame(...)` builds one observation row per:

```text
symbol × decision_bar_close_ms
```

The model sees only information known at that decision close:

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

No extra rolling feature block was kept after the previous benchmark failed.

### 2. Future path labels

`outcome.potential_outcome_frame(...)` computes future path outcomes for each decision row and horizon:

```text
h24, h48, h72
```

`label_tail_exceedances(..., threshold_pct=30.0)` creates direction-specific labels:

```text
up event:
  tail_up = forward_max_return_pct > 30

down event:
  tail_down = forward_min_return_pct < -30
```

It also keeps utility columns:

```text
tail_utility_up
tail_utility_down
```

Utility is still used for selection/reporting, but no longer as the model's training target in this objective.

### 3. Event-lift training data

For `tail_event_lift`, training uses all aligned observations, not only tail rows:

```text
X = known-at-close features for all decision rows
 y = 1 if the direction/horizon produced a >=30% path event, else 0
```

Direction examples:

```text
up/h48:
  y = 1 if forward_max_return_pct over next 48 bars > 30

down/h24:
  y = 1 if forward_min_return_pct over next 24 bars < -30
```

### 4. LightGBM objective

`tail_event_lift` trains LightGBM with:

```text
objective = binary
metric = binary_logloss
```

This asks:

```text
which current states separate future 30% events from normal rows?
```

This is different from the previous `tail_utility_quantile` objective, which trained only on tail rows and optimized severity/utility among known events.

### 5. Selection-efficiency surface

The trained classifier scores validation/current rows. Existing score buckets are reused:

```text
top_1pct
top_2pct
top_5pct
top_10pct
```

Each bucket is evaluated by the existing metrics:

```text
selected_observation_count
selected_tail_count
selected_tail_rate
valid_tail_lift
selected_utility_mean
selected_utility_p90
hpo_score
promotion_threshold_pass_int
```

No new diagnostics were added.

## Benchmark runs

### Baseline

Previous current advanced baseline:

```text
objective: tail_utility_quantile
threshold_pct: 30.0
outcome_horizon: [24]
timeframes: ["1H"]
runtime: 212s
rows: 80
```

### Event-lift run

```text
objective: tail_event_lift
threshold_pct: 30.0
outcome_horizon: [24, 48, 72]
timeframes: ["1H", "4H", "1D"]
runtime: 2258s
models: 60
selection rows: 240
```

The event-lift run is much slower because it trains:

```text
5 Optuna trials × 2 folds × 3 horizons × 2 directions = 60 models
```

and trains on all observations rather than tail-only rows.

## Aggregate comparison

| Metric | Baseline | Event-lift | Change |
|---|---:|---:|---:|
| rows | 80 | 240 | +160 |
| horizons | 1 | 3 | +2 |
| feature_count | 20 | 20 | 0 |
| passing_rows | 51 | 240 | +189 |
| best_hpo_score | 22.297248 | 58.295772 | +161.45% |
| best_valid_tail_lift | 18.613551 | 54.712559 | +193.94% |
| best_profit_proxy_per_selected_obs | 2.820659 | 2.752706 | -2.41% |

## Event-lift by horizon

| Horizon | Rows | Passing rows | Best HPO | Best lift | Max selected tails | Best profit proxy |
|---:|---:|---:|---:|---:|---:|---:|
| h24 | 80 | 80 | 58.295772 | 54.712559 | 4400 | 2.512542 |
| h48 | 80 | 80 | 45.281889 | 41.559719 | 6395 | 2.752706 |
| h72 | 80 | 80 | 32.504793 | 28.607173 | 5335 | 2.470873 |

## Interpretation

Event-lift improved exactly the metric it was meant to improve:

```text
rare-event concentration / lift
```

The strongest result is still h24, especially down-side:

```text
best row: h24 down
hpo_score: 58.295772
valid_tail_lift: 54.712559
selected_utility_mean: 1.378673
selected_utility_p90: 5.123748
```

Compared with baseline:

```text
valid_tail_lift: 18.61 -> 54.71
hpo_score:       22.30 -> 58.30
```

Utility did not improve:

```text
best profit proxy: 2.820659 -> 2.752706
```

That is expected: event-lift optimizes event probability/lift, not severity. It finds buckets with far higher event concentration, while utility remains a second-stage ranking problem.

## Decision

Keep the event-lift objective and advanced horizon/context config.

Do not add more features yet.

Next ponytail follow-up should be one of:

```text
1. make daily config use the best proven horizon/objective in a bounded fixed profile
2. add a small severity rerank after event-lift buckets, using existing utility columns
3. reduce advanced runtime by cutting max_trials or horizons after choosing h24/h48
```

Do not re-add the broad feature block; previous benchmark showed it diluted lift.
