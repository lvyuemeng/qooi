# Normalized mapping + bounded Optuna benchmark

## Why features looked old

The previous event-lift change intentionally changed objective/horizon first, but it kept the old continuous feature allowlist.

Model metadata before this change showed:

```text
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

So the user was right: normalized source/bar mapping was not applied yet. The scanner had a better label/objective but mostly old feature inputs.

## Change applied

No sin/cos.
No broad feature block.
No new diagnostics.

### Bar normalized mapping

Added volatility-scaled versions of existing return features:

```text
return_4bar_vol_scaled
return_24bar_vol_scaled
```

Implementation:

```text
return_vol_168bar = rolling_std(return_1bar, 168 bars)
return_4bar_vol_scaled = return_4bar / return_vol_168bar
return_24bar_vol_scaled = return_24bar / return_vol_168bar
```

### Source normalized mapping

Added normalized source aliases:

```text
funding_rate_bps
oi_delta_pct
taker_buy_pressure
long_short_log_ratio
```

Kept existing raw values for continuity:

```text
funding_rate
oi_delta
taker_buy_sell_ratio
long_short_ratio
```

### Advanced runtime reduction

Kept Optuna and walkforward, but bounded the search:

```text
outcome_horizon: [24, 48, 72] -> [24, 48]
max_trials: 5 -> 3
num_iterations_range: [240, 640] -> [240, 420]
early_stopping_rounds_range: [20, 80] -> [20, 50]
```

Expected models:

```text
old: 5 trials × 2 folds × 3 horizons × 2 directions = 60
new: 3 trials × 2 folds × 2 horizons × 2 directions = 24
```

## Runtime evidence

Previous event-lift advanced run:

```text
runtime = 2258s
models = 60
evidence_tailtree = 2170s
```

Normalized bounded run:

```text
runtime = 829s
models = 24
evidence_tailtree = 683s
```

Runtime reduction:

```text
2258s -> 829s = -63.3%
```

The bottleneck was not feature calculation:

```text
continuous_features = 3.04s
observations = 2.21s
evidence_tailtree = 683.02s
```

So runtime reduction must come mostly from model-count / Optuna bounds, not feature extraction micro-optimization.

## Feature metadata after change

Model metadata now includes 21 continuous features:

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

Feature importance confirms normalized features were used:

```text
return_24bar_vol_scaled: rank 4
long_short_log_ratio:    rank 8
return_4bar_vol_scaled:  rank 9
```

## Metric comparison

Compared against saved event-lift benchmark.

### Previous event-lift all horizons

```text
rows: 240
horizons: 3
feature_count: 20
passing_rows: 240
best_hpo_score: 58.295772
best_lift: 54.712559
best_profit: 2.752706
best_utility_p90: 9.188335
max_selected_tails: 6395
runtime: 2258s
```

### Previous event-lift h24/h48 only

```text
rows: 160
horizons: 2
feature_count: 20
passing_rows: 160
best_hpo_score: 58.295772
best_lift: 54.712559
best_profit: 2.752706
best_utility_p90: 9.188335
max_selected_tails: 6395
```

### Normalized bounded run

```text
rows: 96
horizons: 2
feature_count: 26
passing_rows: 96
best_hpo_score: 60.109815
best_lift: 56.816060
best_profit: 2.746921
best_utility_p90: 9.367261
max_selected_tails: 6585
runtime: 829s
```

## Horizon details

```text
h24:
  best_hpo_score = 60.109815
  best_lift = 56.816060
  best_profit = 2.368639
  max_selected_tails = 4555

h48:
  best_hpo_score = 49.270447
  best_lift = 45.056213
  best_profit = 2.746921
  max_selected_tails = 6585
```

## Interpretation

The normalized mapping did not degrade the main concentration metric. It improved:

```text
best_hpo_score: 58.30 -> 60.11
best_lift:      54.71 -> 56.82
best_utility_p90: 9.19 -> 9.37
max selected tails: 6395 -> 6585
```

Best profit proxy is essentially flat:

```text
2.752706 -> 2.746921
```

Runtime improved materially:

```text
2258s -> 829s
```

## Decision

Keep this change.

Reason:

```text
normalized source/bar features are now actually present and used;
runtime drops 63%;
lift/HPO improve slightly;
Optuna remains enabled.
```

Do not add more features now. The next ponytail step is daily projection from this bounded advanced profile.
