# Normalized feature mapping + bounded Optuna runtime plan

## Correction

The current event-lift benchmark improved the objective, but the feature surface is still mostly the old one.

Current tailtree continuous features from model metadata:

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

That means the objective is new, but several model inputs are still raw/scale-dependent:

```text
return_*bar              raw percent return, only partly normalized by symbol history
oi_delta                 raw open-interest dollar/contract delta
taker_buy_sell_ratio     unbounded ratio, unstable when sell volume is tiny
long_short_ratio         raw crowding ratio, unbounded skew
funding_rate             raw level, no local abnormality context
```

The user is right: normalized source/bar mapping was not applied in the last objective patch.

## Ponytail scope

Do not add another broad feature block.

Do not add sin/cos/Fourier time features.

Do not add new diagnostics.

Do one compact mapping pass:

```text
bar/source raw values -> normalized mechanism features
```

Then keep Optuna but reduce runtime at the config level.

## Feature design

### Bar normalized mapping

Keep current raw returns available, but add normalized forms that tell LightGBM whether a move is large relative to the symbol's own recent noise:

```text
return_4bar_vol_scaled  = return_4bar  / rolling_std(return_1bar, 168)
return_24bar_vol_scaled = return_24bar / rolling_std(return_1bar, 168)
```

This is not a big feature family. It is the normalized version of existing return features.

### Source normalized mapping

Replace scale-dependent source interpretation with bounded/relative forms:

```text
oi_delta_pct                 = (oi_t - oi_{t-1}) / oi_{t-1} * 100
taker_buy_pressure           = (buy - sell) / (buy + sell)
long_short_log_ratio         = log(long_short_ratio)
funding_rate_bps             = funding_rate * 10000
```

Keep age columns:

```text
funding_age_ms
oi_age_ms
taker_age_ms
lsr_age_ms
```

Age columns are not predictive market state; they tell the model/report how stale the source value was.

## Runtime design

Current advanced event-lift run:

```text
5 trials × 2 folds × 3 horizons × 2 directions = 60 models
runtime = 2258s
```

Profile shows the bottleneck is model training/evidence:

```text
evidence_tailtree = 2170s
continuous_features = 0.63s
observations = 0.60s
```

So optimizing feature calculation will not fix runtime.

Keep Optuna, but bound the search:

```text
max_trials: 5 -> 3
outcome_horizon: [24, 48, 72] -> [24, 48]
```

Why remove h72:

```text
h24 best lift/HPO
h48 best profit proxy/support
h72 weaker lift and HPO in the event-lift benchmark
```

New expected model count:

```text
3 trials × 2 folds × 2 horizons × 2 directions = 24 models
```

This keeps Optuna and walkforward while cutting expected runtime by about 60%.

## Benchmark comparison

Use the existing artifact surface only:

```text
tailtree-selection-efficiency.csv
report.md
profile/stages.csv
```

Compare against saved event-lift benchmark:

```text
data/output/potential/benchmarks/event-lift-tailtree-selection-efficiency.csv
```

Key metrics:

```text
runtime
feature_count
passing_rows
best_hpo_score
best_valid_tail_lift
best_profit_proxy_per_selected_obs
h24/h48 horizon rows
```

## Keep/revert rule

Keep if:

```text
runtime drops materially
best lift/HPO does not collapse
h24/h48 remain usable
```

Revert if:

```text
normalized mapping worsens lift/utility heavily without runtime benefit
```
