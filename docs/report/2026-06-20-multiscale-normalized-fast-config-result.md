# Multi-time-scale naming + fast Optuna result

## Applied

Implemented the plan from:

```text
docs/report/2026-06-20-multiscale-normalized-fast-config-plan.md
```

## 1. Long-range bar calculation

The active scanner feature surface does not include the failed long-range primitive bar features:

```text
range_position_720
range_compression_48_720
return_12bar
return_48bar
momentum_accel_4_24
```

Confirmed by grep in `src/qooi/scanner`: zero matches.

Kept the proven baseline 7d volatility denominator:

```text
bar_return_4h_per_vol_7d
bar_return_24h_per_vol_7d
```

This stays because it was part of the better normalized-bounded benchmark. It should be tested separately if we want to replace 7d with a shorter multi-timeframe denominator.

## 2. Normalized feature names

Renamed active feature columns to consistent names.

### Bar features

```text
return_1bar                  -> bar_return_1h_pct
return_4bar                  -> bar_return_4h_pct
return_24bar                 -> bar_return_24h_pct
return_4bar_vol_scaled       -> bar_return_4h_per_vol_7d
return_24bar_vol_scaled      -> bar_return_24h_per_vol_7d
vol_anomaly                  -> bar_volume_1h_to_ma_20h
close_to_range_high_ratio    -> bar_close_position_48h
```

### Source features

```text
funding_rate                 -> funding_rate_raw
funding_rate_bps             -> funding_rate_bps
oi_delta                     -> oi_change_raw
oi_delta_pct                 -> oi_change_pct
taker_buy_sell_ratio         -> taker_buy_sell_ratio_raw
taker_buy_pressure           -> taker_buy_pressure
long_short_ratio             -> lsr_ratio_raw
long_short_log_ratio         -> lsr_log_ratio
```

Age columns remain unchanged:

```text
funding_age_ms
oi_age_ms
taker_age_ms
lsr_age_ms
```

## 3. Faster Optuna config

Changed advanced config from:

```text
3 trials × 2 folds × 2 horizons × 2 directions = 24 models
```

to:

```text
2 trials × 2 folds × 2 horizons × 2 directions = 16 models
```

Kept Optuna:

```toml
kind = "optuna"
max_trials = 2
```

Narrowed search ranges:

```toml
num_leaves_range = [16, 96]
min_data_in_leaf_range = [120, 500]
learning_rate_range = [0.02, 0.08]
num_iterations_range = [180, 320]
early_stopping_rounds_range = [15, 35]
```

Expected effect:

```text
~33% fewer model trainings
shorter per-model cap
minor quality loss acceptable for next exploration loop
```

## Verification

Used `uv`:

```bash
uv run python -m ruff format src/qooi/scanner/state.py src/qooi/scanner/tailrun/core.py configs/potential-advanced-tailtree.toml
uv run python -m ruff check src tests scripts/scanner_backfill.py scripts/scanner_potential.py
uv run python -m ty check
uv run python -m pytest tests/test_state.py -q
```

Result:

```text
ruff: pass
ty: pass
tests/test_state.py: 7 passed
```

Feature extraction sanity check:

```text
rows: 220
missing: []
old_present: []
```

## Next benchmark baseline

The next scanner benchmark should compare the fast/named config against:

```text
data/output/potential/benchmarks/normalized-bounded-tailtree-selection-efficiency.csv
```

Expect some quality movement because feature names changed and model count/search range changed. The values are equivalent for renamed features, but LightGBM model training artifacts will be newly trained under the faster Optuna budget.
