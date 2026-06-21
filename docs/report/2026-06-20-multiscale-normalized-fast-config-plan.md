# Multi-time-scale feature naming + faster Optuna plan

## User direction

```text
1. Do not prefer long-range bar calculations.
2. Prefer multi-time-scale because new altcoin life is shallow.
3. Normalize feature names consistently.
4. Make config faster for the next multi-phase loop.
5. Keep Optuna; minor quality loss is acceptable.
```

## Current state

The failed broad primitive pass has already been reverted from active scanner code.

Active feature surface still includes the proven normalized-bounded baseline:

```text
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
*_age_ms
```

Problems:

```text
- names mix bar-count, source names, and normalization suffixes inconsistently
- no explicit distinction between raw source features and normalized source features
- advanced config is still ~12 min on warm cache for 24 models
- current feature code still talks in 168-bar terms instead of multi-time-scale wording
```

## Design boundary

This pass does **not** re-add broad primitive features.

It keeps the proven values but renames them into a consistent input grammar:

```text
<scope>_<measure>_<window>_<unit_or_transform>
```

Use hours/days, not raw bar counts, because scanner uses multiple timeframes and user preference is multi-time-scale.

## Naming migration

### Bar features

Rename active bar features:

```text
return_1bar                  -> bar_return_1h_pct
return_4bar                  -> bar_return_4h_pct
return_24bar                 -> bar_return_24h_pct
return_4bar_vol_scaled       -> bar_return_4h_per_vol_7d
return_24bar_vol_scaled      -> bar_return_24h_per_vol_7d
vol_anomaly                  -> bar_volume_1h_to_ma_20h
close_to_range_high_ratio    -> bar_close_position_48h
```

The 7d volatility denominator remains because it was part of the proven normalized-bounded baseline. It is not extended to 30d/720 bars. Later if we want zero long-ish windows, test replacing it with 24h/48h denominators in a separate benchmark.

### Source features

Rename active source features:

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

Keep age columns as-is because they already carry source identity and unit:

```text
funding_age_ms
oi_age_ms
taker_age_ms
lsr_age_ms
```

## Long-range rule

Do not add:

```text
range_position_720
range_compression_48_720
return_48bar
symbol tail-rate target encoding
30d range geometry
```

Reason:

```text
- new altcoins have shallow life
- 720-bar geometry mixed current setup with implicit listing-age/history-depth
- the broad primitive pass worsened top-bucket selection quality
```

## Multi-time-scale rule

Use `potential.bars.timeframes` for scale separation:

```text
["1H", "4H", "1D"]
```

Current tailtree features still use the decision timeframe numeric columns plus categorical state from 4H/1D context. This is acceptable for this pass; do not add more numeric long-window bar features.

## Faster config

Bottleneck is model training, not feature calculation:

```text
evidence_tailtree dominates runtime
continuous_features is only a few seconds
```

Current config trains:

```text
3 trials × 2 folds × 2 horizons × 2 directions = 24 models
```

Next multi-phase loop should favor faster iteration:

```text
2 trials × 2 folds × 2 horizons × 2 directions = 16 models
```

Keep Optuna enabled:

```toml
kind = "optuna"
max_trials = 2
```

Also reduce search ranges modestly:

```toml
num_leaves_range = [16, 96]
min_data_in_leaf_range = [120, 500]
learning_rate_range = [0.02, 0.08]
num_iterations_range = [180, 320]
early_stopping_rounds_range = [15, 35]
```

Expected runtime reduction:

```text
~33% fewer model trainings
shorter per-model iteration cap
minor quality loss acceptable for exploration
```

## Verification

Run with `uv`:

```bash
uv run python -m ruff format src/qooi/scanner/state.py src/qooi/scanner/tailrun/core.py
uv run python -m ruff check src tests scripts/scanner_backfill.py scripts/scanner_potential.py
uv run python -m ty check
uv run python -m pytest tests/test_state.py -q
```

Do not run full scanner benchmark unless requested after this config/naming pass.

## Acceptance

This pass is complete if:

```text
- no 720/30d primitive feature names remain in src/qooi/scanner
- tailtree allowlist uses normalized names
- state.py emits those normalized names
- advanced config keeps Optuna but has max_trials = 2
- uv checks pass
```
