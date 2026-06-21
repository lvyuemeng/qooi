# Primitive feature tailtree benchmark

## Run

Current code included the primitive bar feature pass from the theory/tidy plan:

```text
return_12bar
return_48bar
momentum_accel_4_24
realized_vol_ratio_24_168
volume_participation_24_168
range_position_720
range_compression_48_720
```

Benchmark command:

```bash
./.venv/Scripts/python.exe -u scripts/scanner_potential.py --config configs/potential-advanced-tailtree.toml
```

Runtime:

```text
PRIMITIVE_BOUNDED_ADVANCED_SECONDS=748
```

Artifacts saved:

```text
data/output/potential/benchmarks/primitive-bounded-tailtree-selection-efficiency.csv
data/output/potential/benchmarks/primitive-bounded-report.md
```

Comparison baseline:

```text
data/output/potential/benchmarks/normalized-bounded-tailtree-selection-efficiency.csv
```

## Model metadata check

Primitive features reached the model metadata:

```text
model: data/output/potential/advanced-tailtree/models/tailtree-event-lift-advanced-wf-optuna-t0002-f01_24_down.json
continuous_count: 28
missing_required: []
required_present:
  return_12bar
  return_48bar
  momentum_accel_4_24
  realized_vol_ratio_24_168
  volume_participation_24_168
  range_position_720
  range_compression_48_720
```

So the benchmark is a real test of the primitive feature pass, not a wiring miss.

## Aggregate comparison

Best-by-HPO row:

| metric | normalized bounded | primitive pass | delta |
|---|---:|---:|---:|
| feature_count | 26 | 33 | +7 |
| hpo_score | 60.109815 | 54.187079 | -9.85% |
| valid_tail_lift | 56.816060 | 51.614937 | -9.15% |
| selected_profit_proxy_mean | 1.044310 | 0.425051 | -59.30% |
| selected_profit_proxy_p90 | 3.113295 | 1.124663 | -63.88% |
| selected_tail_count | 505 | 460 | -8.91% |
| fit_seconds | 115.474751 | 87.682795 | -24.07% |

Max-across-selection-surface comparison:

| metric | normalized bounded | primitive pass | delta |
|---|---:|---:|---:|
| hpo_score | 60.109815 | 54.187079 | -9.85% |
| valid_tail_lift | 56.816060 | 51.614937 | -9.15% |
| selected_profit_proxy_mean | 2.746921 | 2.331626 | -15.12% |
| selected_profit_proxy_p90 | 9.367261 | 8.225128 | -12.19% |
| selected_tail_count | 6585 | 6715 | +1.97% |
| selected_tail_per_1k_obs | 808.131567 | 765.189584 | -5.31% |
| feature_count | 26 | 33 | +26.92% |
| fit_seconds | 148.321211 | 127.940230 | -13.74% |

## Horizon/direction comparison

Best HPO by horizon/direction:

| horizon | direction | normalized hpo | primitive hpo | verdict |
|---:|---|---:|---:|---|
| h24 | down | 60.109815 | 54.187079 | worse |
| h24 | up | 41.272706 | 41.775763 | tiny better |
| h48 | down | 49.270447 | 48.365476 | worse |
| h48 | up | 28.344385 | 26.874663 | worse |

Best utility/profit by horizon/direction:

| horizon | direction | normalized utility | primitive utility | verdict |
|---:|---|---:|---:|---|
| h24 | down | 1.194829 | 1.369627 | better |
| h24 | up | 2.368639 | 2.331626 | slightly worse |
| h48 | down | 1.360079 | 1.423924 | better |
| h48 | up | 2.746921 | 2.326164 | worse |

## Report surface

Current primitive-pass report promoted:

```text
BEAT-USDT-SWAP ↓ down h24
RE-USDT-SWAP   ↓ down h24
KAT-USDT-SWAP  ↓ down h24
```

Freshness:

```text
Prediction freshness: candidates=89 | stale=0 | age_range=0.6h..0.6h
```

Data/source caveats:

```text
Training bars coverage=16.7% | status=low coverage (17%)
open_interest stale age=3.4h
taker_volume stale age=23.4h
trades stale age=3.1h
```

## Feasibility verdict

Not feasible to keep this primitive bar pass as-is.

Reason:

```text
+7 features increased model surface by 26.92%
main HPO score fell 9.85%
main tail lift fell 9.15%
best utility surface fell 15.12%
best utility p90 fell 12.19%
```

The only strong positive is runtime:

```text
829s normalized-bounded prior run -> 748s primitive run
```

But runtime got faster because this run's fit timings were lower, not because the feature pass made the decision surface better. It is not enough to keep worse evidence quality.

## Decision

Ponytail decision: revert the primitive bar feature pass and keep the proven baseline:

```text
event-lift objective
h24/h48 bounded Optuna
normalized bar/source aliases
```

Keep benchmark artifacts for evidence. Do not delete them.

## Follow-up

Theory base is still useful, but the current implementation was too broad:

```text
range_position_720 and range_compression_48_720 have long warmups and may reduce effective row quality
return_12bar/48bar may duplicate existing return_4bar/24bar splits
volume/vol ratios may add noise without a source/objective-specific interaction
```

Next feature work, if any, should test one primitive family at a time:

```text
A. acceleration only
B. compression/location only
C. participation only
```

using the same selection-efficiency artifact and immediate revert on failure.
