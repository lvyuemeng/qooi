# State/feature computation tidy result

## Applied boundary

The computation boundary is now explicit:

```text
classifier/outcome/transitions = symbolic categorical grammar
continuous_features            = numeric primitive grammar
tailtree                       = consumes both; does not invent features
```

## Code changed

```text
src/qooi/scanner/state.py
src/qooi/scanner/tailrun/core.py
```

## Docs written

```text
docs/report/2026-06-20-theory-based-feature-grammar-evaluation.md
docs/report/2026-06-20-state-feature-computation-tidy-plan.md
```

## Current status after benchmark

The primitive bar feature pass described below was benchmarked in:

```text
docs/report/2026-06-20-primitive-feature-tailtree-benchmark.md
```

Benchmark verdict:

```text
not feasible to keep as-is
```

The code was reverted back to the prior normalized-bounded feature surface:

```text
event-lift objective
h24/h48 bounded Optuna
normalized bar/source aliases
```

The benchmark artifacts were kept:

```text
data/output/potential/benchmarks/primitive-bounded-tailtree-selection-efficiency.csv
data/output/potential/benchmarks/primitive-bounded-report.md
```

## What was tested, then reverted

Added primitive bar features to `_kline_continuous_features()`:

```text
return_12bar
return_48bar
momentum_accel_4_24
realized_vol_ratio_24_168
volume_participation_24_168
range_position_720
range_compression_48_720
```

Added them to `_TAILTREE_CONTINUOUS_TRAIN_FEATURES`.

Existing symbolic classifier/transition semantics were intentionally not changed.

## Theory base

These features are not mechanism-story names. They are primitive numeric states:

```text
velocity:
  return_12bar
  return_48bar

acceleration:
  momentum_accel_4_24

dispersion:
  realized_vol_ratio_24_168

participation:
  volume_participation_24_168

location:
  range_position_720

compression:
  range_compression_48_720
```

They satisfy the scanner feature grammar:

```text
known-at-close
scale-normalized
multi-scale
primitive, not duplicate categorical state labels
```

## Overlap tidy

Classifier already owns symbolic states:

```text
market_stage
structure_trend_state
liquidity_event_type
state_key
context_event
```

Outcome path rows already own symbolic path projections:

```text
core_context
transition_kind
compression_state
expansion_state
transition_path
```

Continuous features now own numeric primitives:

```text
range_position_720
range_compression_48_720
realized_vol_ratio_24_168
volume_participation_24_168
momentum_accel_4_24
```

So the tidy is separation by role, not deletion of useful overlapping concepts:

```text
categorical bucket = classifier/report/transition grammar
numeric scalar     = tailtree primitive magnitude/context grammar
```

## Verification

Static checks:

```text
ruff format: reformatted state.py
ruff check: pass
ty check: pass
```

Tests:

```text
137 passed, 7 skipped, 16 deselected in 4.03s
```

Synthetic feature extraction sanity check:

```text
rows: 760
missing: []
nonnull:
  return_12bar: 748
  return_48bar: 712
  momentum_accel_4_24: 736
  realized_vol_ratio_24_168: 736
  volume_participation_24_168: 737
  range_position_720: 40
  range_compression_48_720: 40
```

The 720-window features are non-null only after the 30d warmup, which is expected.

## Deferred

No slow scanner benchmark was run in this pass. The user previously noted script tests are slow.

Next empirical check, when desired:

```text
run configs/potential-advanced-tailtree.toml
compare against data/output/potential/benchmarks/normalized-bounded-tailtree-selection-efficiency.csv
inspect model metadata for new primitive columns
```

## Not changed

No source rolling z-scores.
No lifecycle prior tail rates.
No sin/cos/Fourier time sampling.
No feature framework or registry.
No classifier state semantic rewrite.
