# Tailtree remaining-phase improvement plan

## Scope

Continue from `2026-06-21-tailtree-explicit-label-api-implementation-plan.md` after phases 1-2 are implemented and smoke-tested.

This pass implements only the remaining minimal ponytail phases:

1. bucket calibration diagnostics,
2. one known-at-close volatility feature pack,
3. narrow objective APIs and explicit dispatch.

HPO score-source switch is not bundled unless diagnostics prove it after benchmark.

## Current baseline from last smoke

Artifact root:

```text
data/output/potential/paired-replay-test
```

Observed after explicit labels:

```text
up:   max objective_hpo_score 36.901774, max side_hpo_score 37.593384, false_direction_rate 0.083547, side_only_rate 0.270846
 down: max objective_hpo_score 50.104264, max side_hpo_score 50.198747, false_direction_rate 0.354610, side_only_rate 0.151170
```

Phenomenon to evaluate:

```text
Down still has high false-direction / weaker side-only separation.
Up is cleaner and benefits more from side-aware scoring.
```

## Phase A — bucket calibration diagnostics

File:

```text
src/qooi/scanner/tailrun/selection.py
```

Add exactly one function:

```python
def calibrated_candidate_replay_frame(replay: pl.DataFrame) -> pl.DataFrame:
    ...
```

Input is the existing paired replay frame. Output is the same frame plus:

```text
selected_bucket_tail_rate
opposite_bucket_tail_rate
selected_bucket_side_only_rate
opposite_bucket_side_only_rate
selected_bucket_tail_both_rate
calibrated_directional_margin
calibrated_side_margin
```

Grouping grain:

```text
outcome_horizon × score_bucket × selected_direction
```

Selection efficiency should read metrics from calibrated replay when available:

```text
paired_calibrated_directional_margin_mean
paired_calibrated_side_margin_mean
```

Do not replace current HPO in this phase.

## Phase B — known-at-close volatility feature pack

Status: attempted and reverted after benchmark. The pack improved down-side score/false-rate but worsened up-side score, side-only rate, and both-tail contamination, so it failed the reversible benchmark rule.

File:

```text
src/qooi/scanner/state.py
```

Add features from OHLCV only:

```text
realized_vol_6h_pct
realized_vol_24h_pct
realized_vol_ratio_6h_24h
abs_return_1h_pct
max_abs_return_24h_pct
range_pct_1h
range_expansion_1h_vs_24h
volume_z_24h
```

Rules:

```text
known at close only
no future/path columns
no threshold/direction config as features
no extra feature family in same pass
```

Then append the same names to `_TAILTREE_CONTINUOUS_TRAIN_FEATURES` in:

```text
src/qooi/scanner/tailrun/core.py
```

If the benchmark worsens materially, revert this feature pack in the same pass.

## Phase C — narrow objectives

Files:

```text
src/qooi/scanner/config.py
src/qooi/scanner/tailtree/model.py
src/qooi/scanner/tailrun/core.py
```

Extend objective literals with:

```text
tail_any_event
tail_side_only
```

Add only two training-value APIs:

```python
def any_event_training_values(...): ...
def side_only_training_values(..., direction: Literal["up", "down"]): ...
```

Semantics:

```text
tail_any_event:
  label = tail_any
  utility = max(tail_utility_up, tail_utility_down)

tail_side_only up:
  label = tail_state == "up"
  utility = max(tail_utility_margin_up, 0)

tail_side_only down:
  label = tail_state == "down"
  utility = max(tail_utility_margin_down, 0)
```

Dispatch remains explicit `if/elif`; no registry, manager, alias objective, or all-in-one objective.

## Tests first

Add tests to existing small files where possible:

```text
tests/test_tailtree_explicit_labels.py
```

New RED tests:

```text
calibrated_candidate_replay_frame adds bucket-level calibrated margins
any_event_training_values labels any tail with max side utility
side_only_training_values labels only orthogonal side state and uses positive utility margin
TrainConfig accepts tail_any_event and tail_side_only
```

State feature test can live in a focused scanner test if needed:

```text
continuous_features_frame emits known-at-close volatility columns
```

## Verification

Use:

```bash
uv run python -m pytest tests/test_tailtree_explicit_labels.py tests/test_scanner_workflow_migration.py tests/test_state.py -q
uv run python -m ruff check src/qooi/scanner/tailtree/model.py src/qooi/scanner/tailrun/selection.py src/qooi/scanner/tailrun/core.py src/qooi/scanner/tailrun/types.py src/qooi/scanner/tailrun/planning.py src/qooi/scanner/tailrun/artifacts.py src/qooi/scanner/workflow.py src/qooi/scanner/state.py src/qooi/scanner/config.py src/qooi/transport/core.py tests/test_tailtree_explicit_labels.py
uv run python -m ty check src/qooi/scanner/tailtree/model.py src/qooi/scanner/tailrun/selection.py src/qooi/scanner/tailrun/core.py src/qooi/scanner/tailrun/types.py src/qooi/scanner/tailrun/planning.py src/qooi/scanner/tailrun/artifacts.py src/qooi/scanner/workflow.py src/qooi/scanner/state.py src/qooi/scanner/config.py src/qooi/transport/core.py tests/test_tailtree_explicit_labels.py
```

Run benchmark smoke via process fork and poll/wait:

```bash
uv run python -m scripts.scanner_potential --config configs/potential-paired-replay-test-tailtree.toml
```

Then compare:

```text
objective_hpo_score
side_hpo_score
paired_side_only_rate
paired_tail_both_rate
paired_false_direction_rate
paired_calibrated_side_margin_mean
```

## Stop criteria

Report unfinished if any phase fails tests, typecheck, smoke, or materially worsens benchmark and needs revert.

Do not switch HPO to side/calibrated score unless the benchmark gives clear evidence in this pass.
