# Tailtree explicit-label normalized implementation plan

## Status

Canonical implementation plan. This replaces the looser 2026-06-21 theory/direction notes for coding decisions.

Reason for normalization:

```text
Do not make one model learn everything at once.
Make labels explicit first. Keep current tail_event_lift baseline until diagnostics prove the next objective.
```

## One vocabulary

Use these names everywhere. Do not introduce synonyms.

| Concept | Canonical name |
|---|---|
| Marginal up threshold event | `tail_up` |
| Marginal down threshold event | `tail_down` |
| Either side threshold event | `tail_any` |
| Both sides cross threshold in same horizon | `tail_both` |
| Orthogonal four-state label | `tail_state` |
| Up utility column | `tail_utility_up` |
| Down utility column | `tail_utility_down` |
| Up minus down utility margin | `tail_utility_margin_up` |
| Down minus up utility margin | `tail_utility_margin_down` |
| Candidate selected side is side-only | `selected_side_only` |
| Candidate opposite side is side-only | `opposite_side_only` |
| Candidate has both-tail path | `selected_tail_both` |
| Replay row is side-only | `side_only_int` |
| Replay row is both-tail | `tail_both_int` |
| Current objective diagnostic | `objective_hpo_score` |
| New label-aware objective diagnostic | `side_hpo_score` |
| Calibrated side margin | `calibrated_side_margin` |

Review prose may call this gray-zone behavior. Data columns use `tail_both`.

## Module boundaries

### `src/qooi/scanner/state.py`

Owns known-at-close feature construction only.

Allowed future additions:

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

Forbidden:

```text
future returns
future path columns
threshold/direction config as input features
```

### `src/qooi/scanner/outcome.py`

Owns future/path outcome rows only. Do not import tailtree/model or selection here.

### `src/qooi/scanner/tailtree/model.py`

Owns:

```text
label_tail_exceedances(...)
tailtree_label_distribution_frame(...)
tailtree_training_frame(...)
event_lift_training_values(...)
TailTreeModel
```

Do not create a new `labels.py` yet. One label function plus one distribution product is enough.

### `src/qooi/scanner/tailtree/evidence.py`

Owns prediction/evidence buckets only:

```text
leaf_evidence_frame(...)
score_bucket_evidence_frame(...)
```

Do not move paired replay here.

### `src/qooi/scanner/tailrun/core.py`

Owns lifecycle:

```text
run_tailtree(...)
_profile_prepared_frames(...)
run_tailtree_job(...)
run_tailtree_fold(...)
```

Allowed side effects:

```text
profile.frame(...)
model JSON write/load
final tailtree run output
```

Forbidden:

```text
label formulas
replay metric formulas
feature engineering formulas
```

### `src/qooi/scanner/tailrun/selection.py`

Owns candidate selection products:

```text
score_bucket_candidate_frame(...)
paired_candidate_replay_frame(...)
candidate_replay_metrics(...)
tailtree_selection_metrics_frame(...)
directional_objective_score(...)
```

Future addition here only if needed:

```text
calibrated_candidate_replay_frame(...)
```

### `src/qooi/scanner/tailrun/types.py`

Owns cross-module dataclasses only. Do not put one-function local products here.

## Phase 1 — explicit labels only

No training behavior change.

### API: `label_tail_exceedances(...)`

File:

```text
src/qooi/scanner/tailtree/model.py
```

Signature remains:

```python
def label_tail_exceedances(
    outcome_frame: pl.DataFrame,
    *,
    threshold_pct: float,
) -> pl.DataFrame:
    ...
```

Required output columns:

```text
tail_up                         existing bool
tail_down                       existing bool
tail_exceedance_value_up         existing float/null
tail_exceedance_value_down       existing float/null
tail_utility_up                  existing float
tail_utility_down                existing float

tail_any                         new bool: tail_up | tail_down
tail_both                        new bool: tail_up & tail_down
tail_state                       new str: none/up/down/both
tail_utility_margin_up           new float: tail_utility_up - tail_utility_down
tail_utility_margin_down         new float: tail_utility_down - tail_utility_up
```

`tail_up` and `tail_down` are marginal event flags, not orthogonal labels.
`tail_state` is the only orthogonal label. Side-only rows are derived as:

```text
tail_state == "up"
tail_state == "down"
```

Do not add separate side-only label columns; they repeat `tail_state` and create naming drift.

Implementation rule:

```text
Append these columns inside `label_tail_exceedances` after existing tail_up/tail_down utility expressions.
```

Do not create a wrapper function like `label_joint_extreme_outcomes`.

### API: `tailtree_label_distribution_frame(...)`

File:

```text
src/qooi/scanner/tailtree/model.py
```

Add:

```python
def tailtree_label_distribution_frame(labeled_outcomes: pl.DataFrame) -> pl.DataFrame:
    ...
```

Output grain:

```text
outcome_horizon × tail_state
```

Output columns:

```text
outcome_horizon
tail_state
row_count
class_rate
tail_up_count
tail_down_count
tail_any_count
tail_both_count
tail_state_up_count
tail_state_down_count
tail_utility_up_mean
tail_utility_down_mean
tail_utility_margin_up_mean
```

Use `tail_*` names only.

### Lifecycle wiring

File:

```text
src/qooi/scanner/tailrun/core.py
```

After labeled outcomes are built:

```python
label_distribution = tailtree_label_distribution_frame(labeled)
profile.frame("scanner", "tailtree", "tailtree_label_distribution", label_distribution)
```

Artifact path should be:

```text
tailtree-label-distribution.csv
```

Do not read this CSV back internally.

### Tests

Create:

```text
tests/test_tailtree_explicit_labels.py
```

Cases:

```text
tail_state == up
tail_state == down
tail_state == both
none
utility margins
distribution class_rate sums to 1 per horizon
```

Commands:

```bash
uv run python -m pytest tests/test_tailtree_explicit_labels.py -q
uv run python -m ruff check src/qooi/scanner/tailtree/model.py tests/test_tailtree_explicit_labels.py
uv run python -m ty check src/qooi/scanner/tailtree/model.py tests/test_tailtree_explicit_labels.py
```

## Phase 2 — replay diagnostics use explicit labels

No new model objective yet.

### `score_bucket_candidate_frame(...)`

File:

```text
src/qooi/scanner/tailrun/selection.py
```

Add selected-side columns:

```text
selected_side_only
selected_tail_both
selected_tail_state
selected_utility_margin
```

Mapping:

```text
up tree:
  selected_side_only = tail_state == "up"
  selected_utility_margin = tail_utility_margin_up

down tree:
  selected_side_only = tail_state == "down"
  selected_utility_margin = tail_utility_margin_down

both directions:
  selected_tail_both = tail_both
  selected_tail_state = tail_state
```

Fallback for tests/old frames:

```text
selected_side_only = selected_tail
selected_tail_both = false
selected_tail_state = null
selected_utility_margin = selected_utility
```

### `paired_candidate_replay_frame(...)`

Add replay columns:

```text
selected_side_only
opposite_side_only
selected_tail_both
opposite_tail_both
selected_utility_margin
opposite_utility_margin
side_only_int
tail_both_int
```

Definitions:

```text
side_only_int = selected_side_only & !opposite_side_only & !selected_tail_both
tail_both_int = selected_tail_both | opposite_tail_both | (selected_tail & opposite_tail)
```

Keep existing columns:

```text
gray_zone_int
false_direction_int
directional_score_margin
```

Do not rename existing artifact columns yet; add new normalized columns beside them.

### `candidate_replay_metrics(...)`

Add metrics:

```text
paired_side_only_rate
paired_tail_both_rate
paired_selected_utility_margin_mean
```

### `tailtree_selection_metrics_frame(...)`

Add output columns:

```text
paired_side_only_rate
paired_tail_both_rate
paired_selected_utility_margin_mean
side_hpo_score
```

Keep existing columns:

```text
hpo_score
base_hpo_score
objective_hpo_score
paired_gray_zone_rate
```

Reason: `objective_hpo_score` is already used by current HPO. `side_hpo_score` is new diagnostics first.

## Phase 3 — bucket calibration

Only after Phase 2 smoke is readable.

### API: `calibrated_candidate_replay_frame(...)`

File:

```text
src/qooi/scanner/tailrun/selection.py
```

Signature:

```python
def calibrated_candidate_replay_frame(replay: pl.DataFrame) -> pl.DataFrame:
    ...
```

Adds:

```text
selected_bucket_tail_rate
opposite_bucket_tail_rate
selected_bucket_side_only_rate
opposite_bucket_side_only_rate
selected_bucket_tail_both_rate
calibrated_directional_margin
calibrated_side_margin
```

Group by:

```text
outcome_horizon
score_bucket
selected_direction
```

Then compute:

```text
calibrated_directional_margin = selected_bucket_tail_rate - opposite_bucket_tail_rate
calibrated_side_margin = selected_bucket_side_only_rate - opposite_bucket_side_only_rate
```

Use these in diagnostics first. Do not switch HPO in the same phase.

## Phase 4 — one feature pack benchmark

Only after labels/replay diagnostics are stable.

### Add known-at-close volatility pack

File:

```text
src/qooi/scanner/state.py
```

Add:

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

File:

```text
src/qooi/scanner/tailrun/core.py
```

Append the same names to `_TAILTREE_CONTINUOUS_TRAIN_FEATURES`.

Benchmark rule:

```text
If side_hpo_score / paired_side_only_rate / paired_tail_both_rate worsens materially, revert the feature pack in the same pass.
```

Do not add state-age/source-disagreement packs in the same pass.

## Phase 5 — optional narrow objectives

Only after Phases 1–4 produce stable artifacts.

### Config names

File:

```text
src/qooi/scanner/config.py
```

Extend objective literal only with narrow objectives:

```python
TailtreeObjective = Literal[
    "tail_severity_gpd",
    "tail_utility_quantile",
    "tail_event_lift",
    "tail_any_event",
    "tail_side_only",
]
```

Do not add all-in-one or alias objective names.

### Training-value APIs

File:

```text
src/qooi/scanner/tailtree/model.py
```

Add only when objective phase starts:

```python
def any_event_training_values(
    observations: pl.DataFrame,
    labeled_outcomes: pl.DataFrame,
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    ...
```

```python
def side_only_training_values(
    observations: pl.DataFrame,
    labeled_outcomes: pl.DataFrame,
    *,
    direction: Literal["up", "down"],
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    ...
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

### Training dispatch

File:

```text
src/qooi/scanner/tailrun/core.py
```

Use explicit branches:

```python
if run.objective == "tail_event_lift":
    train_features, train_values, train_utilities = event_lift_training_values(...)
elif run.objective == "tail_any_event":
    train_features, train_values, train_utilities = any_event_training_values(...)
elif run.objective == "tail_side_only":
    train_features, train_values, train_utilities = side_only_training_values(...)
else:
    training = tailtree_training_frame(...)
    train_features = training.tail_observations
    train_values = training.exceedance_values
    train_utilities = training.utility_values
```

No registry/factory. No multiclass until binaries are evaluated.

## Phase 6 — HPO switch

Only after `side_hpo_score` and calibration prove useful.

Switch fold score source from:

```text
objective_hpo_score
```

to:

```text
side_hpo_score or calibrated side score
```

If multiple folds:

```python
mean_score = sum(trial_scores) / len(trial_scores)
variance = sum((score - mean_score) ** 2 for score in trial_scores) / len(trial_scores)
trial_score = mean_score - fold_std_weight * variance ** 0.5
```

No numpy needed.

## Verification commands

Use these commands.

```bash
uv run python -m pytest tests/test_tailtree_explicit_labels.py -q
uv run python -m pytest tests/test_scanner_workflow_migration.py tests/test_state.py -q
uv run python -m ruff check src/qooi/scanner/tailtree src/qooi/scanner/tailrun tests/test_tailtree_explicit_labels.py tests/test_scanner_workflow_migration.py tests/test_state.py
uv run python -m ty check src/qooi/scanner/tailtree src/qooi/scanner/tailrun tests/test_tailtree_explicit_labels.py tests/test_scanner_workflow_migration.py tests/test_state.py
uv run python -m scripts.scanner_potential --config configs/potential-paired-replay-test-tailtree.toml
```

Smoke artifact checks:

```text
tailtree-label-distribution.csv exists
tailtree-selection-efficiency.csv has side_hpo_score
paired_side_only_rate present
paired_tail_both_rate present
uv commands use the plain `uv run python ...` form
```

## Anti-conflict greps

Run after each phase:

```bash
# Replace <old-alias-pattern> with any non-canonical label/objective name seen during review.
grep -R "<old-alias-pattern>" -n src tests || true
grep -R "ObjectiveManager\|ModelManager\|TrainingRegistry" -n src tests || true
grep -R "candidate_direction_code\|return_threshold_pct" -n src/qooi/scanner/tailrun src/qooi/scanner/tailtree || true
```

Expected:

```text
first grep empty
second grep empty
third grep empty, except config/outcome code outside tailrun/tailtree if searched globally
```

## Summary

Concrete, normalized implementation order:

```text
1. Add explicit `tail_*` label columns inside label_tail_exceedances.
2. Add `tailtree_label_distribution_frame` and artifact.
3. Add side-only/tail-both replay metrics and `side_hpo_score` diagnostics.
4. Add bucket calibration diagnostics.
5. Benchmark one volatility feature pack.
6. Only then add `tail_any_event` / `tail_side_only` objectives.
7. Only then switch HPO to side/calibrated score.
```

This is the ponytail path: one vocabulary, one owner per product, no repeated objective names, no manager classes, no all-in-one objective.
