# Tailtree reduction API graph

## Purpose

Apply ponytail to the current tailtree workflow before adding more objective logic.

The current paired replay diagnostic is useful, but the implementation shape is wrong:

```text
core.py now owns training lifecycle + score frames + paired replay + selection metrics
```

This report turns the reduction target into a concrete API graph. It is a design contract for the next coding pass, not a claim that all APIs already exist.

## Ponytail rule for this refactor

Reduce by ownership, not by moving lines randomly.

```text
one module owns one product role
one function returns one product shape
no thin wrappers
no compatibility shims
no new manager/orchestrator class
no direction/threshold feature leakage
```

Keep the useful new semantic product:

```text
candidate-level paired replay
```

Delete the bad shape:

```text
paired replay helpers embedded inside training core
selection metrics built from ad-hoc row.get(...) in core.py
_train_profile_run returning candidate_replay as a side product
```

## Current mixed-code audit

### `tailrun/core.py`

Current responsibilities found in `core.py`:

```text
✅ lifecycle orchestration
✅ train/load model lifecycle
✅ profile/fold/trial loop
✅ outcome and label construction
✅ model artifact path selection
✅ evidence collection
⚠️ objective-specific training values
❌ score-bucket candidate frame construction
❌ paired candidate replay construction
❌ replay metric aggregation
❌ selection-efficiency row construction
❌ objective HPO formula construction
```

Keep in `core.py`:

```text
run_tailtree(...)
_profile_prepared_frames(...)
_train/profile execution orchestration
model lifecycle calls
artifact/profile side effects
```

Move out of `core.py`:

```text
_score_bucket_candidate_frame
_paired_candidate_replay_frame
_candidate_replay_metrics
_selection_efficiency_frame
```

### `tailtree/model.py`

Current owner:

```text
labels -> training frame -> model train/predict/serialize
```

Belongs here:

```text
label_tail_exceedances(...)
tailtree_training_frame(...)
TailTreeModel.train(...)
TailTreeModel.predict_score(...)
TailTreeModel.predict_leaf(...)
```

Do not move paired replay here. Replay is not model training.

### `tailtree/evidence.py`

Current owner:

```text
model predictions -> evidence buckets
```

Belongs here:

```text
leaf_evidence_frame(...)
score_bucket_evidence_frame(...)
```

Possible small extension:

```text
score_bucket_candidate_frame(...)
```

But the better owner is `tailrun/selection.py` because candidate score buckets are consumed by selection replay, not by downstream rank evidence directly.

### `tailrun/selection.py`

Already owns:

```text
TailtreeSelectionBudgets
TailtreeSelectionFeasibilityPolicy
TailtreeSelectionPolicy
TailtreeSelectionContext
TailtreeCandidateReplay
tailtree_selection_efficiency_frame(...)
select_tailtree_budget_winners(...)
select_tailtree_objective_winners(...)
tailtree_hpo_feedback_frame(...)
write_tailtree_selection_efficiency(...)
```

This is the correct home for paired replay and objective scoring.

Move here:

```text
score_bucket_candidate_frame(...)
paired_candidate_replay_frame(...)
candidate_replay_metrics(...)
directional_objective_score(...)
selection_efficiency_from_evidence(...)
```

### `tailrun/planning.py`

Already owns:

```text
profile run records
trial params
fold specs
execution contexts
selection context construction
```

This should own the new flat job product:

```text
TailtreeObjectiveJob
objective_jobs(...)
```

Do not create a new `jobs.py` yet. `planning.py` is already the planning owner.

### `tailrun/search.py`

Owns:

```text
Optuna dependency loading
trial parameter suggestion
trial feedback aggregation
trial objective score
```

Should remain HPO/search only.

Do not put replay construction here. It may consume `selection_efficiency`, but not build scored candidates.

### `tailrun/types.py`

Owns shared run types and structural protocols.

Use it only for types that cross modules. Do not dump every helper dataclass here.

Add only if the type crosses module boundaries:

```text
TailtreeObjectiveJob
TailtreeJobResult
TailtreeScoredCandidates?  # probably avoid; DataFrame contract is enough
```

If a type is only used inside `selection.py`, keep it inside `selection.py`.

## Current duplication to delete

There are now two selection-efficiency paths:

```text
1. tailrun/selection.py::tailtree_selection_efficiency_frame(...)
2. tailrun/core.py::_selection_efficiency_frame(...)
```

This is the main reduction target.

Ponytail decision:

```text
delete core.py::_selection_efficiency_frame
use tailrun/selection.py as the single selection-efficiency owner
```

But selection.py currently expects candidate-style rows, while core.py currently has evidence rows. So the reduced API graph needs one explicit conversion:

```text
evidence + scored candidates + paired replay -> selection-efficiency rows
```

Do not keep two row builders.

## Target module graph

```text
qooi.scanner.workflow
  -> tailrun.core.run_tailtree(frames, config, profile)

qooi.scanner.tailrun.core
  -> outcome.potential_outcome_frame(...)
  -> tailtree.model.label_tail_exceedances(...)
  -> planning.tailtree_objective_jobs(...)
  -> core.run_tailtree_job(job, prepared, profile)
  -> selection.paired_candidate_replay_frame(scored_candidates)
  -> selection.tailtree_selection_metrics_frame(evidence, replay, context, policy)
  -> TailtreeRunOutput(evidence, models, profile_runs, selection_efficiency)

qooi.scanner.tailrun.planning
  -> tailtree_profile_runs(config)
  -> tailtree_optuna_profiles(config)
  -> tailtree_fold_specs(...)
  -> tailtree_execution_contexts(config, observations)
  -> tailtree_objective_jobs(run, fold, tailtree)

qooi.scanner.tailtree.model
  -> tailtree_training_frame(...)
  -> label_tail_exceedances(...)
  -> TailTreeModel.train(...)
  -> TailTreeModel.from_json(...)

qooi.scanner.tailtree.evidence
  -> leaf_evidence_frame(...)
  -> score_bucket_evidence_frame(...)

qooi.scanner.tailrun.selection
  -> score_bucket_candidate_frame(...)
  -> paired_candidate_replay_frame(...)
  -> tailtree_selection_metrics_frame(...)
  -> directional_objective_score(...)
  -> select_tailtree_budget_winners(...)
  -> tailtree_hpo_feedback_frame(...)
```

## Product graph

```text
TailtreeInputFrames
  -> PotentialOutcomeFrame
  -> LabeledOutcomeFrame
  -> TailtreePreparedFrames
  -> TailtreeObjectiveJob[]
  -> TailtreeJobResult[]
  -> EvidenceFrame
  -> ScoredCandidateFrame
  -> PairedReplayFrame
  -> SelectionMetricFrame
  -> TailtreeRunOutput
```

## Concrete API contracts

### 1. `TailtreeObjectiveJob`

Owner:

```text
tailrun/planning.py or tailrun/types.py
```

Use `planning.py` if only the planner/core consumes it. Move to `types.py` only if `selection.py` or `search.py` needs it.

Shape:

```python
@dataclass(frozen=True)
class TailtreeObjectiveJob:
    run: TailtreeProfileRun
    fold_id: int
    outcome_horizon: int
    direction: TailtreeDirection
    model_path: Path
    label: str
```

No config object inside the job except the already narrow `run`.

Why:

```text
removes nested horizon/direction/model_path construction from _train_profile_run
```

### 2. `tailtree_objective_jobs(...)`

Owner:

```text
tailrun/planning.py
```

Signature:

```python
def tailtree_objective_jobs(
    run: TailtreeProfileRun,
    *,
    fold_id: int,
    tailtree: TailtreeConfig,
) -> tuple[TailtreeObjectiveJob, ...]:
    ...
```

Output order:

```text
horizon-major, direction-minor
```

Concrete order:

```text
h24 up
h24 down
h48 up
h48 down
```

### 3. `TailtreeJobResult`

Owner:

```text
tailrun/types.py
```

Reason: this crosses `core.py` and `selection.py` by carrying evidence/scores/model.

Shape:

```python
@dataclass(frozen=True)
class TailtreeJobResult:
    job: TailtreeObjectiveJob
    evidence: pl.DataFrame
    scored_candidates: pl.DataFrame
    model: TailtreeArtifactTree | None
    score: float
```

But if importing `TailtreeObjectiveJob` from planning into types creates dependency inversion, keep both in `types.py`.

Preferred no-cycle design:

```text
types.py owns TailtreeObjectiveJob and TailtreeJobResult
planning.py constructs TailtreeObjectiveJob
core.py consumes both
selection.py only consumes DataFrames, not the job type
```

### 4. `run_tailtree_job(...)`

Owner:

```text
tailrun/core.py
```

Signature:

```python
def run_tailtree_job(
    job: TailtreeObjectiveJob,
    prepared: TailtreePreparedFrames,
    *,
    config: PotentialConfig,
    profile: ProfileContext,
) -> TailtreeJobResult:
    ...
```

Responsibilities:

```text
filter horizon labels
train/load model
score observations
build one evidence frame
return one scored-candidate frame
```

Forbidden inside this function:

```text
paired replay
selection metric rows
HPO objective formula
Optuna feedback
```

### 5. `event_lift_training_values(...)`

Owner:

```text
tailtree/model.py
```

Current `_event_lift_training_values(...)` is in `core.py`. That is wrong: it is label-to-training-frame logic.

Signature:

```python
def event_lift_training_values(
    observations: pl.DataFrame,
    labeled_outcomes: pl.DataFrame,
    *,
    direction: TailtreeDirection,
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    ...
```

Alternative better reduction:

```text
fold event-lift handling into tailtree_training_frame(..., objective="tail_event_lift")
```

But avoid over-expanding `tailtree_training_frame` unless it removes more code than it adds.

Ponytail decision:

```text
move existing helper to model.py first; do not redesign training-frame API yet
```

### 6. `score_bucket_candidate_frame(...)`

Owner:

```text
tailrun/selection.py
```

Signature:

```python
def score_bucket_candidate_frame(
    tree: TailtreeArtifactTree,
    observations: pl.DataFrame,
    labeled_outcomes: pl.DataFrame,
    *,
    outcome_horizon: int,
    score_quantiles: tuple[float, ...] = (0.99, 0.98, 0.95, 0.90),
) -> pl.DataFrame:
    ...
```

Output grain:

```text
symbol × decision_bar_close_ms × outcome_horizon × direction × score_bucket
```

Columns:

```text
symbol
decision_bar_close_ms
outcome_horizon
direction
tailtree_score
selected_tail
selected_utility
score_bucket
budget_value
```

This replaces `core.py::_score_bucket_candidate_frame`.

### 7. `paired_candidate_replay_frame(...)`

Owner:

```text
tailrun/selection.py
```

Signature:

```python
def paired_candidate_replay_frame(scored_candidates: pl.DataFrame) -> pl.DataFrame:
    ...
```

Input grain:

```text
symbol × decision_bar_close_ms × outcome_horizon × direction × score_bucket
```

Output grain:

```text
symbol × decision_bar_close_ms × outcome_horizon × selected_direction × score_bucket
```

Columns:

```text
symbol
decision_bar_close_ms
outcome_horizon
score_bucket
selected_direction
selected_score
opposite_score
selected_tail
opposite_tail
selected_utility
opposite_utility
directional_score_margin
gray_zone_int
false_direction_int
```

This replaces `core.py::_paired_candidate_replay_frame`.

### 8. `candidate_replay_metrics(...)`

Owner:

```text
tailrun/selection.py
```

Signature:

```python
def candidate_replay_metrics(
    replay: pl.DataFrame,
    *,
    outcome_horizon: int,
    direction: TailtreeDirection,
    score_bucket: str,
) -> dict[str, float]:
    ...
```

Columns returned:

```text
candidate_pair_count
paired_opposite_rate
paired_gray_zone_rate
paired_false_direction_rate
paired_false_direction_cost_mean
paired_directional_margin_mean
```

### 9. `base_hpo_score(...)`

Owner:

```text
tailrun/selection.py
```

Signature:

```python
def base_hpo_score(
    *,
    valid_tail_lift: float,
    utility_mean: float,
    selected_tail_count: int,
) -> float:
    ...
```

Formula for current compatibility:

```text
valid_tail_lift + utility_mean + sqrt(selected_tail_count + 1) / 10
```

This preserves current behavior.

### 10. `directional_objective_score(...)`

Owner:

```text
tailrun/selection.py
```

Diagnostic first. Do not feed Optuna until explicitly switched.

Signature:

```python
def directional_objective_score(
    base_score: float,
    metrics: Mapping[str, float],
    *,
    false_rate_weight: float = 10.0,
    false_cost_weight: float = 1.0,
    margin_weight: float = 5.0,
    gray_weight: float = 0.0,
) -> float:
    ...
```

Theory from current data:

```text
false_direction_rate and margin discriminate better than gray_zone_rate
```

So first formula should be:

```text
base
+ margin_weight * paired_directional_margin_mean
- false_rate_weight * paired_false_direction_rate
- false_cost_weight * paired_false_direction_cost_mean
- gray_weight * paired_gray_zone_rate
```

Default `gray_weight=0.0` initially because gray-zone rates were nearly identical by direction.

### 11. `tailtree_selection_metrics_frame(...)`

Owner:

```text
tailrun/selection.py
```

This is the replacement for `core.py::_selection_efficiency_frame`.

Signature:

```python
def tailtree_selection_metrics_frame(
    evidence: pl.DataFrame,
    replay: pl.DataFrame,
    *,
    context: TailtreeSelectionContext,
    observation_row_count: int,
    eligible_symbol_count: int,
    feature_count: int,
    trained_tree_count: int,
    fit_seconds: float,
) -> pl.DataFrame:
    ...
```

Output:

```text
selection-efficiency frame with existing columns plus paired diagnostics
```

Compatibility rule:

```text
hpo_score = base_hpo_score
objective_hpo_score = directional_objective_score(...)
```

Until user explicitly asks to switch HPO:

```text
hpo_score must remain comparable
```

## Reduced call graph after implementation

```text
run_tailtree(frames, config, profile)
  outcomes = potential_outcome_frame(...)
  labeled = label_tail_exceedances(...)
  prepared = TailtreePreparedFrames(...)

  for context in tailtree_execution_contexts(...):
    jobs = tailtree_objective_jobs(context.run, fold_id=context.fold.fold_id, tailtree=config.evidence.tailtree)
    job_results = [run_tailtree_job(job, prepared_for_fold, config, profile) for job in jobs]

    evidence = concat(result.evidence for result in job_results)
    scores = concat(result.scored_candidates for result in job_results)
    replay = paired_candidate_replay_frame(scores)
    selection = tailtree_selection_metrics_frame(evidence, replay, context=context.selection_context(), ...)
    feedback = _profile_feedback(context.run, score_from_evidence(evidence), evidence, models, seconds)
```

## DataFrame contracts

### `ScoredCandidateFrame`

Grain:

```text
symbol × decision_bar_close_ms × outcome_horizon × direction × score_bucket
```

Required columns:

```text
symbol: str
decision_bar_close_ms: int
outcome_horizon: int
direction: "up" | "down"
tailtree_score: float
selected_tail: bool
selected_utility: float
score_bucket: str
budget_value: float
```

### `PairedReplayFrame`

Grain:

```text
symbol × decision_bar_close_ms × outcome_horizon × selected_direction × score_bucket
```

Required columns:

```text
symbol
decision_bar_close_ms
outcome_horizon
score_bucket
selected_direction
selected_score
opposite_score
selected_tail
opposite_tail
selected_utility
opposite_utility
directional_score_margin
gray_zone_int
false_direction_int
```

### `SelectionMetricFrame`

Existing selection-efficiency columns plus:

```text
base_hpo_score
objective_hpo_score
candidate_pair_count
paired_opposite_rate
paired_gray_zone_rate
paired_false_direction_rate
paired_false_direction_cost_mean
paired_directional_margin_mean
```

Compatibility invariant:

```text
hpo_score == base_hpo_score
```

## Deletion inventory for next coding pass

Delete from `core.py` after moving:

```text
_score_bucket_name
_score_bucket_value
_score_bucket_candidate_frame
_paired_candidate_replay_frame
_candidate_replay_metrics
_series_mean_float
_selection_efficiency_frame
```

Move or delete:

```text
_event_lift_training_values -> tailtree/model.py
```

Remove from `_train_profile_run` return value:

```text
candidate_replay
```

Replace with:

```text
TailtreeJobResult.scored_candidates
```

## Avoided abstractions

Do not add:

```text
TailtreeWorkflowManager
TailtreeReplayEngine
ObjectiveStrategy class hierarchy
contracts.py grab bag
common.py utilities
```

Use plain functions and two dataclasses only if they reduce multi-return values:

```text
TailtreeObjectiveJob
TailtreeJobResult
```

## Implementation phases

### Phase 1 — move without behavior change

Goal:

```text
same outputs, reduced ownership mix
```

Steps:

```text
1. Add TailtreeObjectiveJob and TailtreeJobResult.
2. Add planning.tailtree_objective_jobs(...).
3. Move score/replay/metric helpers to selection.py.
4. Move event_lift_training_values to model.py.
5. Replace _train_profile_run with run_tailtree_job + local concat.
6. Delete core.py helper duplicates.
```

Verification:

```bash
uv run --no-sync python -m ruff check src tests scripts/scanner_backfill.py scripts/scanner_potential.py
uv run --no-sync python -m ty check
uv run --no-sync python -m pytest tests/test_scanner_workflow_migration.py tests/test_state.py -q
uv run --no-sync python -u scripts/scanner_potential.py --config configs/potential-paired-replay-test-tailtree.toml
```

Acceptance:

```text
hpo_score == base_hpo_score
candidate_replay frame exists
paired diagnostic columns exist
no candidate_direction_code
no return_threshold_pct feature
```

### Phase 2 — evaluate objective formula only after reduction

Goal:

```text
no new workflow shape, only one scoring function changes
```

Change only:

```text
directional_objective_score(...)
```

Do not change:

```text
model training
feature list
outcome labels
advanced config
```

## Why this graph answers the user’s concern

The concrete API graph prevents the patch spiral:

```text
new feature = new scoring function / product stage
not more logic inside train loop
```

It also preserves the theoretical sequence:

```text
direction -> threshold event concentration -> utility/replay dominance
```

And it keeps current empirical insight:

```text
down = high lift, high false-direction exposure, negative margin
up = lower lift, cleaner direction, positive margin
```

as an objective-scoring concern, not a model-feature concern.
