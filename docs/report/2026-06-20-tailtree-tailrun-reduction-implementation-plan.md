# Tailtree tailrun reduction implementation plan

## Context read

Read before this code pass:

- `docs/context.md`
- `docs/architecture/scanner.md`
- `docs/report/2026-06-20-tailtree-reduction-api-graph.md`
- `docs/report/2026-06-20-tailtree-workflow-reduction-redesign.md`
- `docs/report/2026-06-20-tailtree-training-workflow-redesign.md`
- `src/qooi/scanner/config.py`
- `src/qooi/scanner/tailrun/core.py`
- `src/qooi/scanner/tailrun/planning.py`
- `src/qooi/scanner/tailrun/selection.py`
- `src/qooi/scanner/tailrun/types.py`
- `src/qooi/scanner/tailtree/model.py`

## Current shape

`tailrun/core.py` still owns too much of the tailtree lifecycle:

```text
profile/fold/trial loop
+ horizon/direction model loop
+ event-lift training target adapter
+ score-bucket candidate frame
+ paired candidate replay
+ replay metric aggregation
+ selection-efficiency row construction
```

This violates the scanner architecture boundary:

```text
tailtree/model.py  -> labels, training frame, train/predict/persist
tailrun/planning.py -> profile/fold/job planning
tailrun/selection.py -> scored candidates, paired replay, selection metrics
tailrun/core.py -> lifecycle orchestration and model side effects
```

## Reduction goal

Implement phase 1 of `2026-06-20-tailtree-reduction-api-graph.md` with behavior kept comparable:

```text
hpo_score == base_hpo_score
paired diagnostic columns still emitted
no direction/threshold feature leakage
no new manager class
no compatibility wrappers
```

## API changes

### 1. Move event-lift target adapter to model layer

Add to `src/qooi/scanner/tailtree/model.py`:

```text
event_lift_training_values(observations, labeled_outcomes, direction)
```

Delete from `tailrun/core.py`:

```text
_event_lift_training_values
```

Reason: this is label-to-training-values logic, not orchestration.

### 2. Add flat objective job/result types

Add to `src/qooi/scanner/tailrun/types.py` only because they cross module boundaries:

```text
TailtreeObjectiveJob
TailtreeJobResult
```

Shape:

```text
job = run + fold_id + horizon + direction + model_path + label
result = job + evidence + scored_candidates + model + score
```

No opaque dicts, no manager class.

### 3. Plan jobs in planning.py

Add:

```text
tailtree_objective_jobs(run, fold_id, tailtree)
```

Order:

```text
horizon-major, direction-minor
```

This deletes inline model path/horizon/direction construction pressure from `core.py`.

### 4. Move candidate/replay/metric products to selection.py

Move from `core.py` to `selection.py`:

```text
_score_bucket_name
_score_bucket_value
_score_bucket_candidate_frame -> score_bucket_candidate_frame
_paired_candidate_replay_frame -> paired_candidate_replay_frame
_candidate_replay_metrics -> candidate_replay_metrics
_series_mean_float
_selection_efficiency_frame -> tailtree_selection_metrics_frame
```

Keep the current base score formula:

```text
base_hpo_score = valid_tail_lift + utility_mean + sqrt(selected_tail_count + 1) / 10
hpo_score = base_hpo_score
```

Keep `objective_hpo_score` diagnostic only.

### 5. Reduce core training loop

Replace `_train_profile_run(...)` with:

```text
run_tailtree_job(job, prepared, config, profile) -> TailtreeJobResult
_train_profile_run(...) -> local concat of job results only, or delete if call sites become direct
```

Ponytail choice for this pass:

- make `_train_profile_run` a small run/fold aggregator if it materially reduces call-site churn;
- do not keep it as a behavior owner;
- if it remains, it must not build replay or selection rows.

## Expected product flow after edit

```text
run_tailtree
  -> prepare outcomes/labels
  -> profile/fold/trial loop
  -> _train_profile_run/run_tailtree_job
       -> TailtreeJobResult(evidence, scored_candidates, model, score)
  -> paired_candidate_replay_frame(all scored candidates)
  -> tailtree_selection_metrics_frame(evidence, replay, context-ish run identity values)
  -> TailtreeRunOutput
```

## Verification

Required grep gates:

```bash
grep -R "def _score_bucket_candidate_frame\|def _paired_candidate_replay_frame\|def _candidate_replay_metrics\|def _selection_efficiency_frame\|def _event_lift_training_values" -n src/qooi/scanner/tailrun/core.py
grep -R "candidate_direction_code\|return_threshold_pct" -n src/qooi/scanner/tailrun src/qooi/scanner/tailtree
```

Expected:

- first grep empty;
- second grep empty except legitimate config/outcome argument names outside feature selection if any.

Run checks:

```bash
uv run --no-sync python -m ruff check src tests scripts/scanner_backfill.py scripts/scanner_potential.py
uv run --no-sync python -m ty check
uv run --no-sync python -m pytest tests/test_scanner_workflow_migration.py tests/test_state.py -q
```

Smoke if dependencies/data allow:

```bash
uv run --no-sync python -u scripts/scanner_potential.py --config configs/potential-paired-replay-test-tailtree.toml
```

## Non-goals

Do not change HPO behavior.
Do not add advanced runs.
Do not add model features.
Do not redesign promotion gates.
Do not introduce strategy/manager classes.
