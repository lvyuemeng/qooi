# Tailtree tailrun dead-code / nesting / method-grouping proposition

## Purpose

Explain the current tailtree tailrun workflow after the reduction pass, then propose the next ponytail cleanup pass:

```text
delete dead paths
flatten duplicated run loops
group behavior onto existing concrete data structures only where it removes free-function plumbing
replace manual JSON dump/load with Pydantic where it already fits
```

No code changes are proposed here as already complete; this is the next implementation plan.

## Current workflow

Current scanner architecture remains:

```text
workflow.py
  -> run_tailtree(frames, config, profile)
  -> rank/output/report
```

Inside `tailrun/core.py::run_tailtree` the current path is:

```text
1. build outcome_frame from observations/source_outcomes/realized
2. label tail exceedances
3. choose known-at-close categorical/continuous features
4. build TailtreePreparedFrames
5. run fixed profiles
6. run optuna profiles
7. concat evidence/profile feedback/selection efficiency
8. return TailtreeRunOutput
```

Per profile/fold/trial:

```text
_train_profile_run(run, fold_prepared, fold_id)
  -> planning.tailtree_objective_jobs(run, fold_id, tailtree)
  -> run_tailtree_job(job, prepared)
       -> filter horizon labels
       -> load/train one direction/horizon model
       -> score observations into score buckets
       -> build evidence
       -> return TailtreeJobResult
  -> concat job evidence
  -> concat scored candidates
  -> return evidence/models/score/scored_candidates
```

Selection surface now lives outside training:

```text
selection.paired_candidate_replay_frame(scored_candidates)
selection.tailtree_selection_metrics_frame(evidence, replay, ...)
```

This is the right direction: model training no longer owns paired replay or selection metric rows.

## Remaining problems

### 1. `run_tailtree` still duplicates fixed and optuna fold execution

Two blocks now do nearly the same work:

```text
fixed profile loop:
  fold_run
  _train_profile_run
  paired replay
  feedback
  selection metrics
  append evidence/models

optuna trial loop:
  fold_run
  _train_profile_run
  paired replay
  feedback
  selection metrics
  append evidence/models
  study.tell
```

Only the outer source differs:

```text
fixed: one run per fixed profile
optuna: trial params + trial score + study feedback
```

Ponytail target:

```text
one helper for one run/fold execution product
```

Recommended product:

```text
TailtreeFoldRunResult(
  evidence,
  models,
  feedback,
  selection_efficiency,
  score,
)
```

Owner:

```text
tailrun/core.py
```

Function:

```text
run_tailtree_fold(run, fold_id, fold_prepared, config, profile) -> TailtreeFoldRunResult
```

Then fixed and optuna loops only decide which runs exist and how trial scores are reported.

### 2. `_train_profile_run` is now an aggregator, not a training owner

After reduction it only does:

```text
jobs = tailtree_objective_jobs(...)
results = [run_tailtree_job(...)]
concat evidence
concat scored candidates
models dict
```

Rename would be cosmetic; ponytail says avoid rename-only churn. Better next step:

```text
inline it into run_tailtree_fold(...)
delete _train_profile_run
```

That removes one misleading middle layer instead of renaming it.

### 3. Old artifact lifecycle path appears dead for current scanner workflow

Observed symbols:

```text
run_frame_split
load_predict
_build_tail_tree_evidence
ReportInputs
PotentialUniverse
PotentialArtifacts
BarFetchResult
SymbolStateBundle
ScanDecision
TailtreeResult
```

Current workflow uses:

```text
workflow.py -> run_tailtree -> TailtreeRunOutput
```

The old path still exists for older report/artifact APIs:

```text
run_frame_split(..., ReportInputs)
_build_tail_tree_evidence(...)
_load_tail_tree_evidence(...)
_write_tailtree_artifacts(...)
```

Ponytail deletion proposal:

```text
if grep confirms no external/tests usage, delete old run_frame_split/load_predict/_build_tail_tree_evidence path in one pass
```

Caution:

```text
artifacts.py still consumes ReportInputs for old artifact helpers
__init__.py exports these old symbols
```

So deletion must include:

```text
1. remove exports from tailrun/__init__.py
2. remove dead types from tailrun/types.py
3. remove dead imports from artifacts/core
4. grep exact symbols after deletion
```

This should happen only if the repository no longer needs the pre-`run_tailtree` public API.

### 4. Old selection-efficiency replay path is likely stale

`selection.py` still has two selection concepts:

```text
current path:
  score_bucket_candidate_frame
  paired_candidate_replay_frame
  tailtree_selection_metrics_frame

older budget replay path:
  TailtreeSelectionContext
  TailtreeSelectionPolicy
  TailtreeSummaryView
  TailtreeCandidateReplay
  tailtree_selection_efficiency_frame
  with_tailtree_selection_identity
```

The older path is self-contained and exported, but current `run_tailtree` does not call it.

There is also a duplicate writer:

```text
tailrun/artifacts.py::write_tailtree_selection_efficiency   # used by workflow.py
tailrun/selection.py::write_tailtree_selection_efficiency   # appears unused except export
```

Ponytail deletion proposal:

```text
keep artifacts.py writer because workflow.py calls it
delete selection.py writer if no external use
```

For the older replay path, choose one of two options:

```text
Option A, delete now:
  if no tests/importers depend on the public API

Option B, fold into current path:
  only if it replaces tailtree_selection_metrics_frame with less code
```

Ponytail preference: Option A unless a live caller exists.

### 5. HPO search module has an unused orchestration path

`search.py` has current live use:

```text
optuna_module
suggest_tailtree_trial_params
```

But `run_tailtree_hpo_study`, `TailtreeContextExecution`, `TailtreeHpoStudyResult`, and `trial_objective_score` appear internally referenced only by that unused orchestration function.

Current `run_tailtree` does Optuna directly, so this is probably stale design drift.

Ponytail proposal:

```text
delete run_tailtree_hpo_study + private result dataclasses if no external callers
keep optuna_module + suggest_tailtree_trial_params only
```

This removes the dependency from `search.py` back into `planning.TailtreeExecutionContext` and helps delete execution-context plumbing.

## Method grouping proposition

The user steer is correct: if a custom data structure exists, put tightly-bound behavior on it instead of passing it through free helpers. But do not create new structures only to hold methods.

### Good method moves

#### `TailtreeProfileRun`

Current repeated logic:

```text
_fold_run(run, fold_id)
trial_id = run.run_id.rsplit("-t", 1)[0] if run.run_source == "optuna" else run.run_id
TrainConfig(...) from run.training
```

Propose methods/properties:

```text
TailtreeProfileRun.for_fold(fold_id) -> TailtreeProfileRun
TailtreeProfileRun.trial_id -> str
TailtreeProfileRun.to_train_config() -> TrainConfig
```

Caution:

```text
to_train_config imports tailtree.model.TrainConfig into planning.py unless placed carefully
```

Ponytail choice:

```text
for_fold + trial_id are safe now
to_train_config only if it avoids repeated TrainConfig construction in multiple places
```

#### `TailtreeObjectiveJob`

Already owns:

```text
run
fold_id
outcome_horizon
direction
model_path
label
```

Potential method:

```text
job.with_evidence_identity(evidence: pl.DataFrame) -> pl.DataFrame
```

This would remove the identity `with_columns(...)` block from `run_tailtree_job`.

Ponytail caution:

```text
DataFrame mutation methods on dataclasses can hide Polars column contracts
```

Recommendation:

```text
skip unless identity attachment repeats in more than one place
```

#### `TailtreeJobResult`

Potential methods:

```text
has_evidence
has_scores
model_entry -> tuple[key, model] | None
```

Ponytail verdict:

```text
skip now; current list comprehensions are clearer and shorter
```

### Bad method moves

Do not move these into dataclasses:

```text
score_bucket_candidate_frame
paired_candidate_replay_frame
tailtree_selection_metrics_frame
```

Reason:

```text
they are DataFrame product transformations, not behavior of one row/config object
```

They belong as flat functions in `selection.py`.

## Pydantic dump/load proposition

`tailtree/model.py::TailTreeModel` manually serializes:

```text
json.dump({"lightgbm_model": self.booster, "metadata": self.metadata.model_dump(...)})
json.load(...)
TreeMetadata.model_validate(data["metadata"])
```

Since `TreeMetadata` is already Pydantic, replace the manual dict/json path with a Pydantic payload:

```text
class TailTreePayload(BaseModel):
    lightgbm_model: str
    metadata: TreeMetadata
```

Then:

```text
TailTreeModel.to_json(path):
  payload = TailTreePayload(lightgbm_model=self.booster, metadata=self.metadata)
  path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")

TailTreeModel.from_json(path):
  payload = TailTreePayload.model_validate_json(path.read_text(encoding="utf-8"))
  return TailTreeModel(booster=payload.lightgbm_model, metadata=payload.metadata)
```

This removes manual `json` import/use and lets Pydantic own validation/dumping.

Do not convert `TailTreeModel` itself to `BaseModel` unless needed. The LightGBM `_booster` reconstruction and prediction methods are normal behavior; a small payload model is enough.

## Proposed next implementation phases

### Phase 1 — dead-code grep and delete obvious stale exports

1. Confirm no callers outside exports for:
   - `selection.py::write_tailtree_selection_efficiency`
   - `run_tailtree_hpo_study`
   - `TailtreeContextExecution`
   - `TailtreeHpoStudyResult`
2. Delete if grep-empty outside defining file / `__init__.py`.
3. Remove corresponding `__init__.py` exports.
4. Run ruff/ty/tests.

### Phase 2 — flatten run/fold execution

1. Add `TailtreeFoldRunResult` only if it removes repeated tuple plumbing.
2. Add `run_tailtree_fold(...)` in `core.py`.
3. Move shared fixed/optuna fold body into it.
4. Delete `_train_profile_run` by inlining its aggregation inside `run_tailtree_fold`.
5. Keep fixed/optuna outer loops explicit.

### Phase 3 — method grouping on existing structures

1. Add `TailtreeProfileRun.for_fold(fold_id)`.
2. Add `TailtreeProfileRun.trial_id`.
3. Replace `_fold_run(...)` and repeated trial-id splits.
4. Do not move DataFrame product functions onto dataclasses.

### Phase 4 — Pydantic model artifact payload

1. Add `TailTreePayload(BaseModel)` in `tailtree/model.py`.
2. Replace manual `json.dump/json.load` in `TailTreeModel.to_json/from_json`.
3. Keep JSON artifact shape unchanged:
   - `lightgbm_model`
   - `metadata`
4. Run existing smoke to prove model persistence still works.

### Phase 5 — old artifact path deletion, only if accepted

This is the larger delete:

```text
run_frame_split/load_predict/_build_tail_tree_evidence/ReportInputs old path
```

Do it only after deciding the old public API is no longer needed. It will shrink `core.py`, `types.py`, `artifacts.py`, and `__init__.py`, but may affect external callers if any exist outside this repo.

## Acceptance checks

```bash
uv run --no-sync python -m ruff check src tests scripts/scanner_backfill.py scripts/scanner_potential.py
uv run --no-sync python -m ty check
uv run --no-sync python -m pytest tests/test_scanner_workflow_migration.py tests/test_state.py -q
uv run --no-sync python -u scripts/scanner_potential.py --config configs/potential-paired-replay-test-tailtree.toml
```

Grep gates:

```bash
grep -R "run_tailtree_hpo_study\|TailtreeContextExecution\|TailtreeHpoStudyResult" -n src tests
grep -R "tailtree_selection_efficiency_frame\|TailtreeCandidateReplay\|TailtreeSummaryView" -n src tests
grep -R "run_frame_split\|_build_tail_tree_evidence\|ReportInputs" -n src tests
```

Expected after deletion phases:

```text
only live current workflow symbols remain
core.py no longer owns old artifact path or duplicated fold body
selection.py owns only current replay/metric products
planning.py owns only profile/fold/job planning, not unused execution context plumbing
```

## Bottom line

Current workflow is directionally correct after the last pass, but the next ponytail pass should be deletion-first:

```text
1. delete stale selection/HPO orchestration APIs
2. flatten duplicated fixed/optuna fold body
3. group repeated run identity behavior onto TailtreeProfileRun
4. replace manual TailTreeModel JSON with a small Pydantic payload
5. only then consider deleting the old run_frame_split/ReportInputs artifact path
```
