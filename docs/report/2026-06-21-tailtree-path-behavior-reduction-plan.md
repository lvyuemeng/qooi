# Tailtree Path-Behavior Reduction Plan

## Goal

Turn the path-behavior redesign into a ponytail/reduction-oriented implementation plan. The target is extensible behavior semantics without accumulating incremental artifacts, patch modules, wrappers, or parallel score surfaces.

The implementation should replace the current vague fixed-horizon side labels as the semantic center with one compact path-behavior product and one canonical action surface.

## Non-goals

Do not add a pile of incremental artifacts such as:

```text
tailtree-path-behavior-distribution.csv
tailtree-selected-path-behavior.csv
tailtree-horizon-action-panel.csv
```

as permanent peers.

Do not create a module-per-concept layout such as:

```text
tailtree/behavior.py
tailtree/utility.py
tailtree/policy.py
tailtree/action.py
```

unless a module has an independent public owner, multiple callers, and real size pressure.

Do not create shims or compatibility wrappers.

Do not keep old/new label systems as two parallel authorities.

## Current reduction audit

Current packages:

```text
src/qooi/scanner/tailtree/
  model.py
  evidence.py
  __init__.py

src/qooi/scanner/tailrun/
  core.py
  selection.py
  artifacts.py
  planning.py
  search.py
  types.py
  __init__.py
```

Current useful owners:

```text
tailtree/model.py:
  labels, training frames, model wrapper, JSON model IO

tailtree/evidence.py:
  model prediction -> evidence rows

tailrun/core.py:
  lifecycle composition: prepare, train/load, score, artifacts

tailrun/selection.py:
  candidate replay, calibrated margins, objective/HPO metrics

tailrun/artifacts.py:
  artifact file writes, stale cleanup, model/profile outputs

tailrun/types.py:
  cross-module dataclasses and protocols
```

Reduction conclusion:

```text
No new module is needed for the first slice.
```

The first implementation should modify existing owners:

```text
path behavior labels        -> tailtree/model.py
behavior evidence columns   -> tailtree/evidence.py if model output changes later
selection/action projection -> tailrun/selection.py
artifact write              -> tailrun/artifacts.py, but only for one canonical artifact
lifecycle wiring            -> tailrun/core.py
```

## Target reduced architecture

### One label product, not layered artifacts

Replace the current label output conceptually with one richer label frame:

```text
TailtreeLabelFrame
```

It is still a `pl.DataFrame`; no new wrapper dataclass unless types become unavoidable.

Grain:

```text
symbol × decision_bar_close_ms × outcome_horizon
```

Required columns:

```text
tail_touch_up
tail_touch_down
tail_touch_any
tail_touch_both
first_touch_side
path_state
path_actionability
path_blocker
path_utility_up
path_utility_down
path_utility_margin_up
path_utility_margin_down
```

Compatibility mapping during migration:

```text
tail_up        -> tail_touch_up
tail_down      -> tail_touch_down
tail_any       -> tail_touch_any
tail_both      -> tail_touch_both
tail_state     -> coarse path_state values for old code until retired
```

But do not keep both vocabularies long-term. Migrate current tests/callers to the new names and delete old aliases in the same migration slice when feasible.

### One canonical scanner action surface

Instead of multiple diagnostic CSV peers, create one canonical candidate/action artifact:

```text
tailtree-action-surface.csv
```

This replaces the need for separate selected-behavior and horizon-action-panel artifacts.

Grain:

```text
profile/trial/fold × symbol × decision_bar_close_ms × action_side × entry_horizon
```

Columns:

```text
symbol
decision_bar_close_ms
action_side
entry_horizon
max_valid_horizon
actionability
path_state_profile
best_path_state
best_utility_margin
clean_horizon_count
chop_horizon_count
reversal_horizon_count
contradicting_horizon_count
calibrated_side_margin
false_direction_rate
blocker_reason
model_tag
trial_id
fold_id
```

This artifact is the user-facing semantic surface. Existing `tailtree-selection-efficiency.csv` can remain as HPO/search feedback, but it should not grow into another report-like candidate surface.

### One distribution appendix, in-memory first

Path behavior distribution is useful but should not become a permanent top-level artifact unless the report needs it externally.

Preferred first implementation:

```text
compute distribution in memory
feed report/profile frame if needed
only write if it is part of the single action surface or selection-efficiency appendix
```

If written, it should be a section inside the existing profile/frames mechanism rather than a new top-level CSV.

## Target API graph

### Existing workflow stays

```text
scripts/scanner_potential.py
  -> qooi.scanner.workflow.run(config_path)
     -> state product
     -> outcome product
     -> tailrun.core.run_tailtree(...)
     -> rank/report
```

No new workflow dispatcher.

### Tailtree label graph

Current:

```text
qooi.scanner.tailtree.model.label_tail_exceedances(outcomes, threshold_pct=...)
```

Target:

```text
qooi.scanner.tailtree.model.label_tail_paths(
    outcomes: pl.DataFrame,
    *,
    threshold_pct: float,
    utility_floor: float,
    margin_floor: float,
    path_efficiency_floor: float,
    late_bar_ratio: float | None,
) -> pl.DataFrame
```

This should replace, not wrap, `label_tail_exceedances` once callers/tests are migrated.

Initial implementation may edit `label_tail_exceedances` directly if that avoids a compatibility period. Preferred final public name:

```text
label_tail_paths
```

Target columns:

```text
tail_touch_up
tail_touch_down
tail_touch_any
tail_touch_both
first_touch_side
path_state
path_actionability
path_blocker
path_utility_up
path_utility_down
path_utility_margin_up
path_utility_margin_down
```

Path states:

```text
none
clean_up
clean_down
up_first_both
down_first_both
chop_both
late_up
late_down
```

Actionability values at label level:

```text
tradable_up
tradable_down
reversal_watch
volatility_watch
gray_zone
no_action
```

### Training graph

Current:

```text
tailtree_training_frame(observations, labeled_outcomes, direction=...)
event_lift_training_values(...)
any_event_training_values(...)
side_only_training_values(...)
```

Reduced target:

```text
tailtree_training_frame(
    observations,
    labeled_paths,
    *,
    target: TailtreeTarget,
    outcome_horizon: int,
) -> TailtreeTrainingFrame
```

Where `TailtreeTarget` is a small literal, not many functions:

```text
path_clean_up
path_clean_down
path_chop
utility_up
utility_down
utility_margin_up
utility_margin_down
```

Reduction rule:

```text
Delete separate narrow training helpers once the generic target selector exists.
```

Do not leave:

```text
any_event_training_values
side_only_training_values
event_lift_training_values
```

as permanent parallel entry points if `tailtree_training_frame(..., target=...)` can own all target selection cleanly.

### Evidence graph

Current evidence functions can remain:

```text
leaf_evidence_frame(...)
score_bucket_evidence_frame(...)
```

Do not add `path_behavior_evidence_frame` in the first slice.

Instead, score/evidence frames should carry target metadata:

```text
target
path_state_target
action_side
outcome_horizon
```

If a future behavior model requires new evidence layout, replace `score_bucket_evidence_frame` with a more general score-bucket evidence function rather than adding another peer.

### Selection graph

Current:

```text
score_bucket_candidate_frame(...)
paired_candidate_replay_frame(...)
calibrated_candidate_replay_frame(...)
tailtree_selection_metrics_frame(...)
```

Target:

```text
tailtree_action_surface_frame(
    scored_candidates: pl.DataFrame,
    labeled_paths: pl.DataFrame,
    *,
    policy: TailtreeActionPolicy,
) -> pl.DataFrame
```

This should absorb selected-behavior and horizon-action-panel responsibilities.

`TailtreeActionPolicy` should be a small frozen dataclass in `tailrun/types.py` only if config already needs to pass more than 3 scalar policy values. Otherwise pass scalar args directly.

Output:

```text
tailtree-action-surface.csv
```

Keep `tailtree_selection_metrics_frame` as search/HPO feedback, but it should consume action-surface aggregates rather than recomputing a second semantic surface.

## Config reduction

Do not add multiple nested config sections immediately.

First slice can use hardcoded policy defaults inside tailtree config constants or reuse existing threshold fields:

```text
threshold_pct
outcome_horizon
selection top_k/top_pct
```

Only add config after artifacts prove the knobs matter.

If config is needed, prefer one compact section:

```toml
[potential.evidence.tailtree.path]
utility_floor = 0.0
margin_floor = 0.0
path_efficiency_floor = 0.0
late_bar_ratio = 0.75
max_chop_probability = 0.25
```

Do not create separate `[behavior]`, `[utility]`, `[horizon_policy]` sections until there is clear independent ownership.

## Artifact reduction

### Keep

```text
tailtree-profile-runs.csv
```

Purpose: run/profile feedback.

```text
tailtree-selection-efficiency.csv
```

Purpose: HPO/search feedback.

```text
tailtree-action-surface.csv
```

Purpose: canonical semantic candidate/action surface.

### Replace or retire

```text
tailtree-label-distribution.csv
```

Either replace with path-state distribution columns inside action surface/profile frames, or keep only during migration. It should not grow as a second semantic report surface.

### Do not add as permanent peer artifacts

```text
tailtree-path-behavior-distribution.csv
tailtree-selected-path-behavior.csv
tailtree-horizon-action-panel.csv
```

These are design concepts, not permanent files.

## Module reduction plan

### Phase 0: design acceptance only

No code.

User reviews this plan and chooses:

```text
A. direct rename/replace current label API now
B. one migration slice with temporary compatibility aliases
```

Preferred ponytail choice: A, if tests can be updated in one pass.

### Phase 1: label reduction

Modify:

```text
src/qooi/scanner/tailtree/model.py
```

Do:

```text
replace current label_tail_exceedances semantics with path-label semantics
or rename it to label_tail_paths and update all callers
```

Delete/avoid:

```text
no new behavior.py
no utility.py
no compatibility wrapper long-term
```

Tests:

```text
tests/test_tailtree_explicit_labels.py
```

Update tests to assert:

```text
tail_touch_up/down
first_touch_side
path_state
path_actionability
path_utility_margin_up/down
```

Verification:

```bash
uv run python -m pytest tests/test_tailtree_explicit_labels.py -q
```

### Phase 2: training target reduction

Modify:

```text
src/qooi/scanner/tailtree/model.py
src/qooi/scanner/tailrun/core.py
src/qooi/scanner/config.py
```

Do:

```text
collapse narrow target helpers into one target-selecting training frame
```

Target API:

```text
tailtree_training_frame(observations, labeled_paths, target=..., outcome_horizon=...)
```

Delete after migration:

```text
any_event_training_values
side_only_training_values
possibly event_lift_training_values if covered by target selector
```

Verification grep:

```bash
grep -R "any_event_training_values\|side_only_training_values" -n src tests
```

Expected after phase: empty.

### Phase 3: action surface reduction

Modify:

```text
src/qooi/scanner/tailrun/selection.py
src/qooi/scanner/tailrun/core.py
src/qooi/scanner/tailrun/artifacts.py
```

Do:

```text
create one tailtree_action_surface_frame(...)
write one tailtree-action-surface.csv
make selection-efficiency consume aggregates from action surface where possible
```

Avoid:

```text
no tailtree-selected-path-behavior.csv
no tailtree-horizon-action-panel.csv
no report-side recomputation
```

Verification:

```bash
grep -R "path-behavior-distribution\|selected-path-behavior\|horizon-action-panel" -n src docs/architecture docs/graph
```

Expected in source/architecture/graph after implementation: empty or explicitly marked rejected in report docs only.

### Phase 4: report surface reduction

Modify only if needed:

```text
src/qooi/scanner/output.py
```

Do:

```text
show canonical Candidate Action Surface
show compact behavior summary derived from action surface
remove overlapping/redundant side-readiness sections if they duplicate the action surface
```

Renderer rule:

```text
No raw path-state computation in output.py.
No CSV read-back for internal transport.
```

### Phase 5: docs graph alignment

Modify:

```text
docs/architecture/scanner.md
docs/graph/scanner.md
docs/graph/tailtree.md
```

Do:

```text
replace tail_up/down as primary vocabulary with tail_touch/path_state/actionability
show one action surface artifact
remove planned multi-artifact bloat
```

## Acceptance gates

### API grep gates

No permanent narrow helper API:

```bash
grep -R "any_event_training_values\|side_only_training_values" -n src tests
```

No permanent planned artifact bloat:

```bash
grep -R "tailtree-path-behavior-distribution\|tailtree-selected-path-behavior\|tailtree-horizon-action-panel" -n src docs/architecture docs/graph
```

No extra modules unless accepted:

```bash
find src/qooi/scanner/tailtree -maxdepth 1 -type f
```

Expected first implementation:

```text
model.py
evidence.py
__init__.py
```

### Quality gates

```bash
uv run python -m pytest tests/test_tailtree_explicit_labels.py tests/test_scanner_workflow_migration.py tests/test_state.py -q
uv run python -m ruff check src/qooi/scanner/tailtree/model.py src/qooi/scanner/tailrun/selection.py src/qooi/scanner/tailrun/core.py src/qooi/scanner/tailrun/artifacts.py src/qooi/scanner/tailrun/types.py src/qooi/scanner/config.py tests/test_tailtree_explicit_labels.py
uv run python -m ty check src/qooi/scanner/tailtree/model.py src/qooi/scanner/tailrun/selection.py src/qooi/scanner/tailrun/core.py src/qooi/scanner/tailrun/artifacts.py src/qooi/scanner/tailrun/types.py src/qooi/scanner/config.py tests/test_tailtree_explicit_labels.py
```

### Smoke gate

```bash
uv run python -m scripts.scanner_potential --config configs/potential-paired-replay-test-tailtree.toml
```

Inspect:

```text
tailtree-action-surface.csv exists
tailtree-selection-efficiency.csv still exists
actionability rows are non-empty
path_state_profile distinguishes clean_down vs down_first_both/chop_both
```

## Final reduced target summary

Preferred end state:

```text
One label API:
  label_tail_paths

One training frame API:
  tailtree_training_frame(..., target=...)

One semantic candidate artifact:
  tailtree-action-surface.csv

One HPO/search feedback artifact:
  tailtree-selection-efficiency.csv

No new tailtree modules in first slice.
No incremental diagnostic CSV pile.
No wrapper APIs.
No raw score averaging across horizons.
No binary final long/short without path_state/actionability.
```

## Recommendation to user

Proceed only after agreeing on these reduction choices:

1. Rename/replace `label_tail_exceedances` with `label_tail_paths`, or keep the old name for one migration slice?
2. Allow deleting `any_event_training_values` and `side_only_training_values` once the target selector exists?
3. Make `tailtree-action-surface.csv` the only new permanent artifact?
4. Keep new behavior logic inside `tailtree/model.py` and `tailrun/selection.py` first, avoiding new modules?

My recommendation:

```text
Yes to all four.
```
