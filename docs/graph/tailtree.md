# Tailtree Graph

Implementation-facing graph for the current tailtree/tailrun path.

## Role

Tailtree is the scanner's LightGBM/GPD tail-event evidence path. It consumes observation/outcome frames and emits evidence rows, JSON model artifacts, profile feedback, and selection-efficiency rows.

Tailtree output is scanner evidence. It is not a trade signal or execution instruction.

## Packages

```text
qooi.scanner.tailtree
  model.py      # labels, training frame, LightGBM model wrapper
  evidence.py   # leaf and score-bucket evidence frames

qooi.scanner.tailrun
  types.py      # concrete run/profile/artifact dataclasses
  planning.py   # profile runs, fixed/optuna run records, walkforward folds
  search.py     # Optuna module loading, trial suggestions, HPO feedback helpers
  core.py       # run_tailtree lifecycle boundary
  artifacts.py  # JSON model/profile/selection-efficiency artifact IO
  selection.py  # selection replay helpers retained for focused budget analysis
```

## Workflow boundary

```text
qooi.scanner.workflow
  -> TailtreeInputFrames(observations, source_outcomes, realized, histories)
  -> qooi.scanner.tailrun.core.run_tailtree(frames, config=config, profile=profile)
  -> qooi.scanner.tailrun.artifacts.write_tailtree_profile_runs(...)
  -> qooi.scanner.tailrun.artifacts.write_tailtree_selection_efficiency(...)
```

`workflow.py` does not own:

```text
TailTreeModel
TrainConfig
Optuna sampling
walkforward fold creation
training-frame construction
evidence dispatch
selection-efficiency row construction
```

## Input and output records

```text
TailtreeInputFrames
  observations
  source_outcomes
  realized
  histories

TailtreePreparedFrames
  observations
  source_outcomes
  realized
  histories
  outcomes
  labeled_outcomes
  categorical_features
  continuous_features
  score_observations
  score_labeled_outcomes

TailtreeRunOutput
  evidence: pl.DataFrame
  models: dict[tuple[int, TailtreeDirection], TailtreeArtifactTree]
  profile_runs: tuple[TailtreeProfileFeedback, ...]
  selection_efficiency: pl.DataFrame
```

## Label and training graph

```text
outcome.potential_outcome_frame(...)
  -> tailtree.model.label_tail_paths(...)
  -> tailtree.model.tailtree_training_frame(...)
  -> TailTreeModel.train(...)
```

Tail path labels:

```text
tail_touch_up      = forward_max_return_pct > threshold_pct
tail_touch_down    = forward_min_return_pct < -threshold_pct
tail_touch_both    = both touches inside the same horizon
first_touch_side   = up | down | tie | none from time_to_max/min
path_state         = none | clean_up | clean_down | up_first_both | down_first_both | chop_both | late_up | late_down
path_actionability = tradable_up | tradable_down | reversal_watch | gray_zone | no_action
```

Current `tail_up`, `tail_down`, `tail_any`, `tail_both`, and `tail_state` remain compatibility excursion columns for existing training/evidence paths. The semantic center is `path_state` + `path_actionability`.

Training population is tail/path rows. Validation/evidence population is outcome-known rows for the split/fold.

## Evidence graph

Objective decides evidence bucket:

```text
tail_severity_gpd
  -> TailTreeModel.predict_leaf
  -> tailtree.evidence.leaf_evidence_frame

tail_utility_quantile
  -> TailTreeModel.predict_score
  -> tailtree.evidence.score_bucket_evidence_frame
```

Current advanced profile uses:

```text
tail_utility_quantile
h24
Optuna
walkforward
```

## Planning/search graph

```text
tailrun.planning.tailtree_profile_runs(config)
tailrun.planning.tailtree_optuna_profiles(config)
tailrun.planning.tailtree_fold_specs(...)
tailrun.planning.tailtree_frame_split(...)

tailrun.search.optuna_module()
tailrun.search.suggest_tailtree_trial_params(...)
tailrun.search.trial_objective_score(...)
```

The advanced config currently runs:

```text
max_trials = 4
max_folds = 2
outcome_horizon = [24]
```

## Artifact graph

Model artifacts:

```text
models/<model_tag>_<horizon>_<direction>.json
```

Feedback artifacts:

```text
tailtree-profile-runs.csv
tailtree-selection-efficiency.csv
tailtree-action-surface.csv
models/tailtree-selection-efficiency.csv
```

Profile artifacts:

```text
profile/stages.csv
profile/frames.csv
profile/summary.md
```

## Selection/action surface grain

`tailtree-action-surface.csv` row grain:

```text
profile/trial/fold × symbol × decision_bar_close_ms × action_side × entry_horizon × score_bucket
```

Core columns:

```text
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
blocker_reason
```

This is the canonical semantic candidate/action surface. It absorbs selected-behavior and horizon-panel diagnostics instead of adding peer artifacts.

## Selection-efficiency grain

`tailtree-selection-efficiency.csv` row grain:

```text
profile/trial/fold × horizon × direction × budget row
```

Core columns:

```text
model_tag
objective
training_profile
trial_id
trial_source
outcome_horizon
tree_direction
budget_family
budget_value
selected_observation_count
selected_tail_count
valid_tail_lift
selected_utility_mean
profit_proxy_per_1k_observed
hpo_score
promotion_threshold_pass_int
fit_seconds
```

This artifact is the scanner's canonical opportunity-selection feedback surface. It is not realized PnL.

## Public tailtree/tailrun calls

```text
qooi.scanner.tailtree.model.label_tail_paths
qooi.scanner.tailtree.model.tailtree_training_frame
qooi.scanner.tailtree.model.tailtree_target_training_values
qooi.scanner.tailtree.model.TailTreeModel
qooi.scanner.tailtree.evidence.leaf_evidence_frame
qooi.scanner.tailtree.evidence.score_bucket_evidence_frame
qooi.scanner.tailrun.core.run_tailtree
qooi.scanner.tailrun.core.load_predict
qooi.scanner.tailrun.core.run_frame_split
qooi.scanner.tailrun.artifacts.write_tailtree_profile_runs
qooi.scanner.tailrun.artifacts.write_tailtree_action_surface
qooi.scanner.tailrun.artifacts.write_tailtree_selection_efficiency
qooi.scanner.tailrun.selection.tailtree_action_surface_frame
```
