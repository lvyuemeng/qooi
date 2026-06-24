# Tailtree Graph

Implementation-facing graph for the current tailtree/tailrun path.

## Role

Tailtree is the scanner's LightGBM/GPD tail-event evidence path. It consumes observation/outcome frames and emits evidence rows, JSON model artifacts, profile feedback, action-surface rows, and selection-efficiency/frontier rows.

Tailtree output is scanner evidence. It is not a trade signal or execution instruction.

## Packages

```text
qooi.scanner.tailtree
  labels.py     # TailEventPolicy, reference fitting, path labels
  model.py      # training frame, target values, LightGBM model wrapper
  evidence.py   # leaf and score-bucket evidence frames

qooi.scanner.tailrun
  types.py      # concrete run/profile/result records plus Pydantic artifact-row serialization
  planning.py   # profile runs, model-id predict records, optuna run records, walkforward folds
  search.py     # Optuna module loading, trial suggestions, HPO feedback helpers
  core.py       # run_tailtree lifecycle boundary, train_features, LocalModelSpec
  artifacts.py  # JSON model/profile/selection-efficiency/frontier artifact IO
  selection.py  # active candidate_dual_guard metrics, replay, action surface, frontier rows
```

## Workflow boundary

```text
qooi.scanner.workflow
  -> TailtreeInputFrames(observations, source_outcomes, realized, histories)
  -> qooi.scanner.tailrun.core.run_tailtree(frames, config=config, profile=profile)
  -> qooi.scanner.tailrun.artifacts.write_tailtree_profile_runs(...)
  -> qooi.scanner.tailrun.artifacts.write_tailtree_selection_efficiency(...)
  -> qooi.scanner.tailrun.artifacts.write_tailtree_frontier_benchmark(...)
  -> qooi.scanner.tailrun.artifacts.write_tailtree_action_surface(...)
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
  action_surface: pl.DataFrame
  selection_error_anatomy: pl.DataFrame
  boundary_anatomy: pl.DataFrame
  contradiction_audit: pl.DataFrame
```

## Label and training graph

```text
outcome.potential_outcome_frame(...)
  -> tailtree.labels.TailEventPolicy(...).label_paths(...)
  -> tailrun.core.train_features(observations, role="opportunity" | "candidate")
  -> tailtree.model.tailtree_training_frame(...)
  -> tailtree.model.tailtree_target_training_values(..., target="tail_event_lift")
  -> TailTreeModel.train(...)
```

Training features are selected by persistent known-at-close contracts. The active feature set includes source-context features when the aligned columns are present.

Current source-context input family:

```text
funding / long-short-ratio / open-interest / taker-pressure
state + transition + run-length + price-divergence columns
```

Excluded from training:

```text
high-cardinality source path strings
current-only books/trades
execution/cost/slippage/wallet fields
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

## Current objective graph

Current advanced profile:

```text
horizon = 24
stage-1 objective = tail_event_lift
search = Optuna
validation = walkforward
final selection objective = candidate_dual_guard
```

Public config split:

```text
potential-tailtree-train.toml    # trains/scans current frontier, emits candidate_dual_guard
potential-tailtree-predict.toml  # loads JSON models by model_id, emits loaded tail_event_lift evidence
```

Graph:

```text
tail_event_lift tree evidence
  -> score_bucket_population_frame(...)
  -> candidate_gate_frame(...)
  -> promoter_target_frame(...)
  -> opposite_guard_target_frame(...)
  -> weak_path_guard_target_frame(...)
  -> LocalModelSpec(promoter_label, promoter_weight, promotion_score, tail_event_lift)
  -> _fit_local_model_scores(...)
  -> LocalModelSpec(opposite_guard_label, opposite_guard_weight, opposite_guard_score, path_guard)
  -> _fit_local_model_scores(...)
  -> LocalModelSpec(weak_path_guard_label, weak_path_guard_weight, weak_path_guard_score, path_guard)
  -> _fit_local_model_scores(...)
  -> dual_guarded_promotion_selection_metrics_frame(...)
  -> frontier_benchmark_frame(...)
```

Only `candidate_dual_guard` is emitted as the final candidate-local selection objective. `candidate_conditional_promoter`, `candidate_opposite_guard`, `continuous_guard_curve`, `two_model_guard`, and suffixed `*_source_blended` rows are removed from active output.

## Evidence graph

Objective decides evidence bucket:

```text
tail_event_lift
  -> TailTreeModel.predict_score
  -> tailtree.evidence.score_bucket_evidence_frame

tail_severity_gpd
  -> TailTreeModel.predict_leaf
  -> tailtree.evidence.leaf_evidence_frame
```

The current public configs use `tail_event_lift`.

## Planning/search graph

```text
tailrun.planning.tailtree_profile_runs(config)
tailrun.planning.tailtree_optuna_profiles(config)
tailrun.planning.tailtree_predict_run(tailtree)
tailrun.planning.tailtree_fold_specs(...)
tailrun.planning.tailtree_frame_split(...)

tailrun.search.optuna_module()
tailrun.search.suggest_tailtree_trial_params(...)
tailrun.search.trial_objective_score(...)
```

The train config currently runs:

```text
outcome_horizon = [24]
walkforward folds
Optuna trials from the profile config
```

Predict config currently resolves:

```text
TailtreeModelRefConfig.model_id -> {model_dir}/{model_id}.json
model_id suffix -> horizon + direction
model JSON metadata.train_config -> profile feedback params
```

Predict-only does not carry fixed training parameters and does not train candidate-local promoter/guard models.

## Artifact graph

Model artifacts:

```text
models/<model_tag>_<horizon>_<direction>.json
```

Feedback artifacts:

```text
tailtree-profile-runs.csv
tailtree-selection-efficiency.csv
tailtree-frontier-benchmark.csv
tailtree-action-surface.csv
tailtree-selection-error-anatomy.csv
tailtree-dual-guard-boundary-anatomy.csv
tailtree-actionability-contradiction-audit.csv
tailtree-source-timeseries-features.csv
tailtree-feature-pack-stability.csv
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

Current objective values:

```text
tail_event_lift       # stage-1 evidence feedback
candidate_dual_guard  # final candidate-local frontier selection
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
behavior_hpo_score
paired_behavior_false_direction_rate
paired_behavior_utility_margin_mean
promotion_threshold_pass_int
fit_seconds
```

Pydantic row/dump boundaries:

```text
TailtreeSelectionEfficiencyRow.model_dump()
TailtreeReplayMetrics.model_dump()
PromotionSelectionMetricRow.model_dump()
DualGuardRowLabels.model_dump()
LocalModelSpec.model_dump()
```

This artifact is the scanner's canonical opportunity-selection feedback surface. It is not realized PnL.

## Recent smoke contract

Verified train smoke:

```text
tailtree-selection-efficiency.csv shape: (3504, 78)
selection objectives: candidate_dual_guard=3456, tail_event_lift=48
tailtree-frontier-benchmark.csv shape: (2107, 86)
frontier objective: candidate_dual_guard only
forbidden objective rows: 0 for source_blended, candidate_conditional_promoter, candidate_opposite_guard, continuous_guard_curve, two_model_guard
fresh model metadata: 17 source-context feature columns
```

Verified predict-only smoke:

```text
tailtree-selection-efficiency.csv shape: (8, 71)
selection objective: loaded tail_event_lift only
frontier benchmark: absent because candidate-local guard models are not trained in predict-only mode
```

## Public tailtree/tailrun calls

```text
qooi.scanner.tailtree.model.tailtree_training_frame
qooi.scanner.tailtree.model.tailtree_target_training_values
qooi.scanner.tailtree.model.TailTreeModel
qooi.scanner.tailtree.labels.TailEventPolicy
qooi.scanner.tailtree.labels.tailtree_label_distribution_frame
qooi.scanner.tailtree.evidence.leaf_evidence_frame
qooi.scanner.tailtree.evidence.score_bucket_evidence_frame
qooi.scanner.tailrun.core.run_tailtree
qooi.scanner.tailrun.core.train_features
qooi.scanner.tailrun.core.LocalModelSpec
qooi.scanner.tailrun.planning.tailtree_predict_run
qooi.scanner.tailrun.artifacts.write_tailtree_profile_runs
qooi.scanner.tailrun.artifacts.write_tailtree_action_surface
qooi.scanner.tailrun.artifacts.write_tailtree_selection_efficiency
qooi.scanner.tailrun.artifacts.write_tailtree_frontier_benchmark
qooi.scanner.tailrun.selection.tailtree_action_surface_frame
qooi.scanner.tailrun.selection.dual_guarded_promotion_selection_metrics_frame
qooi.scanner.tailrun.selection.frontier_benchmark_frame
```
