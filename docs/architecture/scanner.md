# Scanner Architecture

## Purpose

The scanner finds symbols whose known-at-close market/source state is associated with concentrated future tail behavior. It writes research artifacts and review candidates only.

Non-goals:

- no live trading;
- no order sizing;
- no execution/cost/slippage model inside scanner promotion;
- no wallet or exchange-account decisions.

## Current module ownership

```text
src/qooi/scanner/
├── config.py       # PotentialConfig and scanner section config models
├── workflow.py     # outer lifecycle: config, universe, market load, pipe composition, final write
├── state.py        # known-at-close state/features/observation rows
├── outcome.py      # future/path/source outcome rows
├── ladder.py       # fixed ladder evidence path
├── rank.py         # candidate matching, comparable surface, ranking
├── output.py       # market readiness, review decisions, markdown report
├── transitions.py  # transition-pattern analysis
├── tailtree/       # labels, LightGBM/GPD model, prediction/evidence products
└── tailrun/        # tailtree lifecycle, profiles, Optuna, selection metrics, artifacts
```

Do not document removed transitional scanner modules as current APIs. The list above is the current source-of-truth layout.

## Workflow pipe

```text
scripts/scanner_potential.py
  -> qooi.scanner.workflow.run(config_path)
     -> load_config
     -> resolve universe
     -> pipeline.load_market via scanner_market_request/scanner_market_policy
     -> state.classify_states
     -> state.extract_continuous_features
     -> state.potential_observation_frame
     -> outcome.realized_transition_frame
     -> outcome.source_outcomes_frame
     -> outcome.potential_outcome_frame
     -> evidence dispatch
        ladder: ladder.evidence
        tailtree: tailrun.core.run_tailtree
     -> rank.candidate_metric_surface
     -> rank.rank_candidates
     -> output.review_decisions
     -> output.render_report
     -> profile artifacts
```

`workflow.py` is allowed to compose the scanner pipe. Tailtree training, Optuna sampling, fold construction, model persistence, and selection-efficiency rows belong under `tailrun/` and `tailtree/`, not workflow helpers.

## Config profiles

Public scanner configs are reduced to two files:

```text
configs/potential-tailtree-train.toml
configs/potential-tailtree-predict.toml
```

Train config role:

```text
h24 tail_event_lift stage-1 evidence
source-context candidate_dual_guard final selection
Optuna training
walkforward evaluation
larger 160-symbol research surface
selection-efficiency/frontier feedback
```

Predict config role:

```text
load existing model JSONs by model_id
score/report current observations
no fixed-parameter training profile
no candidate-local model training
```

The scanner currently prefers one horizon, `h24`, because daily prediction freshness often approaches the 24h boundary and h24 has stronger tail-label support than shorter horizons. The train config should not train h12/h48 by default while the current frontier and reporting policy are h24-specific.

## Current tailtree objective and parameters

Current tailtree frontier:

```text
stage-1 evidence objective: tail_event_lift
final selection objective: candidate_dual_guard
horizon: h24
validation: walkforward
search: Optuna
source input: active feature set includes 17 persistent source-context columns
artifact frontier: tailtree-frontier-benchmark.csv
```

The final `candidate_dual_guard` row family means:

```text
candidate-local promoter score
  + opposite-direction guard score
  + weak/no-tail path guard score
```

The internal guard models remain implementation ingredients. They are not emitted as final competing objectives.

Current source-context inputs:

```text
funding_level_state
funding_level_transition
funding_price_divergence_24h
funding_direction_run_length
lsr_level_state
lsr_level_transition
lsr_price_divergence_24h
lsr_direction_run_length
lsr_log_ratio_change_24h
oi_flow_state
oi_flow_transition
oi_flow_run_length
oi_change_pct_24h
taker_pressure_state
taker_pressure_transition
taker_pressure_run_length
taker_buy_pressure_24h_mean
```

Excluded high-cardinality source paths:

```text
funding_path_24h
lsr_path_24h
oi_price_flow_path_24h
taker_pressure_path_24h
```

Reason for the current choice:

```text
tail_event_lift keeps the broad extreme-opportunity model target;
pure clean/actionable labels and standalone guard objectives benchmarked worse or unstable;
source-feature input showed the best observed row as candidate_dual_guard_source_blended with selected=75, precision=0.613, false-direction=0.133, utility=3.377 versus fair control candidate_opposite_guard selected=50, precision=0.600, false-direction=0.140, utility=2.991;
the suffix branch was normalized away by folding source-context columns into the single active feature set;
lower-performance output objectives were deleted rather than preserved as active alternatives.
```

Recent verified advanced smoke after row-builder cleanup:

```text
tailtree-selection-efficiency.csv shape: (3504, 78)
selection objectives: candidate_dual_guard=3456, tail_event_lift=48
tailtree-frontier-benchmark.csv shape: (2107, 86)
frontier objective: candidate_dual_guard only
forbidden objective rows: 0 for source_blended, candidate_conditional_promoter, candidate_opposite_guard, continuous_guard_curve, two_model_guard
fresh model metadata: 17 source-context input columns
predict-only selection-efficiency.csv shape: (8, 71), loaded tail_event_lift only
```

Top inspected frontier rows from that smoke:

| objective | selected | precision | false-dir | utility | action |
|---|---:|---:|---:|---:|---|
| candidate_dual_guard | 50 | 0.600 | 0.340 | 5.297 | promote_candidate_frontier |
| candidate_dual_guard | 50 | 0.580 | 0.120 | 3.652 | promote_candidate_frontier |
| candidate_dual_guard | 50 | 0.580 | 0.160 | 3.574 | promote_candidate_frontier |
| candidate_dual_guard | 50 | 0.560 | 0.120 | 3.551 | promote_candidate_frontier |

Architecture docs summarize current choice only. Full empirical reports and failed objective histories stay under `docs/report/`.

## Data and freshness boundaries

Known-at-close state only:

```text
state.py -> observations keyed by symbol/timeframe/bar close
```

Future/path labels only:

```text
outcome.py -> outcome rows keyed by symbol/bar close/horizon
```

Persistent derivative-source families may train tailtree when aligned historically:

```text
funding
open_interest
taker_volume
long_short_ratios
```

Books/trades are current-review context unless a consistent historical artifact contract exists.

Reports must expose:

```text
missing data
stale data
provider-bounded history
current-only sources
coin_too_new symbols
deferred_by_budget rows
```

## Tailtree boundary

Tailtree input:

```text
TailtreeInputFrames(observations, source_outcomes, realized, histories)
```

Tailtree output:

```text
TailtreeRunOutput(
  evidence,
  models,
  profile_runs,
  selection_efficiency,
  action_surface,
  selection_error_anatomy,
  boundary_anatomy,
  contradiction_audit,
)
```

Tailtree internal ownership:

```text
tailtree/labels.py    -> TailEventPolicy, reference fitting, path labels, label distribution
tailtree/model.py     -> target/training values, TailTreeModel
tailtree/evidence.py  -> leaf/score-bucket evidence frames
tailrun/planning.py   -> profile runs, trial params, fold specs, objective jobs
tailrun/core.py       -> train/load/score lifecycle, feature-set selection, local model specs, profile artifact frames
tailrun/selection.py  -> score-bucket candidates, paired replay, selection/HPO metrics, frontier rows
tailrun/types.py      -> cross-module run/result dataclasses and Pydantic artifact-row serialization types
```

Tailtree label vocabulary uses fixed-horizon path semantics:

```text
tail_touch_up/down   # threshold touch facts
first_touch_side     # up/down/tie/none from time_to_max/min
path_state           # none/clean_up/clean_down/up_first_both/down_first_both/chop_both/late_up/late_down
path_actionability   # tradable_up/tradable_down/reversal_watch/gray_zone/no_action
```

Current `tail_up`, `tail_down`, `tail_any`, `tail_both`, and `tail_state` remain compatibility excursion columns during migration, but final scanner suggestions should use `path_state` and actionability, not raw up/down touch flags.

Artifacts:

```text
report.md
tailtree-profile-runs.csv
tailtree-label-distribution.csv
tailtree-selection-efficiency.csv
tailtree-frontier-benchmark.csv
tailtree-action-surface.csv
tailtree-selection-error-anatomy.csv
tailtree-dual-guard-boundary-anatomy.csv
tailtree-actionability-contradiction-audit.csv
tailtree-source-timeseries-features.csv
tailtree-feature-pack-stability.csv
models/*.json
profile/*.csv
```

Model persistence is JSON. The scanner does not pickle models.

## Promotion semantics

`rank.py` builds the comparable candidate surface. `output.review_decisions` applies review gates:

```text
prediction freshness
missing/stale source blockers
support threshold
tail_lift threshold
opposite-direction conflict watch
top-N promote cap
```

Promote/watch/skip rows are research review decisions, not execution instructions.

## Forbidden dependencies

- scanner -> executor/core basket/wallet modules;
- tailtree -> ladder cross-imports;
- state -> future outcome columns;
- outcome -> model/evidence internals;
- report/output code reading CSV artifacts back as an internal transport;
- opaque row probing in model/report boundaries when a typed frame or dataclass/Pydantic row exists.
