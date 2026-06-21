# Tailtree Path-Behavior Design and API Graph

## Purpose

Clarify the improvement proposition before implementation. The current tailtree labels fixed-horizon extrema (`tail_up`, `tail_down`) and then tries to choose a side. That is insufficient because a fixed horizon can contain clean motion, reversal, or chop, and the same decision row can be clean on a short horizon but untradable on a longer horizon.

This design separates:

```text
known-at-close features
  -> future path measurement
  -> path behavior label
  -> side utility label
  -> calibrated horizon action policy
  -> final scanner suggestion
```

The goal is not to create a live-trading signal. The scanner still emits research/review candidates and blockers only.

## Current behavior being replaced or extended

Current tail labels answer:

```text
Did the next H bars touch an up/down threshold?
```

Current simplified label logic:

```text
forward_max_return_pct = max future return within H bars
forward_min_return_pct = min future return within H bars

tail_up   = forward_max_return_pct > threshold_pct
tail_down = forward_min_return_pct < -threshold_pct
tail_both = tail_up and tail_down
tail_state = none | up | down | both
```

Current side utility answers:

```text
If the side threshold was touched, how good was the path after retention, efficiency, speed, and drawdown/rebound penalty?
```

Current weakness:

```text
`tail_down = true` can mean clean short, dump-then-rebound, or broad chop.
`tail_up = true` can mean clean long, pump-then-dump, or broad chop.
```

Therefore `tail_up`/`tail_down` should remain low-level excursion facts, not final action labels.

## Proposed semantics

### Principle 1: behavior before side action

Do not begin with `up` vs `down` as a trade choice.

First ask:

```text
What kind of path happened inside this horizon?
```

Then ask:

```text
Was any side utility-dominant and actionable?
```

Then ask:

```text
Across horizons, is this a candidate, watch, blocker, or no-action row?
```

### Principle 2: horizon is a behavior dimension

A row can be:

```text
6h:  clean_up
24h: up_first_both
48h: chop_both
```

That is not a contradiction. It means:

```text
short-horizon long opportunity, not a swing-long opportunity.
```

Final output must preserve horizon profile instead of collapsing all horizons into one direction.

### Principle 3: utility is quality, not behavior identity

Utility remains important, but it should not define the behavior class alone.

Use:

```text
path_state     = geometric/temporal future behavior
utility_side   = side-specific quality of that behavior
actionability  = policy decision from calibrated behavior + utility
```

## Data products

### 1. Known-at-close observation product

Current owner:

```text
qooi.scanner.state
```

Current public calls:

```text
qooi.scanner.state.classify_states(...)
qooi.scanner.state.extract_continuous_features(...)
qooi.scanner.state.potential_observation_frame(...)
```

Grain:

```text
symbol × decision_timeframe × decision_bar_close_ms
```

Allowed inputs to models:

```text
market_stage
structure_trend_state
liquidity_event_type
atr_percentile
range_width_atr
bar_return_1h_pct
bar_return_4h_pct
bar_return_24h_pct
bar_return_4h_per_vol_7d
bar_return_24h_per_vol_7d
bar_volume_1h_to_ma_20h
bar_close_position_48h
persistent historical source features aligned known-at-close
```

Forbidden model inputs:

```text
forward_max_return_pct
forward_min_return_pct
time_to_max_bar
time_to_min_bar
post_max_drawdown_pct
post_min_rebound_pct
path_efficiency
close_retention_ratio
tail_up
tail_down
tail_state
tail_utility_up
tail_utility_down
tail_utility_margin_up
tail_utility_margin_down
```

Those are outcome/label/diagnostic fields.

### 2. Path outcome product

Current owner should remain:

```text
qooi.scanner.outcome
```

Proposed public object:

```text
PathOutcomeFrame
```

This can be a documented frame contract first, not necessarily a new dataclass immediately.

Grain:

```text
symbol × decision_bar_close_ms × outcome_horizon
```

Columns:

```text
symbol
decision_bar_close_ms
outcome_horizon
forward_max_return_pct
forward_min_return_pct
close_return_pct
time_to_max_bar
time_to_min_bar
close_retention_ratio
path_efficiency
post_max_drawdown_pct
post_min_rebound_pct
```

Responsibility:

```text
Measure what happened after the decision close.
Do not assign model labels.
Do not train models.
Do not decide actionability.
```

### 3. Path behavior product

Proposed owner:

```text
qooi.scanner.tailtree.behavior
```

Reason for separate owner:

```text
Behavior labels are tailtree/model semantics derived from outcome path columns.
They are not raw outcome measurement and not model training itself.
```

Proposed public calls:

```text
qooi.scanner.tailtree.behavior.label_path_behavior(
    outcomes: pl.DataFrame,
    *,
    threshold_pct: float,
    utility_floor: float,
    dominance_floor: float,
    early_bar_limit: int | None = None,
) -> pl.DataFrame

qooi.scanner.tailtree.behavior.path_behavior_distribution_frame(
    labeled: pl.DataFrame,
) -> pl.DataFrame
```

Output grain:

```text
symbol × decision_bar_close_ms × outcome_horizon
```

Core columns:

```text
tail_touch_up: bool
tail_touch_down: bool
tail_touch_any: bool
tail_touch_both: bool
first_touch_side: up | down | none | tie
path_state: none | clean_up | clean_down | up_first_both | down_first_both | chop_both | late_up | late_down
```

Minimal first-version `path_state` values:

```text
none
clean_up
clean_down
up_first_both
down_first_both
chop_both
```

Suggested definitions:

```text
tail_touch_up:
  forward_max_return_pct > threshold_pct

tail_touch_down:
  forward_min_return_pct < -threshold_pct

first_touch_side:
  up if tail_touch_up and (not tail_touch_down or time_to_max_bar < time_to_min_bar)
  down if tail_touch_down and (not tail_touch_up or time_to_min_bar < time_to_max_bar)
  tie if both touched and times are equal/unknown
  none if neither touched

clean_up:
  tail_touch_up and not tail_touch_down and utility_up >= utility_floor

clean_down:
  tail_touch_down and not tail_touch_up and utility_down >= utility_floor

up_first_both:
  tail_touch_both and first_touch_side == up and path is not chop_both

down_first_both:
  tail_touch_both and first_touch_side == down and path is not chop_both

chop_both:
  tail_touch_both and any of:
    - path_efficiency below floor
    - utility dominance weak
    - first_touch_side tie/unknown
    - both utilities positive but margin too small

none:
  no touch or no usable utility
```

Late variants can be added after the minimal version:

```text
late_up:
  tail_touch_up, no down touch, but time_to_max_bar after late_bar_limit

late_down:
  tail_touch_down, no up touch, but time_to_min_bar after late_bar_limit
```

### 4. Path utility product

Proposed owner:

```text
qooi.scanner.tailtree.utility
```

This can initially live beside behavior if the implementation is small. Split only if the code grows or has independent tests.

Proposed public calls:

```text
qooi.scanner.tailtree.utility.path_utility_frame(
    path_behavior: pl.DataFrame,
    *,
    utility_floor: float,
    margin_floor: float,
) -> pl.DataFrame
```

Output grain:

```text
symbol × decision_bar_close_ms × outcome_horizon
```

Core columns:

```text
tail_utility_up
tail_utility_down
tail_utility_margin_up
tail_utility_margin_down
utility_dominant_side: up | down | none
tradable_up: bool
tradable_down: bool
```

Suggested definitions:

```text
utility_dominant_side:
  up if utility_margin_up >= margin_floor
  down if utility_margin_down >= margin_floor
  none otherwise

tradable_up:
  path_state == clean_up
  and utility_dominant_side == up
  and tail_utility_up >= utility_floor

tradable_down:
  path_state == clean_down
  and utility_dominant_side == down
  and tail_utility_down >= utility_floor
```

Important:

```text
up_first_both and down_first_both are not clean directional candidates by default.
They may become watch/reversal states, not trade candidates.
```

### 5. Behavior training product

Proposed owner:

```text
qooi.scanner.tailtree.model
```

Proposed public calls:

```text
qooi.scanner.tailtree.model.path_behavior_training_frame(
    observations: pl.DataFrame,
    behavior_labels: pl.DataFrame,
    *,
    outcome_horizon: int,
) -> PathBehaviorTrainingFrame

qooi.scanner.tailtree.model.utility_training_frame(
    observations: pl.DataFrame,
    utility_labels: pl.DataFrame,
    *,
    outcome_horizon: int,
    side: Literal["up", "down"],
) -> PathUtilityTrainingFrame
```

Training grains:

```text
one training matrix per horizon for behavior
one utility matrix per horizon × side, or one multi-output utility frame if implementation remains simple
```

Input:

```text
known-at-close observation features only
```

Labels:

```text
path_state or path_state binary heads
utility_up/down/margins
tradable_up/down flags
```

### 6. Model outputs

Do not average raw scores across horizons.

For each decision row and horizon, model output should be calibrated into comparable semantic columns:

```text
p_clean_up
p_clean_down
p_up_first_both
p_down_first_both
p_chop_both
expected_utility_up
expected_utility_down
expected_utility_margin_up
expected_utility_margin_down
```

Proposed output frame grain:

```text
symbol × decision_bar_close_ms × outcome_horizon × model_tag × trial_id
```

Proposed public call:

```text
qooi.scanner.tailtree.evidence.path_behavior_evidence_frame(
    behavior_model: TailTreeModel,
    utility_models: Mapping[...],
    observations: pl.DataFrame,
    outcome_horizon: int,
) -> pl.DataFrame
```

This can initially be implemented with current `score_bucket_evidence_frame` machinery, but the graph should distinguish semantic output columns from raw model scores.

### 7. Horizon action panel

Proposed owner:

```text
qooi.scanner.tailrun.selection
```

Reason:

```text
selection owns candidate replay, budget comparison, calibrated margins, and HPO/selection feedback.
A horizon panel is a selection/policy product, not a model internals product.
```

Proposed public calls:

```text
qooi.scanner.tailrun.selection.horizon_action_panel_frame(
    candidate_predictions: pl.DataFrame,
    *,
    entry_horizons: tuple[int, ...],
    confirmation_horizons: tuple[int, ...],
    risk_horizons: tuple[int, ...],
    min_clean_probability: float,
    min_utility_margin: float,
    max_chop_probability: float,
) -> pl.DataFrame

qooi.scanner.tailrun.selection.action_policy_frame(
    horizon_panel: pl.DataFrame,
) -> pl.DataFrame
```

Horizon panel grain:

```text
symbol × decision_bar_close_ms × action_side
```

Columns:

```text
symbol
decision_bar_close_ms
action_side: up | down
entry_horizon
max_valid_horizon
clean_horizon_count
contradicting_horizon_count
chop_horizon_count
best_utility_margin
mean_clean_probability
max_chop_probability
blocker_reason
actionability
```

Suggested actionability values:

```text
trade_candidate
short_horizon_only
volatility_watch
reversal_watch
gray_zone
no_action
```

Policy examples:

```text
trade_candidate:
  clean probability strong on entry and confirmation horizons
  expected utility margin positive
  chop/reversal probability under cap

short_horizon_only:
  clean probability strong on entry horizon
  longer risk horizon has chop/reversal blocker

volatility_watch:
  tail_any / chop probability high
  clean side probability insufficient

reversal_watch:
  down_first_both or up_first_both probability high
  opposite/rebound behavior dominates directional continuation

gray_zone:
  both/chop high or calibrated side margin <= 0
```

## API graph: proposed target

### Shared scanner workflow

```text
scripts/scanner_potential.py
  -> qooi.scanner.workflow.run(config_path)
     -> load_config
     -> resolve universe
     -> load_market
     -> qooi.scanner.state.classify_states
     -> qooi.scanner.state.extract_continuous_features
     -> qooi.scanner.state.potential_observation_frame
     -> qooi.scanner.outcome.path_histories
     -> qooi.scanner.outcome.realized_transition_frame
     -> qooi.scanner.outcome.source_outcomes_frame
     -> qooi.scanner.outcome.potential_outcome_frame
     -> evidence dispatch
        ladder path
        tailtree path
     -> qooi.scanner.rank.candidate_metric_surface
     -> qooi.scanner.rank.rank_candidates
     -> qooi.scanner.output.review_decisions
     -> qooi.scanner.output.render_report
```

No change to the outer dispatch is required first.

### Tailtree path-behavior graph

```text
qooi.scanner.tailrun.core.run_tailtree(
    TailtreeInputFrames(observations, source_outcomes, realized, histories),
    config=config,
    profile=profile,
)
  -> prepare frames / folds
  -> qooi.scanner.tailtree.model.label_tail_exceedances(...)          # keep current excursion labels
  -> qooi.scanner.tailtree.behavior.label_path_behavior(...)          # new behavior labels
  -> qooi.scanner.tailtree.utility.path_utility_frame(...)            # explicit utility labels
  -> qooi.scanner.tailtree.behavior.path_behavior_distribution_frame(...)
  -> for each profile/fold/horizon:
       -> qooi.scanner.tailtree.model.path_behavior_training_frame(...)
       -> TailTreeModel.train(... behavior head ...)
       -> qooi.scanner.tailtree.model.utility_training_frame(...)
       -> TailTreeModel.train(... utility/side head ...)
       -> qooi.scanner.tailtree.evidence.path_behavior_evidence_frame(...)
  -> qooi.scanner.tailrun.selection.score_bucket_candidate_frame(...)
  -> qooi.scanner.tailrun.selection.paired_candidate_replay_frame(...)
  -> qooi.scanner.tailrun.selection.calibrated_candidate_replay_frame(...)
  -> qooi.scanner.tailrun.selection.horizon_action_panel_frame(...)
  -> qooi.scanner.tailrun.selection.action_policy_frame(...)
  -> TailtreeRunOutput(evidence, models, profile_runs, selection_efficiency)
```

### Artifact graph

Current artifacts retained:

```text
report.md
tailtree-profile-runs.csv
tailtree-label-distribution.csv
tailtree-selection-efficiency.csv
models/*.json
profile/*.csv
```

New proposed diagnostic artifacts:

```text
tailtree-path-behavior-distribution.csv
tailtree-selected-path-behavior.csv
tailtree-horizon-action-panel.csv
```

Artifact grains:

```text
tailtree-path-behavior-distribution.csv:
  outcome_horizon × path_state

tailtree-selected-path-behavior.csv:
  outcome_horizon × selected_direction × score_bucket × path_state

tailtree-horizon-action-panel.csv:
  symbol × decision_bar_close_ms × action_side
```

### Report graph

Report should show one canonical candidate/action surface, not overlapping readiness/source/horizon sections.

Proposed report sections:

```text
Tailtree Behavior Summary
  -> path_state distribution by horizon

Tailtree Selection Behavior
  -> selected candidate path_state distribution by direction/bucket

Candidate Action Panel
  -> symbol, entry_horizon, action_side, actionability, path_state_profile, blocker_reason

Watches / Blockers
  -> volatility_watch, reversal_watch, gray_zone rows
```

Renderer remains presentational. Semantic joins and policy labels belong in `tailrun.selection` or a projection owner, not in markdown rendering.

## Config shape

Keep root workflow config as the TOML parse boundary. Do not create a new root config tree.

Add behavior/policy knobs under the existing tailtree section only when needed:

```toml
[potential.evidence.tailtree.behavior]
path_states = ["none", "clean_up", "clean_down", "up_first_both", "down_first_both", "chop_both"]
utility_floor = 0.0
utility_margin_floor = 0.0
path_efficiency_floor = 0.0
late_bar_ratio = 0.75

[potential.evidence.tailtree.horizon_policy]
entry_horizon = [6, 12]
confirmation_horizon = [24]
risk_horizon = [48]
min_clean_probability = 0.5
min_utility_margin = 0.0
max_chop_probability = 0.25
```

Do not add these until the diagnostic artifact proves which knobs matter.

## Training strategy

### Phase 1: diagnostic-only

Implement labels/artifacts but do not change HPO winner selection.

Purpose:

```text
Find whether current selected down candidates are clean_down, down_first_both, or chop_both.
```

Acceptance:

```text
tailtree-path-behavior-distribution.csv exists
tailtree-selected-path-behavior.csv exists
report exposes up/down behavior profile
no model objective switch yet
```

### Phase 2: behavior heads

Train path-state heads per horizon.

Start with binary heads rather than one complex multiclass if that is simpler:

```text
clean_up
clean_down
both_or_chop
up_first_both
down_first_both
```

Reason:

```text
class imbalance is severe; independent heads make rare states visible and inspectable.
```

### Phase 3: utility heads

Train utility/utility-margin predictors per horizon and side.

Do not mix execution/cost/slippage/funding into scanner utility. Those remain downstream concerns.

### Phase 4: horizon action policy

Use calibrated behavior + utility outputs to produce actionability:

```text
trade_candidate
short_horizon_only
volatility_watch
reversal_watch
gray_zone
no_action
```

Do not require all horizons to agree. Use horizon roles.

## Example behavior interpretation

### Case A: clean short entry but rebound risk

Predicted profile:

```text
6h:  p_clean_down high, utility_down positive
24h: p_down_first_both high
48h: p_chop_both high
```

Output:

```text
action_side = down
actionability = short_horizon_only or reversal_watch
entry_horizon = 6
blocker_reason = longer_horizon_rebound_chop
```

Do not output simple `short_candidate` for swing horizon.

### Case B: clean multi-horizon long

Predicted profile:

```text
6h:  p_clean_up high
24h: p_clean_up high
48h: p_chop_both low
```

Output:

```text
action_side = up
actionability = trade_candidate
entry_horizon = 6 or 24
max_valid_horizon = 24 or 48
```

### Case C: pure volatility event

Predicted profile:

```text
24h: p_chop_both high
48h: p_chop_both high
clean side probabilities weak
```

Output:

```text
action_side = none
actionability = volatility_watch or gray_zone
blocker_reason = no_clean_side
```

## Boundary rules

Allowed dependencies:

```text
state -> no future outcome columns
outcome -> no model internals
tailtree.behavior -> outcome path columns only
tailtree.utility -> behavior/outcome path columns only
tailtree.model -> known-at-close features + label frames
tailrun.selection -> model evidence/candidate predictions + policy config
output/report -> typed frames/projections only
```

Forbidden:

```text
state importing outcome/model labels
outcome importing tailtree model code
report computing path_state or actionability from raw columns
rank/report reading generated CSVs back as internal transport
model input feature selection by "column exists" without persistent data contract
raw score averaging across horizons
single global up/down final label without horizon profile
```

## Migration slice recommendation

1. Keep current `tail_up`, `tail_down`, `tail_any`, `tail_both`, `tail_state` as excursion facts.
2. Add path behavior diagnostics from existing outcome columns.
3. Write behavior distribution and selected behavior artifacts.
4. Add report section that exposes behavior by side/horizon.
5. Only then add behavior/utility prediction heads.
6. Only after calibrated evidence is stable, add horizon action panel and final actionability policy.

## Open questions to settle before coding model heads

1. Should first implementation add shorter horizons such as 6/12, or classify only current `[24, 48]` first?
2. Does `clean_up/down` require utility above floor, or only threshold touch without opposite touch?
3. What default makes `chop_both`: both-touch alone, weak utility dominance, low path efficiency, or all of them?
4. Should `down_first_both` default to `reversal_watch` rather than down candidate?
5. Should final candidate table allow `short_horizon_only` as promote, or keep it as watch until enough backtest evidence exists?

## Proposed decision for now

Implement no model-head change until diagnostics prove behavior distribution.

First target:

```text
PathBehavior diagnostics + selected behavior artifact + report panel
```

Then evaluate:

```text
Are selected down rows mostly clean_down, down_first_both, or chop_both?
Does side utility dominance agree with path_state?
Which horizons are clean vs contaminated?
```

Only after those answers should we change training objectives or final candidate promotion.
