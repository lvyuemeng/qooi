# Tailtree training workflow redesign

## Goal

Clarify and redesign the scanner tailtree training workflow so it is adaptable across model/objective families without deep nesting, opaque types, or silent metric divergence.

This is a design report only. No code changes are proposed here beyond the implementation plan.

## User feedback absorbed

Key feedback:

```text
Clarify the workflow adaptable for different model/objective training.
Avoid deep nesting and opaque types by ponytail.
Metrics should be adaptable for various objectives.
In the previous pass, HPO changed and diverged in behavior.
Reflect on that and redesign the training workflow.
```

Accepted.

## Current architecture boundary

From `docs/architecture/scanner.md`:

```text
state.py   -> known-at-close observation/features
outcome.py -> future/path/source outcome rows
tailtree   -> model/evidence products
tailrun    -> lifecycle, profiles, Optuna, selection-efficiency
rank.py    -> comparable candidate surface
output.py  -> review gates/promote/watch
```

The scanner workflow is:

```text
config
-> load market data
-> state
-> outcome
-> tailtree evidence
-> rank
-> review/report
```

Therefore:

```text
known-at-close features stay in state/observation frames
future labels stay in outcome/labeled-outcome frames
objective slicing belongs in tailrun/tailtree training setup
selection/HPO metrics belong in tailrun replay artifacts
promotion gates belong in output.py
```

## Current training workflow, clarified

For current advanced `tail_event_lift`:

```text
for profile run:
  for fold:
    for horizon:
      for direction in up/down:
        build direction-specific labels
        train direction-specific binary event model
        score validation/current observations
        summarize evidence buckets
        append selection-efficiency rows
```

Actual label builder:

```text
_event_lift_training_values(observations, labeled_outcomes, direction)
```

For `direction = up`:

```text
label = tail_up
utility = tail_utility_up
```

For `direction = down`:

```text
label = tail_down
utility = tail_utility_down
```

Model target:

```text
binary selected-direction threshold event
```

Model features:

```text
known-at-close observation features only
```

No feature columns:

```text
candidate_direction_code ❌
return_threshold_pct ❌
```

## Why conflict still appears

The model loop trains two independent questions:

```text
Does this state precede +threshold within horizon?
Does this state precede -threshold within horizon?
```

Both can be true for a high-volatility coin/path:

```text
tail_up = true
tail_down = true
```

or both models can produce material evidence for the same symbol/horizon.

That is not a training contradiction. It means:

```text
the scanner found a both-tail volatility regime, not a clean directional trade
```

Conflict is therefore a candidate-selection problem:

```text
same symbol/time/horizon has material evidence on both sides
```

not a sign that binary up/down event training is wrong.

## Previous problem 1: direction/threshold as features

The paired direction-threshold utility pass added:

```text
candidate_direction_code
return_threshold_pct
```

as model features and trained:

```text
event * log1p(direction_utility)
```

This was wrong because direction and threshold are objective dimensions, not market/source features.

Correct order:

```text
direction -> threshold -> utility
```

Meaning:

```text
1. slice labels by direction
2. apply threshold to define binary event
3. train event concentration model
4. use utility in replay/HPO/selection
```

## Previous problem 2: aggregate HPO penalty changed behavior

The directional HPO replay penalty pass changed:

```text
hpo_score = base_score - opposite_quality - gray_zone_penalty
```

At aggregate score-bucket level:

```text
same horizon + same score bucket + opposite direction
```

That was too coarse. All selection-efficiency rows became gray-zone:

```text
64 / 64 rows
```

So valid tail lift barely moved:

```text
valid_tail_lift max: -1.90%
```

but HPO score collapsed:

```text
hpo_score max: -64.57%
```

because HPO was no longer comparable to the baseline metric. It became a penalty surface rather than a comparable model-quality score.

Reflection:

```text
Changing HPO semantics without preserving a base comparable score hides whether the model got worse or only the metric changed.
```

## Redesign principles

### 1. Separate stages explicitly

The workflow should read as named flat stages:

```text
prepare labels
plan objective jobs
train model
score observations
build candidate replay
score objective metrics
write artifacts
```

Avoid nested branching where each objective rewrites the whole loop.

### 2. Keep objective dimensions out of features

Direction and threshold are not features.

They belong to:

```text
label selection
objective scoring
candidate pairing
```

### 3. Keep metrics comparable

Every objective replay should emit:

```text
base_hpo_score
objective_hpo_score
```

Where:

```text
base_hpo_score = objective-family-neutral concentration/utility score
objective_hpo_score = model/objective-specific score used for optimization
```

The artifact may still keep `hpo_score`, but it should be clear whether it aliases base or objective score.

### 4. Pair at candidate level, not aggregate evidence level

Directional gray-zone metrics must be computed at:

```text
symbol + decision_bar_close_ms + horizon
```

not:

```text
horizon + score_bucket
```

### 5. Use explicit frame contracts, not opaque row probing

Prefer small typed/named DataFrame contracts:

```text
ObjectiveJobFrame
TrainingLabelFrame
EvidenceFrame
CandidateReplayFrame
SelectionMetricFrame
```

These can be simple functions and schema constants, not manager classes.

## Proposed flat workflow

### Stage A — prepare outcomes

Input:

```text
observations
source_outcomes
realized
```

Output:

```text
outcome_frame
labeled_outcome_frame
```

Existing functions:

```text
potential_outcome_frame(..., return_threshold_pct=config.transition.return_threshold_pct)
label_tail_exceedances(outcome_frame, threshold_pct=tailtree.threshold_pct)
```

Contract:

```text
symbol
decision_bar_close_ms
outcome_horizon
tail_up
tail_down
tail_utility_up
tail_utility_down
```

### Stage B — plan objective jobs

Build a flat job table, not nested objective-specific loops.

Columns:

```text
run_id
fold_id
outcome_horizon
direction
objective
model_tag
training_params
```

For current objective:

```text
objective = tail_event_lift
direction = up/down
label_col = tail_up/tail_down
utility_col = tail_utility_up/tail_utility_down
```

This makes different objectives adaptable:

```text
tail_event_lift -> binary label
tail_utility_quantile -> utility target after tail filter
tail_severity_gpd -> exceedance severity target
future objective -> explicit label/target adapter
```

### Stage C — build training frame

Function shape:

```text
training_frame = build_training_frame(observations, labeled_outcomes, objective_job)
```

Output columns:

```text
feature columns only from observations
target_value
utility_value
label_event
```

Important:

```text
direction and threshold do not enter feature columns
```

They control which label/target columns are chosen.

### Stage D — train model

Function shape:

```text
model = train_tailtree(training_frame, objective_job, feature_spec)
```

Input:

```text
feature columns
target_value
objective params
```

Output:

```text
TailtreeArtifactTree
```

The model layer should not know gray-zone or promotion logic.

### Stage E — score observations

Function shape:

```text
scored = score_tailtree(model, score_observations, objective_job)
```

Output:

```text
symbol
decision_bar_close_ms
outcome_horizon
direction
score
score_bucket/leaf_id
```

This is the bridge needed for candidate-level pairing.

Current evidence rows aggregate too early. We need a score/candidate frame before bucket summarization.

### Stage F — build evidence summaries

From scored + labeled outcomes:

```text
evidence = summarize_evidence(scored, labeled_outcomes, objective_job)
```

Output remains current evidence style:

```text
score_bucket
tail_lift
N_total
N_tail_exceedances
tail_utility_mean
```

This keeps existing report/rank surfaces working.

### Stage G — build paired candidate replay

New narrow frame:

```text
candidate_replay = pair_direction_candidates(scored_up, scored_down, labeled_outcomes)
```

Key:

```text
symbol + decision_bar_close_ms + outcome_horizon
```

Columns:

```text
score_up
score_down
lift_up_bucket
lift_down_bucket
utility_up
utility_down
tail_up
tail_down
selected_direction
opposite_direction
selected_quality
opposite_quality
directional_margin
gray_zone_flag
false_direction_cost
```

This is the correct place for gray-zone metrics.

### Stage H — selection/HPO metrics

Input:

```text
evidence summaries
candidate_replay frame
objective_job
selection policy
```

Output:

```text
selection_efficiency rows
```

Required score columns:

```text
base_hpo_score
objective_hpo_score
hpo_score
```

Recommendation:

```text
hpo_score = objective_hpo_score
```

but keep `base_hpo_score` for comparability.

For current direction-aware objective:

```text
base_hpo_score = selected_side_lift + selected_side_utility + log1p(selected_tail_count) - selected_rate

objective_hpo_score =
    base_hpo_score
  - lambda_opposite * paired_opposite_quality_mean
  - lambda_gray * paired_gray_zone_rate
  - lambda_false * paired_false_direction_cost_mean
```

Important: penalties come from candidate-level pair rows, not aggregate evidence rows.

### Stage I — rank/review

Current `output.review_decisions()` already applies:

```text
support threshold
tail_lift threshold
material opposite-direction conflict -> watch/abstain
```

Keep it. Do not move promotion semantics into training.

## Adaptability by objective family

A model objective should define only these adapters:

```text
label columns
training target expression
model objective params
evidence score column
base metric ingredients
objective metric ingredients
```

Example table:

| objective | label slice | model target | evidence | replay objective |
|---|---|---|---|---|
| tail_event_lift | direction + threshold | binary event | score bucket tail lift | lift + utility + paired direction penalties |
| tail_utility_quantile | tail-filtered utility | log utility / quantile | utility bucket | utility with support/lift gates |
| tail_severity_gpd | exceedance severity | severity/GPD | leaf tail params | severity + support + utility |

No objective should require feature hacks like:

```text
candidate_direction_code
return_threshold_pct
```

## Ponytail implementation plan

### Step 1 — clarify current function names only where necessary

Do not rewrite everything.

First code pass should extract small flat helpers around current `_train_profile_run()`:

```text
_objective_label_columns(direction)
_build_event_lift_training_frame(...)
_score_tailtree_observations(...)
```

No behavior change.

Verification:

```text
selection_efficiency identical or acceptably unchanged
```

### Step 2 — emit scored observation frame

Add a frame before aggregate evidence:

```text
profile.frame("scanner", "tailtree", f"scores_{label}", scored)
```

Required columns:

```text
symbol
decision_bar_close_ms
outcome_horizon
direction
score
score_bucket
```

No HPO change yet.

### Step 3 — build candidate-level paired replay frame

Add:

```text
_pair_direction_candidate_replay(scored_frames, labeled_outcomes)
```

Use key:

```text
symbol + decision_bar_close_ms + outcome_horizon
```

Output diagnostics first. No optimization change yet.

### Step 4 — add objective score columns without replacing baseline

Selection-efficiency should include:

```text
base_hpo_score
objective_hpo_score
paired_gray_zone_rate
paired_false_direction_cost_mean
paired_directional_margin_mean
```

At first:

```text
hpo_score = base_hpo_score
```

This prevents metric divergence.

### Step 5 — enable HPO objective switch only after diagnostics are sane

Only after paired replay metrics are verified:

```text
hpo_score = objective_hpo_score
```

Then benchmark.

If the new objective worsens top-bucket lift/utility or increases conflict watches, revert only the objective switch while keeping diagnostics if useful.

## Acceptance checks

For each pass:

```text
uv run --no-sync python -m ruff check src tests scripts/scanner_backfill.py scripts/scanner_potential.py
uv run --no-sync python -m ty check
uv run --no-sync python -m pytest tests/test_scanner_workflow_migration.py tests/test_state.py -q
```

For benchmark passes:

```text
uv run --no-sync python -u scripts/scanner_potential.py --config configs/potential-advanced-tailtree.toml
```

Keep only if:

```text
promoted_conflicts == 0
valid_tail_lift not materially worse
selected_profit_proxy_mean not materially worse
conflict_watches flat or lower
base_hpo_score remains comparable
objective_hpo_score improvement is explainable by paired candidate metrics
```

## Final recommendation

Do not redesign by adding model features or jumping to a new model family.

Redesign by making the workflow explicit and adding a candidate-level paired replay surface:

```text
objective job -> training frame -> model -> scored observations -> evidence summary -> paired candidate replay -> selection metrics
```

This preserves binary up/down concentration training while making direction/threshold/utility metrics adaptable and interpretable across objectives.
