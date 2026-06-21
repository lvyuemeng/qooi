# Tailtree workflow reduction redesign

## Why this report

The last implementation made progress by exposing candidate-level paired replay, but the coding style drifted toward incremental patching:

```text
train loop + scoring + evidence + paired replay + HPO metrics all live in one growing core.py path
```

That is the wrong direction for ponytail. A new feature should reduce the workflow into clearer stages, not keep adding branches and helper patches around a dense loop.

This report clarifies the current training workflow, explains the up/down asymmetry, evaluates the current objective against the scanner goal/theory base, and proposes a reduction-first redesign.

## Current training workflow, as code runs now

Current top-level tailtree flow:

```text
run_tailtree()
  build potential outcomes
  label tail exceedances
  choose training features
  build prepared frames
  for profile/fold/trial:
    _train_profile_run()
    _selection_efficiency_frame()
  concat evidence
  concat selection efficiency
```

Inside `_train_profile_run()`:

```text
for outcome_horizon in config.evidence.tailtree.outcome_horizon:
  for direction in ("up", "down"):
    build horizon+direction training frame
    train or load model
    score validation/current observations
    summarize evidence buckets
append all evidence
build paired candidate replay from score frames
return evidence, models, score, candidate_replay
```

Training target for `tail_event_lift`:

```text
up model:
  label = tail_up
  utility = tail_utility_up

down model:
  label = tail_down
  utility = tail_utility_down
```

Model features remain known-at-close observation columns only. Direction and threshold are not features.

## Current code problem

The current code has a correct idea but poor shape.

It mixes these roles in one file/function path:

```text
outcome slicing
fold planning
objective-specific label extraction
model lifecycle
score-frame construction
evidence summary
candidate replay construction
selection-efficiency scoring
Optuna feedback
```

That causes feature development to happen by patching the middle of the loop. Each patch adds another helper, another return value, or another special-case branch.

This is the complexity growth the user warned about.

## Reduction target

The workflow should be reduced to explicit data products, not expanded with wrappers or managers.

Flat products:

```text
ObjectiveJob
TrainingFrame
ScoredBucketFrame
EvidenceFrame
PairedReplayFrame
SelectionMetricFrame
```

But avoid opaque class bloat. These can be plain dataclasses only when they replace multiple ad-hoc return values. Otherwise, use direct `pl.DataFrame` products with stable column contracts.

The important reduction is:

```text
one stage owns one responsibility
```

not:

```text
one giant loop owns all side effects
```

## Current up/down asymmetry

From the paired replay test artifact:

```text
data/output/potential/paired-replay-test/tailtree-selection-efficiency.csv
```

Direction summary:

```text
down:
  max_lift = 54.120486
  max_hpo  = 57.231896
  max_utility = 1.350225
  mean_false_cost = 2.234210
  mean_margin = -0.220200
  mean_gray = 0.107432

up:
  max_lift = 30.913876
  max_hpo  = 36.892772
  max_utility = 2.783055
  mean_false_cost = 2.732152
  mean_margin = 0.357315
  mean_gray = 0.107436
```

Top-1 bucket examples:

```text
h24 down top_1pct:
  lift = 54.120486
  utility = 1.086564
  false_direction_rate = 0.413745
  false_direction_cost = 2.540712
  score_margin = -0.121536

h24 up top_1pct:
  lift = 30.913876
  utility = 2.713930
  false_direction_rate = 0.035073
  false_direction_cost = 0.850701
  score_margin = 0.449590

h48 down top_1pct:
  lift = 39.585147
  utility = 1.350225
  false_direction_rate = 0.515711
  false_direction_cost = 1.984043
  score_margin = -0.123379

h48 up top_1pct:
  lift = 20.768166
  utility = 2.783055
  false_direction_rate = 0.017990
  false_direction_cost = 8.348569
  score_margin = 0.412398
```

## Why up/down behave differently

The current base objective rewards event concentration:

```text
base_hpo_score = valid_tail_lift + utility_mean + sqrt(selected_tail_count) / 10
```

For down buckets, the down event is rarer:

```text
h24 down valid_tail_rate ~= 0.00447
h24 up   valid_tail_rate ~= 0.02040
h48 down valid_tail_rate ~= 0.00912
h48 up   valid_tail_rate ~= 0.03891
```

Because down tails are rarer, a selected down bucket can produce very high lift even when it is not clean directionally.

That explains:

```text
down lift high
but down false_direction_rate high
and down directional margin negative
```

The down model is identifying a volatile crash-risk regime. But that regime also often has opposite-side realizations or stronger opposite score evidence. So it is good at finding tail risk, not necessarily clean short opportunities.

For up buckets:

```text
up lift lower
up utility higher
up false-direction rate much lower at h24
up directional margin positive
```

The up model appears less concentrated by lift but cleaner directionally, especially h24.

This is a theory-compatible finding:

```text
rare downside events create high lift but are more often embedded in both-tail volatility;
upside tails may be less concentrated but more directional when selected.
```

## Why current objective is incomplete

The current objective is feasible for finding extreme-event concentration:

```text
features -> selected direction threshold event
```

It is not sufficient for the scanner goal:

```text
promote clean directional extreme opportunities
```

because it does not optimize dominance:

```text
selected side should dominate opposite side at symbol/time/horizon
```

Current base HPO ranks down buckets highest because lift dominates:

```text
h24 down top_1pct base_hpo = 57.23
h24 up   top_1pct base_hpo = 36.89
```

But paired replay says:

```text
h24 down false_direction_rate = 0.414, margin = -0.122
h24 up   false_direction_rate = 0.035, margin =  0.450
```

So base HPO and scanner goal diverge.

## Objective feasibility verdict

Current `tail_event_lift` is feasible as a base training objective.

Keep:

```text
binary up/down threshold-event concentration training
```

Do not replace it with utility regression. Do not add direction/threshold features.

But current base HPO is not feasible as the final promotion/search objective because it optimizes:

```text
extreme-event concentration
```

not:

```text
directional dominance after threshold qualification
```

The replay diagnostics now grasp the missing demand, but only as diagnostics.

Therefore the feasible direction is:

```text
keep training objective
reduce workflow
move objective scoring to candidate-paired replay
```

not:

```text
add more model families
add advanced config runs
patch aggregate HPO penalties
```

## Why the previous replay patch was closer but still wrong shape

The paired replay diagnostic surface is correct. The implementation shape is not.

Good:

```text
candidate-level unit = symbol + decision_bar_close_ms + horizon + score_bucket
hpo_score == base_hpo_score
objective_hpo_score diagnostic only
```

Bad:

```text
_train_profile_run returns evidence, models, score, candidate_replay
core.py owns too many products
selection metrics pull from evidence row dicts and replay frame ad hoc
score bucket reconstruction duplicates evidence bucket logic
```

This is patch-growth, not reduction.

## Reduction-first redesign

### Stage 1: plan objective jobs

Replace nested inline loops with a flat job list:

```text
ObjectiveJob:
  run
  fold_id
  outcome_horizon
  direction
  model_path
```

This removes nested branch pressure from the train loop.

The code becomes:

```text
jobs = tailtree_objective_jobs(config, profile_runs, folds)
for job in jobs:
  result = run_objective_job(job, frames)
```

### Stage 2: one job -> one artifact product

One job should produce:

```text
TailtreeJobResult:
  evidence: pl.DataFrame
  scores: pl.DataFrame
  model: TailtreeArtifactTree | None
  score: float
```

This replaces the current multi-return from `_train_profile_run()` and reduces loop-local mutation.

### Stage 3: build replay once from job score products

After all jobs in one run/fold:

```text
score_frame = concat(job.scores)
paired_replay = paired_candidate_replay(score_frame)
evidence = concat(job.evidence)
selection = selection_metrics(evidence, paired_replay)
```

This keeps replay out of model training.

### Stage 4: one scoring contract

Selection metrics should always emit:

```text
base_hpo_score
objective_hpo_score
hpo_score
```

For now:

```text
hpo_score = base_hpo_score
```

Later objective switch is one line/one config field, not a hidden formula change.

### Stage 5: objective scorer is a small flat function

Do not create strategy classes.

Use one function:

```text
directional_objective_score(row, replay_metrics) -> float
```

First theory-based formula:

```text
objective_hpo_score =
    base_hpo_score
  + utility_weight * selected_utility_mean
  + margin_weight * paired_directional_margin_mean
  - gray_weight * paired_gray_zone_rate
  - false_rate_weight * paired_false_direction_rate
  - false_cost_weight * paired_false_direction_cost_mean
```

But do not switch `hpo_score` yet.

## Theory-based scoring implication from current data

The current diagnostic data suggests false-direction **rate** matters more than gray-zone rate.

Gray-zone rate is similar by direction:

```text
down mean_gray ~= 0.107432
up   mean_gray ~= 0.107436
```

So gray-zone alone cannot distinguish up/down.

False-direction rate and margin distinguish much better:

```text
h24 down false_rate = 0.414, margin = -0.122
h24 up   false_rate = 0.035, margin =  0.450
```

Therefore the next objective should focus on:

```text
false_direction_rate
paired_directional_margin_mean
false_direction_cost_mean
```

not just:

```text
gray_zone_rate
```

This also explains why the earlier aggregate gray-zone penalty failed: it penalized a metric that was not discriminating at the right level.

## Goal-based decision

Scanner goal:

```text
promote clean directional extreme opportunities, not generic both-tail volatility
```

Current base HPO chooses:

```text
highest lift even if opposite side is material
```

The paired replay objective should choose:

```text
sufficient lift + positive directional margin + low false-direction rate
```

This may rank a lower-lift up bucket above a higher-lift down bucket if the up bucket is much cleaner.

That is acceptable if the scanner is a promotion/research watch system rather than a pure tail-risk detector.

## Immediate code action recommended

Do not add advanced run now.

Do not switch HPO now.

First reduce the current implementation:

```text
1. Extract objective job planning from run_tailtree().
2. Extract one job runner from _train_profile_run().
3. Make score/evidence/replay/selection explicit products.
4. Remove ad-hoc replay return from _train_profile_run().
5. Keep paired replay diagnostics and current test config.
```

Then run the same small test config to verify behavior is unchanged:

```text
hpo_score == base_hpo_score
candidate_replay rows present
selection-efficiency paired columns present
same promoted/conflict count or explain drift
```

Only after reduction is complete should we modify the objective formula.

## Proposed reduced file/API shape

Stay in existing files unless code shrinks significantly.

Suggested boundaries:

```text
tailrun/core.py
  run_tailtree orchestration only
  objective job loop only

tailrun/selection.py
  score bucket candidate frame
  paired candidate replay
  selection metrics
```

Why move replay to `selection.py`:

```text
paired replay is selection/HPO surface, not training lifecycle
```

This reduces `core.py` and keeps objective metrics near existing selection-efficiency functions.

## Revert/keep recommendation

Keep the diagnostic concept.

Do not keep the current shape long-term.

The next coding pass should be a reduction pass, not a feature pass:

```text
move paired replay helpers out of core.py
flatten job execution
preserve output exactly
```

Acceptance for the reduction pass:

```text
- fewer responsibilities in core.py
- no change to model artifacts
- no change to hpo_score behavior
- same paired diagnostic columns
- same/similar small-config benchmark outputs
- checks pass
```

## Bottom line

Current replay diagnostics grasp our demand better than previous attempts.

The new insight is:

```text
down = high lift, high false-direction exposure, negative margin
up   = lower lift, cleaner direction, positive margin
```

So current base objective is useful but incomplete. It should remain the training objective, while the promotion/search objective should become candidate-level direction-dominance aware.

But before changing HPO behavior, reduce the workflow so future objective work is one clean scoring function, not another patch inside a growing loop.
