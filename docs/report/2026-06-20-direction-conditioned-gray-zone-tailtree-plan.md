# Direction-conditioned gray-zone tailtree experiment plan

## Goal

Solve the gray-zone problem at the model/evaluation level without bloating code.

Current issue:

```text
independent up model + independent down model
=> same symbol can score materially on both sides
=> promotion needs abstention after the fact
```

We want to test whether a direction-conditioned model improves clean directional selection.

## Ponytail scope

Do the smallest viable experiment:

```text
1. add one objective name
2. add one candidate_direction categorical feature inside the training frame
3. train the existing LightGBM binary event-lift path on direction-expanded rows
4. keep using the existing selection-efficiency artifact
5. add gray-zone metrics to that same artifact only if cheap
```

Do not add:

```text
new model family
new artifact family
new long-range bar features
new strategy/executor concepts
large diagnostics tables
```

## Model design

Existing event-lift trains separate models:

```text
up model:   row -> tail_up
 down model: row -> tail_down
```

Experiment trains one direction-conditioned surface by expanding rows:

```text
row, candidate_direction=up   -> tail_up
row, candidate_direction=down -> tail_down
```

The model receives:

```text
candidate_direction
existing categorical states
existing normalized continuous features
```

No new trend geometry. Trend context is already represented by existing multi-time-scale classifier state fields and bar/source normalized fields.

## Objective name

Add:

```text
directional_event_lift
```

It uses the same binary LightGBM path as `tail_event_lift`, but trains on direction-expanded rows.

## Prediction/evidence shape

To avoid broad rewrite, still write direction-specific artifacts:

```text
up artifact
 down artifact
```

But both are trained from the same direction-expanded frame, with `candidate_direction` set to the artifact side during prediction. This keeps downstream evidence/report code mostly unchanged.

## Gray-zone metrics

Keep current promotion abstention.

Add cheap same-symbol conflict metrics at review/report level, already available from ranked rows:

```text
material_conflict_symbols
promoted_conflicts
conflict_watches
```

For benchmark feasibility, inspect:

```text
selection-efficiency top rows
report promoted_conflicts == 0
gray-zone/watch count
clean promoted top_3 quality
```

## Benchmark comparison

Compare against the current fast/named + abstention baseline:

```text
data/output/potential/benchmarks/conflict-abstain-tailtree-selection-efficiency.csv
```

New snapshots:

```text
data/output/potential/benchmarks/directional-grayzone-tailtree-selection-efficiency.csv
data/output/potential/benchmarks/directional-grayzone-report.md
```

Keep if:

```text
promoted_conflicts == 0
best/top_1pct hpo_score and valid_tail_lift are flat or better
utility is not materially worse
clean promoted rows look less gray-zone-heavy
```

Revert if:

```text
selection-efficiency worsens materially
or model quality collapses
or implementation grows beyond the narrow objective path
```

## Verification

```bash
uv run --no-sync python -m ruff check src tests scripts/scanner_backfill.py scripts/scanner_potential.py
uv run --no-sync python -m ty check
uv run --no-sync python -m pytest tests/test_scanner_workflow_migration.py tests/test_state.py -q
uv run --no-sync python -u scripts/scanner_potential.py --config configs/potential-advanced-tailtree.toml
```

## Revision after grill: do not train doubled directions when direction is modeled

User feedback:

```text
We already modeled direction into training. Why still train different directions in double?
```

Accepted. The prior narrow plan kept up/down artifacts to minimize code changes,
but that contradicts the premise of a direction-conditioned model and explains
why runtime did not improve.

### Revised model shape

Do not train:

```text
horizon × direction artifact
```

Train:

```text
horizon artifact with paired direction scoring
```

Training rows may still be expanded:

```text
observation × candidate_direction_code
```

but the artifact is shared. At prediction time, score the same observation twice:

```text
score_up   = model(observation, candidate_direction_code=+1)
score_down = model(observation, candidate_direction_code=-1)
```

Then emit one paired row:

```text
symbol
time
horizon
score_up
score_down
best_direction
directional_margin
conflict_ratio
same_horizon_conflict
```

### Revised objective

Raw binary event lift is insufficient. The objective must tune direction quality:

```text
directional_objective =
    best_side_lift
  + best_side_utility
  + log1p(best_tail_count)
  - selected_rate
  - lambda_opposite * opposite_side_lift
  - lambda_gray * gray_zone_rate
  - lambda_false * false_direction_cost
```

Direction is important because a false direction makes the trade astray. So HPO
should not only ask:

```text
did the selected side have event lift?
```

It must ask:

```text
did the selected side dominate the opposite side enough to trade?
```

### Revised runtime expectation

Previous hybrid experiment:

```text
2 trials × 2 folds × 2 horizons × 2 directions = 16 artifacts
```

Revised shared paired-direction experiment:

```text
2 trials × 2 folds × 2 horizons × 1 shared artifact = 8 artifacts
```

The training frame may be wider/taller, but artifact count should be halved. If
runtime is not materially lower or quality does not improve, reject it.

### Next grill checklist before coding

Before another code pass, answer:

```text
1. Can current TailTreeModel metadata represent one shared artifact whose logical direction is paired, not up/down?
2. Can evidence generation produce paired score buckets keyed by score_up/score_down without duplicating downstream report code?
3. Should same-horizon conflict be hard abstain while cross-horizon conflict is only penalized?
4. What false_direction_cost can be computed from current labeled outcomes without executor assumptions?
5. What benchmark baseline and acceptance thresholds are enough to justify replacing independent event-lift?
```

No coding should continue until this paired artifact/objective surface is settled.

## Revision after grill: train direction + return threshold + utility

User feedback:

```text
Train direction and return threshold, then model utility in the objective.
For code problems, follow ponytail.
For design problems, decide from goal and theory base.
Avoid duplication.
```

Accepted.

### Target product

The shared paired-direction model should not merely predict `event/no-event`.
It should model the trade-relevant tuple:

```text
candidate_direction
return_threshold_pct
utility_of_selected_direction
```

Direction and threshold are part of the prediction/evaluation surface because a
correct scanner candidate is not just:

```text
some tail event occurs
```

It is:

```text
selected direction reaches the selected return threshold with acceptable utility,
while opposite direction risk is not material.
```

### Training row shape

Use one shared artifact and avoid duplicated direction artifacts:

```text
observation × candidate_direction × return_threshold_pct
```

Minimal threshold grid for first pass:

```text
30.0
```

Do not introduce a broad threshold sweep until the single-threshold paired model
is feasible. Later threshold grid can be:

```text
20.0, 30.0, 40.0
```

but only if the first pass wins.

### Labels / utility

For each expanded row:

```text
label_event = selected_direction_path_return >= return_threshold_pct
utility = directional tail utility already computed for up/down
opposite_event = opposite_direction_path_return >= return_threshold_pct
```

The first implementation should reuse existing `tail_utility_up/down` and
`tail_up/down` outcomes. Do not add executor assumptions.

### Objective surface

The HPO/selection objective should include utility and direction correctness:

```text
objective =
    selected_side_lift
  + selected_side_utility
  + log1p(selected_tail_count)
  - selected_rate
  - lambda_opposite * opposite_side_lift
  - lambda_gray * gray_zone_rate
  - lambda_false * false_direction_cost
```

Where:

```text
gray_zone = selected side and opposite side both material
false_direction_cost = selected direction loses to opposite realized tail
```

### Design rule

If the issue is code structure:

```text
ponytail: smallest path, delete duplication, benchmark, revert if worse
```

If the issue is design ambiguity:

```text
choose by scanner goal + theory base
```

Scanner goal:

```text
promote clean directional extreme opportunities, not generic volatility.
```

Theory base:

```text
direction correctness and threshold-conditioned utility are first-class;
both-tail gray zone is watch/abstain unless directional dominance is proven.
```

### Duplication guard

Do not maintain parallel implementations for:

```text
independent direction models
direction-conditioned artifact models
paired-direction shared model
```

The next code pass should be a spike or isolated branch path. If benchmark wins,
collapse to the winning path. If it fails, revert it and keep the current
`tail_event_lift + conflict abstention` baseline.
