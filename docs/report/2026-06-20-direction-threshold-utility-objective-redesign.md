# Direction → threshold → utility objective redesign

## Correction

The previous paired-direction utility implementation misunderstood the design.

It added:

```text
candidate_direction_code
return_threshold_pct
```

as model input features. That is wrong for this scanner architecture.

Direction and threshold are not known-at-close market/source features. They are objective/label-selection dimensions. They belong in outcome/evaluation/HPO logic, not in `state.py` feature columns or LightGBM input columns.

## Architecture consistency

Scanner architecture says:

```text
config -> load market data -> state -> outcome -> tailtree evidence -> rank -> review/report
```

Ownership:

```text
state.py   = known-at-close observation/features
outcome.py = future/path/source labels
rank.py    = comparable candidate surface
output.py  = review gates/promote/watch
```

Therefore:

```text
candidate direction = label/objective slice
return threshold    = label/objective slice
utility             = objective/evaluation weight
```

They must not be modeled as ordinary market features.

## User feedback absorbed

User correction:

```text
Do not model return_threshold_pct and direction in features.
Previously we first filter threshold then concentrate on utility.
We should do the same: first direction, then threshold, then utility.
```

Accepted.

## Correct theory base

The scanner goal is not generic volatility prediction.

It is:

```text
promote clean directional extreme opportunities
```

So the objective order is:

```text
1. direction correctness
2. threshold event concentration
3. utility among threshold-qualified directional events
```

This mirrors the earlier good pattern:

```text
filter to threshold-qualified tails first
then use utility to rank/score quality
```

The corrected design is not:

```text
model(direction, threshold as features) -> utility regression
```

It is:

```text
for each direction:
  build labels from that direction only
  apply threshold to define event
  train event concentration model
  score utility in replay/HPO over selected event buckets
```

## What remains from current baseline

Current baseline is already close to the corrected architecture:

```text
tail_event_lift
```

because `_event_lift_training_values()` already does:

```text
direction -> tail_col = tail_up/tail_down
threshold -> tail_up/tail_down labels from outcome.py threshold_pct
utility   -> available as tail_utility_up/tail_utility_down but not central to train target
```

Current flaw is not that direction/threshold are missing from features.

Current flaw is:

```text
up/down models are optimized independently,
while HPO/selection does not sufficiently penalize opposite-side evidence or false direction.
```

## Correct ponytail improvement

Do not add features.

Do not create a new broad model family first.

Patch the selection/HPO replay objective so existing direction-specific event models are judged by paired-direction quality.

### Keep model training target

Keep binary event concentration:

```text
selected_direction_event_at_threshold
```

This means training remains:

```text
features = known-at-close observation features only
label    = tail_up or tail_down from outcome.py thresholded path labels
```

No feature columns:

```text
candidate_direction_code ❌
return_threshold_pct ❌
```

### Add paired replay metrics

For each evidence row / candidate surface, compute opposite-side materiality from the existing opposite direction evidence rows:

```text
selected_side_lift
selected_side_utility
opposite_side_lift
opposite_side_utility
directional_margin
conflict_ratio
gray_zone_flag
false_direction_cost proxy
```

Start in `tailrun/core.py::_selection_efficiency_frame()` or the selection artifact path, because that is where HPO feedback is produced.

### Corrected HPO/replay score

Replace or augment current HPO score:

```text
hpo_score = valid_tail_lift + utility_mean + sqrt(selected_tails) / 10
```

with a directional score:

```text
directional_hpo_score =
    selected_side_lift
  + selected_side_utility
  + log1p(selected_tail_count)
  - selected_observation_rate
  - lambda_opposite * opposite_side_lift
  - lambda_gray * gray_zone_rate
  - lambda_false * false_direction_cost
```

First ponytail constants:

```text
lambda_opposite = 1.0
lambda_gray = 5.0
lambda_false = 1.0
```

Only keep constants if benchmark improves; otherwise revert.

## Same-horizon vs cross-horizon conflict

Do not collapse all conflicts into one bucket forever.

First pass should focus on same-horizon conflict because it is the cleanest theory signal:

```text
h24 up vs h24 down
h48 up vs h48 down
```

Same-horizon conflict means:

```text
same holding window, both tails material -> direction unreliable
```

Cross-horizon conflict is less decisive:

```text
h24 up vs h48 down
```

and may represent different time-structure. For ponytail first pass:

```text
same-horizon conflict penalty/hard abstain
cross-horizon conflict report-only or weaker penalty
```

## Implementation boundary

Touch only the smallest surfaces:

```text
src/qooi/scanner/tailrun/core.py        # selection/HPO score rows
src/qooi/scanner/output.py              # already does promotion abstention; only adjust if needed
configs/potential-advanced-tailtree.toml # only if objective name/config selector is necessary
```

Avoid:

```text
state.py feature changes
outcome.py label changes unless needed for missing utility fields
model.py new feature columns
rank.py prediction feature hacks
new artifact family
```

## Expected code shape

Prefer flat functions:

```text
_opposite_direction(direction)
_directional_quality(row)
_pair_direction_evidence(evidence)
_directional_hpo_score(row)
```

No manager classes.

No wrapper modules.

No duplicate model path.

## Benchmark acceptance

Compare against:

```text
data/output/potential/benchmarks/conflict-abstain-tailtree-selection-efficiency.csv
```

Keep only if:

```text
promoted_conflicts == 0
same_horizon_conflict promoted count == 0
hpo_score / directional_hpo_score improves or is flat
valid_tail_lift is not materially worse
selected_profit_proxy_mean is not materially worse
conflict_watches do not increase materially
```

If the score worsens, revert.

## Revised conclusion

The failed paired utility regression did not fail because shared artifacts are impossible.

It failed because it violated the design boundary:

```text
direction/threshold became features
utility became model target
```

The next correct ponytail path is:

```text
keep direction-specific threshold-event training
add direction-aware utility/opposite-side penalties to HPO/replay/selection
benchmark
keep only if it improves clean directional promotion
```
