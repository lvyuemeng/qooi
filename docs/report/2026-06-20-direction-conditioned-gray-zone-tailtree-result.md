# Direction-conditioned gray-zone tailtree benchmark result

## Experiment

Plan:

```text
docs/report/2026-06-20-direction-conditioned-gray-zone-tailtree-plan.md
```

Objective tested:

```text
directional_event_lift
```

Training shape:

```text
observation × candidate_direction_code
candidate_direction_code = +1 for up, -1 for down
```

This was the minimal one-direction / direction-conditioned model experiment.

## Important implementation finding

The first attempt used string categorical `candidate_direction`.

That was flawed because prediction calls score one direction at a time, and Polars categorical encoding can map both single-category prediction frames to code `0` separately.

Fix applied before the final benchmark:

```text
candidate_direction_code = +1.0 / -1.0
```

The final benchmark below is the stable numeric-code run.

## Benchmark artifacts

```text
data/output/potential/benchmarks/directional-code-grayzone-report.md
data/output/potential/benchmarks/directional-code-grayzone-tailtree-selection-efficiency.csv
```

Baseline:

```text
data/output/potential/benchmarks/conflict-abstain-report.md
data/output/potential/benchmarks/conflict-abstain-tailtree-selection-efficiency.csv
```

Runtime:

```text
directional-code: 771s
```

Baseline conflict-abstain run was materially faster.

## Selection-efficiency comparison

| metric | conflict-abstain baseline | directional-code | delta |
|---|---:|---:|---:|
| hpo_score max | 63.687865 | 60.733573 | -4.64% |
| valid_tail_lift max | 60.037224 | 54.729814 | -8.84% |
| selected_profit_proxy_mean max | 2.662485 | 2.725273 | +2.36% |
| selected_profit_proxy_p90 max | 9.188335 | 9.224663 | +0.40% |
| selected_tail_count max | 6600 | 6735 | +2.05% |
| selected_tail_per_1k_obs max | 794.426679 | 755.596163 | -4.89% |

Best baseline row:

```text
h24 down top_1pct
hpo=63.687865
lift=60.037224
utility_mean=1.335474
tails=535
```

Best directional-code row:

```text
h24 up top_1pct
hpo=60.733573
lift=54.729814
utility_mean=2.329525
tails=1349
```

## Gray-zone comparison

Baseline report:

```text
promoted=3
conflict_watches=8
promoted_conflicts=0
```

Directional-code report:

```text
promoted=3
conflict_watches=11
promoted_conflicts=0
```

The direction-conditioned model increased gray-zone watches rather than reducing them.

## Feature importance

The model did use direction code:

```text
candidate_direction_code rank ~= 6 by gain
```

Top features included:

```text
bar_return_24h_pct
funding_rate_raw
decision_transition
bar_return_24h_per_vol_7d
bar_return_4h_pct
candidate_direction_code
```

So this was not a wiring miss.

## Feasibility verdict

Not feasible to keep.

Reason:

```text
- HPO worsened
- event concentration/lift worsened
- gray-zone count increased
- runtime increased materially
```

The small utility improvement does not compensate for worse concentration and more ambiguity.

## Why it likely failed

The current direction-conditioned implementation gives the model a direction flag, but it does not create an explicit dominance objective.

It still optimizes binary event probability per expanded row:

```text
candidate_direction + state -> event/no-event
```

That can improve broad utility while still ranking both directions highly for volatile both-tail regimes.

In other words, the model learned:

```text
this state is high-event under a direction flag
```

but did not learn:

```text
one side must dominate the other side
```

The gray zone is therefore not solved by adding direction to event-lift alone.

## Ponytail action

Revert direction-conditioned code/config.

Keep:

```text
tail_event_lift independent up/down model
conflict abstention at promotion
```

Next improvement should be smaller and directly target dominance on the existing evidence surface:

```text
compute directional dominance = best_side_quality - opposite_side_quality
benchmark dominance gates/penalties in selection/report
```

Only if dominance-gated replay succeeds should we revisit model-level training with a pairwise/dominance objective.

## Addendum: user feedback absorbed — direction modeled but still trained twice

The strongest design criticism is valid:

```text
If candidate direction is already inside the training row, why are we still
training separate up/down artifacts?
```

The benchmarked implementation was not a true model reduction. It was a hybrid:

```text
direction-expanded training frame
wrapped inside the old up/down artifact loop
```

So the run still trained:

```text
2 trials × 2 folds × 2 horizons × 2 directions = 16 artifacts
```

and each artifact saw the expanded row shape:

```text
observation × candidate_direction_code
```

That explains the longer runtime:

```text
same doubled direction artifact count
larger training frame
extra direction scoring surface
```

Corrected interpretation:

```text
This benchmark rejects the hybrid implementation.
It does not fully reject a true shared paired-direction model.
```

The real next design, if we continue model-level direction work, must be:

```text
one artifact per horizon/fold/trial
one shared model scores both candidate directions
prediction emits score_up and score_down for the same observation
selection optimizes directional dominance, not raw event probability
```

Target artifact count should be:

```text
2 trials × 2 folds × 2 horizons × 1 shared direction model = 8 artifacts
```

not 16.

Direction is trading-critical. A high event score on the wrong side is not a
small miss; it is the wrong trade. Therefore the objective must directly penalize:

```text
opposite_side_quality
gray_zone_rate
false_direction_cost
```

rather than merely rewarding selected-side event lift.
