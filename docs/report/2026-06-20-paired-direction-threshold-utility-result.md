# Paired direction-threshold utility benchmark result

## Experiment

Implementation note:

```text
docs/report/2026-06-20-paired-direction-threshold-utility-implementation.md
```

Objective tested:

```text
paired_directional_utility
```

Training shape:

```text
observation × candidate_direction_code × return_threshold_pct
candidate_direction_code = +1 for up, -1 for down
return_threshold_pct = 30.0
```

Model artifact shape:

```text
one shared artifact per horizon/fold/trial
```

This fixed the previous hybrid flaw: direction is inside the model, and the scan no longer trains separate up/down model files.

## Runtime / artifact count

Run:

```text
PAIRED_DIRECTION_UTILITY_SECONDS=417
```

Artifacts:

```text
paired_model_files=8
```

Expected artifact count was achieved:

```text
2 trials × 2 folds × 2 horizons × 1 shared artifact = 8
```

So the duplication problem was fixed.

## Benchmark artifacts

```text
data/output/potential/benchmarks/paired-direction-utility-report.md
data/output/potential/benchmarks/paired-direction-utility-tailtree-selection-efficiency.csv
```

Baseline:

```text
data/output/potential/benchmarks/conflict-abstain-report.md
data/output/potential/benchmarks/conflict-abstain-tailtree-selection-efficiency.csv
```

## Selection-efficiency comparison

| metric | conflict-abstain baseline | paired-direction utility | delta |
|---|---:|---:|---:|
| hpo_score max | 63.687865 | 35.671231 | -43.99% |
| valid_tail_lift max | 60.037224 | 33.070499 | -44.92% |
| selected_profit_proxy_mean max | 2.662485 | 2.049974 | -23.01% |
| selected_profit_proxy_p90 max | 9.188335 | 7.789145 | -15.23% |
| selected_tail_count max | 6600 | 6235 | -5.53% |
| selected_tail_per_1k_obs max | 794.426679 | 662.100457 | -16.66% |

Best baseline row:

```text
h24 down top_1pct
hpo=63.687865
lift=60.037224
utility_mean=1.335474
tails=535
```

Best paired row:

```text
h48 down top_1pct
hpo=35.671231
lift=33.070499
utility_mean=0.500732
tails=440
```

## Gray-zone comparison

Baseline:

```text
promoted=3
conflict_watches=8
promoted_conflicts=0
```

Paired-direction utility:

```text
promoted=3
conflict_watches=14
promoted_conflicts=0
```

Gray-zone watches increased materially.

## Model metadata

The model did use the new design fields:

```text
objective = paired_directional_utility
candidate_direction_code in continuous_features = true
return_threshold_pct in continuous_features = true
feature_count = 28
```

Top importance included:

```text
decision_transition
funding_rate_raw
bar_return_24h_pct
lsr_ratio_raw
bar_return_24h_per_vol_7d
bar_close_position_48h
candidate_direction_code
```

So the failure was not a missing feature wiring issue.

## Feasibility verdict

Reject and revert.

Reason:

```text
- artifact duplication was fixed
- but HPO/lift collapsed by ~44%
- utility worsened
- gray-zone watches increased from 8 to 14
```

The shared artifact design is structurally better, but the current utility-regression objective is not good enough.

## Likely reason

The implemented training target was:

```text
event * log1p(direction_utility)
```

with regression scoring. This rewards utility magnitude but weakens event concentration. The top buckets became less concentrated in true directional extreme events.

In other words, the model learned a smoother utility surface, not a sharper direction-threshold event selector.

## Ponytail action

Revert the implementation/config and keep:

```text
tail_event_lift + conflict abstention
```

Next design should not try utility-regression first. It should preserve binary event concentration and add direction/utility only in the HPO/replay objective:

```text
train binary direction-threshold event with one shared artifact
score paired up/down
HPO objective penalizes opposite-side evidence and false-direction cost
```

That is closer to the theory base:

```text
direction first
threshold event concentration second
utility as selection/HPO weighting, not raw regression target
```

## Correction after architecture review

The implementation above is now classified as an architecture mistake, not merely a weak objective.

User correction:

```text
Do not model return_threshold_pct and direction in features.
Direction/threshold are objective dimensions.
Previously we first filtered threshold, then concentrated on utility.
Do the same: first direction, then threshold, then utility.
```

Accepted.

The wrong part was:

```text
candidate_direction_code as feature
return_threshold_pct as feature
utility regression target
```

Correct scanner boundary:

```text
state features = known-at-close market/source features only
outcome labels = direction + threshold event labels
HPO/replay = utility and false-direction penalties
```

So the revised next design is documented in:

```text
docs/report/2026-06-20-direction-threshold-utility-objective-redesign.md
```

Correct next ponytail path:

```text
keep binary direction-specific threshold-event training
add paired-direction utility/opposite-side/gray-zone penalties to selection/HPO replay
avoid new direction/threshold feature columns
```
