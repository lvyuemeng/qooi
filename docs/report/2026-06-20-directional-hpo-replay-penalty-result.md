# Directional HPO replay penalty benchmark result

## Experiment

Implementation note:

```text
docs/report/2026-06-20-directional-hpo-replay-penalty-implementation.md
```

Goal:

```text
keep binary up/down threshold-event concentration training
add same-horizon opposite-side penalty to selection-efficiency HPO replay
no direction/threshold feature columns
```

This was architecture-consistent:

```text
state/model features unchanged
labels remain tail_up/tail_down
utility/opposite-side terms used only in HPO replay
```

## Verification

Before scan:

```text
ruff: pass
ty: pass
pytest tests/test_scanner_workflow_migration.py tests/test_state.py -q: 14 passed
```

Scan:

```text
DIRECTIONAL_HPO_REPLAY_SECONDS=594
```

Artifacts:

```text
data/output/potential/benchmarks/directional-hpo-replay-report.md
data/output/potential/benchmarks/directional-hpo-replay-tailtree-selection-efficiency.csv
```

## Selection-efficiency comparison

Baseline:

```text
data/output/potential/benchmarks/conflict-abstain-tailtree-selection-efficiency.csv
```

| metric | baseline | directional replay | delta |
|---|---:|---:|---:|
| hpo_score max | 63.687865 | 22.565199 | -64.57% |
| valid_tail_lift max | 60.037224 | 58.898147 | -1.90% |
| selected_profit_proxy_mean max | 2.662485 | 2.733977 | +2.69% |
| selected_profit_proxy_p90 max | 9.188335 | 9.226495 | +0.42% |
| selected_tail_count max | 6600 | 6685 | +1.29% |
| selected_tail_per_1k_obs max | 794.426679 | 776.255708 | -2.29% |

Best directional-penalized row:

```text
h48 down top_1pct
hpo_score=22.565199
base_hpo_score=51.075441
lift=46.919830
utility=1.212823
opposite_lift=20.839353
opposite_utility=2.670889
same_horizon_gray_zone=1
```

Best by unpenalized base score:

```text
h24 down top_1pct
base_hpo_score=62.438182
hpo_score=20.886860
lift=58.898147
utility=1.246565
opposite_lift=34.160822
opposite_utility=2.390499
same_horizon_gray_zone=1
```

## Report comparison

Baseline report:

```text
promoted=3
conflict_watches=8
promoted_conflicts=0
```

Directional replay report:

```text
promoted=3
conflict_watches=9
promoted_conflicts=0
```

Promoted rows remained conflict-free, but conflict watches increased from 8 to 9.

## Why reject this pass

The patch was architecture-consistent but too blunt.

It compared aggregate score-bucket evidence:

```text
same horizon + same score bucket + opposite direction
```

Because both directions often have material aggregate evidence for the same score budget, all 64 selection-efficiency rows became gray-zone:

```text
same_horizon_gray_zone_int sum = 64
```

That means the penalty did not discriminate clean vs conflicted candidate symbols. It penalized nearly the whole evidence surface.

This is not useful for HPO selection.

## Ponytail decision

Reject and revert the aggregate score-bucket penalty patch.

Keep:

```text
tail_event_lift + promotion-level conflict abstention
```

## Correct next pass

The objective must pair at candidate/symbol level, not aggregate evidence row level.

Correct unit:

```text
symbol + decision_bar_close_ms + horizon
```

For each validation/current candidate:

```text
score_up
score_down
selected_direction
opposite_direction
selected_side_quality
opposite_side_quality
false_direction_cost
gray_zone_flag
```

Then summarize those paired candidate rows into HPO/replay metrics.

This keeps the correct objective order:

```text
direction -> threshold -> utility
```

without adding direction/threshold as model features.
