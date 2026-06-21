# Extreme-event objective/horizon ponytail plan

## Direction

User approved the next step:

```text
objective should learn event probability/lift over all states
horizon should test h24/h48/h72 for 30% event formation
constrain by ponytail
follow same benchmark comparison
explain feature -> label process
```

## Ponytail boundary

Do:

```text
- add one objective value: tail_event_lift
- keep the existing tailtree-selection-efficiency.csv comparison surface
- change advanced config to 1H/4H/1D and h24/h48/h72
- benchmark against the existing current advanced artifact snapshot
```

Do not:

```text
- add more diagnostics
- add broad feature blocks
- add sin/cos/Fourier features
- add model framework/classes
- add another artifact family
```

## Design

### Existing path

Current `tail_utility_quantile` path trains on tail rows:

```text
known-at-close observations
+ future path outcomes
-> tail rows only
-> LightGBM learns utility/severity among events
-> all rows route through leaves for denominators/lift
```

This is good for severity, but weak for rare-event discovery because the model does not learn normal-vs-event separation directly.

### New path

Add `tail_event_lift` as the minimum event-probability objective:

```text
known-at-close observations
+ future path outcomes
-> all rows
-> binary target: did the future path exceed ±30% for this direction/horizon?
-> LightGBM binary classifier
-> all rows route through leaves
-> existing selection-efficiency measures event lift/support/utility
```

The label is direction-specific:

```text
up:   forward_max_return_pct >= threshold_pct
down: forward_min_return_pct <= -threshold_pct
```

Utility columns stay available for ranking/reporting, but the model split target becomes event presence instead of event severity.

## Config change

Advanced only:

```toml
[potential.bars]
timeframes = ["1H", "4H", "1D"]

[potential.evidence.tailtree]
threshold_pct = 30.0
outcome_horizon = [24, 48, 72]

[[potential.evidence.tailtree.profiles]]
objective = "tail_event_lift"
```

Daily remains a later decision after advanced benchmark.

## Benchmark protocol

Baseline:

```text
data/output/potential/benchmarks/baseline-tailtree-selection-efficiency.csv
```

Event-lift run:

```bash
python scripts/scanner_potential.py --config configs/potential-advanced-tailtree.toml
```

Compare existing columns only:

```text
hpo_score
valid_tail_lift
selected_tail_count
selected_observation_count
selected_utility_mean
selected_utility_p90
profit_proxy_per_selected_obs
promotion_threshold_pass_int
```

Keep only if the selection-efficiency surface improves or reveals a clearly better horizon with enough support.

## Implementation steps

1. Inspect model/objective hook.
2. Add `tail_event_lift` objective support with minimum branching.
3. Update advanced config only.
4. Add/adjust one small unit test for objective value acceptance/label behavior if nearby tests exist.
5. Run ruff/ty/tests.
6. Run advanced benchmark.
7. Compare against baseline and explain feature -> label flow.
