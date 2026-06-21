# Tailtree Advanced Stabilization Benchmark

## Scope

Used the enhanced configuration requested for a longer stabilization benchmark:

```text
configs/potential-advanced-tailtree.toml
```

Configuration changes applied:

```text
outcome_horizon = [12, 24, 48]
profile_id = "event-lift-walkforward-optuna-h12-h24-h48"
model_tag = "tailtree-event-lift-advanced-wf-optuna-h12-h24-h48"
max_folds = 3
```

Existing advanced scale retained:

```text
max_symbols = 160
bars.days = 180
training.max_trials = 2
walkforward train_days = 90
walkforward valid_days = 21
walkforward step_days = 21
```

This run is materially longer than the paired replay smoke: 2 Optuna trials x 3 walkforward folds x 3 horizons x 2 sides.

## Command

```text
uv run python -m scripts.scanner_potential --config configs/potential-advanced-tailtree.toml
```

Completed successfully:

```text
data\output\potential\advanced-tailtree\report.md
```

The generated report contains:

```text
## Tailtree Action Surface
```

## Artifacts

```text
tailtree-action-surface.csv:       3,779,170 rows x 21 columns
tailtree-selection-efficiency.csv: 144 rows x 50 columns
tailtree-label-distribution.csv:   12 rows x 13 columns
tailtree-profile-runs.csv:         6 rows x 15 columns
```

Profile run count matches 2 trials x 3 folds. Each fold produced 6 models: 3 horizons x 2 sides.

## Label distribution

| horizon | state | rows | rate |
|---:|---|---:|---:|
| 12 | both | 102 | 0.000138 |
| 12 | down | 1,393 | 0.001890 |
| 12 | none | 728,674 | 0.988634 |
| 12 | up | 6,882 | 0.009337 |
| 24 | both | 413 | 0.000517 |
| 24 | down | 3,230 | 0.004047 |
| 24 | none | 780,734 | 0.977948 |
| 24 | up | 13,961 | 0.017488 |
| 48 | both | 606 | 0.001122 |
| 48 | down | 4,435 | 0.008212 |
| 48 | none | 517,118 | 0.957557 |
| 48 | up | 17,880 | 0.033109 |

Behavior remains asymmetric across every horizon:

```text
up-only events are materially more common than down-only events
both-tail events grow with horizon but remain low as a raw class rate
```

## Action surface summary

| actionability | rows | rate |
|---|---:|---:|
| gray_zone | 95,365 | 0.025 |
| no_action | 2,963,322 | 0.784 |
| reversal_watch | 573,221 | 0.152 |
| trade_candidate | 147,262 | 0.039 |

By side:

| side | actionability | rows |
|---|---|---:|
| down | gray_zone | 47,699 |
| down | no_action | 1,377,650 |
| down | reversal_watch | 465,401 |
| down | trade_candidate | 1,980 |
| up | gray_zone | 47,666 |
| up | no_action | 1,585,672 |
| up | reversal_watch | 107,820 |
| up | trade_candidate | 145,282 |

The report-level side gate blocks down despite 1,980 row-local down candidates.

## Promotion gate result

| side | report status | trade candidates | mean calibrated side margin | false direction rate |
|---|---|---:|---:|---:|
| up | candidate annotation | 145,282 | 0.129 | 0.057 |
| down | market-state only; do not promote short | 1,980 | -0.098 | 0.246 |

Interpretation:

```text
up passes aggregate side gate across the advanced run

down fails aggregate side gate because calibrated side margin is negative and false direction is high
```

## Fold consistency

| fold | side | rows | trade candidates | mean side margin | false direction rate |
|---:|---|---:|---:|---:|---:|
| 0 | down | 595,993 | 1,895 | -0.110 | 0.227 |
| 0 | up | 594,213 | 35,665 | 0.151 | 0.055 |
| 1 | down | 607,197 | 85 | -0.093 | 0.204 |
| 1 | up | 606,557 | 36,060 | 0.117 | 0.035 |
| 2 | down | 689,540 | 0 | -0.091 | 0.298 |
| 2 | up | 685,670 | 73,557 | 0.121 | 0.079 |

Consistency conclusion:

- up has positive mean side margin in all 3 folds
- up false direction remains below 0.10 in all 3 folds
- down has negative mean side margin in all 3 folds
- down trade candidates collapse from 1,895 to 85 to 0 across folds
- down false direction is high in all folds and worsens in fold 2

## Horizon consistency

| horizon | side | rows | trade candidates | mean side margin | false direction rate |
|---:|---|---:|---:|---:|---:|
| 12 | down | 543,374 | 85 | -0.060 | 0.223 |
| 12 | up | 537,166 | 30,130 | 0.082 | 0.029 |
| 24 | down | 662,371 | 0 | -0.104 | 0.234 |
| 24 | up | 662,301 | 49,530 | 0.130 | 0.050 |
| 48 | down | 686,985 | 1,895 | -0.122 | 0.275 |
| 48 | up | 686,973 | 65,622 | 0.165 | 0.087 |

Horizon conclusion:

- up passes side-margin sign at 12/24/48
- up false direction rises with horizon but remains below the 0.20 report gate
- down fails side-margin sign at 12/24/48
- down false direction is above 0.20 at every horizon
- down local candidates are unstable by horizon: nearly none at 12, none at 24, only a small 48h cluster

This supports treating down as market state rather than short signal.

## Selection-efficiency by horizon

| horizon | side | max side_hpo | side-only rate | false direction rate | calibrated side margin | calibrated directional margin | selected utility margin |
|---:|---|---:|---:|---:|---:|---:|---:|
| 12 | down | 109.539 | 0.0367 | 0.2703 | -0.0679 | -0.2323 | -0.0440 |
| 12 | up | 105.845 | 0.1362 | 0.0282 | 0.1034 | 0.3174 | 1.0100 |
| 24 | down | 45.650 | 0.0744 | 0.2706 | -0.1011 | -0.1968 | -0.1112 |
| 24 | up | 55.721 | 0.2202 | 0.0530 | 0.1584 | 0.2951 | 0.6306 |
| 48 | down | 40.937 | 0.1166 | 0.2714 | -0.0946 | -0.1369 | -0.0493 |
| 48 | up | 24.797 | 0.2926 | 0.0866 | 0.1941 | 0.2890 | 0.4992 |

Selection-efficiency conclusion:

- down has attractive raw side_hpo at 12h, but calibrated side/directional margins are negative
- this is exactly why side_hpo alone is insufficient for promotion
- up has positive calibrated margins for all horizons

## Feasibility assessment

### Up side

Feasible as a research candidate annotation.

Evidence:

- positive side margin across folds
- positive side margin across horizons
- false direction below gate across folds/horizons
- large and stable candidate population

Production extent:

```text
OK for non-executing scanner annotation / watchlist ranking evidence
not OK for live trading authorization
```

### Down side

Not feasible as a short signal.

Evidence:

- negative side margin in every fold
- negative side margin in every horizon
- false direction above 0.20 in every horizon
- candidate count unstable and collapses across folds
- many selected down rows are better interpreted as reversal/volatility market state

Production extent:

```text
OK as market-state risk / reversal warning / blocker
not OK as short promotion
```

## Decision

Keep current policy:

```text
up: candidate annotation

down: market-state only; do not promote short
```

This is more robust after the enhanced benchmark than after the smaller smoke run.

## Next step

Do not add feature packs. Do not add new artifacts.

Recommended next slice:

1. Wire the report-level promotion status into ranked scanner rows as annotation only.
2. Preserve down as `market_state` / `blocker_reason`, not `short_candidate`.
3. Run one more backfill over a different date window before allowing up candidate annotation into a daily workflow.
4. Keep promotion gates visible in every report so feasibility remains auditable.
