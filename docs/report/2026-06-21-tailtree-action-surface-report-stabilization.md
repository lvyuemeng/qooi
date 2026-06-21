# Tailtree Action-Surface Report Stabilization

## Scope

Stabilized the tailtree path-behavior output without adding new artifacts or opaque manager surfaces.

Implemented:

- report integration for existing `tailtree-action-surface.csv`
- explicit side promotion gate lines in `report.md`
- down-side production policy as market-state only when aggregate gates fail
- test coverage for report action-surface lines and down-side non-promotion wording

No new diagnostic CSVs were added. The single semantic candidate artifact remains:

```text
tailtree-action-surface.csv
```

## Code touched

```text
src/qooi/scanner/output.py
src/qooi/scanner/workflow.py
tests/test_tailtree_explicit_labels.py
```

Existing reduction surfaces remain:

```text
label_tail_paths(...)
tailtree_target_training_values(..., target=...)
tailtree_action_surface_frame(...)
```

## Report behavior

`render_report(...)` now includes:

```text
## Tailtree Action Surface
```

This section reads the already-produced action surface and prints:

- actionability counts/rates
- side/action split
- blocker reason counts
- path-state profile by side
- promotion gate summary by side

The report does not create another artifact.

## Promotion gate semantics

The gate is report-level side interpretation, not row-level relabeling.

Per side:

```text
trade_candidates = count(actionability == "trade_candidate")
mean_calibrated_side_margin = mean(calibrated_side_margin)
false_direction_rate = mean(false_direction_int)
```

Policy:

```text
candidate annotation if:
  trade_candidates > 0
  mean_calibrated_side_margin > 0
  false_direction_rate < 0.20

otherwise:
  market-state only
```

For down specifically, failed gate is rendered as:

```text
market-state only; do not promote short
```

This intentionally preserves row-level path evidence while preventing side-level promotion when the aggregate market behavior is bad.

## Verification

```text
uv run python -m pytest tests/test_tailtree_explicit_labels.py tests/test_scanner_workflow_migration.py tests/test_state.py -q
22 passed
```

```text
uv run python -m ruff check ...
All checks passed
```

```text
uv run python -m ty check ...
All checks passed
```

## Benchmark smoke

Command:

```text
uv run python -m scripts.scanner_potential --config configs/potential-paired-replay-test-tailtree.toml
```

Result:

```text
data\output\potential\paired-replay-test\report.md
```

Report contains the new section:

```text
## Tailtree Action Surface
```

Artifact shapes:

```text
tailtree-action-surface.csv:       369678 rows x 21 columns
tailtree-selection-efficiency.csv: 16 rows x 50 columns
tailtree-label-distribution.csv:   8 rows x 13 columns
```

## Report output excerpt

```text
- action surface rows=369_678
- actionability gray_zone: rows=16_468 rate=0.045
- actionability no_action: rows=266_665 rate=0.721
- actionability reversal_watch: rows=68_590 rate=0.186
- actionability trade_candidate: rows=17_955 rate=0.049
```

Side/action split:

| side | actionability | rows |
|---|---|---:|
| down | gray_zone | 8313 |
| down | no_action | 124822 |
| down | reversal_watch | 51115 |
| down | trade_candidate | 590 |
| up | gray_zone | 8155 |
| up | no_action | 141843 |
| up | reversal_watch | 17475 |
| up | trade_candidate | 17365 |

Blockers:

| side | blocker | rows |
|---|---|---:|
| down | both_or_mixed_path | 8313 |
| down | no_clean_side | 124822 |
| down | opposite_tail_dominates | 51115 |
| up | both_or_mixed_path | 8155 |
| up | no_clean_side | 141843 |
| up | opposite_tail_dominates | 17475 |

Path states:

| side | best_path_state | rows |
|---|---|---:|
| down | clean_down | 21665 |
| down | clean_up | 42914 |
| down | down_first_both | 485 |
| down | none | 114061 |
| down | up_first_both | 5715 |
| up | clean_down | 21095 |
| up | clean_up | 42708 |
| up | down_first_both | 325 |
| up | none | 115045 |
| up | up_first_both | 5665 |

## Promotion gate result

| side | report status | trade candidates | mean calibrated side margin | false direction rate |
|---|---|---:|---:|---:|
| up | candidate annotation | 17365 | 0.125018 | 0.094542 |
| down | market-state only; do not promote short | 590 | -0.088374 | 0.276536 |

The down row-level surface still finds 590 local clean-side candidates, but the aggregate side gate fails because down has negative calibrated side margin and high false-direction rate. Therefore down is not promoted.

## Selection-efficiency summary

| direction | max objective_hpo | max side_hpo | mean side-only | mean false-direction | mean calibrated side margin | mean calibrated directional margin | mean selected utility margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| down | 39.618311 | 39.991212 | 0.141352 | 0.301563 | -0.067886 | -0.155214 | 0.117524 |
| up | 32.729725 | 33.540472 | 0.271665 | 0.097892 | 0.149045 | 0.299485 | 0.571146 |

## Behavior evaluation

Up remains feasible as a research candidate annotation:

- trade candidates: 17,365
- positive mean calibrated side margin: 0.125
- false-direction rate below gate: 0.095

Down is not feasible as a promoted short signal:

- negative mean calibrated side margin: -0.088
- high false-direction rate: 0.277
- many down-selected rows resolve as clean-up/none/mixed states

Current interpretation:

```text
up = candidate annotation / research watch

down = market-state risk / reversal warning / blocker, not short candidate
```

## Production suitability

Suitable only as:

```text
research scanner annotation
candidate review surface
market-state warning layer
```

Not suitable as:

```text
live execution signal
automatic short promotion
authorization for trading
```

Production extent that is acceptable:

```text
show up trade_candidate rows as watchlist/ranking evidence
show down rows as market-state warnings only
require manual review before any strategy test
```

## Next step

Do not add feature packs or new artifacts. Next stabilization should be:

1. keep report-level promotion gates explicit
2. wire the gate status into ranked scanner output only as annotation
3. run a broader walkforward/backfill benchmark
4. if up gate remains stable, allow up-only candidate promotion into research ranking
5. keep down disabled as short promotion unless calibrated side margin and false-direction gates both pass across multiple windows
