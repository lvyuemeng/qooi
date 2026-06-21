# Tailtree Path-Behavior Reduction Completion Report

## Scope

Completed the ponytail reduction slice for tailtree path-behavior semantics:

- `label_tail_paths(...)` is now the canonical label API.
- `label_tail_exceedances(...)` compatibility alias was removed from source/tests/architecture/graph docs.
- `tailtree_target_training_values(..., target=...)` is the canonical target-specific binary training API.
- Narrow helpers `event_lift_training_values`, `any_event_training_values`, and `side_only_training_values` were removed from source/tests.
- `tailtree-action-surface.csv` remains the single semantic candidate/action artifact; no selected-behavior/horizon-panel artifact pile was added.

## Verification

```text
uv run python -m pytest tests/test_tailtree_explicit_labels.py tests/test_scanner_workflow_migration.py tests/test_state.py -q
21 passed
```

```text
uv run python -m ruff check ...
All checks passed
```

```text
uv run python -m ty check ...
All checks passed
```

Reduction grep:

```text
grep -R --exclude='*.pyc' "label_tail_exceedances\|def event_lift_training_values\|def any_event_training_values\|def side_only_training_values\|any_event_training_values\|side_only_training_values" -n src tests docs/architecture docs/graph
# no output
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

Artifact shapes:

```text
tailtree-action-surface.csv:       373434 rows x 21 columns
tailtree-selection-efficiency.csv: 16 rows x 50 columns
tailtree-label-distribution.csv:   8 rows x 13 columns
```

## Label distribution

| horizon | state | rows | rate |
|---:|---|---:|---:|
| 24 | both | 378 | 0.000613 |
| 24 | down | 2570 | 0.004169 |
| 24 | none | 601776 | 0.976104 |
| 24 | up | 11784 | 0.019114 |
| 48 | both | 553 | 0.001332 |
| 48 | down | 3352 | 0.008076 |
| 48 | none | 396921 | 0.956293 |
| 48 | up | 14236 | 0.034298 |

Down-only path events remain much rarer than up-only events: about 4.6x rarer at 24h and 4.3x rarer at 48h.

## Action surface summary

| actionability | rows | rate |
|---|---:|---:|
| gray_zone | 16440 | 0.044024 |
| no_action | 264599 | 0.708556 |
| reversal_watch | 74250 | 0.198830 |
| trade_candidate | 18145 | 0.048590 |

By side:

| side | actionability | rows |
|---|---|---:|
| down | gray_zone | 8400 |
| down | no_action | 119094 |
| down | reversal_watch | 59225 |
| up | gray_zone | 8040 |
| up | no_action | 145505 |
| up | reversal_watch | 15025 |
| up | trade_candidate | 18145 |

The current policy emits zero down `trade_candidate` rows. Down is expressed as market state (`reversal_watch`, `gray_zone`, `no_action`) rather than a clean short signal.

## Path-state profile

| action_side | best_path_state | rows |
|---|---|---:|
| down | clean_down | 21690 |
| down | clean_up | 44393 |
| down | down_first_both | 600 |
| down | none | 114256 |
| down | up_first_both | 5780 |
| up | clean_down | 21050 |
| up | clean_up | 43926 |
| up | down_first_both | 455 |
| up | none | 115759 |
| up | up_first_both | 5525 |

A selected down side frequently lands on rows whose realized path is clean-up, none, or mixed. This supports the decision to treat down as market-state evidence until calibration improves.

## Selection efficiency

| direction | max objective_hpo | max side_hpo | mean side-only | mean both-tail | mean false-direction | mean calibrated side margin | mean calibrated directional margin | mean selected utility margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| down | 36.033813 | 36.083628 | 0.137563 | 0.110277 | 0.367655 | -0.086280 | -0.256240 | 0.037308 |
| up | 39.405399 | 40.318106 | 0.280715 | 0.106374 | 0.066418 | 0.160300 | 0.394791 | 0.964967 |

## Behavior evaluation

Up side is feasible as a candidate surface in this smoke run:

- positive calibrated side margin
- positive calibrated directional margin
- low false-direction rate
- non-trivial trade-candidate population

Down side is not feasible as a production short signal in this smoke run:

- negative calibrated side margin
- negative calibrated directional margin
- high false-direction rate
- zero trade candidates under clean-side policy
- many selected down rows resolve as clean-up/none/mixed path states

Therefore the correct current interpretation is:

```text
up = candidate signal, still needs wider validation

down = market-state warning / reversal-watch / blocker, not directional short candidate
```

## Production suitability

Suitable now only as a research/smoke-grade semantic surface and diagnostic gate:

- ✅ useful for separating trade candidates from market-state warnings
- ✅ useful for blocking contaminated down signals
- ✅ useful for reporting why a side is not actionable
- ⚠️ not production trading-ready as a live directional executor input
- ⚠️ not yet validated across enough windows, symbols, and walkforward regimes
- ❌ down side should not be promoted to short candidate in the current benchmark

A limited production extent may be acceptable as a non-executing scanner annotation:

```text
up trade_candidate = watchlist/ranking candidate only

down reversal_watch/gray_zone = market-state/risk warning only
```

No live trading authorization follows from these artifacts.

## Next step if not production-ready

Next slice should make action policy configurable and validate stability, not add another feature pack:

1. Keep down as market-state output, not short signal.
2. Add report sections that read `tailtree-action-surface.csv` directly.
3. Run broader walkforward/backfill benchmark across more dates/symbols.
4. Evaluate promotion gates by side:
   - calibrated_side_margin > 0
   - calibrated_directional_margin > 0
   - false_direction_rate cap
   - enough trade_candidate count
5. Only after stable up-side evidence, consider integrating up candidates into ranked scanner output.
6. Only if down calibrated margins become positive should short-candidate behavior be re-enabled.
