# Test Architecture

The test suite is intentionally being reduced from broad white-box coverage toward ownership and boundary coverage. Passing tests are not enough: each test file should have a clear module owner and should avoid preserving stale compatibility APIs.

## Current test ownership map

| Test file | Owner | Status | Direction |
|---|---|---|---|
| `tests/test_module_boundaries.py` | architecture boundaries | keep | Guard forbidden imports before API expansion. |
| `tests/test_scanner_potential.py` | scanner workflow/evidence | compressed | Keeps workflow artifact smoke, config/universe selection, known-at-close source/evidence/history invariants, decision gates, and transition matching; helper-by-helper assertions removed. |
| `tests/test_classifier_diagnostics.py` | scanner classifiers | keep | Small classifier health contract. |
| `tests/test_classifier.py` | deterministic structure classifier | keep | Rename/scope only if classifier ownership moves. |
| `tests/test_sources.py` | source providers/artifacts | keep, then split if needed | Tests source normalization and manifests; old accumulation vocabulary retired. |
| `tests/test_sources_coverage.py` | source coverage | keep | Small coverage contract. |
| `tests/test_research_diagnostics.py` | research artifact tables | keep, then compress | Keep table-contract coverage; avoid over-testing internal helper names. |
| `tests/test_research_data.py` | research frame loading | keep | Verify known-at-close and coverage behavior. |
| `tests/test_research_backtest.py` | research config/backtest bridge | review | Keep only if it protects promoted signal/evaluation bridge. |
| `tests/test_dynamic_contracts.py` | dynamic contracts | keep | Lightweight optional AI boundary. |
| `tests/test_dynamic_behavior.py` | dynamic state preparation/evaluation | compress | Keep import-safe and output contracts; reduce config micro-tests. |
| `tests/test_dynamic_vq_rssm.py` | optional VQ-RSSM implementation | compressed | Reduced from broad implementation coverage to a small optional-Torch smoke suite plus checkpoint, sequence inference, schedule, spec-validation, and core math/codebook contracts. |
| `tests/test_strategy_registry.py` | strategy registry/specs | compressed | Reduced from variant-by-variant assertions to public signal-column contracts, structural known-at-close invariants, selected entry/gate/exit behavior, and unpromoted catalog placement. |
| `tests/test_backtest_executor.py` | executor accounting | compressed | Reduced cache-backed executor coverage to one report/schema/PnL-direction smoke plus synthetic diagnostics, recovery sizing/blocking, intrabar ambiguity, and loss-cooldown contracts. |
| `tests/test_pipeline_integration.py` | core process-bar pipeline | compressed | Reduced to one process-bar contract per behavior: entry/idling, default held-signal independence, policy exits, flip reversal, holding/time-stop lifecycle, explicit multi-entry, recovery flow, reverse-gated recovery, and hard-stop preemption. |
| `tests/test_basket.py` | basket lifecycle | keep/compress | Keep invariants, not every scenario variant. |
| `tests/test_exits.py` | exit policy | keep/compress | Keep stop/target/trailing/time-stop semantics; owns skip-trailing during recovery after moving that invariant out of pipeline integration. |
| `tests/test_recovery.py` | recovery policies | keep/compress | Keep policy invariants. |
| `tests/test_evaluate.py` | reporting/metrics formatting | compressed | Reduced format/detail tests to report metric contracts, derived bucket/section contracts, attribution diagnostics, recommendation/ranking decisions, and safe missing-column handling. |
| `tests/test_data.py` | exchange/cache/research-data bridge | split/compress | Separate exchange cache tests from research frame tests if it grows. |
| `tests/test_state.py` | state providers | keep/compress | Keep ID parsing and state-source evaluation invariants. |
| `tests/test_flow_pipeline.py` | indicator feature helpers | review/compress | Owns remaining OFI/regime flow contracts, including the merged legacy pipeline parity check on synthetic data. |
| `tests/test_backtest.py` | legacy indicator smoke | merged/deleted | Removed the one-test cache-dependent file after moving its pipeline parity invariant into `tests/test_flow_pipeline.py`. |

## Reduction rules

1. Rename old-vocabulary test files when the module has been renamed; do not keep `accumulation`/`ai` names for scanner/dynamic tests.
2. Delete tests for intentionally retired APIs instead of adding compatibility wrappers.
3. Prefer black-box contracts at module boundaries over helper-by-helper tests.
4. Keep side-effect tests focused on artifact existence, schema, and missing-data diagnostics.
5. Keep optional AI tests lazy/skipped when optional dependencies are unavailable.
6. Compress broad scenario tests by preserving unique invariants, not every historical scenario.
7. After each prune/rename batch, run `uv run ruff check src tests scripts` and `uv run pytest -q`.

## Next compression targets

1. Data/cache tests — split exchange cache, coverage, and research-frame bridge if it keeps growing.
