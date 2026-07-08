# Load hotpath/helper reduction follow-up

## Goal

Continue the `load.py` / `state.py` / `coverage.py` cleanup with one behavior-preserving speed slice:

- reduce repeated whole-frame source merges/filters in `load.py`;
- keep bars and sources separate because their shapes differ;
- avoid new wrapper classes, registries, or generic product loaders;
- preserve explicit missing/stale diagnostics.

## Current evidence

After the previous schema slice:

```text
src/qooi/pipeline/load.py      901 lines, 40 defs, 12 frame column checks
src/qooi/scanner/state.py     1449 lines, 28 defs, 14 frame column checks
src/qooi/pipeline/coverage.py  652 lines, 14 defs, 4 frame column checks
```

The next obvious `load.py` hotpath is source execution:

```python
for job in plan.jobs:
    local = _symbol_frame(existing, job.symbol)
    fetched = ...
    existing = merge_frames(existing, fetched, ("symbol", "timestamp"))
```

For every source job this scans/merges the whole product cache. With multiple symbols and source products this is unnecessary: source backfill/latest jobs are symbol-local, so the cache can be split once by symbol, updated locally, and concatenated once per product.

## Target slice

### Phase 1 — source product cache by symbol

Inside `_execute_source_jobs`:

1. Build `by_symbol: dict[str, DataFrame]` once from the existing source cache.
2. Merge `current_snapshot` fetches into symbol-local frames.
3. Run backfill/latest refresh against the symbol-local frame.
4. Write the product cache once after all symbol updates.

This removes repeated full-product `_symbol_frame(existing, symbol)` filters and repeated full-product merge writes inside the job loop.

### Phase 2 — small helper deletion if unused

After Phase 1, grep whether `_symbol_frame` still has callers. If none, delete it.

Do not delete `_merge_source` unless it becomes a pure one-liner at a single call site; it still names the source merge contract.

### Phase 3 — state.py only if net-shorter

Do not touch family feature math unless a change deletes repeated schema checks without obscuring the calculations. In this slice, `state.py` is a candidate only for documentation/constant consistency, not a forced edit.

## Verification

Focused gates:

```bash
uv run ruff check src/qooi/pipeline/load.py src/qooi/pipeline/coverage.py src/qooi/scanner/state.py scripts/check_module_layout.py tests/test_cache_behavior.py tests/test_pipeline_validate.py tests/test_bar_coverage_health.py tests/test_state.py
uv run pytest tests/test_cache_behavior.py tests/test_pipeline_validate.py tests/test_bar_coverage_health.py tests/test_state.py -q
uv run python scripts/check_module_layout.py
```

Ad-hoc verifier:

- Create a tiny existing source cache with two symbols.
- Exercise symbol-local source merge semantics by calling the public `load_market` path with cache-only / low-budget planning where possible, or assert the static reduction properties if network calls are not used.
- Assert `_symbol_frame` is deleted if Phase 2 applies.

## Done / not done

Done when source execution no longer repeatedly filters/merges whole source cache per job and focused gates pass.

Not in this slice: concurrent backfill scheduling, generic product loaders, or feature math changes.

## Implementation result

Completed:

- Changed `_execute_source_jobs` to split an existing source product cache once by `symbol`, update current/latest/backfill jobs against symbol-local frames, then concatenate/save once per product.
- Replaced the new source-symbol loops with `get_column(...).unique().drop_nulls().to_list()` instead of row-dict iteration.
- Deleted `_symbol_frame`; added a layout guard for `def _symbol_frame(` in `scripts/check_module_layout.py`.
- Left `state.py` unchanged in this slice because the net-shorter speed target was in `load.py`.

Verified:

```text
All checks passed!
27 passed
module layout check passed
ad-hoc verification passed: source cache updates are symbol-local, untouched symbols are preserved, _symbol_frame is deleted
```
