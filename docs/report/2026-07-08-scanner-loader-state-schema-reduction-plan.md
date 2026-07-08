# Scanner loader/state schema reduction plan

## Goal

Optimize `state.py`, `load.py`, and `coverage.py` without changing scanner semantics:

- reduce helper-function noise;
- make schemas explicit and consistent;
- reduce runtime `"col" in frame.columns` checks where the upstream product schema owns the column;
- delete obsolete `scripts/scanner_backfill.py` with no shim.

## Boundaries

- Keep layers separated: pipeline coverage/load stays in `qooi.pipeline`; scanner state construction stays in `qooi.scanner`.
- No compatibility wrappers for deleted scripts.
- No data masking: missing/stale data remains explicit in coverage/report artifacts.
- Do not redesign the tailtree model or feature math in this slice.

## Current audit

```text
src/qooi/scanner/state.py       1450 lines, 28 defs, 3 classes
src/qooi/pipeline/load.py        898 lines, 40 defs, 8 classes
src/qooi/pipeline/coverage.py    670 lines, 23 defs, 6 classes
scripts/scanner_backfill.py      161 lines, 8 defs, no importers
```

`scanner_backfill.py` is unreferenced by active source/tests/docs search. Its behavior is now covered by `load_market(...)` plus the active build/predict scripts, so deletion is the correct ponytail action.

## Target slice

### Phase 1 — delete obsolete backfill CLI

Delete:

```text
scripts/scanner_backfill.py
```

No replacement shim. Existing active workflow is:

```text
scripts/01_build_features.py -> scripts/02_train.py -> scripts/03_predict.py
```

### Phase 2 — coverage schema ownership

Move scalar timestamp/span/freshness helpers onto the owning coverage types:

- `ProductCoverageSpec.timestamp_bounds(frame)` owns timestamp min/max extraction.
- `ProductCoverageSpec.span_days(earliest, latest)` owns interval span math.
- `ProductCoverageSpec.is_fresh(latest)` owns freshness policy.
- `ProductCoverageSpec.max_possible_rows(...)` owns interval-aligned possible-row math.
- `CoverageState.candidate_job(policy)` owns whether and how a state becomes a fetch job.
- `CoverageState.with_allocation(...)` replaces loose `_allocated_state`.

Expected deletion/reduction:

```text
_timestamp_min
_timestamp_max
_span_days
_fresh
_max_possible_rows
_candidate_jobs_for_states
_allocated_state
```

### Phase 3 — load schema constants and direct frame ops

Introduce small owned constants for recurring cache schemas:

```text
KEY_COLUMNS = ("symbol", "timestamp")
LOADER_COLUMNS = ("_fetch_error", "_source_error")
```

Use them to reduce repeated literal schema checks and make source/bars cache contracts obvious. Avoid a new framework or generic product loader; bars and sources remain separate because their result shapes differ.

### Phase 4 — state source schema map

Replace repeated ad-hoc source family prefix/value-column checks with a compact source schema map near the existing source continuous builders:

```text
SOURCE_PREFIX = {...}
SOURCE_VALUE_CANDIDATES = {...}
```

Keep family-specific calculations explicit where behavior differs. Do not force all source families through one generic builder.

## Verification

Focused gates after edits:

```bash
uv run ruff check src/qooi/pipeline/coverage.py src/qooi/pipeline/load.py src/qooi/scanner/state.py tests/test_bar_coverage_health.py tests/test_cache_behavior.py tests/test_pipeline_validate.py tests/test_state.py
uv run pytest tests/test_bar_coverage_health.py tests/test_cache_behavior.py tests/test_pipeline_validate.py tests/test_state.py -q
uv run python scripts/check_module_layout.py
```

Ad-hoc reduction verifier:

- assert `scripts/scanner_backfill.py` is absent;
- assert no active file references `scanner_backfill`;
- assert deleted helper names are absent from `coverage.py` where applicable;
- instantiate coverage planning on a tiny frame to confirm statuses/jobs still work.

## Done / not done

Done when the above checks pass and the deleted script is gone.

Not in this slice: large rewrite of `state.py` feature math, generic product loader, or scanner CLI redesign.

## Implementation result

Completed:

- Deleted `scripts/scanner_backfill.py`.
- Added `scripts/check_module_layout.py` guard for the retired script/import snippet.
- Moved coverage timestamp bounds, span, freshness, cursor, max-possible-row, candidate-job, and allocation behavior onto `ProductCoverageSpec` / `CoverageState`.
- Removed retired coverage helpers:
  - `_timestamp_min`
  - `_timestamp_max`
  - `_span_days`
  - `_fresh`
  - `_max_possible_rows`
  - `_candidate_jobs_for_states`
  - `_allocated_state`
  - `_priority`
  - `_cursor`
- Added explicit loader schema constants for bar/source merge keys, bar loader columns, and source identity aliases.
- Added explicit state source prefix/value-candidate maps while keeping family-specific source feature math explicit.

Verified:

```text
All checks passed!
27 passed
module layout check passed
ad-hoc verification passed: scanner_backfill deleted, retired coverage helpers absent, coverage planning/allocation still works
```
