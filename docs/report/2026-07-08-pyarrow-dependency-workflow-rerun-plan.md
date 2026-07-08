# PyArrow dependency cleanup and workflow rerun plan

## Goal

Remove the direct heavy optional `pyarrow` dependency where possible, sync the uv environment, rerun the 01→02→03 tailtree workflow, then commit and push in small stages.

## Audit

Current `pyproject.toml` has no `pyarrow` entry in `feature-research`.

Code references:

- No direct `import pyarrow`.
- No direct pandas import.
- `src/qooi/scanner/tailrun/features.py` uses `DataFrame.to_pandas()` for tsflex integration; therefore pandas must remain available transitively through the feature-research stack unless the tsflex path is redesigned.

Decision:

- Remove direct `pyarrow` from project metadata/lock if workflow still runs.
- Do not remove pandas in this slice because the tsflex conversion path currently needs pandas-shaped input.

## Verification plan

1. `uv lock --python 3.12`
2. `uv sync --python 3.12 --dev --group tailtree --group feature-research`
3. Check `pyproject.toml` has no `pyarrow` direct dependency.
4. Run focused checks:
   - `uv run ruff check ...`
   - `uv run pytest ...`
   - `uv run python scripts/check_module_layout.py`
5. Rerun workflow:
   - `uv run python scripts/01_build_features.py`
   - `QOOI_TRAIN_MAX_TRIALS=5 uv run python scripts/02_train.py`
   - `uv run python scripts/03_predict.py`
6. Verify board/report freshness and rows.

## Commit stages

1. `[chore]: remove direct pyarrow dependency`
2. If workflow outputs/stage docs need committing, stage them separately; otherwise report that generated `data/` artifacts were not committed.

## Implementation result

Completed:

- Removed direct `pyarrow` from `feature-research`.
- Kept and declared `pandas>=2.3` because tsflex feature extraction consumes pandas frames.
- Replaced Polars `.to_pandas()` calls in `src/qooi/scanner/tailrun/features.py` with `_pandas_frame(...)`, a simple list-backed conversion that does not require `pyarrow`.
- Regenerated `uv.lock` with Python 3.12 and synced `--group tailtree --group feature-research`.

Workflow rerun:

```text
uv run python scripts/01_build_features.py
QOOI_TRAIN_MAX_TRIALS=5 uv run python scripts/02_train.py
uv run python scripts/03_predict.py
```

Result:

```text
board_rows=30
fresh_rows=30
stale_rows=0
oldest_decision_age_hours=1.51
best_score=0.534399430293138
```
