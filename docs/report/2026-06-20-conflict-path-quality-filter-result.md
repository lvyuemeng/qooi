# Conflict path quality filter result

## Problem confirmed

The candidate board showed same-symbol opposite-direction rows:

```text
RE-USDT-SWAP down h24 promoted, utility=3.11
RE-USDT-SWAP up h24 watch, utility=7.78, reason=conflicting weaker up direction
```

The word `weaker` was misleading. It meant lower `rank_score`, not lower all-around opportunity quality.

Current rank score is lift-heavy:

```text
tail_lift + stability + log1p(support) + utility / 10
```

So an opposite side with higher utility could still be marked weaker if its lift was lower.

## Applied fix

Changed `src/qooi/scanner/output.py` review filtering:

```text
1. group current candidate rows by symbol
2. choose one best side per symbol before promote/watch
3. tie-break same-symbol sides by:
   tail_lift + log1p(support) + utility_proxy
4. append note to the winning row if an opposite side was dropped
```

This keeps the board one-symbol/one-side and avoids presenting the losing conflict as another watch item.

## Why in output.py, not model/rank.py

This is a review/report-surface issue:

```text
model evidence may contain up and down rows
candidate ranking may inspect both
user-facing promotion should not show both as actionable watches
```

No tailtree retraining logic changed.

## Verification

Checks:

```bash
uv run --no-sync python -m ruff check src tests scripts/scanner_backfill.py scripts/scanner_potential.py
uv run --no-sync python -m ty check
uv run --no-sync python -m pytest tests/test_scanner_workflow_migration.py tests/test_state.py -q
```

Result:

```text
ruff: pass
ty: pass
14 passed
```

Report verification run:

```bash
uv run --no-sync python -u scripts/scanner_potential.py --config configs/potential-advanced-tailtree.toml
```

Runtime:

```text
CONFLICT_FILTER_VERIFY_SECONDS=371
```

New report candidate board:

```text
O-USDT-SWAP down h48 ... resolved opposite direction: up
RE-USDT-SWAP down h24 ... resolved opposite direction: up
BEAT-USDT-SWAP down h24 ... resolved opposite direction: up
```

Duplicate opposite-direction check over report rows:

```text
opposite_direction_duplicates: []
```

## Remaining note

This fix resolves conflict presentation. It does not change the model evidence or HPO surface. If we want the model/rank score itself to emphasize utility more, that should be a separate benchmarked change because it can affect promotion order and selection efficiency.
