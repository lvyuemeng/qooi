# Conflict path quality filter plan

## Problem

Current candidate board can show the same symbol on both sides:

```text
RE-USDT-SWAP down h24 promoted, utility=3.11
RE-USDT-SWAP up h24 watch, utility=7.78, reason=conflicting weaker up direction
```

This is confusing because `weaker` is based on `rank_score`, and `rank_score` is mostly:

```text
tail_lift + log(support) + stability + utility / 10
```

So utility has too little influence. A lower-utility side can beat a much higher-utility opposite side because lift is larger.

## Root cause

Two separate issues:

```text
1. review_decisions() keeps the losing opposite side as a watch row.
2. best_direction_by_symbol uses raw rank_score, not a side-quality score that balances lift/support/utility.
```

The report therefore presents conflict as a user-facing watch item instead of resolving it before promotion.

## Ponytail fix

Do not add new artifacts or frameworks.

Do one small boundary fix in `output.review_decisions()`:

```text
1. compute side_quality_score per row:
   tail_lift + log1p(support) + utility_proxy

2. keep only the best row per symbol before promote/watch/skip

3. if dropped opposite sides existed, append note to the winning row reason:
   resolved opposite direction: up/down
```

Why `utility_proxy` not `/10` here:

```text
candidate board is review/promotion, not HPO replay;
if two directions conflict on the same symbol, utility should matter strongly.
```

## What this changes

Before:

```text
same symbol can appear as promote and conflicting watch
```

After:

```text
one symbol, one side, one horizon row on candidate board
```

This does not retrain tailtree and does not change model evidence. It only filters the report/review surface.

## Verification

```bash
uv run python -m ruff check src tests scripts/scanner_backfill.py scripts/scanner_potential.py
uv run python -m ty check
uv run python -m pytest tests/test_output.py tests/test_state.py -q
```

If `uv run` is still blocked by the stale websocket install, use the project python executable for local tests and report the uv blocker explicitly.
