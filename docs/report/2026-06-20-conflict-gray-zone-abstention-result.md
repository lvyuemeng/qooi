# Conflict gray-zone abstention result

## Decision

Use abstention, not rank lagging.

```text
If a symbol has material up and down evidence, do not promote it.
Keep it as watch with direction-conflict reason.
```

## Why not just lag rank

Rank penalty is too weak for this failure mode:

```text
high-lift conflicted symbols can still remain top_3 after a penalty
```

The issue is epistemic ambiguity:

```text
the same current state maps to historically meaningful up-tail and down-tail buckets
```

That is not a weaker signal. It is a gray-zone signal.

Ponytail answer:

```text
abstain from promotion
```

## Applied code

Changed:

```text
src/qooi/scanner/output.py
tests/test_scanner_workflow_migration.py
```

Implementation:

```text
1. inspect all current ranked rows before same-symbol collapse
2. mark symbols with material evidence in both directions
3. keep one best display row per symbol
4. force conflicted symbols to watch before top_k promotion
5. backfill promotion slots with clean symbols
```

Material conflict uses existing selection thresholds:

```text
support >= selection.min_selected_observation_count
tail_lift >= selection.min_valid_tail_lift
rank_score > 0
```

No new config, no retrain, no extra artifact family.

## Verification checks

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

## Real scanner verification

Command:

```bash
uv run --no-sync python -u scripts/scanner_potential.py --config configs/potential-advanced-tailtree.toml
```

Runtime:

```text
CONFLICT_ABSTAIN_VERIFY_SECONDS=270
```

Snapshot:

```text
data/output/potential/benchmarks/conflict-abstain-report.md
data/output/potential/benchmarks/conflict-abstain-tailtree-selection-efficiency.csv
```

## New candidate board behavior

Previously top conflicted symbols were promoted.

Now:

```text
O-USDT-SWAP    watch | direction conflict: down vs up; abstain from promotion
RE-USDT-SWAP   watch | direction conflict: down vs up; abstain from promotion
BEAT-USDT-SWAP watch | direction conflict: down vs up; abstain from promotion
```

Promoted rows are backfilled by non-conflicted candidates:

```text
HOME-USDT-SWAP down h24
OFC-USDT-SWAP  up h48
SHIB-USDT-SWAP down h24
```

Programmatic check:

```text
promoted: 3
conflict_watches: 8
promoted_conflicts: []
```

Report blocker summary:

```text
direction conflict: 11 symbols
outside promote top_3: 37 symbols
```

## Interpretation

The gray zone is expected in volatile shallow-life altcoins because the up/down tailtree surfaces are independent path-event models. A symbol can be a high-volatility both-tail bucket rather than a clean directional candidate.

Scanner promotion should select clean directional asymmetry, not both-tail ambiguity.

## Next ponytail improvement

Do not add another model yet.

Next smallest improvement is report wording:

```text
Promote: clean directional candidates
Watch: conflict/gray-zone candidates
```

If we later want to improve model confidence rather than only promotion filtering, benchmark one narrow addition:

```text
directional dominance = best_side_quality - opposite_side_quality
```

But only after this abstention behavior is accepted.
