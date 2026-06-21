# Conflict gray-zone abstention plan

## Inspection

The current report now hides the losing opposite direction, but the promoted row still says:

```text
resolved opposite direction: up
```

That is still not convincing. It means the model found enough evidence for both sides on the same symbol.

This is a gray-zone prediction, not a clean opportunity.

## Why the gray zone exists

Tailtree trains/evaluates up and down as separate event surfaces:

```text
up event model
 down event model
h24 / h48 horizons
```

For volatile shallow-life altcoins, the same current state can historically precede both:

```text
large upward excursions
large downward excursions
```

That can happen because:

```text
1. h24/h48 path labels overlap in time
2. extreme altcoins are high-volatility, both-tail regimes
3. score buckets are historical state buckets, not a deterministic side classifier
4. utility and lift can disagree
```

So the problem is not merely a report duplicate. The symbol itself is ambiguous.

## Decision

For promotion, remove gray-zone symbols.

Do not just lag rank.

Reason:

```text
rank penalty can still leave a conflicted high-lift symbol in top_3;
for scanner promotion, ambiguity should become abstention.
```

Ponytail rule:

```text
If a symbol has material opposite-direction evidence, do not promote it.
Keep it as watch with reason: direction conflict.
```

## Material conflict definition

Use existing thresholds. No new config.

An opposite side is material when it passes the same basic evidence gates:

```text
support >= selection.min_selected_observation_count
tail_lift >= selection.min_valid_tail_lift
rank_score > 0
```

If no tailtree selection config exists, fallback:

```text
support > 0
tail_lift > 1
rank_score > 0
```

## Implementation

Patch only `review_decisions()`:

```text
1. inspect all ranked rows before symbol collapse
2. build conflict_symbols for symbols with material up and down evidence
3. keep the one best side for display, as today
4. if symbol in conflict_symbols, force action=watch before top_k promotion
5. reason: direction conflict: down vs up; abstain from promotion
```

No model retraining. No new artifact. No new config.

## Expected report behavior

Before:

```text
✅ RE-USDT-SWAP down ... resolved opposite direction: up
```

After:

```text
👀 RE-USDT-SWAP down ... direction conflict: down vs up; abstain from promotion
```

The top promoted list backfills with clean symbols if available.

## Verification

```bash
uv run --no-sync python -m ruff check src tests scripts/scanner_backfill.py scripts/scanner_potential.py
uv run --no-sync python -m ty check
uv run --no-sync python -m pytest tests/test_scanner_workflow_migration.py tests/test_state.py -q
uv run --no-sync python -u scripts/scanner_potential.py --config configs/potential-advanced-tailtree.toml
```
