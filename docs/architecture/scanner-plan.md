# Scanner refinement plan

Derived from empirical scan analysis (160 symbols, 5m17s run time, 269 passing tests).

## Found: prior-cycle deliverables

- Removed backtest/replay pipeline entirely (97% missing holdout rate, no research signal)
- Compact diagnostic artifacts replacing multi-GB raw CSVs (2.7 GB → 3.5 MB)
- Evidence gate strengthened: market_background excluded, requires >= usable_unstable status
- Shared Polars expression helpers extracted to `qooi.scanner` package `__init__.py`
- Decision chain collapsed from 9 if/elif branches into 9-rule tuple table
- Baseline computation shared across evidence levels (10 → 2 group_by passes)
- Dead backtest schemas/functions removed from candidates module (-203 lines)
- Graph docs synced to current artifact names, API surface, and removed B backtest

## Active: code bugs

### Bug 1 — market_background still selected (evidence.py:907)

Root cause: `select_potential_evidence_level()` filters `best_status` with `selection_level_rank >= 1`, then joins `best_level` back to unscored records on `(outcome_horizon, statistical_direction, selection_status_rank)`. Market_background rows with the same status_rank rejoin via shared columns.

Fix: filter `scored` in the `best_level` join path with the same `(selection_level_rank >= 1)` expression.

### Bug 2 — negative information gain selected

Rows where conditioning INCREASES entropy (info_gain < 0) pass the selection gate. Conditioning that adds uncertainty should not be presented as evidence.

Fix: add `information_gain_bits > 0` and `transition_information_gain_bits > 0` to selection criteria.

### Bug 3 — `missing: context` column is constant noise

Every review row shows `missing: context`. The absence of a message source is structural, not per-coin.

Fix: remove from per-row table. Summarize once in scan scope.

## Active: report ergonomics

### Provenance labels on merged review rows

The current merged row puts decision-path outputs (Direction, Suggestion) beside evidence-path outputs (Info, Rank, Tier) without indicating which comes from where.

Proposed header:

```
| Tier | Symbol | Info | Rank | ↓ evidence ↑ | Dir | Suggestion | ↓ decision ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
```

Columns grouped under `← evidence →` and `← decision →` sub-headers with a legend line explaining each path's data provenance:

- Evidence: cross-coin historical state → outcome distributions, info gain in bits, composite rank
- Decision: coin-specific latest kline + sources, transition patterns, scanner rule table

### Tier thresholds relative to evidence distribution

Current T1/T2 thresholds are absolute (0.3/0.1 bits). Observed distribution:

```
transition_info_gain: mean 0.006, median 0.003, max 0.12, 75th %ile 0.009
information_gain:     mean 0.052, median 0.027, max 0.51, 75th %ile 0.107
```

Proposed: T1 = top decile, T2 = top quartile, computed from the current scan's distribution rather than hardcoded.

## Architecture decisions

### Train/detect separation — no coverage gate

Evidence is trained on ALL coins with outcome history. Detection applies that evidence to ANY coin regardless of its own history depth. Newly listed coins with shallow kline history still receive valid evidence scores from the cross-coin pool.

### Decision ≠ Evidence

The decision path (coin-specific, current-state, all sources) and evidence path (cross-coin, historical, kline-primary) answer different questions. They are two lenses on the same symbol, not one merged verdict. The report should present them as such.

## What we accept

- Crypto short-horizon returns have low predictability from kline-regime states. Baseline entropy ~1.30 bits out of max 1.585 — near-random. The scanner correctly measures this. No model to fix.
- Context source (messages/social) is structurally absent. Surface it once, not per-row.
- New coins with shallow history produce `transition_path_missing`. This is correct — the classifier can't extract n-gram patterns from insufficient data. Cross-coin evidence still applies.
