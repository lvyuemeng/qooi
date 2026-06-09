# Research Architecture

## Purpose

Research modules build deterministic, known-at-close diagnostic tables and promotion evidence. The active research direction is manual-classifier-first. AI/learned-state research is isolated in `qooi.dynamic` and documented separately.

## Owned modules

```text
src/qooi/research/config.py           # strict command config
src/qooi/research/data.py             # cache-backed frame prep
src/qooi/research/artifacts.py        # artifact bundle and schema helpers
src/qooi/research/patterns.py         # shared table pipe
src/qooi/research/behavior_tables.py  # behavior-state diagnostics and taxonomy
src/qooi/research/candidates.py       # candidate pattern diagnostics
src/qooi/research/rule_primitives.py  # taxonomy-derived rule primitives
src/qooi/research/reports.py          # command-facing report helpers
```

## Responsibilities

- Build deterministic known-at-close research frames.
- Construct pattern, outcome, metric, and scored-pattern tables.
- Keep future returns/outcomes out of state construction.
- Preserve empty promotion exports as valid rejection evidence.
- Produce artifact bundles, warnings, and summaries for command-facing reports.
- Support promotion analysis before any idea becomes a normal strategy signal.

## Shared pipe

```text
source/prepared wide frames
  -> ResearchFrame
  -> PatternTable
  -> OutcomeTable
  -> MetricTable
  -> ScoredPatternTable
  -> ArtifactBundle
```

| Contract | Role |
|---|---|
| `ResearchFrame` | Known-at-close state/event rows. No future labels. |
| `PatternTable` | Candidate grouping units. No forward labels. |
| `OutcomeTable` | First table where future returns may appear. |
| `MetricTable` | Aggregate measurements before gates. |
| `ScoredPatternTable` | Metrics plus candidate/promotion fields. |
| `ArtifactBundle` | Named tables, summary, warnings, metadata. |

## Non-responsibilities

- No live trading authorization.
- No basket lifecycle mutation.
- No scanner-specific candidate ranking ownership.
- No provider/source collection outside research-data preparation.
- No AI/learned-state training; that belongs in `qooi.dynamic`.

## Allowed dependencies

- `qooi.research` modules.
- `qooi.exchange` cache/data modules through `research.data`.
- `qooi.strategies` for deterministic classifier/signal features used by research tables.
- `qooi.core` only in explicit report/backtest orchestration paths.

## Forbidden dependencies

- Table-building modules must not import executor, basket, trading clients, or strategy signal computation unless they are explicitly in a backtest/report orchestration path.
- Research tables must not call exchange APIs directly; data preparation owns cache/source loading.
- Research outputs must not authorize allocation.
- Dynamic learned states must not be mixed into deterministic research unless a future promotion changes this architecture explicitly.

## Stage 1: manual classifier / dynamic transitions

Inputs:

- known-at-close classifier labels;
- known-at-close liquidity event labels;
- deterministic context columns;
- forward returns only after pattern construction.

Default state columns:

```text
market_stage_reduced
h4_market_stage_reduced
structure_trend_state
```

Outputs:

```text
timeframe-classifier.csv
state-transition-graph.csv
transition-information.csv
transition-ngram-quality.csv
none-event-context-quality.csv
scored-patterns.csv
promotion-candidates.csv
```

## Promotion / integration boundary

A research candidate can become a strategy hypothesis only after no-lookahead, count, symbol, time-split, economic-materiality, invalid-state, and rationale gates pass. It must then become explicit strategy signal columns and pass execution-aware backtests.

Concrete research table/report surfaces live in `docs/graph/research.md`.
