# Strategy / Signal Architecture

## Purpose

The strategy area transforms prepared research or market frames into explicit signal columns when a hypothesis is mature enough to evaluate through execution-aware backtests.

This area is intentionally marked **unstable**. Current computation is spread across several project parts, so `qooi.strategies` is not yet an ideal design boundary. Treat existing strategy modules as implementation details until a promoted hypothesis is expressed through a tested signal contract.

## Owned modules

```text
src/qooi/strategies/indicators.py # indicator predicates and feature helpers
src/qooi/strategies/structure.py  # deterministic market-structure feature/classifier helpers
src/qooi/strategies/semantics.py  # shared strategy/classifier semantic enums and columns
src/qooi/strategies/specs.py      # strategy specs, signal frame computation, diagnostics
src/qooi/strategies/catalog.py    # strategy registry/metadata
src/qooi/strategies/portfolio.py  # portfolio qualification/allocation helpers
```

## Responsibilities

- Build known-at-close indicator, structure, and signal columns.
- Express promoted hypotheses as explicit strategy signal contracts.
- Keep signal identity, signal strength, entry intent, position state, and exit intent separate.
- Provide strategy metadata and diagnostics for execution-aware testing.

## Required output columns

Every promoted strategy path should produce:

- `raw_entry_signal`
- `entry_signal`
- `position_signal`
- `exit_signal`
- `signal_strength`
- `signal_id`

## Signal rules

- Entry signals are event-like basket-opening candidates.
- Position signals are held directional thesis state.
- Exit signals are explicit strategy-owned exit intent.
- Signal strength is quality/confidence, not direction.
- Signal ID identifies the rule or hypothesis, not symbol or basket ID.
- Strategy computation should be Polars-native and known-at-close.

## Non-responsibilities

- No fetching/cache writes.
- No source-provider calls.
- No scanner evidence construction or ranking.
- No research promotion gates.
- No basket caps.
- No sizing decisions.
- No fills/fees/equity accounting.
- No recovery mutation.
- No live trading authorization.

## Allowed dependencies

- Polars and pure computation helpers.
- Other `qooi.strategies` modules.

## Forbidden dependencies

- `qooi.exchange`
- `qooi.sources`
- `qooi.scanner`
- `qooi.dynamic`
- `qooi.core.basket`
- `qooi.core.executor`
- `qooi.core.recovery`

## Promotion / integration boundary

Research/scanner findings must be adapted into explicit signal columns before they can be tested through `qooi.core` execution. Until that happens, they remain research artifacts, not strategies.

Concrete strategy implementation surfaces live in `docs/graph/strategy.md`.
