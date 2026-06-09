# Core Basket / Execution Architecture

## Purpose

The core layer is the execution simulation, basket lifecycle, recovery, state, and evaluation boundary. It consumes prepared signal columns and market bars, produces or accepts basket actions, and accounts fills, fees, trades, equity, and diagnostics.

## Owned modules

```text
src/qooi/core/__init__.py    # BarMarket, BarSignal, PipelineContext, process_bar
src/qooi/core/basket.py      # Basket, BasketBook, BasketAction, exits, lifecycle
src/qooi/core/recovery.py    # recovery proposal policies
src/qooi/core/executor.py    # BacktestExecutor and LiveExecutor
src/qooi/core/evaluate.py    # diagnostics and report formatting
src/qooi/core/metrics.py     # metrics
src/qooi/core/state.py       # live/backtest basket state reconstruction
src/qooi/core/config.py      # AssetConfig, PairConfig, universes
src/qooi/core/plot.py        # diagnostic plotting helpers
```

## Responsibilities

- Convert prepared signal columns and market bars into action proposals.
- Own basket lifecycle mutation through `BasketBook`.
- Account accepted actions, fills, fees, trades, equity, and drawdown.
- Enforce basket caps, lifecycle transitions, and recovery proposal constraints.
- Produce execution-aware diagnostics and reports.
- Reconstruct soft/live state where explicitly requested.

## Pipeline contract

`process_bar()`:

- consumes one market bar and one signal context;
- evaluates existing baskets independently;
- returns `BasketAction` proposals;
- does not mutate baskets;
- does not account fills;
- does not apply lifecycle actions;
- does not advance trailing state or bars-held counters.

`BasketBook`:

- owns lifecycle mutation;
- enforces basket caps;
- applies accepted actions;
- advances untouched active baskets;
- provides immutable snapshots for executor accounting.

`BacktestExecutor`:

- computes or consumes signal columns;
- builds `PipelineContext`;
- accepts/accounts action proposals;
- applies lifecycle mutations after accounting;
- marks equity and drawdown;
- returns trades/equity/diagnostics.

## Non-responsibilities

- No exchange/source collection policy.
- No scanner evidence construction or ranking.
- No research promotion decision by itself.
- No learned-state training.
- No hidden mutation inside evaluation/report formatting.

## Allowed dependencies

- `qooi.strategies` for signal computation in executor convenience paths.
- `qooi.exchange.trading` and state-source IO only at explicit live/trading boundaries.
- Other `qooi.core` modules.

## Forbidden dependencies

- Scanner candidate/evidence modules as execution inputs.
- Dynamic learned-state modules as execution policy.
- Source provider modules for report/evaluation logic.

## Recovery boundary

Recovery policies return proposals only. Recovery does not mutate baskets or open/close positions directly. Unsized recovery exposure is blocked under research defaults.

## Evaluation boundary

Evaluation computes metrics and formats reports. It must not mutate execution state or silently compare incompatible runs.

Concrete execution/evaluation surfaces live in `docs/graph/core.md`.
