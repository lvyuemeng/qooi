# Data / Exchange Architecture

## Purpose

The data/exchange layer fetches, normalizes, caches, and audits market/exchange data. It must not know strategy, scanner, basket, recovery, research-report, or promotion policy.

## Owned modules

```text
src/qooi/exchange/market.py      # resource-first OKX/CCXT market clients
src/qooi/exchange/store.py       # parquet cache, history planning, coverage validation
src/qooi/exchange/discovery.py   # swap candidate discovery/ranking inputs
src/qooi/exchange/universe.py    # broad/potential universe collection and OKX mapping
src/qooi/exchange/context.py     # OKX source-family market context dispatch
src/qooi/exchange/trading.py     # thin OKX trading/signal-bot IO wrapper
```

## Responsibilities

- Normalize exchange/provider JSON before the DataFrame boundary.
- Store/load OHLCV, books, funding, and related Parquet cache artifacts.
- Plan requested history horizon from days, minimum bars, and bar size.
- Validate actual coverage, gaps, duplicates, freshness, and refresh requirements.
- Return data-only coverage metadata and cache refresh events.
- Discover bounded universes and exchange instrument mappings.
- Keep trading IO thin and separated from strategy/scanner/research policy.

## Non-responsibilities

- No strategy indicator computation.
- No scanner ranking policy.
- No research promotion policy.
- No basket lifecycle mutation.
- No executor accounting.
- No report interpretation.
- No hardcoded secrets.

## Allowed dependencies

- Provider/CCXT/OKX clients and standard IO/cache helpers.
- `qooi.sources` only for current data-context dispatch and source result/manifest integration.

## Forbidden dependencies

- `qooi.scanner`
- `qooi.research` policy/report modules
- `qooi.strategies`
- `qooi.core.basket`
- `qooi.core.executor`
- `qooi.core.recovery`
- `qooi.dynamic`

## Core contracts

- `HistoryRequest`
- `HistoryRefreshRequest`
- `HistoryRefreshResult`
- `HistoryTarget`
- `HistoryCoverage`
- `MarketContextRequest`
- `MarketContextResult`
- `DiscoveryConfig`
- `PotentialUniverseRequest`

Resource vocabulary:

```text
bars() / bars_since()
book() / books()
funding()
archives()
```

## Source policy

Source policy is selected outside this layer:

- `swap`: swap signal and swap execution, default.
- `spot_signal_swap_exec`: spot signal with swap execution.
- `spot`: spot-only research approximation.

Concrete exchange/cache/universe mappings live in `docs/graph/data.md`.
