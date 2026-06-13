# Exchange Architecture

## Purpose

`qooi.exchange` owns exchange-facing IO and exchange cache state. It does not own source-family demand, scanner source context, source artifact semantics, strategy computation, research interpretation, basket lifecycle, executor accounting, or learned-state logic.

## Module layout

| Module | Ownership |
|---|---|
| `qooi.exchange.market` | Low-level market resource clients and provider resource vocabulary. |
| `qooi.exchange.store` | OHLCV/cache paths, history planning, cache merge, coverage validation. |
| `qooi.exchange.discovery` | OKX instrument/ticker discovery for exchange-backed symbols. |
| `qooi.exchange.universe` | OKX-first universe mapping and temporary broad-source ranking bridge. |
| `qooi.exchange.trading` | Thin trading/signal-bot IO and environment loading. |

Removed target: `qooi.exchange.context`. Source context collection is source demand, not exchange architecture.

## Contracts

| Module | Contracts |
|---|---|
| `market` | `SyncExchange`, `AsyncExchange`, `OkxSyncExchange`, `OkxAsyncExchange`, `OkxBarsRequest`, `OkxBarsAudit`, `BookSnapshot` |
| `store` | `CacheStore`, `AsyncCacheStore`, `HistoryRequest`, `HistoryRefreshRequest`, `HistoryRefreshResult`, `HistoryCoverage`, `HistoryTarget` |
| `discovery` | `DiscoveryConfig`, `DiscoveryWorkflowConfig`, `DiscoveryResult` |
| `universe` | `PotentialUniverseRequest`, `PotentialUniverseResult`, broad workflow configs/results |
| `trading` | `TradingClient`, `BotIdentity`, `PositionState`, `load_okx_env()` |

## Dependency policy

```text
exchange.market    -> external clients / stdlib only
exchange.store     -> exchange.market
exchange.discovery -> sources.okx
exchange.universe  -> exchange.discovery + broad source provider wrappers
exchange.trading   -> external trading IO only
```

Forbidden dependencies:

```text
exchange -> scanner policy/evidence/report modules
exchange -> research policy/report interpretation
exchange -> strategies
exchange -> core basket/executor/recovery
exchange -> dynamic/learned-state modules
exchange -> sources.context / source collection orchestration
```

## Bar cache refresh ownership

`qooi.exchange.store` executes OHLCV/cache refresh requests, but it does not decide scanner source policy. Scanner workflow supplies a decisive `HistoryRefreshRequest` from top-level `PotentialConfig.refresh_mode`:

```text
refresh_mode="cache_only"   -> read cache, report coverage, no exchange fetch
refresh_mode="incremental"  -> refresh stale/missing cache incrementally
refresh_mode="force"        -> rebuild requested cache window
```

`[potential.source].refresh_mode` must not affect `exchange.store` bar requests. Source collection may inherit the top-level mode, but that inheritance is resolved in `qooi.sources.context`, not in exchange.

## Demand boundary

Exchange answers:

```text
Can this market resource be fetched?
Where is exchange cache stored?
Is exchange cache deep/fresh/valid enough?
Which OKX instrument maps to a symbol?
Can a thin trading IO call be made from environment credentials?
```

Exchange does not answer:

```text
Which source families does scanner need?
Which source artifact represents a family?
Which source rows count as current/history evidence?
Which source symbols should be fetched for scanner context?
```

Those belong to `qooi.sources`.

## Callers

```text
scanner.workflow -> exchange.store + exchange.discovery
research.data    -> exchange.store + exchange.market
research.reports -> exchange.store for cache audit only
live/core IO     -> exchange.market + exchange.trading
sources.okx      -> exchange.market only for low-level market helpers
```

Concrete target APIs live in `docs/graph/exchange.md`.
