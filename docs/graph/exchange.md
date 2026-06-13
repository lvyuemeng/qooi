# Exchange API Graph

Target API graph for lean exchange modules.

## Package exports

```text
qooi.exchange
  -> OkxSyncExchange
  -> OkxAsyncExchange
  -> TradingClient
  -> load_okx_env
```

## Dependency graph

```text
exchange.market

exchange.store
  -> exchange.market

exchange.discovery
  -> sources.okx

exchange.universe
  -> exchange.discovery
  -> sources provider wrappers

exchange.trading
```

No target edge:

```text
exchange.context
exchange -> sources.context
exchange -> scanner/research/strategy policy
```

## `qooi.exchange.market`

Purpose: low-level provider market resources.

Contracts:

```text
SyncExchange
AsyncExchange
OkxSyncExchange
OkxAsyncExchange
OkxBarsRequest
OkxBarsAudit
BookSnapshot
```

Resource vocabulary:

```text
bars(request)
bars_since(request)
book(symbol, depth)
books(symbols, depth)
funding(symbol)
archives(...)
```

## `qooi.exchange.store`

Purpose: exchange cache and history coverage.

Contracts:

```text
CacheStore
AsyncCacheStore
HistoryRequest
HistoryRefreshRequest
HistoryRefreshResult
HistoryTarget
HistoryCoverage
CacheRefreshEvent
```

API:

```text
bar_refresh_request(
    symbol: str,
    timeframe: str,
    *,
    days: int,
    history_days: int,
    refresh_mode: RefreshMode,
) -> HistoryRefreshRequest

plan_history(request: HistoryRequest) -> HistoryTarget
validate_history(frame, target) -> HistoryCoverage
history_coverage_frame(coverages) -> DataFrame
history_coverage_row(coverage) -> dict
history_coverage_error_row(request, error) -> dict
bar_freshness_threshold_hours(timeframe) -> float
```

Refresh mapping:

```text
refresh_mode="cache_only"  -> HistoryRefreshRequest(refresh=False, cache_only=True)
refresh_mode="incremental" -> HistoryRefreshRequest(refresh=True, incremental=True)
refresh_mode="force"       -> HistoryRefreshRequest(refresh=True, incremental=False)
```

No exchange API accepts or reads `SourceConfig`.

Flow:

```text
HistoryRefreshRequest
  -> plan_history
  -> CacheStore / AsyncCacheStore
  -> exchange.market
  -> cache files
  -> HistoryRefreshResult + HistoryCoverage
```

## `qooi.exchange.discovery`

Purpose: OKX exchange-backed symbol discovery.

Contracts:

```text
DiscoveryConfig
DiscoveryWorkflowConfig
DiscoveryResult
```

API:

```text
discover_candidates(config) -> DiscoveryResult
rank_discovery_frame(frame, config) -> DataFrame
select_candidate_symbols(frame, limit) -> tuple[str, ...]
empty_discovery_frame() -> DataFrame
```

Flow:

```text
discover_candidates
  -> sources.okx.fetch_okx_instruments
  -> sources.okx.fetch_okx_tickers
  -> DiscoveryResult
```

## `qooi.exchange.universe`

Purpose: OKX-first universe mapping. Broad-source ranking remains here only as a temporary bridge until promoted or split.

Contracts:

```text
PotentialUniverseRequest
PotentialUniverseResult
BroadWorkflowConfig
BroadDiscoveryResult
```

API:

```text
collect_potential_universe(request) -> PotentialUniverseResult
collect_potential_board_universe(config) -> BroadDiscoveryResult
build_okx_first_potential_universe(...) -> PotentialUniverseResult
map_broad_to_okx(...) -> DataFrame
select_deep_symbols(...) -> tuple[str, ...]
select_board_pool_symbols(...) -> tuple[str, ...]
```

Flow:

```text
collect_potential_universe
  -> exchange.discovery.discover_candidates
  -> optional broad provider frames
  -> map_broad_to_okx
  -> PotentialUniverseResult
```

## `qooi.exchange.trading`

Purpose: thin live/trading IO.

Contracts/API:

```text
TradingClient
BotIdentity
PositionState
load_okx_env(...)
```

No strategy, scanner, basket, executor, or research policy lives here.

## Target callers

```text
scanner.workflow
  -> exchange.store
  -> exchange.discovery
  -> sources.context

research.data
  -> exchange.market
  -> exchange.store

research.reports
  -> exchange.store for cache audit

sources.okx
  -> exchange.market low-level helpers only

live/core IO
  -> exchange.market / exchange.trading
```
