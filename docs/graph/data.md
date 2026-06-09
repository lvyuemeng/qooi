# Data / Exchange Module Graph

```text
qooi.exchange.market
  OkxSyncExchange
    bars() / bars_since()
    book()
    funding()
    archives()
  OkxAsyncExchange
    bars() / bars_since()
    book() / books()
    funding()
  CcxtSyncExchange
  CcxtBooksStream

qooi.exchange.store
  HistoryRequest
  HistoryRefreshRequest
  HistoryTarget
  HistoryCoverage
  CacheStore / AsyncCacheStore
    bars()
    funding()
    books()
    many()
    validate_history()

qooi.exchange.discovery
  discover_candidates()
  rank_discovery_frame()
  select_candidate_symbols()

qooi.exchange.universe
  collect_broad_sources()
  collect_potential_board_universe()
  collect_potential_universe()
  build_okx_first_potential_universe()

qooi.exchange.context
  collect_market_context(MarketContextRequest)
    -> books/trades/funding/open_interest/taker/long_short/rubik source frames

qooi.exchange.trading
  load_okx_env()
  TradingClient
  BotIdentity
```

Callers:

```text
scanner.workflow -> exchange.store + exchange.discovery + exchange.context
research.data    -> exchange.store
core/live paths   -> exchange.trading / exchange.market as IO only
```
