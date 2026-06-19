# OKX API v5 truth surface

This document records the OKX endpoints and field mappings currently owned by `qooi.transport.okx`. It is provider/API truth for the scanner pipeline, not a TODO list.

## Base URLs

```text
REST: https://www.okx.com
WS:   wss://ws.okx.com:8443/ws/v5/public
```

## Owner module

```text
src/qooi/transport/okx.py
```

Public transport objects:

```text
OkxClient
OkxWsClient
SourceResult
SourceManifestRow
Manifest
okx_retry_policy
collect_okx_ws_books
```

Transport owns provider requests, response normalization, manifests, and sanitized provider warnings. Scanner/pipeline code decides cache policy, coverage policy, model training, ranking, and reporting.

## REST endpoints used

| Method | Endpoint | Client method | Purpose |
|---|---|---|---|
| GET | `/api/v5/market/candles` | `OkxClient.bars` | latest candle page |
| GET | `/api/v5/market/history-candles` | `OkxClient.history_candles`, `bars_since` | historical candles |
| GET | `/api/v5/market/books` | `OkxClient.book_snapshot` | current order-book snapshot |
| GET | `/api/v5/market/trades` | `OkxClient.recent_trades` | current recent trades |
| GET | `/api/v5/public/funding-rate-history` | `OkxClient.funding_history` | funding history |
| GET | `/api/v5/public/funding-rate` | `OkxClient.funding_rate` | latest funding rate |
| GET | `/api/v5/rubik/stat/contracts/open-interest-history` | `OkxClient.open_interest` | Rubik open-interest history |
| GET | `/api/v5/rubik/stat/taker-volume-contract` | `OkxClient.taker_volume` | Rubik taker-volume history |
| GET | `/api/v5/rubik/stat/contracts/long-short-account-ratio-contract` | `OkxClient.long_short_ratio` | Rubik long/short ratio history |
| GET | `/api/v5/public/instruments` | `OkxClient.instruments` | instrument discovery |
| GET | `/api/v5/market/tickers` | `OkxClient.tickers` | ticker/discovery liquidity surface |

## WebSocket endpoint used

```text
wss://ws.okx.com:8443/ws/v5/public
```

Current websocket path:

```text
OkxWsClient.connect
OkxWsClient.subscribe
OkxWsClient.messages
collect_okx_ws_books(symbols, max_samples=..., channel="books5")
```

## Field naming rule

OKX response fields are camelCase. `qooi.transport.okx` normalizes scanner/pipeline columns to snake_case before returning frames.

Examples:

```text
instId      -> inst_id
instType    -> inst_type
baseCcy     -> base_ccy
quoteCcy    -> quote_ccy
settleCcy   -> settle_ccy
ctVal       -> ct_val
ctValCcy    -> ct_val_ccy
listTime    -> list_time
bidPx       -> bid_px
askPx       -> ask_px
volCcy24h   -> quote_volume_24h
fundingRate -> funding_rate
fundingTime -> funding_time
```

## Candle rows

Endpoints:

```text
/api/v5/market/candles
/api/v5/market/history-candles
```

Input params:

```text
instId
bar
limit
after  # history pagination
```

`_parse_bars` uses the first six OKX row fields:

```text
timestamp, open, high, low, close, volume
```

Returned frame:

```text
timestamp: Int64
open: Float64
high: Float64
low: Float64
close: Float64
volume: Float64
```

## Discovery rows

Instrument endpoint:

```text
/api/v5/public/instruments?instType=SWAP
```

Ticker endpoint:

```text
/api/v5/market/tickers?instType=SWAP
```

Normalized discovery columns include:

```text
inst_id
inst_type
state
ct_val
list_time
last
bid_px
ask_px
quote_volume_24h
spread_bps
```

`pipeline.discovery.rank_discovery` consumes these normalized frames.

## Source result contract

Provider source methods return:

```text
SourceResult(
  frame: pl.DataFrame,
  manifest: pl.DataFrame,
  telemetry: pl.DataFrame,
)
```

Manifest columns:

```text
timestamp
symbol
source
phase
status
backend
endpoint
rows
range_start
range_end
coverage_pct
warning
stop_reason
```

Scanner reports should use materialized frame freshness/coverage for usability and manifest rows for provider provenance.

## Pagination / provider-bound caveats

- OKX funding history paginates older data through `after`.
- OKX Rubik history endpoints use `end` for older pages inside the loader/pipeline boundary.
- Rubik `1H` history has a provider lookback cap; near-cap fresh rows should be classified as `provider_bounded`, not missing.
- Books/trades are current snapshot/recent context unless a consistent historical artifact contract exists.

## Security boundary

Do not hardcode:

```text
API keys
secret keys
wallet/account labels
exchange account IDs
```

This doc is endpoint/field truth only.
