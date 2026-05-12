# OKX API reference

Python SDK: [`python-okx`](https://pypi.org/project/python-okx/) (`pip install python-okx`)

Docs: <https://www.okx.com/docs-v5/zh/>

---

## Instrument types (`instType`)

| Value | Meaning |
|-------|---------|
| `SPOT` | Spot |
| `MARGIN` | Margin |
| `SWAP` | Perpetual swap |
| `FUTURES` | Futures / delivery |
| `OPTION` | Option |

## Account modes (`acctLv`)

| Value | Mode |
|-------|------|
| 1 | Spot mode |
| 2 | Contract (futures) mode |
| 3 | Cross-currency margin mode |
| 4 | Portfolio margin mode |

---

## Market data — candles

### REST endpoints (MarketData SDK)

```python
# Instrument candles (for any instType: SPOT, SWAP, FUTURES, OPTION)
md.get_candlesticks(instId, bar, after, before, limit)
# Up to 300 bars per call, limit max 300

md.get_history_candlesticks(instId, bar, after, before, limit)
# Up to 100 per page, paginate with `after` for older bars
# Goes years back for well-established instruments

# Index candles (underlying spot index, predates any swap listing)
md.get_index_candlesticks(instId, bar, limit)
# Up to 1440 candles per granularity
# instId e.g. "XAU-USDT" — the underlying index

md.get_index_components(instId)

# Mark price candles
md.get_mark_price_candlesticks(instId, bar, after, before, limit)
```

### Key insight: index vs instrument candles

- **Instrument candles** (`get_candlesticks*`): Only return data from the instrument's listing date. E.g. `XAU-USDT-SWAP` was listed 2025-12-31.
- **Index candles** (`get_index_candlesticks`): Return underlying spot index data, which exists independently and can predate any swap/futures listing. Use index ID like `XAU-USDT` (not swap ID).

### Supported bars

Standard: `1s`, `1m`, `3m`, `5m`, `15m`, `30m`, `1H`, `2H`, `4H`, `6H`, `12H`, `1D`, `2D`, `3D`, `1W`, `1M`, `3M`

UTC variants (daily boundary at 00:00 UTC): `6Hutc`, `12Hutc`, `1Dutc`, `2Dutc`, `3Dutc`, `1Wutc`, `1Mutc`, `3Mutc`

### WebSocket channels

```text
candle1m, candle5m, candle15m, candle30m, candle1H, candle2H, candle4H,
candle6H, candle12H, candle1D, candle2D, candle3D, candle5D, candle1W,
candle1M, candle3M, candle1s
```

UTC variants: `candle6Hutc`, `candle12Hutc`, `candle1Dutc`, `candle2Dutc`, etc.

WS path: `/ws/v5/business`

### Candle response fields

```text
ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm
```

`confirm = "1"` means the bar is confirmed (closed). Raw strings — cast to float.

---

## Account (Account SDK)

```python
account_api.get_account_balance(ccy=None)   # all currencies if ccy omitted
account_api.get_account_config()            # acctLv, position mode, etc.
account_api.get_positions()                 # current open positions
```

---

## Trading — place order (Trade SDK)

```python
trade_api.place_order(
    instId="BTC-USDT-SWAP",
    tdMode="isolated",   # isolated | cross | cash
    side="buy",          # buy | sell
    ordType="limit",     # market | limit | post_only | fok | ioc
    sz="100",            # quantity (contracts or currency)
    px="19000",          # price (required for limit orders)
    posSide="net",       # net | long | short (required for SWAP/FUTURES in long_short_mode)
)
```

### `tdMode` values

| Value | Meaning |
|-------|---------|
| `isolated` | Isolated margin |
| `cross` | Cross margin |
| `cash` | Non-margin (spot) |
| `spot_isolated` | Spot isolated (lead trading only) |

### `posSide` values

| Mode | posSide |
|------|---------|
| Net mode (buy/sell) | `net` |
| Long/short mode | `long` or `short` |

---

## Public data (PublicData SDK)

```python
# Instrument definitions
pub.get_instruments(instType="SWAP", instId=None)

# Funding rate
pub.get_funding_rate(instId)
pub.get_funding_rate_history(instId, after, before, limit)

# Open interest
pub.get_open_interest(instId)

# Mark price
pub.get_mark_price(instId)

# Limit price
pub.get_limit_price(instId)

# Position tiers (leverage info)
pub.get_position_tiers(instType, uly=None, instId=None, tdMode=None)

# Historical market data (downloadable ZIP files)
pub.get_historical_market_data(module, instType, dateAggrType, begin, end, ...)
```

### Module values for historical market data

| Value | Data type |
|-------|-----------|
| 1 | Trades |
| 2 | 1min candles |
| 3 | Funding rate |
| 4 | Order book depth (400-level) |
| 5 | Order book depth (5000-level) |
| 6 | Order book depth (50-level) |
| 11 | Borrow interest rate |

---

## Other useful endpoints

```python
# Order book
market_data_api.get_order_book(instId, sz=None)

# Ticker
market_data_api.get_ticker(instId)
market_data_api.get_tickers(instType=None)

# Recent trades
market_data_api.get_trades(instId, limit=None)
market_data_api.get_trades_history(instId, after, before, limit)
```

---

## Signal Bot (tradingBot)

The OKX Signal Bot is a server-driven execution engine. TP/SL are handled
autonomously by OKX — the client only sends entry signals and close-position
requests. This is the execution layer for qooi's 1H strategies.

### Setup (one-time, see `scripts/setup_signal.py`)

1. `POST /api/v5/tradingBot/signal/create-signal`
   - Creates a signal channel → returns `signalChanId` + token

2. `POST /api/v5/tradingBot/signal/order-algo`
   - Creates a signal strategy with entry params and exit TP/SL
   - `entryType: "3"` (fixed contracts), `subOrdType: "9"` (TradingView signal)
   - `exitSettingParam.tpSlType: "price"` — TP/SL as % price change from entry
   - Returns `algoId`

### Trading (per bar, see `scripts/trade.py`)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `GET /signal/positions` | GET | Query current position: `pos` field (+N = long, -N = short, "0" = flat) |
| `POST /signal/sub-order` | POST | Push an entry order (side, sz, px, ordType) — TP/SL NOT attached (set at creation) |
| `POST /signal/close-position` | POST | Close all positions for this instrument |
| `GET /signal/orders-algo-details` | GET | Bot-level P&L, config, frozen balance (not position state) |

### Key differences from regular OKX trading

- **TP/SL are set at strategy creation**, not per-order. The server handles them
  autonomously — no trailing stop, no time stop, no client-side risk management
  needed for basic stop/target.
- **`signal/positions`** is the server-side source of truth for position state.
  It tells you quantity, average price, P&L per instrument. This is what qooi
  uses instead of file-persisted `_last_side` (stateless GitHub Actions fix).
- **`sub-order` pushes a signal event**, not a direct order. The signal bot
  translates the signal into orders based on the strategy config (entryType,
  contract sizing, etc.).
- **Signal channels** are webhook endpoints that can also receive TradingView
  alerts. qooi uses them programmatically via the REST API (not webhooks).

### Python SDK

qooi wraps these in `TradingClient` (`src/qooi/exchange/trading.py`):

```python
tc.signal_create(name, desc)                    → signalChanId
tc.signal_create_order_algo(chan_id, ...)       → algoId
tc.signal_get_positions(algo_id)                → [{instId, pos, avgPx, ...}]
tc.signal_push_sub_order(algo_id, ...)          → enter
tc.signal_close_position(algo_id, ...)          → exit
tc.signal_stop(algo_id, chan_id)                → cancel algo
```
