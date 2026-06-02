# OKX API Notes

Date: 2026-06-01

Python SDK: `python-okx`

Docs: <https://www.okx.com/docs-v5/zh/>

## Instruments

| `instType` | Meaning |
|---|---|
| `SPOT` | Spot. |
| `MARGIN` | Margin. |
| `SWAP` | Perpetual swap. |
| `FUTURES` | Delivery futures. |
| `OPTION` | Options. |

## Market Data

Candles:

```python
md.get_candlesticks(instId, bar, after, before, limit)
md.get_history_candlesticks(instId, bar, after, before, limit)
md.get_index_candlesticks(instId, bar, limit)
md.get_mark_price_candlesticks(instId, bar, after, before, limit)
```

Order book and trades:

```python
market_data_api.get_order_book(instId, sz=None)
market_data_api.get_trades(instId, limit=None)
market_data_api.get_trades_history(instId, after, before, limit)
```

Funding and public data:

```python
pub.get_instruments(instType="SWAP", instId=None)
pub.get_funding_rate(instId)
pub.get_funding_rate_history(instId, after, before, limit)
pub.get_open_interest(instId)
pub.get_historical_market_data(module, instType, dateAggrType, begin, end, ...)
```

Historical market-data modules:

| Module | Data |
|---:|---|
| `1` | Trades. |
| `2` | 1m candles. |
| `3` | Funding rate. |
| `4` | 400-level order book. |
| `5` | 5000-level order book. |
| `6` | 50-level order book. |
| `11` | Borrow interest rate. |

## Candle Notes

- Instrument candles begin at instrument listing.
- Index candles can predate swap/futures listings.
- Standard bars include `1m`, `5m`, `15m`, `1H`, `4H`, `1D`, `1W`, `1M`.
- Candle fields are `ts`, `o`, `h`, `l`, `c`, `vol`, `volCcy`, `volCcyQuote`, `confirm`.
- `confirm = "1"` means the candle is closed.

## Accumulation Scanner Mapping

These public endpoints can populate offline accumulation scanner evidence without API secrets. See `docs/architecture/accumulation.md` for scanner boundaries and missing-data behavior.

| Need | Endpoint | Historical? | Scanner Use |
|---|---|---|---|
| Universe | `/api/v5/public/instruments` | Current | Discover swap symbols and contract metadata. |
| Liquidity | `/api/v5/market/tickers` | Current | Rank candidates before deeper collection. |
| Price structure | `/api/v5/market/candles`, `/api/v5/market/history-candles` | Yes | Returns, moving averages, range, volatility, and drawdown. |
| Book support | `/api/v5/market/books` | Current or sampled | Depth imbalance, spread, slope, and bid support. |
| Recent tape | `/api/v5/market/trades`, `/api/v5/market/history-trades` | Recent or limited | Buy ratio, large-sell absorption, and resilience context. |
| Funding | `/api/v5/public/funding-rate-history` | Limited history | Funding crowding context. |
| Open interest | `/api/v5/public/open-interest` | Current unless sampled | Derivatives positioning context. |
| Deep history | Historical market data modules | Downloadable archives | Optional replay source for trades, candles, books, and funding. |

## Signal Bot

qooi uses OKX Signal Bot as a server-driven execution path. TP/SL are set on strategy creation, not per order.

Core endpoints:

| Endpoint | Purpose |
|---|---|
| `POST /tradingBot/signal/create-signal` | Create signal channel. |
| `POST /tradingBot/signal/order-algo` | Create signal strategy. |
| `GET /signal/positions` | Query server-side position. |
| `POST /signal/sub-order` | Push entry event. |
| `POST /signal/close-position` | Close position. |

qooi wrapper: `src/qooi/exchange/trading.py`.
