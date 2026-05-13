# qooi — domain context

## Domain

Quantitative trading strategy research on crypto (OKX exchange).
Perpetual futures (swap) for execution, spot data for signal computation.

## Stack

- **Python management**: `uv` — no global Python state
- **Python version**: 3.12
- **DataFrame library**: Polars (primary)
- **Exchange SDK**: python-okx (trading), ccxt (market data)
- **Validation**: pydantic models for data contracts

## Data source

OKX Market API — free, no auth for public market data.
Instruments: SPOT, SWAP (perp), FUTURES, OPTION.

- `get_candlesticks` — up to 300 recent bars via OKX SDK
- `get_candlesticks_history` — paginated, 100 per page, goes back years
- `MarketData` client in `src/qooi/exchange/market.py`
- `CacheStore` in `src/qooi/exchange/store.py` auto-paginates; call `refresh(days=1000, min_bars=2000)` for deep history
- CCXT backend via `CcxtBackend` supports LBank, Gate, KuCoin for 2000+ bars

## Current Strategies: 1H Dual-Strategy Ensemble

Two complementary strategies running on different assets, both at 1H timeframe.
TP/SL handled by OKX Signal Bot server-side. Client sends ENTER (sub-order) or
CLOSE (close-position) signals only.

### Momentum Burst (ETH-USDT-SWAP)

```text
1H OHLCV → add_indicators → momentum_1h_signal
         → 6-bar return > 0.3%, EMA50>EMA200, ADX>20, volume > 1.5× avg
         → session filter 08-22 UTC, trend maturity ≥20 bars
         → signal = 1 (long) / -1 (short) / 0 (flat)
```

Trend-following strategy: enters in the direction of established momentum,
backed by ADX trend strength and volume confirmation. Exits via OKX server-side
TP/SL or client-side trend-flip detection.

### RSI Reversion (SOL-USDT-SWAP)

```text
1H OHLCV → add_indicators → rsi_reversion_signal
         → RSI(14) < 30 → bounce > 25 with confirmation bar
         → EMA50>EMA200, ADX>20, session 08-22 UTC
         → signal = 1 (long) / 0 (flat)  [long only]
```

Mean-reversion strategy: buys oversold bounces within confirmed uptrends.
The confirmation bar rule prevents entries into continuing sell-offs.
Exits via OKX server-side TP/SL or client-side RSI > 50 / trend-flip.

### Why Two Strategies?

Backtest evidence (93 trades over 83 days):

- Momentum burst wins during trend accelerations (ETH, 67% WR)
- RSI reversion wins during sharp oversold bounces (SOL, 78% WR)
- They are complementary (uncorrelated entry triggers) and produce a
  smoother combined equity curve than either alone.

Key files:

- `src/qooi/strategies/momentum_1h.py` — 1H momentum burst state-machine with tiered exits
- `src/qooi/strategies/rsi_reversion.py` — 1H RSI mean-reversion state-machine
- `src/qooi/core/signal.py` — `compute_momentum_1h()`, `compute_rsi_reversion_1h()`
- `src/qooi/exchange/trading.py` — TradingClient + signal bot endpoints
- `scripts/trade.py` — live trading entry point (test, live)

## Glossary

| Term | Definition |
|------|-----------|
| bar | OHLCV candle granularity (1m, 5m, 15m, 1H, 4H, 1D, etc.) |
| instrument | Product ID e.g. `BTC-USDT`, `ETH-USDT-SWAP` |
| SWAP | Perpetual swap (no expiry, funding rate) |
| ct_val | Contract value in base currency (ETH=0.1, SOL=1, BTC=0.01) |
| ATR | Average True Range — volatility measure |
| ADX | Average Directional Index — trend strength (0-100, >20 = trending) |
| OFI | Order Flow Imbalance — directional volume fraction |
| IC | Information Coefficient — rank correlation of signal with forward returns |
| momentum burst | 6-bar directional persistence signal with volume and session filters |
| RSI reversion | Oversold bounce in confirmed uptrend, long-only, with confirmation bar |
| circuit breaker | 2 consecutive losses → suspend asset until 20-bar high/low break |
| trend maturity | EMA50/200 direction must persist for ≥20 consecutive bars before entry |
| signal bot | OKX server-driven execution engine — handles TP/SL autonomously |

## Data depth notes

- For 1D: `days=1000` yields ~3 years (1100 bars)
- For 4H: `days=500` yields ~11 months (2000 bars)
- For 1H: `days=200` yields ~3 months (2000 bars)
- OKX SDK limit: 300 bars per request. Use CCXT (LBank) or CacheStore for deep history.
- Signal cache accumulates via `MarketData.candles(cache=True)` — merges new bars with existing.
