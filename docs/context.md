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

## Current Strategy: OFI Flow (Spot-sourced, Swap-executed)

Signal is computed from spot candles (clean volume data) and executed on perpetual swaps
(positions persist, margin efficient).  Prices are 99.9% correlated between spot and swap.

```
spot OHLCV → add_indicators → add_regime_features → add_ofi_flow_columns
           → magnitude filter (|OFI| ≥ sig_threshold) → signal
           → execute on SWAP (limit entry, market exit)
```

Key files:
- `src/qooi/strategies/flow_pipeline.py` — OFI flow, regime features, regime gate
- `src/qooi/exchange/trading.py` — LiveExecutor, PortfolioRunner, TradingClient
- `scripts/trade.py` — entry point (testnet, live, backtest)

## Glossary

| Term | Definition |
|------|-----------|
| bar | OHLCV candle granularity (1m, 5m, 15m, 1H, 4H, 1D, etc.) |
| instrument | Product ID e.g. `BTC-USDT`, `ETH-USDT-SWAP` |
| SWAP | Perpetual swap (no expiry, funding rate) |
| ct_val | Contract value in base currency (ETH=0.1, SOL=1, BTC=0.01) |
| ATR | Average True Range — volatility measure |
| OFI | Order Flow Imbalance — directional volume fraction |
| IC | Information Coefficient — rank correlation of signal with forward returns |
| sig_threshold | Per-asset magnitude filter for OFI flow signals |

## Data depth notes

- For 1D: `days=1000` yields ~3 years (1100 bars)
- For 4H: `days=500` yields ~11 months (2000 bars)
- For 1H: `days=200` yields ~3 months (2000 bars)
- OKX SDK limit: 300 bars per request. Use CCXT (LBank) or CacheStore for deep history.
- Signal cache accumulates via `MarketData.candles(cache=True)` — merges new bars with existing.
