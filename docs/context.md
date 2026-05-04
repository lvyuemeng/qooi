# qooi — domain context

## Domain

Quantitative trading strategy research on crypto (OKX exchange).

## Stack

- **Python management**: `uv` — no global Python state
- **Python version**: 3.11–3.13
- **DataFrame library**: Polars (primary), pandas only via `to_pandas()` for SDK interop

## Data source

OKX Market API — free, no auth for public market data. Instruments: SPOT, SWAP (perp), FUTURES, OPTION.

- `get_candlesticks` — up to 300 recent bars
- `get_candlesticks_history` — paginated, 100 per page, goes back years
- `MarketData` client in `src/qooi/exchange/market.py`
- `CacheStore` in `src/qooi/exchange/store.py` auto-paginates; call `refresh(days=1000, min_bars=2000)` for deep history

## Glossary

| Term | Definition |
|------|-----------|
| bar | OHLCV candle granularity (1m, 5m, 15m, 1H, 4H, 1D, etc.) |
| instrument | Product ID e.g. `BTC-USDT`, `XAU-USDT-SWAP` |
| SWAP | Perpetual swap (no expiry, funding rate) |
| ATR | Average True Range — volatility measure |
| ADX | Average Directional Index — trend strength |
| VM | VuManChu Swing Free — range filter channel strategy |

## Workflow

1. Fetch OHLCV via `CacheStore.refresh(inst_id, bar, days, min_bars)` → cached as Parquet
2. Add indicators via `add_indicators(df)`
3. Compute signal via strategy function (`sma_cross_signal`, `ema_vumanchu_signal`, `trend_pullback_signal`, etc.)
4. Run `Backtest` (single) or `WalkForwardBacktest` → metrics
5. Chart via `plot_backtest`

## Data depth notes

- For 1D: `days=1000` yields ~3 years (1100 bars) — enough for multi-filter strategies
- For 4H: `days=500` yields ~11 months (2000 bars)
- For 1H: `days=200` yields ~3 months (2000 bars)
- XAU-USDT-SWAP: only ~125 bars of 1D (listed 2025-12-31); use 4H for more data
