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

## Architecture: Four-Layer Pipeline

```
┌──────────────────────────────────────────────────┐
│           4. Exits — Stop / Target / Trail / Time │
├──────────────────────────────────────────────────┤
│           3. Recovery — Grid / Martingale / Hedge │
├──────────────────────────────────────────────────┤
│           2. Basket — Dedup / Exposure / Limit   │
├──────────────────────────────────────────────────┤
│           1. Signal — Registry → Strategy Dispatch│
└──────────────────────────────────────────────────┘
```

Data flow: OHLCV → add_indicators → pipeline.process_bar() → list[BasketAction] → Executor.

Same pipeline for backtest (BacktestExecutor.simulate) and live (LiveExecutor.execute).

### Layer 1: Signal (`core/registry.py`, `core/indicators.py`)

Strategy registry maps names (`momentum_1h`, `rsi_reversion`) to signal functions.
`indicators.py` provides `compute_momentum_1h()`, `compute_rsi_reversion_1h()` —
full state-machine pipelines that re-run on each invocation (matches backtest exactly).

### Layer 2: Basket (`core/basket.py`)

`BasketManager` isolates parallel signals per instrument. Each Basket tracks
entry price, size, position state, recovery level, and trail data.
`BasketAction` is the unified action type consumed by all executors.

### Layer 3: Recovery (`core/recovery.py`)

Grid, martingale reversal, and hedge strategies for drawdown recovery.
Activated when a basket exceeds loss thresholds. Produces BasketActions
for grid adds, direction reversals, and opposing hedges.

### Layer 4: Exits (`core/exits.py`)

Tiered exit evaluation in priority order: hard stop → trailing stop →
breakeven → target → time. `TrailTracker` maintains highest/lowest since
entry for trailing stop calculation.

### Executor (`core/executor.py`)

Two executors consume the same `list[BasketAction]`:
- `LiveExecutor` — `place_order``, ``cancel_order`, ``amend_order` via direct OKX TradeAPI
- `BacktestExecutor` — simulate fills against OHLCV bars, track equity curve

### Current Strategies

| Strategy | Asset | Signal | Recovery | Exit |
|----------|-------|--------|----------|------|
| `momentum_1h` | ETH | 6-bar return > 0.3%, ADX>20, vol>1.5× | none | stop/target/trail/time |
| `rsi_reversion` | SOL | RSI(14)<30 bounce, uptrend | none | stop/target/RSI exit/time |

Both via `process_bar()` in `core/pipeline.py` — same entry point for backtest and live.

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
