# qooi — domain context

## Domain

Quantitative trading strategy research on crypto (OKX exchange).
Perpetual futures (swap) for execution, spot data for signal computation.

## Stack

- **Python management**: `uv` — no global Python state
- **Python version**: 3.12
- **DataFrame library**: Polars (primary)
- **Exchange SDK**: python-okx (trading), ccxt (market data)
- **Ty**: type checker for core modules
- **Ruff**: linter

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
┌──────────────────────────────────────────────────────┐
│  4. Exits — Stop / Target / Trail / Time / Breakeven │
├──────────────────────────────────────────────────────┤
│  3. Recovery — Grid / Martingale / Hedge             │
│     Recovery returns list[BasketAction].             │
│     Martingale: EXIT original + ENTER opposite.      │
│     Trailing/breakeven paused during active recovery.│
├──────────────────────────────────────────────────────┤
│  2. Basket — Dedup / Exposure / Limit / Sizing       │
│     BasketManager creates fully-sized Baskets.       │
│     BasketAction is the unified executor API.        │
├──────────────────────────────────────────────────────┤
│  1. Signal — OkxSignalConfig.compute() → Strategy    │
│     Backtest uses pre-computed signal column.        │
│     Live uses pair.okx.compute() → OKX API.          │
└──────────────────────────────────────────────────────┘
```

Data flow: OHLCV → add_indicators → core.process_bar() → list[BasketAction] → Executor.

Same pipeline for backtest (BacktestExecutor.run) and live (LiveExecutor.execute).

### Layer 1: Signal (`core/config.py` (OkxSignalConfig.compute), `core/indicators.py`)

`OkxSignalConfig.compute()` dispatches strategy names (`momentum_1h`, `rsi_reversion`)
to signal functions in `core/indicators.py`. Backtest mode uses pre-computed signal
column via `signal_src` parameter on `process_bar()` — no live API calls.

### Layer 2: Basket (`core/basket.py`)

`BasketManager` isolates parallel signals per instrument. Owns sizing
(`size_position()`), stop/target computation (`compute_stop_target()`),
and Basket lifecycle (`create()`, `remove()`). Each `Basket` tracks
entry price, size, position state, recovery level, trail data, and
cumulative loss. `BasketAction` is the unified action type consumed by
all executors. `basket.add_to_position()` handles weighted-average
entry price updates without caller mutation.

### Layer 3: Recovery (`core/recovery.py`)

Grid, martingale reversal, and hedge strategies for drawdown recovery.
Returns `list[BasketAction]` — may be empty, single action, or multiple
(EXIT + ENTER for martingale reversal). Martingale reversal size computed
from loss amount / zone ATR. `RecoveryKind` enum: NONE, GRID, MARTINGALE, HEDGE.

During active recovery, trailing/breakeven exits are paused (only hard
stop active) to avoid interference. Controlled by `basket.recovery_activated`
and `basket.recovery_level > 0` → passed as `skip_trailing=True` to
`evaluate_exits()`.

### Layer 4: Exits (`core/basket.py` — evaluate_exits() + TrailTracker)

Tiered exit evaluation in priority order: hard stop → trailing stop →
breakeven → target → time. `TrailTracker` maintains highest/lowest since
entry for trailing stop calculation. `ExitReason` enum includes STOP,
TRAILING, BREAKEVEN, TIME, SIGNAL_ENTRY, SIGNAL_FLIP, MARTINGALE,
HEDGE_DRAWDOWN, GRID_LEVEL, GLOBAL_LOSS_LIMIT.

### Executor (`core/executor.py`)

Two executors consume the same `list[BasketAction]`:
- `LiveExecutor` — `place()`, `cancel()`, `close_position()`, `amend()` via direct OKX TradeAPI. State persists to `data/state/baskets.json`.
- `BacktestExecutor.run()` — loops `process_bar()` with pre-computed signal column. Computes PnL from BasketAction stream. Tracks portfolio drawdown (5% stop). `run_report()` returns a `Report`.

### Backtest Styles (`core/styles.py`) — strategy-independent

Backtest styles run a `trades_fn` (any function producing `(trades, equity)`)
repeatedly under different slicing regimes. Pure functions, zero strategy imports.

| Style | Function | Description |
|-------|----------|-------------|
| walk-forward | `walk_forward(trades_fn, df, train, test, step)` | Slide train→test windows, report OOS metrics |
| rolling window | `rolling_window(trades_fn, df, lookback, step)` | Fixed lookback window, slide forward |
| cross-validate | `cross_validate(trades_fn, df, folds)` | K-fold cross-validation across time segments |

All return `StyleResult` with `WindowSlice` list, combined OOS metrics, and stability stats.

### Evaluation (`core/evaluate.py`) — strategy-independent

Takes raw trades + equity and produces formatted output.

| Component | Description |
|-----------|-------------|
| `Report.from_raw(trades, equity, pair)` | Build from `BacktestExecutor.run()` output |
| `report.summary()` | Terse one-liner: trades, ret%, WR%, PL, Sharpe |
| `report.table()` | Multi-line aligned metrics block |
| `compare(*reports)` | Side-by-side ensemble comparison table |
| `format_table(headers, rows)` | Generic aligned-column formatter |

### Current Strategies

| Strategy | Asset | Signal | Recovery | Exit |
|----------|-------|--------|----------|------|
| `momentum_1h` | ETH | 6-bar return > 0.3%, ADX>20, vol>1.5× | grid/martingale/hedge (pipeline) | stop/target/trail/time |
| `rsi_reversion` | SOL | RSI(14)<30 bounce, uptrend | grid/martingale/hedge (pipeline) | stop/target/RSI exit/time |

Both via `process_bar()` in `core/__init__.py` — same entry point for backtest and live.

Recovery is available to all strategies through the pipeline's Layer 3.
Martingale reversal: closes original position and opens opposite with
computed size. During recovery, trailing/breakeven are paused to avoid
interfering with grid accumulation or reversal logic.

### Why Two Strategies?

Backtest evidence (93 trades over 83 days):

- Momentum burst wins during trend accelerations (ETH, 67% WR)
- RSI reversion wins during sharp oversold bounces (SOL, 78% WR)
- They are complementary (uncorrelated entry triggers) and produce a
  smoother combined equity curve than either alone.

Key files:

- `src/qooi/core/__init__.py` — `process_bar()` pipeline entry point
- `src/qooi/core/basket.py` — Basket, BasketManager, BasketAction, evaluate_exits
- `src/qooi/core/recovery.py` — RecoveryConfig, RecoveryKind, grid/martingale/hedge
- `src/qooi/core/executor.py` — LiveExecutor, BacktestExecutor, state persistence
- `src/qooi/core/config.py` — AssetConfig, OkxSignalConfig, PairConfig, PAIRS
- `src/qooi/core/indicators.py` — compute_momentum_1h(), compute_rsi_reversion_1h()
- `src/qooi/strategies/momentum_1h.py` — 1H momentum burst state-machine
- `src/qooi/strategies/rsi_reversion.py` — 1H RSI mean-reversion state-machine
- `src/qooi/exchange/trading.py` — TradingClient + signal bot + direct TradeAPI
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
| martingale reversal | EXIT original + ENTER opposite with computed size from loss |
| cumulative_loss | Total realized loss tracked per Basket |
| trailing pause | Trailing/breakeven disabled during active recovery to avoid interference |
| global loss limit | Force-close basket when cumulative loss exceeds threshold |
| portfolio drawdown stop | 5% drawdown from peak equity → force-close all baskets |
| signal_src | Backtest pre-computed signal column — bypasses live API |
| BasketAction | Unified payload: action kind, side, size, price, stop/target, reason |

## Data depth notes

- For 1D: `days=1000` yields ~3 years (1100 bars)
- For 4H: `days=500` yields ~11 months (2000 bars)
- For 1H: `days=200` yields ~3 months (2000 bars)
- OKX SDK limit: 300 bars per request. Use CCXT (LBank) or CacheStore for deep history.
- Signal cache accumulates via `MarketData.candles(cache=True)` — merges new bars with existing.
