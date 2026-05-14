# qooi — domain context

## Domain

Quantitative trading strategy research on crypto (OKX exchange).
Perpetual futures (swap) for execution, spot data for signal computation.

## Stack

- **Python management**: `uv` — no global Python state
- **Python version**: 3.12
- **DataFrame library**: Polars (primary)
- **Exchange SDK**: python-okx (trading), ccxt (market data)
- **Ty**: type checker; run with `uv run ty check <path>`
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
│     BasketBook owns lifecycle state.                 │
│     BasketAction is the unified executor API.        │
├──────────────────────────────────────────────────────┤
│  1. Signal — Composable specs in qooi.strategies     │
│     Backtest/live precompute the signal column.      │
│     process_bar() consumes signal_src only.          │
└──────────────────────────────────────────────────────┘
```

Data flow: OHLCV → strategies.indicators.add_indicators() inside
strategies.compute_signal_frame() → core.process_bar(signal_src=..., strategy_id=...)
→ list[BasketAction] → Executor.

Same pipeline for backtest (BacktestExecutor.run) and live (LiveExecutor.execute).

### Layer 1: Signal (`strategies/compose.py`, `strategies/specs.py`)

Signal generation is composable and functionality-focused. Strategy specs combine
feature builders, entry conditions, filters, and hold policies. Strategy selection
is an orchestration input (`momentum_burst`, `rsi_bounce_reversion`, or retained
`flow_pipeline`), not a property of `PairConfig` or `OkxSignalConfig`. Old names
such as `momentum_1h` and `rsi_reversion` are intentionally rejected. `process_bar()`
receives a precomputed `signal_src` value and a runtime `strategy_id` label for
basket identity; it does not fetch market data or dispatch strategy functions.

### Layer 2: Basket (`core/basket.py`)

`BasketBook` owns basket lifecycle state for a pipeline run/session. `BasketManager`
is a stateless sizing/factory policy (`size_position()`, `compute_stop_target()`,
`create()`, `remove()`). Each `Basket` tracks entry price, size, position state,
recovery level, trail data, and cumulative loss. `BasketAction` is the unified
action type consumed by all executors. `basket.add_to_position()` handles
weighted-average entry price updates after execution actions are consumed.

### Layer 3: Recovery (`core/recovery.py`)

Grid, martingale reversal, and hedge strategies for drawdown recovery.
Returns `list[BasketAction]` — may be empty, single action, or multiple
(EXIT + ENTER for martingale reversal). Martingale reversal size computed
from contract-value loss / zone ATR value. `RecoveryKind` enum: NONE, GRID,
MARTINGALE, HEDGE.

Grid and hedge are experimental until recovery risk semantics are complete.
Grid compounds exposure and must respect `max_loss_pct`. Hedge requires explicit
hedge-group unwind logic; without that, it can freeze exposure rather than recover.

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
- `LiveExecutor` — action execution via OKX. Hard position/order truth comes from `OkxStateProvider`; JSON stores only soft strategy state.
- `BacktestExecutor.run()` — in-memory `BacktestStateProvider`, precomputed signal column, BasketAction PnL, and portfolio drawdown stop. `run_report()` returns a `Report`.

Validation commands:

```bash
uv run ruff check <path>
uv run ty check <path>
uv run pytest <path>
uv run python scripts/backtest.py --mode base|grid|martingale|hedge
```

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
| `report.summary()` | Terse one-liner: trades, WR%, PF, expectancy, trade/active-bar Sharpe |
| `report.table()` | Multi-line aligned metrics block |
| `compare(*reports)` | Side-by-side ensemble comparison table |
| `format_table(headers, rows)` | Generic aligned-column formatter |

### Current Strategies

| Strategy | Asset | Signal | Recovery | Exit |
|----------|-------|--------|----------|------|
| `momentum_burst` | any configured asset | N-bar momentum, trend/session/volume/structure filters | grid/martingale/hedge (pipeline) | stop/target/trail/time |
| `rsi_bounce_reversion` | any configured asset | RSI oversold bounce, uptrend/session/structure filters | grid/martingale/hedge (pipeline) | stop/target/trail/time |

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

## Backtest Status

- [x] Pipeline backtest runs end-to-end
- [x] Basket lifecycle verified by white-box tests
- [x] Recovery branches verified by white-box tests
- [x] Exit branches verified by white-box tests
- [x] Multiple sequential baskets reopen correctly
- [x] Multiple active baskets possible (hedge path)
- [x] Recovery modes validated by report workflow (`base`, `grid`, `martingale`, `hedge`)
- [ ] LiveExecutor places real direct orders
- [ ] Active-bar or trade-level robust Sharpe added

## Metric Caveat

Current Sharpe / Sortino are calendar-bar metrics on sparse equity series.
For low-frequency systems with many flat bars, annualized ratios can become
numerically extreme. `Report` now exposes trade-level metrics too:

- trade count
- win rate
- profit factor
- avg win / avg loss
- expectancy
- trade Sharpe
- median trade return

Use Sharpe / Sortino only as secondary diagnostics.

Current metric stack:

- primary: trade count, win rate, profit factor, avg win/loss, expectancy %, expectancy $
- secondary: trade Sharpe, median trade %, active-bar Sharpe, active-bar %
- tertiary: calendar Sharpe / Sortino / annualized return / annualized vol

Sparse-equity systems should be judged by primary + secondary metrics first.

Key files:

- `src/qooi/core/__init__.py` — `process_bar()` pipeline entry point
- `src/qooi/core/basket.py` — Basket, BasketBook, BasketManager, BasketAction, evaluate_exits
- `src/qooi/core/recovery.py` — RecoveryConfig, RecoveryKind, grid/martingale/hedge
- `src/qooi/core/executor.py` — LiveExecutor, BacktestExecutor
- `src/qooi/core/state.py` — OkxStateProvider, BacktestStateProvider, soft-state stores
- `src/qooi/core/config.py` — AssetConfig, OkxSignalConfig, PairConfig, PAIRS, RESEARCH_PAIRS
- `src/qooi/strategies/compose.py` — apply composable StrategySpec to OHLCV
- `src/qooi/strategies/specs.py` — composable strategy specs and canonical strategy names
- `src/qooi/strategies/indicators.py` — generic technical indicator precompute
- `src/qooi/strategies/features.py` — reusable feature builders
- `src/qooi/strategies/conditions.py` — reusable condition/filter expressions
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
