# qooi

Crypto quantitative trading toolkit — **1H momentum/mean-reversion strategies** executed via
**OKX Signal Bot** (server-driven TP/SL, client-side signal generation).

```bash
uv sync
uv run python scripts/trade.py testnet
```

## Architecture

```text
OKX REST API (public, no auth)
    ↓ MarketData.candles(cache=True)    ← 1H OHLCV, auto-cached to Parquet
    ↓ strategies.compute_signal_frame() ← indicators + composable StrategySpec
    ↓ core.process_bar()                ← Signal → Basket → Recovery → Exits
    ↓ list[BasketAction]                ← Unified action stream
    ↓ Executor (Live / Backtest)        ← place/cancel/amend or simulate fills
```

Signals run identically in backtest and live. Strategy selection is a runtime input,
not part of pair/execution config.

## Active Strategies

| Strategy | Asset | Timeframe | Entry Logic | Exit Logic |
|----------|-------|-----------|-------------|------------|
| `momentum_burst` | any configured asset | 1H | 6-bar return > 0.3%, EMA50>EMA200, ADX>20, vol > 1.5× avg, session 08-22 UTC | OKX TP/SL plus client close on trend-flip |
| `rsi_bounce_reversion` | any configured asset | 1H | RSI(14) < 30 → bounce > 25 with confirmation, EMA50>EMA200, ADX>20, session 08-22 UTC | OKX TP/SL plus client close on trend-flip or RSI > 50 |

Both strategies use tiered exits in backtest (hard stop, target, trailing, time stop).
In live trading, the OKX Signal Bot handles stop-loss and take-profit autonomously.
The client only sends **ENTER** and **CLOSE** signals.

## Backtest Results (1H Ensemble, 83 days)

| Asset | Strategy | Trades | Avg Ret | WR | PL | Sharpe |
|-------|----------|--------|---------|-----|----|--------|
| ETH | momentum_burst | 24 | +0.88% | 67% | 1.84 | +0.45 |
| SOL | rsi_bounce_reversion | 9 | +0.48% | 78% | 1.04 | +0.51 |
| **Ensemble** | **all** | **93** | **+0.21%** | **53%** | **1.32** | **+0.14** |

## Features

| Layer | Location | API key |
|-------|----------|---------|
| OHLCV / order book | `src/qooi/exchange/market.py` | No |
| Signal bot (OKX trading API) | `src/qooi/exchange/trading.py` | Yes (`.env`) |
| Parquet data cache | `src/qooi/exchange/store.py` | No |
| Technical indicators | `src/qooi/strategies/indicators.py` | No |
| Loop-based backtest (shared with live) | `src/qooi/core/executor.py` | No |
| Strategy evaluation | `src/qooi/core/evaluate.py` | No |
| Composable strategies | `src/qooi/strategies/` | No |

## Usage

```bash
# Auto-creates signal bot on first run, then trades every 1H
uv run python scripts/trade.py test

# Backtest a strategy on cached data
uv run python scripts/backtest.py --strategy momentum_burst

# Sweep recovery profiles
uv run python scripts/backtest.py --mode base
uv run python scripts/backtest.py --mode grid
uv run python scripts/backtest.py --mode martingale
uv run python scripts/backtest.py --mode hedge

# Custom backtest:
uv run python -c "
from qooi.core.executor import BacktestExecutor
from qooi.core.config import PAIRS
import polars as pl

df = pl.read_parquet('data/cache/ETH_USDT_1H.parquet')
pair = PAIRS[0]
bt = BacktestExecutor(initial_capital=pair.asset.capital)
trades, equity = bt.run(df, pair, strategy='momentum_burst')
print(f'Trades={len(trades)}  Final equity=${equity[-1]:.0f}')
"

# Switch between test / live:
uv run python scripts/trade.py test
uv run python scripts/trade.py live dry
```

## Environment

- **Python 3.12** via `uv`
- **Polars** DataFrame library
- **python-okx** SDK for trading
- **ty** for type checking
- GitHub Actions: 1H cron schedule on test environment

## Quality

```bash
uv run ruff check src/ tests/ scripts/   # lint
uv run ty check src/                      # type check
uv run pytest tests/ -v                   # unit + integration tests
```

## Project structure

```text
src/qooi/
  core/              ← __init__.py (process_bar), basket.py (Basket + exits),
                       decide.py (AssetConfig), config.py (PairConfig),
                       executor.py (Live + Backtest), indicators.py,
                       recovery.py (grid/martingale/hedge)
  exchange/          ← market.py, trading.py, store.py
  strategies/        ← compose.py, specs.py, features.py, conditions.py,
                       indicators.py, flow_pipeline.py, portfolio.py
scripts/
  trade.py           ← live trading entry point (auto-creates bot on first run)
  backtest.py        ← CLI backtest runner
docs/
  okx-api.md         ← OKX API reference
  testnet.md         ← test validation workflow
  context.md         ← domain context & glossary
  tests.md           ← instrument config & backtest results
```

## Live Deployment

1. Create OKX API keys with trade permissions on your test account
2. Set up `.env.test`:

   ```ini
   OKX_API_KEY_TEST=your_key
   OKX_SECRET_KEY_TEST=your_secret
   OKX_PASSPHRASE_TEST=your_passphrase
   OKX_FLAG=1
   ```

3. Run trade.py — auto-creates signal bot on first run, then trades every hour:

   ```bash
   uv run python scripts/trade.py test
   ```

4. After observing 1-2 weeks of paper-trading results, switch to live.

For production, copy `.env.test` → `.env.live` with live credentials and run:

```bash
uv run python scripts/trade.py live dry    # dry run first
uv run python scripts/trade.py live live   # real orders
```
