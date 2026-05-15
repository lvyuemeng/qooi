# qooi

Crypto quantitative trading toolkit for OKX swap research and signal-bot execution.
The current research focus is 1H mean-reversion with fixed, robust, and adaptive
statistical indicators.

```bash
uv sync
uv run python scripts/trade.py testnet
```

## Architecture

```text
OKX REST API (public, no auth)
    ↓ CacheStore.load_history()         ← planned OHLCV target + coverage audit
    ↓ strategies.compute_signal_frame() ← indicators + composable StrategySpec
    ↓ core.process_bar()                ← Bar context → BasketAction proposals
    ↓ BasketBook / Executor             ← lifecycle mutation, fills, fees, equity
```

Signals run identically in backtest and live. Strategy selection is a runtime input,
not part of pair/execution config.

## Active Strategies

| Strategy | Asset | Timeframe | Entry Logic | Exit Logic |
|----------|-------|-----------|-------------|------------|
| `zscore_mean_reversion` | configured swap assets | 1H | fixed rolling close Z-score extremes with ADX gate | Z-score reversion plus basket exits |
| `robust_zscore_mean_reversion` | configured swap assets | 1H | rolling median/MAD Z-score extremes with ADX gate | robust Z-score reversion plus basket exits |
| `adaptive_zscore_mean_reversion` | configured swap assets | 1H | blended fixed/EWMA/robust Z-score extremes with ADX and volatility-ratio gates | dynamic Z-score reversion plus basket exits |
| `rsi_bounce_reversion` | configured swap assets | 1H | RSI oversold bounce with trend/session filters | RSI/trend thesis failure plus basket exits |
| `momentum_burst` | configured swap assets | 1H | 6-bar momentum, trend, volume, ADX, and session filters | trend thesis failure plus basket exits |

Strategies use tiered exits in backtest: hard stop, target, trailing stop, time stop,
and explicit strategy exit. In live trading, the OKX Signal Bot handles stop-loss and
take-profit autonomously; the client sends entry and close signals.

## Data Readiness

Research backtests request `730` days and `12,000` bars by default. Exchange retention
or pagination may return less, so every run includes `HistoryCoverage` metadata and
cache warnings. Use `--refresh-cache` to fetch again and `--min-coverage-pct` to fail
fast when data is too shallow.

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
uv run python scripts/backtest.py --strategy robust_zscore_mean_reversion --symbol ETH-USDT-SWAP --diagnostics

# Refresh cache and require at least 90% target coverage
uv run python scripts/backtest.py --strategy robust_zscore_mean_reversion --symbol ETH-USDT-SWAP --refresh-cache --min-coverage-pct 90

# Compare current benchmark set
uv run python scripts/backtest.py --benchmark --diagnostics --data-source swap

# Sweep recovery profiles
uv run python scripts/backtest.py --mode base
uv run python scripts/backtest.py --mode grid
uv run python scripts/backtest.py --mode martingale
uv run python scripts/backtest.py --mode hedge

# Custom backtest:
uv run python -c "
from qooi.core.executor import BacktestExecutor
from qooi.core.config import PAIRS
from qooi.strategies import robust_zscore_mean_reversion_spec
import polars as pl

df = pl.read_parquet('data/cache/ETH_USDT_SWAP_1H.parquet')
pair = PAIRS[0]
bt = BacktestExecutor(initial_capital=pair.asset.capital)
report = bt.run_report(df, pair, strategy=robust_zscore_mean_reversion_spec())
print(report.summary())
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
  core/              ← __init__.py (process_bar), basket.py, config.py,
                       executor.py, evaluate.py, metrics.py, recovery.py,
                       state.py, styles.py
  exchange/          ← market.py, trading.py, store.py
  strategies/        ← specs.py, features.py, conditions.py, indicators.py,
                       portfolio.py
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
