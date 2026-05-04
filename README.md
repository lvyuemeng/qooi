# qooi

Crypto quantitative trading toolkit built on **OKX** (spot/perpetuals) — market data, backtesting, and strategy evaluation in a local-first, Polars-native pipeline.

```bash
uv sync
uv run python scripts/demo.py
```

## Pipeline

```text
OKX REST API
    ↓ MarketData.fetch()          ← public OHLCV, no auth
    ↓ CacheStore.refresh()        ← Parquet cache (avoid re-fetching)
    ↓ add_indicators()            ← SMA, EMA, RSI, ATR, Bollinger Bands
    ↓ signal expression           ← sma_cross / ema_cross / bollinger
    ↓ Backtest.run()              ← vectorized with slippage, spread, borrow cost
    ↓ compute_metrics()           ← Sharpe, Sortino, win rate, IC/IR, drawdown…
    ↓ plot()                      ← equity curve + daily returns + signal markers
```

Everything runs locally — the only remote call is to the OKX public API.

## Features

| Layer | Location | API key |
|-------|----------|---------|
| OHLCV / ticker / order book | `src/qooi/exchange/market.py` | No |
| Place / cancel / amend orders | `src/qooi/exchange/trading.py` | Yes (`.env`) |
| Parquet data cache | `src/qooi/exchange/store.py` | No |
| Technical indicators | `src/qooi/exchange/indicator.py` | No |
| Vectorized backtest (costs + walk-forward) | `src/qooi/exchange/backtest.py` | No |
| Full pipeline (load → backtest → plot) | `src/qooi/exchange/pipeline.py` | No |
| Strategy evaluation | `src/qooi/exchange/eval.py` | No |
| Signal expressions | `src/qooi/strategies/` | No |

## Usage

```bash
# Full pipeline in one call:
uv run python scripts/demo.py

# Custom strategy:
uv run python -c "
from qooi.exchange.pipeline import Pipeline
from qooi.strategies import bollinger_signal
s = Pipeline().run('ETH-USDT', '4H', days=60, capital=5000, signal_expr=bollinger_signal(20, 2))
print(s.eval)
"

# Place a live order (requires .env):
uv run python scripts/trade_okx.py

# Switch between live / testnet API keys:
uv run python -c "import shutil; shutil.copy2('.env.test', '.env')"
```

## Project structure

See `AGENTS.md` for full conventions (coding style, structures, lint rules).
