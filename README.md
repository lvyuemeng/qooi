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
    ↓ add_indicators()                  ← SMA, EMA, RSI, ATR, ADX, Bollinger Bands
    ↓ strategy signal (state-machine)   ← momentum_1h / rsi_reversion
    ↓ decide_idle / decide_active       ← shared decision engine
    ↓ OKX Signal Bot (TP/SL: server)    ← sub-order, close-position
```

Signals run identically in backtest and live — same state-machine, same indicators,
same `SignalResult` structure.

## Active Strategies

| Strategy | Asset | Timeframe | Entry Logic | Exit Logic |
|----------|-------|-----------|-------------|------------|
| `momentum_1h` | ETH | 1H | 6-bar return > 0.3%, EMA50>EMA200, ADX>20, vol > 1.5× avg, session 08-22 UTC | OKX TP 2.0% / SL 2.5%; client close on trend-flip |
| `rsi_reversion` | SOL | 1H | RSI(14) < 30 → bounce > 25 with confirmation, EMA50>EMA200, ADX>20, session 08-22 UTC | OKX TP 2.0% / SL 2.0%; client close on trend-flip or RSI > 50 |

Both strategies use tiered exits in backtest (hard stop, target, trailing, time stop).
In live trading, the OKX Signal Bot handles stop-loss and take-profit autonomously.
The client only sends **ENTER** and **CLOSE** signals.

## Backtest Results (1H Ensemble, 83 days)

| Asset | Strategy | Trades | Avg Ret | WR | PL | Sharpe |
|-------|----------|--------|---------|-----|----|--------|
| ETH | momentum_1h | 24 | +0.88% | 67% | 1.84 | +0.45 |
| SOL | rsi_reversion | 9 | +0.48% | 78% | 1.04 | +0.51 |
| **Ensemble** | **all** | **93** | **+0.21%** | **53%** | **1.32** | **+0.14** |

## Features

| Layer | Location | API key |
|-------|----------|---------|
| OHLCV / order book | `src/qooi/exchange/market.py` | No |
| Signal bot (OKX trading API) | `src/qooi/exchange/trading.py` | Yes (`.env`) |
| Parquet data cache | `src/qooi/exchange/store.py` | No |
| Technical indicators | `src/qooi/exchange/indicator.py` | No |
| Loop-based backtest (shared with live) | `src/qooi/exchange/backtest.py` | No |
| Strategy evaluation | `src/qooi/exchange/eval.py` | No |
| Signal state-machines | `src/qooi/strategies/` | No |

## Usage

```bash
# One-time setup: create OKX signal channels + strategies
uv run python scripts/setup_signal.py testnet

# Every 1H: compute signals, decide, push to OKX
uv run python scripts/trade.py testnet

# Backtest a strategy on cached data
uv run python scripts/backtest.py

# Custom backtest:
uv run python -c "
from qooi.exchange.backtest import Backtest, RiskConfig
from qooi.core.signal import compute_dataframe
import polars as pl

df = pl.read_parquet('data/cache/ETH_USDT_1H.parquet')
# compute_dataframe runs indicators + strategy signal
bt = Backtest(data=df, signal_expr=pl.col('signal'), initial_capital=500,
               risk=RiskConfig(max_leverage=2.0, max_risk_pct=0.50, ct_val=0.1))
result = bt.run()
print(f'Sharpe={result.metrics.sharpe_ratio:.2f}  WR={result.metrics.win_rate_pct:.0f}%')
"

# Switch between testnet / live API keys:
export OKX_ENV=test && uv run python scripts/trade.py testnet
export OKX_ENV=live && uv run python scripts/trade.py live dry
```

## Environment

- **Python 3.12** via `uv`
- **Polars** DataFrame library
- **python-okx** SDK for trading
- GitHub Actions: 1H cron schedule on testnet

## Project structure

```text
src/qooi/
  core/              ← signal.py (compute_*_1h), decide.py (decision engine)
  exchange/          ← market.py, trading.py, backtest.py, eval.py, indicator.py
  strategies/        ← momentum_1h.py, rsi_reversion.py, momentum.py,
                       ema_pullback.py, ema_pullback_v2.py, flow_pipeline.py,
                       portfolio.py
scripts/
  setup_signal.py    ← one-time OKX signal channel + strategy creation
  trade.py           ← live trading entry point (GitHub Actions or manual)
  backtest.py        ← CLI backtest runner
docs/
  okx-api.md         ← OKX API reference
  testnet.md         ← testnet validation workflow
  context.md         ← domain context & glossary
  tests.md           ← instrument config & backtest results
```

## Live Deployment

1. Create OKX API keys with trade permissions on your testnet account
2. Set up testnet `.env.test`:

   ```ini
   OKX_API_KEY_TEST=your_key
   OKX_SECRET_KEY_TEST=your_secret
   OKX_PASSPHRASE_TEST=your_passphrase
   OKX_FLAG=1
   ```

3. One-time: `uv run python scripts/setup_signal.py testnet`
4. Every hour: `uv run python scripts/trade.py testnet`
5. After observing 1-2 weeks of paper-trading results, switch to live.

For production, copy `.env.test` → `.env.live` with live credentials and run:

```bash
uv run python scripts/setup_signal.py live
uv run python scripts/trade.py live dry    # dry run first
uv run python scripts/trade.py live live   # real orders
```
