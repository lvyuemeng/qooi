# qooi

Crypto quantitative trading toolkit — market data, backtesting, evaluation.

```bash
uv sync
uv run python scripts/demo.py
```

## Quick start

```bash
# Fetch BTC daily data, compute indicators, run backtest, print metrics, save chart
uv run python scripts/demo.py
```

Output: Sharpe, win rate, drawdown, IC/IR, equity curve chart → `data/charts/`

## Project

| Layer | Package | Auth needed |
|-------|---------|-------------|
| Market data | `src/qooi/exchange/market.py` | No |
| Trade | `src/qooi/exchange/trading.py` | API key via `.env` |
| Cache | `src/qooi/exchange/store.py` | No |
| Indicators | `src/qooi/exchange/indicator.py` | No |
| Backtest | `src/qooi/exchange/backtest.py` | No |
| Pipeline | `src/qooi/exchange/pipeline.py` | No |
| Evaluation | `src/qooi/exchange/eval.py` | No |

See `AGENTS.md` for full project conventions.
