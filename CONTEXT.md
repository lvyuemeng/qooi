# qooi — domain context

## Domain

Quantitative trading strategy research on China A-share stock market.

## Stack

- **Quant engine**: [BigQuant SDK](https://bigquant.com/wiki/doc/vac4qwmQr4) — data query (`dai`), local backtest (`bigtrader`), distributed compute (`fai`)
- **Python management**: `uv` — no global Python state
- **Python version**: 3.11–3.13

## Data sources

| Priority | Source | Scope | Cost |
|----------|--------|-------|------|
| 1 | **TickFlow** (free tier) | A-share daily K-line, ETFs, futures, HK, US | Free, no auth |
| 2 | **OKX Market API** | Crypto spot/perp/futures/option OHLCV, order book, ticker | Free, no auth |
| 3 | **BigQuant DAI** | Full A-share data (real-time, minute, fundamentals) | Paid plan needed |
| 4 | **MOOTDX** (fallback) | TDX local/online data | Free, requires TDX |

## Preferences

- **DataFrame library**: Polars preferred over pandas. Use `result.pl()` instead of `result.df()` when calling BigQuant DAI queries.

## Glossary

| Term | Definition |
|------|-----------|
| A-share | China A-share stocks traded on Shanghai/Shenzhen exchanges |
| DAI | BigQuant's data query interface — SQL over Arrow Flight to cloud data |
| BigTrader | BigQuant's local backtesting engine |
| FAI | BigQuant's distributed computing framework |
| bar1d | Daily OHLCV bar data |
| instrument | Stock ticker code, e.g. `000001.SZ` |
| benchmark | Index used as performance baseline, e.g. `000300.SH` (CSI 300) |

## Workflow

1. **Research** — Paste `scripts/run_notebook.py` into AI Studio web, run to get factor reports + CSV signals
2. **Download** CSVs from AI Studio to `data/signals/<strategy>.csv`
3. **Backtest locally** — `uv run python scripts/backtest_csv.py --csv data/signals/<strategy>.csv`
4. BigQuant AI Studio (cloud) only when DAI data access is needed

## Research pipeline

- `scripts/run_notebook.py` — complete factor research + 7 strategies, runs in AI Studio web
- `src/qooi/research/` — local factor analysis module (Polars, for when DAI is available)
- `scripts/backtest_csv.py` — local backtest using signal CSV + TickFlow bar data
