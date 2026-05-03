# qooi — domain context

## Domain

Quantitative trading strategy research on China A-share stock market.

## Stack

- **Quant engine**: [BigQuant SDK](https://bigquant.com/wiki/doc/vac4qwmQr4) — data query (`dai`), local backtest (`bigtrader`), distributed compute (`fai`)
- **Python management**: `uv` — no global Python state
- **Python version**: 3.11–3.13

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

1. Explore data via `dai.query()` locally
2. Develop strategy in `src/`, run backtest via `bigtrader.run()` in `scripts/`
3. BigQuant AI Studio (cloud) only when distributed compute is essential
