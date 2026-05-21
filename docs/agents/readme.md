# AGENTS.md — qooi agent configuration

## Agent skills

### Issue tracker

Issues are tracked as local markdown files under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Labels use the five canonical defaults: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context repo — one `docs/context.md` for domain glossary, plus `docs/adr/` for architecture decisions. See `docs/agents/domain.md`.

---

## Project conventions

### Stack

| Tool | Convention |
|------|-----------|
| Python manager | `uv` (no global state) |
| Python version | 3.11–3.13 |
| DataFrame lib | **Polars** (pandas only via `to_pandas()` for SDK interop) |
| Data sources | Crypto: OKX (`qooi.exchange.MarketData`) |
| API keys | `.env` file (never committed) |

### Running code

```bash
uv sync                              # install dependencies
uv run python scripts/demo.py        # default: BTC-USDT 1D 365d
uv run python scripts/demo.py ETH-USDT 4H 120        # ETH-USDT 4H 120d
uv run python scripts/demo.py XAU-USDT-SWAP 1D 120   # XAU gold perp 1D
uv run python scripts/strategy_test.py               # VuManChu on BTC 1D
uv run python scripts/strategy_test.py ETH-USDT 4H   # VuManChu on ETH 4H
uv run python scripts/risk_test.py                   # risk configs on BTC 1D
```

### Supported assets

Any OKX instrument ID works. Common crypto: `BTC-USDT`, `ETH-USDT`, `SOL-USDT`, `XRP-USDT`, `DOGE-USDT`. Commodity perp: `XAU-USDT-SWAP` (gold, listed 2025-12-31, ~125 bars of 1D data). Use 4H/1H for more data points.

See `docs/tests.md` for cross-asset strategy results, `docs/okx-api.md` for API endpoint reference, `docs/context.md` for domain glossary.

### Project structure

```text
qooi/
├── src/qooi/
│   └── exchange/
│       ├── market.py      ← OKX public market data (no auth)
│       ├── trading.py     ← OKX orders & balance (API key via .env)
│       ├── store.py       ← OHLCV cache (Parquet)
│       ├── indicator.py   ← SMA, RSI, ATR, Bollinger
│       ├── research.py    ← config-first research + backtest orchestration
│       ├── chart.py       ← standalone charting (equity curve, signals)
│       └── eval.py        ← strategy evaluation (Sharpe, win rate, IC, IR, drawdown…)
│   └── strategies/
│       └── __init__.py    ← signal expressions (sma_cross, ema_cross, bollinger)
├── scripts/
│   ├── demo.py            ← single entry: fetch → cache → backtest → plot → evaluate
│   ├── risk_test.py       ← risk analysis tests
│   └── strategy_test.py   ← strategy tests
├── data/
│   ├── cache/             ← Parquet OHLCV cache (gitignored)
│   ├── charts/            ← generated PNG charts (gitignored)
│   └── signals/           ← strategy signal CSVs
├── docs/
│   ├── agents/            ← agent configuration files
│   ├── context.md         ← domain glossary
│   └── adr/               ← architecture decision records
├── README.md              ← project overview
└── pyproject.toml         ← project metadata & dependencies (read this for canonical info)
```

### Coding style

- All functions must be type-annotated
- `ruff check` must pass before commit (runs on `src/qooi/` and `scripts/`)
- `ruff format` is the formatter (double quotes, 100 char width)
