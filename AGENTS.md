# AGENTS.md — qooi agent configuration

## Agent skills

### Issue tracker

Issues are tracked as local markdown files under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Labels use the five canonical defaults: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context repo — one `CONTEXT.md` at root plus `docs/adr/` for architecture decisions. See `docs/agents/domain.md`.

---

## Project conventions

### Stack

| Tool | Convention |
|------|-----------|
| Python manager | `uv` (no global state) |
| Python version | 3.11–3.13 |
| DataFrame lib | **Polars** (pandas only via `to_pandas()` for SDK interop) |
| Data sources | Crypto: OKX (`qooi.exchange.MarketData`) / A-share: TickFlow (inactive) |
| API keys | `.env` file (never committed), use `scripts/okx_profile.py live\|test` to switch |

### Running code

```bash
uv sync                              # install dependencies
uv run python scripts/demo.py        # run the demo pipeline
uv run python scripts/trade_okx.py   # place a live/testnet order (requires .env)
```

### Project structure

```
qooi/
├── src/qooi/
│   └── exchange/
│       ├── market.py      ← OKX public market data (no auth)
│       ├── trading.py     ← OKX orders & balance (API key via .env)
│       ├── store.py       ← OHLCV cache (Parquet)
│       ├── indicator.py   ← SMA, RSI, ATR, Bollinger
│       ├── backtest.py    ← vectorized backtest with costs + walk-forward
│       ├── pipeline.py    ← Pipeline: load → indicators → signal → backtest → evaluate → plot
│       └── eval.py        ← strategy evaluation (Sharpe, win rate, IC, IR, drawdown…)
│   └── strategies/
│       └── __init__.py    ← signal expressions (sma_cross, ema_cross, bollinger)
├── scripts/
│   ├── demo.py            ← single entry: fetch → cache → backtest → plot → evaluate
│   └── trade_okx.py       ← place orders (profile-aware)
├── data/
│   ├── cache/             ← Parquet OHLCV cache (gitignored)
│   ├── charts/            ← generated PNG charts (gitignored)
│   └── signals/           ← strategy signal CSVs
├── AGENTS.md              ← this file
├── CONTEXT.md             ← domain glossary
└── README.md              ← project overview
```

### Coding style

- All functions must be type-annotated
- `ruff check` must pass before commit (runs on `src/qooi/` and `scripts/`)
- `ruff format` is the formatter (double quotes, 100 char width)
