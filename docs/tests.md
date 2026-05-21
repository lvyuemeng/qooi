# Qooi — 1H Dual-Strategy Ensemble

## Instruments

All trading is on OKX perpetual swaps via the OKX Signal Bot (server-driven TP/SL).

| Symbol | Strategy | ctVal | Leverage | Capital | TP | SL |
|--------|----------|-------|----------|---------|-----|-----|
| ETH-USDT-SWAP | runtime selected | 0.1 ETH/ct | 2× | $500 | 2.0% | 2.5% |
| SOL-USDT-SWAP | runtime selected | 1 SOL/ct | 3× | $200 | 2.0% | 2.0% |

## Signal Pipelines

### Momentum Burst (ETH)

```text
1H OHLCV → compute_signal_frame("momentum_burst")
         → 6-bar return > 0.3%, EMA50>EMA200, ADX>20, volume > 1.5× avg
         → session 08-22 UTC, trend maturity ≥20 bars
         → signal = 1 / -1 / 0
```

### RSI Reversion (SOL)

```text
1H OHLCV → compute_signal_frame("rsi_bounce_reversion")
         → RSI(14) < 30 → bounce > 25 with confirmation
         → EMA50>EMA200, ADX>20, session 08-22 UTC
         → signal = 1 / 0  (long only)
```

## Backtest Results (1H, current pipeline)

| Asset | Strategy | Trades | WR | PL | Avg Win | Avg Loss | Calendar Sharpe |
|-------|----------|--------|----|----|---------|----------|-----------------|
| ETH | momentum_burst | 28 | 64% | 2.88 | +1.56% | 0.98% | unstable |
| SOL | rsi_bounce_reversion | 11 | 64% | 2.08 | +0.88% | 0.74% | unstable |

Calendar Sharpe/Sortino on sparse 1H equity are not primary truth. Use trade
count, win rate, profit factor, avg win/loss, and expectancy first.

Backtest profiles:

```bash
uv run python scripts/research.py --config configs/research/base-backtest.toml
uv run python scripts/research.py --config configs/research/grid-backtest.toml
uv run python scripts/research.py --config configs/research/martingale-backtest.toml
uv run python scripts/research.py --config configs/research/hedge-backtest.toml
```

Use profile sweep to separate weak edge from weak exposure / conservative sizing.

## Backtest ↔ Live Parity

| Feature | Backtest | Live |
|---------|----------|------|
| Signal computation | `compute_signal_frame()` on full DataFrame | Same — re-runs on cached data each invocation |
| Indicators | `strategies.indicators.add_indicators()` | Same |
| Decision engine | `process_bar()` (pipeline) | `process_bar()` (pipeline) |
| Exit mode | `signal_flip_only` + tiered exits in pipeline | `signal_flip_only` via OKX signal bot |
| Stop-loss | Strategy state-machine (hard stop at 1.5–1.8× ATR) | OKX Signal Bot server-side TP/SL + direct TradeAPI close_position() |
| Take-profit | Strategy state-machine (target at 1.2–1.5× ATR) | OKX Signal Bot server-side TP/SL |
| Trailing stop | Strategy state-machine (2.0× ATR from high) | Not available in OKX signal bot |
| Time stop | Strategy state-machine (6–12 bars) | Not available |
| Circuit breaker | 2 losses → suspend until 20-bar high/low | Backtest only (GitHub Actions is stateless) |
| Position state | Signal column (1/-1/0) | `GET /signal/positions` (`pos` field) |
| Order type | Market (backtest) / Limit (live) | Limit entry via signal bot or direct TradeAPI place() |

## Data Sources

| Source | Depth | Use |
|--------|-------|-----|
| OKX SDK | 300 bars/request | Live signals, accumulating cache |
| OKX Parquet cache | Grows over time | Backtest |
| CCXT (LBank) | 2000+ bars | Historical spot (reference only) |

## Running

```bash
# Auto-creates bots on first run, then trades
uv run python scripts/trade.py test

# Backtest (all cached data)
uv run python scripts/research.py

# Live trading — production
uv run python scripts/trade.py live dry    # dry run
uv run python scripts/trade.py live live   # real orders
```

## Account Setup

Perpetual swaps require OKX account in "single-currency margin" or "multi-currency margin"
mode. Error `51010` means the account is in "simple" mode — switch in OKX settings.

For the signal bot, ensure:

1. `.env.test` has `OKX_API_KEY_TEST`, `OKX_SECRET_KEY_TEST`, `OKX_PASSPHRASE_TEST`
2. `OKX_FLAG=1` for test, `OKX_FLAG=0` for live
3. Run `uv run python scripts/trade.py test` — auto-creates signal bot on first run
