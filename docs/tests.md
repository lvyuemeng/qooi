# Qooi — 1H Dual-Strategy Ensemble

## Instruments

All trading is on OKX perpetual swaps via the OKX Signal Bot (server-driven TP/SL).

| Symbol | Strategy | ctVal | Leverage | Capital | TP | SL |
|--------|----------|-------|----------|---------|-----|-----|
| ETH-USDT-SWAP | momentum_1h | 0.1 ETH/ct | 2× | $500 | 2.0% | 2.5% |
| SOL-USDT-SWAP | rsi_reversion | 1 SOL/ct | 3× | $200 | 2.0% | 2.0% |

## Signal Pipelines

### Momentum Burst (ETH)

```text
1H OHLCV → add_indicators → momentum_1h_signal
         → 6-bar return > 0.3%, EMA50>EMA200, ADX>20, volume > 1.5× avg
         → session 08-22 UTC, trend maturity ≥20 bars
         → signal = 1 / -1 / 0
```

### RSI Reversion (SOL)

```text
1H OHLCV → add_indicators → rsi_reversion_signal
         → RSI(14) < 30 → bounce > 25 with confirmation
         → EMA50>EMA200, ADX>20, session 08-22 UTC
         → signal = 1 / 0  (long only)
```

## Backtest Results (1H, 83 days, 4 assets)

| Asset | Strategy | Trades | Avg Ret | WR | PL | Sharpe |
|-------|----------|--------|---------|-----|----|--------|
| ETH | momentum_1h | 24 | +0.88% | 67% | 1.84 | +0.45 |
| SOL | rsi_reversion | 9 | +0.48% | 78% | 1.04 | +0.51 |
| BTC | momentum_1h | 26 | -0.08% | 42% | 1.06 | -0.09 |
| XAU | momentum_1h | 2 | -0.21% | 0% | — | — |
| **Ensemble** | **all** | **93** | **+0.21%** | **53%** | **1.32** | **+0.14** |

BTC excluded from live trading due to persistent negative expectancy across all
tested strategies and timeframes. XAU-USDT had insufficient data (34 days).

## Backtest ↔ Live Parity

| Feature | Backtest | Live |
|---------|----------|------|
| Signal computation | `momentum_1h_signal()` / `rsi_reversion_signal()` on full DataFrame | Same — re-runs on cached data each invocation |
| Indicators | `add_indicators()` | Same |
| Decision engine | `decide_idle()` / `decide_active()` | Same |
| Exit mode | `signal_flip_only` (tiered exits in strategy state-machine) | `signal_flip_only` |
| Stop-loss | Strategy state-machine (hard stop at 1.5–1.8× ATR) | OKX Signal Bot server-side TP/SL |
| Take-profit | Strategy state-machine (target at 1.2–1.5× ATR) | OKX Signal Bot server-side TP/SL |
| Trailing stop | Strategy state-machine (2.0× ATR from high) | Not available in OKX signal bot |
| Time stop | Strategy state-machine (6–12 bars) | Not available |
| Circuit breaker | 2 losses → suspend until 20-bar high/low | Backtest only (GitHub Actions is stateless) |
| Position state | Signal column (1/-1/0) | `GET /signal/positions` (`pos` field) |
| Order type | Market (backtest) / Limit (live) | Limit entry via signal bot |

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
uv run python scripts/backtest.py

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
