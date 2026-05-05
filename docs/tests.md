# Qooi — proficient strategy summary

## Test config

| Param | Value |
|---|---|
| Capital | 10,000 USDT |
| Cost | 0.005%/side (OKX VIP0 maker, limit orders) |
| Leverage | 2× max (signal-clipped, drawdown-adaptive) |
| Risk | ATR stop 2.0×, ATR target 3.0× |
| Data | LBank (CCXT) for backtest; OKX SDK for live |

---

## 1. Multi-Factor Intraday Ensemble (4H) — unified backtest + live

### Pipeline

```
OHLCV → indicators → ensemble → OFI flow → micro confirm → adaptive gate → stop/target/trailing → portfolio
```

All layers shared between backtest (`Backtest`) and live executor (`LiveExecutor`).

### Results (4H, 2023-2026, leverage=2.0)

| Asset | Sharpe | DD | Return | Trades | WR | OOS Sharpe | Overfit |
|---|---|---|---|---|---|---|---|
| **ETH-USDT** | **0.71** | 12.2% | +18.2% | 79 | 54% | **0.58** | 0.40 |
| SOL-USDT | -0.12 | 14.0% | +1.9% | 68 | 56% | -0.15 | 0.46 |

OOS Sharpe positive on ETH (0.58). Overfit ratio 0.40 = wins 60% of unseen test windows.

### Symbol naming convention

| Context | Format | Example |
|---|---|---|
| OKX SDK (live) | `ETH-USDT` | dash-separated, spot |
| OKX swap/futures | `ETH-USDT-SWAP` | perpetual swap |
| CCXT (backtest) | `ETH/USDT` | slash-separated |

`MarketData` auto-converts. `TradingClient` uses OKX format (`ETH-USDT`).

### Testnet readiness

| Component | Status | Note |
|---|---|---|
| Signal pipeline | ✅ | Multi-factor ensemble + OFI + adaptive gate |
| Risk management | ✅ | Stop-loss / take-profit / trailing via `RiskConfig` |
| Cost model | ✅ | OKX maker 0.005%/side via `CostModel` |
| Leverage control | ✅ | Signal clipping + position cap + drawdown-adaptive halving |
| Order execution | ✅ | `post_only` limit orders, 120s timeout auto-cancel |
| Position check | ✅ | Prevents duplicate orders |
| Dry run mode | ✅ | `dry_run=True` simulates without API keys |
| Live mode | ✅ | `live=False`→testnet, `live=True`→production (separate secrets) |
| Walk-forward OOS | ✅ | ETH 4H: Sharpe 0.58 across rolling train/test windows |
| Portfolio runner | ✅ | `PortfolioRunner` + `PortfolioConfig` for multi-asset deployment |

---

## 2. Daily strategies (1D, 2018-2026, LBank)

| Asset | Strategy | Sharpe | DD | Return | Trades | WR |
|---|---|---|---|---|---|---|
| BTC | Trend Pullback | **0.80** | 55% | +1132% | 56 | 52% |
| ETH | Trend Pullback | **0.35** | 32% | +45% | 13 | 69% |
| SOL | Trend Pullback | **0.64** | 61% | +114% | 19 | 79% |

---

## 3. Backtest ↔ Live parity

| Feature | Backtest (`backtest.py`) | Live (`trading.py`) |
|---|---|---|
| Signal pipeline | `signal_expr` + `run()` | `SignalSource` + `step()` |
| Risk config | `RiskConfig(max_leverage, atr_stop_mult, atr_target_mult)` | Same `RiskConfig` |
| Cost | `CostModel(commission_pct, slippage_pct)` | Same `CostModel` |
| Stop-loss | `_check_exit()` every bar | `step()` every bar via `RiskConfig` |
| Take-profit | ATR target on entry | Same logic |
| Trailing stop | Trailing high/low + distance × ATR | Same logic |
| Leverage | `_clipped_signal()` + `_enter_position()` cap | `clipped` + `signal_abs` + dynamic drawdown halving |
| Order type | Implicit market-at-close | `post_only` limit (maker 0.005%) |
| Timeout | None (instant fill) | 120s auto-cancel |
| Equity tracking | `equity` array + `compute_metrics` | `self._equity` array + `report()` via `compute_metrics` |
| Walk-forward | `WalkForwardBacktest` | Not applicable (live) |

---

## 4. Data sources

| Source | OHLCV depth | Order book | Funding rate | Geo-access |
|---|---|---|---|---|
| **OKX SDK** | 300-1200 bars (2023+) | Live (REST) | 3-month history | Open |
| **LBank (CCXT)** | 3000-10000 bars (2018+) | Live (REST) | — | Open |
| **Bitfinex** | 10000 bars (2019-2023) | Live (REST) | Yes | Open |
| Gate, Bitstamp, KuCoin, MEXC | 500-999 bars | Live (REST) | Yes | Open |
| Binance, Bybit | Blocked | — | — | Proxy-needed |
