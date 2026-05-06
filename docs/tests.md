# Qooi — strategy summary

## Test config

| Param | Value |
|---|---|
| Capital | 10,000 USDT |
| Cost | 0.005%/side (OKX VIP0 maker fee) |
| Leverage | 0.4× max (signal-clipped, drawdown-adaptive) |
| Risk | ATR stop 2.0×, ATR target 3.0×, trail distance 1.0× |
| Signal | OFI flow with |sig| > 0.4 magnitude filter |
| Data | OKX (parquet cache) for backtest; OKX SDK for live |

---

## 1. OFI Flow Signal (4H) — unified backtest + live

### Pipeline

```
OHLCV → add_indicators → add_regime_features → add_ofi_flow_columns → magnitude filter (|OFI| ≥ 0.4)
```

Risk management (`State`, `PositionState`, `RiskConfig`) is fully separate from the signal pipeline. Either can be changed independently.

### Results (4H, market orders, scale-invariant OFI flow)

| Asset | Threshold | Sharpe | DD | Return | Trades | WR |
|-------|-----------|--------|-----|--------|--------|-----|
| **SOL-USDT** | 0.35 | **+1.88** | 7.2% | +30.7% | 248 | 51% |
| **BTC-USDT** | 0.25 | **+1.52** | 6.8% | +19.7% | 447 | 57% |
| XRP-USDT | 0.45 | +0.18 | 9.4% | +1.7% | 71 | 54% |

\* ETH-USDT not included — spot cache was overwritten (SWAP data not equivalent).

**Exit reason distribution (SOL, lev=0.5, th=0.35):**
- Stop-loss: 70
- Target: 27
- Trailing stop: 69
- Signal-based exit: 553

**Sides:** long=169, short=550 (short-biased in the test period).

### Symbol naming convention

| Context | Format | Example |
|---|---|---|
| OKX SDK (live) | `ETH-USDT` | dash-separated, spot |
| OKX swap/futures | `ETH-USDT-SWAP` | perpetual swap |
| CCXT (backtest) | `ETH/USDT` | slash-separated |

`MarketData` auto-converts. `TradingClient` uses OKX format (`ETH-USDT`).

### Signal quality

| Metric | Ensemble (old) | OFI flow (current) |
|---|---|---|
| IC (4-bar forward) | +0.021 | +0.051 |
| Hit rate (4-bar) | 49% | 51% |
| Sharpe (market orders) | -0.04 | +1.37 |
| Interpretability | Opaque ensemble | "Order flow imbalance" |

## 2. Testnet readiness

| Component | Status | Note |
|---|---|---|
| Signal pipeline | ✅ | OFI flow, magnitude filter at 0.4 |
| Risk management | ✅ | Stop-loss / target / trailing via `RiskConfig` + ADR D1/D2 |
| Cost model | ✅ | OKX maker 0.005%/side via `CostModel` |
| Leverage control | ✅ | 0.4× max, signal-clipped, drawdown-adaptive halving |
| Order execution | ✅ | `post_only` (ETH) / `limit` (SOL) per-pair, 8h timeout |
| Duplicate prevention | ✅ | `sync()` cancels old orders, log-fallback if API down |
| Dry run mode | ✅ | `dry_run=True` simulates without API keys |
| Live mode | ✅ | `testnet` → OKX demo, `live` → production (separate secrets) |
| Portfolio runner | ✅ | `PortfolioRunner` + `PortfolioConfig` for multi-asset |
| Pre-flight status | ✅ | Balance, positions, pending orders printed before execution |

## 3. Backtest ↔ Live parity

| Feature | Backtest (`backtest.py`) | Live (`trading.py`) |
|---|---|---|
| State machine | `State` enum (IDLE → PENDING → ACTIVE → EXITING) | Same `State` enum |
| Position state | `PositionState` with stops/targets/trails | Same `PositionState` |
| Risk config | `RiskConfig(max_leverage, atr_stop, atr_target, trail_act, trail_dist)` | Same `RiskConfig` |
| Cost | `CostModel(commission_pct=0.00005)` | Same `CostModel` |
| Stop-loss | `check_exit()` every bar | `_decide_from_active()` → `check_exit()` |
| Take-profit | ATR target on entry | Same logic |
| Trailing stop | Activation at 2× ATR profit, trail at 1× ATR | Same logic (D2) |
| Signal-based exit | Exit on signal flip | Same logic (D1) |
| Breakeven stop | Via stop exit (not tracked separately) | Moves stop to entry at +1 ATR |
| Adaptive threshold | `_thresh()` based on PnL EMA | Same `_thresh()` on real PnL |
| Order type | `ord_type="market"` or `"limit"` | `post_only` or `limit` per-pair |
| Limit fill | Fills if `bar.low <= order_px` | Exchange fills, checked via `pending()` |
| Timeout | 2 bars (8h for 4h) | 2× bar duration (8h for 4h, D4) |

## 4. Known gaps (deferred)

| Gap | Priority | Adr ref |
|-----|----------|---------|
| Frozen-capital gate (>50% blocked → skip) | High | C1 |
| SOL data depth (400 bars → need 2000+) | High | — |
| OOS walk-forward validation for OFI flow | High | — |
| Trailing target (move target up in trend) | Medium | ADR-002 |
| Time-based exit in ACTIVE (N bars max) | Medium | ADR-002 |
| Market order fallback (taker after N fails) | Low | ADR-002 |

## 5. Data sources

| Source | OHLCV depth | Order book | Funding rate | Geo-access |
|---|---|---|---|---|
| **OKX SDK** | 300-1200 bars (2023+) | Live (REST) | 3-month history | Open |
| **LBank (CCXT)** | 3000-10000 bars (2018+) | Live (REST) | — | Open |
| **Bitfinex** | 10000 bars (2019-2023) | Live (REST) | Yes | Open |
| Gate, Bitstamp, KuCoin, MEXC | 500-999 bars | Live (REST) | Yes | Open |
| Binance, Bybit | Blocked | — | — | Proxy-needed |

## 6. Running

```bash
# Backtest report
uv run python scripts/backtest_report.py

# Live trading (testnet)
uv run python scripts/trade.py testnet

# Live trading (production — manual confirmation required)
uv run python scripts/trade.py live dry
uv run python scripts/trade.py live live  # places real orders
```
