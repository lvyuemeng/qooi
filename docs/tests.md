# Cross-asset strategy evaluation

## Test config

| Param | Value |
|---|---|
| Capital | 10,000 USDT |
| Cost | 0.2%/side (old model; real OKX VIP0: maker 0.005%, taker 0.05%) |
| Leverage | 1× fixed |
| Walk-forward | train=6, test=2, holdout=1, step=1, rebalance=30 bars |

## 1. Daily strategies (OKX 1000-1100 bars, 2023-2026)

### Results

| Asset | Strategy | Ret | Sharpe | DD | Trades | WR | PF | WFS |
|---|---|---|---|---|---|---|---|---|
| BTC | SMA(10,30) | +4384% | 0.12 | 59% | 44 | 66% | 2.84 | — |
| BTC | BB(20,2) | +127% | 0.49 | 33% | 74 | 46% | 0.78 | — |
| BTC | VM | +1593% | 0.08 | 69% | 76 | 61% | 2.15 | -0.73 |
| **BTC** | **TP** | **+1132%** | **0.80** | **55%** | **56** | **52%** | **1.58** | **0.28** |
| ETH | SMA(10,30) | +11% | 0.03 | 70% | 42 | 60% | 1.67 | — |
| ETH | BB(20,2) | -48% | -0.99 | 63% | 51 | 69% | 2.85 | — |
| ETH | VM | -21% | -0.17 | 61% | 72 | 64% | 1.18 | -0.16 |
| **ETH** | **TP** | **+45%** | **0.35** | **32%** | **13** | **69%** | **1.86** | **0.47** |
| SOL | SMA(10,30) | +389% | 0.96 | 55% | 33 | 67% | 2.31 | — |
| SOL | BB(20,2) | -57% | -0.79 | 66% | 57 | 53% | 2.33 | — |
| SOL | VM | +51% | 0.20 | 81% | 64 | 59% | 1.77 | -0.36 |
| **SOL** | **TP** | **+114%** | **0.64** | **61%** | **19** | **79%** | **3.64** | **0.30** |

**WFS** = walk-forward OOS Sharpe. TP is the only strategy with positive OOS Sharpe on all assets.

### Verified with deep data (LBank 2974 bars, 2018-2026)

| Source | Period | Bars | Trades | WR | Ret | Sharpe |
|---|---|---|---|---|---|---|
| Gate | 2017-2020 | 999 | 15 | 47% | +57% | 0.27 |
| Bitstamp | 2017-2019 | 999 | 21 | 48% | +552% | 1.40 |
| **LBank** | **2018-2026** | **2974** | **56** | **52%** | **+1132%** | **0.80** |

## 2. Intraday strategies (LBank deep data)

### OHLCV-only strategies (1H, 20000 bars)

| Strategy | Trades | WR | Ret | DD | Sharpe | Direction accuracy |
|---|---|---|---|---|---|---|
| SMA(10,30) | 62 | 53% | -16% | 35% | -0.54 | 50.2% |
| BB(20,2) | 120 | 53% | +6% | 9% | -0.19 | 49.8% |
| VM | 199 | 51% | -8% | 9% | -0.95 | 50.1% |
| TP | 28 | 32% | -10% | 15% | -0.80 | 48.5% |
| **Mean reversion (BB+ADX)** | **286** | **54%** | **-16%** | **35%** | **-0.58** | **37.4%** |

Direction accuracy of BB mean-reversion on 1H: **37.4%** (worse than random). Intraday OHLCV-only strategies fail because the market continues through BB extremes rather than reverting. Correct intraday strategies require **order book or funding rate data**.

### Order Book Imbalance strategy (synthetic, 1H 20000 bars)

| Scenario | Imbalance | Trades | WR | Ret | DD |
|---|---|---|---|---|---|
| Slight bid (55/45) | +0.10 | 0 | — | 0% | 0% |
| Strong bid (70/30) | +0.40 | 840 | 39% | +315% | 22% |
| Alternating (half bull/half bear) | ±0.20 | 396 | 53% | +7% | 34% |

**Entry gates**: ATR regime, volume confirmation, imbalance threshold (scaled by volatility), momentum agreement (3-bar). **Execution**: limit order at best bid/ask (maker fee 0.005%). **Risk**: 1.5× ATR stop, 2:1 R:R TP, trailing stop after +1× ATR, 5-bar cooldown, position sizing scales with imbalance.

### Real OKX costs (VIP0)

| Type | Per side | Round trip |
|---|---|---|
| Maker (limit / post-only) | 0.005% | 0.01% |
| Taker (market) | 0.05% | 0.10% |
| Old backtest model | 0.20% | 0.40% |

## 3. Intraday strategy framework (`src/qooi/strategies/intraday.py`)

| Strategy | Signal | Data needed | Frequency |
|---|---|---|---|
| **Funding rate reversal** | Extreme funding rate (>0.2%/8h) | OKX WS/REST | 1-3/week |
| **Order book imbalance** | (bid − ask) / (bid + ask) | CCXT `fetch_order_book` | 5-50/day |
| **Volume divergence** | Volume Z-score, peaked before reversal | OHLCV + vol | 0-3/day |

## 4. Data sources

| Source | Access | Notes |
|---|---|---|
| **OKX** (SDK) | Working | 1100 1D bars (2023-2026), funding rate, order book |
| **LBank** (CCXT) | Working | 2974 1D bars (2018-2026), 10000+ 1H/4H |
| **Gate / Bitstamp** (CCXT) | Working | 999 1D bars (2017-2020) |
| **Kraken** (CCXT) | Working | 721 1D bars (2024-2026) |
| Binance, Bybit, KuCoin | Blocked | Geographic restriction; use `UnifiedMarket(exchange_id, proxy=...)` |
