# Qooi — Perpetual Futures Strategy

## Instruments

All trading is on OKX perpetual swaps (futures). Positions persist on exchange across runs.

| Symbol | ctVal | Min | Threshold | Leverage | Capital |
|--------|-------|-----|-----------|----------|---------|
| ETH-USDT-SWAP | 0.1 ETH/ct | 1 ct | 0.25 | 2× | $500 |
| SOL-USDT-SWAP | 1 SOL/ct | 1 ct | 0.35 | 3× | $200 |
| BTC-USDT-SWAP | 0.01 BTC/ct | 1 ct | 0.25 | 2× | $1,000 |

## Signal Pipeline

```
OHLCV → add_indicators → add_regime_features → add_ofi_flow_columns
      → magnitude filter (|OFI| ≥ sig_threshold) → signal
```

OFI flow normalization: `net_flow / vol_total` (fraction of directional volume). Scale-invariant across assets.

Thresholds are per-asset, set via `sig_threshold` in pair config. Adaptive entry gate uses PnL-based EMA.

## Design: Limit as Signal Verification

- **Entry**: limit orders (post_only or limit). An unfilled order = the market disagreed with the signal = prevented loss.
- **Exit**: market orders. When risk triggers, guaranteed execution.
- **Fill rate**: directly measures signal accuracy.

## Backtest Results (Swap Data)

Swap data is limited to 300 bars (OKX API restriction). Cache accumulates over time via `MarketData.candles(cache=True)`.

| Asset | Bars | Threshold | Sharpe | DD | Trades | Note |
|-------|------|-----------|--------|-----|--------|------|
| SOL-USDT-SWAP | 300 | 0.35 | -2.21 | 2.7% | 34 | Insufficient data |
| BTC-USDT-SWAP | 300 | 0.25 | -1.24 | 3.4% | 59 | Insufficient data |
| ETH-USDT-SWAP | 300 | 0.25 | -4.62 | 9.3% | 55 | Insufficient data |

Spot data (historical, for reference only — spot is no longer used):

| Asset | Bars | Sharpe | DD | Trades | WR |
|-------|------|--------|-----|--------|-----|
| SOL-USDT | 1,999 | +1.88 | 7.2% | 248 | 51% |
| BTC-USDT | 2,000 | +1.52 | 6.8% | 447 | 57% |

## Backtest ↔ Live Parity

| Feature | Backtest (`backtest.py`) | Live (`trading.py`) |
|---|---|---|
| State machine | `State` enum (IDLE→PENDING→ACTIVE→EXITING) | Same |
| Risk config | `RiskConfig(atr_stop=2.0, atr_target=3.0, trail=1.0)` | Same |
| Stop-loss | `check_exit()` every bar | `_decide_from_active()` → `check_exit()` |
| Trailing stop | Activation at 2×ATR profit, trail at 1×ATR | Same (D2) |
| Signal-based exit | Exit on signal flip | Same (D1) |
| Breakeven stop | Via stop exit | Moves stop to entry at +1 ATR |
| Adaptive threshold | PnL EMA-based | Same, scaled per-asset |
| Order type | Market (backtest) / Limit (live entry) / Market (live exit) | Per-path |
| Limit fill simulation | `bar.low <= order_px` | Exchange API |
| Timeout | 2 bars (8h for 4h) | 8h (D4) |

## Data Sources

| Source | Depth | Use |
|--------|-------|-----|
| OKX SDK | 300 bars/request | Live signals, accumulating cache |
| OKX parquet cache | Grows over time | Backtest |
| LBank | 2,000+ bars | Historical spot (reference only) |

## Running

```bash
# Backtest (swap data)
uv run python scripts/backtest.py

# Live trading — testnet
uv run python scripts/trade.py testnet

# Live trading — production (requires "LIVE" confirmation)
uv run python scripts/trade.py live dry    # dry run
uv run python scripts/trade.py live live   # real orders
```

## Account Setup

Perpetual swaps require OKX account in "single-currency margin" or "multi-currency margin" mode. Error `51010` means the account is in "simple" mode — switch in OKX settings, transfer USDT to swap sub-account.
