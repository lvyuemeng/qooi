# Test Validation

## Setup (One-Time)

Perpetual swaps require a margin-enabled account.

1. **OKX test/demo** → Settings → Account Mode → switch to **"Single-currency margin"**
2. Transfer USDT from spot sub-account to swap sub-account
3. Set up `.env.test`:

   ```ini
   OKX_API_KEY_TEST=your_key
   OKX_SECRET_KEY_TEST=your_secret
   OKX_PASSPHRASE_TEST=your_passphrase
   OKX_FLAG=1
   ```

4. Run trade.py — auto-creates signal bot on first run, then trades:

   ```bash
   uv run python scripts/trade.py test
   ```

## Daily Routine

```bash
uv run python scripts/trade.py test
```

This does everything: fetches 1H candles, runs the strategy state-machine,
queries OKX position state, decides, and pushes orders to the signal bot.

### What the output means

```text
  ETH-USDT-SWAP        strategy=momentum_burst sig=+1 atr=42.5 pos=flat action=enter
    ORDER buy sz=2 px=2320.50 sl=2250.0 tp=2367.0

  SOL-USDT-SWAP        strategy=rsi_bounce_reversion sig=0 atr=3.8 pos=flat action=hold (weak_signal)
```

Action labels:

| Output | Meaning |
|--------|---------|
| `action=enter` | Signal fired — pushing order to OKX signal bot. Server handles TP/SL. |
| `action=hold (weak_signal)` | No signal this bar — nothing to do |
| `action=hold (holding)` | Position active — signal hasn't flipped |
| `action=exit (signal_flipped)` | Signal reversed direction — closing position via signal bot |
| `ORDER FAILED` | API error — check account mode, balance, or signal bot state |

Position state is queried from `GET /signal/positions` (server-side truth).
`pos=flat` means no position. `pos=buy` or `pos=sell` means holding.

### What the strategies are doing

| Strategy | Asset | What to expect |
|----------|-------|---------------|
| `momentum_burst` | any configured asset | Enters on strong 1H directional bursts (6-bar return > 0.3%) with volume confirmation. Signal stays at ±1 until trend flips. Frequent signals during trending markets, quiet during chop. |
| `rsi_bounce_reversion` | any configured asset | Enters only on oversold bounces in uptrends (RSI < 30 then recovers). Long-only. Less frequent than momentum — expect 2-4 signals per week. |

## Fast Validation (Local)

Run multiple times to observe different signal states:

```bash
# See current signal state
uv run python scripts/trade.py test

# Wait 1 hour for next bar...

# See updated state (new candle, new signal)
uv run python scripts/trade.py test
```

If a strategy has a signal, you'll see `action=enter` and an `ORDER ...` line.

## GitHub Actions

The system runs automatically **every 1 hour** via GitHub Actions
(`.github/workflows/trade-test-1h.yml`). Logs are uploaded as artifacts
and retained for 90 days.

Manual trigger: GitHub → Actions → `qooi-test-1h` → Run workflow.

On the Actions runner, orders execute on the OKX test environment (server-side),
so position state persists across GitHub Actions invocations. The strategy
re-runs the full state-machine on each invocation (matching backtest).

## Cancel Stale Orders

If orders accumulate from rapid local testing:

```bash
uv run python -c "
from qooi.exchange.trading import TradingClient, load_okx_env
import os
os.environ['OKX_ENV'] = 'test'
load_okx_env()
tc = TradingClient()
# Use signal_stop to cancel an algo
"
```

## FAQ

**Q: Why no orders placed even with signal?**
A: Check if swap sub-account has USDT. The signal bot needs available balance.

**Q: Error 51010 "account mode"?**
A: Account is in "simple" mode. Switch to "single-currency margin" in OKX settings.

**Q: Why is pos=flat after a previous ORDER?**
A: The order may still be pending (limit order not filled yet). Check the OKX
web interface to see if the order is filled or pending.

**Q: Why does the signal change from 1 to 0 and back to 1 without a trade?**
A: If the signal was 1 (long) and then the position was closed by OKX server-side
TP/SL, the signal needs to flip to 0 before re-entering. The state-machine
tracks EMA trend state — re-entry requires the full entry condition set again.
