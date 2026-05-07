# Testnet Validation

## Setup (One-Time)

Perpetual swaps require a margin-enabled account.

1. **OKX testnet/demo** → Settings → Account Mode → switch to **"Single-currency margin"**
2. Transfer USDT from spot sub-account to swap sub-account
3. Verify: `uv run python scripts/trade.py testnet` should not show error 51010

## Daily Routine

### Check state

```bash
uv run python scripts/trade.py testnet
```

This does everything: computes signals, syncs with exchange, manages risk, places/closes orders, prints a portfolio summary.

### What the output means

```
=== pre-flight ===
  USDT   avail=4932.04 frozen=0      ← your swap margin balance
  positions: 1 open                    ← open futures positions
  pending orders: 0                    ← unfilled limit orders
==================

  SOL-USDT-SWAP   sig=+0.403 th=0.350  ← signal above threshold → entry
```

Possible outcomes:

| Output | Meaning |
|--------|---------|
| `ORDER buy sz=1 px=88.2 id=...` | Limit order placed — waiting for fill |
| `skip (weak_signal)` | Signal below threshold — nothing to do |
| `skip (order_filled)` | Previous order filled — now ACTIVE, managing risk |
| `skip (holding)` | Position active — stops/targets/trails updated |
| `ORDER sell sz=1 px=89.5` | Exit triggered — closing position |
| `ERROR [51010]` | Account mode is "simple" — switch to margin mode |
| `ERROR [51020]` | Order below minimum — increase capital or leverage |

### Portfolio summary

```
## qooi Portfolio

Total: $5,001.30 (+0.03% since inception)
  USDT free:   $4,932.04
  USDT margin: $80.00

  Positions:
    SOL-USDT-SWAP   long  1ct @ $88.20 → $89.50  upl=+$1.30 (+1.6%)

**ETH-USDT-SWAP**: T=0 | no trades
**SOL-USDT-SWAP**: T=1 | 1 open @ $88.20
```

- **Total**: free USDT + margin + unrealized PnL
- **upl**: unrealized profit/loss on open positions
- **T**: number of orders placed this session

## Fast Validation (Local)

Run twice, 2-3 minutes apart, to observe the fill lifecycle:

```bash
# Run 1 — place orders
uv run python scripts/trade.py testnet

# Wait 2-3 min for OKX to process fills...

# Run 2 — check fill status, manage risk
uv run python scripts/trade.py testnet
```

If signals are strong enough (> threshold), you'll see:
1. Run 1: `ORDER buy` → limit order placed
2. Run 2: `skip (order_filled)` → position active → `skip (holding)` → risk managed

## GitHub Actions

The system also runs automatically every 4 hours via GitHub Actions (`.github/workflows/trade-testnet-4h.yml`). Logs are uploaded as artifacts and retained for 90 days.

Manual trigger: GitHub → Actions → `qooi-testnet-4h` → Run workflow.

On the Actions runner, orders persist because it's a Linux environment with stable network. The 4h cycle matches the signal's bar frequency.

## Cancel Stale Orders

If orders accumulate from rapid local testing:

```bash
uv run python scripts/_cleanup.py
```

This cancels all pending orders and prints the clean state.

## FAQ

**Q: Why no orders placed even with strong signal?**
A: Check if swap sub-account has USDT. The margin gate requires `free_usdt >= 1.2 × required_margin`.

**Q: Error 51010 "account mode"?**
A: Account is in "simple" mode. Switch to "single-currency margin" in OKX settings.

**Q: Why does the Summary say "no trades yet" even though I placed orders?**
A: The reporting needs at least one completed buy→sell pair to compute PnL. Single buys show as "N open positions, no closed trades."
