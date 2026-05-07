# Local Testnet Validation

## Why Local

GitHub Actions 4h cycle is too slow for testnet — OKX testnet orders expire in
~15 minutes.  By the time the next cycle runs, the order is gone.  We never see
fill → manage → exit.

On your local machine, you control the timing.

## Pre-flight

```bash
# 1. Cancel any stale orders from previous runs
uv run python scripts/_cleanup.py

# 2. Verify the swap instruments work
uv run python -c "from qooi.exchange.market import MarketData; md=MarketData('okx'); print(md.candles('SOL-USDT-SWAP','4h',limit=5))"

# 3. Check balance
uv run python -c "from qooi.exchange.trading import TradingClient; tc=TradingClient(); [print(b) for b in tc.balance()]"
```

## Validation — Fast Cycle

Run `trade.py` multiple times at ~2-5 minute intervals to observe:

```
=== Run 1 ===
uv run python scripts/trade.py testnet
# → orders placed, state = PENDING

=== Run 2 (2 min later) ===  
uv run python scripts/trade.py testnet
# → sync() finds pending order → checks fill → if filled → ACTIVE
# → risk rules activate → stops/targets/trails set

=== Run 3 (5 min later) ===
uv run python scripts/trade.py testnet  
# → if in ACTIVE → manage risk → check exits
# → if signal flips → market close order
# → Summary shows realized PnL, positions, portfolio value
```

## What to Observe

1. **Fill**: limit order appears in `pending()` → `fillSz` grows → `FILLED`
2. **Transition**: PENDING → ACTIVE via `_decide_from_pending`
3. **Risk**: stops/targets logged, trailing stop activates at 2×ATR profit
4. **Exit**: signal flip → `Decision.exit` → market close order
5. **Report**: Summary shows `Total: $X (+Y%)` with position breakdown

## GitHub Actions (Production)

The 4h schedule works for production because orders persist indefinitely.
Manual `workflow_dispatch` trigger available for on-demand runs.

```bash
# Trigger from GitHub UI: Actions → qooi-testnet-4h → Run workflow
# Or use gh CLI:
gh workflow run trade-testnet-4h.yml
```
