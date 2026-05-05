# Testnet workflow

## Architecture

```
Layer 1 (offline)           Layer 2 (online — API key needed)
compute_signal()             LiveExecutor.step()
      │                            │
      ▼                            ▼
data/signals/                 Read signal file
  ETH_USDT_4h.json ──────────► Check staleness
                               Get live OBI
                               Place limit order
                               Cancel if timeout
                               Log → JSONL
```

Signal computation uses cached OHLCV — no API key, no internet needed.
Executor reads signal file, places orders via OKX demo (flag='1').

## Setup (once)

```bash
# 1. Create API keys for testnet at okx.com → API Management
#    Select "Demo Trading" (flag='1')
#    Permissions: Read + Trade

# 2. Create .env
echo 'OKX_API_KEY=your_demo_key' > .env
echo 'OKX_SECRET_KEY=your_demo_secret' >> .env
echo 'OKX_PASSPHRASE=your_passphrase' >> .env

# 3. Sync
uv sync
```

## Workflow for offline / part-time users

### Option A: Manual (check once per bar)

```bash
# At bar close (e.g. 10:00 for 4H bar), run:
uv run python -c "
from qooi.exchange.live_executor import compute_signal
compute_signal('ETH-USDT', '4h')
print('Signal written to data/signals/')
"

# Then anytime within next 4h, execute:
uv run python -c "
from qooi.exchange.live_executor import LiveExecutor
e = LiveExecutor(symbol='ETH-USDT', timeframe='4h', dry_run=False)
r = e.step()
print('Order placed' if r else 'No signal')
"
```

### Option B: Cron / Task Scheduler (auto, no computer always on)

**Windows Task Scheduler:**
```
Trigger: Daily, every 4 hours starting at 02:00 UTC
Action: uv run python -m qooi.exec.signal_runner ETH-USDT 4h
        uv run python -m qooi.exec.signal_runner ETH-USDT 4h --live
```

**Linux cron:**
```cron
0 */4 * * * cd /path/to/qooi && uv run python -m qooi.exec.signal_runner ETH-USDT 4h
5 */4 * * * cd /path/to/qooi && uv run python -m qooi.exec.signal_runner ETH-USDT 4h --live
```

### Option C: Cloud free tier (always-on, $0)

Deploy to AWS Lambda / Google Cloud Functions / Vercel cron:

```python
# lambda_handler.py — trigger every 4h
from qooi.exchange.live_executor import compute_signal
def handler(event, context):
    compute_signal("ETH-USDT", "4h")
    return {"status": "ok"}
```

Signal file stored on S3/GCS. Executor reads from cloud bucket.

## Dry run → live progression

```
Phase 1: Dry run (2 weeks)
  e = LiveExecutor(dry_run=True)
  -> Verify: log file shows correct signal decisions
  -> Verify: orders would have been profitable (compare signal vs subsequent price)

Phase 2: Micro-sized live (1 week)
  e = LiveExecutor(dry_run=False, capital=10, max_position_pct=0.01)
  -> Real testnet orders at 0.1 USDT size
  -> Verify: orders fill, P&L tracks, no API errors

Phase 3: Scaled live (ongoing)
  e = LiveExecutor(dry_run=False, capital=100, max_position_pct=0.03)
  -> Normal position sizing
```

## Monitoring

```bash
# View recent events
cat data/logs/exec_ETH_USDT_4h.jsonl | tail -5 | jq .

# Count orders by status
cat data/logs/exec_ETH_USDT_4h.jsonl | jq 'select(.event=="order") | .status' | sort | uniq -c

# Profit/loss analysis
uv run python -c "
import polars as pl
log = pl.read_ndjson('data/logs/exec_ETH_USDT_4h.jsonl')
orders = log.filter(pl.col('event')=='order')
print(f'Orders: {len(orders)}')
print(f'Side mix: {(orders[\"side\"]==\"buy\").sum()}L / {(orders[\"side\"]==\"sell\").sum()}S')
"
```

## Per-pair configuration reference

```python
from qooi.exchange.live_executor import LiveExecutor

# Conservative — ETH 4H (best single Sharpe 0.95, DD 12%)
e = LiveExecutor(symbol="ETH-USDT", timeframe="4h", capital=100, max_position_pct=0.03)

# Aggressive — SOL 4H (Sharpe 0.67, DD 5.3%)
e = LiveExecutor(symbol="SOL-USDT", timeframe="4h", capital=50, max_position_pct=0.05)

# Portfolio — run both, share capital allocation via allocate_portfolio_weights
```
