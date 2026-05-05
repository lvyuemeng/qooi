# Testnet deployment

## Architecture

```
compute_signal()                   LiveExecutor.step()
  (offline, no API key)              (online, reads signal file)
      │                                    │
      ▼                                    ▼
data/signals/                         Read → check staleness
  ETH_USDT_4h.json                    Place limit order (post_only)
                                      Cancel if timeout (120s)
                                      Log → data/logs/
```

**Layers** (all in `src/qooi/exchange/trading.py`):

| Class/Function | Purpose | Needs API key |
|---|---|---|
| `TradingClient` | place/cancel orders, balance, positions | Yes |
| `compute_signal(symbol, tf)` | run pipeline → write JSON signal file | No |
| `LiveExecutor` | read signal file → place limits → log | No (dry_run) / Yes (live) |
| `PortfolioRunner` | run N executors from config | Same as executor |

## Setup

```bash
# .env file
echo 'OKX_API_KEY=demo_key' > .env
echo 'OKX_SECRET_KEY=demo_secret' >> .env
echo 'OKX_PASSPHRASE=demo_pass' >> .env

uv sync
```

## Usage

### Dry run (verify)

```python
from qooi.exchange.trading import LiveExecutor, PortfolioConfig, PortfolioRunner

# Single asset
e = LiveExecutor(symbol="ETH-USDT", timeframe="4h", dry_run=True)
e.step()

# Portfolio
config = PortfolioConfig(
    pairs=[
        {"symbol": "ETH-USDT", "tf": "4h", "capital": 100, "risk_pct": 0.03},
        {"symbol": "SOL-USDT", "tf": "4h", "capital": 50, "risk_pct": 0.05},
    ],
    dry_run=True,
)
PortfolioRunner(config).step()
```

### Live (testnet)

```python
LiveExecutor(symbol="ETH-USDT", timeframe="4h", dry_run=False).step()
```

## Offline / not always on

### Manual (two-step)

```bash
# Step 1: compute (anytime, offline)
uv run python -c "from qooi.exchange.trading import compute_signal; compute_signal('ETH-USDT','4h')"

# Step 2: execute (within 4h of bar close)
uv run python -c "from qooi.exchange.trading import LiveExecutor; LiveExecutor(symbol='ETH-USDT',timeframe='4h',dry_run=False).step()"
```

### Cron / Task Scheduler

```cron
0 */4 * * * cd /path/to/qooi && uv run python -m qooi.exec.run ETH-USDT 4h
```

### GitHub Actions (recommended — free, no server, native Python)

Deploy via `.github/workflows/trade-4h.yml`:

1. Set secrets: GitHub repo → Settings → Secrets → Actions
   - `OKX_API_KEY`, `OKX_SECRET_KEY`, `OKX_PASSPHRASE`
2. Push — workflow runs every 4h automatically
3. Manual test: Actions tab → `qooi-4h` → Run workflow
4. Logs: uploaded as artifacts (90-day retention)

**Start/Stop**: disable workflow in Actions tab. Re-enable to resume. No delete needed.

**Cost**: free (180 min/month, 2000 min free limit).

## Progression

```
Phase 1: Dry run (2 weeks) — verify signals, no orders
Phase 2: Micro live     (1 week) — capital=10, risk_pct=0.01
Phase 3: Scaled live    (ongoing) — capital=100, risk_pct=0.03
```

## Monitoring

```bash
# Recent events
tail -5 data/logs/exec_ETH_USDT_4h.jsonl

# Portfolio summary
cat data/logs/portfolio_summary.txt

# Live counts
cat data/logs/exec_ETH_USDT_4h.jsonl | jq 'select(.event=="order") | .status' | sort | uniq -c
```
