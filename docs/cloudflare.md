# Cloudflare Workers deployment

Deploy signal computation + execution on CF Workers (free tier: 100k req/day).

## What runs where

```
Cloudflare Worker (every 4h cron trigger)
  │
  ├─ compute_signal("ETH-USDT", "4h")  → R2 bucket "qooi-signals"
  ├─ read OBI via CCXT (fetch from OKX REST)
  └─ place limit order via OKX API (testnet flag='1')
```

State stored in R2 (S3-compatible object storage, free 10GB). No server needed.

## Setup

### 1. Install wrangler

```bash
npm install -g wrangler
wrangler login
```

### 2. Create Worker

```bash
mkdir qooi-worker && cd qooi-worker
wrangler init qooi-worker
```

### 3. `wrangler.toml`

```toml
name = "qooi-worker"
main = "src/index.js"
compatibility_date = "2025-01-01"

[[r2_buckets]]
binding = "SIGNALS"
bucket_name = "qooi-signals"

[triggers]
crons = ["0 */4 * * *"]  # every 4h

[vars]
DRY_RUN = "true"
OKX_FLAG = "1"
```

### 4. Set secrets

```bash
wrangler secret put OKX_API_KEY
wrangler secret put OKX_SECRET_KEY
wrangler secret put OKX_PASSPHRASE

# Create R2 bucket
wrangler r2 bucket create qooi-signals
```

### 5. `src/index.js` (Worker code)

Since CF Workers run JS, not Python — the strategy logic must be compiled or reimplemented. Use **Python via Docker**:

```dockerfile
# Dockerfile — Python layer for CF Workers
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install uv && uv sync --frozen
CMD ["python", "-m", "qooi.exec.cloudflare_entry"]
```

Or use **Pyodide** (WASM Python in browser) via `cloudflare/python-workers`:

```toml
# wrangler.toml
main = "src/index.py"
```

```python
# src/index.py
from js import Response, fetch
import json

async def on_fetch(request):
    return Response.new("qooi worker running")
```

### 6. Deploy

```bash
wrangler deploy
```

### 7. Test

```bash
curl https://qooi-worker.YOUR_SUBDOMAIN.workers.dev
```

## Start / Stop / Update

**Start** (if disabled):
```bash
wrangler deploy  # re-deploys
```

**Stop** (disable cron + delete Worker):
```bash
# Comment out [triggers] section in wrangler.toml
wrangler deploy  # cron removed
# OR delete Worker entirely:
wrangler delete
```

**Stop temporarily without deleting:**
- Remove cron trigger from wrangler.toml → `wrangler deploy`
- Worker stays deployed but won't run on schedule

**Update code:**
```bash
# Edit src/index.py, then:
wrangler deploy
```

**Update secrets (rotating keys):**
```bash
wrangler secret put OKX_API_KEY
# enter new value
wrangler deploy  # no code change needed if only secrets changed
```

## Monitoring

```bash
# Real-time logs
wrangler tail

# Metrics
wrangler kv:key list --binding=TRADES  # if using KV for trade log

# R2 signal files
wrangler r2 object get qooi-signals/ETH_USDT_4h.json
```

## Migration: dry_run → live

1. Deploy with `DRY_RUN = "true"` (default)
2. Monitor logs via `wrangler tail` for 1-2 weeks
3. Verify signals are correct, no API errors
4. Change `DRY_RUN = "false"` → `wrangler deploy`
5. Orders now sent to OKX testnet

## Cost

- CF Workers free tier: 100k requests/day, 3M/month
- R2: 10GB free, 10M Class A ops/month
- Our usage (1 request every 4h = 6/day) → well within free tier
- Zero cost for always-on deployment
