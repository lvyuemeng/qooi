# Workflow FAQ

## How to start/stop

| Workflow | Start | Stop |
|---|---|---|
| `qooi-testnet-4h` (auto) | Enabled by default. Push to `main` triggers schedule. | GitHub → Actions → click workflow → `...` → Disable workflow |
| `qooi-live-4h` (manual) | Actions → `qooi-live-4h` → Run workflow → fill form | Only runs when manually triggered (no schedule) |

## How to read P&L / metrics

```python
from qooi.exchange.trading import LiveExecutor

report = LiveExecutor.report("ETH-USDT", "4h")
print(report)
# {'trades': 24, 'win_rate_pct': 54.2, 'sharpe': 0.87, 'mean_ret_pct': 0.152}
```

Or read raw JSONL:

```bash
cat data/logs/exec_ETH_USDT_4h.jsonl | jq 'select(.event=="order")'

# GitHub: download artifact from Actions run → unzip → same format
```

## How leverage + position size works

```text
sz = capital × max_position_pct × |signal| × leverage / entry_px
```

| Param | Default | Meaning |
|---|---|---|
| `capital` | 1000 | USDT allocated to this pair |
| `risk_pct` | 0.03 (3%) | % of capital risked per trade |
| `leverage` | 1.0 | Multiplier. 2.0 = 2× position size |
| `|signal|` | 0.0–1.0 | Confidence from ensemble |

Example: `capital=100, risk_pct=0.03, leverage=1.0, signal=0.5, entry_px=2000`
→ `sz = 100 × 0.03 × 0.5 × 1.0 / 2000 = 0.00075 BTC`

## Initial capital

Set in workflow YAML `capital` field. Current defaults:

- ETH: 100 USDT
- SOL: 50 USDT

Change by editing the `dict(symbol=..., capital=...)` in workflow file.

## Live vs testnet difference

| | Testnet | Live |
|---|---|---|
| OKX flag | `"1"` | `"0"` |
| Secrets | `OKX_API_KEY` | `OKX_API_KEY_LIVE` |
| Schedule | Every 4h | Manual only |
| Safety gate | None | Must type `LIVE` |
| Dry run | Default `false` | Default `true` (must toggle) |
| Funds | Simulated | Real USDT on OKX |
