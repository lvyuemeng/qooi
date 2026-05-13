# Workflow FAQ

## How to start/stop

| Workflow | Start | Stop |
|---|---|---|
| `qooi-test-1h` (auto) | Enabled by default. Push to `main` triggers schedule. | GitHub → Actions → click workflow → `...` → Disable workflow |
| `qooi-live-1h` (manual) | Actions → `qooi-live-1h` → Run workflow → fill form | Only runs when manually triggered (no schedule) |

## Live vs test difference

| | Test | Live |
|---|---|---|
| OKX flag | `"1"` | `"0"` |
| Secrets | `OKX_API_KEY_TEST` | `OKX_API_KEY_LIVE` |
| Schedule | Every 1h | Manual only |
| Safety gate | None | Must type `LIVE` |
| Dry run | Default `false` | Default `true` (must toggle) |
| Funds | Simulated | Real USDT on OKX |
