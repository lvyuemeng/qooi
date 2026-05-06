# ADR-002: Dynamic Risk & Order Management Strategy

**Status:** accepted  
**Date:** 2026-05-06  
**Depends on:** ADR-001 (State Machine)  
**Deciders:** qooi project

## Context

ADR-001 established a state machine (`IDLE → PENDING → ACTIVE → EXITING`) with dynamic risk rules (breakeven stop, signal decay, price chasing, scale-out, circuit breaker). Phases A and B delivered retry logic, log fallback, per-pair order types, and order management via exchange state.

A full audit of every risk and order management path revealed six gaps. Four are high-impact and consistent with the current phase architecture.

## Audit Findings

### Profit-Stop (Take Profit)

| Component | Status |
|-----------|--------|
| Static target at 3× ATR on entry | ✅ `PositionState.enter_long()` |
| Target hit → market exit | ✅ `check_exit()` |
| Trailing target (move up in trend) | ❌ Target never moves |
| Partial scale-out at 50% target | ⚠️ Implemented, untested |

**Issue:** A static 3× ATR target caps profit in strong trends. SOL is up ~13%; a position would exit at ~3.6% (3× 1.2% ATR) while the trend continues.

### Loss-Stop (Stop Loss)

| Component | Status |
|-----------|--------|
| Initial stop at 2× ATR | ✅ |
| Breakeven stop at +1× ATR | ⚠️ One-shot: `stop_price != entry_price` guard prevents further moves |
| Trailing stop | ❌ `trail_high` starts at entry, distance = 2× ATR (same as initial stop) |
| `trailing_activation_mult` | ❌ Defined in `RiskConfig` but never used |
| Volatility-adjusted stop | ❌ Comment placeholder only |

**Issue:** The trailing stop provides zero additional protection — it triggers at the same 2× ATR distance as the initial stop. The activation threshold (2× ATR profit before trail activates) is defined but never checked.

### Order Fill

| Component | Status |
|-----------|--------|
| Limit order at ask/bid | ✅ |
| Fill check via `pending()` | ✅ |
| Price chasing (0.5% threshold) | ✅ |
| Market fallback after N attempts | ❌ |
| Timeout: 120s for 4h bars | ❌ Order cancelled before bar closes |

**Issue:** `limit_timeout_sec=120` is meaningless for 4h bars. Should be timeframe-derived (e.g., 2 bars = 8h).

### Order Cancel

| Component | Status |
|-----------|--------|
| Time decay (120s) | ⚠️ Too short for 4h |
| Signal flipped → cancel | ⚠️ `entry_sig` is 0.0 for resumed orders — check `if entry_sig and ...` silently skips |
| Signal weakened → reduce size | ⚠️ Same — dead code for cross-run orders |

**Issue:** `_adopt_order()` creates `OrderPayload` without the `signal` field. The OKX API response doesn't carry our custom data. Signal decay never fires on resumed orders.

### Order Amend/Move

| Component | Status |
|-----------|--------|
| Price chase in PENDING | ✅ |
| Breakeven stop in ACTIVE | ⚠️ One-shot |
| Target amendment | ❌ |
| Trailing stop tightening | ❌ Distance stays at 2× ATR |

### Position Exit (ACTIVE)

| Component | Status |
|-----------|--------|
| Stop/target/trail → market exit | ✅ |
| **Signal-based exit** | ❌ **Critical backtest/live divergence** |
| Time-based exit (N bars max) | ❌ |

**The signal-based exit gap:** `Backtest._run_single()` exits when `p_prev * p < 0` (signal flips direction). `LiveExecutor._decide_from_active()` never checks the current signal — it holds indefinitely regardless of what the new signal says. This means the backtest Sharpe of 0.71 is unrepresentative of live performance.

## Decision

Four fixes, all within the existing state machine architecture. No new files, no new states, no changed interfaces.

### Fix 1: Signal-Based Exit from ACTIVE (backtest parity)

In `_decide_from_active()`:

```python
# Signal flipped → exit (backtest parity)
d = 1 if self._position.order.side == "buy" else -1
if d * sr.signal < 0:
    return Decision.exit("signal_flipped", cur_close)
```

Placed before the hard exit check. This is the single most important fix — it closes the backtest/live divergence identified in ADR-001.

### Fix 2: Trailing Stop Activation

In `_decide_from_active()`, replace the commented-out volatility section:

```python
# Trailing stop activation: only trail after activation_mult × ATR profit
activation_price = self._position.entry_price + d * self._risk.trailing_activation_mult * atr_est
if d * (cur_close - activation_price) > 0:
    trail_dist = self._risk.trailing_distance_mult * atr_est
    new_stop = cur_close - d * trail_dist
    if d * (new_stop - self._position.stop_price) > 0:
        self._position.stop_price = new_stop
```

Update `RiskConfig` defaults: `trailing_distance_mult=1.0` (tight trail after activation, down from 2.0).

### Fix 3: Signal Preservation Through Resume

In `_adopt_order()`, read the `signal` field from the order payload if present. In `_resume_from_logs()`, pass the signal value. The log already stores `signal` in `OrderPayload` — it just wasn't being passed through.

### Fix 4: Timeframe-Aware Timeout

In `LiveExecutor.__init__`:

```python
bar_duration = {"1h": 3600, "4h": 14400, "1d": 86400}.get(timeframe, 14400)
self._timeout = limit_timeout_sec if limit_timeout_sec > 0 else (bar_duration * 2)
```

Default `limit_timeout_sec` in `PortfolioConfig` changes from 120 to 0 (0 = derive from timeframe).

### What Is NOT Changed

- State machine — same four states, same dispatch
- Decision dataclass — no new fields needed
- `_place()`, `_cancel()`, `_apply_amend()` — unchanged
- Signal pipeline — untouched
- `PortfolioRunner` — untouched (C1 frozen-capital gate deferred to Phase C)
- `Backtest` — already has signal-based exit; C2 port still deferred

### Deferred to Phase C

- **C1: Frozen-capital gate** — ✅ Implemented. `PortfolioRunner.step(tc=)` skips IDLE executors when >50% balance frozen.
- **C2: Honest backtest** — ✅ Done. `Backtest._run_single()` uses `State` + `PositionState` with limit-order simulation.
- **Market orders** — ✅ SOL switched to `ord_type="market"` for guaranteed fills.
- **Trailing target** — ❌ Tested 3 variants (unconditional, pre-exit, signal-gated). All degraded Sharpe. OFI flow + static 3× ATR target is optimal.
- **Time-based exit** — ❌ Tested N∈{8,12,24,48}. None improved metrics. Winning positions last longer; cutting them reduces returns.
- **SOL data depth** — ✅ Refreshed to 1999 bars from LBank.
- **OOS walk-forward** — ✅ Run on 4 assets. SOL shows OOS Sharpe +0.19 (weak but positive). ETH SWAP incompatible. BTC needs per-asset threshold calibration.

### New Findings (2026-05-06)

- **Trailing target is consistently harmful** for OFI flow signal. The signal's predictive window is short — trailing the target moves it beyond the signal's horizon, causing exits via stops. Signal-based exit (D1) already handles position management.
- **Time-based exit is also harmful.** Winning trades tend to hold longer; forcing exits after N bars cuts winners short.
- **OFI flow magnitude is not cross-asset comparable.** BTC has std=0.03 vs ETH std≈0.3. A fixed 0.4 threshold filters out all BTC trades. Per-asset percentile-based thresholds needed for multi-asset.
- **ETH SWAP data ≠ spot data.** Perpetual swap OHLCV differs from spot, making swap data unsuitable for spot strategy backtesting.

## Consequences

### Positive

- **Backtest/live parity**: signal-based exit closes the biggest divergence
- **Tighter risk control**: trailing stop activates at 2× ATR profit, trails at 1× ATR (was: trails at 2× ATR from entry)
- **Signal decay works cross-run**: resumed orders check whether signal has weakened
- **Timeouts match timeframes**: 4h bars get 8h timeout instead of 120s
- **All changes are local**: four methods modified, no new files, no API changes

### Negative

- Signal-based exit may reduce win rate (positions exited earlier on signal flips). The backtest with signal-based exits shows Sharpe 0.71 — this is the honest number.
- Trailing stop tightening may cause premature exits in volatile markets. The 1× ATR trail distance should be validated against backtest data before going live.
