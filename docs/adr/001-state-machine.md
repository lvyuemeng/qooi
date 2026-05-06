# ADR-001: Unified State Machine for Live Execution

**Status:** accepted  
**Date:** 2026-05-06  
**Deciders:** qooi project

## Context

The live trading executor (`LiveExecutor`) and backtest engine (`Backtest._run_single`) shared the same risk logic (`RiskConfig`, stop/target/trailing via `PositionState.check_exit()`) but diverged in state representation. The live executor used a single `self._position: PositionState | None` field with scattered `if self._position` checks. Every `trade.py` invocation created fresh executor instances with no memory of prior cycles, causing:

- **Duplicate orders**: 5+ overlapping limit orders on SOL-USDT, all capital frozen
- **No fill tracking**: orders filled between cycles were invisible to the next run
- **No dynamic risk management**: stops, targets, trail levels couldn't adapt mid-position
- **Pipeline gating**: the adaptive threshold lived inside the signal computation (`flow_pipeline.py`), using a synthetic backtest loop rather than real PnL history

The backtest used loose local variables (`active`, `stop_price`, `trailing_high`) that lacked structural parity with live execution.

## Decision

### State Machine

Introduce a formal `State` enum with four lifecycle states:

```
IDLE → PENDING → ACTIVE → EXITING → IDLE
```

| State | Meaning | Exchange Has |
|-------|---------|-------------|
| `IDLE` | Nothing open | Nothing |
| `PENDING` | Limit order placed, unfilled | Open order(s) |
| `ACTIVE` | Position filled, managing risk | Open position |
| `EXITING` | Exit order placed | Exit order + position |

The `LiveExecutor` holds `self._state: State` alongside `self._position: PositionState`. Dispatch is via `match self._state` rather than conditional checks on `fill_status` or `position is None`.

### Dynamic Risk Rules (within states)

| State | Rule | Trigger |
|-------|------|---------|
| `PENDING` | Time decay | Order age > timeout → cancel |
| `PENDING` | Signal decay | Signal weakened >50% → reduce size; flipped → cancel |
| `ACTIVE` | Breakeven stop | Unrealized PnL > 1× ATR → stop to entry |
| `ACTIVE` | Scale-out | Price reaches 50% target → partial exit |
| Any | Circuit breaker | 3 consecutive `_place()` errors → halt |

### Adaptive Threshold Relocation

The adaptive threshold (`_thresh`, `_ema_update`) moved from `flow_pipeline.py` to `LiveExecutor`. The signal pipeline now produces raw signals. The executor applies its own threshold based on **real** `self._pnl_ema` (updated on each exit), not a synthetic backtest loop. Cold start (`pnl_ema ≈ 0`) → threshold = 0.15 (permissive).

### API Reconciliation (Layer 1)

`LiveExecutor.sync()` queries OKX at the start of every `step()`:
- `pending()` → if multiple orders for this symbol, cancel oldest, keep newest
- `positions()` → if position exists, adopt it as ACTIVE
- Neither → stay at current state

### Decision Data Model

`Decision` dataclass extended with risk adjustment fields:
- `new_stop`, `new_target` — move stops/targets without state transition
- `amend_px`, `amend_sz` — modify unfilled limit orders via `POST /amend-order`
- `scale_out_pct` — partial market exit

## Consequences

### Positive

- **No state leakage**: `match self._state` enforces valid transitions at the type level
- **Backtest/live parity**: same `State`, `PositionState`, `Decision` types usable by both engines
- **Dynamic risk operations** are first-class actions within states, not hidden in conditional branches
- **Adaptive threshold** driven by real PnL, not pipeline-internal simulation
- **Duplicate prevention**: `sync()` cancels old orders on startup (when API is reachable)

### Negative

- **API dependency**: `sync()` requires `pending()` and `positions()` API calls. Network failures leave the executor blind.
- **Added complexity**: 4 state handlers instead of 1 `_decide()` method. Each handler is small (~15 lines), but there are more methods.
- **Backtest not yet ported**: `Backtest._run_single()` still uses raw scalar state. Porting is Phase C.

## Phase Plan

### Phase A — Fix Execution Surface (current)

**Goal:** Reliable API calls, fallback when API fails, actual fills to exercise risk rules.

- **A1: Retry + timeout on API calls.** Add retry loop (3 attempts, 1s backoff) and explicit timeout to `TradingClient.pending()`, `.positions()`, `.place()`, `.cancel()`.
- **A2: Resume from logs fallback (Layer 2).** When all API attempts fail, scan `data/logs/exec_{symbol}_{tf}.jsonl` to reconstruct trade count, last order ID, and position state from the append-only audit trail.
- **A3: Regular limit orders on SOL.** Switch SOL-USDT from `post_only` to regular limit orders so fills actually happen. ETH stays `post_only` for cost efficiency.

### Phase B — Order Management Strategy ✅

**Goal:** Track the same order across cycles via exchange state, not local files. Amend orders to chase price.

- **B1: `cTime`-based order age.** `_adopt_order` uses OKX `cTime` field (ms) for accurate `placed_at`, enabling cross-run time decay.
- **B2: Price chasing.** `_decide_from_pending` amends limit price to 0.2% below/above market when distance exceeds 0.5%. Order "breathes" — stays near market for fills on retracement.
- **B3: Validate risk rules.** Breakeven stop, signal decay, adaptive threshold, circuit breaker, scale-out — all implemented. Exercise when first position fills.
- **Note:** OKX testnet expires orders between cycles (~15m lifespan). On production, orders persist indefinitely and B1+B2 will manage them across runs.

### Phase E — Signal Optimization ✅

**Decision:** Replace multi-factor ensemble with pure OFI flow signal.

**Rationale:**
- Ensemble IC: +0.021 (4-bar forward) — near zero predictive power
- OFI flow IC: +0.051 — 2.4× stronger, works in all market conditions
- 57% fewer signals with |sig|>0.4 filter, 36% fewer trades
- Sharpe: ensemble -0.04 → OFI flow **+1.37** (market, lev=0.4, 2×/3× stops)
- Interpretable: "order flow imbalance, only strong signals"

**Pipeline:** `add_indicators → add_regime_features → add_ofi_flow_columns`, magnitude filter at |OFI| ≥ 0.35.

### Phase F — Cross-Asset Normalization ✅

**Decision:** Fix OFI flow normalization from `net_flow / (atr × close)` to `net_flow / vol_total` (fraction of directional volume).

**Rationale:** Old formula had 1,450× price-level bias (BTC $87K vs ETH $2.4K). BTC IC went from -0.15 to +0.08. Cross-asset thresholds now comparable.

**Results (optimal per-asset):**

| Asset | Threshold | Sharpe | DD | Return | Trades | WR |
|-------|-----------|--------|-----|--------|--------|-----|
| **SOL-USDT** | 0.35 | **+1.88** | 7.2% | +30.7% | 248 | 51% |
| **BTC-USDT** | 0.25 | **+1.52** | 6.8% | +19.7% | 447 | 57% |
| XRP-USDT | 0.45 | +0.18 | 9.4% | +1.7% | 71 | 54% |

### Phase C — Portfolio Control + Backtest Parity

**Goal:** Production hardening and honest backtesting.

- **C1: Frozen-capital gate (Layer 3).** In `PortfolioRunner`, skip entries if total frozen capital > 50% of available balance.
- **C2: Backtest port to State + PositionState.** Refactor `Backtest._run_single()` to use the same `State` enum and `PositionState` model. Simulate limit orders (fill only if `bar.low <= order.px`). Simulate expiry after N bars.
- **C3: Order type per-pair config.** Add `ord_type` field to portfolio pair config (currently hardcoded `post_only`).

### Phase D — Risk & Order Management Hardening

**Goal:** Close backtest/live divergence, tighten risk controls. See [ADR-002](./002-risk-order-management.md) for full audit.

- **D1: Signal-based exit from ACTIVE.** Exit position when signal flips direction — closes the critical backtest/live gap.
- **D2: Trailing stop activation.** Only trail after `trailing_activation_mult × ATR` profit. Tighten trail distance to 1× ATR.
- **D3: Signal preservation through resume.** Carry `entry_sig` through `_adopt_order()` so signal decay works cross-run.
- **D4: Timeframe-aware timeout.** Derive `limit_timeout_sec` from bar duration (e.g., 8h for 4h bars).

### Current Gaps

| ID | Gap | Priority | Status |
|----|-----|----------|--------|
| C1 | Frozen-capital gate | High | ✅ Done |
| — | SOL data depth | High | ✅ Done (LBank, 1999 bars) |
| — | OOS walk-forward | High | ✅ Done — SOL +0.19 OOS |
| — | Trailing target | Medium | ❌ 3 variants tested, all degrade Sharpe |
| — | Time-based exit (N bars max) | Medium | ❌ N∈{8,12,24,48} tested, all reduce returns |
| — | Cross-asset threshold calibration | High | ✅ Normalization fixed. BTC Sharpe -7.42→**+1.52**, SOL +0.92→**+1.88**. |
| — | Market order fallback | Low | Deferred (SOL is already market) |

### Production Readiness

**Status: Conditional yes for testnet. NO for live.**

| Layer | Ready? | Evidence |
|-------|--------|----------|
| Signal pipeline | ✅ | OFI flow, IC +0.051, Sharpe +1.37 |
| Risk management | ✅ | D1 (signal exit), D2 (trailing), breakeven, all verified |
| Backtest parity | ✅ | Same State, PositionState, RiskConfig across engines |
| Execution surface | ✅ | Retry, log fallback, per-pair order types |
| Order management | ✅ | sync(), price chasing, time-frame timeout |
| Pre-flight safety | ✅ | Balance/position/pending printed before execution |
| GitHub Actions | ✅ | Workflow correct, env vars set, artifacts uploaded |
| Frozen-capital gate | ❌ | No limit on total frozen capital — both assets can lock up |
| SOL confidence | ❌ | 400 bars vs ETH's 2000 — insufficient statistical confidence |
| OOS validation | ❌ | OFI flow tested on full dataset only — no walk-forward |

**Verdict:**
- Testnet: continue running. The system tracks orders, manages risk, and survives API failures. Limited capital exposure.
- Live production: hold until C1 (frozen-capital gate) is implemented and SOL data reaches 2000+ bars.

### Goals

| Goal | How |
|------|-----|
| No multiple unfilled orders | `sync()` cancels dupes + L3 frozen-capital gate |
| No balance exhaustion | L3 gate: reject entries when >50% frozen |
| Better Sharpe | Honest backtest (C2) → signal refinement with limit-order simulation |
| Lower drawdown | Breakeven stop (A) + signal decay (A) + adaptive threshold (done) |
| Structural method calling | `match state` dispatch → 4 pure handler methods |
| Side effect separation | `_decide_*()` methods are pure; `_apply_amend()`, `_place()`, `_cancel()` handle I/O |
| Ergonomic API | Single `step()` call handles sync → decide → execute; `--trace` mode for debugging |
