"""Live executor tests — full branch coverage of state machine, sync, execute.

Covers every branch in:
  sync()       — 6 branches
  _decide()    — 10 branches
  _decide_idle / _decide_pending / _decide_active  — all paths
  _execute()   — 5 action branches
  check_fill() — 5 branches
"""

import json
import time

import pytest

from qooi.exchange.backtest import RiskConfig
from qooi.exchange.trading import (
    Decision,
    FillFacts,
    FillStatus,
    LiveExecutor,
    OrderPayload,
    PositionState,
    SignalResult,
    SkipPayload,
    State,
    SyncFacts,
)

# ── helpers ──────────────────────────────────────────────────────────────────


def _obi(ask=2500.0, bid=2498.0, imb=0.5):
    class O:
        ask_price = ask
        bid_price = bid
        imbalance_5 = imb
    return O()


def _sig(signal=0.50, flow=0.50, threshold=0.25):
    return SignalResult(
        symbol="ETH-USDT-SWAP", timeframe="4h",
        signal=signal, flow=flow, threshold=threshold,
        timestamp=0, computed_at=0,
    )


def _pos(side="buy", sz=1, px=2500.0, signal=0.5, status=FillStatus.PLACED):
    return PositionState(
        order=OrderPayload(side=side, sz=sz, px=px, placed_at=time.time(), signal=signal),
        fill_status=status,
        entry_price=px,
    )


# ══════════════════════════════════════════════════════════════════════════════
# sync — order detection
# ══════════════════════════════════════════════════════════════════════════════


class TestSyncOrderDetection:
    """sync() must detect orders from BOTH exchange API and local log."""

    def test_sync_api_position_adopted(self):
        """API returns a position → SyncFacts.position is set."""
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        # Patch _client to return mock data
        class FakeClient:
            def pending(self): return []
            def positions(self): return [{"instId": "ETH-USDT-SWAP", "pos": "1", "avgPx": "2500"}]
        exe._client = FakeClient()
        exe._dry = False
        facts = exe.sync()
        assert facts.position is not None
        assert facts.position["instId"] == "ETH-USDT-SWAP"
        assert facts.api_ok

    def test_sync_api_order_found(self):
        """API returns pending order → SyncFacts.order is set."""
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        class FakeClient:
            def pending(self): return [{"instId": "ETH-USDT-SWAP", "ordId": "oid1", "side": "sell", "sz": "1", "px": "2276"}]
            def positions(self): return []
        exe._client = FakeClient()
        exe._dry = False
        facts = exe.sync()
        assert facts.order is not None
        assert facts.order["ordId"] == "oid1"
        assert facts.api_ok

    def test_sync_api_duplicates_detected(self):
        """API returns multiple pending orders → duplicates captured."""
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        class FakeClient:
            def pending(self): return [
                {"instId": "ETH-USDT-SWAP", "ordId": "oid1", "cTime": "1000"},
                {"instId": "ETH-USDT-SWAP", "ordId": "oid2", "cTime": "2000"},
            ]
            def positions(self): return []
        exe._client = FakeClient()
        exe._dry = False
        facts = exe.sync()
        assert facts.order is not None
        assert facts.duplicates is not None
        assert len(facts.duplicates) == 1

    def test_sync_api_empty_log_recovery(self, tmp_path, monkeypatch):
        """Bug 1 grill: API returns OK with 0 pending — sync MUST read log."""
        import qooi.exchange.trading as tmod
        monkeypatch.setattr(tmod, "LOG_DIR", tmp_path)
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        # No API orders
        class FakeClient:
            def pending(self): return []
            def positions(self): return []
        exe._client = FakeClient()
        exe._dry = False
        # Write a recent uncancelled order to the log
        log_file = tmp_path / "exec_ETH_USDT_SWAP_4h.jsonl"
        log_file.write_text(json.dumps({
            "ts": int(time.time() * 1000) - 10000,
            "event": "order",
            "symbol": "ETH-USDT-SWAP", "tf": "4h",
            "payload": {"ord_id": "log_oid", "side": "sell", "sz": 1.0,
                        "px": 2276.0, "placed_at": time.time() - 5, "signal": -0.319},
        }) + "\n")
        facts = exe.sync()
        assert facts.order is None, "API returned no order"
        assert facts.log_order is not None, "Log recovery MUST find the recent order"
        assert facts.log_order["ordId"] == "log_oid"
        assert facts.api_ok

    def test_sync_api_fail_falls_back_to_log(self, tmp_path, monkeypatch):
        """API fails → log recovery only."""
        import qooi.exchange.trading as tmod
        monkeypatch.setattr(tmod, "LOG_DIR", tmp_path)
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        class FakeClient:
            def pending(self): raise ConnectionError("down")
            def positions(self): raise ConnectionError("down")
        exe._client = FakeClient()
        exe._dry = False
        # Write a recent order
        log_file = tmp_path / "exec_ETH_USDT_SWAP_4h.jsonl"
        log_file.write_text(json.dumps({
            "ts": int(time.time() * 1000) - 10000,
            "event": "order",
            "symbol": "ETH-USDT-SWAP", "tf": "4h",
            "payload": {"ord_id": "fail_oid", "side": "buy", "sz": 1.0,
                        "px": 2500.0, "placed_at": time.time() - 5, "signal": 0.5},
        }) + "\n")
        facts = exe.sync()
        assert not facts.api_ok
        assert facts.log_order is not None
        assert facts.log_order["ordId"] == "fail_oid"

    def test_sync_dry_run_returns_empty(self):
        """dry_run with no client → empty SyncFacts."""
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        facts = exe.sync()
        assert not facts.api_ok
        assert facts.order is None
        assert facts.position is None


# ══════════════════════════════════════════════════════════════════════════════
# _decide — dispatch
# ══════════════════════════════════════════════════════════════════════════════


class TestDecideDispatch:
    def test_idle_dispatches_to_idle_method(self):
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        exe._state = State.IDLE
        d = exe._decide(_sig(0.1), _obi(), SyncFacts(), FillFacts())
        assert "weak_signal" in d.detail

    def test_pending_dispatches_to_pending_method(self):
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        exe._state = State.PENDING
        exe._position = _pos()
        d = exe._decide(_sig(0.5), _obi(), SyncFacts(), FillFacts())
        assert d.action != "enter"
        assert d.action in ("skip", "amend", "exit")

    def test_active_dispatches_to_active_method(self):
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        exe._state = State.ACTIVE
        pos = PositionState.enter_long(2500.0, 50.0, RiskConfig(), 0)
        pos.order.side = "buy"; pos.fill_status = FillStatus.FILLED; pos.entry_price = 2500.0
        exe._position = pos
        d = exe._decide(_sig(0.5), _obi(), SyncFacts(), FillFacts())
        assert d.action in ("skip", "amend", "exit")

    def test_sync_position_adopts_to_active(self):
        """facts.position → state=ACTIVE."""
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        facts = SyncFacts(position={"instId": "ETH-USDT-SWAP", "pos": "1", "avgPx": "2500"})
        d = exe._decide(_sig(0.5), _obi(), facts, FillFacts())
        assert exe._state == State.ACTIVE

    def test_sync_order_adopts_to_pending(self):
        """facts.order → state=PENDING."""
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        facts = SyncFacts(order={"instId": "ETH-USDT-SWAP", "ordId": "o1", "side": "buy", "sz": "1", "px": "2500"})
        d = exe._decide(_sig(0.5), _obi(), facts, FillFacts())
        assert exe._state == State.PENDING

    def test_sync_log_order_adopts_to_pending(self):
        """facts.log_order → state=PENDING (Bug 1 fix)."""
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        facts = SyncFacts(log_order={"instId": "ETH-USDT-SWAP", "ordId": "log1", "side": "sell", "sz": "1", "px": "2276"}, api_ok=True)
        d = exe._decide(_sig(-0.4), _obi(imb=-0.5), facts, FillFacts())
        assert exe._state == State.PENDING, f"Log order must be adopted, got {exe._state}"

    def test_fill_missing_clears_position_to_idle(self):
        """fill.missing → _position=None, state=IDLE."""
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        exe._state = State.PENDING
        exe._position = _pos()
        d = exe._decide(_sig(0.5), _obi(), SyncFacts(), FillFacts(missing=True))
        assert exe._position is None, "Missing fill must clear position"
        assert exe._state == State.IDLE

    def test_fill_filled_updates_status(self):
        """fill.filled → fill_status=FILLED, entry_price updated."""
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        exe._state = State.PENDING
        exe._position = _pos()
        d = exe._decide(_sig(0.5), _obi(), SyncFacts(), FillFacts(filled=True, filled_px=2510.0, filled_sz=1.0))
        assert exe._position.fill_status == FillStatus.FILLED
        assert exe._position.entry_price == 2510.0

    def test_fill_partial_updates_status(self):
        """fill.partial → fill_status=PARTIAL."""
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        exe._state = State.PENDING
        exe._position = _pos()
        d = exe._decide(_sig(0.5), _obi(), SyncFacts(), FillFacts(partial=True))
        assert exe._position.fill_status == FillStatus.PARTIAL


# ══════════════════════════════════════════════════════════════════════════════
# _decide_idle — entry gate
# ══════════════════════════════════════════════════════════════════════════════


class TestDecideIdle:
    def test_weak_signal_below_threshold(self):
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        exe._signal_threshold = 0.25
        d = exe._decide_idle(_sig(0.10), _obi())
        assert d.action == "skip"
        assert "weak_signal" in d.detail

    def test_clipped_to_zero_max_leverage_zero(self):
        r = RiskConfig(max_leverage=0.0)
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True, risk=r)
        exe._signal_threshold = 0.25
        d = exe._decide_idle(_sig(0.50), _obi())
        assert d.action == "skip"
        assert "clipped_to_zero" in d.detail

    def test_strong_signal_buy(self):
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True, initial_capital=10000, max_position_pct=0.5, leverage=2.0)
        exe._signal_threshold = 0.25
        d = exe._decide_idle(_sig(0.50), _obi())
        assert d.action == "enter"
        assert d.side == "buy"
        assert d.sz > 0

    def test_strong_signal_sell(self):
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True, initial_capital=10000, max_position_pct=0.5, leverage=2.0)
        exe._signal_threshold = 0.25
        d = exe._decide_idle(_sig(-0.50), _obi())
        assert d.action == "enter"
        assert d.side == "sell"

    def test_insufficient_margin(self, monkeypatch):
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True, initial_capital=500, max_position_pct=0.5, leverage=2.0, ct_val=0.1)
        exe._signal_threshold = 0.25
        monkeypatch.setattr(exe, "_free_usdt", lambda: 0.0)
        d = exe._decide_idle(_sig(0.50), _obi())
        assert d.action == "skip"
        assert "insufficient_margin" in d.detail


# ══════════════════════════════════════════════════════════════════════════════
# _decide_pending — while order is live
# ══════════════════════════════════════════════════════════════════════════════


class TestDecidePending:
    def test_position_lost(self):
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        exe._state = State.PENDING  # no _position set
        d = exe._decide_pending(_sig(0.5), _obi())
        assert d.action == "skip"
        assert "position_lost" in d.detail

    def test_filled_transitions_to_active(self):
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        exe._state = State.PENDING
        exe._position = _pos(status=FillStatus.FILLED)
        d = exe._decide_pending(_sig(0.5), _obi())
        assert d.action == "amend"
        assert d.detail == "order_filled"
        assert d.new_state == State.ACTIVE

    def test_partial_fill_skips(self):
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        exe._state = State.PENDING
        exe._position = _pos(status=FillStatus.PARTIAL)
        d = exe._decide_pending(_sig(0.5), _obi())
        assert d.action == "skip"
        assert "partial_fill" in d.detail
        assert d.new_state == State.PENDING

    def test_timeout(self):
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True, limit_timeout_sec=0)
        exe._state = State.PENDING
        old = 999999.0  # placed far in the past
        exe._position = PositionState(
            order=OrderPayload(side="buy", sz=1, px=2500, placed_at=old, signal=0.5),
            fill_status=FillStatus.PLACED, entry_price=2500,
        )
        d = exe._decide_pending(_sig(0.5), _obi())
        assert d.action == "exit"
        assert "timeout" in d.detail

    def test_signal_flipped(self):
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True, limit_timeout_sec=99999)
        exe._state = State.PENDING
        exe._position = _pos(side="buy", signal=0.5)
        # Signal flips from +0.5 to -0.4
        d = exe._decide_pending(_sig(-0.4), _obi(imb=-0.5))
        assert d.action == "exit"
        assert "signal_flipped" in d.detail

    def test_signal_weakened_reduce_size(self):
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True, limit_timeout_sec=99999)
        exe._state = State.PENDING
        exe._position = _pos(side="buy", signal=0.5)
        # Signal weakens to 0.2 (less than 50% of 0.5)
        d = exe._decide_pending(_sig(0.2), _obi())
        assert d.action == "amend"
        assert "signal_weakened" in d.detail
        assert d.amend_sz is not None

    def test_signal_weakened_to_zero_exits(self):
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True, limit_timeout_sec=99999)
        exe._state = State.PENDING
        exe._position = _pos(side="buy", sz=0.000001, signal=0.5)
        # Signal weakens below zero after halving → exit
        d = exe._decide_pending(_sig(0.01), _obi())
        assert d.action == "exit"
        assert "signal_weakened" in d.detail

    def test_price_chase(self):
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True, limit_timeout_sec=99999)
        exe._state = State.PENDING
        exe._position = _pos(side="buy", px=2500.0)
        # Market moved >0.5% away
        d = exe._decide_pending(_sig(0.5), _obi(ask=2520.0))
        assert d.action == "amend"
        assert d.amend_px is not None

    def test_order_outstanding(self):
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True, limit_timeout_sec=99999)
        exe._state = State.PENDING
        exe._position = _pos()
        d = exe._decide_pending(_sig(0.5), _obi())
        assert d.action == "skip"
        assert "order_outstanding" in d.detail
        assert d.new_state == State.PENDING


# ══════════════════════════════════════════════════════════════════════════════
# _decide_active — risk management
# ══════════════════════════════════════════════════════════════════════════════


class TestDecideActive:
    def test_position_lost(self):
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        exe._state = State.ACTIVE  # no _position
        d = exe._decide_active(_sig(0.5), _obi())
        assert "position_lost" in d.detail

    def test_signal_flip_exit(self):
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        pos = PositionState.enter_long(2500.0, 50.0, RiskConfig(), 0)
        pos.order.side = "buy"; pos.fill_status = FillStatus.FILLED; pos.entry_price = 2500.0
        exe._position = pos; exe._state = State.ACTIVE
        d = exe._decide_active(_sig(-0.4), _obi())
        assert d.action == "exit"
        assert "signal_flipped" in d.detail

    def test_time_exit(self):
        r = RiskConfig(max_bars_held=5)
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True, risk=r)
        pos = PositionState.enter_long(2500.0, 50.0, r, 0)
        pos.order.side = "buy"; pos.fill_status = FillStatus.FILLED; pos.entry_price = 2500.0
        pos.bars_held = 10
        exe._position = pos; exe._state = State.ACTIVE
        d = exe._decide_active(_sig(0.5), _obi())
        assert d.action == "exit"
        assert "time" in d.detail

    def test_stop_exit(self):
        pos = PositionState.enter_long(2500.0, 50.0, RiskConfig(), 0)
        pos.order.side = "buy"; pos.fill_status = FillStatus.FILLED; pos.entry_price = 2500.0
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        exe._position = pos; exe._state = State.ACTIVE
        d = exe._decide_active(_sig(0.5), _obi(ask=2390.0))
        assert d.action == "exit"
        assert d.detail == "stop"

    def test_target_exit(self):
        pos = PositionState.enter_long(2500.0, 50.0, RiskConfig(), 0)
        pos.order.side = "buy"; pos.fill_status = FillStatus.FILLED; pos.entry_price = 2500.0
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        exe._position = pos; exe._state = State.ACTIVE
        d = exe._decide_active(_sig(0.5), _obi(ask=2660.0))
        assert d.action == "exit"
        assert d.detail == "target"

    def test_breakeven(self):
        r = RiskConfig(atr_stop_mult=2.0)
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True, risk=r)
        pos = PositionState.enter_long(2500.0, 50.0, r, 0)
        pos.order.side = "buy"; pos.fill_status = FillStatus.FILLED; pos.entry_price = 2500.0
        pos.stop_price = 2400.0  # not yet at breakeven
        exe._position = pos; exe._state = State.ACTIVE
        # Price up 5%, ATR ~2%, profit > ATR
        d = exe._decide_active(_sig(0.5), _obi(ask=2625.0))
        assert d.action == "amend"
        assert d.detail == "breakeven_stop"
        assert d.new_stop == 2500.0
        assert d.new_state == State.ACTIVE

    def test_holding(self):
        r = RiskConfig(max_bars_held=0)
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True, risk=r)
        pos = PositionState.enter_long(2500.0, 50.0, r, 0)
        pos.order.side = "buy"; pos.fill_status = FillStatus.FILLED; pos.entry_price = 2500.0
        pos.stop_price = 2500.0  # already at breakeven
        exe._position = pos; exe._state = State.ACTIVE
        d = exe._decide_active(_sig(0.5), _obi(ask=2520.0))
        assert d.action == "skip"
        assert d.detail == "holding"
        assert d.new_state == State.ACTIVE


# ══════════════════════════════════════════════════════════════════════════════
# _execute — side effects
# ══════════════════════════════════════════════════════════════════════════════


class TestExecute:
    def test_enter_executes_dry_placement(self):
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True, initial_capital=10000, max_position_pct=0.5, leverage=2.0)
        exe._signal_threshold = 0.25
        exe._state = State.IDLE
        d = Decision.enter("buy", 1, 2500.0)
        result = exe._execute(d, _sig(0.5), _obi())
        assert result is not None
        assert exe._state == State.PENDING
        assert exe._position is not None
        assert "dry_" in exe._position.order.ord_id

    def test_exit_filled_places_market_close(self):
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        pos = PositionState.enter_long(2500.0, 50.0, RiskConfig(), 0)
        pos.order.side = "buy"; pos.fill_status = FillStatus.FILLED; pos.entry_price = 2500.0
        pos.order.px = 2500.0; pos.order.sz = 1; pos.order.signal = 0.5
        exe._position = pos; exe._state = State.ACTIVE
        d = Decision.exit("stop", 2400.0)
        exe._execute(d, _sig(0.5), _obi())
        assert exe._state == State.IDLE

    def test_exit_pending_cancels(self):
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        pos = _pos()
        exe._position = pos; exe._state = State.PENDING
        d = Decision.exit("timeout", 2490.0)
        exe._execute(d, _sig(0.5), _obi())
        assert exe._position is None
        assert exe._state == State.IDLE

    def test_amend_applies_new_stop(self):
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        pos = _pos()
        exe._position = pos; exe._state = State.ACTIVE
        d = Decision.amend("breakeven", new_stop=2500.0, new_state=State.ACTIVE)
        exe._execute(d, _sig(0.5), _obi())
        assert exe._position.stop_price == 2500.0
        assert exe._state == State.ACTIVE

    def test_skip_logs_and_stays(self):
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        d = Decision.skip("holding", new_state=State.ACTIVE)
        exe._state = State.ACTIVE
        exe._execute(d, _sig(0.5), _obi())
        assert exe._state == State.ACTIVE


# ══════════════════════════════════════════════════════════════════════════════
# check_fill — fill status detection
# ══════════════════════════════════════════════════════════════════════════════


class TestCheckFill:
    def test_no_position_returns_empty(self):
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        f = exe.check_fill()
        assert not f.filled and not f.partial and not f.missing

    def test_already_filled_returns_empty(self):
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        exe._position = _pos(status=FillStatus.FILLED)
        f = exe.check_fill()
        assert not f.filled and not f.partial and not f.missing

    def test_dry_run_returns_empty(self):
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        exe._position = _pos()
        f = exe.check_fill()
        assert not f.filled

    def test_order_not_in_pending_becomes_missing(self):
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        class FakeClient:
            def pending(self): return []
        exe._client = FakeClient()
        exe._dry = False
        old = time.time() - 400  # > 5 min ago
        exe._position = PositionState(
            order=OrderPayload(side="buy", sz=1, px=2500, placed_at=old, signal=0.5, ord_id="old"),
            fill_status=FillStatus.PLACED, entry_price=2500,
        )
        f = exe.check_fill()
        assert f.missing, "Order absent from pending >5 min must be marked missing"

    def test_order_found_filled(self):
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        class FakeClient:
            def pending(self): return [{"ordId": "oid1", "fillSz": "1.0", "fillPx": "2510"}]
        exe._client = FakeClient()
        exe._dry = False
        exe._position = PositionState(
            order=OrderPayload(side="buy", sz=1, px=2500, placed_at=time.time(), signal=0.5, ord_id="oid1"),
            fill_status=FillStatus.PLACED, entry_price=2500,
        )
        f = exe.check_fill()
        assert f.filled
        assert f.filled_px == 2510.0


# ══════════════════════════════════════════════════════════════════════════════
# check_exit — stop/target
# ══════════════════════════════════════════════════════════════════════════════


class TestCheckExit:
    def test_long_stop(self):
        pos = PositionState.enter_long(2500.0, 50.0, RiskConfig(), 0)
        pos.order.side = "buy"
        assert pos.check_exit(2399.0, 50.0, RiskConfig()) == "stop"

    def test_long_target(self):
        pos = PositionState.enter_long(2500.0, 50.0, RiskConfig(), 0)
        pos.order.side = "buy"
        assert pos.check_exit(2651.0, 50.0, RiskConfig()) == "target"

    def test_long_no_exit(self):
        pos = PositionState.enter_long(2500.0, 50.0, RiskConfig(), 0)
        pos.order.side = "buy"
        assert pos.check_exit(2520.0, 50.0, RiskConfig()) is None

    def test_short_stop(self):
        pos = PositionState.enter_short(2500.0, 50.0, RiskConfig(), 0)
        pos.order.side = "sell"
        assert pos.check_exit(2601.0, 50.0, RiskConfig()) == "stop"

    def test_short_target(self):
        pos = PositionState.enter_short(2500.0, 50.0, RiskConfig(), 0)
        pos.order.side = "sell"
        assert pos.check_exit(2349.0, 50.0, RiskConfig()) == "target"

    def test_target_disabled_when_negative(self):
        """target_price=-1 means no target set."""
        pos = PositionState(entry_price=2500, stop_price=2400, target_price=-1, order=OrderPayload(side="buy"))
        assert pos.check_exit(3000.0, 50.0, RiskConfig()) is None

    def test_stop_disabled_when_negative(self):
        """stop_price=-1 means no stop set."""
        pos = PositionState(entry_price=2500, stop_price=-1, target_price=2650, order=OrderPayload(side="buy"))
        assert pos.check_exit(10.0, 50.0, RiskConfig()) is None


# ══════════════════════════════════════════════════════════════════════════════
# _read_log_order — log parsing
# ══════════════════════════════════════════════════════════════════════════════


class TestReadLogOrder:
    def test_no_log_file(self, tmp_path, monkeypatch):
        import qooi.exchange.trading as tmod
        monkeypatch.setattr(tmod, "LOG_DIR", tmp_path)
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        assert exe._read_log_order() is None

    def test_cancelled_order_not_returned(self, tmp_path, monkeypatch):
        import qooi.exchange.trading as tmod
        monkeypatch.setattr(tmod, "LOG_DIR", tmp_path)
        log_file = tmp_path / "exec_ETH_USDT_SWAP_4h.jsonl"
        log_file.write_text(
            json.dumps({"ts": 1, "event": "order", "payload": {"ord_id": "oid1", "side": "buy", "sz": 1, "px": 2500, "placed_at": time.time() - 5, "signal": 0.5}}) + "\n" +
            json.dumps({"ts": 2, "event": "cancel", "payload": {"ord_id": "oid1", "reason": "timeout"}}) + "\n"
        )
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        assert exe._read_log_order() is None, "Cancelled orders must not be returned"

    def test_stale_order_not_returned(self, tmp_path, monkeypatch):
        import qooi.exchange.trading as tmod
        monkeypatch.setattr(tmod, "LOG_DIR", tmp_path)
        log_file = tmp_path / "exec_ETH_USDT_SWAP_4h.jsonl"
        old = time.time() - 100000  # very old
        log_file.write_text(json.dumps({
            "ts": 1, "event": "order",
            "payload": {"ord_id": "stale", "side": "buy", "sz": 1, "px": 2500,
                        "placed_at": old, "signal": 0.5},
        }) + "\n")
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        assert exe._read_log_order() is None, "Stale orders (past timeout) must not be returned"


# ══════════════════════════════════════════════════════════════════════════════
# Duplicate prevention — end-to-end
# ══════════════════════════════════════════════════════════════════════════════


class TestDuplicatePrevention:
    def test_pending_state_prevents_duplicate_entry(self):
        """In PENDING state, _decide dispatches to _decide_pending, never enter."""
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        exe._state = State.PENDING
        exe._position = _pos()
        d = exe._decide(_sig(0.6), _obi(), SyncFacts(), FillFacts())
        assert d.action != "enter", f"PENDING must not enter, got {d.action}"

    def test_sync_empty_with_log_order_prevents_duplicate(self, tmp_path, monkeypatch):
        """Bug 1 end-to-end: API empty, log has order → PENDING → skip, not enter."""
        import qooi.exchange.trading as tmod
        monkeypatch.setattr(tmod, "LOG_DIR", tmp_path)
        log_file = tmp_path / "exec_ETH_USDT_SWAP_4h.jsonl"
        log_file.write_text(json.dumps({
            "ts": int(time.time() * 1000) - 10000,
            "event": "order",
            "symbol": "ETH-USDT-SWAP", "tf": "4h",
            "payload": {"ord_id": "dup_oid", "side": "sell", "sz": 1.0,
                        "px": 2276.0, "placed_at": time.time() - 5, "signal": -0.319},
        }) + "\n")
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        class FakeClient:
            def pending(self): return []
            def positions(self): return []
        exe._client = FakeClient()
        exe._dry = False
        # This simulates what step() does:
        facts = exe.sync()
        sr = _sig(-0.4, threshold=0.25)
        d = exe._decide(sr, _obi(imb=-0.5), facts, FillFacts())
        assert d.action != "enter", f"Duplicate prevented: got {d.action} (log order should cause PENDING → skip)"

    def test_phantom_cancel_allows_reentry(self):
        """Bug 2: After cancel clears position to IDLE, strong signal re-enters."""
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True, initial_capital=10000, max_position_pct=0.5, leverage=2.0)
        exe._state = State.PENDING
        exe._position = _pos()
        # Simulate fill.missing → position=None, state=IDLE
        d = exe._decide(_sig(0.5), _obi(), SyncFacts(), FillFacts(missing=True))
        assert exe._state == State.IDLE, "After missing fill, state must be IDLE"
        assert exe._position is None
        # Next cycle: strong signal → should enter
        d2 = exe._decide(_sig(0.5), _obi(), SyncFacts(), FillFacts())
        assert d2.action == "enter", f"After phantom cancel, strong signal must re-enter, got {d2.action}: {d2.detail}"


# ══════════════════════════════════════════════════════════════════════════════
# Adaptive threshold
# ══════════════════════════════════════════════════════════════════════════════


class TestAdaptiveThreshold:
    def test_min_when_ema_zero(self):
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        exe._signal_threshold = 0.25; exe._pnl_ema = 0.0
        assert abs(exe._entry_threshold() - 0.15625) < 0.01

    def test_positive_ema_lowers(self):
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        exe._signal_threshold = 0.25; exe._pnl_ema = 0.01
        assert exe._entry_threshold() < 0.25

    def test_negative_ema_raises(self):
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True)
        exe._signal_threshold = 0.25; exe._pnl_ema = -0.03
        assert exe._entry_threshold() > 0.25


# ══════════════════════════════════════════════════════════════════════════════
# Position sizing
# ══════════════════════════════════════════════════════════════════════════════


class TestPositionSizing:
    def test_contract_sizing_min_one(self):
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True, initial_capital=500, max_position_pct=0.50, leverage=2.0, ct_val=0.1)
        exe._equity = [500]
        assert exe._compute_size(0.3, 2500.0) == 1

    def test_small_notional_returns_zero(self):
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True, initial_capital=1, max_position_pct=0.01, leverage=1.0, ct_val=1.0)
        exe._equity = [1]
        assert exe._compute_size(0.5, 2500.0) == 0

    def test_drawdown_halves_leverage(self):
        exe = LiveExecutor(symbol="ETH-USDT-SWAP", dry_run=True, initial_capital=10000, max_position_pct=0.5, leverage=2.0, ct_val=1.0)
        exe._equity = [10000, 8000]  # 20% drawdown from peak
        sz_normal = exe._compute_size(0.5, 100.0)
        exe._equity = [10000]
        sz_full = exe._compute_size(0.5, 100.0)
        assert sz_normal < sz_full, "Drawdown should reduce position size"
