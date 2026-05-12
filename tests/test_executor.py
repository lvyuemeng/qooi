"""Stateless executor tests — full state machine + normalization coverage.

Covers:
  _reconstruct()   — IDLE PENDING ACTIVE partial-fill algo-recon
  _decide()        — idle idle_sell pending pending_fill pending_timeout
                     pending_flip pending_chase pending_outstanding
                     active_flip active_time active_holding active_trail
                     active_notrail
  _execute()       — enter exit amend (dry-run only)
  normalization    — stop/target ATR scaling, size calc, trail math
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from qooi.exchange.backtest import RiskConfig
from qooi.exchange.trading import (
    AssetConfig,
    Decision,
    ExchangeSnapshot,
    PositionState,
    ReconstructedState,
    SignalResult,
    State,
    StatelessExecutor,
)

# ── helpers ──────────────────────────────────────────────────────────────────


@dataclass
class _Obi:
    ask_price: float = 2500.0
    bid_price: float = 2498.0


def _obi(ask=2500.0, bid=2498.0):
    return _Obi(ask_price=ask, bid_price=bid)


def _sig(**kw) -> SignalResult:
    d = dict(
        symbol="ETH-USDT-SWAP",
        timeframe="4h",
        timestamp=int(time.time()),
        signal=0.5,
        flow=0.5,
        threshold=0.25,
        atr=50.0,
    )
    d.update(kw)
    return SignalResult(**d)


def _cfg(**kw) -> AssetConfig:
    d = dict(
        symbol="ETH-USDT-SWAP",
        sig_symbol="ETH-USDT",
        timeframe="4h",
        capital=500,
        max_risk_pct=0.50,
        leverage=2.0,
        ct_val=0.1,
        atr_stop_mult=2.0,
        atr_target_mult=3.0,
        trail_activation_mult=2.0,
        trail_distance_mult=1.0,
        signal_threshold=0.25,
        ord_type="post_only",
        td_mode="isolated",
    )
    d.update(kw)
    return AssetConfig(**d)


def _snap(**kw) -> ExchangeSnapshot:
    d = dict(orders=[], positions=[], algo_orders=[], usdt_balance=10000.0, usdt_frozen=0.0)
    d.update(kw)
    return ExchangeSnapshot(**d)


def _rs(state: State = State.IDLE, **kw) -> ReconstructedState:
    d = dict(state=state, symbol="ETH-USDT-SWAP", signal=0.5, flow=0.5, atr_estimate=50.0)
    d.update(kw)
    return ReconstructedState(**d)


def _rs_pending(**kw):
    d: dict = dict(
        state=State.PENDING,
        ord_id="123",
        cl_ord_id="abc",
        side="buy",
        sz="10",
        acc_fill_sz="0",
        ord_px="2500",
        ord_state="live",
        ord_ctime=str(int(time.time() * 1000) - 60000),
        age_sec=60.0,
    )
    d.update(kw)
    return _rs(**d)


def _rs_active(**kw):
    d: dict = dict(
        state=State.ACTIVE,
        side="buy",
        sz="10",
        pos_id="p1",
        pos_side="long",
        pos_sz="10",
        avg_px="2500",
        mark_px="2550",
        upl="50",
        margin="500",
        sl_trigger_px="2400",
        sl_ord_px="2400",
        tp_trigger_px="2700",
        tp_ord_px="2700",
    )
    d.update(kw)
    return _rs(**d)


# ── state reconstruction ─────────────────────────────────────────────────────


class TestReconstruct:
    def test_empty_snap_is_idle(self):
        s = StatelessExecutor(_cfg())._reconstruct(_snap(), _sig(), _obi())
        assert s.state == State.IDLE
        assert s.signal == 0.5

    def test_pending_order_reconstructed(self):
        now = int(time.time() * 1000)
        snap = _snap(
            orders=[
                {
                    "instId": "ETH-USDT-SWAP",
                    "ordId": "123",
                    "side": "buy",
                    "sz": "10",
                    "px": "2500",
                    "state": "live",
                    "cTime": str(now - 120000),
                    "accFillSz": "0",
                }
            ]
        )
        s = StatelessExecutor(_cfg())._reconstruct(snap, _sig(), _obi())
        assert s.state == State.PENDING
        assert s.ord_id == "123"
        assert s.age_sec == 120.0

    def test_filled_order_with_position_is_active(self):
        snap = _snap(
            orders=[
                {
                    "instId": "ETH-USDT-SWAP",
                    "ordId": "123",
                    "side": "buy",
                    "sz": "10",
                    "px": "2500",
                    "state": "filled",
                    "cTime": "1000",
                    "accFillSz": "10",
                }
            ],
            positions=[
                {
                    "instId": "ETH-USDT-SWAP",
                    "posId": "p1",
                    "posSide": "long",
                    "pos": "10",
                    "avgPx": "2500",
                    "markPx": "2550",
                    "upl": "50",
                }
            ],
        )
        s = StatelessExecutor(_cfg())._reconstruct(snap, _sig(), _obi())
        assert s.state == State.ACTIVE
        assert s.pos_id == "p1"

    def test_position_without_order_is_active(self):
        snap = _snap(
            positions=[
                {
                    "instId": "ETH-USDT-SWAP",
                    "posId": "p1",
                    "posSide": "long",
                    "pos": "10",
                    "avgPx": "2500",
                    "markPx": "2550",
                    "upl": "50",
                    "cTime": str(int(time.time() * 1000) - 7200000),
                }
            ]
        )
        s = StatelessExecutor(_cfg())._reconstruct(snap, _sig(), _obi())
        assert s.state == State.ACTIVE
        assert s.side == "buy"
        assert s.age_sec == 7200.0

    def test_partially_filled_stays_pending(self):
        snap = _snap(
            orders=[
                {
                    "instId": "ETH-USDT-SWAP",
                    "ordId": "123",
                    "side": "buy",
                    "sz": "10",
                    "px": "2500",
                    "state": "partially_filled",
                    "cTime": "1000",
                    "accFillSz": "3",
                }
            ]
        )
        s = StatelessExecutor(_cfg())._reconstruct(snap, _sig(), _obi())
        assert s.state == State.PENDING

    def test_algo_sl_tp_picked_up(self):
        snap = _snap(
            algo_orders=[
                {
                    "instId": "ETH-USDT-SWAP",
                    "algoId": "a1",
                    "slTriggerPx": "2400",
                    "slOrdPx": "2400",
                    "tpTriggerPx": "2700",
                    "tpOrdPx": "2700",
                }
            ]
        )
        s = StatelessExecutor(_cfg())._reconstruct(snap, _sig(), _obi())
        assert s.algo_sl_id == "a1"
        assert s.sl_trigger_px == "2400"
        assert s.algo_tp_id == "a1"
        assert s.tp_trigger_px == "2700"


# ── normalization ────────────────────────────────────────────────────────────


class TestNormalization:
    def test_stop_distance_scales_with_atr(self):
        exe = StatelessExecutor(_cfg(atr_stop_mult=2.0))
        s_lo = _rs(state=State.IDLE, atr_estimate=25.0, signal=0.5)
        s_hi = _rs(state=State.IDLE, atr_estimate=100.0, signal=0.5)
        d_lo = exe._decide(s_lo, _obi(ask=2500, bid=2498))
        d_hi = exe._decide(s_hi, _obi(ask=2500, bid=2498))
        # entry_px = ask * 0.9995 with skew; regime_strength=0 -> 1.25x stop_mult
        assert d_lo.stop_px == round(2500 * 0.9995 - 2.0 * 1.25 * 25.0, 2)
        assert d_hi.stop_px == round(2500 * 0.9995 - 2.0 * 1.25 * 100.0, 2)

    def test_target_distance_scales_with_atr(self):
        exe = StatelessExecutor(_cfg(atr_target_mult=3.0))
        s = _rs(state=State.IDLE, atr_estimate=50.0, signal=0.5)
        d = exe._decide(s, _obi(ask=2500, bid=2498))
        # regime_strength=0 -> 0.6x target_mult
        assert d.target_px == round(2500 * 0.9995 + 3.0 * 0.6 * 50.0, 2)

    def test_size_derived_from_risk_and_stop_distance(self):
        exe = StatelessExecutor(
            _cfg(capital=1000, max_risk_pct=0.20, ct_val=0.1, atr_stop_mult=2.0)
        )
        s = _rs(state=State.IDLE, atr_estimate=50.0, signal=0.5)
        d = exe._decide(s, _obi(ask=2500, bid=2498))
        risk_per_ct = abs(2500 - (2500 - 100)) * 0.1
        risk_sz = max(1, int(200 / risk_per_ct))  # 20
        margin_sz = int((1000 * 2.0) / (0.1 * 2500))  # 8
        assert d.sz == min(risk_sz, margin_sz)

    def test_trail_activation_requires_profit_atr(self):
        exe = StatelessExecutor(_cfg(trail_activation_mult=2.0, trail_distance_mult=1.0))
        s = _rs_active(mark_px="2580", sl_trigger_px="2400", avg_px="2500", atr_estimate=50.0)
        d = exe._decide(s, _obi(ask=2580, bid=2578))
        assert d.detail == "holding"

    def test_trail_triggers_when_profit_exceeds_activation(self):
        exe = StatelessExecutor(_cfg(trail_activation_mult=2.0, trail_distance_mult=1.0))
        s = _rs_active(mark_px="2650", sl_trigger_px="2400", avg_px="2500", atr_estimate=50.0)
        d = exe._decide(s, _obi(ask=2650, bid=2648))
        assert d.action == "amend"
        assert d.detail == "trail_update"
        assert d.amend_sl_trigger_px == "2600.0"

    def test_trail_does_not_loosen_stop(self):
        exe = StatelessExecutor(_cfg(trail_activation_mult=2.0, trail_distance_mult=1.0))
        s = _rs_active(
            mark_px="2600",
            sl_trigger_px="2450",
            avg_px="2500",
            atr_estimate=50.0,
            side="sell",
            pos_side="short",
            signal=-0.5,
        )
        d = exe._decide(s, _obi(ask=2602, bid=2600))
        assert d.detail == "holding"


# ── decision dispatch ────────────────────────────────────────────────────────


class TestDecide:
    def test_idle_skips_weak_signal(self):
        d = StatelessExecutor(_cfg(signal_threshold=0.25))._decide(
            _rs(State.IDLE, signal=0.1), _obi()
        )
        assert d.action == "skip"
        assert d.detail == "weak_signal"

    def test_idle_enters_long_on_strong_signal(self):
        d = StatelessExecutor(_cfg(signal_threshold=0.25))._decide(
            _rs(State.IDLE, signal=0.5, atr_estimate=50.0), _obi(ask=2500, bid=2498)
        )
        assert d.action == "enter"
        assert d.side == "buy"
        assert d.entry_px == round(2500 * 0.9995, 2)
        assert d.stop_px == round(d.entry_px - 2.0 * 1.25 * 50.0, 2)
        assert d.target_px == round(d.entry_px + 3.0 * 0.6 * 50.0, 2)

    def test_idle_enters_short_on_negative_signal(self):
        d = StatelessExecutor(_cfg(signal_threshold=0.25))._decide(
            _rs(State.IDLE, signal=-0.5, atr_estimate=50.0), _obi(ask=2500, bid=2498)
        )
        assert d.action == "enter"
        assert d.side == "sell"
        assert d.entry_px == round(2498 * 1.0005, 2)
        assert d.stop_px == round(d.entry_px + 2.0 * 1.25 * 50.0, 2)

    def test_pending_filled_transitions_to_active(self):
        s = _rs_pending(acc_fill_sz="10", ord_state="filled")
        s.state = State.PENDING
        d = StatelessExecutor(_cfg())._decide(s, _obi())
        assert d.action == "skip"
        assert d.detail == "order_filled"
        assert d.new_state == State.ACTIVE

    def test_pending_timeout_exits(self):
        d = StatelessExecutor(_cfg(limit_timeout_sec=10))._decide(
            _rs_pending(age_sec=100.0), _obi()
        )
        assert d.action == "exit"
        assert d.detail == "timeout"

    def test_pending_signal_flip_exits(self):
        d = StatelessExecutor(_cfg())._decide(_rs_pending(signal=-0.5, side="buy"), _obi())
        assert d.action == "exit"
        assert d.detail == "signal_flipped"

    def test_pending_outstanding_holds(self):
        d = StatelessExecutor(_cfg(limit_timeout_sec=99999))._decide(
            _rs_pending(ord_px="2500", signal=0.5, side="buy", age_sec=10), _obi(ask=2502, bid=2500)
        )
        assert d.action == "skip"

    def test_pending_price_chases(self):
        d = StatelessExecutor(_cfg(limit_timeout_sec=99999))._decide(
            _rs_pending(ord_px="2500", signal=0.5, side="buy", age_sec=10), _obi(ask=2530, bid=2528)
        )
        assert d.action == "amend"
        assert d.detail == "price_chase"

    def test_active_signal_flip_exits(self):
        d = StatelessExecutor(_cfg())._decide(
            _rs_active(signal=-0.5, side="buy", pos_side="long"), _obi()
        )
        assert d.action == "exit"

    def test_active_max_bars_exits(self):
        d = StatelessExecutor(_cfg(max_bars_held=5))._decide(
            _rs_active(mark_px="2550", bars_held=6), _obi()
        )
        assert d.action == "exit"
        assert d.detail == "time"


# ── execution (dry-run) ──────────────────────────────────────────────────────


class TestExecute:
    def test_enter_no_client_is_noop(self):
        d = Decision.enter("buy", 10, 2500, stop_px=2400, target_px=2700)
        StatelessExecutor(_cfg())._execute(d, _rs(State.IDLE), _obi(), None)

    def test_exit_no_client_is_noop(self):
        d = Decision.exit("signal_flipped")
        StatelessExecutor(_cfg())._execute(d, _rs_pending(ord_id="123"), _obi(), None)


# ── PositionState backtest-compat ────────────────────────────────────────────


class TestPositionState:
    def test_enter_long_stop_target(self):
        risk = RiskConfig(atr_stop_mult=2.0, atr_target_mult=3.0)
        pos = PositionState.enter_long(2500, 50, risk, int(time.time() * 1000))
        assert pos.stop_price == 2400.0
        assert pos.target_price == 2650.0

    def test_enter_short_stop_target(self):
        risk = RiskConfig(atr_stop_mult=2.0, atr_target_mult=3.0)
        pos = PositionState.enter_short(2500, 50, risk, int(time.time() * 1000))
        assert pos.stop_price == 2600.0
        assert pos.target_price == 2350.0

    def test_check_exit_stop_target_trailing(self):
        risk = RiskConfig(atr_stop_mult=2.0, atr_target_mult=3.0, trailing_distance_mult=1.0)
        pos = PositionState.enter_long(2500, 50, risk, 0)
        pos.order.side = "buy"
        assert pos.check_exit(2390, 50, risk) == "stop"
        assert pos.check_exit(2650, 50, risk) == "target"
        pos.trail_high = 2600.0
        assert pos.check_exit(2540, 50, risk) == "trailing_stop"


# ── ExchangeSnapshot query ───────────────────────────────────────────────────


class TestExchangeSnapshot:
    def test_snapshot_queries_by_symbol(self):
        snap = _snap(
            orders=[{"instId": "ETH-USDT-SWAP", "ordId": "123"}],
            positions=[{"instId": "ETH-USDT-SWAP", "posId": "p1"}],
            algo_orders=[{"instId": "ETH-USDT-SWAP", "algoId": "a1"}],
        )
        assert snap.order_for_symbol("ETH-USDT-SWAP")["ordId"] == "123"
        assert snap.position_for_symbol("ETH-USDT-SWAP")["posId"] == "p1"
        assert snap.algos_for_symbol("ETH-USDT-SWAP")[0]["algoId"] == "a1"
        assert snap.order_for_symbol("BTC-USDT-SWAP") is None
