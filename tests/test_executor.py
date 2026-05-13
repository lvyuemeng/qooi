"""Decision engine tests — shared between live and backtest via qooi.core."""

from __future__ import annotations

from qooi.core.decide import (
    AssetConfig,
    compute_stop_target,
    compute_sz,
    decide_active,
    decide_idle,
)
from qooi.core.indicators import SignalResult


def _sig(**kw) -> SignalResult:
    d = dict(
        symbol="ETH-USDT-SWAP",
        timeframe="4h",
        timestamp=0,
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
        leverage=2.0,
        ct_val=0.1,
        signal_threshold=0.25,
    )
    d.update(kw)
    return AssetConfig(**d)


class TestDecideIdle:
    def test_weak_signal_holds(self):
        d = decide_idle(_sig(signal=0.1), entry_px=2500, side="buy", cfg=_cfg())
        assert d.action.value == "hold"

    def test_strong_buy_enters(self):
        d = decide_idle(_sig(signal=0.5, atr=50.0), entry_px=2500, side="buy", cfg=_cfg())
        assert d.action.value == "enter"
        assert d.side == "buy"
        assert d.sz > 0

    def test_strong_sell_enters(self):
        d = decide_idle(_sig(signal=-0.5, atr=50.0), entry_px=2498, side="sell", cfg=_cfg())
        assert d.action.value == "enter"
        assert d.side == "sell"

    def test_momentum_opposing_holds(self):
        d = decide_idle(_sig(signal=0.5, mom_fast=-0.5), entry_px=2500, side="buy", cfg=_cfg())
        assert d.detail == "momentum_opposing"

    def test_low_volume_holds(self):
        d = decide_idle(_sig(signal=0.5, vol_conf=0.2), entry_px=2500, side="buy", cfg=_cfg())
        assert d.detail == "low_volume"


class TestDecideActive:
    def test_signal_flip_exits(self):
        d = decide_active(_sig(signal=-0.5), pos_side="buy", cfg=_cfg())
        assert d.action.value == "exit"

    def test_same_direction_holds(self):
        d = decide_active(_sig(signal=0.5), pos_side="buy", cfg=_cfg())
        assert d.action.value == "hold"


class TestSizing:
    def test_stop_target_scales_with_atr(self):
        sl, tp = compute_stop_target("buy", 2500, 50, _cfg())
        assert sl == round(2500 - 2.0 * 1.25 * 50, 2)
        assert tp == round(2500 + 3.0 * 0.6 * 50, 2)

    def test_stop_target_regime_strong_trend(self):
        sl, tp = compute_stop_target("buy", 2500, 50, _cfg(), regime_strength=0.8)
        assert sl == round(2500 - 2.0 * 0.5 * 50, 2)
        assert tp == round(2500 + 3.0 * 0.8 * 50, 2)

    def test_size_respects_margin_cap(self):
        sz = compute_sz(2500, 2400, _cfg())
        assert sz > 0
        max_notional = 500 * 2.0
        actual_notional = sz * 0.1 * 2500
        assert actual_notional <= max_notional + 1
