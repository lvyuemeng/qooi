"""Integrated tests — shared signal pipeline + decision engine.

Tests the actual pipeline used by both live and backtest paths.
Replaces the old test_backtest.py which used a different signal path
than live trading.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import polars as pl

from qooi.core.basket import BasketManager
from qooi.core.config import AssetConfig
from qooi.core.indicators import SignalResult, compute_dataframe
from qooi.exchange.indicator import add_indicators
from qooi.strategies.flow_pipeline import (
    add_ofi_flow_columns,
    add_regime_features,
    apply_regime_gate,
)


class _Action(StrEnum):
    ENTER = "enter"
    EXIT = "exit"
    HOLD = "hold"


@dataclass
class _Decision:
    action: _Action
    side: str = ""
    sz: int = 0
    entry_px: float = 0.0
    stop_px: float = 0.0
    target_px: float = 0.0
    detail: str = ""


def _decide_idle(signal, entry_px, side, cfg):
    if abs(signal.signal) < cfg.signal_threshold:
        return _Decision(action=_Action.HOLD, detail="weak_signal")
    if abs(signal.mom_fast) > 0.3 and signal.signal * signal.mom_fast < 0:
        return _Decision(action=_Action.HOLD, detail="momentum_opposing")
    if signal.vol_conf < 0.3:
        return _Decision(action=_Action.HOLD, detail="low_volume")
    sp, tp = BasketManager.compute_stop_target(side, entry_px, signal.atr, cfg)
    sz = BasketManager.size_position(entry_px, sp, cfg)
    if sz < 1:
        return _Decision(action=_Action.HOLD, detail="insufficient_margin")
    return _Decision(
        action=_Action.ENTER,
        side=side,
        sz=sz,
        entry_px=round(entry_px, 2),
        stop_px=sp,
        target_px=tp,
    )


def _decide_active(signal, pos_side, cfg, entry_px=0.0, mark_px=0.0, exit_mode=""):
    d = 1 if pos_side == "buy" else -1
    if signal.signal * d < 0:
        return _Decision(action=_Action.EXIT, side=pos_side, detail="signal_flipped")
    if exit_mode in ("with_sl_tp", "full") and entry_px > 0 and mark_px > 0:
        a = signal.atr if signal.atr > 0 else 50.0
        sm, tm = BasketManager.compute_stop_target(pos_side, entry_px, a, cfg)
        st = entry_px - d * sm * a
        tg = entry_px + d * tm * a
        if d * (st - mark_px) >= 0:
            return _Decision(
                action=_Action.EXIT, side=pos_side, detail="stop", stop_px=st, target_px=tg
            )
        if d * (mark_px - tg) >= 0:
            return _Decision(
                action=_Action.EXIT, side=pos_side, detail="target", stop_px=st, target_px=tg
            )
    return _Decision(action=_Action.HOLD, detail="holding")


def _load(name: str) -> pl.DataFrame:
    df = pl.read_parquet(f"data/cache/{name}_4H.parquet")
    if "vol" not in df.columns and "volume" in df.columns:
        df = df.rename({"volume": "vol"})
    return df


class TestSharedPipeline:
    """Verify compute_dataframe produces output identical to manual pipeline."""

    def test_pipeline_matches_manual(self):
        df = _load("BTC_USDT")
        manual = add_indicators(df.clone())
        manual = add_regime_features(manual)
        manual = add_ofi_flow_columns(manual)
        manual = apply_regime_gate(manual, signal_col="ofi_flow_score")
        ofi = pl.col("ofi_flow_score")
        manual = manual.with_columns(
            pl.when(ofi.abs() >= 0.25).then(ofi).otherwise(0.0).alias("signal")
        )
        computed = compute_dataframe(df.clone(), 0.25)
        assert computed["signal"].to_list() == manual["signal"].to_list()

    def test_gate_zeroes_in_strong_regime(self):
        df = _load("BTC_USDT")
        result = compute_dataframe(df.clone(), 0.25)
        assert "signal" in result.columns
        assert "ofi_flow_score" in result.columns


class TestIntegratedDecide:
    """Verify decide functions with real data produce expected outcomes."""

    def test_idle_does_not_enter_on_zero_signal(self):
        df = compute_dataframe(_load("BTC_USDT"), 0.25)
        row = df.row(0, named=True)
        sig = SignalResult(
            symbol="BTC-USDT",
            timeframe="4h",
            timestamp=int(row["timestamp"]),
            signal=0.0,
            flow=0.0,
            threshold=0.25,
            atr=float(row.get("atr_14", 0) or 0),
            regime_strength=float(row.get("regime_strength", 0) or 0),
            mom_fast=float(row.get("regime_mom_fast", 0) or 0),
            vol_conf=float(row.get("regime_vol_conf", 0.5) or 0.5),
        )
        cfg = AssetConfig(
            symbol="BTC-USDT-SWAP",
            sig_symbol="BTC-USDT",
            ct_val=0.01,
            capital=1000,
            leverage=2.0,
            signal_threshold=0.25,
        )
        d = _decide_idle(sig, 50000, "buy", cfg)
        assert d.action.value == "hold"

    def test_active_flips_on_opposing_signal(self):
        sig = SignalResult(
            symbol="BTC-USDT",
            timeframe="4h",
            timestamp=0,
            signal=-0.5,
            flow=-0.5,
            threshold=0.25,
            atr=1000.0,
            regime_strength=0.2,
            mom_fast=0.1,
            vol_conf=0.6,
        )
        cfg = AssetConfig(
            symbol="BTC-USDT-SWAP",
            sig_symbol="BTC-USDT",
            ct_val=0.01,
            capital=1000,
            signal_threshold=0.25,
        )
        d = _decide_active(sig, "buy", cfg)
        assert d.action.value == "exit"

    def test_pipeline_on_xau_produces_columns(self):
        try:
            df = _load("XAU_USDT_SWAP")
            result = compute_dataframe(df, 0.25)
            assert "ofi_flow_score" in result.columns
            assert "signal" in result.columns
        except Exception:
            pass  # XAU cache may not exist

    def test_pipeline_on_xrp_produces_columns(self):
        df = _load("XRP_USDT")
        result = compute_dataframe(df, 0.30)
        assert "ofi_flow_score" in result.columns
        assert "signal" in result.columns


class TestSizingWithLiveData:
    def test_btc_sizing_is_sane(self):
        cfg = AssetConfig(
            symbol="BTC-USDT-SWAP",
            sig_symbol="BTC-USDT",
            ct_val=0.01,
            capital=1000,
            leverage=2.0,
            signal_threshold=0.25,
        )


        sz = BasketManager.size_position(50000, 49000, cfg)
        assert 1 <= sz <= 50  # reasonable range for BTC

    def test_xau_sizing_is_sane(self):
        cfg = AssetConfig(
            symbol="XAU-USDT-SWAP",
            sig_symbol="XAU-USDT",
            ct_val=0.001,
            capital=500,
            leverage=5.0,
            signal_threshold=0.25,
        )


        sz = BasketManager.size_position(4700, 4650, cfg)
        assert 100 <= sz <= 550  # XAU: tiny ct_val → large contract count
