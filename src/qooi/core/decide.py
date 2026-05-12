"""Shared decision engine — identical for live trading and backtesting.

Both paths call the same functions: what changes is only the I/O layer
(fetching prices vs reading cached data; calling OKX vs simulating fills).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from qooi.core.signal import SignalResult


class Action(StrEnum):
    ENTER = "enter"
    EXIT = "exit"
    HOLD = "hold"


@dataclass
class Decision:
    action: Action
    side: str = ""
    sz: int = 0
    entry_px: float = 0.0
    stop_px: float = 0.0
    target_px: float = 0.0
    detail: str = ""


@dataclass
class AssetConfig:
    symbol: str
    sig_symbol: str = ""
    timeframe: str = "4h"
    capital: float = 500.0
    max_risk_pct: float = 0.50
    leverage: float = 2.0
    ct_val: float = 0.1
    atr_stop_mult: float = 2.0
    atr_target_mult: float = 3.0
    signal_threshold: float = 0.25
    ord_type: str = "limit"


def compute_stop_target(
    side: str,
    entry_px: float,
    atr: float,
    cfg: AssetConfig,
    regime_strength: float = 0.0,
) -> tuple[float, float]:
    d = 1 if side == "buy" else -1
    if regime_strength > 0.7:
        stop_mult = cfg.atr_stop_mult * 0.5
        target_mult = cfg.atr_target_mult * 0.8
    elif regime_strength > 0.3:
        stop_mult = cfg.atr_stop_mult * 0.75
        target_mult = cfg.atr_target_mult * 1.2
    else:
        stop_mult = cfg.atr_stop_mult * 1.25
        target_mult = cfg.atr_target_mult * 0.6
    return (
        round(entry_px - d * stop_mult * atr, 2),
        round(entry_px + d * target_mult * atr, 2),
    )


def compute_sz(
    entry_px: float,
    stop_px: float,
    cfg: AssetConfig,
) -> int:
    risk_per_ct = abs(entry_px - stop_px) * cfg.ct_val
    if risk_per_ct <= 0:
        return 0
    max_risk = cfg.capital * cfg.max_risk_pct
    sz = max(1, int(max_risk / risk_per_ct))
    notional_per_ct = cfg.ct_val * entry_px
    max_sz = int(cfg.capital * cfg.leverage / max(notional_per_ct, 1e-9))
    return max(1, min(sz, max_sz))


def decide_idle(
    signal: SignalResult,
    entry_px: float,
    side: str,
    cfg: AssetConfig,
) -> Decision:
    if abs(signal.signal) < cfg.signal_threshold:
        return Decision(action=Action.HOLD, detail="weak_signal")
    if abs(signal.mom_fast) > 0.3 and signal.signal * signal.mom_fast < 0:
        return Decision(action=Action.HOLD, detail="momentum_opposing")
    if signal.vol_conf < 0.3:
        return Decision(action=Action.HOLD, detail="low_volume")

    stop_px, target_px = compute_stop_target(
        side, entry_px, signal.atr, cfg, signal.regime_strength
    )
    sz = compute_sz(entry_px, stop_px, cfg)
    if sz < 1:
        return Decision(action=Action.HOLD, detail="insufficient_margin")

    return Decision(
        action=Action.ENTER,
        side=side,
        sz=sz,
        entry_px=round(entry_px, 2),
        stop_px=stop_px,
        target_px=target_px,
    )


def decide_active(
    signal: SignalResult,
    pos_side: str,
    cfg: AssetConfig,
) -> Decision:
    d = 1 if pos_side == "buy" else -1
    if signal.signal * d < 0:
        return Decision(action=Action.EXIT, side=pos_side, detail="signal_flipped")
    return Decision(action=Action.HOLD, detail="holding")
