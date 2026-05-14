"""Recovery — grid, martingale, and hedge strategies for drawdown recovery.

Produces list[BasketAction] for grid adds, direction reversals, and hedges.
Activated when a basket is in a losing state beyond thresholds.

Martingale reversal produces TWO actions: EXIT (close original) + ENTER
(open opposite with computed size from loss).  This maps to OKX
close-position + place order.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from qooi.core.basket import ActionKind, Basket, BasketAction, ExitReason


class RecoveryKind(StrEnum):
    NONE = "none"
    GRID = "grid"
    MARTINGALE = "martingale"
    HEDGE = "hedge"


@dataclass
class RecoveryConfig:
    strategy: RecoveryKind = RecoveryKind.NONE
    zone_atr: float = 2.0
    multiplier: float = 2.0
    max_levels: int = 3
    max_loss_pct: float = 10.0
    breakeven_atr: float = 1.0


def evaluate(
    basket: Basket,
    bar_close: float,
    atr: float,
    config: RecoveryConfig,
    current_level: int,
) -> list[BasketAction]:
    """Evaluate recovery logic. Returns list of actions (may be empty)."""

    if config.strategy == RecoveryKind.NONE:
        return []

    if not basket.is_active:
        return []

    d = 1 if basket.side == "buy" else -1
    loss_pct = d * (bar_close / basket.entry_px - 1) * 100 if basket.entry_px > 0 else 0

    if config.strategy == RecoveryKind.GRID:
        return _grid(basket, bar_close, atr, config, current_level, loss_pct, d)
    if config.strategy == RecoveryKind.MARTINGALE:
        return _martingale(basket, bar_close, atr, config, current_level, loss_pct, d)
    if config.strategy == RecoveryKind.HEDGE:
        return _hedge(basket, bar_close, config, loss_pct)

    return []


def _grid(
    basket: Basket,
    bar_close: float,
    atr: float,
    config: RecoveryConfig,
    level: int,
    loss_pct: float,
    d: int,
) -> list[BasketAction]:
    if level >= config.max_levels:
        return []
    if basket.current_sz <= 0:
        return []

    target_px = basket.entry_px - d * config.zone_atr * atr * (level + 1)
    if d * (bar_close - target_px) <= 0:
        return [
            BasketAction(
                basket_id=basket.basket_id,
                action=ActionKind.ADD_GRID,
                side=basket.side,
                sz=basket.current_sz * config.multiplier,
                px=bar_close,
                reason=f"grid_level_{level + 1}",
            )
        ]
    return []


def _martingale(
    basket: Basket,
    bar_close: float,
    atr: float,
    config: RecoveryConfig,
    level: int,
    loss_pct: float,
    d: int,
) -> list[BasketAction]:
    if level >= config.max_levels or loss_pct > -config.zone_atr:
        return []

    reversal_side = "sell" if basket.side == "buy" else "buy"
    reversal_sz = _reversal_size(basket, bar_close, atr, config)

    return [
        BasketAction(
            basket_id=basket.basket_id,
            action=ActionKind.EXIT,
            side=basket.side,
            reason=ExitReason.MARTINGALE.value,
            fraction=1.0,
        ),
        BasketAction(
            basket_id=basket.basket_id + "_reversal",
            action=ActionKind.ENTER,
            side=reversal_side,
            sz=max(1, reversal_sz),
            px=bar_close,
            reason=ExitReason.MARTINGALE.value,
        ),
    ]


def _hedge(
    basket: Basket,
    bar_close: float,
    config: RecoveryConfig,
    loss_pct: float,
) -> list[BasketAction]:
    if loss_pct < -config.zone_atr:
        hedge_side = "sell" if basket.side == "buy" else "buy"
        return [
            BasketAction(
                basket_id=basket.basket_id,
                action=ActionKind.HEDGE,
                side=hedge_side,
                sz=basket.current_sz,
                px=bar_close,
                reason=ExitReason.HEDGE_DRAWDOWN.value,
            )
        ]
    return []


def _reversal_size(basket: Basket, bar_close: float, atr: float, config: RecoveryConfig) -> int:
    loss_amt = abs(basket.entry_px - bar_close) * basket.current_sz * basket.entry_px
    zone_value = config.zone_atr * atr
    if zone_value <= 0 or basket.entry_px <= 0:
        return 1
    sz = int(loss_amt / zone_value / basket.entry_px)
    return max(1, sz)
