"""Recovery — grid, martingale, and hedge strategies for drawdown recovery.

Produces BasketActions for grid adds, direction reversals, and hedges.
Activated when a basket is in a losing state beyond thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass

from qooi.core.basket import Basket, BasketAction


@dataclass
class RecoveryConfig:
    strategy: str = "none"
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
) -> BasketAction | None:
    """Evaluate recovery logic for a basket. Returns an action or None."""

    if config.strategy == "none" or config.strategy == "":
        return None

    if not basket.is_active:
        return None

    d = 1 if basket.side == "buy" else -1
    loss_pct = d * (bar_close / basket.entry_px - 1) * 100 if basket.entry_px > 0 else 0

    if config.strategy == "grid":
        return _grid(basket, bar_close, atr, config, current_level, loss_pct, d)
    if config.strategy == "martingale":
        return _martingale(basket, bar_close, config, current_level, loss_pct, d)
    if config.strategy == "hedge":
        return _hedge(basket, bar_close, config, loss_pct)

    return None


def _grid(
    basket: Basket,
    bar_close: float,
    atr: float,
    config: RecoveryConfig,
    level: int,
    loss_pct: float,
    d: int,
) -> BasketAction | None:
    if level >= config.max_levels:
        return None

    target_px = basket.entry_px - d * config.zone_atr * atr * (level + 1)
    if d * (bar_close - target_px) <= 0:
        return BasketAction(
            basket_id=basket.basket_id,
            action="add_grid",
            side=basket.side,
            sz=basket.current_sz * config.multiplier,
            px=bar_close,
            reason=f"grid_level_{level + 1}",
        )
    return None


def _martingale(
    basket: Basket,
    bar_close: float,
    config: RecoveryConfig,
    level: int,
    loss_pct: float,
    d: int,
) -> BasketAction | None:
    if level >= config.max_levels:
        return None

    if level >= config.max_levels or loss_pct > -config.zone_atr:
        return None

    return BasketAction(
        basket_id=basket.basket_id,
        action="exit",
        side=basket.side,
        reason="martingale_reverse",
        fraction=1.0,
    )


def _hedge(
    basket: Basket,
    bar_close: float,
    config: RecoveryConfig,
    loss_pct: float,
) -> BasketAction | None:
    if loss_pct < -config.zone_atr:
        hedge_side = "sell" if basket.side == "buy" else "buy"
        return BasketAction(
            basket_id=basket.basket_id,
            action="hedge",
            side=hedge_side,
            sz=basket.current_sz,
            px=bar_close,
            reason="hedge_on_drawdown",
        )
    return None
