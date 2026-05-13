"""Exit engine — stop, target, trail, time, funding exits in priority order.

Produces BasketActions for the executor.  Used identically by backtest
and live trading.
"""

from __future__ import annotations

from dataclasses import dataclass

from qooi.core.basket import Basket, BasketAction


@dataclass
class ExitConfig:
    stop_mult: float = 1.5
    target_mult: float = 1.3
    trail_mult: float = 2.0
    max_bars: int = 10
    breakeven_after_target: bool = False
    session_end_utc: int = 22


@dataclass
class TrailTracker:
    trail_high: float = 0.0
    trail_low: float = float("inf")
    target_hit: bool = False

    def update(self, high: float, low: float) -> None:
        if high > self.trail_high:
            self.trail_high = high
        if low < self.trail_low:
            self.trail_low = low


def evaluate(
    basket: Basket,
    bar_close: float,
    bar_high: float,
    bar_low: float,
    atr: float,
    trail: TrailTracker,
    config: ExitConfig,
) -> BasketAction | None:
    """Check all exit conditions, return action or None."""

    bars = basket.bars_in_pos
    entry = basket.entry_px
    d = 1 if basket.side == "buy" else -1

    trail.update(bar_high, bar_low)

    stop_px = entry - d * config.stop_mult * atr
    target_px = entry + d * config.target_mult * atr

    # Level 1: hard stop (always active until target hit)
    if not trail.target_hit:
        if d * (stop_px - bar_close) >= 0:
            return BasketAction(
                basket_id=basket.basket_id,
                action="exit",
                side=basket.side,
                reason="stop",
                px=bar_close,
                fraction=1.0,
            )

    # Level 2: trailing stop (only after target hit)
    if trail.target_hit:
        trail_stop = (
            trail.trail_high - config.trail_mult * atr
            if d > 0
            else trail.trail_low + config.trail_mult * atr
        )
        if d > 0 and bar_close <= trail_stop:
            return BasketAction(
                basket_id=basket.basket_id,
                action="exit",
                side=basket.side,
                reason="trailing_stop",
                px=bar_close,
                fraction=1.0,
            )
        elif d < 0 and bar_close >= trail_stop:
            return BasketAction(
                basket_id=basket.basket_id,
                action="exit",
                side=basket.side,
                reason="trailing_stop",
                px=bar_close,
                fraction=1.0,
            )

    # Level 3: breakeven stop (after target, move stop to entry)
    if config.breakeven_after_target and trail.target_hit:
        if d > 0 and bar_close <= entry:
            return BasketAction(
                basket_id=basket.basket_id,
                action="exit",
                side=basket.side,
                reason="breakeven",
                px=bar_close,
                fraction=1.0,
            )
        elif d < 0 and bar_close >= entry:
            return BasketAction(
                basket_id=basket.basket_id,
                action="exit",
                side=basket.side,
                reason="breakeven",
                px=bar_close,
                fraction=1.0,
            )

    # Level 4: target hit (activates trailing mode)
    if not trail.target_hit and d * (bar_close - target_px) >= 0:
        trail.target_hit = True

    # Level 5: time stop
    if not trail.target_hit and bars >= config.max_bars:
        return BasketAction(
            basket_id=basket.basket_id,
            action="exit",
            side=basket.side,
            reason="time",
            px=bar_close,
            fraction=1.0,
        )

    return None
