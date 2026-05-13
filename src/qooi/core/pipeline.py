"""Pipeline — compose four layers into a unified BasketAction stream.

Same entry point for backtest and live trading.
"""

from __future__ import annotations

import polars as pl

from qooi.core.basket import Basket, BasketAction, BasketManager
from qooi.core.config import PairConfig
from qooi.core.exits import ExitConfig, TrailTracker
from qooi.core.exits import evaluate as evaluate_exits
from qooi.core.recovery import RecoveryConfig
from qooi.core.recovery import evaluate as evaluate_recovery
from qooi.core.registry import resolve as resolve_strategy


def process_bar(
    df: pl.DataFrame,
    baskets: list[Basket],
    pair: PairConfig,
    exit_cfg: ExitConfig | None = None,
    recovery_cfg: RecoveryConfig | None = None,
) -> list[BasketAction]:
    """Run all four layers. Returns BasketActions for executor."""

    exit_cfg = exit_cfg or ExitConfig()
    recovery_cfg = recovery_cfg or RecoveryConfig()

    actions: list[BasketAction] = []
    bar_idx = df.height - 1

    # ---- Layer 1: Signal ----
    entry = resolve_strategy(pair.okx.strategy)
    signal = entry.compute(pair.asset.sig_symbol) if entry else None

    close = float(df["close"][bar_idx])
    high = float(df["high"][bar_idx])
    low = float(df["low"][bar_idx])
    atr = float(df["atr_14"][bar_idx] or 1.0)

    sym = pair.asset.symbol
    mgr = BasketManager()
    basket = next((b for b in baskets if b.basket_id == f"{sym}-{pair.okx.strategy}"), None)

    # ---- Layer 2: Basket — signal → basket lifecycle ----
    if signal is None:
        return actions

    sig_val = signal.signal

    if basket is None or basket.is_idle:
        if sig_val != 0 and mgr.can_open(sym, baskets):
            side = "buy" if sig_val > 0 else "sell"
            basket = mgr.create(sym, pair.okx.strategy, side, close)
            baskets.append(basket)
            actions.append(
                BasketAction(
                    basket_id=basket.basket_id,
                    action="enter",
                    side=side,
                    sz=0,
                    px=close,
                    reason="signal_entry",
                )
            )
    elif basket.is_active:
        d = 1 if basket.side == "buy" else -1
        if sig_val * d < 0:
            actions.append(
                BasketAction(
                    basket_id=basket.basket_id,
                    action="exit",
                    side=basket.side,
                    reason="signal_flip",
                    fraction=1.0,
                )
            )
            mgr.remove(basket)
            return actions

        # ---- Layer 3: Recovery ----
        r_action = evaluate_recovery(
            basket,
            close,
            atr,
            recovery_cfg,
            basket.recovery_level,
        )
        if r_action:
            actions.append(r_action)
            if r_action.action == "add_grid":
                basket.recovery_level += 1
                basket.recovery_activated = True
            elif r_action.reason == "martingale_reverse":
                mgr.remove(basket)
            return actions

        # ---- Layer 4: Exits ----
        trail = TrailTracker(
            trail_high=basket.trail_high,
            trail_low=basket.trail_low,
            target_hit=basket.target_hit,
        )
        e_action = evaluate_exits(basket, close, high, low, atr, trail, exit_cfg)
        if e_action:
            actions.append(e_action)
            mgr.remove(basket)
        else:
            basket.trail_high = trail.trail_high
            basket.trail_low = trail.trail_low
            basket.target_hit = trail.target_hit
            basket.bars_in_pos += 1

    return actions
