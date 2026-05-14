"""Core signal pipeline — shared across live trading and backtesting.

process_bar() composes the four layers (Signal -> Basket -> Recovery -> Exits)
and returns a list of BasketActions for the executor.
"""

from __future__ import annotations

import polars as pl

from qooi.core.basket import (
    ActionKind,
    Basket,
    BasketAction,
    BasketManager,
    ExitConfig,
    ExitReason,
    TrailTracker,
    evaluate_exits,
)
from qooi.core.config import PairConfig
from qooi.core.recovery import RecoveryConfig
from qooi.core.recovery import evaluate as evaluate_recovery


def process_bar(
    df: pl.DataFrame,
    baskets: list[Basket],
    pair: PairConfig,
    exit_cfg: ExitConfig | None = None,
    recovery_cfg: RecoveryConfig | None = None,
    *,
    signal_src: float | None = None,
) -> list[BasketAction]:
    """Run all four layers. Returns BasketActions for executor."""

    exit_cfg = exit_cfg or ExitConfig()
    recovery_cfg = recovery_cfg or RecoveryConfig()

    actions: list[BasketAction] = []
    bar_idx = df.height - 1

    # ---- Layer 1: Signal ----
    if signal_src is not None:
        sig_val = signal_src
    else:
        signal = pair.okx.compute(pair.asset.sig_symbol)
        if signal is None:
            return actions
        sig_val = signal.signal

    close = float(df["close"][bar_idx])
    high = float(df["high"][bar_idx])
    low = float(df["low"][bar_idx])
    atr = float(df["atr_14"][bar_idx] or 1.0)

    sym = pair.asset.symbol
    mgr = BasketManager()
    basket = next((b for b in baskets if b.basket_id == f"{sym}-{pair.okx.strategy}"), None)

    # ---- Layer 2: Basket — signal → basket lifecycle ----
    if basket is None or basket.is_idle:
        if sig_val != 0 and mgr.can_open(sym, baskets):
            side = "buy" if sig_val > 0 else "sell"
            stop_px, target_px = mgr.compute_stop_target(side, close, atr, pair.asset)
            sz = mgr.size_position(close, stop_px, pair.asset)
            basket = mgr.create(sym, pair.okx.strategy, side, close, sz, stop_px, target_px)
            baskets.append(basket)
            actions.append(
                BasketAction(
                    basket_id=basket.basket_id,
                    action=ActionKind.ENTER,
                    side=side,
                    sz=sz,
                    px=close,
                    stop_px=stop_px,
                    target_px=target_px,
                    reason=ExitReason.SIGNAL_ENTRY.value,
                )
            )
    elif basket.is_active:
        d = 1 if basket.side == "buy" else -1
        if sig_val * d < 0:
            actions.append(
                BasketAction(
                    basket_id=basket.basket_id,
                    action=ActionKind.EXIT,
                    side=basket.side,
                    reason=ExitReason.SIGNAL_FLIP.value,
                    fraction=1.0,
                )
            )
            mgr.remove(basket)
            return actions

        # ---- Layer 3: Recovery ----
        r_actions = evaluate_recovery(
            basket,
            close,
            atr,
            recovery_cfg,
            basket.recovery_level,
        )
        if r_actions:
            for r_a in r_actions:
                actions.append(r_a)
                if r_a.action == ActionKind.ADD_GRID:
                    basket.add_to_position(r_a.sz, r_a.px)
                    basket.recovery_level += 1
                    basket.recovery_activated = True
                elif r_a.action == ActionKind.EXIT and r_a.reason == ExitReason.MARTINGALE.value:
                    mgr.remove(basket)
            return actions

        # ---- Layer 4: Exits ----
        trail = TrailTracker(
            trail_high=basket.trail_high,
            trail_low=basket.trail_low,
            target_hit=basket.target_hit,
        )
        e_action = evaluate_exits(
            basket,
            close,
            high,
            low,
            atr,
            trail,
            exit_cfg,
            skip_trailing=basket.recovery_activated and basket.recovery_level > 0,
        )
        if e_action:
            actions.append(e_action)
            mgr.remove(basket)
        else:
            basket.trail_high = trail.trail_high
            basket.trail_low = trail.trail_low
            basket.target_hit = trail.target_hit
            basket.bars_in_pos += 1

    return actions
