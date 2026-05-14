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
    BasketBook,
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
    baskets: list[Basket] | BasketBook,
    pair: PairConfig,
    exit_cfg: ExitConfig | None = None,
    recovery_cfg: RecoveryConfig | None = None,
    *,
    signal_src: float | None = None,
    strategy_id: str = "default",
) -> list[BasketAction]:
    """Run all four layers. Returns BasketActions for executor."""

    exit_cfg = exit_cfg or ExitConfig()
    recovery_cfg = recovery_cfg or RecoveryConfig()

    actions: list[BasketAction] = []
    bar_idx = df.height - 1

    # ---- Layer 1: Signal ----
    if signal_src is None:
        return actions
    sig_val = signal_src

    close = float(df["close"][bar_idx])
    high = float(df["high"][bar_idx])
    low = float(df["low"][bar_idx])
    atr = float(df["atr_14"][bar_idx] or 1.0)

    sym = pair.asset.symbol
    book = baskets if isinstance(baskets, BasketBook) else BasketBook(baskets)
    basket = book.for_pair(sym, strategy_id)

    def _snapshot_exit(reason: str) -> BasketAction:
        assert basket is not None
        return BasketAction(
            basket_id=basket.basket_id,
            action=ActionKind.EXIT,
            side=basket.side,
            sz=basket.current_sz,
            px=close,
            reason=reason,
            fraction=1.0,
        )

    # ---- Layer 2: Basket — signal → basket lifecycle ----
    if basket is None or basket.is_idle:
        if sig_val != 0 and book.can_open(sym):
            side = "buy" if sig_val > 0 else "sell"
            stop_px, target_px = book.policy.compute_stop_target(side, close, atr, pair.asset)
            sz = book.policy.size_position(close, stop_px, pair.asset)
            basket = book.open(sym, strategy_id, side, close, sz, stop_px, target_px)
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
            actions.append(_snapshot_exit(ExitReason.SIGNAL_FLIP.value))
            book.close(basket)
            return actions
        if sig_val == 0:
            actions.append(_snapshot_exit(ExitReason.SIGNAL_ZERO.value))
            book.close(basket)
            return actions

        # ---- Layer 3: Recovery ----
        r_actions = evaluate_recovery(
            basket,
            close,
            atr,
            recovery_cfg,
            basket.recovery_level,
            ct_val=pair.asset.ct_val,
        )
        if r_actions:
            for r_a in r_actions:
                if r_a.action == ActionKind.EXIT:
                    r_a.sz = basket.current_sz
                    r_a.px = close
                actions.append(r_a)
                if r_a.action in (ActionKind.ADD_GRID, ActionKind.HEDGE):
                    basket.recovery_level += 1
                    basket.recovery_activated = True
                elif r_a.action == ActionKind.EXIT and r_a.reason == ExitReason.MARTINGALE.value:
                    book.close(basket)
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
            e_action.sz = basket.current_sz
            e_action.px = close
            actions.append(e_action)
            book.close(basket)
        else:
            basket.trail_high = trail.trail_high
            basket.trail_low = trail.trail_low
            basket.target_hit = trail.target_hit
            basket.bars_in_pos += 1

    return actions
