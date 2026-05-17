"""Recovery — grid, martingale, and hedge strategies for drawdown recovery.

Produces list[BasketAction] for grid adds, direction reversals, and hedges.
Activated when a basket is in a losing state beyond thresholds.

Martingale reversal produces TWO actions: EXIT (close original) + ENTER
(open opposite with computed size from loss).  This maps to OKX
close-position + place order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from qooi.core.basket import ActionKind, Basket, BasketAction, ExitReason


@dataclass(frozen=True)
class RecoveryMarket:
    close: float
    atr: float


@dataclass
class RecoveryContext:
    current_level: int
    ct_val: float = 1.0
    signal_position: float = 0.0
    signal_entry: float = 0.0
    zscore: float | None = None
    zscore_delta: float | None = None
    short_momentum_return: float | None = None
    lower_wick_ratio: float | None = None
    upper_wick_ratio: float | None = None
    volatility_ratio: float | None = None
    trend_return: float | None = None
    adx: float | None = None


class RecoveryPolicy(Protocol):
    def evaluate(
        self, basket: Basket, market: RecoveryMarket, ctx: RecoveryContext
    ) -> list[BasketAction]: ...


@dataclass(frozen=True)
class NoRecovery:
    def evaluate(
        self, basket: Basket, market: RecoveryMarket, ctx: RecoveryContext
    ) -> list[BasketAction]:
        return []


@dataclass(frozen=True)
class RecoverySettings:
    zone_atr: float = 2.0
    multiplier: float = 2.0
    max_levels: int = 3
    max_loss_pct: float = 100.0
    breakeven_atr: float = 1.0


@dataclass(frozen=True)
class GridRecovery(RecoverySettings):
    def evaluate(
        self, basket: Basket, market: RecoveryMarket, ctx: RecoveryContext
    ) -> list[BasketAction]:
        shared = _shared_guard(basket, market.close, self)
        if shared is not None:
            return shared
        d, loss_pct = _direction_and_loss(basket, market.close)
        return _grid(basket, market.close, market.atr, self, ctx.current_level, loss_pct, d)


@dataclass(frozen=True)
class MartingaleRecovery(RecoverySettings):
    def evaluate(
        self, basket: Basket, market: RecoveryMarket, ctx: RecoveryContext
    ) -> list[BasketAction]:
        shared = _shared_guard(basket, market.close, self)
        if shared is not None:
            return shared
        d, loss_pct = _direction_and_loss(basket, market.close)
        return _martingale(
            basket, market.close, market.atr, self, ctx.current_level, loss_pct, d, ctx.ct_val
        )


@dataclass(frozen=True)
class ReverseRecovery(RecoverySettings):
    require_opposite_signal: bool = True

    def evaluate(
        self, basket: Basket, market: RecoveryMarket, ctx: RecoveryContext
    ) -> list[BasketAction]:
        shared = _shared_guard(basket, market.close, self)
        if shared is not None:
            return shared
        d, loss_pct = _direction_and_loss(basket, market.close)
        if self.require_opposite_signal:
            opposite_thesis = ctx.signal_entry * d < 0 or ctx.signal_position * d < 0
            if not opposite_thesis:
                return []
        return _martingale(
            basket, market.close, market.atr, self, ctx.current_level, loss_pct, d, ctx.ct_val
        )


@dataclass(frozen=True)
class HedgeRecovery(RecoverySettings):
    def evaluate(
        self, basket: Basket, market: RecoveryMarket, ctx: RecoveryContext
    ) -> list[BasketAction]:
        shared = _shared_guard(basket, market.close, self)
        if shared is not None:
            return shared
        _d, loss_pct = _direction_and_loss(basket, market.close)
        return _hedge(basket, market.close, self, loss_pct)


@dataclass(frozen=True)
class ZScoreReversionRecovery(RecoverySettings):
    multiplier: float = 1.0
    max_levels: int = 1
    wick_min: float = 0.35
    volatility_ratio_max: float = 1.5
    trend_return_max: float = 0.03
    adx_max: float = 35.0

    def evaluate(
        self, basket: Basket, market: RecoveryMarket, ctx: RecoveryContext
    ) -> list[BasketAction]:
        shared = _shared_guard(basket, market.close, self)
        if shared is not None:
            return shared
        d, loss_pct = _direction_and_loss(basket, market.close)
        if not _zscore_recovery_allowed(basket, ctx, self, d):
            return []
        return _grid(
            basket,
            market.close,
            market.atr,
            self,
            ctx.current_level,
            loss_pct,
            d,
            reason_prefix="zscore_recovery_level",
        )


def evaluate(
    basket: Basket,
    bar_close: float,
    atr: float,
    config: RecoveryPolicy,
    current_level: int,
    *,
    ct_val: float = 1.0,
    signal_position: float = 0.0,
    signal_entry: float = 0.0,
    zscore: float | None = None,
    zscore_delta: float | None = None,
    short_momentum_return: float | None = None,
    lower_wick_ratio: float | None = None,
    upper_wick_ratio: float | None = None,
    volatility_ratio: float | None = None,
    trend_return: float | None = None,
    adx: float | None = None,
) -> list[BasketAction]:
    """Evaluate recovery behavior. Returns list of actions (may be empty)."""
    return config.evaluate(
        basket,
        RecoveryMarket(close=bar_close, atr=atr),
        RecoveryContext(
            current_level=current_level,
            ct_val=ct_val,
            signal_position=signal_position,
            signal_entry=signal_entry,
            zscore=zscore,
            zscore_delta=zscore_delta,
            short_momentum_return=short_momentum_return,
            lower_wick_ratio=lower_wick_ratio,
            upper_wick_ratio=upper_wick_ratio,
            volatility_ratio=volatility_ratio,
            trend_return=trend_return,
            adx=adx,
        ),
    )


def _direction_and_loss(basket: Basket, bar_close: float) -> tuple[int, float]:
    d = 1 if basket.side == "buy" else -1
    loss_pct = d * (bar_close / basket.entry_px - 1) * 100 if basket.entry_px > 0 else 0.0
    return d, loss_pct


def _shared_guard(
    basket: Basket, bar_close: float, config: RecoverySettings
) -> list[BasketAction] | None:
    if not basket.is_active or basket.current_sz <= 0:
        return []
    _d, loss_pct = _direction_and_loss(basket, bar_close)
    if loss_pct < -abs(config.max_loss_pct):
        return [
            BasketAction(
                basket_id=basket.basket_id,
                action=ActionKind.EXIT,
                side=basket.side,
                sz=basket.current_sz,
                px=bar_close,
                reason=ExitReason.GLOBAL_LOSS_LIMIT.value,
                fraction=1.0,
            )
        ]
    return None


def _grid(
    basket: Basket,
    bar_close: float,
    atr: float,
    config: RecoverySettings,
    level: int,
    loss_pct: float,
    d: int,
    *,
    reason_prefix: str = "grid_level",
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
                reason=f"{reason_prefix}_{level + 1}",
            )
        ]
    return []


def _zscore_recovery_allowed(
    basket: Basket,
    ctx: RecoveryContext,
    config: ZScoreReversionRecovery,
    d: int,
) -> bool:
    if ctx.current_level >= config.max_levels:
        return False
    if ctx.zscore is None or ctx.zscore_delta is None:
        return False
    if ctx.volatility_ratio is not None and ctx.volatility_ratio > config.volatility_ratio_max:
        return False
    if ctx.adx is not None and ctx.adx > config.adx_max:
        return False

    if basket.side == "buy":
        if ctx.zscore >= 0:
            return False
        z_compressing = ctx.zscore_delta > 0
        momentum_toward_mean = (
            ctx.short_momentum_return is not None and ctx.short_momentum_return > 0
        )
        rejection = ctx.lower_wick_ratio is not None and ctx.lower_wick_ratio >= config.wick_min
        trend_continuation = (
            ctx.trend_return is not None
            and ctx.trend_return < -config.trend_return_max
            and ctx.zscore_delta < 0
        )
    else:
        if ctx.zscore <= 0:
            return False
        z_compressing = ctx.zscore_delta < 0
        momentum_toward_mean = (
            ctx.short_momentum_return is not None and ctx.short_momentum_return < 0
        )
        rejection = ctx.upper_wick_ratio is not None and ctx.upper_wick_ratio >= config.wick_min
        trend_continuation = (
            ctx.trend_return is not None
            and ctx.trend_return > config.trend_return_max
            and ctx.zscore_delta > 0
        )
    if trend_continuation:
        return False
    return z_compressing and (momentum_toward_mean or rejection) and d != 0


def _martingale(
    basket: Basket,
    bar_close: float,
    atr: float,
    config: RecoverySettings,
    level: int,
    loss_pct: float,
    d: int,
    ct_val: float,
) -> list[BasketAction]:
    if level >= config.max_levels or loss_pct > -config.zone_atr:
        return []

    reversal_side = "sell" if basket.side == "buy" else "buy"
    reversal_sz = _reversal_size(basket, bar_close, atr, config, ct_val)

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
            sz=reversal_sz,
            px=bar_close,
            reason=ExitReason.MARTINGALE.value,
        ),
    ]


def _hedge(
    basket: Basket,
    bar_close: float,
    config: RecoverySettings,
    loss_pct: float,
) -> list[BasketAction]:
    if basket.recovery_level > 0:
        return []
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


def _reversal_size(
    basket: Basket, bar_close: float, atr: float, config: RecoverySettings, ct_val: float
) -> int:
    loss_usd = abs(basket.entry_px - bar_close) * basket.current_sz * ct_val
    zone_profit_per_contract = config.zone_atr * atr * ct_val
    if zone_profit_per_contract <= 0 or basket.entry_px <= 0:
        return 1
    sz = math.ceil(loss_usd / zone_profit_per_contract)
    return max(1, sz)
