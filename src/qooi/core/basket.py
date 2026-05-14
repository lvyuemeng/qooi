"""Basket management + exit engine — isolates signals, manages positions, evaluates exits.

Each Basket holds one or more Positions for a single instrument + strategy + direction.
Exit logic is integrated because exits read Basket state directly (entry_px, bars_in_pos,
trail_high, trail_low, target_hit).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ActionKind(StrEnum):
    ENTER = "enter"
    EXIT = "exit"
    ADD_GRID = "add_grid"
    HEDGE = "hedge"


class BasketState(StrEnum):
    IDLE = "idle"
    ACTIVE = "active"
    SUSPENDED = "suspended"


class ExitReason(StrEnum):
    STOP = "stop"
    TRAILING = "trailing_stop"
    BREAKEVEN = "breakeven"
    TIME = "time"
    SIGNAL_ENTRY = "signal_entry"
    SIGNAL_FLIP = "signal_flip"
    MARTINGALE = "martingale_reverse"
    HEDGE_DRAWDOWN = "hedge_on_drawdown"
    GRID_LEVEL = "grid_level"
    GLOBAL_LOSS_LIMIT = "global_loss_limit"
    SIGNAL_ZERO = "signal_zero"


@dataclass
class Position:
    symbol: str
    side: str
    sz: float
    avg_px: float
    order_id: str = ""


@dataclass
class Basket:
    basket_id: str
    symbol: str
    strategy: str
    side: str
    state: BasketState = BasketState.IDLE
    positions: list[Position] = field(default_factory=list)
    entry_px: float = 0.0
    current_sz: float = 0.0
    stop_px: float = 0.0
    target_px: float = 0.0
    entry_bar: int = 0
    bars_in_pos: int = 0
    loss_streak: int = 0
    trail_high: float = 0.0
    trail_low: float = float("inf")
    target_hit: bool = False
    recovery_activated: bool = False
    recovery_level: int = 0
    cumulative_loss: float = 0.0
    suspended_long: bool = False
    suspended_short: bool = False
    suspension_px: float = 0.0

    @property
    def is_idle(self) -> bool:
        return self.state == BasketState.IDLE

    @property
    def is_active(self) -> bool:
        return self.state == BasketState.ACTIVE

    @property
    def is_suspended(self) -> bool:
        return self.state == BasketState.SUSPENDED

    def add_to_position(self, sz: float, px: float) -> None:
        """Add to existing position, updating avg entry price."""
        total_sz = self.current_sz + sz
        if total_sz > 0:
            self.entry_px = (self.entry_px * self.current_sz + px * sz) / total_sz
        self.current_sz = total_sz


@dataclass
class BasketAction:
    basket_id: str
    action: ActionKind
    side: str = ""
    sz: float = 0.0
    px: float = 0.0
    stop_px: float = 0.0
    target_px: float = 0.0
    fraction: float = 1.0
    reason: str = ""
    order_type: str = "limit"


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


class BasketManager:
    """Stateless basket policy/factory.

    Basket state is owned by ``BasketBook`` or by legacy caller-owned lists.
    """

    def __init__(self, max_baskets: int = 5, max_per_symbol: int = 1):
        self.max_baskets = max_baskets
        self.max_per_symbol = max_per_symbol

    def can_open(self, symbol: str, active: list[Basket]) -> bool:
        active_baskets = [b for b in active if b.is_active]
        if len(active_baskets) >= self.max_baskets:
            return False
        symbol_count = sum(1 for b in active_baskets if b.symbol == symbol)
        return symbol_count < self.max_per_symbol

    def create(
        self,
        symbol: str,
        strategy: str,
        side: str,
        entry_px: float,
        sz: float,
        stop_px: float,
        target_px: float,
    ) -> Basket:
        # Reuse existing idle basket if present, otherwise create
        bid = f"{symbol}-{strategy}"
        return Basket(
            basket_id=bid,
            symbol=symbol,
            strategy=strategy,
            side=side,
            state=BasketState.ACTIVE,
            entry_px=entry_px,
            current_sz=sz,
            stop_px=stop_px,
            target_px=target_px,
        )

    @staticmethod
    def remove(basket: Basket) -> None:
        """Full reset — clears state, positions, and all accumulators."""
        basket.state = BasketState.IDLE
        basket.positions.clear()
        basket.bars_in_pos = 0
        basket.trail_high = 0.0
        basket.trail_low = float("inf")
        basket.target_hit = False
        basket.recovery_activated = False
        basket.recovery_level = 0
        basket.cumulative_loss = 0.0
        basket.current_sz = 0.0

    @staticmethod
    def size_position(entry_px: float, stop_px: float, cfg) -> int:
        risk_per_ct = abs(entry_px - stop_px) * cfg.ct_val
        if risk_per_ct <= 0:
            return 0
        max_risk = cfg.capital * cfg.max_risk_pct
        sz = max(1, int(max_risk / risk_per_ct))
        notional_per_ct = cfg.ct_val * entry_px
        max_sz = int(cfg.capital * cfg.leverage / max(notional_per_ct, 1e-9))
        return max(1, min(sz, max_sz))

    @staticmethod
    def compute_stop_target(side: str, entry_px: float, atr: float, cfg) -> tuple[float, float]:
        d = 1 if side == "buy" else -1
        return (
            round(entry_px - d * cfg.atr_stop_mult * atr, 2),
            round(entry_px + d * cfg.atr_target_mult * atr, 2),
        )


class BasketBook:
    """Owns basket lifecycle state for one pipeline run/session."""

    def __init__(
        self,
        baskets: list[Basket] | None = None,
        *,
        max_baskets: int = 5,
        max_per_symbol: int = 1,
    ):
        self.baskets = baskets if baskets is not None else []
        self.policy = BasketManager(max_baskets=max_baskets, max_per_symbol=max_per_symbol)

    def get(self, basket_id: str) -> Basket | None:
        return next((b for b in self.baskets if b.basket_id == basket_id), None)

    def for_pair(self, symbol: str, strategy: str) -> Basket | None:
        return self.get(f"{symbol}-{strategy}")

    def active(self) -> list[Basket]:
        return [b for b in self.baskets if b.is_active]

    def active_exposure(self) -> float:
        return sum(abs(b.current_sz) for b in self.baskets if b.is_active)

    def can_open(self, symbol: str) -> bool:
        return self.policy.can_open(symbol, self.baskets)

    def replace_or_add(self, basket: Basket) -> None:
        for i, existing in enumerate(self.baskets):
            if existing.basket_id == basket.basket_id:
                self.baskets[i] = basket
                return
        self.baskets.append(basket)

    def open(
        self,
        symbol: str,
        strategy: str,
        side: str,
        entry_px: float,
        sz: float,
        stop_px: float,
        target_px: float,
    ) -> Basket:
        basket = self.policy.create(symbol, strategy, side, entry_px, sz, stop_px, target_px)
        self.replace_or_add(basket)
        return basket

    def close(self, basket: Basket) -> None:
        self.policy.remove(basket)


def evaluate_exits(
    basket: Basket,
    bar_close: float,
    bar_high: float,
    bar_low: float,
    atr: float,
    trail: TrailTracker,
    config: ExitConfig,
    *,
    skip_trailing: bool = False,
) -> BasketAction | None:
    bars = basket.bars_in_pos
    entry = basket.entry_px
    d = 1 if basket.side == "buy" else -1

    trail.update(bar_high, bar_low)

    stop_px = entry - d * config.stop_mult * atr
    target_px = entry + d * config.target_mult * atr

    if not trail.target_hit:
        if d * (stop_px - bar_close) >= 0:
            return BasketAction(
                basket_id=basket.basket_id,
                action=ActionKind.EXIT,
                side=basket.side,
                reason=ExitReason.STOP.value,
                px=bar_close,
                fraction=1.0,
            )

    if not skip_trailing and trail.target_hit:
        trail_stop = (
            trail.trail_high - config.trail_mult * atr
            if d > 0
            else trail.trail_low + config.trail_mult * atr
        )
        if d > 0 and bar_close <= trail_stop:
            return BasketAction(
                basket_id=basket.basket_id,
                action=ActionKind.EXIT,
                side=basket.side,
                reason=ExitReason.TRAILING.value,
                px=bar_close,
                fraction=1.0,
            )
        elif d < 0 and bar_close >= trail_stop:
            return BasketAction(
                basket_id=basket.basket_id,
                action=ActionKind.EXIT,
                side=basket.side,
                reason=ExitReason.TRAILING.value,
                px=bar_close,
                fraction=1.0,
            )

    if not skip_trailing and config.breakeven_after_target and trail.target_hit:
        if d > 0 and bar_close <= entry:
            return BasketAction(
                basket_id=basket.basket_id,
                action=ActionKind.EXIT,
                side=basket.side,
                reason=ExitReason.BREAKEVEN.value,
                px=bar_close,
                fraction=1.0,
            )
        elif d < 0 and bar_close >= entry:
            return BasketAction(
                basket_id=basket.basket_id,
                action=ActionKind.EXIT,
                side=basket.side,
                reason=ExitReason.BREAKEVEN.value,
                px=bar_close,
                fraction=1.0,
            )

    if not trail.target_hit and d * (bar_close - target_px) >= 0:
        trail.target_hit = True

    if bars >= config.max_bars:
        return BasketAction(
            basket_id=basket.basket_id,
            action=ActionKind.EXIT,
            side=basket.side,
            reason=ExitReason.TIME.value,
            px=bar_close,
            fraction=1.0,
        )

    return None
