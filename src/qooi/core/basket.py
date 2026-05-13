"""Basket management — isolate parallel signals, dedup, portfolio exposure.

Each Basket is a container that holds one or more Positions for a single
instrument + strategy + direction.  Baskets are identified by basket_id
(derived from signal_chan_name) and tracked per bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
    state: str = "idle"
    positions: list[Position] = field(default_factory=list)
    entry_px: float = 0.0
    current_sz: float = 0.0
    entry_bar: int = 0
    bars_in_pos: int = 0
    loss_streak: int = 0
    trail_high: float = 0.0
    trail_low: float = 0.0
    target_hit: bool = False
    recovery_activated: bool = False
    recovery_level: int = 0
    suspended_long: bool = False
    suspended_short: bool = False
    suspension_px: float = 0.0

    @property
    def is_idle(self) -> bool:
        return self.state == "idle"

    @property
    def is_active(self) -> bool:
        return self.state == "active"

    @property
    def is_suspended(self) -> bool:
        return self.state == "suspended"


@dataclass
class BasketAction:
    basket_id: str
    action: str
    side: str = ""
    sz: float = 0.0
    px: float = 0.0
    stop_px: float = 0.0
    target_px: float = 0.0
    fraction: float = 1.0
    reason: str = ""
    order_type: str = "limit"


class BasketManager:
    """Creates, deduplicates, and limits baskets per instrument."""

    def __init__(self, max_baskets: int = 5, max_per_symbol: int = 1):
        self.max_baskets = max_baskets
        self.max_per_symbol = max_per_symbol

    def can_open(self, symbol: str, active: list[Basket]) -> bool:
        if len(active) >= self.max_baskets:
            return False
        symbol_count = sum(1 for b in active if b.symbol == symbol)
        return symbol_count < self.max_per_symbol

    def create(self, symbol: str, strategy: str, side: str, entry_px: float) -> Basket:
        return Basket(
            basket_id=f"{symbol}-{strategy}",
            symbol=symbol,
            strategy=strategy,
            side=side,
            state="active",
            entry_px=entry_px,
            current_sz=0.0,
        )

    def remove(self, basket: Basket) -> None:
        basket.state = "idle"
        basket.positions.clear()
