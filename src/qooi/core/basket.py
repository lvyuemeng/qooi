"""Basket management + exit engine — isolates signals, manages positions, evaluates exits.

Each Basket holds one or more Positions for a single instrument + strategy + direction.
Exit logic is integrated because exits read Basket state directly (entry_px, bars_in_pos,
trail_high, trail_low, target_hit).
"""

from __future__ import annotations

import math
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
    STRATEGY_EXIT = "strategy_exit"
    THESIS_FAILED = "thesis_failed"
    BLOCKED_ENTRY = "blocked_entry"


@dataclass(frozen=True)
class SizingDecision:
    contracts: float
    risk_per_contract: float
    risk_budget_usd: float
    risk_sized_contracts: float
    max_notional_usd: float
    notional_sized_contracts: float
    binding_cap: str
    blocked_reason: str = ""


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


@dataclass(frozen=True)
class BasketSnapshot:
    basket_id: str
    symbol: str
    strategy: str
    side: str
    entry_px: float
    current_sz: float
    stop_px: float
    target_px: float
    bars_in_pos: int
    recovery_level: int
    recovery_activated: bool


@dataclass
class BasketLimits:
    max_total: int = 5
    max_per_symbol: int = 3
    max_per_strategy_symbol: int = 3


@dataclass
class BasketAction:
    basket_id: str
    action: ActionKind
    side: str = ""
    sz: float = 0.0
    px: float = 0.0
    stop_px: float = 0.0
    target_px: float = 0.0
    entry_px: float = 0.0
    fraction: float = 1.0
    reason: str = ""
    order_type: str = "limit"
    bars_held: int = 0
    signal_id: str = ""
    signal_strength: float = 1.0
    sizing: SizingDecision | None = None
    snapshot: BasketSnapshot | None = None


@dataclass
class ExitConfig:
    drawdown_stop_pct: float | None = None
    no_drawdown_stop: bool = False
    stop_mult: float = 1.5
    target_mult: float = 1.3
    trail_mult: float = 2.0
    max_bars: int = 10
    breakeven_after_target: bool = False
    loss_cooldown_bars: int = 0
    session_end_utc: int = 22


HARD_EXIT_REASONS = {ExitReason.STOP.value, ExitReason.GLOBAL_LOSS_LIMIT.value}


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

    def __init__(
        self,
        max_baskets: int = 5,
        max_per_symbol: int = 3,
        max_per_strategy_symbol: int = 3,
    ):
        self.limits = BasketLimits(
            max_total=max_baskets,
            max_per_symbol=max_per_symbol,
            max_per_strategy_symbol=max_per_strategy_symbol,
        )

    def can_open(
        self, symbol: str, strategy: str | list[Basket] = "", active: list[Basket] | None = None
    ) -> bool:
        if active is None and isinstance(strategy, list):
            active = strategy
            strategy = ""
        active = active or []
        active_baskets = [b for b in active if b.is_active]
        if len(active_baskets) >= self.limits.max_total:
            return False
        symbol_count = sum(1 for b in active_baskets if b.symbol == symbol)
        if symbol_count >= self.limits.max_per_symbol:
            return False
        strategy_count = sum(
            1 for b in active_baskets if b.symbol == symbol and b.strategy == str(strategy)
        )
        return strategy_count < self.limits.max_per_strategy_symbol

    def create(
        self,
        symbol: str,
        strategy: str,
        side: str,
        entry_px: float,
        sz: float,
        stop_px: float,
        target_px: float,
        entry_ts: int = 0,
        sequence: int = 0,
    ) -> Basket:
        direction = "long" if side == "buy" else "short"
        unique = entry_ts if entry_ts else sequence or 1
        suffix = sequence or 1
        bid = f"{symbol}-{strategy}-{direction}-{unique}-{suffix}"
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
    def size_decision(
        entry_px: float,
        stop_px: float,
        cfg,
        *,
        signal_strength: float = 1.0,
        lot_multiplier: float = 1.0,
    ) -> SizingDecision:
        risk_per_ct = abs(entry_px - stop_px) * cfg.ct_val
        if risk_per_ct <= 0:
            return SizingDecision(0, 0.0, 0.0, 0, 0.0, 0, "none", "zero_risk_distance")
        strength = max(0.0, float(signal_strength or 0.0))
        max_risk = cfg.capital * cfg.max_risk_pct * strength * max(0.0, lot_multiplier)
        lot_size = max(float(getattr(cfg, "lot_size", 1.0)), 1e-9)
        min_contracts = max(float(getattr(cfg, "min_contracts", 1.0)), 0.0)

        def _round_lot(size: float) -> float:
            lots = math.floor(max(size, 0.0) / lot_size + 1e-9)
            return round(lots * lot_size, 10)

        risk_sz = _round_lot(max_risk / risk_per_ct)
        notional_per_ct = cfg.ct_val * entry_px
        notional_fraction = float(getattr(cfg, "max_notional_pct_per_basket", 1.0))
        max_notional = cfg.capital * cfg.leverage * max(0.0, notional_fraction)
        notional_sz = _round_lot(max_notional / max(notional_per_ct, 1e-9))
        raw_contracts = min(risk_sz, notional_sz)
        if raw_contracts + 1e-9 < min_contracts:
            binding = "risk" if risk_sz <= notional_sz else "notional"
            return SizingDecision(
                0,
                risk_per_ct,
                max_risk,
                risk_sz,
                max_notional,
                notional_sz,
                binding,
                f"below_min_contracts_{min_contracts:g}",
            )
        binding = "risk" if risk_sz <= notional_sz else "notional"
        return SizingDecision(
            round(raw_contracts, 10),
            risk_per_ct,
            max_risk,
            risk_sz,
            max_notional,
            notional_sz,
            binding,
        )

    @staticmethod
    def size_position(entry_px: float, stop_px: float, cfg) -> float:
        return BasketManager.size_decision(entry_px, stop_px, cfg).contracts

    @staticmethod
    def compute_stop_target(side: str, entry_px: float, atr: float, cfg) -> tuple[float, float]:
        d = 1 if side == "buy" else -1
        tick_size = max(float(getattr(cfg, "tick_size", 0.01)), 1e-9)

        def _round_tick(px: float) -> float:
            return round(round(px / tick_size) * tick_size, 10)

        return (
            _round_tick(entry_px - d * cfg.atr_stop_mult * atr),
            _round_tick(entry_px + d * cfg.atr_target_mult * atr),
        )


class BasketBook:
    """Owns basket lifecycle state for one pipeline run/session."""

    def __init__(
        self,
        baskets: list[Basket] | None = None,
        *,
        max_baskets: int = 5,
        max_per_symbol: int = 3,
        max_per_strategy_symbol: int = 3,
    ):
        self.baskets = baskets if baskets is not None else []
        self.policy = BasketManager(
            max_baskets=max_baskets,
            max_per_symbol=max_per_symbol,
            max_per_strategy_symbol=max_per_strategy_symbol,
        )

    def get(self, basket_id: str) -> Basket | None:
        return next((b for b in self.baskets if b.basket_id == basket_id), None)

    def for_pair(self, symbol: str, strategy: str) -> Basket | None:
        return next(
            (
                b
                for b in self.baskets
                if b.symbol == symbol and b.strategy == strategy and b.is_active
            ),
            None,
        )

    def active_for_symbol(self, symbol: str) -> list[Basket]:
        return [b for b in self.baskets if b.symbol == symbol and b.is_active]

    def active_for_strategy(self, symbol: str, strategy: str) -> list[Basket]:
        return [
            b for b in self.baskets if b.symbol == symbol and b.strategy == strategy and b.is_active
        ]

    def active(self) -> list[Basket]:
        return [b for b in self.baskets if b.is_active]

    def active_exposure(self) -> float:
        return sum(abs(b.current_sz) for b in self.baskets if b.is_active)

    def can_open(self, symbol: str, strategy: str = "", direction: str = "") -> bool:
        return self.policy.can_open(symbol, strategy, self.baskets)

    def open_block_reason(self, symbol: str, strategy: str = "") -> str:
        active_baskets = self.active()
        if len(active_baskets) >= self.policy.limits.max_total:
            return "max_total"
        symbol_count = sum(1 for b in active_baskets if b.symbol == symbol)
        if symbol_count >= self.policy.limits.max_per_symbol:
            return "max_per_symbol"
        if (
            sum(1 for b in active_baskets if b.symbol == symbol and b.strategy == strategy)
            >= self.policy.limits.max_per_strategy_symbol
        ):
            return "max_per_strategy_symbol"
        return ""

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
        entry_ts: int = 0,
    ) -> Basket:
        sequence = len(self.baskets) + 1
        basket = self.policy.create(
            symbol, strategy, side, entry_px, sz, stop_px, target_px, entry_ts, sequence
        )
        self.replace_or_add(basket)
        return basket

    def close(self, basket: Basket) -> None:
        self.policy.remove(basket)

    @staticmethod
    def snapshot(basket: Basket) -> BasketSnapshot:
        return BasketSnapshot(
            basket_id=basket.basket_id,
            symbol=basket.symbol,
            strategy=basket.strategy,
            side=basket.side,
            entry_px=basket.entry_px,
            current_sz=basket.current_sz,
            stop_px=basket.stop_px,
            target_px=basket.target_px,
            bars_in_pos=basket.bars_in_pos,
            recovery_level=basket.recovery_level,
            recovery_activated=basket.recovery_activated,
        )

    def apply_action(self, action: BasketAction) -> None:
        basket = self.get(action.basket_id)
        if action.action == ActionKind.ENTER:
            if basket is None:
                basket = Basket(
                    basket_id=action.basket_id,
                    symbol=(action.snapshot.symbol if action.snapshot else ""),
                    strategy=(action.snapshot.strategy if action.snapshot else ""),
                    side=action.side,
                    state=BasketState.ACTIVE,
                    entry_px=action.entry_px or action.px,
                    current_sz=action.sz,
                    stop_px=action.stop_px,
                    target_px=action.target_px,
                )
                self.replace_or_add(basket)
            else:
                basket.state = BasketState.ACTIVE
                basket.side = action.side or basket.side
                basket.entry_px = action.entry_px or action.px
                basket.current_sz = action.sz
                basket.stop_px = action.stop_px
                basket.target_px = action.target_px
            return
        if basket is None:
            return
        if action.action == ActionKind.ADD_GRID:
            basket.add_to_position(action.sz, action.px)
            basket.recovery_level += 1
            basket.recovery_activated = True
        elif action.action == ActionKind.HEDGE:
            hedge_id = f"{action.basket_id}_hedge"
            self.replace_or_add(
                Basket(
                    basket_id=hedge_id,
                    symbol=basket.symbol,
                    strategy=basket.strategy,
                    side=action.side,
                    state=BasketState.ACTIVE,
                    entry_px=action.entry_px or action.px,
                    current_sz=action.sz,
                    stop_px=action.stop_px,
                    target_px=action.target_px,
                    recovery_activated=True,
                    recovery_level=basket.recovery_level + 1,
                )
            )
            basket.recovery_level += 1
            basket.recovery_activated = True
        elif action.action == ActionKind.EXIT:
            self.close(basket)

    def apply_actions(self, actions: list[BasketAction]) -> None:
        for action in actions:
            self.apply_action(action)

    def advance_bar(self, close: float, high: float, low: float, *, skip_ids: set[str]) -> None:
        for basket in self.active():
            if basket.basket_id in skip_ids:
                continue
            if high > basket.trail_high:
                basket.trail_high = high
            if low < basket.trail_low:
                basket.trail_low = low
            d = 1 if basket.side == "buy" else -1
            target_hit = high >= basket.target_px if d > 0 else low <= basket.target_px
            if basket.target_px > 0 and not basket.target_hit and target_hit:
                basket.target_hit = True
            basket.bars_in_pos += 1


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

    stop_px = basket.stop_px if basket.stop_px > 0 else entry - d * config.stop_mult * atr
    target_px = basket.target_px if basket.target_px > 0 else entry + d * config.target_mult * atr

    if not trail.target_hit:
        stop_hit = bar_low <= stop_px if d > 0 else bar_high >= stop_px
        if stop_hit:
            return BasketAction(
                basket_id=basket.basket_id,
                action=ActionKind.EXIT,
                side=basket.side,
                reason=ExitReason.STOP.value,
                px=stop_px,
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

    target_hit = bar_high >= target_px if d > 0 else bar_low <= target_px
    if not trail.target_hit and target_hit:
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


def evaluate_hard_exits(
    basket: Basket,
    bar_close: float,
    bar_high: float,
    bar_low: float,
    atr: float,
    config: ExitConfig,
) -> BasketAction | None:
    """Evaluate non-negotiable risk exits that must outrank recovery."""
    entry = basket.entry_px
    d = 1 if basket.side == "buy" else -1
    stop_px = basket.stop_px if basket.stop_px > 0 else entry - d * config.stop_mult * atr
    if basket.target_hit:
        return None
    stop_hit = bar_low <= stop_px if d > 0 else bar_high >= stop_px
    if not stop_hit:
        return None
    return BasketAction(
        basket_id=basket.basket_id,
        action=ActionKind.EXIT,
        side=basket.side,
        reason=ExitReason.STOP.value,
        px=stop_px,
        fraction=1.0,
    )
