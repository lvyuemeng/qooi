"""State provider interfaces for backtest and live basket reconstruction."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from qooi.core.basket import Basket, BasketState, Position
from qooi.core.config import PairConfig

STATE_DIR = Path("data") / "state"
DEFAULT_SOFT_STATE_PATH = STATE_DIR / "baskets.json"
CLIENT_ID_PREFIX = "qooi"

SOFT_FIELDS = (
    "recovery_level",
    "trail_high",
    "trail_low",
    "target_hit",
    "bars_in_pos",
    "recovery_activated",
    "loss_streak",
    "suspended_long",
    "suspended_short",
    "suspension_px",
)


@dataclass(frozen=True)
class PositionSnapshot:
    symbol: str
    strategy: str
    basket_id: str
    side: str
    size: float
    avg_px: float
    active: bool


@dataclass(frozen=True)
class OkxOrderSource:
    symbol: str
    side: str
    size: float
    price: float
    order_id: str
    client_id: str
    basket_id: str


@dataclass(frozen=True)
class OkxPositionSource:
    symbol: str
    side: str
    size: float
    avg_px: float


@dataclass(frozen=True)
class BasketStateSource:
    basket_id: str
    symbol: str
    strategy: str
    branch: str = ""
    position: OkxPositionSource | None = None
    orders: tuple[OkxOrderSource, ...] = ()


@dataclass(frozen=True)
class EvaluatedBasketState:
    basket_id: str
    symbol: str
    strategy: str
    branch: str
    state: BasketState
    side: str
    entry_px: float
    current_sz: float
    positions: tuple[PositionSnapshot, ...]
    recovery_level: int = 0
    recovery_activated: bool = False


class StateProvider(Protocol):
    def load(self, pairs: list[PairConfig], strategy_id: str = "default") -> list[Basket]:
        """Load basket state for a pipeline run."""

    def save_soft(self, baskets: list[Basket]) -> None:
        """Persist soft strategy state only."""


class SoftStateStore(Protocol):
    def read(self) -> dict[str, dict]:
        """Read persisted soft basket state."""

    def write(self, baskets: list[Basket]) -> None:
        """Write persisted soft basket state."""


class JsonSoftStateStore:
    """JSON persistence for soft basket state only."""

    def __init__(self, path: Path = DEFAULT_SOFT_STATE_PATH):
        self.path = path

    def read(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def write(self, baskets: list[Basket]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {b.basket_id: _soft_state(b) for b in baskets}
        self.path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


class NullSoftStateStore:
    """No-op soft state store for tests and stateless runs."""

    def read(self) -> dict[str, dict]:
        return {}

    def write(self, baskets: list[Basket]) -> None:
        return None


class BacktestStateProvider:
    """In-memory backtest state provider; never reads or writes files."""

    def __init__(self, baskets: list[Basket] | None = None):
        self.baskets = baskets if baskets is not None else []

    def load(self, pairs: list[PairConfig], strategy_id: str = "default") -> list[Basket]:
        return self.baskets

    def save_soft(self, baskets: list[Basket]) -> None:
        self.baskets = baskets


class OkxStateProvider:
    """Rebuild basket state from OKX hard truth only."""

    def __init__(self, trading_client, soft_store: SoftStateStore | None = None):
        self._tc = trading_client
        self._soft_store = soft_store

    def load(self, pairs: list[PairConfig], strategy_id: str = "default") -> list[Basket]:
        positions = self._fetch_positions()
        sources: dict[str, BasketStateSource] = {}

        for pair in pairs:
            symbol = pair.asset.symbol
            strategy = strategy_id
            basket_id = format_basket_id(symbol, strategy)
            position = position_source_from_okx(_find_position(positions, symbol))
            candidates = (
                basket_id,
                format_basket_id(symbol, strategy, "hedge"),
                format_basket_id(symbol, strategy, "reversal"),
            )
            order_sources = [
                order_source
                for order in self._fetch_orders(symbol)
                if (order_source := order_source_from_okx(order, candidates)) is not None
            ]
            base_orders = tuple(o for o in order_sources if o.basket_id == basket_id)
            sources[basket_id] = BasketStateSource(
                basket_id=basket_id,
                symbol=symbol,
                strategy=strategy,
                position=position,
                orders=base_orders,
            )
            for order_source in order_sources:
                if order_source.basket_id == basket_id or order_source.basket_id in sources:
                    continue
                parsed = parse_basket_id(order_source.basket_id)
                if parsed is None:
                    continue
                branch_orders = tuple(
                    o for o in order_sources if o.basket_id == order_source.basket_id
                )
                sources[order_source.basket_id] = BasketStateSource(
                    basket_id=order_source.basket_id,
                    symbol=parsed[0],
                    strategy=parsed[1],
                    branch=parsed[2],
                    orders=branch_orders,
                )

        return [
            basket_from_evaluated_state(evaluate_basket_source(source))
            for source in sources.values()
        ]

    def save_soft(self, baskets: list[Basket]) -> None:
        if self._soft_store is not None:
            self._soft_store.write(baskets)

    def _fetch_positions(self) -> list[dict]:
        try:
            return self._tc.positions(inst_type="SWAP")
        except Exception:
            return []

    def _fetch_orders(self, inst_id: str) -> list[dict]:
        try:
            return self._tc.orders(inst_id, inst_type="SWAP")
        except Exception:
            return []


def _soft_state(basket: Basket) -> dict:
    return {field: getattr(basket, field) for field in SOFT_FIELDS}


def position_source_from_okx(position: dict | None) -> OkxPositionSource | None:
    if not position:
        return None
    qty = _position_qty(position)
    if qty == 0:
        return None
    return OkxPositionSource(
        symbol=str(position.get("instId", "")),
        side="buy" if qty > 0 else "sell",
        size=abs(qty),
        avg_px=float(position.get("avgPx", 0) or 0),
    )


def order_source_from_okx(order: dict, candidates: tuple[str, ...]) -> OkxOrderSource | None:
    basket_id = basket_id_from_okx_order(order, candidates)
    if not basket_id:
        return None
    return OkxOrderSource(
        symbol=str(order.get("instId", "")),
        side=str(order.get("side", "")),
        size=float(order.get("sz", 0) or 0),
        price=float(order.get("px", 0) or 0),
        order_id=str(order.get("ordId", "")),
        client_id=str(order.get("clOrdId") or order.get("algoClOrdId") or ""),
        basket_id=basket_id,
    )


def basket_id_from_okx_order(order: dict, candidates: tuple[str, ...]) -> str:
    client_id = str(order.get("clOrdId") or order.get("algoClOrdId") or "")
    if not client_id:
        return ""
    for candidate in sorted(candidates, key=lambda item: len(item), reverse=True):
        encoded = format_okx_client_id(candidate)
        if client_id == encoded or client_id.startswith(encoded):
            return candidate
    return ""


def position_snapshots_from_orders(
    symbol: str, orders: tuple[OkxOrderSource, ...]
) -> tuple[PositionSnapshot, ...]:
    return tuple(
        PositionSnapshot(
            symbol=symbol,
            strategy="",
            basket_id=order.basket_id,
            side=order.side,
            size=order.size,
            avg_px=order.price,
            active=order.size != 0,
        )
        for order in orders
    )


def evaluate_basket_source(source: BasketStateSource) -> EvaluatedBasketState:
    if source.position is None:
        return EvaluatedBasketState(
            basket_id=source.basket_id,
            symbol=source.symbol,
            strategy=source.strategy,
            branch=source.branch,
            state=BasketState.IDLE,
            side="",
            entry_px=0.0,
            current_sz=0.0,
            positions=position_snapshots_from_orders(source.symbol, source.orders),
        )
    return EvaluatedBasketState(
        basket_id=source.basket_id,
        symbol=source.symbol,
        strategy=source.strategy,
        branch=source.branch,
        state=BasketState.ACTIVE,
        side=source.position.side,
        entry_px=source.position.avg_px,
        current_sz=source.position.size,
        positions=position_snapshots_from_orders(source.symbol, source.orders),
    )


def basket_from_evaluated_state(state: EvaluatedBasketState) -> Basket:
    return Basket(
        basket_id=state.basket_id,
        symbol=state.symbol,
        strategy=state.strategy,
        side=state.side,
        state=state.state,
        positions=[
            Position(
                symbol=pos.symbol,
                side=pos.side,
                sz=pos.size,
                avg_px=pos.avg_px,
            )
            for pos in state.positions
        ],
        entry_px=state.entry_px,
        current_sz=state.current_sz,
        recovery_level=state.recovery_level,
        recovery_activated=state.recovery_activated,
    )


def format_basket_id(symbol: str, strategy_id: str, branch: str = "") -> str:
    base = f"{symbol}-{strategy_id}"
    return f"{base}_{branch}" if branch else base


def parse_basket_id(basket_id: str) -> tuple[str, str, str] | None:
    symbol_suffix = "-SWAP"
    if symbol_suffix not in basket_id:
        return None
    symbol_end = basket_id.index(symbol_suffix) + len(symbol_suffix)
    if symbol_end >= len(basket_id) or basket_id[symbol_end] != "-":
        return None
    symbol = basket_id[:symbol_end]
    rest = basket_id[symbol_end + 1 :]
    strategy = rest
    branch = ""
    for suffix in ("hedge", "reversal"):
        marker = f"_{suffix}"
        if rest.endswith(marker):
            strategy = rest[: -len(marker)]
            branch = suffix
            break
    if not strategy:
        return None
    return symbol, strategy, branch


def format_okx_client_id(basket_id: str, suffix: str = "") -> str:
    """Encode a local basket id into an OKX-safe client id."""
    raw = f"{CLIENT_ID_PREFIX}-{basket_id}"
    if suffix:
        raw = f"{raw}-{suffix}"
    safe = re.sub(r"[^A-Za-z0-9]", "", raw)
    return safe[:32]


def parse_okx_client_id(client_id: str) -> str:
    """Return the encoded basket identity token from an OKX client id."""
    safe_prefix = CLIENT_ID_PREFIX
    if not client_id.startswith(safe_prefix):
        return ""
    return client_id[len(safe_prefix) :]


def _find_position(positions: list[dict], symbol: str) -> dict | None:
    return next((pos for pos in positions if pos.get("instId") == symbol), None)


def _position_qty(position: dict) -> float:
    return float(position.get("pos", 0) or 0)

