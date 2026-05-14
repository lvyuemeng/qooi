"""State provider interfaces for backtest and live basket reconstruction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from qooi.core.basket import Basket, BasketState, Position
from qooi.core.config import PairConfig

STATE_DIR = Path("data") / "state"
DEFAULT_SOFT_STATE_PATH = STATE_DIR / "baskets.json"

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
    """Rebuild basket state from OKX hard truth plus soft local strategy state."""

    def __init__(self, trading_client, soft_store: SoftStateStore | None = None):
        self._tc = trading_client
        self._soft_store = soft_store or JsonSoftStateStore()

    def load(self, pairs: list[PairConfig], strategy_id: str = "default") -> list[Basket]:
        positions = self._fetch_positions()
        soft_state = self._soft_store.read()
        baskets: list[Basket] = []

        for pair in pairs:
            symbol = pair.asset.symbol
            strategy = strategy_id
            basket_id = f"{symbol}-{strategy}"
            pos = _find_position(positions, symbol)
            saved = soft_state.get(basket_id, {})

            if pos and _position_qty(pos) != 0:
                qty = _position_qty(pos)
                basket = Basket(
                    basket_id=basket_id,
                    symbol=symbol,
                    strategy=strategy,
                    side="buy" if qty > 0 else "sell",
                    state=BasketState.ACTIVE,
                    entry_px=float(pos.get("avgPx", 0) or 0),
                    current_sz=abs(qty),
                )
                _apply_soft_state(basket, saved)
                for order in self._fetch_orders(symbol):
                    basket.positions.append(
                        Position(
                            symbol=symbol,
                            side=order.get("side", ""),
                            sz=float(order.get("sz", 0) or 0),
                            avg_px=float(order.get("px", 0) or 0),
                            order_id=order.get("ordId", ""),
                        )
                    )
            else:
                basket = Basket(basket_id=basket_id, symbol=symbol, strategy=strategy, side="")
            baskets.append(basket)

        return baskets

    def save_soft(self, baskets: list[Basket]) -> None:
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


def _apply_soft_state(basket: Basket, saved: dict) -> None:
    for field in SOFT_FIELDS:
        if field in saved:
            setattr(basket, field, saved[field])
    if basket.recovery_level > 0:
        basket.recovery_activated = True


def _find_position(positions: list[dict], symbol: str) -> dict | None:
    return next((pos for pos in positions if pos.get("instId") == symbol), None)


def _position_qty(position: dict) -> float:
    return float(position.get("pos", 0) or 0)
