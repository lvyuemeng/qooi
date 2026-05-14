"""Executor — maps BasketActions to real-world orders.

Two executors: LiveExecutor (direct OKX API) and BacktestExecutor (simulate).
Both consume the same list[BasketAction] from the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

from qooi.core.basket import ActionKind, Basket, BasketAction, BasketState, Position

STATE_DIR = Path("data") / "state"
STATE_PATH = STATE_DIR / "baskets.json"


class LiveExecutor:
    """Execute BasketActions via direct OKX Trading API calls.

    State management: loads soft accumulators (trail_high, trail_low,
    recovery_level, target_hit, bars_in_pos) from data/state/baskets.json
    on startup.  Hard truth (position side, size, avg price) comes from
    OKX API via TradingClient.
    """

    def __init__(self, tc, md):
        self._tc = tc
        self._md = md

    def load_state(self, pairs) -> list[Basket]:
        """Reconstruct Basket state from OKX positions + persisted JSON.

        Hard truth from API: side, size, avgPx, open orders.
        Soft accumulators from disk: trail_high, trail_low, recovery_level,
        target_hit, bars_in_pos, recovery_activated.
        """
        positions = self._fetch_positions()
        persisted = _read_state()

        baskets: list[Basket] = []
        for p in pairs:
            sym = p.asset.symbol
            bid = f"{sym}-{p.okx.strategy}"

            pos = next((px for px in positions if px.get("instId") == sym), None)
            saved = persisted.get(bid, {})

            if pos and float(pos.get("pos", 0)) != 0:
                qty = float(pos.get("pos", 0))
                basket = Basket(
                    basket_id=bid,
                    symbol=sym,
                    strategy=p.okx.strategy,
                    side="buy" if qty > 0 else "sell",
                    state=BasketState.ACTIVE,
                    entry_px=float(pos.get("avgPx", 0)),
                    current_sz=abs(qty),
                    recovery_level=saved.get("recovery_level", 0),
                    trail_high=saved.get("trail_high", 0.0),
                    trail_low=saved.get("trail_low", 0.0),
                    target_hit=saved.get("target_hit", False),
                    bars_in_pos=saved.get("bars_in_pos", 0),
                    recovery_activated=saved.get("recovery_activated", False),
                    loss_streak=saved.get("loss_streak", 0),
                    suspended_long=saved.get("suspended_long", False),
                    suspended_short=saved.get("suspended_short", False),
                    suspension_px=saved.get("suspension_px", 0.0),
                )
                for o in self._fetch_orders(sym):
                    basket.positions.append(
                        Position(
                            symbol=sym,
                            side=o.get("side", ""),
                            sz=float(o.get("sz", 0)),
                            avg_px=float(o.get("px", 0)),
                            order_id=o.get("ordId", ""),
                        )
                    )
                if saved.get("recovery_level", 0) > 0:
                    basket.recovery_activated = True
            else:
                basket = Basket(basket_id=bid, symbol=sym, strategy=p.okx.strategy, side="")

            baskets.append(basket)
        return baskets

    def save_state(self, baskets: list[Basket]) -> None:
        _write_state(baskets)

    def execute(self, actions: list[BasketAction], dry_run: bool = False) -> None:
        for a in actions:
            if dry_run:
                self._log(a)
                continue
            try:
                self._dispatch(a)
            except Exception as e:
                print(f"    EXEC FAILED [{a.action}]: {e}")

    def _dispatch(self, a: BasketAction) -> None:
        if a.action == ActionKind.ENTER:
            px = a.px or self._entry_px(a.side, a.basket_id)
            sz = int(a.sz) if a.sz > 0 else 1
            print(f"    ORDER {a.side} sz={sz} px={px} ({a.reason})")

        elif a.action == ActionKind.EXIT:
            print(f"    CLOSE {a.side} ({a.reason})")

        elif a.action == ActionKind.ADD_GRID:
            px = a.px or self._entry_px(a.side, a.basket_id)
            print(f"    GRID ADD {a.side} sz={a.sz} px={px} ({a.reason})")

        elif a.action == ActionKind.HEDGE:
            print(f"    HEDGE {a.side} sz={a.sz} ({a.reason})")

    def _entry_px(self, side: str, symbol: str) -> float:
        obi = self._md.ob_snapshot(symbol, limit=1)
        if not obi:
            return 0.0
        return obi.ask_price if side == "buy" else obi.bid_price

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

    def _log(self, a: BasketAction) -> None:
        print(f"    {a.action.upper():10s} {a.side:5s} sz={a.sz:.0f} px={a.px:.2f} ({a.reason})")


class BacktestExecutor:
    """Simulate BasketActions against OHLCV bars for backtesting."""

    def __init__(self):
        self.trades: list[dict] = []
        self.equity: list[float] = [10000.0]

    def simulate(self, actions: list[BasketAction], bar: dict) -> None:
        for a in actions:
            if a.action == ActionKind.ENTER:
                pass
            elif a.action == ActionKind.EXIT:
                pass


def _read_state() -> dict[str, dict]:
    if not STATE_PATH.exists():
        return {}
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_state(baskets: list[Basket]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    data: dict[str, dict] = {}
    for b in baskets:
        data[b.basket_id] = {
            "recovery_level": b.recovery_level,
            "trail_high": b.trail_high,
            "trail_low": b.trail_low,
            "target_hit": b.target_hit,
            "bars_in_pos": b.bars_in_pos,
            "recovery_activated": b.recovery_activated,
            "loss_streak": b.loss_streak,
            "suspended_long": b.suspended_long,
            "suspended_short": b.suspended_short,
            "suspension_px": b.suspension_px,
        }
    STATE_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
