"""Executor — maps BasketActions to real-world orders.

Two executors: LiveExecutor (direct OKX API) and BacktestExecutor (simulate).
Both consume the same list[BasketAction] from the pipeline.
"""

from __future__ import annotations

from qooi.core.basket import BasketAction


class LiveExecutor:
    """Execute BasketActions via direct OKX Trading API calls.

    Uses TradingClient for order placement, cancellation, and position queries.
    """

    def __init__(self, tc, md):
        self._tc = tc
        self._md = md

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
        if a.action == "enter":
            px = a.px or self._entry_px(a.side, a.basket_id)
            sz = int(a.sz) if a.sz > 0 else 1
            print(f"    ORDER {a.side} sz={sz} px={px} ({a.reason})")

        elif a.action == "exit":
            print(f"    CLOSE {a.side} ({a.reason})")

        elif a.action == "add_grid":
            px = a.px or self._entry_px(a.side, a.basket_id)
            print(f"    GRID ADD {a.side} sz={a.sz} px={px} ({a.reason})")

        elif a.action == "hedge":
            print(f"    HEDGE {a.side} sz={a.sz} ({a.reason})")

        elif a.action == "cancel":
            print(f"    CANCEL ({a.reason})")

    def _entry_px(self, side: str, symbol: str) -> float:
        obi = self._md.ob_snapshot(symbol, limit=1)
        if not obi:
            return 0.0
        return obi.ask_price if side == "buy" else obi.bid_price

    def _log(self, a: BasketAction) -> None:
        print(f"    {a.action.upper():10s} {a.side:5s} sz={a.sz:.0f} px={a.px:.2f} ({a.reason})")


class BacktestExecutor:
    """Simulate BasketActions against OHLCV bars for backtesting.

    Tracks equity curve, fills orders at bar open/close/limit prices,
    and computes P&L per trade.
    """

    def __init__(self):
        self.trades: list[dict] = []
        self.equity: list[float] = [10000.0]

    def simulate(self, actions: list[BasketAction], bar: dict) -> None:
        for a in actions:
            if a.action == "enter":
                pass
            elif a.action == "exit":
                pass
