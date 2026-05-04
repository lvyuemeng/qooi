"""Real-time order book via WebSocket — ccxt.pro or OKX native WS.

Sub-second order book updates for intraday strategies.  The ``ObClient``
keeps a local snapshot of bid/ask levels and provides a ``snapshot()``
method that returns the current imbalance signal ready for the strategy.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ObSnapshot:
    """Current state of the order book — ready for the OBI strategy."""

    timestamp: int = 0
    bid_price: float = 0.0
    ask_price: float = 0.0
    bid_vol_depth_5: float = 0.0
    ask_vol_depth_5: float = 0.0
    bid_vol_depth_25: float = 0.0
    ask_vol_depth_25: float = 0.0

    @property
    def imbalance_5(self) -> float:
        total = self.bid_vol_depth_5 + self.ask_vol_depth_5
        return (self.bid_vol_depth_5 - self.ask_vol_depth_5) / total if total > 0 else 0.0

    @property
    def imbalance_25(self) -> float:
        total = self.bid_vol_depth_25 + self.ask_vol_depth_25
        return (self.bid_vol_depth_25 - self.ask_vol_depth_25) / total if total > 0 else 0.0

    def to_row(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "ob_bid_price": self.bid_price,
            "ob_ask_price": self.ask_price,
            "ob_bid_vol": self.bid_vol_depth_25,
            "ob_ask_vol": self.ask_vol_depth_25,
        }

    @classmethod
    def from_ccxt_book(cls, book: dict) -> ObSnapshot:
        bids = book.get("bids", [])
        asks = book.get("asks", [])
        return cls(
            timestamp=int(book.get("timestamp", 0)),
            bid_price=bids[0][0] if bids else 0.0,
            ask_price=asks[0][0] if asks else 0.0,
            bid_vol_depth_5=sum(b[1] for b in bids[:5]),
            ask_vol_depth_5=sum(a[1] for a in asks[:5]),
            bid_vol_depth_25=sum(b[1] for b in bids[:25]),
            ask_vol_depth_25=sum(a[1] for a in asks[:25]),
        )


class ObClient:
    """WebSocket order book client.

    Two backends:

    - **ccxt.pro** (default)::
          client = ObClient("BTC/USDT", exchange_id="okx")
          async for snap in client.stream():
              print(snap.imbalance_5)

    - **OKX native WS**::
          client = ObClient("BTC-USDT", exchange_id="okx", native_ws=True)
          async for snap in client.stream():
              print(snap.imbalance_5)

    Stops after ``max_snapshots`` (0 = infinite).
    """

    def __init__(
        self,
        symbol: str,
        exchange_id: str = "okx",
        depth: int = 25,
        native_ws: bool = False,
        max_snapshots: int = 0,
    ) -> None:
        self._symbol = symbol
        self._exchange_id = exchange_id
        self._depth = depth
        self._native = native_ws
        self._max = max_snapshots

    async def stream(self) -> AsyncIterator[ObSnapshot]:
        if self._native:
            async for snap in self._stream_okx():
                yield snap
        else:
            async for snap in self._stream_ccxt_pro():
                yield snap

    async def _stream_ccxt_pro(self) -> AsyncIterator[ObSnapshot]:
        import ccxt.pro as ccxtpro

        exchange_class = getattr(ccxtpro, self._exchange_id)
        ex = exchange_class({"enableRateLimit": True})
        count = 0
        while self._max == 0 or count < self._max:
            book = await ex.watch_order_book(self._symbol, limit=self._depth)
            yield ObSnapshot.from_ccxt_book(book)
            count += 1

    async def _stream_okx(self) -> AsyncIterator[ObSnapshot]:
        import asyncio
        import json

        from okx.websocket.ws_public import WsPublicAsync

        ws = WsPublicAsync(url="wss://ws.okx.com:8443/ws/v5/public")
        await ws.connect()
        inst_id = self._symbol.replace("/", "-").split("-USDT")[0] + "-USDT"
        args = [
            {
                "channel": "books",
                "instId": inst_id,
            }
        ]

        queue: asyncio.Queue[ObSnapshot] = asyncio.Queue()
        store = {"bids": {}, "asks": {}}
        count = 0

        async def on_message(msg: str) -> None:
            nonlocal count
            data = json.loads(msg)
            if "data" not in data or not data["data"]:
                return
            snap, store["bids"], store["asks"] = self._apply_okx_snapshot(
                data["data"][0], store["bids"], store["asks"]
            )
            if snap:
                await queue.put(snap)

        ws.callback = on_message
        await ws.subscribe(args, callback=on_message, id="ob-sub")

        while self._max == 0 or count < self._max:
            try:
                snap = await asyncio.wait_for(queue.get(), timeout=30)
                yield snap
                count += 1
            except TimeoutError:
                break

        await ws.websocket.close()

    @staticmethod
    def _apply_okx_snapshot(
        data: dict, bids_store: dict, asks_store: dict
    ) -> tuple[ObSnapshot | None, dict, dict]:
        if data.get("action") == "snapshot":
            bids_store = {float(p): float(s) for p, s, *_ in data.get("bids", [])}
            asks_store = {float(p): float(s) for p, s, *_ in data.get("asks", [])}
        elif data.get("action") == "update":
            for p, s, *_ in data.get("bids", []):
                p_f, s_f = float(p), float(s)
                if s_f == 0:
                    bids_store.pop(p_f, None)
                else:
                    bids_store[p_f] = s_f
            for p, s, *_ in data.get("asks", []):
                p_f, s_f = float(p), float(s)
                if s_f == 0:
                    asks_store.pop(p_f, None)
                else:
                    asks_store[p_f] = s_f

        if not bids_store or not asks_store:
            return None, bids_store, asks_store

        bid_prices = sorted(bids_store.keys(), reverse=True)
        ask_prices = sorted(asks_store.keys())

        return (
            ObSnapshot(
                timestamp=int(data.get("ts", 0)),
                bid_price=bid_prices[0] if bid_prices else 0.0,
                ask_price=ask_prices[0] if ask_prices else 0.0,
                bid_vol_depth_5=sum(bids_store[p] for p in bid_prices[:5]),
                ask_vol_depth_5=sum(asks_store[p] for p in ask_prices[:5]),
                bid_vol_depth_25=sum(bids_store[p] for p in bid_prices[:25]),
                ask_vol_depth_25=sum(asks_store[p] for p in ask_prices[:25]),
            ),
            bids_store,
            asks_store,
        )


# Import for the type hint
try:
    from collections.abc import AsyncIterator
except ImportError:
    from collections.abc import AsyncIterator
