"""Exchange backend layer — abstracts ccxt (sync + async WS) and OKX SDK.

Architecture::

    ExchangeBackend  (sync: fetch_ohlcv / fetch_order_book)
         |
         +-- CcxtBackend     wraps ccxt.{sync,async_support,pro}
         |
         +-- OkxSdkBackend   wraps okx.MarketData / PublicData (funding rate)

    MarketData  (thin facade — no backend branching in methods)
         |
         +-- .candles(...)             → polars DataFrame
         +-- .candles_range(...)       → deep paginated DataFrame
         +-- .ob_snapshot(symbol)      → ObSnapshot (REST, sync)
         +-- .ob_stream(symbol)        → AsyncIterator[ObSnapshot] (WS)

Usage::

    # Synchronous (backtesting)
    md = MarketData("lbank")
    df = md.candles("BTC/USDT", "1d", limit=100)

    # Async with WebSocket (live)
    md = await MarketData.async_("okx")
    async for snap in md.ob_stream("BTC/USDT", params={"depth": "books"}):
        print(snap.imbalance_5)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import polars as pl

# ---------------------------------------------------------------------------
# ObSnapshot — shared order-book model
# ---------------------------------------------------------------------------


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
            "ob_bid_vol_5": self.bid_vol_depth_5,
            "ob_ask_vol_5": self.ask_vol_depth_5,
            "ob_bid_vol_25": self.bid_vol_depth_25,
            "ob_ask_vol_25": self.ask_vol_depth_25,
            "ob_bid_vol": self.bid_vol_depth_25,
            "ob_ask_vol": self.ask_vol_depth_25,
            "ob_imbalance_5": self.imbalance_5,
            "ob_imbalance_25": self.imbalance_25,
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


# ---------------------------------------------------------------------------
# ExchangeBackend — abstract transport
# ---------------------------------------------------------------------------


class ExchangeBackend:
    """Synchronous exchange backend — REST only.

    Subclasses implement: fetch_ohlcv, fetch_order_book, maybe funding_rate.

    The async counterpart for WS is created via ``.async_pro()`` which
    returns a ``CcxtProBackend`` sharing the same exchange id + proxy.
    """

    def __init__(self, exchange_id: str = "okx", proxy: str | None = None) -> None:
        self._exchange_id = exchange_id
        self._proxy = proxy

    @property
    def exchange_id(self) -> str:
        return self._exchange_id

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1d", limit: int = 500) -> list[list]:
        raise NotImplementedError

    def fetch_ohlcv_range(
        self, symbol: str, timeframe: str = "1d", since: str = "2020-01-01", limit: int = 3000
    ) -> pl.DataFrame:
        data: list[list] = []
        since_ms = int(datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000)
        while len(data) < limit:
            chunk = self.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=500)
            if not chunk:
                break
            data.extend(chunk)
            since_ms = chunk[-1][0] + 1
            if len(chunk) < 500:
                break
        if len(data) > limit:
            data = data[:limit]
        return _parse_ohlcv(data)

    def fetch_order_book(self, symbol: str, limit: int = 25) -> ObSnapshot:
        raise NotImplementedError

    def funding_rate_history(
        self, inst_id: str = "BTC-USDT-SWAP", limit: int = 100
    ) -> pl.DataFrame:
        return pl.DataFrame()

    def market_data_history(
        self,
        module: int,
        inst_type: str,
        date_aggr_type: str,
        begin: str,
        end: str,
        inst_id_list: list[str] | None = None,
        inst_family_list: list[str] | None = None,
    ) -> list[dict]:
        return []

    def close(self) -> None:
        pass


class CcxtBackend(ExchangeBackend):
    """CCXT synchronous REST backend."""

    def __init__(self, exchange_id: str, proxy: str | None = None) -> None:
        super().__init__(exchange_id, proxy)
        import ccxt

        klass = getattr(ccxt, exchange_id)
        config: dict[str, Any] = {"enableRateLimit": True}
        if proxy:
            config["proxies"] = {"https": proxy, "http": proxy}
        self._ex = klass(config)
        try:
            self._ex.load_markets()
        except Exception as e:
            msg = f"Cannot connect to {exchange_id}"
            raise ConnectionError(msg + (f" via proxy {proxy}" if proxy else "")) from e

    def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1d", limit: int = 500, since: int | None = None
    ) -> list[list]:
        return self._ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)  # type: ignore[no-any-return]

    def fetch_order_book(self, symbol: str, limit: int = 25) -> ObSnapshot:
        raw = self._ex.fetch_order_book(symbol, limit=limit)
        return ObSnapshot.from_ccxt_book(raw)

    def close(self) -> None:
        close = getattr(self._ex, "close", None)
        if callable(close):
            close()


class OkxSdkBackend(ExchangeBackend):
    """OKX native Python SDK backend.

    Handles candles / funding rate via the synchronous Python SDK.
    Order book comes from ccxt (OKX is always available).
    """

    def __init__(self, proxy: str | None = None) -> None:
        super().__init__("okx", proxy)
        from okx.MarketData import MarketAPI

        self._api = MarketAPI(flag="1", debug=False)
        self._ccxt: CcxtBackend | None = None  # lazy init — only needed for OB

    def _ensure_ccxt(self) -> CcxtBackend:
        if self._ccxt is None:
            self._ccxt = CcxtBackend("okx", self._proxy)
        return self._ccxt

    def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1d", limit: int = 500, since: int | None = None
    ) -> list[list]:
        # Use OKX SDK directly for candles (fast, no CCXT connection needed)
        try:
            resp = self._api.get_candlesticks(instId=symbol, bar=timeframe, limit=str(limit))
            if resp.get("code") != "0":
                raise RuntimeError(f"OKX SDK error: {resp.get('msg', resp)}")
            data = []
            for r in resp.get("data", []):
                data.append(
                    [int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])]
                )
            return data
        except Exception:
            pass
        return self._ensure_ccxt().fetch_ohlcv(symbol, timeframe, limit=limit)

    def fetch_ohlcv_range(
        self, symbol: str, timeframe: str = "1d", since: str = "2020-01-01", limit: int = 3000
    ) -> pl.DataFrame:
        """Use OKX native history candles for contiguous deep pagination."""
        since_ms = int(datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000)
        page_limit = min(100, limit)
        after = ""
        rows: list[list] = []
        seen: set[int] = set()

        while len(rows) < limit:
            resp = self._api.get_history_candlesticks(
                instId=symbol,
                after=after,
                bar=_okx_timeframe(timeframe),
                limit=str(page_limit),
            )
            if resp.get("code") != "0":
                return ExchangeBackend.fetch_ohlcv_range(self, symbol, timeframe, since, limit)

            chunk = resp.get("data", [])
            if not chunk:
                break

            oldest_ts = None
            for candle in chunk:
                ts = int(candle[0])
                oldest_ts = ts if oldest_ts is None else min(oldest_ts, ts)
                if ts in seen:
                    continue
                seen.add(ts)
                rows.append(
                    [
                        ts,
                        float(candle[1]),
                        float(candle[2]),
                        float(candle[3]),
                        float(candle[4]),
                        float(candle[5]),
                    ]
                )

            if oldest_ts is None or oldest_ts <= since_ms or len(chunk) < page_limit:
                break
            after = str(oldest_ts)

        if not rows:
            return pl.DataFrame()

        return _parse_ohlcv(rows).filter(pl.col("timestamp") >= since_ms).tail(limit)

    def fetch_order_book(self, symbol: str, limit: int = 25) -> ObSnapshot:
        # Try OKX SDK natively first (no CCXT connection)
        try:
            resp = self._api.get_orderbook(instId=symbol, sz=str(limit))
            if resp.get("code") == "0" and resp.get("data"):
                data = resp["data"][0]
                bids = data.get("bids", [])
                asks = data.get("asks", [])
                timestamp = int(data.get("ts", "0"))
                bid_price = float(bids[0][0]) if bids else 0.0
                ask_price = float(asks[0][0]) if asks else 0.0
                return ObSnapshot(
                    timestamp=timestamp,
                    bid_price=bid_price,
                    ask_price=ask_price,
                    bid_vol_depth_5=sum(float(b[1]) for b in bids[:5]),
                    ask_vol_depth_5=sum(float(a[1]) for a in asks[:5]),
                    bid_vol_depth_25=sum(float(b[1]) for b in bids[:25]),
                    ask_vol_depth_25=sum(float(a[1]) for a in asks[:25]),
                )
        except Exception:
            pass
        # Fallback to CCXT (if available)
        return self._ensure_ccxt().fetch_order_book(symbol, limit)

    def close(self) -> None:
        if self._ccxt:
            self._ccxt.close()

    def funding_rate_history(
        self, inst_id: str = "BTC-USDT-SWAP", limit: int = 100
    ) -> pl.DataFrame:
        from okx.PublicData import PublicAPI

        pub = PublicAPI(flag="1")
        resp = pub.funding_rate_history(instId=inst_id, limit=str(limit))
        if resp.get("code") != "0":
            raise RuntimeError(f"OKX funding rate error: {resp.get('msg', resp)}")
        rows = resp.get("data", [])
        if not rows:
            return pl.DataFrame()
        return pl.DataFrame(
            [
                {
                    "timestamp": int(r["fundingTime"]),
                    "funding_rate": float(r["realizedRate"]),
                    "funding_time": int(r["fundingTime"]),
                }
                for r in rows
            ]
        ).sort("timestamp")

    def market_data_history(
        self,
        module: int,
        inst_type: str,
        date_aggr_type: str,
        begin: str,
        end: str,
        inst_id_list: list[str] | None = None,
        inst_family_list: list[str] | None = None,
    ) -> list[dict]:
        """Return downloadable historical-market-data metadata from OKX.

        The documented OKX modules include:
          - ``3`` funding rate
          - ``4`` order book depth (400-level)
          - ``5`` order book depth (5000-level)
          - ``6`` order book depth (50-level)
        """
        from okx.PublicData import PublicAPI

        pub = PublicAPI(flag="1")
        resp = pub.get_market_data_history(
            module=str(module),
            instType=inst_type,
            dateAggrType=date_aggr_type,
            begin=begin,
            end=end,
            instIdList=",".join(inst_id_list) if inst_id_list else None,
            instFamilyList=",".join(inst_family_list) if inst_family_list else None,
        )
        if resp.get("code") != "0":
            raise RuntimeError(f"OKX historical market data error: {resp.get('msg', resp)}")
        return resp.get("data", [])


# ---------------------------------------------------------------------------
# CcxtProBackend — async WebSocket overlay
# ---------------------------------------------------------------------------


class CcxtProBackend:
    """Async exchange backend for WebSocket (ccxt.pro).

    Created via ``MarketData.async_()``.  Shares the same exchange_id
    and proxy as the synchronous backend.
    """

    def __init__(self, exchange_id: str, proxy: str | None = None) -> None:
        import ccxt.pro as ccxtpro

        klass = getattr(ccxtpro, exchange_id)
        config: dict[str, Any] = {"enableRateLimit": True}
        if proxy:
            config["proxies"] = {"https": proxy, "http": proxy}
        self._ex = klass(config)

    async def watch_ob(
        self, symbol: str, limit: int = 25, params: dict[str, Any] | None = None
    ) -> AsyncIterator[ObSnapshot]:
        while True:
            book = await self._ex.watch_order_book(symbol, limit=limit, params=params or {})
            yield ObSnapshot.from_ccxt_book(book)

    async def close(self) -> None:
        await self._ex.close()


# ---------------------------------------------------------------------------
# MarketData — thin facade
# ---------------------------------------------------------------------------


class MarketData:
    """Unified market data — REST (sync) + WS (async) with same API.

    Usage — synchronous (backtesting / scripts)::

        md = MarketData("lbank")
        df = md.candles("BTC/USDT", "1d", limit=100)
        snap = md.ob_snapshot("BTC/USDT", limit=25)   # ObSnapshot

    Usage — async with WebSocket (live / paper trading)::

        md = await MarketData.async_("okx")
        async for snap in md.ob_stream("BTC/USDT"):
            print(snap.imbalance_5)
        await md.close()
    """

    def __init__(self, exchange_id: str = "okx", proxy: str | None = None) -> None:
        self._backend: ExchangeBackend
        self._async_backend: CcxtProBackend | None = None
        self._proxy = proxy
        if exchange_id == "okx":
            self._backend = OkxSdkBackend(proxy)
        else:
            self._backend = CcxtBackend(exchange_id, proxy)
        self._exchange_id = exchange_id

    @classmethod
    async def async_(cls, exchange_id: str = "okx", proxy: str | None = None) -> MarketData:
        """Create instance with async WS support."""
        md = cls(exchange_id, proxy)
        md._async_backend = CcxtProBackend(exchange_id, proxy)
        return md

    @property
    def exchange_id(self) -> str:
        return self._exchange_id

    @property
    def proxy(self) -> str | None:
        return self._proxy

    # -- OHLCV -----------------------------------------------------------

    def candles(
        self, symbol: str, timeframe: str = "1d", limit: int = 300, since: int | None = None
    ) -> pl.DataFrame:
        raw = self._backend.fetch_ohlcv(
            symbol,
            _normalize_timeframe(timeframe),
            limit=limit,
            since=since,
        )
        return _parse_ohlcv(raw)

    def candles_range(
        self, symbol: str, timeframe: str = "1d", since: str = "2020-01-01", limit: int = 3000
    ) -> pl.DataFrame:
        return self._backend.fetch_ohlcv_range(
            symbol,
            _normalize_timeframe(timeframe),
            since,
            limit,
        )

    # -- Order book ------------------------------------------------------

    def ob_snapshot(self, symbol: str, limit: int = 25) -> ObSnapshot:
        """REST order book snapshot (synchronous)."""
        return self._backend.fetch_order_book(symbol, limit=limit)

    async def ob_stream(
        self, symbol: str, limit: int = 25, params: dict[str, Any] | None = None
    ) -> AsyncIterator[ObSnapshot]:
        """WebSocket order-book stream.

        For OKX, ``params`` can pass ``{"depth": "books"}``, ``"books5"``,
        ``"books50-l2-tbt"`` or ``"bbo-tbt"`` as supported by CCXT Pro.
        """
        if self._async_backend is None:
            raise RuntimeError("Use MarketData.async_() for WebSocket access")
        async for snap in self._async_backend.watch_ob(symbol, limit=limit, params=params):
            yield snap

    # -- Funding rate ----------------------------------------------------

    def funding_rate_history(
        self, inst_id: str = "BTC-USDT-SWAP", limit: int = 100
    ) -> pl.DataFrame:
        return self._backend.funding_rate_history(inst_id, limit)

    def market_data_history(
        self,
        module: int,
        inst_type: str,
        date_aggr_type: str,
        begin: str,
        end: str,
        inst_id_list: list[str] | None = None,
        inst_family_list: list[str] | None = None,
    ) -> list[dict]:
        return self._backend.market_data_history(
            module,
            inst_type,
            date_aggr_type,
            begin,
            end,
            inst_id_list,
            inst_family_list,
        )

    # -- Lifecycle -------------------------------------------------------

    async def close(self) -> None:
        if self._async_backend:
            await self._async_backend.close()
        self._backend.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_ohlcv(raw: list[list]) -> pl.DataFrame:
    if not raw:
        return pl.DataFrame()
    return pl.DataFrame(
        [
            {
                "timestamp": int(r[0]),
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "vol": float(r[5]),
            }
            for r in raw
        ]
    ).sort("timestamp")


def _normalize_timeframe(timeframe: str) -> str:
    if timeframe.endswith("H"):
        return f"{timeframe[:-1]}h"
    if timeframe.endswith("D"):
        return f"{timeframe[:-1]}d"
    if timeframe.endswith("W"):
        return f"{timeframe[:-1]}w"
    return timeframe


def _okx_timeframe(timeframe: str) -> str:
    if timeframe.endswith("h"):
        return f"{timeframe[:-1]}H"
    if timeframe.endswith("d"):
        return f"{timeframe[:-1]}D"
    if timeframe.endswith("w"):
        return f"{timeframe[:-1]}W"
    return timeframe


def _days_since(since: str) -> int:
    return max(
        1, (datetime.now(UTC) - datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=UTC)).days
    )
