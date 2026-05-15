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
from pathlib import Path
from typing import Any, Protocol

import polars as pl

# ---------------------------------------------------------------------------
# Protocols — contracts for backend providers
# ---------------------------------------------------------------------------


class OhlcvProvider(Protocol):
    """Any backend that can fetch OHLCV candles."""

    def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1d", limit: int = 500, since: int | None = None
    ) -> list[list]: ...

    def fetch_ohlcv_range(
        self, symbol: str, timeframe: str = "1d", since: str = "2020-01-01", limit: int = 3000
    ) -> pl.DataFrame: ...


class OrderBookProvider(Protocol):
    """Any backend that can fetch order book snapshots and OHLCV (fallback)."""

    def fetch_order_book(self, symbol: str, limit: int = 25) -> ObSnapshot: ...

    def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1d", limit: int = 500, since: int | None = None
    ) -> list[list]: ...

    def close(self) -> None: ...


class StreamProvider(Protocol):
    """Any backend that can stream order book via WebSocket."""

    async def watch_ob(
        self, symbol: str, limit: int = 25, params: dict[str, Any] | None = None
    ) -> AsyncIterator[ObSnapshot]: ...

    async def close(self) -> None: ...


class FundingRateProvider(Protocol):
    """Any backend that can fetch funding rate history."""

    def funding_rate_history(
        self, inst_id: str = "BTC-USDT-SWAP", limit: int = 100
    ) -> pl.DataFrame: ...


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
        self._last_ohlcv_audit: tuple[str, ...] = ()

    @property
    def exchange_id(self) -> str:
        return self._exchange_id

    def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1d", limit: int = 500, since: int | None = None
    ) -> list[list]:
        raise NotImplementedError

    def fetch_ohlcv_range(
        self, symbol: str, timeframe: str = "1d", since: str = "2020-01-01", limit: int = 3000
    ) -> pl.DataFrame:
        data: list[list] = []
        since_ms = int(datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000)
        pages = 0
        stop_reason = "limit_reached"
        while len(data) < limit:
            chunk = self.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=500)
            if not chunk:
                stop_reason = "empty_page"
                break
            pages += 1
            data.extend(chunk)
            since_ms = chunk[-1][0] + 1
            if len(chunk) < 500:
                stop_reason = "short_page"
                break
        if len(data) > limit:
            data = data[:limit]
        self._last_ohlcv_audit = (
            f"fetch_backend={self.exchange_id}",
            "fetch_endpoint=ccxt_fetch_ohlcv",
            f"fetch_pages={pages}",
            "fetch_page_limit=500",
            f"fetch_stop={stop_reason}",
            "fetch_cursor=since",
        )
        return _parse_ohlcv(data)

    def fetch_order_book(self, symbol: str, limit: int = 25) -> ObSnapshot:
        raise NotImplementedError

    @property
    def last_ohlcv_audit(self) -> tuple[str, ...]:
        return self._last_ohlcv_audit

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
    """CCXT synchronous REST backend — lazy initialisation.

    ``load_markets()`` is deferred until first use to avoid connection
    errors at construction time (especially when CCXT is only a fallback
    and the primary backend handles all operations).
    """

    def __init__(self, exchange_id: str, proxy: str | None = None) -> None:
        super().__init__(exchange_id, proxy)
        import ccxt

        klass = getattr(ccxt, exchange_id)
        config: dict[str, Any] = {"enableRateLimit": True}
        if proxy:
            config["proxies"] = {"https": proxy, "http": proxy}
        self._ex = klass(config)
        self._markets_loaded = False

    def _ensure_markets(self) -> None:
        if self._markets_loaded:
            return
        try:
            self._ex.load_markets()
            self._markets_loaded = True
        except Exception as e:
            msg = f"Cannot connect to {self._exchange_id}"
            raise ConnectionError(msg + (f" via proxy {self._proxy}" if self._proxy else "")) from e

    def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1d", limit: int = 500, since: int | None = None
    ) -> list[list]:
        self._ensure_markets()
        return self._ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)  # type: ignore[no-any-return]

    def fetch_order_book(self, symbol: str, limit: int = 25) -> ObSnapshot:
        self._ensure_markets()
        raw = self._ex.fetch_order_book(symbol, limit=limit)
        return ObSnapshot.from_ccxt_book(raw)

    def close(self) -> None:
        close = getattr(self._ex, "close", None)
        if callable(close):
            close()


class OkxSdkBackend(ExchangeBackend):
    """OKX native Python SDK backend.

    OHLCV:   OKX SDK directly (fast, no CCXT).
    Order book: OKX SDK first, falls back to provided OrderBookProvider.
    Funding rate: OKX PublicData API.
    """

    def __init__(
        self, proxy: str | None = None, *, order_book: OrderBookProvider | None = None
    ) -> None:
        super().__init__("okx", proxy)
        from okx.MarketData import MarketAPI

        self._api = MarketAPI(flag="1", debug=False)
        # Composition: OB fallback passed explicitly, not lazily created.
        self._ob_fallback: OrderBookProvider | None = order_book

    def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1d", limit: int = 500, since: int | None = None
    ) -> list[list]:
        try:
            resp = self._api.get_candlesticks(instId=symbol, bar=timeframe, limit=str(limit))
            if resp.get("code") != "0":
                raise RuntimeError(f"OKX SDK error: {resp.get('msg', resp)}")
            return [
                [int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])]
                for r in resp.get("data", [])
            ]
        except Exception:
            pass
        if self._ob_fallback:
            return self._ob_fallback.fetch_ohlcv(symbol, timeframe, limit=limit)
        raise RuntimeError(f"OKX SDK failed for {symbol}, no fallback configured")

    def fetch_ohlcv_range(
        self, symbol: str, timeframe: str = "1d", since: str = "2020-01-01", limit: int = 3000
    ) -> pl.DataFrame:
        """Use OKX native history candles for contiguous deep pagination."""
        since_ms = int(datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000)
        page_limit = min(100, limit)
        after = ""
        rows: list[list] = []
        seen: set[int] = set()
        pages = 0
        duplicate_count = 0
        stop_reason = "limit_reached"
        first_page_range = "fetch_first_page=n/a"
        last_page_range = "fetch_last_page=n/a"

        while len(rows) < limit:
            try:
                resp = self._api.get_history_candlesticks(
                    instId=symbol,
                    after=after,
                    bar=_okx_timeframe(timeframe),
                    limit=str(page_limit),
                )
            except Exception as exc:
                stop_reason = f"page_error_{type(exc).__name__}"
                if rows:
                    break
                self._last_ohlcv_audit = (
                    "fetch_backend=okx",
                    "fetch_endpoint=history_candlesticks",
                    f"fetch_pages={pages}",
                    f"fetch_page_limit={page_limit}",
                    f"fetch_stop={stop_reason}_fallback",
                    "fetch_cursor=after",
                    f"fetch_duplicates={duplicate_count}",
                )
                return ExchangeBackend.fetch_ohlcv_range(self, symbol, timeframe, since, limit)
            if resp.get("code") != "0":
                self._last_ohlcv_audit = (
                    "fetch_backend=okx",
                    "fetch_endpoint=history_candlesticks",
                    f"fetch_pages={pages}",
                    f"fetch_page_limit={page_limit}",
                    "fetch_stop=sdk_error_fallback",
                    "fetch_cursor=after",
                    f"fetch_duplicates={duplicate_count}",
                )
                return ExchangeBackend.fetch_ohlcv_range(self, symbol, timeframe, since, limit)

            chunk = resp.get("data", [])
            if not chunk:
                stop_reason = "empty_page"
                break
            pages += 1
            chunk_ts = [int(candle[0]) for candle in chunk]
            page_range = f"{min(chunk_ts)}..{max(chunk_ts)}"
            if pages == 1:
                first_page_range = f"fetch_first_page={page_range}"
            last_page_range = f"fetch_last_page={page_range}"

            oldest_ts = None
            for candle in chunk:
                ts = int(candle[0])
                oldest_ts = ts if oldest_ts is None else min(oldest_ts, ts)
                if ts in seen:
                    duplicate_count += 1
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
                if oldest_ts is None:
                    stop_reason = "missing_oldest_ts"
                elif oldest_ts <= since_ms:
                    stop_reason = "reached_since"
                else:
                    stop_reason = "short_page"
                break
            after = str(oldest_ts)

        self._last_ohlcv_audit = (
            "fetch_backend=okx",
            "fetch_endpoint=history_candlesticks",
            f"fetch_pages={pages}",
            f"fetch_page_limit={page_limit}",
            f"fetch_stop={stop_reason}",
            "fetch_cursor=after",
            f"fetch_oldest_ts={min(seen) if seen else 'n/a'}",
            f"fetch_since_ms={since_ms}",
            f"fetch_duplicates={duplicate_count}",
            first_page_range,
            last_page_range,
        )
        if not rows:
            return pl.DataFrame()

        return _parse_ohlcv(rows).filter(pl.col("timestamp") >= since_ms).tail(limit)

    def fetch_order_book(self, symbol: str, limit: int = 25) -> ObSnapshot:
        try:
            resp = self._api.get_orderbook(instId=symbol, sz=str(limit))
            if resp.get("code") == "0" and resp.get("data"):
                data = resp["data"][0]
                bids = data.get("bids", [])
                asks = data.get("asks", [])
                timestamp = int(data.get("ts", "0"))
                return ObSnapshot(
                    timestamp=timestamp,
                    bid_price=float(bids[0][0]) if bids else 0.0,
                    ask_price=float(asks[0][0]) if asks else 0.0,
                    bid_vol_depth_5=sum(float(b[1]) for b in bids[:5]),
                    ask_vol_depth_5=sum(float(a[1]) for a in asks[:5]),
                    bid_vol_depth_25=sum(float(b[1]) for b in bids[:25]),
                    ask_vol_depth_25=sum(float(a[1]) for a in asks[:25]),
                )
        except Exception:
            pass
        if self._ob_fallback:
            return self._ob_fallback.fetch_order_book(symbol, limit)
        raise RuntimeError(f"OKX SDK order book failed for {symbol}, no fallback")

    def close(self) -> None:
        if self._ob_fallback is not None:
            try:
                self._ob_fallback.close()
            except AttributeError:
                pass

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
    """Unified market data — context manager for sync and async use.

    Usage — synchronous (backtesting / scripts)::

        with MarketData("lbank") as md:
            df = md.candles("BTC/USDT", "1d", limit=100)
            snap = md.ob_snapshot("BTC/USDT", limit=25)

    Usage — async with WebSocket (live / paper trading)::

        async with MarketData("okx") as md:
            async for snap in md.ob_stream("BTC/USDT"):
                print(snap.imbalance_5)

    Backward-compatible API (without context manager)::

        md = MarketData("okx")
        df = md.candles("BTC-USDT-SWAP", "4h", limit=500)
    """

    # Registry: exchange_id → backend factory.
    # Add new exchanges here without touching __init__.
    _registry: dict[str, type[ExchangeBackend]] = {
        "okx": OkxSdkBackend,
    }
    _fallback: type[ExchangeBackend] = CcxtBackend

    def __init__(self, exchange_id: str = "okx", proxy: str | None = None) -> None:
        self._backend: ExchangeBackend
        self._async_backend: CcxtProBackend | None = None
        self._proxy = proxy

        backend_cls = self._registry.get(exchange_id, self._fallback)
        if backend_cls is OkxSdkBackend:
            # OKX gets a CCXT fallback for order book via composition.
            self._backend = OkxSdkBackend(proxy, order_book=CcxtBackend("okx", proxy))
        else:
            self._backend = backend_cls(exchange_id, proxy)
        self._exchange_id = exchange_id

    # -- context manager (sync) ------------------------------------------
    def __enter__(self) -> MarketData:
        return self

    def __exit__(self, *_: Any) -> None:
        self._backend.close()

    # -- async context manager (replaces async_ classmethod) -------------
    async def __aenter__(self) -> MarketData:
        self._async_backend = CcxtProBackend(self._exchange_id, self._proxy)
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._async_backend:
            await self._async_backend.close()
        self._backend.close()

    # -- compat: kept for existing callers, delegates to __aenter__ ------
    @classmethod
    async def async_(cls, exchange_id: str = "okx", proxy: str | None = None) -> MarketData:
        """Create instance with async WS support.  Prefer ``async with``."""
        md = cls(exchange_id, proxy)
        return await md.__aenter__()

    @property
    def exchange_id(self) -> str:
        return self._exchange_id

    @property
    def proxy(self) -> str | None:
        return self._proxy

    @property
    def last_ohlcv_audit(self) -> tuple[str, ...]:
        return self._backend.last_ohlcv_audit

    # -- OHLCV -----------------------------------------------------------

    def candles(
        self,
        symbol: str,
        timeframe: str = "1d",
        limit: int = 300,
        since: int | None = None,
        cache: bool = False,
    ) -> pl.DataFrame:
        raw = self._backend.fetch_ohlcv(
            symbol,
            _normalize_timeframe(timeframe),
            limit=limit,
            since=since,
        )
        df = _parse_ohlcv(raw)

        if cache and not df.is_empty():
            cache_path = _cache_path(symbol, timeframe)
            if cache_path.exists():
                cached = pl.read_parquet(cache_path)
                for col in cached.columns:
                    if col not in df.columns:
                        df = df.with_columns(pl.lit(None).cast(cached[col].dtype).alias(col))
                df = df.select(cached.columns)
                merged = pl.concat([cached, df]).unique(subset=["timestamp"]).sort("timestamp")
                merged.write_parquet(cache_path)
            else:
                df.write_parquet(cache_path)

        return df

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


def _cache_path(symbol: str, timeframe: str) -> Path:
    """Parquet cache path: data/cache/{S}_{T}.parquet"""
    cache_dir = Path(__file__).resolve().parent.parent.parent.parent / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe = symbol.replace("-", "_").replace("/", "_").upper()
    tf = timeframe.replace(" ", "").upper()
    return cache_dir / f"{safe}_{tf}.parquet"
