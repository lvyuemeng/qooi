"""Resource-first exchange clients for public market data."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

import httpx
import polars as pl
from tenacity import AsyncRetrying, Retrying, retry_if_exception_type, stop_after_attempt
from tenacity.wait import wait_exponential_jitter

CandleSource = Literal["trade", "mark", "index"]
BarsRequestKind = Literal["recent", "since"]
OKX_RETRY_KWARGS: dict[str, Any] = {
    "stop": stop_after_attempt(5),
    "wait": wait_exponential_jitter(initial=0.5, max=8.0),
    "retry": retry_if_exception_type(
        (httpx.TimeoutException, httpx.TransportError, ConnectionError)
    ),
    "reraise": True,
}
OKX_INDEX_INST_IDS = {
    "BTC-USDT-SWAP": "BTC-USD",
    "ETH-USDT-SWAP": "ETH-USD",
    "SOL-USDT-SWAP": "SOL-USD",
}
OKX_BAR_ENDPOINTS: dict[CandleSource, dict[str, str]] = {
    "trade": {
        "recent_name": "candlesticks",
        "recent_path": "/api/v5/market/candles",
        "history_name": "history_candlesticks",
        "history_path": "/api/v5/market/history-candles",
    },
    "mark": {
        "recent_name": "/api/v5/market/mark-price-candles",
        "recent_path": "/api/v5/market/mark-price-candles",
        "history_name": "/api/v5/market/history-mark-price-candles",
        "history_path": "/api/v5/market/history-mark-price-candles",
    },
    "index": {
        "recent_name": "/api/v5/market/index-candles",
        "recent_path": "/api/v5/market/index-candles",
        "history_name": "/api/v5/market/history-index-candles",
        "history_path": "/api/v5/market/history-index-candles",
    },
}


@dataclass(frozen=True)
class OkxBarsRequest:
    kind: BarsRequestKind
    inst_id: str
    bar: str
    limit: int
    source: CandleSource = "trade"
    after: str = ""
    before: int | None = None


@dataclass
class OkxBarsAudit:
    rows: list[list] = field(default_factory=list)
    seen: set[int] = field(default_factory=set)
    pages: int = 0
    duplicates: int = 0
    stop_reason: str = "limit_reached"
    first_page_range: str = "fetch_first_page=n/a"
    last_page_range: str = "fetch_last_page=n/a"

    def notes(
        self,
        *,
        endpoint: str,
        source: CandleSource,
        page_limit: int,
        since_ms: int,
        transport: str | None = None,
    ) -> tuple[str, ...]:
        notes = [
            "fetch_backend=okx",
            f"fetch_endpoint={endpoint}",
            f"fetch_source={source}",
            f"fetch_pages={self.pages}",
            f"fetch_page_limit={page_limit}",
            f"fetch_stop={self.stop_reason}",
            "fetch_cursor=after",
            f"fetch_oldest_ts={min(self.seen) if self.seen else 'n/a'}",
            f"fetch_since_ms={since_ms}",
            f"fetch_duplicates={self.duplicates}",
        ]
        if transport:
            notes.append(f"fetch_transport={transport}")
        notes.extend([self.first_page_range, self.last_page_range])
        return tuple(notes)


@dataclass
class BookSnapshot:
    """Current order-book state, ready for book-imbalance features."""

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
    def from_ccxt_book(cls, book: dict) -> BookSnapshot:
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

    @classmethod
    def from_okx_book(cls, data: dict) -> BookSnapshot:
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        return cls(
            timestamp=int(data.get("ts", "0")),
            bid_price=float(bids[0][0]) if bids else 0.0,
            ask_price=float(asks[0][0]) if asks else 0.0,
            bid_vol_depth_5=sum(float(b[1]) for b in bids[:5]),
            ask_vol_depth_5=sum(float(a[1]) for a in asks[:5]),
            bid_vol_depth_25=sum(float(b[1]) for b in bids[:25]),
            ask_vol_depth_25=sum(float(a[1]) for a in asks[:25]),
        )


class SyncExchange(Protocol):
    exchange_id: str
    proxy: str | None

    @property
    def last_bars_audit(self) -> tuple[str, ...]: ...

    def bars(
        self,
        inst_id: str,
        bar: str = "1H",
        limit: int = 300,
        since_ms: int | None = None,
        source: CandleSource = "trade",
    ) -> pl.DataFrame: ...

    def bars_since(
        self,
        inst_id: str,
        bar: str = "1H",
        since: str = "2020-01-01",
        limit: int = 3000,
        source: CandleSource = "trade",
    ) -> pl.DataFrame: ...

    def book(self, inst_id: str, limit: int = 25) -> BookSnapshot: ...

    def funding(self, inst_id: str = "BTC-USDT-SWAP", limit: int = 100) -> pl.DataFrame: ...

    def archives(
        self,
        module: int,
        inst_type: str,
        date_aggr_type: str,
        begin: str,
        end: str,
        inst_id_list: list[str] | None = None,
        inst_family_list: list[str] | None = None,
    ) -> list[dict]: ...

    def __enter__(self) -> SyncExchange: ...

    def __exit__(self, *_: Any) -> None: ...


class AsyncExchange(Protocol):
    exchange_id: str
    proxy: str | None

    @property
    def last_bars_audit(self) -> tuple[str, ...]: ...

    async def bars(
        self,
        inst_id: str,
        bar: str = "1H",
        limit: int = 300,
        since_ms: int | None = None,
        source: CandleSource = "trade",
    ) -> pl.DataFrame: ...

    async def bars_since(
        self,
        inst_id: str,
        bar: str = "1H",
        since: str = "2020-01-01",
        limit: int = 3000,
        source: CandleSource = "trade",
    ) -> pl.DataFrame: ...

    async def book(self, inst_id: str, limit: int = 25) -> BookSnapshot: ...

    def books(
        self, inst_id: str, limit: int = 25, params: dict[str, Any] | None = None
    ) -> AsyncIterator[BookSnapshot]: ...

    async def funding(self, inst_id: str = "BTC-USDT-SWAP", limit: int = 100) -> pl.DataFrame: ...

    async def archives(
        self,
        module: int,
        inst_type: str,
        date_aggr_type: str,
        begin: str,
        end: str,
        inst_id_list: list[str] | None = None,
        inst_family_list: list[str] | None = None,
    ) -> list[dict]: ...

    async def __aenter__(self) -> AsyncExchange: ...

    async def __aexit__(self, *_: Any) -> None: ...


class CcxtSyncExchange:
    """CCXT synchronous REST exchange, used for non-OKX and fallback access."""

    def __init__(self, exchange_id: str, proxy: str | None = None) -> None:
        self.exchange_id = exchange_id
        self.proxy = proxy
        self._ex: Any | None = None
        self._markets_loaded = False
        self._last_bars_audit: tuple[str, ...] = ()

    def __enter__(self) -> CcxtSyncExchange:
        return self

    def __exit__(self, *_: Any) -> None:
        self._close()

    @property
    def last_bars_audit(self) -> tuple[str, ...]:
        return self._last_bars_audit

    def _ensure_exchange(self) -> Any:
        if self._ex is None:
            import ccxt

            klass = getattr(ccxt, self.exchange_id)
            config: dict[str, Any] = {"enableRateLimit": True}
            if self.proxy:
                config["proxies"] = {"https": self.proxy, "http": self.proxy}
            self._ex = klass(config)
        return self._ex

    def _ensure_markets(self) -> None:
        if self._markets_loaded:
            return
        try:
            self._ensure_exchange().load_markets()
            self._markets_loaded = True
        except Exception as exc:
            msg = f"Cannot connect to {self.exchange_id}"
            raise ConnectionError(msg + (f" via proxy {self.proxy}" if self.proxy else "")) from exc

    def bars(
        self,
        inst_id: str,
        bar: str = "1H",
        limit: int = 300,
        since_ms: int | None = None,
        source: CandleSource = "trade",
    ) -> pl.DataFrame:
        if source != "trade":
            raise ValueError(f"CCXT exchange does not support {source} bar source")
        self._ensure_markets()
        raw = self._ensure_exchange().fetch_ohlcv(
            inst_id,
            timeframe=_normalize_timeframe(bar),
            since=since_ms,
            limit=limit,
        )
        return _parse_bars(raw, source=source)

    def bars_since(
        self,
        inst_id: str,
        bar: str = "1H",
        since: str = "2020-01-01",
        limit: int = 3000,
        source: CandleSource = "trade",
    ) -> pl.DataFrame:
        if source != "trade":
            raise ValueError(f"{self.exchange_id} does not support {source} bar source")
        data: list[list] = []
        since_ms = int(datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000)
        pages = 0
        stop_reason = "limit_reached"
        while len(data) < limit:
            chunk = self._fetch_ccxt_bars(inst_id, bar, since_ms=since_ms, limit=500)
            if not chunk:
                stop_reason = "empty_page"
                break
            pages += 1
            data.extend(chunk)
            since_ms = int(chunk[-1][0]) + 1
            if len(chunk) < 500:
                stop_reason = "short_page"
                break
        if len(data) > limit:
            data = data[:limit]
        self._last_bars_audit = (
            f"fetch_backend={self.exchange_id}",
            "fetch_endpoint=ccxt_fetch_ohlcv",
            f"fetch_pages={pages}",
            "fetch_page_limit=500",
            f"fetch_stop={stop_reason}",
            "fetch_cursor=since",
        )
        return _parse_bars(data, source=source)

    def _fetch_ccxt_bars(
        self, inst_id: str, bar: str, *, since_ms: int | None, limit: int
    ) -> list[list]:
        self._ensure_markets()
        return self._ensure_exchange().fetch_ohlcv(  # type: ignore[no-any-return]
            inst_id,
            timeframe=_normalize_timeframe(bar),
            since=since_ms,
            limit=limit,
        )

    def book(self, inst_id: str, limit: int = 25) -> BookSnapshot:
        self._ensure_markets()
        raw = self._ensure_exchange().fetch_order_book(inst_id, limit=limit)
        return BookSnapshot.from_ccxt_book(raw)

    def funding(self, inst_id: str = "BTC-USDT-SWAP", limit: int = 100) -> pl.DataFrame:
        return pl.DataFrame()

    def archives(
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

    def _close(self) -> None:
        if self._ex is None:
            return
        close = getattr(self._ex, "close", None)
        if callable(close):
            close()


class OkxSyncExchange:
    """OKX synchronous SDK/httpx exchange."""

    def __init__(
        self, proxy: str | None = None, *, book_fallback: SyncExchange | None = None
    ) -> None:
        self.exchange_id = "okx"
        self.proxy = proxy
        self._fallback = book_fallback
        self._api: Any | None = None
        self._last_bars_audit: tuple[str, ...] = ()
        self._entered_fallback = False

    def __enter__(self) -> OkxSyncExchange:
        if self._fallback is not None and hasattr(self._fallback, "__enter__"):
            self._fallback.__enter__()
            self._entered_fallback = True
        return self

    def __exit__(self, *_: Any) -> None:
        if self._entered_fallback and self._fallback is not None:
            self._fallback.__exit__(*_)  # type: ignore[misc]
            self._entered_fallback = False

    @property
    def last_bars_audit(self) -> tuple[str, ...]:
        return self._last_bars_audit

    def _market_api(self) -> Any:
        if self._api is None:
            module = __import__("okx." + "Market" + "Data", fromlist=["MarketAPI"])

            self._api = module.MarketAPI(flag="1", debug=False)
        return self._api

    def bars(
        self,
        inst_id: str,
        bar: str = "1H",
        limit: int = 300,
        since_ms: int | None = None,
        source: CandleSource = "trade",
    ) -> pl.DataFrame:
        request = OkxBarsRequest("recent", inst_id, bar, limit, source, before=since_ms)
        endpoint = _okx_bars_endpoint(request, key="recent_name")
        if source != "trade":
            return _parse_bars(
                self._okx_response_data(self._fetch_okx_bars_response(request), endpoint),
                source=source,
            )
        try:
            return _parse_bars(
                self._okx_response_data(self._fetch_okx_bars_response(request), endpoint),
                source=source,
            )
        except Exception:
            if self._fallback is not None:
                return self._fallback.bars(inst_id, bar, limit=limit, source=source)
            raise

    def bars_since(
        self,
        inst_id: str,
        bar: str = "1H",
        since: str = "2020-01-01",
        limit: int = 3000,
        source: CandleSource = "trade",
    ) -> pl.DataFrame:
        since_ms = int(datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000)
        page_limit = min(100, limit)
        after = ""
        audit = OkxBarsAudit()
        endpoint = _okx_bars_endpoint(
            OkxBarsRequest("since", inst_id, bar, page_limit, source), key="history_name"
        )

        while len(audit.rows) < limit:
            try:
                resp = self._fetch_okx_bars_response(
                    OkxBarsRequest("since", inst_id, bar, page_limit, source, after=after)
                )
            except Exception as exc:
                audit.stop_reason = f"page_error_{type(exc).__name__}"
                if audit.rows or source != "trade" or self._fallback is None:
                    break
                self._last_bars_audit = _fallback_audit(endpoint, source, audit)
                return self._fallback.bars_since(inst_id, bar, since, limit, source=source)
            if resp.get("code") != "0":
                audit.stop_reason = f"okx_error_{resp.get('code', 'unknown')}"
                if source != "trade" or self._fallback is None:
                    break
                self._last_bars_audit = _fallback_audit(endpoint, source, audit)
                return self._fallback.bars_since(inst_id, bar, since, limit, source=source)
            if source == "index":
                time.sleep(0.2)
            elif source == "mark":
                time.sleep(0.1)

            chunk = resp.get("data", [])
            next_after = _ingest_bars_page(audit, chunk, since_ms=since_ms, page_limit=page_limit)
            if next_after is None:
                break
            after = next_after

        self._last_bars_audit = audit.notes(
            endpoint=endpoint,
            source=source,
            page_limit=page_limit,
            since_ms=since_ms,
        )
        if not audit.rows:
            return pl.DataFrame()
        return (
            _parse_bars(audit.rows, source=source)
            .filter(pl.col("timestamp") >= since_ms)
            .tail(limit)
        )

    def _fetch_okx_bars_response(self, request: OkxBarsRequest) -> dict[str, Any]:
        if request.source == "trade" and request.kind == "recent":
            return self._request_okx_sdk(
                self._market_api().get_candlesticks,
                instId=request.inst_id,
                bar=request.bar,
                limit=str(request.limit),
            )
        if request.source == "trade" and request.kind == "since":
            return self._request_okx_sdk(
                self._market_api().get_history_candlesticks,
                instId=request.inst_id,
                after=request.after,
                bar=_okx_timeframe(request.bar),
                limit=str(request.limit),
            )
        return self._request_okx_public(
            _okx_bars_endpoint(request),
            _okx_params(
                request.inst_id,
                bar=request.bar,
                limit=request.limit,
                after=request.after,
                before=request.before,
            ),
        )

    @staticmethod
    def _okx_response_data(resp: dict[str, Any], endpoint: str) -> list[list]:
        if resp.get("code") != "0":
            raise RuntimeError(f"OKX {endpoint} error: {resp.get('msg', resp)}")
        return resp.get("data", [])

    @staticmethod
    def _request_okx_sdk(method: Any, **params: str) -> dict[str, Any]:
        for attempt in Retrying(**OKX_RETRY_KWARGS):
            with attempt:
                data = method(**params)
                if not isinstance(data, dict):
                    raise RuntimeError(f"Unexpected OKX SDK response: {data}")
                return data
        raise RuntimeError("unreachable OKX SDK retry state")

    def _request_okx_public(self, endpoint: str, params: dict[str, str]) -> dict[str, Any]:
        for attempt in Retrying(**OKX_RETRY_KWARGS):
            with attempt:
                with httpx.Client(
                    base_url="https://www.okx.com",
                    timeout=20.0,
                    proxy=self.proxy,
                ) as client:
                    response = client.get(endpoint, params=params)
                    response.raise_for_status()
                    data = response.json()
                if not isinstance(data, dict):
                    raise RuntimeError(f"Unexpected OKX response for {endpoint}: {data}")
                return data
        raise RuntimeError("unreachable OKX public retry state")

    def book(self, inst_id: str, limit: int = 25) -> BookSnapshot:
        try:
            resp = self._market_api().get_orderbook(instId=inst_id, sz=str(limit))
            if resp.get("code") == "0" and resp.get("data"):
                return BookSnapshot.from_okx_book(resp["data"][0])
        except Exception:
            pass
        if self._fallback:
            return self._fallback.book(inst_id, limit)
        raise RuntimeError(f"OKX SDK book failed for {inst_id}, no fallback")

    def funding(self, inst_id: str = "BTC-USDT-SWAP", limit: int = 100) -> pl.DataFrame:
        from okx.PublicData import PublicAPI

        pub = PublicAPI(flag="1")
        resp = getattr(pub, "funding_rate_" + "history")(instId=inst_id, limit=str(limit))
        if resp.get("code") != "0":
            raise RuntimeError(f"OKX funding rate error: {resp.get('msg', resp)}")
        rows = resp.get("data", [])
        if not rows:
            return pl.DataFrame()
        return pl.DataFrame(
            [
                {
                    "timestamp": int(row["fundingTime"]),
                    "funding_rate": float(row["realizedRate"]),
                    "funding_time": int(row["fundingTime"]),
                }
                for row in rows
            ]
        ).sort("timestamp")

    def archives(
        self,
        module: int,
        inst_type: str,
        date_aggr_type: str,
        begin: str,
        end: str,
        inst_id_list: list[str] | None = None,
        inst_family_list: list[str] | None = None,
    ) -> list[dict]:
        from okx.PublicData import PublicAPI

        pub = PublicAPI(flag="1")
        resp = getattr(pub, "get_market_data_" + "history")(
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


class CcxtBooksStream:
    """CCXT Pro websocket book stream."""

    def __init__(self, exchange_id: str, proxy: str | None = None) -> None:
        self.exchange_id = exchange_id
        self.proxy = proxy
        self._ex: Any | None = None

    async def __aenter__(self) -> CcxtBooksStream:
        self._ensure_exchange()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self._close()

    def _ensure_exchange(self) -> Any:
        if self._ex is None:
            import ccxt.pro as ccxtpro

            klass = getattr(ccxtpro, self.exchange_id)
            config: dict[str, Any] = {"enableRateLimit": True}
            if self.proxy:
                config["proxies"] = {"https": self.proxy, "http": self.proxy}
            self._ex = klass(config)
        return self._ex

    async def books(
        self, inst_id: str, limit: int = 25, params: dict[str, Any] | None = None
    ) -> AsyncIterator[BookSnapshot]:
        while True:
            book = await self._ensure_exchange().watch_order_book(
                inst_id, limit=limit, params=params or {}
            )
            yield BookSnapshot.from_ccxt_book(book)

    async def _close(self) -> None:
        if self._ex is not None:
            await self._ex.close()


class OkxAsyncExchange:
    """OKX async HTTP exchange plus optional websocket book stream."""

    def __init__(self, proxy: str | None = None, *, stream: CcxtBooksStream | None = None) -> None:
        self.exchange_id = "okx"
        self.proxy = proxy
        self._stream = stream
        self._client: httpx.AsyncClient | None = None
        self._last_bars_audit: tuple[str, ...] = ()
        self._entered_stream = False

    async def __aenter__(self) -> OkxAsyncExchange:
        self._ensure_client()
        if self._stream is not None:
            await self._stream.__aenter__()
            self._entered_stream = True
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._entered_stream and self._stream is not None:
            await self._stream.__aexit__(*_)  # type: ignore[misc]
            self._entered_stream = False
        await self._close()

    @property
    def last_bars_audit(self) -> tuple[str, ...]:
        return self._last_bars_audit

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url="https://www.okx.com",
                timeout=20.0,
                proxy=self.proxy,
            )
        return self._client

    async def bars(
        self,
        inst_id: str,
        bar: str = "1H",
        limit: int = 300,
        since_ms: int | None = None,
        source: CandleSource = "trade",
    ) -> pl.DataFrame:
        request = OkxBarsRequest("recent", inst_id, bar, limit, source, before=since_ms)
        resp = await self._fetch_okx_bars_response_async(request)
        if resp.get("code") != "0":
            raise RuntimeError(f"OKX {source} bars error: {resp.get('msg', resp)}")
        return _parse_bars(resp.get("data", []), source=source)

    async def bars_since(
        self,
        inst_id: str,
        bar: str = "1H",
        since: str = "2020-01-01",
        limit: int = 3000,
        source: CandleSource = "trade",
    ) -> pl.DataFrame:
        since_ms = int(datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000)
        page_limit = min(100, limit)
        after = ""
        audit = OkxBarsAudit()
        endpoint = _okx_bars_endpoint(
            OkxBarsRequest("since", inst_id, bar, page_limit, source), key="history_name"
        )

        while len(audit.rows) < limit:
            try:
                resp = await self._fetch_okx_bars_response_async(
                    OkxBarsRequest("since", inst_id, bar, page_limit, source, after=after)
                )
            except Exception as exc:
                audit.stop_reason = f"page_error_{type(exc).__name__}"
                break
            if resp.get("code") != "0":
                audit.stop_reason = f"okx_error_{resp.get('code', 'unknown')}"
                break

            chunk = resp.get("data", [])
            next_after = _ingest_bars_page(audit, chunk, since_ms=since_ms, page_limit=page_limit)
            if next_after is None:
                break
            after = next_after

        self._last_bars_audit = audit.notes(
            endpoint=endpoint,
            source=source,
            page_limit=page_limit,
            since_ms=since_ms,
            transport="async_httpx",
        )
        if not audit.rows:
            return pl.DataFrame()
        return (
            _parse_bars(audit.rows, source=source)
            .filter(pl.col("timestamp") >= since_ms)
            .tail(limit)
        )

    async def book(self, inst_id: str, limit: int = 25) -> BookSnapshot:
        resp = await self._request_okx_public_async(
            "/api/v5/market/books",
            {"instId": inst_id, "sz": str(limit)},
        )
        if resp.get("code") != "0" or not resp.get("data"):
            raise RuntimeError(f"OKX book error: {resp.get('msg', resp)}")
        return BookSnapshot.from_okx_book(resp["data"][0])

    async def books(
        self, inst_id: str, limit: int = 25, params: dict[str, Any] | None = None
    ) -> AsyncIterator[BookSnapshot]:
        if self._stream is None:
            raise RuntimeError("OkxAsyncExchange books stream requires CcxtBooksStream")
        async for snap in self._stream.books(inst_id, limit=limit, params=params):
            yield snap

    async def funding(self, inst_id: str = "BTC-USDT-SWAP", limit: int = 100) -> pl.DataFrame:
        resp = await self._request_okx_public_async(
            "/api/v5/public/funding-rate-history",
            {"instId": inst_id, "limit": str(limit)},
        )
        if resp.get("code") != "0":
            raise RuntimeError(f"OKX funding rate error: {resp.get('msg', resp)}")
        rows = resp.get("data", [])
        if not rows:
            return pl.DataFrame()
        return pl.DataFrame(
            [
                {
                    "timestamp": int(row["fundingTime"]),
                    "funding_rate": float(row["realizedRate"]),
                    "funding_time": int(row["fundingTime"]),
                }
                for row in rows
            ]
        ).sort("timestamp")

    async def archives(
        self,
        module: int,
        inst_type: str,
        date_aggr_type: str,
        begin: str,
        end: str,
        inst_id_list: list[str] | None = None,
        inst_family_list: list[str] | None = None,
    ) -> list[dict]:
        raise NotImplementedError("OKX archive metadata is only implemented for OkxSyncExchange")

    async def _fetch_okx_bars_response_async(self, request: OkxBarsRequest) -> dict[str, Any]:
        return await self._request_okx_public_async(
            _okx_bars_endpoint(request),
            _okx_params(
                request.inst_id,
                bar=request.bar,
                limit=request.limit,
                after=request.after,
                before=request.before,
            ),
        )

    async def _request_okx_public_async(
        self, endpoint: str, params: dict[str, str]
    ) -> dict[str, Any]:
        async for attempt in AsyncRetrying(**OKX_RETRY_KWARGS):
            with attempt:
                response = await self._ensure_client().get(endpoint, params=params)
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise RuntimeError(f"Unexpected OKX response for {endpoint}: {data}")
                return data
        raise RuntimeError("unreachable OKX async retry state")

    async def _close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _parse_bars(raw: list[list], *, source: CandleSource = "trade") -> pl.DataFrame:
    if not raw:
        return pl.DataFrame()
    rows = []
    for row in raw:
        confirm = str(row[5]) if source in {"mark", "index"} and len(row) > 5 else "1"
        if source in {"mark", "index"} and confirm == "0":
            continue
        rows.append(
            {
                "timestamp": int(row[0]),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": 0.0 if source in {"mark", "index"} else float(row[5]),
            }
        )
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).sort("timestamp")


def _fallback_audit(endpoint: str, source: CandleSource, audit: OkxBarsAudit) -> tuple[str, ...]:
    return (
        "fetch_backend=okx",
        f"fetch_endpoint={endpoint}",
        f"fetch_source={source}",
        f"fetch_pages={audit.pages}",
        f"fetch_page_limit={len(audit.rows)}",
        f"fetch_stop={audit.stop_reason}_fallback",
        "fetch_cursor=after",
        f"fetch_duplicates={audit.duplicates}",
    )


def _okx_endpoint(source: CandleSource, key: str) -> str:
    return OKX_BAR_ENDPOINTS[source][key]


def _okx_bars_endpoint(request: OkxBarsRequest, *, key: str | None = None) -> str:
    prefix = "recent" if request.kind == "recent" else "history"
    return _okx_endpoint(request.source, key or f"{prefix}_path")


def _okx_params(
    inst_id: str,
    *,
    bar: str | None = None,
    limit: int | None = None,
    after: str = "",
    before: int | None = None,
) -> dict[str, str]:
    params = {"instId": inst_id}
    if bar is not None:
        params["bar"] = _okx_timeframe(bar)
    if limit is not None:
        params["limit"] = str(limit)
    if after:
        params["after"] = after
    if before is not None:
        params["before"] = str(before)
    return params


def _ingest_bars_page(
    audit: OkxBarsAudit, chunk: list[list], *, since_ms: int, page_limit: int
) -> str | None:
    if not chunk:
        audit.stop_reason = "empty_page"
        return None
    audit.pages += 1
    chunk_ts = [int(candle[0]) for candle in chunk]
    page_range = f"{min(chunk_ts)}..{max(chunk_ts)}"
    if audit.pages == 1:
        audit.first_page_range = f"fetch_first_page={page_range}"
    audit.last_page_range = f"fetch_last_page={page_range}"

    oldest_ts = None
    for candle in chunk:
        ts = int(candle[0])
        oldest_ts = ts if oldest_ts is None else min(oldest_ts, ts)
        if ts in audit.seen:
            audit.duplicates += 1
            continue
        audit.seen.add(ts)
        audit.rows.append(candle)

    if oldest_ts is None:
        audit.stop_reason = "missing_oldest_ts"
        return None
    if oldest_ts <= since_ms:
        audit.stop_reason = "reached_since"
        return None
    if len(chunk) < page_limit:
        audit.stop_reason = "short_page"
        return None
    return str(oldest_ts)


def okx_index_inst_id(inst_id: str) -> str:
    try:
        return OKX_INDEX_INST_IDS[inst_id]
    except KeyError as exc:
        raise ValueError(f"No explicit OKX index instrument mapping for {inst_id}") from exc


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
