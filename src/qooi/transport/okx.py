"""OKX transport — OkxClient + WebSocket + bar parsing."""

# ruff: noqa: E501, E701, E702
from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, TypedDict

import httpx
import polars as pl
import websockets
from tenacity import AsyncRetrying
from websockets.asyncio.client import ClientConnection

from qooi.transport.core import BaseHttpClient, HttpError, RetryPolicy, request_json, sanitize_error

OKX_BASE_URL = "https://www.okx.com"
OKX_WS_PUBLIC_URL = "wss://ws.okx.com:8443/ws/v5/public"


class SourceManifestRow(TypedDict):
    timestamp: int
    symbol: str
    source: str
    phase: str
    status: str
    backend: str
    endpoint: str
    rows: int
    range_start: int | None
    range_end: int | None
    coverage_pct: float | None
    warning: str
    stop_reason: str


@dataclass(frozen=True)
class SourceResult:
    frame: pl.DataFrame = field(default_factory=pl.DataFrame)
    manifest: pl.DataFrame = field(default_factory=pl.DataFrame)
    telemetry: pl.DataFrame = field(default_factory=pl.DataFrame)


@dataclass(frozen=True)
class Manifest:
    schema: dict[str, pl.DataType]

    def frame(self, rows: list[SourceManifestRow]) -> pl.DataFrame:
        if not rows:
            return pl.DataFrame(schema=self.schema)
        frame = pl.DataFrame(rows)
        for col, dtype in self.schema.items():
            if col not in frame.columns:
                frame = frame.with_columns(pl.lit(None).cast(dtype).alias(col))
        return frame.select(pl.col(col).cast(dtype) for col, dtype in self.schema.items())

    def row(
        self,
        *,
        symbol: str,
        source: str,
        phase: str,
        status: str,
        backend: str = "",
        endpoint: str = "",
        rows: int = 0,
        range_start: int | None = None,
        range_end: int | None = None,
        coverage_pct: float | None = None,
        warning: str = "",
        stop_reason: str = "",
        timestamp: int | None = None,
    ) -> SourceManifestRow:
        return {
            "timestamp": timestamp if timestamp is not None else now_ms(),
            "symbol": symbol,
            "source": source,
            "phase": phase,
            "status": status,
            "backend": backend,
            "endpoint": endpoint,
            "rows": rows,
            "range_start": range_start,
            "range_end": range_end,
            "coverage_pct": coverage_pct,
            "warning": warning,
            "stop_reason": stop_reason,
        }


TRANSPORT_MANIFEST = Manifest(
    {
        "timestamp": pl.Int64,
        "symbol": pl.String,
        "source": pl.String,
        "phase": pl.String,
        "status": pl.String,
        "backend": pl.String,
        "endpoint": pl.String,
        "rows": pl.Int64,
        "range_start": pl.Int64,
        "range_end": pl.Int64,
        "coverage_pct": pl.Float64,
        "warning": pl.String,
        "stop_reason": pl.String,
    }
)


def now_ms() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)


# ── helpers ──


def _retryable_okx_exception(exc: BaseException) -> bool:
    if isinstance(exc, HttpError):
        return exc.category in {"rate_limited", "timeout_or_too_broad", "transport_error"}
    return isinstance(exc, httpx.TimeoutException | httpx.TransportError | ConnectionError)


def okx_retry_policy() -> RetryPolicy:
    return RetryPolicy(retry_on=_retryable_okx_exception)


def _okx_error_category(payload: dict[str, Any]) -> str | None:
    return None if payload.get("code") == "0" else "api_error"


def _okx_timeframe(timeframe: str) -> str:
    if timeframe.endswith("h"):
        return f"{timeframe[:-1]}H"
    if timeframe.endswith("d"):
        return f"{timeframe[:-1]}D"
    if timeframe.endswith("w"):
        return f"{timeframe[:-1]}W"
    return timeframe


# ── Bar parsing ──


def _parse_bars(raw: list[list], *, source: str = "trade") -> pl.DataFrame:
    if not raw:
        return pl.DataFrame()
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    df = pl.DataFrame([row[:6] for row in raw], schema=columns, orient="row")
    if source in {"mark", "index"}:
        df = df.with_columns(pl.col("volume").fill_null(0.0))
    return df.with_columns(
        pl.col("timestamp").cast(pl.Int64),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
    ).sort("timestamp")


# ── Normalize ──


def _rename_snake(df: pl.DataFrame, mapping: dict[str, str]) -> pl.DataFrame:
    for old, new in mapping.items():
        if old in df.columns:
            df = df.rename({old: new})
    return df


def _normalize_funding(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(schema={"funding_rate": pl.Float64, "funding_time": pl.Int64})
    frame = _rename_snake(frame, {"fundingRate": "funding_rate", "fundingTime": "funding_time"})
    if "funding_rate" not in frame.columns and "funding_time" not in frame.columns:
        return frame
    if "funding_time" not in frame.columns:
        frame = frame.with_columns(pl.lit(0).alias("funding_time"))
    keep = (
        ["timestamp", "funding_rate", "funding_time", "realized_rate"]
        if "realized_rate" in frame.columns
        else ["timestamp", "funding_rate", "funding_time"]
    )
    return frame.select([c for c in keep if c in frame.columns]).sort(
        "funding_time" if "funding_time" in frame.columns else "timestamp"
    )


def _normalize_instruments(data: list[dict[str, Any]]) -> pl.DataFrame:
    frame = _rename_snake(
        pl.DataFrame(data) if data else pl.DataFrame(),
        {
            "instId": "inst_id",
            "instType": "inst_type",
            "baseCcy": "base_ccy",
            "quoteCcy": "quote_ccy",
            "settleCcy": "settle_ccy",
            "ctVal": "ct_val",
            "ctValCcy": "ct_val_ccy",
            "listTime": "list_time",
        },
    )
    if frame.is_empty():
        return frame
    return frame.with_columns(
        pl.col("ct_val").cast(pl.Float64, strict=False),
        pl.col("list_time").cast(pl.Int64, strict=False),
    )


def _normalize_tickers(data: list[dict[str, Any]]) -> pl.DataFrame:
    frame = _rename_snake(
        pl.DataFrame(data) if data else pl.DataFrame(),
        {
            "instId": "inst_id",
            "volCcy24h": "quote_volume_24h",
            "bidPx": "bid_px",
            "askPx": "ask_px",
        },
    )
    if frame.is_empty():
        return frame
    frame = frame.with_columns(
        pl.col("quote_volume_24h").cast(pl.Float64, strict=False),
        pl.col("bid_px").cast(pl.Float64, strict=False),
        pl.col("ask_px").cast(pl.Float64, strict=False),
        pl.col("last").cast(pl.Float64, strict=False),
    )
    return frame.with_columns(
        pl.when(((pl.col("bid_px") + pl.col("ask_px")) / 2.0) > 0)
        .then(
            (pl.col("ask_px") - pl.col("bid_px"))
            / ((pl.col("bid_px") + pl.col("ask_px")) / 2.0)
            * 10_000.0
        )
        .otherwise(None)
        .alias("spread_bps")
    )


def _float_or_none(value: object) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _int_or_zero(value: object) -> int:
    try:
        return int(value) if value not in (None, "") else 0
    except (TypeError, ValueError):
        return 0


def _rubik_value(row: object, key: str, index: int) -> object:
    if isinstance(row, dict):
        return row.get(key)
    if isinstance(row, list | tuple) and len(row) > index:
        return row[index]
    return None


def _normalize_open_interest(data: list[object]) -> pl.DataFrame:
    rows = [
        {
            "timestamp": _int_or_zero(_rubik_value(row, "ts", 0)),
            "open_interest": _float_or_none(_rubik_value(row, "oi", 1)),
            "open_interest_ccy": _float_or_none(_rubik_value(row, "oiCcy", 2)),
            "open_interest_usd": _float_or_none(_rubik_value(row, "oiUsd", 3)),
        }
        for row in data
    ]
    if not rows:
        return pl.DataFrame()
    return (
        pl.DataFrame(rows)
        .with_columns(
            pl.col("timestamp").cast(pl.Int64, strict=False),
            pl.col("open_interest").cast(pl.Float64, strict=False),
            pl.col("open_interest_ccy").cast(pl.Float64, strict=False),
            pl.col("open_interest_usd").cast(pl.Float64, strict=False),
        )
        .sort("timestamp")
    )


def _normalize_taker_volume(data: list[object], unit: str) -> pl.DataFrame:
    rows = [
        {
            "timestamp": _int_or_zero(_rubik_value(row, "ts", 0)),
            "taker_sell_volume": _float_or_none(_rubik_value(row, "sellVol", 1)),
            "taker_buy_volume": _float_or_none(_rubik_value(row, "buyVol", 2)),
            "taker_volume_unit": unit,
        }
        for row in data
    ]
    if not rows:
        return pl.DataFrame()
    return (
        pl.DataFrame(rows)
        .with_columns(
            pl.col("timestamp").cast(pl.Int64, strict=False),
            pl.col("taker_sell_volume").cast(pl.Float64, strict=False),
            pl.col("taker_buy_volume").cast(pl.Float64, strict=False),
        )
        .sort("timestamp")
    )


def _normalize_long_short_ratio(data: list[object]) -> pl.DataFrame:
    rows = [
        {
            "timestamp": _int_or_zero(_rubik_value(row, "ts", 0)),
            "long_short_account_ratio": _float_or_none(_rubik_value(row, "ratio", 1)),
        }
        for row in data
    ]
    if not rows:
        return pl.DataFrame()
    return (
        pl.DataFrame(rows)
        .with_columns(
            pl.col("timestamp").cast(pl.Int64, strict=False),
            pl.col("long_short_account_ratio").cast(pl.Float64, strict=False),
        )
        .sort("timestamp")
    )


# ═══════════════════════════════════════════════════════════════════════════
# OkxClient — all OKX data access
# ═══════════════════════════════════════════════════════════════════════════


class OkxClient(BaseHttpClient):
    """All OKX data access. Context manager: async with OkxClient() as okx: ..."""

    def __init__(self, *, timeout: float = 20.0, proxy: str | None = None) -> None:
        super().__init__(OKX_BASE_URL, timeout=timeout, proxy=proxy)

    async def request(
        self,
        endpoint: str,
        params: dict[str, str] | None = None,
        *,
        error_classifier: Callable[[dict[str, Any]], str | None] | None = None,
    ) -> dict[str, Any]:
        retry_kwargs = okx_retry_policy().to_kwargs()
        classified_error = False
        try:
            async for attempt in AsyncRetrying(**retry_kwargs):
                with attempt:
                    payload = await request_json(
                        self.client,
                        endpoint,
                        params=params,
                        error_classifier=error_classifier or _okx_error_category,
                    )
                    return payload
        except HttpError:
            classified_error = True
            raise
        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.TransportError) as exc:
            if not classified_error:
                raise HttpError("transport_error", type(exc).__name__, endpoint=endpoint) from None
            raise

    # ── Private fetch ──

    async def _fetch_frame(
        self,
        *,
        endpoint: str,
        params: dict[str, str] | None = None,
        source: str = "",
        symbol: str = "*",
        normalizer: Callable | None = None,
    ) -> SourceResult:
        try:
            raw = await self.request(endpoint, params=params)
            data = raw.get("data", [])
            frame_data = data if isinstance(data, list) else ([data] if data else [])
            frame = (
                normalizer(frame_data)
                if normalizer
                else pl.DataFrame(frame_data)
                if frame_data
                else pl.DataFrame()
            )
            return SourceResult(
                frame,
                TRANSPORT_MANIFEST.frame(
                    [
                        TRANSPORT_MANIFEST.row(
                            symbol=symbol,
                            source=source,
                            phase="collect-market",
                            status="ok" if not frame.is_empty() else "missing",
                            backend="okx",
                            endpoint=endpoint,
                            rows=frame.height,
                            warning="" if not frame.is_empty() else f"{source}_empty",
                        )
                    ]
                ),
            )
        except Exception as exc:
            return SourceResult(
                pl.DataFrame(),
                TRANSPORT_MANIFEST.frame(
                    [
                        TRANSPORT_MANIFEST.row(
                            symbol=symbol,
                            source=source,
                            phase="collect-market",
                            status="failed",
                            backend="okx",
                            endpoint=endpoint,
                            warning=sanitize_error(exc),
                            stop_reason="http_error",
                        )
                    ]
                ),
            )

    # ── Bars ──

    async def bars(
        self, inst_id: str, *, bar: str = "1H", limit: int = 300, source: str = "trade"
    ) -> pl.DataFrame:
        resp = await self.request(
            "/api/v5/market/candles",
            params={"instId": inst_id, "bar": _okx_timeframe(bar), "limit": str(limit)},
        )
        return _parse_bars(resp.get("data", []), source=source)

    async def history_candles(
        self,
        inst_id: str,
        *,
        bar: str = "1H",
        limit: int = 100,
        after: str | None = None,
        source: str = "trade",
    ) -> pl.DataFrame:
        params = {"instId": inst_id, "bar": _okx_timeframe(bar), "limit": str(min(100, limit))}
        if after is not None:
            params["after"] = after
        resp = await self.request("/api/v5/market/history-candles", params=params)
        return _parse_bars(resp.get("data", []), source=source)

    async def bars_since(
        self,
        inst_id: str,
        *,
        bar: str = "1H",
        since: str = "2020-01-01",
        limit: int = 3000,
        source: str = "trade",
    ) -> pl.DataFrame:
        since_ms = int(datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000)
        page_limit = min(100, limit)
        after = ""
        all_rows: list[list] = []
        seen: set[int] = set()

        while len(all_rows) < limit:
            page = await self.history_candles(
                inst_id,
                bar=bar,
                limit=page_limit,
                after=after or None,
                source=source,
            )
            if page.is_empty():
                break
            chunk = page.select("timestamp", "open", "high", "low", "close", "volume").rows()
            if not chunk:
                break
            for candle in chunk:
                ts = int(candle[0])
                if ts not in seen:
                    seen.add(ts)
                    all_rows.append(list(candle))
            oldest = min(int(c[0]) for c in chunk)
            if oldest <= since_ms or len(chunk) < page_limit:
                break
            after = str(oldest)

        return _parse_bars(all_rows, source=source)

    # ── Sources ──

    async def book_snapshot(self, symbol: str, *, limit: int = 5) -> SourceResult:
        return await self._fetch_frame(
            endpoint="/api/v5/market/books",
            params={"instId": symbol, "sz": str(limit)},
            source="books",
            symbol=symbol,
            normalizer=lambda d: pl.DataFrame(d[0]) if d else pl.DataFrame(),
        )

    async def recent_trades(self, symbol: str, *, limit: int = 100) -> SourceResult:
        return await self._fetch_frame(
            endpoint="/api/v5/market/trades",
            params={"instId": symbol, "limit": str(limit)},
            source="trades",
            symbol=symbol,
        )

    async def funding_history(
        self, symbol: str, *, after: str | None = None, limit: int = 100
    ) -> SourceResult:
        params: dict[str, str] = {"instId": symbol, "limit": str(limit)}
        if after:
            params["after"] = after
        return await self._fetch_frame(
            endpoint="/api/v5/public/funding-rate-history",
            params=params,
            source="funding",
            symbol=symbol,
            normalizer=lambda d: _normalize_funding(pl.DataFrame(d)),
        )

    async def funding_rate(self, symbol: str) -> SourceResult:
        return await self._fetch_frame(
            endpoint="/api/v5/public/funding-rate",
            params={"instId": symbol},
            source="funding_current",
            symbol=symbol,
            normalizer=lambda d: pl.DataFrame(
                [
                    {
                        "symbol": symbol,
                        "timestamp": int(d[0].get("fundingTime", 0) if d else 0) or now_ms(),
                        "funding_rate": float(d[0].get("fundingRate", 0) if d else 0),
                    }
                ]
            ),
        )

    async def open_interest(
        self,
        symbol: str,
        *,
        period: str = "1H",
        limit: int = 100,
        end: str | None = None,
    ) -> SourceResult:
        params = {"instId": symbol, "period": period, "limit": str(limit)}
        if end is not None:
            params["end"] = end
        return await self._fetch_frame(
            endpoint="/api/v5/rubik/stat/contracts/open-interest-history",
            params=params,
            source="open_interest_history",
            symbol=symbol,
            normalizer=_normalize_open_interest,
        )

    async def taker_volume(
        self,
        symbol: str,
        *,
        period: str = "1H",
        unit: str = "1",
        limit: int = 100,
        end: str | None = None,
    ) -> SourceResult:
        params = {"instId": symbol, "period": period, "unit": unit, "limit": str(limit)}
        if end is not None:
            params["end"] = end
        return await self._fetch_frame(
            endpoint="/api/v5/rubik/stat/taker-volume-contract",
            params=params,
            source="taker_volume_contract",
            symbol=symbol,
            normalizer=lambda data: _normalize_taker_volume(data, unit),
        )

    async def long_short_ratio(
        self,
        symbol: str,
        *,
        period: str = "1H",
        limit: int = 100,
        end: str | None = None,
    ) -> SourceResult:
        params = {"instId": symbol, "period": period, "limit": str(limit)}
        if end is not None:
            params["end"] = end
        return await self._fetch_frame(
            endpoint="/api/v5/rubik/stat/contracts/long-short-account-ratio-contract",
            params=params,
            source="long_short_ratio_contract",
            symbol=symbol,
            normalizer=_normalize_long_short_ratio,
        )

    async def instruments(self, inst_type: str = "SWAP") -> SourceResult:
        return await self._fetch_frame(
            endpoint="/api/v5/public/instruments",
            params={"instType": inst_type},
            source="discovery",
            symbol="*",
            normalizer=_normalize_instruments,
        )

    async def tickers(self, inst_type: str = "SWAP") -> SourceResult:
        return await self._fetch_frame(
            endpoint="/api/v5/market/tickers",
            params={"instType": inst_type},
            source="discovery",
            symbol="*",
            normalizer=_normalize_tickers,
        )


# ═══════════════════════════════════════════════════════════════════════════
# WebSocket
# ═══════════════════════════════════════════════════════════════════════════


class OkxWsClient:
    def __init__(self) -> None:
        self._ws: ClientConnection | None = None

    async def connect(self, url: str = OKX_WS_PUBLIC_URL) -> None:
        self._ws = await websockets.connect(url)

    async def subscribe(self, channels: tuple[str, ...], symbols: tuple[str, ...]) -> None:
        if self._ws is None:
            raise RuntimeError("Not connected")
        args = [{"channel": ch, "instId": sym} for sym in symbols for ch in channels]
        await self._ws.send(json.dumps({"op": "subscribe", "args": args}))

    async def messages(self) -> AsyncIterator[dict[str, Any]]:
        if self._ws is None:
            raise RuntimeError("Not connected")
        async for raw in self._ws:
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                continue

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None


async def collect_okx_ws_books(
    symbols: tuple[str, ...], *, max_samples: int = 0, channel: str = "books5"
) -> pl.DataFrame:
    client = OkxWsClient()
    await client.connect(OKX_WS_PUBLIC_URL)
    await client.subscribe((channel,), symbols)
    rows: list[dict] = []
    async for msg in client.messages():
        data = msg.get("data", [])
        for row in data if isinstance(data, list) else []:
            if not isinstance(row, dict):
                continue
            bids = row.get("bids", [])
            asks = row.get("asks", [])
            rows.append(
                {
                    "timestamp": int(row.get("ts", 0)),
                    "ob_bid_price": float(bids[0][0]) if bids else 0.0,
                    "ob_ask_price": float(asks[0][0]) if asks else 0.0,
                    "ob_bid_vol_5": sum(float(b[1]) for b in bids[:5]),
                    "ob_ask_vol_5": sum(float(a[1]) for a in asks[:5]),
                    "ob_bid_vol_25": sum(float(b[1]) for b in bids[:25]),
                    "ob_ask_vol_25": sum(float(a[1]) for a in asks[:25]),
                }
            )
        if max_samples and len(rows) >= max_samples:
            break
    await client.close()
    return pl.DataFrame(rows) if rows else pl.DataFrame()
