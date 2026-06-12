"""Public OKX WebSocket source helpers."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import Any

import polars as pl

from qooi.sources.manifest import manifest_frame, now_ms, source_manifest_row

OKX_WS_PUBLIC_URL = "wss://ws.okx.com:8443/ws/v5/public"


@dataclass(frozen=True)
class OkxWsCollectionResult:
    trades: pl.DataFrame
    books: pl.DataFrame
    manifest: pl.DataFrame


def okx_ws_subscribe_message(
    symbols: tuple[str, ...], *, channels: tuple[str, ...] = ("trades", "books5")
) -> dict[str, object]:
    args = [{"channel": channel, "instId": symbol} for symbol in symbols for channel in channels]
    return {"op": "subscribe", "args": args}


def normalize_okx_ws_trades(
    payload: dict[str, Any],
    *,
    contract_values: dict[str, float | None] | None = None,
) -> pl.DataFrame:
    arg = payload.get("arg") if isinstance(payload.get("arg"), dict) else {}
    symbol = str(arg.get("instId") or "")
    rows = payload.get("data") if isinstance(payload.get("data"), list) else []
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        price = _float(row.get("px"))
        size = _float(row.get("sz"))
        timestamp = _int(row.get("ts"))
        side = str(row.get("side") or "").lower()
        if price is None or size is None or timestamp is None or side not in {"buy", "sell"}:
            continue
        contract_value = (contract_values or {}).get(symbol)
        notional = price * size * contract_value if contract_value is not None else price * size
        out.append(
            {
                "symbol": symbol,
                "timestamp": timestamp,
                "trade_id": str(row.get("tradeId") or row.get("id") or timestamp),
                "price": price,
                "size": size,
                "side": side,
                "notional_usd": notional,
            }
        )
    return pl.DataFrame(
        out,
        schema={
            "symbol": pl.String,
            "timestamp": pl.Int64,
            "trade_id": pl.String,
            "price": pl.Float64,
            "size": pl.Float64,
            "side": pl.String,
            "notional_usd": pl.Float64,
        },
    )


def normalize_okx_ws_books(
    payload: dict[str, Any],
    *,
    contract_values: dict[str, float | None] | None = None,
) -> pl.DataFrame:
    arg = payload.get("arg") if isinstance(payload.get("arg"), dict) else {}
    symbol = str(arg.get("instId") or "")
    rows = payload.get("data") if isinstance(payload.get("data"), list) else []
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        bids = _levels(row.get("bids"))
        asks = _levels(row.get("asks"))
        timestamp = _int(row.get("ts"))
        if timestamp is None or not bids or not asks:
            continue
        contract_value = (contract_values or {}).get(symbol)
        bid_price = bids[0][0]
        ask_price = asks[0][0]
        out.append(
            {
                "symbol": symbol,
                "timestamp": timestamp,
                "ob_bid_price": bid_price,
                "ob_ask_price": ask_price,
                "ob_bid_vol_5": _depth_notional(bids, 5, contract_value),
                "ob_ask_vol_5": _depth_notional(asks, 5, contract_value),
                "ob_bid_vol_10": _depth_notional(bids, 10, contract_value),
                "ob_ask_vol_10": _depth_notional(asks, 10, contract_value),
                "ob_bid_vol_25": _depth_notional(bids, 25, contract_value),
                "ob_ask_vol_25": _depth_notional(asks, 25, contract_value),
                "ob_bid_vol": _depth_notional(bids, len(bids), contract_value),
                "ob_ask_vol": _depth_notional(asks, len(asks), contract_value),
                "bid_depth_bps_10": _depth_within_bps(bids, bid_price, 10.0, contract_value),
                "ask_depth_bps_10": _depth_within_bps(asks, ask_price, 10.0, contract_value),
                "bid_depth_bps_25": _depth_within_bps(bids, bid_price, 25.0, contract_value),
                "ask_depth_bps_25": _depth_within_bps(asks, ask_price, 25.0, contract_value),
                "bid_depth_bps_50": _depth_within_bps(bids, bid_price, 50.0, contract_value),
                "ask_depth_bps_50": _depth_within_bps(asks, ask_price, 50.0, contract_value),
                "spread_bps": _spread_bps(bid_price, ask_price),
            }
        )
    frame = pl.DataFrame(out, schema=_BOOK_SCHEMA)
    if frame.is_empty():
        return frame
    return frame.with_columns(
        [
            _imbalance_expr("ob_bid_vol_5", "ob_ask_vol_5", "ob_imbalance_5"),
            _imbalance_expr("ob_bid_vol_10", "ob_ask_vol_10", "ob_imbalance_10"),
            _imbalance_expr("ob_bid_vol_25", "ob_ask_vol_25", "ob_imbalance_25"),
        ]
    )


async def collect_okx_ws_public(
    symbols: tuple[str, ...],
    *,
    message_source: AsyncIterable[dict[str, Any]] | None = None,
    channels: tuple[str, ...] = ("trades", "books5"),
    contract_values: dict[str, float | None] | None = None,
    duration_seconds: float = 60.0,
    stale_after_ms: int = 10_000,
) -> OkxWsCollectionResult:
    source = message_source or _websocket_message_source(symbols, channels, duration_seconds)
    trades = []
    books = []
    errors: list[str] = []
    async for message in source:
        if message.get("event") == "error":
            errors.append(str(message.get("msg") or message.get("code") or "okx_ws_error"))
            continue
        channel = _message_channel(message)
        if channel == "trades":
            frame = normalize_okx_ws_trades(message, contract_values=contract_values)
            if not frame.is_empty():
                trades.append(frame)
        elif channel and channel.startswith("books"):
            frame = normalize_okx_ws_books(message, contract_values=contract_values)
            if not frame.is_empty():
                books.append(frame)
    trade_frame = (
        pl.concat(trades, how="vertical_relaxed") if trades else normalize_okx_ws_trades({})
    )
    book_frame = pl.concat(books, how="vertical_relaxed") if books else normalize_okx_ws_books({})
    manifest = _collection_manifest(
        symbols,
        trade_frame,
        book_frame,
        errors=errors,
        stale_after_ms=stale_after_ms,
    )
    return OkxWsCollectionResult(trade_frame, book_frame, manifest)


async def _websocket_message_source(
    symbols: tuple[str, ...], channels: tuple[str, ...], duration_seconds: float
) -> AsyncIterable[dict[str, Any]]:
    try:
        import websockets  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "websockets package is required for live OKX WebSocket collection"
        ) from exc
    subscribe = json.dumps(okx_ws_subscribe_message(symbols, channels=channels))
    deadline = asyncio.get_running_loop().time() + duration_seconds
    async with websockets.connect(OKX_WS_PUBLIC_URL) as websocket:
        await websocket.send(subscribe)
        while asyncio.get_running_loop().time() < deadline:
            timeout = max(0.0, deadline - asyncio.get_running_loop().time())
            try:
                raw = await asyncio.wait_for(websocket.recv(), timeout=timeout)
            except TimeoutError:
                break
            payload = json.loads(raw)
            if isinstance(payload, dict):
                yield payload


def _collection_manifest(
    symbols: tuple[str, ...],
    trades: pl.DataFrame,
    books: pl.DataFrame,
    *,
    errors: list[str],
    stale_after_ms: int,
) -> pl.DataFrame:
    rows = []
    now = now_ms()
    for symbol in symbols:
        rows.append(
            _source_row(
                symbol,
                "okx_ws_trades",
                trades.filter(pl.col("symbol") == symbol) if not trades.is_empty() else trades,
                errors=errors,
                now=now,
                stale_after_ms=stale_after_ms,
            )
        )
        rows.append(
            _source_row(
                symbol,
                "okx_ws_books",
                books.filter(pl.col("symbol") == symbol) if not books.is_empty() else books,
                errors=errors,
                now=now,
                stale_after_ms=stale_after_ms,
            )
        )
    return manifest_frame(rows)


def _source_row(
    symbol: str,
    source: str,
    frame: pl.DataFrame,
    *,
    errors: list[str],
    now: int,
    stale_after_ms: int,
) -> dict[str, Any]:
    range_start = _range_min(frame)
    range_end = _range_max(frame)
    stale = range_end is not None and now - range_end > stale_after_ms
    warning = ";".join([*errors, "okx_ws_stale" if stale else ""]).strip(";")
    status = (
        "failed" if errors and frame.is_empty() else "ok" if not frame.is_empty() else "missing"
    )
    if stale and status == "ok":
        status = "partial"
    return source_manifest_row(
        symbol=symbol,
        source=source,
        phase="collect-realtime",
        status=status,
        backend="okx_ws_public",
        endpoint=OKX_WS_PUBLIC_URL,
        rows=frame.height,
        range_start=range_start,
        range_end=range_end,
        warning=warning,
    )


def _message_channel(message: dict[str, Any]) -> str:
    arg = message.get("arg") if isinstance(message.get("arg"), dict) else {}
    return str(arg.get("channel") or "")


def _levels(value: Any) -> list[tuple[float, float]]:
    if not isinstance(value, list):
        return []
    levels = []
    for row in value:
        if not isinstance(row, list) or len(row) < 2:
            continue
        price = _float(row[0])
        size = _float(row[1])
        if price is not None and size is not None:
            levels.append((price, size))
    return levels


def _depth_notional(
    levels: list[tuple[float, float]], limit: int, contract_value: float | None
) -> float:
    return sum(price * size * (contract_value or 1.0) for price, size in levels[:limit])


def _depth_within_bps(
    levels: list[tuple[float, float]],
    reference_price: float,
    bps: float,
    contract_value: float | None,
) -> float:
    max_distance = reference_price * bps / 10_000.0
    return sum(
        price * size * (contract_value or 1.0)
        for price, size in levels
        if abs(price - reference_price) <= max_distance
    )


def _spread_bps(bid: float, ask: float) -> float | None:
    mid = (bid + ask) / 2.0
    return ((ask - bid) / mid * 10_000.0) if mid > 0.0 else None


def _imbalance_expr(bid_col: str, ask_col: str, alias: str) -> pl.Expr:
    return (
        pl.when((pl.col(bid_col) + pl.col(ask_col)) > 0.0)
        .then((pl.col(bid_col) - pl.col(ask_col)) / (pl.col(bid_col) + pl.col(ask_col)))
        .otherwise(None)
        .alias(alias)
    )


def _range_min(frame: pl.DataFrame) -> int | None:
    return (
        int(frame["timestamp"].min())
        if not frame.is_empty() and "timestamp" in frame.columns
        else None
    )


def _range_max(frame: pl.DataFrame) -> int | None:
    return (
        int(frame["timestamp"].max())
        if not frame.is_empty() and "timestamp" in frame.columns
        else None
    )


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


_BOOK_SCHEMA = {
    "symbol": pl.String,
    "timestamp": pl.Int64,
    "ob_bid_price": pl.Float64,
    "ob_ask_price": pl.Float64,
    "ob_bid_vol_5": pl.Float64,
    "ob_ask_vol_5": pl.Float64,
    "ob_bid_vol_10": pl.Float64,
    "ob_ask_vol_10": pl.Float64,
    "ob_bid_vol_25": pl.Float64,
    "ob_ask_vol_25": pl.Float64,
    "ob_bid_vol": pl.Float64,
    "ob_ask_vol": pl.Float64,
    "bid_depth_bps_10": pl.Float64,
    "ask_depth_bps_10": pl.Float64,
    "bid_depth_bps_25": pl.Float64,
    "ask_depth_bps_25": pl.Float64,
    "bid_depth_bps_50": pl.Float64,
    "ask_depth_bps_50": pl.Float64,
    "spread_bps": pl.Float64,
}
