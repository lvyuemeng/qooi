"""Public OKX source helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import polars as pl
from tenacity import AsyncRetrying

from qooi.exchange.market import OKX_RETRY_KWARGS, BookSnapshot
from qooi.sources.http import gather_source_results, sanitize_error
from qooi.sources.manifest import manifest_frame, source_manifest_row
from qooi.sources.models import SourceResult

OKX_BASE_URL = "https://www.okx.com"


async def fetch_okx_instruments_async(
    client: httpx.AsyncClient, inst_type: str = "SWAP"
) -> SourceResult:
    return await _fetch_okx_frame(
        client,
        endpoint="/api/v5/public/instruments",
        params={"instType": inst_type},
        source="discovery",
        symbol="*",
        normalizer=_normalize_instruments,
    )


async def fetch_okx_tickers_async(
    client: httpx.AsyncClient, inst_type: str = "SWAP"
) -> SourceResult:
    return await _fetch_okx_frame(
        client,
        endpoint="/api/v5/market/tickers",
        params={"instType": inst_type},
        source="discovery",
        symbol="*",
        normalizer=_normalize_tickers,
    )


async def fetch_okx_recent_trades_async(
    client: httpx.AsyncClient,
    inst_id: str,
    *,
    limit: int = 100,
    contract_value: float | None = None,
    contract_value_currency: str | None = None,
    contract_base_currency: str | None = None,
) -> SourceResult:
    notional_supported = _contract_value_supported(
        contract_value_currency, contract_base_currency=contract_base_currency
    )
    result = await _fetch_okx_frame(
        client,
        endpoint="/api/v5/market/trades",
        params={"instId": inst_id, "limit": str(limit)},
        source="trades",
        symbol=inst_id,
        normalizer=lambda rows: normalize_okx_trades(
            rows,
            contract_value=contract_value,
            contract_value_currency=contract_value_currency,
            contract_base_currency=contract_base_currency,
        ),
    )
    if contract_value is None or not notional_supported:
        warning = (
            "contract_metadata_missing"
            if contract_value is None
            else "unsupported_contract_value_currency"
        )
        return SourceResult(
            result.frame,
            _append_manifest_warning(result.manifest, warning, status="partial"),
        )
    return result


async def fetch_okx_funding_history_async(
    client: httpx.AsyncClient, inst_id: str, *, limit: int = 100
) -> SourceResult:
    return await _fetch_okx_frame(
        client,
        endpoint="/api/v5/public/funding-rate-history",
        params={"instId": inst_id, "limit": str(limit)},
        source="funding",
        symbol=inst_id,
        normalizer=_normalize_funding,
    )


async def fetch_okx_open_interest_async(client: httpx.AsyncClient, inst_id: str) -> SourceResult:
    return await _fetch_okx_frame(
        client,
        endpoint="/api/v5/public/open-interest",
        params={"instType": "SWAP", "instId": inst_id},
        source="open_interest",
        symbol=inst_id,
        normalizer=_normalize_open_interest,
    )


async def fetch_okx_book_snapshot_async(
    client: httpx.AsyncClient, inst_id: str, *, limit: int = 25
) -> SourceResult:
    return await _fetch_okx_frame(
        client,
        endpoint="/api/v5/market/books",
        params={"instId": inst_id, "sz": str(limit)},
        source="books",
        symbol=inst_id,
        normalizer=lambda rows: _normalize_books(rows, inst_id),
    )


async def gather_okx_source_results(
    calls: list[Callable[[httpx.AsyncClient], Awaitable[SourceResult]]],
    *,
    concurrency: int = 3,
) -> list[SourceResult]:
    return await gather_source_results(calls, base_url=OKX_BASE_URL, concurrency=concurrency)


def fetch_okx_instruments(inst_type: str = "SWAP") -> SourceResult:
    return asyncio.run(_fetch_sync(fetch_okx_instruments_async, inst_type))


def fetch_okx_tickers(inst_type: str = "SWAP") -> SourceResult:
    return asyncio.run(_fetch_sync(fetch_okx_tickers_async, inst_type))


async def _fetch_sync(func: Callable[..., Awaitable[SourceResult]], *args: Any) -> SourceResult:
    async with httpx.AsyncClient(base_url=OKX_BASE_URL, timeout=20.0) as client:
        return await func(client, *args)


async def _fetch_okx_frame(
    client: httpx.AsyncClient,
    *,
    endpoint: str,
    params: dict[str, str],
    source: str,
    symbol: str,
    normalizer: Callable[[list[dict[str, Any]]], pl.DataFrame],
) -> SourceResult:
    try:
        async for attempt in AsyncRetrying(**OKX_RETRY_KWARGS):
            with attempt:
                response = await client.get(endpoint, params=params)
                response.raise_for_status()
        payload = response.json()
        if payload.get("code") != "0":
            msg = str(payload.get("msg") or payload.get("code") or "unknown")
            return _failed_result(symbol, source, endpoint, f"okx_error_{payload.get('code')}", msg)
        rows = payload.get("data", [])
        frame = normalizer(rows)
        return SourceResult(
            frame,
            manifest_frame(
                [
                    source_manifest_row(
                        symbol=symbol,
                        source=source,
                        phase="collect-market" if source != "discovery" else "discover",
                        status="ok" if not frame.is_empty() else "missing",
                        backend="okx_rest",
                        endpoint=endpoint,
                        rows=frame.height,
                        range_start=_min_timestamp(frame),
                        range_end=_max_timestamp(frame),
                        warning="" if not frame.is_empty() else f"{source}_missing",
                    )
                ]
            ),
        )
    except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.TransportError) as exc:
        return _failed_result(symbol, source, endpoint, "http_error", sanitize_error(exc))


def normalize_okx_trades(
    rows: list[dict[str, Any]],
    *,
    contract_value: float | None = None,
    contract_value_currency: str | None = None,
    contract_base_currency: str | None = None,
) -> pl.DataFrame:
    out = []
    notional_enabled = contract_value is not None and _contract_value_supported(
        contract_value_currency, contract_base_currency=contract_base_currency
    )
    for row in rows:
        price = _float(row.get("px"))
        size = _float(row.get("sz"))
        out.append(
            {
                "timestamp": _int(row.get("ts")),
                "trade_id": str(row.get("tradeId", "")),
                "price": price,
                "size": size,
                "side": str(row.get("side", "")),
                "notional_usd": price * size * float(contract_value) if notional_enabled else None,
            }
        )
    return pl.DataFrame(out).sort("timestamp") if out else pl.DataFrame()


def _normalize_instruments(rows: list[dict[str, Any]]) -> pl.DataFrame:
    out = []
    for row in rows:
        inst_id = str(row.get("instId", ""))
        inferred_base, inferred_quote = _infer_base_quote(inst_id)
        out.append(
            {
                "inst_id": inst_id,
                "symbol": inst_id,
                "inst_type": str(row.get("instType", "")),
                "state": str(row.get("state", "")),
                "base_ccy": str(row.get("baseCcy") or inferred_base),
                "quote_ccy": str(row.get("quoteCcy") or inferred_quote),
                "settle_ccy": str(row.get("settleCcy", "")),
                "ct_val": _float_or_none(row.get("ctVal")),
                "ct_val_ccy": str(row.get("ctValCcy", "")),
                "list_time": _int_or_none(row.get("listTime")),
            }
        )
    return pl.DataFrame(out) if out else pl.DataFrame()


def _normalize_tickers(rows: list[dict[str, Any]]) -> pl.DataFrame:
    out = []
    for row in rows:
        bid = _float_or_none(row.get("bidPx"))
        ask = _float_or_none(row.get("askPx"))
        mid = (bid + ask) / 2.0 if bid and ask else None
        out.append(
            {
                "inst_id": str(row.get("instId", "")),
                "quote_volume_24h": _float_or_none(row.get("volCcy24h")),
                "last": _float_or_none(row.get("last")),
                "bid_px": bid,
                "ask_px": ask,
                "spread_bps": ((ask - bid) / mid * 10_000.0)
                if mid and ask is not None and bid is not None
                else None,
            }
        )
    return pl.DataFrame(out) if out else pl.DataFrame()


def _normalize_funding(rows: list[dict[str, Any]]) -> pl.DataFrame:
    out = [
        {
            "timestamp": _int(row.get("fundingTime")),
            "funding_time": _int(row.get("fundingTime")),
            "funding_rate": _float(row.get("fundingRate")),
        }
        for row in rows
    ]
    return pl.DataFrame(out).sort("timestamp") if out else pl.DataFrame()


def _normalize_open_interest(rows: list[dict[str, Any]]) -> pl.DataFrame:
    out = [
        {
            "timestamp": _int(row.get("ts")),
            "open_interest": _float(row.get("oi")),
            "open_interest_ccy": _float_or_none(row.get("oiCcy")),
        }
        for row in rows
    ]
    return pl.DataFrame(out).sort("timestamp") if out else pl.DataFrame()


def _normalize_books(rows: list[dict[str, Any]], inst_id: str) -> pl.DataFrame:
    out = []
    for row in rows:
        book = BookSnapshot.from_okx_book(row).to_row()
        book["symbol"] = inst_id
        out.append(book)
    return pl.DataFrame(out).sort("timestamp") if out else pl.DataFrame()


def _failed_result(
    symbol: str, source: str, endpoint: str, stop_reason: str, warning: str
) -> SourceResult:
    return SourceResult(
        pl.DataFrame(),
        manifest_frame(
            [
                source_manifest_row(
                    symbol=symbol,
                    source=source,
                    phase="collect-market" if source != "discovery" else "discover",
                    status="failed",
                    backend="okx_rest",
                    endpoint=endpoint,
                    warning=warning,
                    stop_reason=stop_reason,
                )
            ]
        ),
    )


def _append_manifest_warning(manifest: pl.DataFrame, warning: str, *, status: str) -> pl.DataFrame:
    if manifest.is_empty():
        return manifest
    return manifest.with_columns(
        [
            pl.lit(status).alias("status"),
            pl.concat_str([pl.col("warning").fill_null(""), pl.lit(warning)], separator=";")
            .str.strip_chars(";")
            .alias("warning"),
        ]
    )


def _min_timestamp(frame: pl.DataFrame) -> int | None:
    return (
        int(frame["timestamp"].min())
        if "timestamp" in frame.columns and not frame.is_empty()
        else None
    )


def _max_timestamp(frame: pl.DataFrame) -> int | None:
    return (
        int(frame["timestamp"].max())
        if "timestamp" in frame.columns and not frame.is_empty()
        else None
    )


def _float(value: Any) -> float:
    return float(value or 0.0)


def _float_or_none(value: Any) -> float | None:
    return None if value in {None, ""} else float(value)


def _int(value: Any) -> int:
    return int(value or 0)


def _int_or_none(value: Any) -> int | None:
    return None if value in {None, ""} else int(value)


def _contract_value_supported(
    contract_value_currency: str | None, *, contract_base_currency: str | None = None
) -> bool:
    currency = (contract_value_currency or "").upper()
    base = (contract_base_currency or "").upper()
    return currency in {"USD", "USDT"} or bool(base and currency == base)


def _infer_base_quote(inst_id: str) -> tuple[str, str]:
    parts = inst_id.split("-")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "", ""
