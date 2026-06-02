"""Public Polymarket source helpers."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

import httpx
import polars as pl

from qooi.sources.http import sanitize_error
from qooi.sources.manifest import manifest_frame, now_ms, source_manifest_row
from qooi.sources.models import SourceResult

POLYMARKET_GAMMA_BASE_URL = "https://gamma-api.polymarket.com"


async def fetch_polymarket_search_async(
    client: httpx.AsyncClient,
    query: str,
    *,
    symbol: str,
    limit_per_type: int = 10,
) -> SourceResult:
    return await _fetch_polymarket_frame(
        client,
        endpoint="/public-search",
        params={"q": query, "limit_per_type": str(limit_per_type), "search_profiles": "false"},
        source="polymarket_markets",
        symbol=symbol,
        normalizer=lambda payload: normalize_polymarket_search_results(
            payload, symbol=symbol, query=query
        ),
    )


async def fetch_polymarket_events_async(
    client: httpx.AsyncClient,
    *,
    symbol: str,
    query: str = "",
    limit: int = 25,
    active: bool = True,
    closed: bool = False,
) -> SourceResult:
    return await _fetch_polymarket_frame(
        client,
        endpoint="/events",
        params={"limit": str(limit), "active": str(active).lower(), "closed": str(closed).lower()},
        source="polymarket_events",
        symbol=symbol,
        normalizer=lambda payload: normalize_polymarket_events(
            _payload_rows(payload), symbol=symbol, query=query
        ),
    )


async def fetch_polymarket_markets_async(
    client: httpx.AsyncClient,
    *,
    symbol: str,
    query: str = "",
    limit: int = 25,
    closed: bool = False,
) -> SourceResult:
    return await _fetch_polymarket_frame(
        client,
        endpoint="/markets",
        params={"limit": str(limit), "closed": str(closed).lower()},
        source="polymarket_markets",
        symbol=symbol,
        normalizer=lambda payload: normalize_polymarket_markets(
            _payload_rows(payload), symbol=symbol, query=query
        ),
    )


async def _fetch_polymarket_frame(
    client: httpx.AsyncClient,
    *,
    endpoint: str,
    params: dict[str, str],
    source: str,
    symbol: str,
    normalizer: Callable[[Any], pl.DataFrame],
) -> SourceResult:
    try:
        response = await client.get(endpoint, params=params)
        response.raise_for_status()
        frame = normalizer(response.json())
        return SourceResult(
            frame,
            manifest_frame(
                [
                    source_manifest_row(
                        symbol=symbol,
                        source=source,
                        phase="collect-context",
                        status="ok" if not frame.is_empty() else "missing",
                        backend="polymarket_gamma",
                        endpoint=endpoint,
                        rows=frame.height,
                        range_start=_min_timestamp(frame),
                        range_end=_max_timestamp(frame),
                        warning="" if not frame.is_empty() else "polymarket_unmatched",
                    )
                ]
            ),
        )
    except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.TransportError) as exc:
        return SourceResult(
            pl.DataFrame(),
            manifest_frame(
                [
                    source_manifest_row(
                        symbol=symbol,
                        source=source,
                        phase="collect-context",
                        status="failed",
                        backend="polymarket_gamma",
                        endpoint=endpoint,
                        warning=sanitize_error(exc),
                        stop_reason="http_error",
                    )
                ]
            ),
        )


def normalize_polymarket_search_results(
    payload: dict[str, Any], *, symbol: str, query: str
) -> pl.DataFrame:
    markets = []
    for event in payload.get("events") or []:
        for market in event.get("markets") or []:
            row = _market_row(market, symbol=symbol, query=query)
            row["event_id"] = str(event.get("id") or row["event_id"])
            row["category"] = str(event.get("category") or row["category"] or "")
            markets.append(row)
    return pl.DataFrame(markets) if markets else pl.DataFrame()


def normalize_polymarket_events(
    rows: list[dict[str, Any]], *, symbol: str, query: str
) -> pl.DataFrame:
    ts = now_ms()
    out = []
    for row in rows:
        markets = row.get("markets") or []
        out.append(
            {
                "symbol": symbol,
                "timestamp": ts,
                "query": query,
                "provider": "polymarket_gamma",
                "event_id": str(row.get("id") or ""),
                "slug": str(row.get("slug") or ""),
                "title": str(row.get("title") or ""),
                "description": str(row.get("description") or ""),
                "category": str(row.get("category") or ""),
                "active": _bool_or_none(row.get("active")),
                "closed": _bool_or_none(row.get("closed")),
                "start_time": _iso_ms(row.get("startDate") or row.get("createdAt")),
                "end_time": _iso_ms(row.get("endDate") or row.get("closedTime")),
                "volume_24h": _float_or_none(row.get("volume24hr")),
                "volume_1w": _float_or_none(row.get("volume1wk")),
                "volume_1mo": _float_or_none(row.get("volume1mo")),
                "volume_total": _float_or_none(row.get("volume")),
                "liquidity": _float_or_none(row.get("liquidity")),
                "open_interest": _float_or_none(row.get("openInterest")),
                "market_count": len(markets),
                "comment_count": _int_or_none(row.get("commentCount")),
                "matched_alias": query,
                "match_method": "query_match" if query else "unmatched",
                "url": f"https://polymarket.com/event/{row.get('slug')}" if row.get("slug") else "",
                "data_quality_warning": "" if query else "polymarket_alias_missing",
            }
        )
    return pl.DataFrame(out) if out else pl.DataFrame()


def normalize_polymarket_markets(
    rows: list[dict[str, Any]], *, symbol: str, query: str
) -> pl.DataFrame:
    out = [_market_row(row, symbol=symbol, query=query) for row in rows]
    return pl.DataFrame(out) if out else pl.DataFrame()


def _market_row(row: dict[str, Any], *, symbol: str, query: str) -> dict[str, Any]:
    yes_price, no_price = _outcome_prices(row.get("outcomePrices"))
    return {
        "symbol": symbol,
        "timestamp": now_ms(),
        "query": query,
        "provider": "polymarket_gamma",
        "market_id": str(row.get("id") or ""),
        "event_id": _first_event_id(row),
        "slug": str(row.get("slug") or ""),
        "question": str(row.get("question") or ""),
        "description": str(row.get("description") or ""),
        "category": str(row.get("category") or ""),
        "active": _bool_or_none(row.get("active")),
        "closed": _bool_or_none(row.get("closed")),
        "start_time": _iso_ms(row.get("startDate") or row.get("startDateIso")),
        "end_time": _iso_ms(row.get("endDate") or row.get("endDateIso")),
        "volume_24h": _float_or_none(row.get("volume24hr")),
        "volume_1w": _float_or_none(row.get("volume1wk")),
        "volume_1mo": _float_or_none(row.get("volume1mo")),
        "volume_total": _float_or_none(row.get("volumeNum") or row.get("volume")),
        "liquidity": _float_or_none(row.get("liquidityNum") or row.get("liquidity")),
        "open_interest": _float_or_none(row.get("openInterest")),
        "yes_price": yes_price,
        "no_price": no_price,
        "last_trade_price": _float_or_none(row.get("lastTradePrice")),
        "best_bid": _float_or_none(row.get("bestBid")),
        "best_ask": _float_or_none(row.get("bestAsk")),
        "spread": _float_or_none(row.get("spread")),
        "price_change_1h": _float_or_none(row.get("oneHourPriceChange")),
        "price_change_1d": _float_or_none(row.get("oneDayPriceChange")),
        "matched_alias": query,
        "match_method": "query_match" if query else "unmatched",
        "url": f"https://polymarket.com/market/{row.get('slug')}" if row.get("slug") else "",
        "data_quality_warning": "" if query else "polymarket_alias_missing",
    }


def _payload_rows(payload: Any) -> list[dict[str, Any]]:
    return payload if isinstance(payload, list) else []


def _outcome_prices(raw: Any) -> tuple[float | None, float | None]:
    if isinstance(raw, str):
        raw = raw.strip().strip("[]").replace('"', "").split(",") if raw else []
    if not isinstance(raw, list):
        return None, None
    prices = [_float_or_none(value) for value in raw]
    return (prices[0] if prices else None, prices[1] if len(prices) > 1 else None)


def _first_event_id(row: dict[str, Any]) -> str:
    events = row.get("events") or []
    if events and isinstance(events[0], dict):
        return str(events[0].get("id") or "")
    return str(row.get("eventId") or row.get("event_id") or "")


def _iso_ms(value: Any) -> int | None:
    if not value:
        return None
    try:
        return int(datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return None


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


def _bool_or_none(value: Any) -> bool | None:
    return None if value is None else bool(value)


def _float_or_none(value: Any) -> float | None:
    return None if value in {None, ""} else float(value)


def _int_or_none(value: Any) -> int | None:
    return None if value in {None, ""} else int(value)
