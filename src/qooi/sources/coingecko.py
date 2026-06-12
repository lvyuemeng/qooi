"""CoinGecko broad market source helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx
import polars as pl

from qooi.sources.http import request_json_value, sanitize_error
from qooi.sources.manifest import manifest_frame, now_ms, source_manifest_row
from qooi.sources.models import SourceResult

COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

BROAD_MARKET_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Int64,
    "provider": pl.String,
    "coin_id": pl.String,
    "base_ccy": pl.String,
    "name": pl.String,
    "rank": pl.Int64,
    "price_usd": pl.Float64,
    "market_cap_usd": pl.Float64,
    "volume_24h_usd": pl.Float64,
    "volume_24h_change_pct": pl.Float64,
    "price_change_pct_1h": pl.Float64,
    "price_change_pct_24h": pl.Float64,
    "last_updated": pl.Int64,
    "trending_rank": pl.Int64,
    "trending_score": pl.Float64,
    "heat_source": pl.String,
}


async def fetch_coingecko_markets(
    client: httpx.AsyncClient,
    *,
    page: int,
    per_page: int = 250,
    api_key: str = "",
    vs_currency: str = "usd",
    order: str = "volume_desc",
    price_change_percentage: tuple[str, ...] = ("1h", "24h"),
) -> SourceResult:
    endpoint = "/coins/markets"
    params = {
        "vs_currency": vs_currency,
        "order": order,
        "per_page": str(min(max(per_page, 1), 250)),
        "page": str(page),
        "price_change_percentage": ",".join(price_change_percentage),
        "sparkline": "false",
    }
    headers = {"x-cg-demo-api-key": api_key} if api_key else None
    try:
        payload = await request_json_value(client, endpoint, params=params, headers=headers)
        frame = normalize_coingecko_markets(payload)
        return SourceResult(
            frame,
            manifest_frame(
                [
                    source_manifest_row(
                        symbol="*",
                        source="coingecko_markets",
                        phase="discover-broad",
                        status="ok" if not frame.is_empty() else "missing",
                        backend="coingecko",
                        endpoint=endpoint,
                        rows=frame.height,
                        range_start=_min_timestamp(frame),
                        range_end=_max_timestamp(frame),
                        warning="" if not frame.is_empty() else "coingecko_markets_empty",
                    )
                ]
            ),
        )
    except Exception as exc:
        return SourceResult(
            empty_broad_market_frame(),
            _failed_manifest("coingecko_markets", "coingecko", endpoint, exc),
        )


async def fetch_coingecko_trending(
    client: httpx.AsyncClient,
    *,
    api_key: str = "",
) -> SourceResult:
    endpoint = "/search/trending"
    headers = {"x-cg-demo-api-key": api_key} if api_key else None
    try:
        payload = await request_json_value(client, endpoint, headers=headers)
        frame = normalize_coingecko_trending(payload)
        return SourceResult(
            frame,
            manifest_frame(
                [
                    source_manifest_row(
                        symbol="*",
                        source="coingecko_trending",
                        phase="discover-broad",
                        status="ok" if not frame.is_empty() else "missing",
                        backend="coingecko",
                        endpoint=endpoint,
                        rows=frame.height,
                        range_start=_min_timestamp(frame) if not frame.is_empty() else None,
                        range_end=_max_timestamp(frame) if not frame.is_empty() else None,
                        warning="" if not frame.is_empty() else "coingecko_trending_empty",
                    )
                ]
            ),
        )
    except Exception as exc:
        return SourceResult(
            empty_broad_market_frame(),
            _failed_manifest("coingecko_trending", "coingecko", endpoint, exc),
        )


def normalize_coingecko_markets(payload: object) -> pl.DataFrame:
    if not isinstance(payload, list):
        return empty_broad_market_frame()
    ts = now_ms()
    rows = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "timestamp": ts,
                "provider": "coingecko",
                "coin_id": str(item.get("id") or ""),
                "base_ccy": str(item.get("symbol") or "").upper(),
                "name": str(item.get("name") or ""),
                "rank": _int_or_none(item.get("market_cap_rank")),
                "price_usd": _float_or_none(item.get("current_price")),
                "market_cap_usd": _float_or_none(item.get("market_cap")),
                "volume_24h_usd": _float_or_none(item.get("total_volume")),
                "volume_24h_change_pct": None,
                "price_change_pct_1h": _float_or_none(
                    item.get("price_change_percentage_1h_in_currency")
                ),
                "price_change_pct_24h": _float_or_none(
                    item.get("price_change_percentage_24h_in_currency")
                    or item.get("price_change_percentage_24h")
                ),
                "last_updated": _iso_ms(item.get("last_updated")),
                "trending_rank": None,
                "trending_score": None,
                "heat_source": "",
            }
        )
    return _coerce(pl.DataFrame(rows)) if rows else empty_broad_market_frame()


def normalize_coingecko_trending(payload: object) -> pl.DataFrame:
    if not isinstance(payload, dict):
        return empty_broad_market_frame()
    coins = payload.get("coins")
    if not isinstance(coins, list):
        return empty_broad_market_frame()
    ts = now_ms()
    rows = []
    for index, row in enumerate(coins, start=1):
        if not isinstance(row, dict):
            continue
        item = row.get("item") if isinstance(row.get("item"), dict) else row
        if not isinstance(item, dict):
            continue
        data = item.get("data") if isinstance(item.get("data"), dict) else {}
        price_change = (
            data.get("price_change_percentage_24h")
            if isinstance(data.get("price_change_percentage_24h"), dict)
            else {}
        )
        rows.append(
            {
                "timestamp": ts,
                "provider": "coingecko_trending",
                "coin_id": str(item.get("id") or item.get("coin_id") or ""),
                "base_ccy": str(item.get("symbol") or "").upper(),
                "name": str(item.get("name") or ""),
                "rank": _int_or_none(item.get("market_cap_rank")),
                "price_usd": None,
                "market_cap_usd": _float_or_none(data.get("market_cap")),
                "volume_24h_usd": _float_or_none(data.get("total_volume")),
                "volume_24h_change_pct": None,
                "price_change_pct_1h": None,
                "price_change_pct_24h": _float_or_none(price_change.get("usd")),
                "last_updated": None,
                "trending_rank": index,
                "trending_score": _float_or_none(item.get("score")),
                "heat_source": "coingecko_trending",
            }
        )
    return _coerce(pl.DataFrame(rows)) if rows else empty_broad_market_frame()


def empty_broad_market_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=BROAD_MARKET_SCHEMA)


def _failed_manifest(source: str, backend: str, endpoint: str, exc: BaseException) -> pl.DataFrame:
    return manifest_frame(
        [
            source_manifest_row(
                symbol="*",
                source=source,
                phase="discover-broad",
                status="failed",
                backend=backend,
                endpoint=endpoint,
                warning=sanitize_error(exc),
                stop_reason="http_error",
            )
        ]
    )


def _coerce(frame: pl.DataFrame) -> pl.DataFrame:
    for col, dtype in BROAD_MARKET_SCHEMA.items():
        if col not in frame.columns:
            frame = frame.with_columns(pl.lit(None).cast(dtype).alias(col))
        else:
            frame = frame.with_columns(pl.col(col).cast(dtype, strict=False).alias(col))
    return frame.select(BROAD_MARKET_SCHEMA.keys())


def _iso_ms(value: Any) -> int | None:
    if not value:
        return None
    try:
        return int(datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return None


def _min_timestamp(frame: pl.DataFrame) -> int | None:
    return int(frame["timestamp"].min()) if not frame.is_empty() else None


def _max_timestamp(frame: pl.DataFrame) -> int | None:
    return int(frame["timestamp"].max()) if not frame.is_empty() else None


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, dict):
        for key in ("usd", "value"):
            parsed = _float_or_none(value.get(key))
            if parsed is not None:
                return parsed
        return None
    if isinstance(value, str):
        value = value.replace("$", "").replace(",", "").strip()
        if not value:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    return None if value in {None, ""} else int(value)
