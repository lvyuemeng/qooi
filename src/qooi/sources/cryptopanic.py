"""CryptoPanic broad news source helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx
import polars as pl

from qooi.sources.http import request_json_value, sanitize_error
from qooi.sources.manifest import manifest_frame, source_manifest_row
from qooi.sources.models import SourceResult

CRYPTOPANIC_BASE_URL = "https://cryptopanic.com/api/v1"

BROAD_NEWS_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Int64,
    "provider": pl.String,
    "source_id": pl.String,
    "title": pl.String,
    "url": pl.String,
    "base_ccy": pl.String,
    "sentiment": pl.String,
}


async def fetch_cryptopanic_global_posts(
    client: httpx.AsyncClient, *, api_key: str, limit: int = 100
) -> SourceResult:
    endpoint = "/posts/"
    try:
        payload = await request_json_value(
            client,
            endpoint,
            params={"auth_token": api_key, "public": "true", "limit": str(limit)},
        )
        frame = normalize_cryptopanic_posts(payload)
        return SourceResult(
            frame,
            manifest_frame(
                [
                    source_manifest_row(
                        symbol="*",
                        source="cryptopanic_posts",
                        phase="discover-broad",
                        status="ok" if not frame.is_empty() else "missing",
                        backend="cryptopanic",
                        endpoint=endpoint,
                        rows=frame.height,
                        range_start=_min_timestamp(frame),
                        range_end=_max_timestamp(frame),
                        warning="" if not frame.is_empty() else "cryptopanic_posts_empty",
                    )
                ]
            ),
        )
    except Exception as exc:
        return SourceResult(
            empty_broad_news_frame(),
            manifest_frame(
                [
                    source_manifest_row(
                        symbol="*",
                        source="cryptopanic_posts",
                        phase="discover-broad",
                        status="failed",
                        backend="cryptopanic",
                        endpoint=endpoint,
                        warning=sanitize_error(exc),
                        stop_reason="http_error",
                    )
                ]
            ),
        )


def normalize_cryptopanic_posts(payload: object) -> pl.DataFrame:
    if not isinstance(payload, dict):
        return empty_broad_news_frame()
    results = payload.get("results")
    if not isinstance(results, list):
        return empty_broad_news_frame()
    rows = []
    for item in results:
        if not isinstance(item, dict):
            continue
        currencies = item.get("currencies") or []
        bases = _currency_codes(currencies)
        if not bases:
            bases = [""]
        for base in bases:
            rows.append(
                {
                    "timestamp": _iso_ms(item.get("published_at")),
                    "provider": "cryptopanic",
                    "source_id": str(item.get("id") or ""),
                    "title": str(item.get("title") or ""),
                    "url": str(item.get("url") or ""),
                    "base_ccy": base,
                    "sentiment": str(item.get("kind") or ""),
                }
            )
    return _coerce(pl.DataFrame(rows)) if rows else empty_broad_news_frame()


def empty_broad_news_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=BROAD_NEWS_SCHEMA)


def missing_api_key_result(*, api_key_env: str) -> SourceResult:
    return SourceResult(
        empty_broad_news_frame(),
        manifest_frame(
            [
                source_manifest_row(
                    symbol="*",
                    source="cryptopanic_posts",
                    phase="discover-broad",
                    status="missing",
                    backend="cryptopanic",
                    endpoint="/posts/",
                    warning="missing_api_key",
                    stop_reason=api_key_env,
                )
            ]
        ),
    )


def _coerce(frame: pl.DataFrame) -> pl.DataFrame:
    for col, dtype in BROAD_NEWS_SCHEMA.items():
        if col not in frame.columns:
            frame = frame.with_columns(pl.lit(None).cast(dtype).alias(col))
        else:
            frame = frame.with_columns(pl.col(col).cast(dtype, strict=False).alias(col))
    return frame.select(BROAD_NEWS_SCHEMA.keys())


def _currency_codes(currencies: object) -> list[str]:
    if not isinstance(currencies, list):
        return []
    codes = []
    for row in currencies:
        if isinstance(row, dict):
            code = str(row.get("code") or row.get("symbol") or "").upper()
            if code:
                codes.append(code)
    return codes


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

