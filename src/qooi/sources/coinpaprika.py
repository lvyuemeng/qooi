"""CoinPaprika broad market source helpers."""

from __future__ import annotations

from typing import Any

import httpx
import polars as pl

from qooi.sources.http import request_json_value, sanitize_error
from qooi.sources.manifest import manifest_frame, now_ms, source_manifest_row
from qooi.sources.models import SourceResult

COINPAPRIKA_BASE_URL = "https://api.coinpaprika.com/v1"

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


async def fetch_coinpaprika_tickers(
    client: httpx.AsyncClient, *, quotes: str = "USD"
) -> SourceResult:
    endpoint = "/tickers"
    try:
        payload = await request_json_value(client, endpoint, params={"quotes": quotes})
        frame = normalize_coinpaprika_tickers(payload)
        return SourceResult(
            frame,
            manifest_frame(
                [
                    source_manifest_row(
                        symbol="*",
                        source="coinpaprika_tickers",
                        phase="discover-broad",
                        status="ok" if not frame.is_empty() else "missing",
                        backend="coinpaprika",
                        endpoint=endpoint,
                        rows=frame.height,
                        range_start=_min_timestamp(frame),
                        range_end=_max_timestamp(frame),
                        warning="" if not frame.is_empty() else "coinpaprika_tickers_empty",
                    )
                ]
            ),
        )
    except Exception as exc:
        return SourceResult(
            empty_broad_market_frame(),
            manifest_frame(
                [
                    source_manifest_row(
                        symbol="*",
                        source="coinpaprika_tickers",
                        phase="discover-broad",
                        status="failed",
                        backend="coinpaprika",
                        endpoint=endpoint,
                        warning=sanitize_error(exc),
                        stop_reason="http_error",
                    )
                ]
            ),
        )


def normalize_coinpaprika_tickers(payload: object) -> pl.DataFrame:
    if not isinstance(payload, list):
        return empty_broad_market_frame()
    ts = now_ms()
    rows = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        quote = item.get("quotes") or {}
        usd = quote.get("USD") if isinstance(quote, dict) else {}
        if not isinstance(usd, dict):
            usd = {}
        rows.append(
            {
                "timestamp": ts,
                "provider": "coinpaprika",
                "coin_id": str(item.get("id") or ""),
                "base_ccy": str(item.get("symbol") or "").upper(),
                "name": str(item.get("name") or ""),
                "rank": _int_or_none(item.get("rank")),
                "price_usd": _float_or_none(usd.get("price")),
                "market_cap_usd": _float_or_none(usd.get("market_cap")),
                "volume_24h_usd": _float_or_none(usd.get("volume_24h")),
                "volume_24h_change_pct": _float_or_none(usd.get("volume_24h_change_24h")),
                "price_change_pct_1h": _float_or_none(usd.get("percent_change_1h")),
                "price_change_pct_24h": _float_or_none(usd.get("percent_change_24h")),
                "last_updated": None,
                "trending_rank": None,
                "trending_score": None,
                "heat_source": "",
            }
        )
    return _coerce(pl.DataFrame(rows)) if rows else empty_broad_market_frame()


def empty_broad_market_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=BROAD_MARKET_SCHEMA)


def _coerce(frame: pl.DataFrame) -> pl.DataFrame:
    for col, dtype in BROAD_MARKET_SCHEMA.items():
        if col not in frame.columns:
            frame = frame.with_columns(pl.lit(None).cast(dtype).alias(col))
        else:
            frame = frame.with_columns(pl.col(col).cast(dtype, strict=False).alias(col))
    return frame.select(BROAD_MARKET_SCHEMA.keys())


def _min_timestamp(frame: pl.DataFrame) -> int | None:
    return int(frame["timestamp"].min()) if not frame.is_empty() else None


def _max_timestamp(frame: pl.DataFrame) -> int | None:
    return int(frame["timestamp"].max()) if not frame.is_empty() else None


def _float_or_none(value: Any) -> float | None:
    return None if value in {None, ""} else float(value)


def _int_or_none(value: Any) -> int | None:
    return None if value in {None, ""} else int(value)
