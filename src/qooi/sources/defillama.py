"""DeFiLlama broad protocol source helpers."""

from __future__ import annotations

from typing import Any

import httpx
import polars as pl

from qooi.sources.http import request_json_value_async, sanitize_error
from qooi.sources.manifest import manifest_frame, now_ms, source_manifest_row
from qooi.sources.models import SourceResult

DEFILLAMA_BASE_URL = "https://api.llama.fi"

BROAD_PROTOCOL_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Int64,
    "provider": pl.String,
    "protocol": pl.String,
    "base_ccy": pl.String,
    "name": pl.String,
    "category": pl.String,
    "chains": pl.String,
    "tvl_usd": pl.Float64,
    "tvl_change_1d_pct": pl.Float64,
    "tvl_change_7d_pct": pl.Float64,
}


async def fetch_defillama_protocols_async(client: httpx.AsyncClient) -> SourceResult:
    endpoint = "/protocols"
    try:
        payload = await request_json_value_async(client, endpoint)
        frame = normalize_defillama_protocols(payload)
        return SourceResult(
            frame,
            manifest_frame(
                [
                    source_manifest_row(
                        symbol="*",
                        source="defillama_protocols",
                        phase="discover-broad",
                        status="ok" if not frame.is_empty() else "missing",
                        backend="defillama",
                        endpoint=endpoint,
                        rows=frame.height,
                        range_start=_min_timestamp(frame),
                        range_end=_max_timestamp(frame),
                        warning="" if not frame.is_empty() else "defillama_protocols_empty",
                    )
                ]
            ),
        )
    except Exception as exc:
        return SourceResult(
            empty_broad_protocol_frame(),
            manifest_frame(
                [
                    source_manifest_row(
                        symbol="*",
                        source="defillama_protocols",
                        phase="discover-broad",
                        status="failed",
                        backend="defillama",
                        endpoint=endpoint,
                        warning=sanitize_error(exc),
                        stop_reason="http_error",
                    )
                ]
            ),
        )


def normalize_defillama_protocols(payload: object) -> pl.DataFrame:
    if not isinstance(payload, list):
        return empty_broad_protocol_frame()
    ts = now_ms()
    rows = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "timestamp": ts,
                "provider": "defillama",
                "protocol": str(item.get("slug") or item.get("name") or ""),
                "base_ccy": str(item.get("symbol") or "").upper(),
                "name": str(item.get("name") or ""),
                "category": str(item.get("category") or ""),
                "chains": _join_strings(item.get("chains")),
                "tvl_usd": _float_or_none(item.get("tvl")),
                "tvl_change_1d_pct": _float_or_none(item.get("change_1d")),
                "tvl_change_7d_pct": _float_or_none(item.get("change_7d")),
            }
        )
    return _coerce(pl.DataFrame(rows)) if rows else empty_broad_protocol_frame()


def empty_broad_protocol_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=BROAD_PROTOCOL_SCHEMA)


def _coerce(frame: pl.DataFrame) -> pl.DataFrame:
    for col, dtype in BROAD_PROTOCOL_SCHEMA.items():
        if col not in frame.columns:
            frame = frame.with_columns(pl.lit(None).cast(dtype).alias(col))
        else:
            frame = frame.with_columns(pl.col(col).cast(dtype, strict=False).alias(col))
    return frame.select(BROAD_PROTOCOL_SCHEMA.keys())


def _join_strings(value: Any) -> str:
    if isinstance(value, list):
        return ",".join(str(item) for item in value if item)
    return str(value or "")


def _min_timestamp(frame: pl.DataFrame) -> int | None:
    return int(frame["timestamp"].min()) if not frame.is_empty() else None


def _max_timestamp(frame: pl.DataFrame) -> int | None:
    return int(frame["timestamp"].max()) if not frame.is_empty() else None


def _float_or_none(value: Any) -> float | None:
    return None if value in {None, ""} else float(value)
