"""Source manifest helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import polars as pl

from qooi.sources.schema import SOURCE_MANIFEST_SCHEMA


def now_ms() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)


def empty_source_manifest_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=SOURCE_MANIFEST_SCHEMA)


def manifest_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return empty_source_manifest_frame()
    frame = pl.DataFrame(rows)
    for col, dtype in SOURCE_MANIFEST_SCHEMA.items():
        if col not in frame.columns:
            frame = frame.with_columns(pl.lit(None).cast(dtype).alias(col))
    return frame.select(pl.col(col).cast(dtype) for col, dtype in SOURCE_MANIFEST_SCHEMA.items())


def source_manifest_row(
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
) -> dict[str, Any]:
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
