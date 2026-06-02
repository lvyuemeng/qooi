"""Reusable source schemas shared by scanner source manifests."""

from __future__ import annotations

import polars as pl

SOURCE_MANIFEST_SCHEMA: dict[str, pl.DataType] = {
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
