"""Reusable source collectors and source artifact helpers."""

from __future__ import annotations

import polars as pl

HOUR_MS = 60 * 60 * 1000
DAY_MS = 24 * HOUR_MS


def funding_min_rows(days: int) -> int:
    return max(1, int(days * DAY_MS / (8 * HOUR_MS)))


def period_min_rows(days: int, period: str) -> int:
    period_ms = period_ms_value(period)
    if period_ms <= 0:
        return 0
    return max(1, int(days * DAY_MS / period_ms))


def period_ms_value(period: str) -> int:
    value = period.strip().upper()
    if value.endswith("H"):
        return int(value[:-1]) * HOUR_MS
    if value.endswith("D"):
        return int(value[:-1]) * DAY_MS
    if value.endswith("M"):
        return int(value[:-1]) * 60 * 1000
    return 0


def normalize_funding_artifact_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Return a funding artifact frame with explicit history/current row semantics.

    Legacy cached rows may predate ``funding_source_kind`` and ``known_at_ms``.
    Treat those rows as historical funding events and fill known-at time from the
    funding event timestamp so every persisted row has one explicit kind.
    """
    if frame.is_empty():
        return frame
    out = frame
    if "funding_source_kind" not in out.columns:
        out = out.with_columns(pl.lit("history").alias("funding_source_kind"))
    else:
        out = out.with_columns(
            pl.when(pl.col("funding_source_kind").is_null() | (pl.col("funding_source_kind") == ""))
            .then(pl.lit("history"))
            .otherwise(pl.col("funding_source_kind"))
            .alias("funding_source_kind")
        )
    fallback_col = "funding_time" if "funding_time" in out.columns else "timestamp"
    if "known_at_ms" not in out.columns:
        out = out.with_columns(pl.col(fallback_col).alias("known_at_ms"))
    else:
        out = out.with_columns(
            pl.when(pl.col("known_at_ms").is_null())
            .then(pl.col(fallback_col))
            .otherwise(pl.col("known_at_ms"))
            .alias("known_at_ms")
        )
    if "next_funding_rate" not in out.columns:
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("next_funding_rate"))
    if "next_funding_time" not in out.columns:
        out = out.with_columns(pl.lit(None, dtype=pl.Int64).alias("next_funding_time"))
    return out
