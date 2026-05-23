"""Normalize prepared research frames into shared pipe contracts."""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl

from qooi.research.contracts import RESEARCH_FRAME_SCHEMA, empty_frame, ensure_columns


def normalize_research_frame(
    frame: pl.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    state_columns: Iterable[str],
    event_column: str,
    context_columns: Iterable[str] = (),
    state_source: str = "classifier",
) -> pl.DataFrame:
    """Convert a wide known-at-close frame into long state/event rows."""
    if frame.is_empty():
        return empty_frame(RESEARCH_FRAME_SCHEMA)
    base_cols = [
        column for column in ("timestamp", "open", "high", "low", "close") if column in frame
    ]
    context_cols = [column for column in context_columns if column in frame]
    states = [column for column in state_columns if column in frame]
    if not states:
        return empty_frame(RESEARCH_FRAME_SCHEMA)
    work = frame.with_columns(
        pl.lit(symbol).alias("symbol"),
        pl.lit(timeframe).alias("timeframe"),
        pl.lit(event_column).alias("event_column"),
        pl.col(event_column).cast(pl.Utf8).alias("event_value")
        if event_column in frame.columns
        else pl.lit(None, dtype=pl.Utf8).alias("event_value"),
    )
    rows = []
    for state_column in states:
        selected = work.select(
            "symbol",
            "timeframe",
            *base_cols,
            pl.lit(state_source).alias("state_source"),
            pl.lit(state_column).alias("state_column"),
            pl.col(state_column).cast(pl.Utf8).alias("state_value"),
            "event_column",
            "event_value",
            *context_cols,
        )
        rows.append(ensure_columns(selected, RESEARCH_FRAME_SCHEMA))
    return concat_research_frames(rows)


def concat_research_frames(frames: Iterable[pl.DataFrame]) -> pl.DataFrame:
    non_empty = [frame for frame in frames if not frame.is_empty()]
    if not non_empty:
        return empty_frame(RESEARCH_FRAME_SCHEMA)
    return ensure_columns(pl.concat(non_empty, how="diagonal_relaxed"), RESEARCH_FRAME_SCHEMA)


def validate_research_frame(frame: pl.DataFrame) -> pl.DataFrame:
    missing = [column for column in RESEARCH_FRAME_SCHEMA if column not in frame.columns]
    if missing:
        raise ValueError("ResearchFrame missing columns: " + ", ".join(missing))
    return ensure_columns(frame, RESEARCH_FRAME_SCHEMA)
