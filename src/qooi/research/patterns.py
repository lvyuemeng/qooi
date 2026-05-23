"""Materialize research patterns from normalized state/event rows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import polars as pl

from qooi.research.contracts import PATTERN_TABLE_SCHEMA, empty_frame, ensure_columns

DEFAULT_INVALID_VALUES = ("warmup", "unknown", "data_error")


def materialize_static_patterns(
    research_frame: pl.DataFrame, spec: Mapping[str, object] | None = None
) -> pl.DataFrame:
    if research_frame.is_empty():
        return empty_frame(PATTERN_TABLE_SCHEMA)
    source = str((spec or {}).get("pattern_source", "static_state"))
    return _with_pattern_id(
        research_frame.select(
            pl.lit("static_state").alias("pattern_family"),
            pl.lit(source).alias("pattern_source"),
            "symbol",
            "timeframe",
            "timestamp",
            "state_source",
            "state_column",
            pl.col("state_value").alias("pattern_value"),
            "event_value",
            pl.lit(None, dtype=pl.Utf8).alias("side"),
            pl.lit(1).alias("ngram_length"),
            _invalid_expr("state_value", _invalid_values(spec)).alias("invalid_state_present"),
        )
    )


def materialize_transition_patterns(
    research_frame: pl.DataFrame, spec: Mapping[str, object] | None = None
) -> pl.DataFrame:
    if research_frame.is_empty():
        return empty_frame(PATTERN_TABLE_SCHEMA)
    ngram_lengths = tuple(int(v) for v in (spec or {}).get("ngram_lengths", (2,)))
    source = str((spec or {}).get("pattern_source", "transition"))
    sort_cols = ["symbol", "timeframe", "state_source", "state_column", "timestamp"]
    work = research_frame.sort([column for column in sort_cols if column in research_frame.columns])
    frames = []
    group = ["symbol", "timeframe", "state_source", "state_column"]
    for length in ngram_lengths:
        if length < 2:
            continue
        value_exprs = [
            pl.col("state_value").shift(offset).over(group) for offset in range(length - 1, -1, -1)
        ]
        ngram = pl.concat_str([expr.cast(pl.Utf8) for expr in value_exprs], separator="->")
        ready = pl.all_horizontal([expr.is_not_null() for expr in value_exprs])
        family = "transition" if length == 2 else "transition_ngram"
        frame = work.with_columns(
            ngram.alias("pattern_value"),
            ready.alias("_ready"),
        ).filter(pl.col("_ready"))
        frame = frame.select(
            pl.lit(family).alias("pattern_family"),
            pl.lit(source).alias("pattern_source"),
            "symbol",
            "timeframe",
            "timestamp",
            "state_source",
            "state_column",
            "pattern_value",
            "event_value",
            pl.lit(None, dtype=pl.Utf8).alias("side"),
            pl.lit(length).alias("ngram_length"),
            _invalid_expr("pattern_value", _invalid_values(spec)).alias("invalid_state_present"),
        )
        frames.append(_with_pattern_id(frame))
    return concat_patterns(frames)


def materialize_none_event_context_patterns(
    research_frame: pl.DataFrame, spec: Mapping[str, object] | None = None
) -> pl.DataFrame:
    if research_frame.is_empty():
        return empty_frame(PATTERN_TABLE_SCHEMA)
    context_columns = [
        column for column in (spec or {}).get("context_columns", ()) if column in research_frame
    ]
    if not context_columns:
        return empty_frame(PATTERN_TABLE_SCHEMA)
    source = str((spec or {}).get("pattern_source", "none_event_context"))
    rows = []
    none_rows = research_frame.filter(pl.col("event_value") == "none")
    for column in context_columns:
        rows.append(
            _with_pattern_id(
                none_rows.select(
                    pl.lit("none_event_context").alias("pattern_family"),
                    pl.lit(source).alias("pattern_source"),
                    "symbol",
                    "timeframe",
                    "timestamp",
                    "state_source",
                    pl.lit(column).alias("state_column"),
                    pl.col(column).cast(pl.Utf8).alias("pattern_value"),
                    "event_value",
                    pl.lit(None, dtype=pl.Utf8).alias("side"),
                    pl.lit(1).alias("ngram_length"),
                    _invalid_expr(column, _invalid_values(spec)).alias("invalid_state_present"),
                )
            )
        )
    return concat_patterns(rows)


def concat_patterns(patterns: Iterable[pl.DataFrame]) -> pl.DataFrame:
    non_empty = [frame for frame in patterns if not frame.is_empty()]
    if not non_empty:
        return empty_frame(PATTERN_TABLE_SCHEMA)
    return ensure_columns(pl.concat(non_empty, how="diagonal_relaxed"), PATTERN_TABLE_SCHEMA)


def _with_pattern_id(frame: pl.DataFrame) -> pl.DataFrame:
    work = frame.with_columns(
        pl.concat_str(
            [
                pl.col("pattern_family"),
                pl.col("pattern_source"),
                pl.col("state_source"),
                pl.col("state_column"),
                pl.col("pattern_value").fill_null("null"),
                pl.col("event_value").fill_null("null"),
                pl.col("ngram_length").cast(pl.Utf8),
            ],
            separator="|",
        ).alias("pattern_id")
    )
    return ensure_columns(work, PATTERN_TABLE_SCHEMA)


def _invalid_values(spec: Mapping[str, object] | None) -> tuple[str, ...]:
    values = (spec or {}).get("invalid_values", DEFAULT_INVALID_VALUES)
    return tuple(str(value) for value in values)


def _invalid_expr(column: str, invalid_values: tuple[str, ...]) -> pl.Expr:
    expr = pl.lit(False)
    for value in invalid_values:
        expr = expr | pl.col(column).cast(pl.Utf8).str.contains(value, literal=True)
    return expr
