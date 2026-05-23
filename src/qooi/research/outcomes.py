"""Attach forward outcomes to materialized research patterns."""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl

from qooi.research.contracts import OUTCOME_TABLE_SCHEMA, empty_frame, ensure_columns

_BULLISH_EVENTS = {"failed_breakout_low", "bullish_reclaim", "breakout_acceptance_high"}
_BEARISH_EVENTS = {"failed_breakout_high", "bearish_reclaim", "breakout_acceptance_low"}


def side_from_event(event_value: str | pl.Expr | None = None) -> str | pl.Expr | None:
    if isinstance(event_value, str):
        if event_value in _BULLISH_EVENTS:
            return "long"
        if event_value in _BEARISH_EVENTS:
            return "short"
        return None
    expr = event_value if isinstance(event_value, pl.Expr) else pl.col("event_value")
    return (
        pl.when(expr.is_in(_BULLISH_EVENTS))
        .then(pl.lit("long"))
        .when(expr.is_in(_BEARISH_EVENTS))
        .then(pl.lit("short"))
        .otherwise(None)
    )


def attach_forward_outcomes(
    patterns: pl.DataFrame, market_frame: pl.DataFrame, horizons: Iterable[int]
) -> pl.DataFrame:
    if patterns.is_empty():
        return empty_frame(OUTCOME_TABLE_SCHEMA)
    if market_frame.is_empty() or "close" not in market_frame.columns:
        return empty_frame(OUTCOME_TABLE_SCHEMA)
    keys = [column for column in ("symbol", "timeframe", "timestamp") if column in patterns.columns]
    market = market_frame
    if "symbol" not in market.columns and "symbol" in patterns.columns:
        market = market.with_columns(pl.lit(patterns.select("symbol").item(0, 0)).alias("symbol"))
    if "timeframe" not in market.columns and "timeframe" in patterns.columns:
        market = market.with_columns(
            pl.lit(patterns.select("timeframe").item(0, 0)).alias("timeframe")
        )
    sort_cols = [
        column for column in ("symbol", "timeframe", "timestamp") if column in market.columns
    ]
    market = market.sort(sort_cols) if sort_cols else market.sort("timestamp")
    frames = []
    group_cols = [column for column in ("symbol", "timeframe") if column in market.columns]
    for horizon in horizons:
        future_close = (
            pl.col("close").shift(-int(horizon)).over(group_cols)
            if group_cols
            else pl.col("close").shift(-int(horizon))
        )
        returns = market.with_columns(
            pl.lit(int(horizon)).alias("horizon"),
            ((future_close - pl.col("close")) / pl.col("close") * 100.0).alias(
                "forward_return_pct"
            ),
        ).with_columns(
            pl.when(pl.col("forward_return_pct") > 0)
            .then(pl.lit("up"))
            .when(pl.col("forward_return_pct") < 0)
            .then(pl.lit("down"))
            .otherwise(pl.lit("flat"))
            .alias("forward_direction")
        )
        joined = patterns.join(
            returns.select(*keys, "horizon", "forward_return_pct", "forward_direction"),
            on=keys,
            how="left",
        )
        frames.append(joined)
    return attach_side_returns(pl.concat(frames, how="diagonal_relaxed"))


def attach_side_returns(outcomes: pl.DataFrame) -> pl.DataFrame:
    if outcomes.is_empty():
        return empty_frame(OUTCOME_TABLE_SCHEMA)
    work = outcomes.with_columns(
        pl.coalesce([pl.col("side"), side_from_event(pl.col("event_value"))]).alias("side")
    ).with_columns(
        pl.when(pl.col("side") == "short")
        .then(-pl.col("forward_return_pct"))
        .otherwise(pl.col("forward_return_pct"))
        .alias("side_return_pct")
    )
    return ensure_columns(work, OUTCOME_TABLE_SCHEMA)
