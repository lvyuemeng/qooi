"""Potential scanner package."""

from __future__ import annotations

import polars as pl


def pct_change_expr(next_col: str, base_col: str) -> pl.Expr:
    base = pl.when(pl.col(base_col).abs() > 1e-12).then(pl.col(base_col)).otherwise(None)
    return (pl.col(next_col) - pl.col(base_col)) / base * 100.0


def outcome_bucket_expr(return_threshold_pct: float) -> pl.Expr:
    return (
        pl.when(pl.col("forward_return_pct") > return_threshold_pct)
        .then(pl.lit("up"))
        .when(pl.col("forward_return_pct") < -return_threshold_pct)
        .then(pl.lit("down"))
        .otherwise(pl.lit("flat"))
    )


def entropy_term(col: str) -> pl.Expr:
    probability = pl.col(col).cast(pl.Float64)
    return pl.when(probability > 0.0).then(-probability * probability.log(2)).otherwise(0.0)


def entropy_expr(up_col: str, down_col: str, flat_col: str) -> pl.Expr:
    return entropy_term(up_col) + entropy_term(down_col) + entropy_term(flat_col)
