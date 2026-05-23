"""Shared candidate and promotion gates for scored research patterns."""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

from qooi.research.contracts import SCORED_PATTERN_SCHEMA, ensure_columns


def apply_candidate_gate(metrics: pl.DataFrame, thresholds: Mapping[str, object]) -> pl.DataFrame:
    min_rows = int(thresholds.get("min_rows", 30))
    omega = float(thresholds.get("omega_threshold", 1.5))
    pwpr = float(thresholds.get("pwpr_threshold", 2.0))
    scored = metrics.with_columns(
        (
            (pl.col("rows") >= min_rows)
            & (~pl.col("invalid_state_present").fill_null(False))
            & (pl.col("omega_ratio").fill_null(0.0) > omega)
            & (pl.col("pwpr").fill_null(0.0) > pwpr)
            & (pl.col("mean_side_return_pct").fill_null(0.0) > 0)
        ).alias("passes_candidate_gate"),
        pl.concat_str(
            [
                _reason(pl.col("rows") < min_rows, "rows,"),
                _reason(pl.col("invalid_state_present").fill_null(False), "invalid_state,"),
                _reason(pl.col("omega_ratio").fill_null(0.0) <= omega, "omega,"),
                _reason(pl.col("pwpr").fill_null(0.0) <= pwpr, "pwpr,"),
                _reason(pl.col("mean_side_return_pct").fill_null(0.0) <= 0, "direction,"),
            ]
        )
        .str.strip_chars_end(",")
        .alias("gate_failure_reasons"),
        pl.lit(False).alias("passes_promotion_gate"),
        pl.lit("unscored").alias("promotion_failure_reasons"),
        pl.lit(0).alias("sufficient_symbols"),
        pl.lit(0.0).alias("symbol_direction_agreement_pct"),
        pl.lit(0).alias("time_splits"),
        pl.lit(0).alias("sufficient_time_splits"),
        pl.lit(0.0).alias("time_split_sign_agreement_pct"),
        pl.lit(False).alias("time_stable"),
    )
    return ensure_columns(scored, SCORED_PATTERN_SCHEMA)


def symbol_support(metrics: pl.DataFrame, keys: list[str], min_symbol_rows: int) -> pl.DataFrame:
    if metrics.is_empty():
        return metrics
    return (
        metrics.filter(pl.col("rows") >= min_symbol_rows)
        .group_by(keys)
        .agg(
            pl.len().alias("sufficient_symbols"),
            ((pl.col("mean_side_return_pct") > 0).mean() * 100.0)
            .round(0)
            .alias("symbol_direction_agreement_pct"),
        )
    )


def time_split_support(
    metrics: pl.DataFrame, keys: list[str], config: Mapping[str, object]
) -> pl.DataFrame:
    min_rows = int(config.get("min_time_split_rows", 15))
    splits = int(config.get("time_splits", 2))
    if metrics.is_empty() or "time_split" not in metrics.columns:
        return pl.DataFrame()
    return (
        metrics.filter(pl.col("rows") >= min_rows)
        .group_by(keys)
        .agg(
            pl.lit(splits).alias("time_splits"),
            pl.len().alias("sufficient_time_splits"),
            ((pl.col("mean_side_return_pct") > 0).mean() * 100.0)
            .round(0)
            .alias("time_split_sign_agreement_pct"),
        )
    )


def apply_promotion_gate(scored: pl.DataFrame, thresholds: Mapping[str, object]) -> pl.DataFrame:
    min_rows = int(thresholds.get("promotion_min_rows", 50))
    min_symbols = int(thresholds.get("promotion_min_symbols", 3))
    min_splits = int(thresholds.get("promotion_min_time_splits", 2))
    symbol_agreement = float(thresholds.get("promotion_symbol_agreement_pct", 67.0))
    time_agreement = float(thresholds.get("promotion_time_agreement_pct", 100.0))
    out = scored.with_columns(
        (
            (pl.col("sufficient_time_splits") >= min_splits)
            & (pl.col("time_split_sign_agreement_pct") >= time_agreement)
        ).alias("time_stable")
    ).with_columns(
        (
            pl.col("passes_candidate_gate")
            & (pl.col("rows") >= min_rows)
            & (pl.col("sufficient_symbols") >= min_symbols)
            & (pl.col("symbol_direction_agreement_pct") >= symbol_agreement)
            & pl.col("time_stable")
        ).alias("passes_promotion_gate"),
        pl.concat_str(
            [
                _reason(~pl.col("passes_candidate_gate"), "candidate_gate,"),
                _reason(pl.col("rows") < min_rows, "promotion_rows,"),
                _reason(pl.col("sufficient_symbols") < min_symbols, "symbols,"),
                _reason(
                    pl.col("symbol_direction_agreement_pct") < symbol_agreement, "symbol_agreement,"
                ),
                _reason(pl.col("sufficient_time_splits") < min_splits, "time_splits,"),
                _reason(
                    pl.col("time_split_sign_agreement_pct") < time_agreement, "time_agreement,"
                ),
            ]
        )
        .str.strip_chars_end(",")
        .alias("promotion_failure_reasons"),
    )
    return ensure_columns(out, SCORED_PATTERN_SCHEMA)


def _reason(condition: pl.Expr, text: str) -> pl.Expr:
    return pl.when(condition).then(pl.lit(text)).otherwise(pl.lit(""))
