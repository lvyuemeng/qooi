"""Shared research table contracts for behavior-driven state analysis."""

from __future__ import annotations

import polars as pl

RESEARCH_FRAME_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.Utf8,
    "timeframe": pl.Utf8,
    "timestamp": pl.Int64,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "state_source": pl.Utf8,
    "state_column": pl.Utf8,
    "state_value": pl.Utf8,
    "event_column": pl.Utf8,
    "event_value": pl.Utf8,
}

PATTERN_TABLE_SCHEMA: dict[str, pl.DataType] = {
    "pattern_id": pl.Utf8,
    "pattern_family": pl.Utf8,
    "pattern_source": pl.Utf8,
    "symbol": pl.Utf8,
    "timeframe": pl.Utf8,
    "timestamp": pl.Int64,
    "state_source": pl.Utf8,
    "state_column": pl.Utf8,
    "pattern_value": pl.Utf8,
    "event_value": pl.Utf8,
    "side": pl.Utf8,
    "ngram_length": pl.Int64,
    "invalid_state_present": pl.Boolean,
}

OUTCOME_TABLE_SCHEMA: dict[str, pl.DataType] = {
    **PATTERN_TABLE_SCHEMA,
    "horizon": pl.Int64,
    "forward_return_pct": pl.Float64,
    "side_return_pct": pl.Float64,
    "forward_direction": pl.Utf8,
}

METRIC_TABLE_SCHEMA: dict[str, pl.DataType] = {
    "pattern_id": pl.Utf8,
    "pattern_family": pl.Utf8,
    "pattern_source": pl.Utf8,
    "symbol": pl.Utf8,
    "horizon": pl.Int64,
    "side": pl.Utf8,
    "rows": pl.Int64,
    "positive_rate": pl.Float64,
    "negative_rate": pl.Float64,
    "positive_mean": pl.Float64,
    "negative_mean_abs": pl.Float64,
    "omega_ratio": pl.Float64,
    "pwpr": pl.Float64,
    "sortino_zero": pl.Float64,
    "mean_side_return_pct": pl.Float64,
    "transition_information": pl.Float64,
    "conditional_transition_information": pl.Float64,
    "normalized_transition_information": pl.Float64,
    "normalized_conditional_transition_information": pl.Float64,
    "bias_warning": pl.Utf8,
    "sufficient": pl.Boolean,
    "invalid_state_present": pl.Boolean,
}

SCORED_PATTERN_SCHEMA: dict[str, pl.DataType] = {
    **METRIC_TABLE_SCHEMA,
    "passes_candidate_gate": pl.Boolean,
    "gate_failure_reasons": pl.Utf8,
    "passes_promotion_gate": pl.Boolean,
    "promotion_failure_reasons": pl.Utf8,
    "sufficient_symbols": pl.Int64,
    "symbol_direction_agreement_pct": pl.Float64,
    "time_splits": pl.Int64,
    "sufficient_time_splits": pl.Int64,
    "time_split_sign_agreement_pct": pl.Float64,
    "time_stable": pl.Boolean,
}


def ensure_columns(frame: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """Return a dataframe containing schema columns, preserving extra columns."""
    additions = [
        pl.lit(None).cast(dtype).alias(column)
        for column, dtype in schema.items()
        if column not in frame.columns
    ]
    work = frame.with_columns(additions) if additions else frame
    casts = [pl.col(column).cast(dtype).alias(column) for column, dtype in schema.items()]
    extras = [pl.col(column) for column in work.columns if column not in schema]
    return work.select([*casts, *extras])


def empty_frame(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)
