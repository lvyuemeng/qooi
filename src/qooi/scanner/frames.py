"""Shared scanner observation and outcome frame builders."""

from __future__ import annotations

import polars as pl

from qooi.scanner import outcome_bucket_expr

SOURCE_KLINE_RECENT_WINDOW_MS = 30 * 24 * 60 * 60 * 1000


POTENTIAL_OBSERVATION_SCHEMA = {
    "symbol": pl.String,
    "decision_timeframe": pl.String,
    "decision_bar_close_ms": pl.Int64,
    "background_regime": pl.String,
    "background_structure": pl.String,
    "background_range": pl.String,
    "background_vol": pl.String,
    "swing_regime": pl.String,
    "swing_core": pl.String,
    "swing_range": pl.String,
    "swing_transition": pl.String,
    "decision_direction": pl.String,
    "decision_regime": pl.String,
    "decision_core": pl.String,
    "decision_range": pl.String,
    "decision_vol": pl.String,
    "decision_event": pl.String,
    "decision_event_age_bucket": pl.String,
    "decision_transition": pl.String,
    "source_family": pl.String,
    "source_state": pl.String,
    "source_direction": pl.String,
    "source_known_at_ms": pl.Int64,
    "source_age_ms": pl.Int64,
    "source_freshness": pl.String,
    "market_alignment": pl.String,
    "source_market_alignment": pl.String,
    "risk_context": pl.String,
}


def potential_observation_frame(
    kline_history: pl.DataFrame,
    source_events: pl.DataFrame,
    continuous_features: pl.DataFrame | None = None,
    *,
    decision_timeframe: str,
    max_source_staleness_hours: int,
) -> pl.DataFrame:
    if (
        kline_history.is_empty()
        or decision_timeframe not in kline_history.get_column("timeframe").to_list()
    ):
        return pl.DataFrame(schema=POTENTIAL_OBSERVATION_SCHEMA)
    decision = _potential_state_columns(kline_history, decision_timeframe, "decision").rename(
        {"bar_close_ms": "decision_bar_close_ms"}
    )
    if decision.is_empty():
        return pl.DataFrame(schema=POTENTIAL_OBSERVATION_SCHEMA)
    observations = decision.with_columns(pl.lit(decision_timeframe).alias("decision_timeframe"))
    for timeframe, prefix in (("4H", "swing"), ("1D", "background")):
        state = _potential_state_columns(kline_history, timeframe, prefix)
        if state.is_empty():
            observations = observations.with_columns(
                *[
                    pl.lit(None, dtype=pl.String).alias(column)
                    for column in _potential_state_output_columns(prefix)
                ]
            )
            continue
        observations = observations.sort("symbol", "decision_bar_close_ms").join_asof(
            state.sort("symbol", "bar_close_ms"),
            left_on="decision_bar_close_ms",
            right_on="bar_close_ms",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
    observations = observations.with_columns(
        pl.when(pl.col("background_regime") == pl.col("swing_regime"))
        .then(pl.lit("background_swing_aligned"))
        .when(pl.col("background_regime").is_null() | pl.col("swing_regime").is_null())
        .then(pl.lit("market_context_missing"))
        .otherwise(pl.lit("background_swing_conflict"))
        .alias("market_alignment"),
        pl.concat_str("decision_range", "decision_vol", separator="|").alias("risk_context"),
    )
    if source_events.is_empty():
        return _join_continuous_features(
            _potential_observation_without_source(observations), continuous_features
        )
    source_rows = source_events.filter(pl.col("source_state").is_not_null()).select(
        "symbol",
        "source_family",
        "source_state",
        "source_direction",
        pl.col("known_at_ms").alias("source_known_at_ms"),
    )
    if source_rows.is_empty():
        return _join_continuous_features(
            _potential_observation_without_source(observations), continuous_features
        )
    frames = []
    max_age_ms = max_source_staleness_hours * 60 * 60 * 1000
    for family in source_rows.get_column("source_family").drop_nulls().unique().to_list():
        family_source = source_rows.filter(pl.col("source_family") == family)
        frames.append(
            observations.sort("symbol", "decision_bar_close_ms")
            .join_asof(
                family_source.sort("symbol", "source_known_at_ms"),
                left_on="decision_bar_close_ms",
                right_on="source_known_at_ms",
                by="symbol",
                strategy="backward",
                check_sortedness=False,
            )
            .with_columns(
                (pl.col("decision_bar_close_ms") - pl.col("source_known_at_ms")).alias(
                    "source_age_ms"
                ),
                pl.when(pl.col("source_known_at_ms").is_null())
                .then(pl.lit("missing"))
                .when((pl.col("decision_bar_close_ms") - pl.col("source_known_at_ms")) > max_age_ms)
                .then(pl.lit("stale"))
                .otherwise(pl.lit("fresh"))
                .alias("source_freshness"),
                pl.when(pl.col("source_direction").is_null())
                .then(pl.lit("source_missing"))
                .when(pl.col("source_direction") == pl.col("decision_direction"))
                .then(pl.lit("source_agrees_with_decision"))
                .when(pl.col("source_direction") == "neutral")
                .then(pl.lit("source_neutral"))
                .otherwise(pl.lit("source_conflicts_with_decision"))
                .alias("source_market_alignment"),
            )
        )
    if not frames:
        result = _potential_observation_without_source(observations)
    else:
        result = pl.concat(frames, how="vertical_relaxed").select(
            *POTENTIAL_OBSERVATION_SCHEMA.keys()
        )

    return _join_continuous_features(result, continuous_features)


def _join_continuous_features(
    observations: pl.DataFrame, continuous_features: pl.DataFrame | None
) -> pl.DataFrame:
    if continuous_features is None or continuous_features.is_empty():
        return observations
    if not {
        "symbol",
        "decision_bar_close_ms",
    }.issubset(observations.columns) or not {"symbol", "timestamp"}.issubset(
        continuous_features.columns
    ):
        return observations
    cf_cols = [
        c
        for c in continuous_features.columns
        if c not in ("symbol", "timestamp") and c not in observations.columns
    ]
    if not cf_cols:
        return observations
    return observations.join(
        continuous_features.select(["symbol", "timestamp"] + cf_cols).unique(
            subset=["symbol", "timestamp"], keep="last"
        ),
        left_on=["symbol", "decision_bar_close_ms"],
        right_on=["symbol", "timestamp"],
        how="left",
    )


def _potential_observation_without_source(observations: pl.DataFrame) -> pl.DataFrame:
    return observations.with_columns(
        pl.lit(None, dtype=pl.String).alias("source_family"),
        pl.lit(None, dtype=pl.String).alias("source_state"),
        pl.lit(None, dtype=pl.String).alias("source_direction"),
        pl.lit(None, dtype=pl.Int64).alias("source_known_at_ms"),
        pl.lit(None, dtype=pl.Int64).alias("source_age_ms"),
        pl.lit("missing").alias("source_freshness"),
        pl.lit("source_missing").alias("source_market_alignment"),
    ).select(*POTENTIAL_OBSERVATION_SCHEMA.keys())


def _potential_state_columns(
    kline_history: pl.DataFrame, timeframe: str, prefix: str
) -> pl.DataFrame:
    frame = kline_history.filter(pl.col("timeframe") == timeframe)
    if frame.is_empty():
        return pl.DataFrame()
    if prefix == "background":
        return frame.select(
            "symbol",
            "bar_close_ms",
            pl.col("regime_state").alias("background_regime"),
            pl.col("structure_state").alias("background_structure"),
            pl.col("range_state").alias("background_range"),
            pl.col("vol_state").alias("background_vol"),
        )
    if prefix == "swing":
        return frame.with_columns(
            pl.concat_str("core_context", "transition_kind", separator="|").alias(
                "swing_transition"
            )
        ).select(
            "symbol",
            "bar_close_ms",
            pl.col("regime_state").alias("swing_regime"),
            pl.col("core_context").alias("swing_core"),
            pl.col("range_state").alias("swing_range"),
            "swing_transition",
        )
    return frame.with_columns(
        pl.concat_str("core_context", "transition_kind", separator="|").alias("decision_transition")
    ).select(
        "symbol",
        "bar_close_ms",
        pl.col("direction_hint").alias("decision_direction"),
        pl.col("regime_state").alias("decision_regime"),
        pl.col("core_context").alias("decision_core"),
        pl.col("range_state").alias("decision_range"),
        pl.col("vol_state").alias("decision_vol"),
        pl.col("event_state").alias("decision_event"),
        pl.col("event_age_bucket").alias("decision_event_age_bucket"),
        "decision_transition",
    )


def _potential_state_output_columns(prefix: str) -> tuple[str, ...]:
    if prefix == "background":
        return ("background_regime", "background_structure", "background_range", "background_vol")
    if prefix == "swing":
        return ("swing_regime", "swing_core", "swing_range", "swing_transition")
    return (
        "decision_direction",
        "decision_regime",
        "decision_core",
        "decision_range",
        "decision_vol",
        "decision_event",
        "decision_event_age_bucket",
        "decision_transition",
    )


def potential_outcome_frame(
    observations: pl.DataFrame,
    source_outcomes: pl.DataFrame,
    realized_transitions: pl.DataFrame,
    *,
    return_threshold_pct: float,
) -> pl.DataFrame:
    market = (
        observations.unique(
            subset=["symbol", "decision_timeframe", "decision_bar_close_ms"], keep="first"
        )
        .join(
            realized_transitions,
            left_on=("symbol", "decision_timeframe", "decision_bar_close_ms"),
            right_on=("symbol", "timeframe", "bar_close_ms"),
            how="inner",
        )
        .filter(pl.col("terminal_core_context").is_not_null())
        .with_columns(
            pl.lit(None, dtype=pl.String).alias("source_family"),
            pl.lit(None, dtype=pl.String).alias("source_state"),
            pl.lit(None, dtype=pl.String).alias("source_direction"),
            pl.lit(None, dtype=pl.Int64).alias("source_known_at_ms"),
            pl.lit(None, dtype=pl.Int64).alias("source_age_ms"),
            pl.lit("missing").alias("source_freshness"),
            pl.lit("source_missing").alias("source_market_alignment"),
            pl.lit(None, dtype=pl.Float64).alias("forward_return_pct"),
            pl.lit(None, dtype=pl.Float64).alias("forward_min_return_pct"),
            pl.lit(None, dtype=pl.Float64).alias("forward_max_return_pct"),
            pl.lit(None, dtype=pl.Float64).alias("path_range_pct"),
            _terminal_direction_bucket_expr().alias("outcome_bucket"),
            pl.lit(False).alias("tail_up"),
            pl.lit(False).alias("tail_down"),
        )
    )
    frames = [market] if not market.is_empty() else []
    if not source_outcomes.is_empty():
        scored = source_outcomes.filter(
            pl.col("outcome_available") & pl.col("source_state").is_not_null()
        ).with_columns(
            outcome_bucket_expr(return_threshold_pct).alias("outcome_bucket"),
            (pl.col("forward_max_return_pct") >= return_threshold_pct).alias("tail_up"),
            (pl.col("forward_min_return_pct") <= -return_threshold_pct).alias("tail_down"),
        )
        source_joined = observations.join(
            scored.select(
                "symbol",
                "source_family",
                "source_state",
                "known_at_ms",
                "outcome_horizon",
                "forward_return_pct",
                "forward_min_return_pct",
                "forward_max_return_pct",
                "path_range_pct",
                "outcome_bucket",
                "tail_up",
                "tail_down",
            ),
            left_on=("symbol", "source_family", "source_state", "source_known_at_ms"),
            right_on=("symbol", "source_family", "source_state", "known_at_ms"),
            how="inner",
        )
        if not source_joined.is_empty():
            frames.append(
                source_joined.join(
                    realized_transitions,
                    left_on=(
                        "symbol",
                        "decision_timeframe",
                        "decision_bar_close_ms",
                        "outcome_horizon",
                    ),
                    right_on=("symbol", "timeframe", "bar_close_ms", "outcome_horizon"),
                    how="inner",
                ).filter(pl.col("terminal_core_context").is_not_null())
            )
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def _terminal_direction_bucket_expr() -> pl.Expr:
    return (
        pl.when(pl.col("terminal_direction").str.contains("bull|up"))
        .then(pl.lit("up"))
        .when(pl.col("terminal_direction").str.contains("bear|down"))
        .then(pl.lit("down"))
        .otherwise(pl.lit("flat"))
    )


__all__ = [
    "POTENTIAL_OBSERVATION_SCHEMA",
    "SOURCE_KLINE_RECENT_WINDOW_MS",
    "potential_observation_frame",
    "potential_outcome_frame",
]
