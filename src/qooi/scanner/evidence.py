"""Unified potential scanner observation and evidence surfaces."""

from __future__ import annotations

import polars as pl

from qooi.scanner import entropy_expr, outcome_bucket_expr

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

POTENTIAL_EVIDENCE_SCHEMA = {
    "evidence_level": pl.String,
    "outcome_horizon": pl.Int64,
    "background_regime": pl.String,
    "swing_core": pl.String,
    "decision_core": pl.String,
    "decision_transition": pl.String,
    "source_family": pl.String,
    "source_state": pl.String,
    "risk_context": pl.String,
    "baseline_observations": pl.UInt32,
    "conditioned_observations": pl.UInt32,
    "symbol_count": pl.UInt32,
    "baseline_p_up": pl.Float64,
    "baseline_p_down": pl.Float64,
    "baseline_p_flat": pl.Float64,
    "conditioned_p_up": pl.Float64,
    "conditioned_p_down": pl.Float64,
    "conditioned_p_flat": pl.Float64,
    "lift_up": pl.Float64,
    "lift_down": pl.Float64,
    "lift_flat": pl.Float64,
    "baseline_entropy_bits": pl.Float64,
    "conditioned_entropy_bits": pl.Float64,
    "information_gain_bits": pl.Float64,
    "baseline_direction_change_rate": pl.Float64,
    "conditioned_direction_change_rate": pl.Float64,
    "direction_transition_information_gain_bits": pl.Float64,
    "baseline_core_change_rate": pl.Float64,
    "conditioned_core_change_rate": pl.Float64,
    "core_transition_information_gain_bits": pl.Float64,
    "transition_information_gain_bits": pl.Float64,
    "tail_up_rate": pl.Float64,
    "tail_down_rate": pl.Float64,
    "baseline_returned_to_origin_rate": pl.Float64,
    "returned_to_origin_rate": pl.Float64,
    "avg_forward_max_return_pct": pl.Float64,
    "avg_forward_min_return_pct": pl.Float64,
    "avg_path_range_pct": pl.Float64,
    "path_skew": pl.Float64,
    "recent_conditioned_observations": pl.UInt32,
    "recent_symbol_count": pl.UInt32,
    "recent_information_gain_bits": pl.Float64,
    "recent_transition_information_gain_bits": pl.Float64,
    "information_stability": pl.Float64,
    "transition_information_stability": pl.Float64,
    "parent_evidence_level": pl.String,
    "parent_information_gain_bits": pl.Float64,
    "information_gain_over_parent": pl.Float64,
    "parent_transition_information_gain_bits": pl.Float64,
    "transition_information_gain_over_parent": pl.Float64,
    "evidence_status": pl.String,
    "transition_status": pl.String,
    "selected_evidence_level": pl.Boolean,
    "statistical_direction": pl.String,
    "research_suggestion": pl.String,
}


def potential_observation_frame(
    kline_history: pl.DataFrame,
    source_events: pl.DataFrame,
    *,
    decision_timeframe: str,
    max_source_staleness_hours: int,
) -> pl.DataFrame:
    if kline_history.is_empty() or decision_timeframe not in kline_history.get_column(
        "timeframe"
    ).to_list():
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
        return _potential_observation_without_source(observations)
    source_rows = source_events.filter(pl.col("source_state").is_not_null()).select(
        "symbol",
        "source_family",
        "source_state",
        "source_direction",
        pl.col("known_at_ms").alias("source_known_at_ms"),
    )
    if source_rows.is_empty():
        return _potential_observation_without_source(observations)
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
        return _potential_observation_without_source(observations)
    return pl.concat(frames, how="vertical_relaxed").select(*POTENTIAL_OBSERVATION_SCHEMA.keys())


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
        pl.concat_str("core_context", "transition_kind", separator="|").alias(
            "decision_transition"
        )
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


def potential_evidence_frame(
    observations: pl.DataFrame,
    source_outcomes: pl.DataFrame,
    realized_transitions: pl.DataFrame,
    *,
    return_threshold_pct: float,
) -> pl.DataFrame:
    if observations.is_empty() or realized_transitions.is_empty():
        return pl.DataFrame(schema=POTENTIAL_EVIDENCE_SCHEMA)
    joined = potential_outcome_frame(
        observations,
        source_outcomes,
        realized_transitions,
        return_threshold_pct=return_threshold_pct,
    )
    if joined.is_empty():
        return pl.DataFrame(schema=POTENTIAL_EVIDENCE_SCHEMA)
    market = joined.unique(
        subset=["symbol", "decision_bar_close_ms", "outcome_horizon"], keep="first"
    )
    market_baseline = _outcome_baseline(market)
    levels = [
        _potential_level_metrics(market, "market_background", ["background_regime"],
                                 baseline=market_baseline),
        _potential_level_metrics(market, "market_swing",
                                 ["background_regime", "swing_core"],
                                 baseline=market_baseline),
        _potential_level_metrics(
            market,
            "market_decision",
            ["background_regime", "swing_core", "decision_core", "decision_transition"],
            baseline=market_baseline,
        ),
        _potential_level_metrics(
            joined.filter(pl.col("source_state").is_not_null()),
            "market_decision_source",
            [
                "background_regime",
                "swing_core",
                "decision_core",
                "decision_transition",
                "source_family",
                "source_state",
            ],
            baseline=market_baseline,
        ),
        _potential_level_metrics(
            joined.filter(pl.col("source_state").is_not_null()),
            "market_decision_source_risk",
            [
                "background_regime",
                "swing_core",
                "decision_core",
                "decision_transition",
                "source_family",
                "source_state",
                "risk_context",
            ],
            baseline=market_baseline,
        ),
    ]
    level_frames = [frame for frame in levels if not frame.is_empty()]
    if not level_frames:
        return pl.DataFrame(schema=POTENTIAL_EVIDENCE_SCHEMA)
    evidence = pl.concat(level_frames, how="vertical_relaxed")
    latest = joined.get_column("decision_bar_close_ms").drop_nulls().max()
    recent_joined = (
        joined.filter(
            pl.col("decision_bar_close_ms") >= int(latest) - SOURCE_KLINE_RECENT_WINDOW_MS
        )
        if latest is not None
        else pl.DataFrame()
    )
    recent_market = recent_joined.unique(
        subset=["symbol", "decision_bar_close_ms", "outcome_horizon"], keep="first"
    ) if not recent_joined.is_empty() else pl.DataFrame()
    recent_baseline = _outcome_baseline(recent_market)
    recent_levels = [
        _potential_level_metrics(recent_market, "market_background",
                                 ["background_regime"], baseline=recent_baseline),
        _potential_level_metrics(recent_market, "market_swing",
                                 ["background_regime", "swing_core"],
                                 baseline=recent_baseline),
        _potential_level_metrics(
            recent_market,
            "market_decision",
            ["background_regime", "swing_core", "decision_core", "decision_transition"],
            baseline=recent_baseline,
        ),
        _potential_level_metrics(
            recent_joined.filter(pl.col("source_state").is_not_null())
            if not recent_joined.is_empty()
            else pl.DataFrame(),
            "market_decision_source",
            [
                "background_regime",
                "swing_core",
                "decision_core",
                "decision_transition",
                "source_family",
                "source_state",
            ],
            baseline=recent_baseline,
        ),
        _potential_level_metrics(
            recent_joined.filter(pl.col("source_state").is_not_null())
            if not recent_joined.is_empty()
            else pl.DataFrame(),
            "market_decision_source_risk",
            [
                "background_regime",
                "swing_core",
                "decision_core",
                "decision_transition",
                "source_family",
                "source_state",
                "risk_context",
            ],
            baseline=recent_baseline,
        ),
    ]
    recent_level_frames = [frame for frame in recent_levels if not frame.is_empty()]
    recent = (
        pl.concat(recent_level_frames, how="vertical_relaxed")
        if recent_level_frames
        else pl.DataFrame()
    )
    if recent.is_empty():
        evidence = evidence.with_columns(
            pl.lit(0, dtype=pl.UInt32).alias("recent_conditioned_observations"),
            pl.lit(0, dtype=pl.UInt32).alias("recent_symbol_count"),
            pl.lit(0.0).alias("recent_information_gain_bits"),
            pl.lit(0.0).alias("recent_transition_information_gain_bits"),
        )
    else:
        join_cols = _potential_evidence_identity_columns()
        evidence = evidence.join(
            recent.select(
                *join_cols,
                pl.col("conditioned_observations").alias("recent_conditioned_observations"),
                pl.col("symbol_count").alias("recent_symbol_count"),
                pl.col("information_gain_bits").alias("recent_information_gain_bits"),
                pl.col("transition_information_gain_bits").alias(
                    "recent_transition_information_gain_bits"
                ),
            ),
            on=join_cols,
            how="left",
        ).with_columns(
            pl.col("recent_conditioned_observations").fill_null(0).cast(pl.UInt32),
            pl.col("recent_symbol_count").fill_null(0).cast(pl.UInt32),
            pl.col("recent_information_gain_bits").fill_null(0.0),
            pl.col("recent_transition_information_gain_bits").fill_null(0.0),
        )
    evidence = evidence.with_columns(
        (
            pl.col("recent_information_gain_bits")
            / pl.when(pl.col("information_gain_bits").abs() > 1e-12)
            .then(pl.col("information_gain_bits").abs())
            .otherwise(None)
        )
        .fill_null(0.0)
        .alias("information_stability"),
        (
            pl.col("recent_transition_information_gain_bits")
            / pl.when(pl.col("transition_information_gain_bits").abs() > 1e-12)
            .then(pl.col("transition_information_gain_bits").abs())
            .otherwise(None)
        )
        .fill_null(0.0)
        .alias("transition_information_stability"),
    ).with_columns(
        pl.when(
            pl.col("conditioned_p_up")
            > pl.max_horizontal("conditioned_p_down", "conditioned_p_flat")
        )
        .then(pl.lit("up"))
        .when(
            pl.col("conditioned_p_down")
            > pl.max_horizontal("conditioned_p_up", "conditioned_p_flat")
        )
        .then(pl.lit("down"))
        .otherwise(pl.lit("flat"))
        .alias("statistical_direction")
    )
    evidence = add_potential_parent_gain(evidence)
    evidence = evidence.with_columns(
        pl.when(
            (pl.col("conditioned_observations") >= 100)
            & (pl.col("symbol_count") >= 20)
            & (pl.col("information_gain_bits") > 0.0)
            & (pl.col("recent_conditioned_observations") >= 30)
            & (pl.col("recent_information_gain_bits") > 0.0)
        )
        .then(pl.lit("usable_stable_information"))
        .when(
            (pl.col("conditioned_observations") >= 100)
            & (pl.col("symbol_count") >= 20)
            & (pl.col("information_gain_bits") > 0.0)
        )
        .then(pl.lit("usable_unstable_information"))
        .when(
            (pl.col("conditioned_observations") >= 50)
            & (pl.col("symbol_count") >= 12)
            & (pl.col("information_gain_bits") > 0.0)
        )
        .then(pl.lit("exploratory_information"))
        .otherwise(pl.lit("insufficient_information"))
        .alias("evidence_status"),
        pl.when(
            (pl.col("conditioned_observations") >= 100)
            & (pl.col("symbol_count") >= 20)
            & (pl.col("transition_information_gain_bits") > 0.0)
            & (pl.col("recent_conditioned_observations") >= 30)
            & (pl.col("recent_transition_information_gain_bits") > 0.0)
        )
        .then(pl.lit("usable_stable_transition_information"))
        .when(
            (pl.col("conditioned_observations") >= 100)
            & (pl.col("symbol_count") >= 20)
            & (pl.col("transition_information_gain_bits") > 0.0)
        )
        .then(pl.lit("usable_unstable_transition_information"))
        .when(
            (pl.col("conditioned_observations") >= 50)
            & (pl.col("symbol_count") >= 12)
            & (pl.col("transition_information_gain_bits") > 0.0)
        )
        .then(pl.lit("exploratory_transition_information"))
        .otherwise(pl.lit("insufficient_transition_information"))
        .alias("transition_status"),
    )
    return select_potential_evidence_level(evidence).with_columns(
        _potential_research_suggestion_expr().alias("research_suggestion")
    ).select(*POTENTIAL_EVIDENCE_SCHEMA.keys())


def potential_outcome_frame(
    observations: pl.DataFrame,
    source_outcomes: pl.DataFrame,
    realized_transitions: pl.DataFrame,
    *,
    return_threshold_pct: float,
) -> pl.DataFrame:
    market = observations.unique(
        subset=["symbol", "decision_timeframe", "decision_bar_close_ms"], keep="first"
    ).join(
        realized_transitions,
        left_on=("symbol", "decision_timeframe", "decision_bar_close_ms"),
        right_on=("symbol", "timeframe", "bar_close_ms"),
        how="inner",
    ).filter(pl.col("terminal_core_context").is_not_null()).with_columns(
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


def _outcome_baseline(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame()
    return frame.group_by("outcome_horizon").agg(
        pl.len().cast(pl.UInt32).alias("baseline_observations"),
        (pl.col("outcome_bucket") == "up").mean().alias("baseline_p_up"),
        (pl.col("outcome_bucket") == "down").mean().alias("baseline_p_down"),
        (pl.col("outcome_bucket") == "flat").mean().alias("baseline_p_flat"),
        pl.col("direction_changed").mean().alias("baseline_direction_change_rate"),
        pl.col("core_context_changed").mean().alias("baseline_core_change_rate"),
        pl.col("returned_to_origin").mean().alias("baseline_returned_to_origin_rate"),
    ).with_columns(
        entropy_expr("baseline_p_up", "baseline_p_down", "baseline_p_flat").alias(
            "baseline_entropy_bits"
        ),
        _binary_entropy_expr("baseline_direction_change_rate").alias(
            "baseline_direction_change_entropy_bits"
        ),
        _binary_entropy_expr("baseline_core_change_rate").alias(
            "baseline_core_change_entropy_bits"
        ),
    )


def _potential_level_metrics(
    frame: pl.DataFrame,
    evidence_level: str,
    group_columns: list[str],
    *,
    baseline: pl.DataFrame | None = None,
) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame()
    clean = frame.filter(pl.all_horizontal(pl.col(col).is_not_null() for col in group_columns))
    if clean.is_empty():
        return pl.DataFrame()
    if baseline is None or baseline.is_empty():
        baseline = clean.group_by("outcome_horizon").agg(
            pl.len().cast(pl.UInt32).alias("baseline_observations"),
            (pl.col("outcome_bucket") == "up").mean().alias("baseline_p_up"),
            (pl.col("outcome_bucket") == "down").mean().alias("baseline_p_down"),
            (pl.col("outcome_bucket") == "flat").mean().alias("baseline_p_flat"),
            pl.col("direction_changed").mean().alias("baseline_direction_change_rate"),
            pl.col("core_context_changed").mean().alias("baseline_core_change_rate"),
            pl.col("returned_to_origin").mean().alias("baseline_returned_to_origin_rate"),
        ).with_columns(
            entropy_expr("baseline_p_up", "baseline_p_down", "baseline_p_flat").alias(
                "baseline_entropy_bits"
            ),
            _binary_entropy_expr("baseline_direction_change_rate").alias(
                "baseline_direction_change_entropy_bits"
            ),
            _binary_entropy_expr("baseline_core_change_rate").alias(
                "baseline_core_change_entropy_bits"
            ),
        )
    conditioned = clean.group_by("outcome_horizon", *group_columns).agg(
        pl.len().cast(pl.UInt32).alias("conditioned_observations"),
        pl.col("symbol").n_unique().cast(pl.UInt32).alias("symbol_count"),
        (pl.col("outcome_bucket") == "up").mean().alias("conditioned_p_up"),
        (pl.col("outcome_bucket") == "down").mean().alias("conditioned_p_down"),
        (pl.col("outcome_bucket") == "flat").mean().alias("conditioned_p_flat"),
        pl.col("tail_up").mean().alias("tail_up_rate"),
        pl.col("tail_down").mean().alias("tail_down_rate"),
        pl.col("direction_changed").mean().alias("conditioned_direction_change_rate"),
        pl.col("core_context_changed").mean().alias("conditioned_core_change_rate"),
        pl.col("returned_to_origin").mean().alias("returned_to_origin_rate"),
        pl.col("forward_max_return_pct").mean().alias("avg_forward_max_return_pct"),
        pl.col("forward_min_return_pct").mean().alias("avg_forward_min_return_pct"),
        pl.col("path_range_pct").mean().alias("avg_path_range_pct"),
    ).with_columns(
        entropy_expr("conditioned_p_up", "conditioned_p_down", "conditioned_p_flat").alias(
            "conditioned_entropy_bits"
        ),
        _binary_entropy_expr("conditioned_direction_change_rate").alias(
            "conditioned_direction_change_entropy_bits"
        ),
        _binary_entropy_expr("conditioned_core_change_rate").alias(
            "conditioned_core_change_entropy_bits"
        ),
    )
    return conditioned.join(baseline, on="outcome_horizon", how="left").with_columns(
        pl.lit(evidence_level).alias("evidence_level"),
        (pl.col("conditioned_p_up") - pl.col("baseline_p_up")).alias("lift_up"),
        (pl.col("conditioned_p_down") - pl.col("baseline_p_down")).alias("lift_down"),
        (pl.col("conditioned_p_flat") - pl.col("baseline_p_flat")).alias("lift_flat"),
        (pl.col("baseline_entropy_bits") - pl.col("conditioned_entropy_bits")).alias(
            "information_gain_bits"
        ),
        (
            pl.col("baseline_direction_change_entropy_bits")
            - pl.col("conditioned_direction_change_entropy_bits")
        ).alias("direction_transition_information_gain_bits"),
        (
            pl.col("baseline_core_change_entropy_bits")
            - pl.col("conditioned_core_change_entropy_bits")
        ).alias("core_transition_information_gain_bits"),
        (pl.col("tail_up_rate") - pl.col("tail_down_rate")).alias("path_skew"),
        *[
            pl.lit(None, dtype=pl.String).alias(col)
            for col in _potential_evidence_role_columns()
            if col not in group_columns
        ],
    ).with_columns(
        pl.max_horizontal(
            "direction_transition_information_gain_bits", "core_transition_information_gain_bits"
        ).alias("transition_information_gain_bits")
    ).select(
        "evidence_level",
        "outcome_horizon",
        *_potential_evidence_role_columns(),
        "baseline_observations",
        "conditioned_observations",
        "symbol_count",
        "baseline_p_up",
        "baseline_p_down",
        "baseline_p_flat",
        "conditioned_p_up",
        "conditioned_p_down",
        "conditioned_p_flat",
        "lift_up",
        "lift_down",
        "lift_flat",
        "baseline_entropy_bits",
        "conditioned_entropy_bits",
        "information_gain_bits",
        "baseline_direction_change_rate",
        "conditioned_direction_change_rate",
        "direction_transition_information_gain_bits",
        "baseline_core_change_rate",
        "conditioned_core_change_rate",
        "core_transition_information_gain_bits",
        "transition_information_gain_bits",
        "tail_up_rate",
        "tail_down_rate",
        "baseline_returned_to_origin_rate",
        "returned_to_origin_rate",
        "avg_forward_max_return_pct",
        "avg_forward_min_return_pct",
        "avg_path_range_pct",
        "path_skew",
    )


def _potential_evidence_role_columns() -> tuple[str, ...]:
    return (
        "background_regime",
        "swing_core",
        "decision_core",
        "decision_transition",
        "source_family",
        "source_state",
        "risk_context",
    )


def _potential_evidence_identity_columns() -> tuple[str, ...]:
    return ("evidence_level", "outcome_horizon", *_potential_evidence_role_columns())


def add_potential_parent_gain(evidence: pl.DataFrame) -> pl.DataFrame:
    keyed = evidence.with_columns(
        pl.when(pl.col("evidence_level") == "market_swing")
        .then(pl.lit("market_background"))
        .when(pl.col("evidence_level") == "market_decision")
        .then(pl.lit("market_swing"))
        .when(pl.col("evidence_level") == "market_decision_source")
        .then(pl.lit("market_decision"))
        .when(pl.col("evidence_level") == "market_decision_source_risk")
        .then(pl.lit("market_decision_source"))
        .otherwise(None)
        .alias("parent_evidence_level")
    )
    parent = keyed.select(
        "evidence_level",
        "outcome_horizon",
        *_potential_evidence_role_columns(),
        pl.col("information_gain_bits").alias("parent_information_gain_bits"),
        pl.col("transition_information_gain_bits").alias(
            "parent_transition_information_gain_bits"
        ),
    )
    roots = keyed.filter(pl.col("parent_evidence_level").is_null()).with_columns(
        pl.lit(None, dtype=pl.Float64).alias("parent_information_gain_bits"),
        pl.lit(None, dtype=pl.Float64).alias("parent_transition_information_gain_bits"),
    )
    swing = keyed.filter(pl.col("evidence_level") == "market_swing").join(
        parent.filter(pl.col("evidence_level") == "market_background").select(
            "outcome_horizon",
            "background_regime",
            "parent_information_gain_bits",
            "parent_transition_information_gain_bits",
        ),
        on=("outcome_horizon", "background_regime"),
        how="left",
    )
    decision = keyed.filter(pl.col("evidence_level") == "market_decision").join(
        parent.filter(pl.col("evidence_level") == "market_swing").select(
            "outcome_horizon",
            "background_regime",
            "swing_core",
            "parent_information_gain_bits",
            "parent_transition_information_gain_bits",
        ),
        on=("outcome_horizon", "background_regime", "swing_core"),
        how="left",
    )
    source = keyed.filter(pl.col("evidence_level") == "market_decision_source").join(
        parent.filter(pl.col("evidence_level") == "market_decision").select(
            "outcome_horizon",
            "background_regime",
            "swing_core",
            "decision_core",
            "decision_transition",
            "parent_information_gain_bits",
            "parent_transition_information_gain_bits",
        ),
        on=(
            "outcome_horizon",
            "background_regime",
            "swing_core",
            "decision_core",
            "decision_transition",
        ),
        how="left",
    )
    risk = keyed.filter(pl.col("evidence_level") == "market_decision_source_risk").join(
        parent.filter(pl.col("evidence_level") == "market_decision_source").select(
            "outcome_horizon",
            "background_regime",
            "swing_core",
            "decision_core",
            "decision_transition",
            "source_family",
            "source_state",
            "parent_information_gain_bits",
            "parent_transition_information_gain_bits",
        ),
        on=(
            "outcome_horizon",
            "background_regime",
            "swing_core",
            "decision_core",
            "decision_transition",
            "source_family",
            "source_state",
        ),
        how="left",
    )
    children = [frame for frame in (roots, swing, decision, source, risk) if not frame.is_empty()]
    return pl.concat(children, how="vertical_relaxed").with_columns(
        (pl.col("information_gain_bits") - pl.col("parent_information_gain_bits")).alias(
            "information_gain_over_parent"
        ),
        (
            pl.col("transition_information_gain_bits")
            - pl.col("parent_transition_information_gain_bits")
        ).alias("transition_information_gain_over_parent"),
    )


def select_potential_evidence_level(evidence: pl.DataFrame) -> pl.DataFrame:
    improves_parent = pl.col("parent_evidence_level").is_null() | (
        (pl.col("information_gain_over_parent") > 0.0)
        | (pl.col("transition_information_gain_over_parent") > 0.0)
    )
    scored = evidence.with_columns(
        pl.when((pl.col("evidence_status") == "usable_stable_information") & improves_parent)
        .then(3)
        .when((pl.col("evidence_status") == "usable_unstable_information") & improves_parent)
        .then(2)
        .when((pl.col("evidence_status") == "exploratory_information") & improves_parent)
        .then(1)
        .when(
            (pl.col("transition_status") == "usable_stable_transition_information")
            & improves_parent
        )
        .then(3)
        .when(
            (pl.col("transition_status") == "usable_unstable_transition_information")
            & improves_parent
        )
        .then(2)
        .when(
            (pl.col("transition_status") == "exploratory_transition_information")
            & improves_parent
        )
        .then(1)
        .otherwise(0)
        .alias("selection_status_rank"),
        pl.when(pl.col("evidence_level") == "market_background")
        .then(0)
        .when(pl.col("evidence_level") == "market_swing")
        .then(1)
        .when(pl.col("evidence_level") == "market_decision")
        .then(2)
        .when(pl.col("evidence_level") == "market_decision_source")
        .then(3)
        .when(pl.col("evidence_level") == "market_decision_source_risk")
        .then(4)
        .otherwise(5)
        .alias("selection_level_rank"),
    )
    best_status = scored.filter(
        (pl.col("selection_status_rank") >= 2)
        & (pl.col("selection_level_rank") >= 1)
        & (pl.col("information_gain_bits") > 0.0)
    ).group_by(
        "outcome_horizon", "statistical_direction"
    ).agg(pl.col("selection_status_rank").max().alias("selection_status_rank"))
    if best_status.is_empty():
        return scored.with_columns(pl.lit(False).alias("selected_evidence_level")).drop(
            "selection_status_rank", "selection_level_rank"
        )
    best_level = scored.filter(
        (pl.col("selection_status_rank") >= 2) & (pl.col("selection_level_rank") >= 1)
    ).join(
        best_status, on=("outcome_horizon", "statistical_direction", "selection_status_rank")
    ).group_by("outcome_horizon", "statistical_direction", "selection_status_rank").agg(
        pl.col("selection_level_rank").min().alias("selection_level_rank")
    )
    return scored.join(
        best_level.with_columns(pl.lit(True).alias("selected_evidence_level")),
        on=(
            "outcome_horizon",
            "statistical_direction",
            "selection_status_rank",
            "selection_level_rank",
        ),
        how="left",
    ).with_columns(pl.col("selected_evidence_level").fill_null(False)).drop(
        "selection_status_rank", "selection_level_rank"
    )


def _potential_research_suggestion_expr() -> pl.Expr:
    selected = pl.col("selected_evidence_level")
    return (
        pl.when(
            selected
            & (pl.col("returned_to_origin_rate") >= 0.25)
            & (pl.col("path_skew").abs() <= 0.10)
        )
        .then(pl.lit("chop_avoid"))
        .when(
            selected
            & (
                pl.col("conditioned_direction_change_rate")
                > pl.col("baseline_direction_change_rate")
            )
            & (pl.col("avg_path_range_pct") > 0.0)
        )
        .then(pl.lit("volatility_expansion_watch"))
        .when(
            selected
            & (pl.col("conditioned_core_change_rate") > pl.col("baseline_core_change_rate"))
        )
        .then(pl.lit("rapid_trend_watch"))
        .when(
            selected
            & (pl.col("returned_to_origin_rate") > pl.col("baseline_returned_to_origin_rate"))
        )
        .then(pl.lit("mean_reversion_watch"))
        .otherwise(pl.lit("insufficient_evidence"))
    )


def _binary_entropy_expr(probability_col: str) -> pl.Expr:
    probability = pl.col(probability_col).cast(pl.Float64)
    complement = 1.0 - probability
    return (
        pl.when(probability > 0.0).then(-probability * probability.log(2)).otherwise(0.0)
        + pl.when(complement > 0.0).then(-complement * complement.log(2)).otherwise(0.0)
    )
