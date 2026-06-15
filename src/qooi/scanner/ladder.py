"""Fixed categorical ladder evidence path for the scanner."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from qooi.scanner import entropy_expr
from qooi.scanner.outcome import potential_outcome_frame

SOURCE_KLINE_RECENT_WINDOW_MS = 30 * 24 * 60 * 60 * 1000


@dataclass(frozen=True)
class LadderResult:
    """Ladder path pipeline result. Every field has a concrete type."""

    evidence: pl.DataFrame
    candidates: pl.DataFrame
    ranked: pl.DataFrame
    selection_efficiency: pl.DataFrame
    sections: tuple


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
        _potential_level_metrics(
            market, "market_background", ["background_regime"], baseline=market_baseline
        ),
        _potential_level_metrics(
            market, "market_swing", ["background_regime", "swing_core"], baseline=market_baseline
        ),
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
    recent_market = (
        recent_joined.unique(
            subset=["symbol", "decision_bar_close_ms", "outcome_horizon"], keep="first"
        )
        if not recent_joined.is_empty()
        else pl.DataFrame()
    )
    recent_baseline = _outcome_baseline(recent_market)
    recent_levels = [
        _potential_level_metrics(
            recent_market, "market_background", ["background_regime"], baseline=recent_baseline
        ),
        _potential_level_metrics(
            recent_market,
            "market_swing",
            ["background_regime", "swing_core"],
            baseline=recent_baseline,
        ),
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
    return (
        select_potential_evidence_level(evidence)
        .with_columns(_potential_research_suggestion_expr().alias("research_suggestion"))
        .select(*POTENTIAL_EVIDENCE_SCHEMA.keys())
    )


def _outcome_baseline(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame()
    return (
        frame.group_by("outcome_horizon")
        .agg(
            pl.len().cast(pl.UInt32).alias("baseline_observations"),
            (pl.col("outcome_bucket") == "up").mean().alias("baseline_p_up"),
            (pl.col("outcome_bucket") == "down").mean().alias("baseline_p_down"),
            (pl.col("outcome_bucket") == "flat").mean().alias("baseline_p_flat"),
            pl.col("direction_changed").mean().alias("baseline_direction_change_rate"),
            pl.col("core_context_changed").mean().alias("baseline_core_change_rate"),
            pl.col("returned_to_origin").mean().alias("baseline_returned_to_origin_rate"),
        )
        .with_columns(
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
        baseline = (
            clean.group_by("outcome_horizon")
            .agg(
                pl.len().cast(pl.UInt32).alias("baseline_observations"),
                (pl.col("outcome_bucket") == "up").mean().alias("baseline_p_up"),
                (pl.col("outcome_bucket") == "down").mean().alias("baseline_p_down"),
                (pl.col("outcome_bucket") == "flat").mean().alias("baseline_p_flat"),
                pl.col("direction_changed").mean().alias("baseline_direction_change_rate"),
                pl.col("core_context_changed").mean().alias("baseline_core_change_rate"),
                pl.col("returned_to_origin").mean().alias("baseline_returned_to_origin_rate"),
            )
            .with_columns(
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
        )
    conditioned = (
        clean.group_by("outcome_horizon", *group_columns)
        .agg(
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
        )
        .with_columns(
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
    )
    return (
        conditioned.join(baseline, on="outcome_horizon", how="left")
        .with_columns(
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
        )
        .with_columns(
            pl.max_horizontal(
                "direction_transition_information_gain_bits",
                "core_transition_information_gain_bits",
            ).alias("transition_information_gain_bits")
        )
        .select(
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
        pl.col("transition_information_gain_bits").alias("parent_transition_information_gain_bits"),
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
    is_source_level = pl.col("evidence_level").str.contains("_source")
    absolute_source = (
        is_source_level & (pl.col("information_gain_bits") > 0.1) & (pl.col("symbol_count") >= 5)
    )
    passes = improves_parent | absolute_source
    scored = evidence.with_columns(
        pl.when((pl.col("evidence_status") == "usable_stable_information") & passes)
        .then(3)
        .when((pl.col("evidence_status") == "usable_unstable_information") & passes)
        .then(2)
        .when((pl.col("evidence_status") == "exploratory_information") & passes)
        .then(1)
        .when((pl.col("transition_status") == "usable_stable_transition_information") & passes)
        .then(3)
        .when((pl.col("transition_status") == "usable_unstable_transition_information") & passes)
        .then(2)
        .when((pl.col("transition_status") == "exploratory_transition_information") & passes)
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
    best_status = (
        scored.filter(
            (pl.col("selection_status_rank") >= 2)
            & (pl.col("selection_level_rank") >= 1)
            & (pl.col("information_gain_bits") > 0.0)
        )
        .group_by("outcome_horizon", "statistical_direction")
        .agg(pl.col("selection_status_rank").max().alias("selection_status_rank"))
    )
    if best_status.is_empty():
        return scored.with_columns(pl.lit(False).alias("selected_evidence_level")).drop(
            "selection_status_rank", "selection_level_rank"
        )
    eligible = scored.filter(
        (pl.col("selection_status_rank") >= 2)
        & (pl.col("selection_level_rank") >= 1)
        & (pl.col("information_gain_bits") > 0.0)
    )
    best_level = (
        eligible.join(
            best_status, on=("outcome_horizon", "statistical_direction", "selection_status_rank")
        )
        .group_by("outcome_horizon", "statistical_direction", "selection_status_rank")
        .agg(pl.col("selection_level_rank").min().alias("selection_level_rank"))
    )
    return (
        scored.join(
            best_level.with_columns(pl.lit(True).alias("selected_evidence_level")),
            on=(
                "outcome_horizon",
                "statistical_direction",
                "selection_status_rank",
                "selection_level_rank",
            ),
            how="left",
        )
        .with_columns(pl.col("selected_evidence_level").fill_null(False))
        .drop("selection_status_rank", "selection_level_rank")
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
    return pl.when(probability > 0.0).then(-probability * probability.log(2)).otherwise(
        0.0
    ) + pl.when(complement > 0.0).then(-complement * complement.log(2)).otherwise(0.0)


__all__ = [
    "LadderResult",
    "POTENTIAL_EVIDENCE_SCHEMA",
    "add_potential_parent_gain",
    "potential_evidence_frame",
    "select_potential_evidence_level",
]
