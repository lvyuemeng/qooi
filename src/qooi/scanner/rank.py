"""Promoted scanner candidate ranking boundary."""

from __future__ import annotations

from typing import Protocol

import polars as pl


class TailTreePredictor(Protocol):
    def predict_leaf(self, features: pl.DataFrame) -> pl.DataFrame: ...


EVIDENCE_LEVEL_COLUMNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("market_background", ("background_regime",)),
    ("market_swing", ("background_regime", "swing_core")),
    (
        "market_decision",
        ("background_regime", "swing_core", "decision_core", "decision_transition"),
    ),
    (
        "market_decision_source",
        (
            "background_regime",
            "swing_core",
            "decision_core",
            "decision_transition",
            "source_family",
            "source_state",
        ),
    ),
    (
        "market_decision_source_risk",
        (
            "background_regime",
            "swing_core",
            "decision_core",
            "decision_transition",
            "source_family",
            "source_state",
            "risk_context",
        ),
    ),
)

OBSERVATION_COLUMNS = (
    "symbol",
    "decision_timeframe",
    "decision_bar_close_ms",
    "source_freshness",
    "source_age_ms",
    "market_alignment",
    "source_market_alignment",
)

EVIDENCE_VALUE_COLUMNS = (
    "conditioned_observations",
    "symbol_count",
    "conditioned_p_up",
    "conditioned_p_down",
    "conditioned_p_flat",
    "lift_up",
    "lift_down",
    "lift_flat",
    "information_gain_bits",
    "transition_information_gain_bits",
    "tail_up_rate",
    "tail_down_rate",
    "avg_forward_max_return_pct",
    "avg_forward_min_return_pct",
    "avg_path_range_pct",
    "path_skew",
    "returned_to_origin_rate",
    "information_stability",
    "transition_information_stability",
    "evidence_status",
    "transition_status",
    "statistical_direction",
    "research_suggestion",
)

CANDIDATE_EVIDENCE_SCHEMA = {
    "symbol": pl.String,
    "decision_timeframe": pl.String,
    "decision_bar_close_ms": pl.Int64,
    "outcome_horizon": pl.Int64,
    "matched_evidence_level": pl.String,
    "candidate_status": pl.String,
    "statistical_direction": pl.String,
    "research_suggestion": pl.String,
    "conditioned_observations": pl.UInt32,
    "symbol_count": pl.UInt32,
    "conditioned_p_up": pl.Float64,
    "conditioned_p_down": pl.Float64,
    "conditioned_p_flat": pl.Float64,
    "lift_up": pl.Float64,
    "lift_down": pl.Float64,
    "lift_flat": pl.Float64,
    "information_gain_bits": pl.Float64,
    "transition_information_gain_bits": pl.Float64,
    "tail_up_rate": pl.Float64,
    "tail_down_rate": pl.Float64,
    "avg_forward_max_return_pct": pl.Float64,
    "avg_forward_min_return_pct": pl.Float64,
    "avg_path_range_pct": pl.Float64,
    "path_skew": pl.Float64,
    "returned_to_origin_rate": pl.Float64,
    "information_stability": pl.Float64,
    "transition_information_stability": pl.Float64,
    "evidence_status": pl.String,
    "transition_status": pl.String,
    "source_freshness": pl.String,
    "source_age_ms": pl.Int64,
    "market_alignment": pl.String,
    "source_market_alignment": pl.String,
    # Tree path columns (nullable — ladder path produces null)
    "tail_lift": pl.Float64,
    "gpd_shape_xi": pl.Float64,
    "gpd_scale_sigma": pl.Float64,
    "tail_lift_stability": pl.Float64,
    "N_tail_exceedances": pl.UInt32,
    "leaf_id": pl.Int32,
    "tree_direction": pl.String,
    "leaf_path": pl.String,
}


CANDIDATE_RANK_SCHEMA = CANDIDATE_EVIDENCE_SCHEMA | {
    "rank_information_component": pl.Float64,
    "rank_transition_component": pl.Float64,
    "rank_tail_component": pl.Float64,
    "rank_path_component": pl.Float64,
    "rank_stability_component": pl.Float64,
    "rank_quality_component": pl.Float64,
    "rank_penalty_component": pl.Float64,
    "rank_score": pl.Float64,
    "rank_reason": pl.String,
}


def candidate_evidence_frame(
    observations: pl.DataFrame,
    evidence: pl.DataFrame,
    *,
    latest_only: bool = True,
    tree_up: TailTreePredictor | None = None,
    tree_down: TailTreePredictor | None = None,
) -> pl.DataFrame:
    """Match observation rows to selected evidence rows."""
    if observations.is_empty():
        return pl.DataFrame(schema=CANDIDATE_EVIDENCE_SCHEMA)
    if tree_up is not None or tree_down is not None:
        return _candidate_from_trees(
            observations, evidence, tree_up, tree_down, latest_only=latest_only
        )
    base = _candidate_observations(observations, latest_only=latest_only)
    selected = _selected_evidence(evidence)
    if selected.is_empty():
        return _unmatched_candidates(base, "no_selected_evidence")
    frames = [
        _matches_for_level(base, selected, level, columns)
        for level, columns in EVIDENCE_LEVEL_COLUMNS
    ]
    non_empty_frames = [frame for frame in frames if not frame.is_empty()]
    matched = (
        pl.concat(non_empty_frames, how="diagonal_relaxed")
        if non_empty_frames
        else pl.DataFrame(schema=CANDIDATE_EVIDENCE_SCHEMA)
    )
    matched = _select_schema(_best_candidate_per_observation(matched), CANDIDATE_EVIDENCE_SCHEMA)
    unmatched = _unmatched_observations(base, matched)
    if not unmatched.is_empty():
        matched = pl.concat([matched, unmatched], how="vertical_relaxed")
    return _select_schema(matched, CANDIDATE_EVIDENCE_SCHEMA)


def _candidate_from_trees(
    observations: pl.DataFrame,
    evidence: pl.DataFrame,
    tree_up: TailTreePredictor | None,
    tree_down: TailTreePredictor | None,
    *,
    latest_only: bool = True,
) -> pl.DataFrame:
    base = _candidate_observations(observations, latest_only=latest_only)
    if base.is_empty():
        return pl.DataFrame(schema=CANDIDATE_EVIDENCE_SCHEMA)

    frames = []
    for tree, direction in ((tree_up, "up"), (tree_down, "down")):
        if tree is None:
            continue
        try:
            with_leaf = tree.predict_leaf(base)
        except Exception:
            continue
        dir_evidence = (
            evidence.filter(pl.col("tree_direction") == direction)
            if "tree_direction" in evidence.columns
            else evidence
        )
        if dir_evidence.is_empty():
            continue
        matched = with_leaf.join(
            dir_evidence,
            left_on="leaf_id",
            right_on="leaf_id",
            how="inner",
        )
        if matched.is_empty():
            continue
        matched = matched.with_columns(
            pl.lit(f"tree_{direction}").alias("matched_evidence_level"),
            pl.lit("matched_evidence").alias("candidate_status"),
        )
        frames.append(matched)

    if not frames:
        return _unmatched_candidates(base, "no_tree_match")

    result = pl.concat(frames, how="diagonal_relaxed")
    return _select_schema(result, CANDIDATE_EVIDENCE_SCHEMA)


def _candidate_observations(observations: pl.DataFrame, *, latest_only: bool) -> pl.DataFrame:
    base = observations
    if latest_only:
        latest = base.group_by("symbol").agg(
            pl.col("decision_bar_close_ms").max().alias("decision_bar_close_ms")
        )
        base = base.join(latest, on=("symbol", "decision_bar_close_ms"), how="inner")
    return base


def _selected_evidence(evidence: pl.DataFrame) -> pl.DataFrame:
    if evidence.is_empty() or "selected_evidence_level" not in evidence.columns:
        return pl.DataFrame()
    return evidence.filter(pl.col("selected_evidence_level"))


def _matches_for_level(
    observations: pl.DataFrame,
    selected: pl.DataFrame,
    level: str,
    columns: tuple[str, ...],
) -> pl.DataFrame:
    evidence = selected.filter(pl.col("evidence_level") == level)
    if evidence.is_empty() or any(column not in observations.columns for column in columns):
        return pl.DataFrame(schema=CANDIDATE_EVIDENCE_SCHEMA)
    value_columns = [column for column in EVIDENCE_VALUE_COLUMNS if column in evidence.columns]
    joined = observations.join(
        evidence.select("outcome_horizon", *columns, *value_columns),
        on=list(columns),
        how="inner",
    )
    if joined.is_empty():
        return pl.DataFrame(schema=CANDIDATE_EVIDENCE_SCHEMA)
    return joined.with_columns(
        pl.lit(level).alias("matched_evidence_level"),
        pl.lit("matched_evidence").alias("candidate_status"),
    )


def _best_candidate_per_observation(matched: pl.DataFrame) -> pl.DataFrame:
    if matched.is_empty():
        return matched
    return (
        matched.with_columns(
            pl.when(pl.col("matched_evidence_level") == "market_background")
            .then(0)
            .when(pl.col("matched_evidence_level") == "market_swing")
            .then(1)
            .when(pl.col("matched_evidence_level") == "market_decision")
            .then(2)
            .when(pl.col("matched_evidence_level") == "market_decision_source")
            .then(3)
            .when(pl.col("matched_evidence_level") == "market_decision_source_risk")
            .then(4)
            .otherwise(5)
            .alias("_level_rank"),
            pl.col("information_gain_bits").fill_null(0.0).alias("_information_rank"),
            pl.col("transition_information_gain_bits").fill_null(0.0).alias("_transition_rank"),
        )
        .sort(
            [
                "symbol",
                "decision_timeframe",
                "decision_bar_close_ms",
                "outcome_horizon",
                "_level_rank",
                "_information_rank",
                "_transition_rank",
            ],
            descending=[False, False, False, False, True, True, True],
        )
        .unique(
            subset=(
                "symbol",
                "decision_timeframe",
                "decision_bar_close_ms",
                "outcome_horizon",
            ),
            keep="first",
        )
        .drop("_level_rank", "_information_rank", "_transition_rank")
    )


def _unmatched_observations(observations: pl.DataFrame, matched: pl.DataFrame) -> pl.DataFrame:
    if observations.is_empty():
        return pl.DataFrame(schema=CANDIDATE_EVIDENCE_SCHEMA)
    if matched.is_empty():
        return _unmatched_candidates(observations, "no_matching_evidence")
    matched_keys = matched.select("symbol", "decision_timeframe", "decision_bar_close_ms").unique()
    missing = observations.join(
        matched_keys,
        on=("symbol", "decision_timeframe", "decision_bar_close_ms"),
        how="anti",
    )
    return _unmatched_candidates(missing, "no_matching_evidence")


def _unmatched_candidates(observations: pl.DataFrame, status: str) -> pl.DataFrame:
    if observations.is_empty():
        return pl.DataFrame(schema=CANDIDATE_EVIDENCE_SCHEMA)
    frame = observations.with_columns(
        pl.lit(None, dtype=pl.Int64).alias("outcome_horizon"),
        pl.lit(None, dtype=pl.String).alias("matched_evidence_level"),
        pl.lit(status).alias("candidate_status"),
        pl.lit(None, dtype=pl.String).alias("statistical_direction"),
        pl.lit("insufficient_evidence").alias("research_suggestion"),
    )
    return _select_schema(frame, CANDIDATE_EVIDENCE_SCHEMA)


def rank_candidate_evidence(candidates: pl.DataFrame) -> pl.DataFrame:
    """Add transparent review-ordering components to candidate rows.

    Detects tail_lift presence → uses tail-first scoring.
    Falls back to entropy-first scoring for ladder path.
    """
    if candidates.is_empty():
        return pl.DataFrame(schema=CANDIDATE_RANK_SCHEMA)

    has_tail_lift = (
        "tail_lift" in candidates.columns and candidates["tail_lift"].drop_nulls().len() > 0
    )

    ranked = _select_schema(candidates, CANDIDATE_EVIDENCE_SCHEMA)

    if has_tail_lift:
        ranked = ranked.with_columns(
            pl.col("tail_lift").fill_null(1.0).alias("rank_information_component"),
            pl.lit(0.0).alias("rank_transition_component"),
            pl.max_horizontal(pl.col("tail_lift").fill_null(1.0), pl.lit(1.0)).alias(
                "rank_tail_component"
            ),
            (pl.col("tail_lift_stability").fill_null(0.0)).alias("rank_stability_component"),
            (pl.col("N_tail_exceedances").fill_null(0).cast(pl.Float64) + 1.0)
            .log()
            .alias("rank_path_component"),
        )
    else:
        ranked = ranked.with_columns(
            pl.max_horizontal(pl.col("information_gain_bits").fill_null(0.0), pl.lit(0.0)).alias(
                "rank_information_component"
            ),
            pl.max_horizontal(
                pl.col("transition_information_gain_bits").fill_null(0.0), pl.lit(0.0)
            ).alias("rank_transition_component"),
            pl.max_horizontal(
                pl.col("tail_up_rate").fill_null(0.0), pl.col("tail_down_rate").fill_null(0.0)
            ).alias("rank_tail_component"),
            (
                pl.max_horizontal(pl.col("avg_path_range_pct").fill_null(0.0), pl.lit(0.0)) / 10.0
            ).alias("rank_path_component"),
            pl.min_horizontal(
                pl.max_horizontal(
                    pl.col("information_stability").fill_null(0.0),
                    pl.col("transition_information_stability").fill_null(0.0),
                ),
                pl.lit(2.0),
            ).alias("rank_stability_component"),
        )

    ranked = ranked.with_columns(
        (pl.col("conditioned_observations").fill_null(0).cast(pl.Float64) + 1.0)
        .log()
        .alias("rank_quality_component")
        if not has_tail_lift
        else pl.lit(0.0).alias("rank_quality_component"),
    )

    # Fix: need proper quality component for both paths
    if has_tail_lift:
        ranked = ranked.with_columns(
            ((pl.col("N_tail_exceedances").fill_null(0).cast(pl.Float64) + 1.0).log() / 10.0).alias(
                "rank_quality_component"
            ),
        )
    else:
        ranked = ranked.with_columns(
            (
                (pl.col("conditioned_observations").fill_null(0).cast(pl.Float64) + 1.0).log()
                + (pl.col("symbol_count").fill_null(0).cast(pl.Float64) + 1.0).log()
            ).alias("rank_quality_component"),
        )

    ranked = ranked.with_columns(
        (
            pl.when(pl.col("source_freshness") == "stale").then(pl.lit(1.0)).otherwise(pl.lit(0.0))
            + pl.when(pl.col("candidate_status") != "matched_evidence")
            .then(pl.lit(2.0))
            .otherwise(pl.lit(0.0))
        ).alias("rank_penalty_component"),
    ).with_columns(
        (
            pl.col("rank_information_component")
            + pl.col("rank_transition_component")
            + pl.col("rank_tail_component")
            + pl.col("rank_path_component")
            + pl.col("rank_stability_component")
            + pl.col("rank_quality_component")
            - pl.col("rank_penalty_component")
        ).alias("rank_score"),
        pl.when(pl.col("candidate_status") == "matched_evidence")
        .then(pl.lit("matched_selected_evidence"))
        .otherwise(pl.col("candidate_status"))
        .alias("rank_reason"),
    )
    ranked = _select_schema(ranked.sort("rank_score", descending=True), CANDIDATE_RANK_SCHEMA)
    if has_tail_lift and "tree_direction" in ranked.columns:
        ranked = ranked.unique(
            subset=["symbol", "tree_direction"], keep="first", maintain_order=True
        )
    return ranked


def _select_schema(frame: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if frame.is_empty() and not frame.columns:
        return pl.DataFrame(schema=schema)
    filled = frame.with_columns(
        *[
            pl.lit(None, dtype=dtype).alias(column)
            for column, dtype in schema.items()
            if column not in frame.columns
        ]
    )
    return filled.select(*(pl.col(column).cast(dtype) for column, dtype in schema.items()))


def rank_candidates(candidates: pl.DataFrame) -> pl.DataFrame:
    """Rank promoted candidate rows for review output."""

    return rank_candidate_evidence(candidates)


__all__ = [
    "CANDIDATE_EVIDENCE_SCHEMA",
    "CANDIDATE_RANK_SCHEMA",
    "candidate_evidence_frame",
    "rank_candidate_evidence",
    "rank_candidates",
]
