"""Promoted scanner candidate ranking boundary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Protocol, TypeAlias

import polars as pl

PolarsDtype: TypeAlias = type[pl.DataType] | pl.DataType


class TailTreePredictor(Protocol):
    def predict_leaf(self, features: pl.DataFrame) -> pl.DataFrame: ...


TailTreeDirection: TypeAlias = Literal["up", "down"]
TailTreeModelKey: TypeAlias = tuple[int, TailTreeDirection]


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

CANDIDATE_EVIDENCE_SCHEMA: dict[str, PolarsDtype] = {
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
    "required_missing_source_count": pl.Int64,
    "required_stale_source_count": pl.Int64,
    "provider_bounded_source_count": pl.Int64,
    "optional_absent_source_count": pl.Int64,
    "source_penalty_score": pl.Float64,
    # Tree path columns (nullable — ladder path produces null)
    "tail_lift": pl.Float64,
    "gpd_shape_xi": pl.Float64,
    "gpd_scale_sigma": pl.Float64,
    "tail_lift_stability": pl.Float64,
    "N_total": pl.UInt32,
    "N_tail_exceedances": pl.UInt32,
    "tail_utility_mean": pl.Float64,
    "tail_utility_p90": pl.Float64,
    "leaf_id": pl.Int32,
    "tree_direction": pl.String,
    "score_bucket": pl.String,
    "tailtree_score": pl.Float64,
    "score_min": pl.Float64,
    "score_max": pl.Float64,
    "leaf_path": pl.String,
}


CANDIDATE_RANK_SCHEMA: dict[str, PolarsDtype] = CANDIDATE_EVIDENCE_SCHEMA | {
    "rank_information_component": pl.Float64,
    "rank_transition_component": pl.Float64,
    "rank_tail_component": pl.Float64,
    "rank_path_component": pl.Float64,
    "rank_stability_component": pl.Float64,
    "rank_quality_component": pl.Float64,
    "rank_penalty_component": pl.Float64,
    "rank_score": pl.Float64,
    "profit_proxy_score": pl.Float64,
    "profit_proxy_per_selected_obs": pl.Float64,
    "profit_proxy_per_1k_observed": pl.Float64,
    "promotion_score": pl.Float64,
    "rank_reason": pl.String,
}


CANDIDATE_HORIZON_CONSISTENCY_SCHEMA: dict[str, PolarsDtype] = {
    "symbol": pl.String,
    "decision_timeframe": pl.String,
    "tree_direction": pl.String,
    "horizon_count": pl.UInt32,
    "strong_horizon_count": pl.UInt32,
    "horizon_span_bars": pl.Int64,
    "best_outcome_horizon": pl.Int64,
    "best_rank_score": pl.Float64,
    "best_tail_lift": pl.Float64,
    "best_tail_utility_score": pl.Float64,
    "direction_consistency_score": pl.Float64,
    "opposite_direction_count": pl.UInt32,
    "opposite_direction_best_rank_score": pl.Float64,
    "conflict_penalty_score": pl.Float64,
    "consistency_rank_score": pl.Float64,
}


def candidate_horizon_consistency_frame(candidate_rank: pl.DataFrame) -> pl.DataFrame:
    """Summarize same-direction horizon agreement without averaging raw scores."""
    if candidate_rank.is_empty() or "tree_direction" not in candidate_rank.columns:
        return pl.DataFrame(schema=CANDIDATE_HORIZON_CONSISTENCY_SCHEMA)
    required = {"symbol", "decision_timeframe", "outcome_horizon", "tree_direction"}
    if not required.issubset(candidate_rank.columns):
        return pl.DataFrame(schema=CANDIDATE_HORIZON_CONSISTENCY_SCHEMA)
    base = candidate_rank.filter(pl.col("candidate_status") == "matched_evidence")
    if base.is_empty():
        return pl.DataFrame(schema=CANDIDATE_HORIZON_CONSISTENCY_SCHEMA)
    tail_lift_expr = (
        pl.col("tail_lift").fill_null(0.0).fill_nan(0.0)
        if "tail_lift" in base.columns
        else pl.lit(0.0)
    )
    tail_count_expr = (
        pl.col("N_tail_exceedances").fill_null(0).cast(pl.Float64)
        if "N_tail_exceedances" in base.columns
        else pl.lit(0.0)
    )
    tail_utility_expr = tail_lift_expr * (tail_count_expr + 1.0).log()
    rank_score_expr = (
        pl.col("rank_score").fill_null(0.0).fill_nan(0.0)
        if "rank_score" in base.columns
        else tail_utility_expr
    )
    enriched = base.with_columns(
        tail_lift_expr.alias("_tail_lift_value"),
        tail_utility_expr.alias("_tail_utility_score"),
        rank_score_expr.alias("_rank_score_value"),
        ((tail_lift_expr >= 1.5) & (rank_score_expr > 0.0))
        .cast(pl.UInt32)
        .alias("_strong_horizon_flag"),
    )
    per_horizon = enriched.sort(
        [
            "symbol",
            "decision_timeframe",
            "tree_direction",
            "outcome_horizon",
            "_rank_score_value",
        ],
        descending=[False, False, False, False, True],
    ).unique(
        subset=["symbol", "decision_timeframe", "tree_direction", "outcome_horizon"],
        keep="first",
        maintain_order=True,
    )
    grouped = per_horizon.group_by(
        "symbol", "decision_timeframe", "tree_direction", maintain_order=True
    ).agg(
        pl.col("outcome_horizon").n_unique().cast(pl.UInt32).alias("horizon_count"),
        pl.col("_strong_horizon_flag").sum().cast(pl.UInt32).alias("strong_horizon_count"),
        (pl.col("outcome_horizon").max() - pl.col("outcome_horizon").min()).alias(
            "horizon_span_bars"
        ),
        pl.col("outcome_horizon").sort_by("_rank_score_value", descending=True)
        .first()
        .alias("best_outcome_horizon"),
        pl.col("_rank_score_value").max().alias("best_rank_score"),
        pl.col("_tail_lift_value").max().alias("best_tail_lift"),
        pl.col("_tail_utility_score").max().alias("best_tail_utility_score"),
    )
    opposite = grouped.rename(
        {
            "tree_direction": "_opposite_tree_direction",
            "horizon_count": "opposite_direction_count",
            "best_rank_score": "opposite_direction_best_rank_score",
        }
    ).select(
        "symbol",
        "decision_timeframe",
        "_opposite_tree_direction",
        "opposite_direction_count",
        "opposite_direction_best_rank_score",
    )
    panel = grouped.with_columns(
        pl.when(pl.col("tree_direction") == "up")
        .then(pl.lit("down"))
        .otherwise(pl.lit("up"))
        .alias("_opposite_tree_direction"),
        (
            pl.col("strong_horizon_count").cast(pl.Float64)
            * (1.0 + pl.col("best_tail_lift").fill_null(0.0).fill_nan(0.0)).log()
            * (1.0 + pl.col("best_rank_score").fill_null(0.0).fill_nan(0.0)).log()
        ).alias("direction_consistency_score"),
    ).join(
        opposite,
        on=["symbol", "decision_timeframe", "_opposite_tree_direction"],
        how="left",
    )
    panel = panel.with_columns(
        pl.col("opposite_direction_count").fill_null(0).cast(pl.UInt32),
        pl.col("opposite_direction_best_rank_score").fill_null(0.0),
    ).with_columns(
        pl.when(pl.col("opposite_direction_best_rank_score") > pl.col("best_rank_score"))
        .then(pl.col("opposite_direction_best_rank_score") - pl.col("best_rank_score"))
        .otherwise(0.0)
        .alias("conflict_penalty_score")
    ).with_columns(
        (pl.col("direction_consistency_score") - pl.col("conflict_penalty_score")).alias(
            "consistency_rank_score"
        )
    )
    return _select_schema(
        panel.drop("_opposite_tree_direction").sort(
            "consistency_rank_score", descending=True
        ),
        CANDIDATE_HORIZON_CONSISTENCY_SCHEMA,
    )


def candidate_evidence_frame(
    observations: pl.DataFrame,
    evidence: pl.DataFrame,
    *,
    latest_only: bool = True,
    tree_models: Mapping[TailTreeModelKey, TailTreePredictor] | None = None,
) -> pl.DataFrame:
    """Match observation rows to selected evidence rows."""
    if observations.is_empty():
        return pl.DataFrame(schema=CANDIDATE_EVIDENCE_SCHEMA)
    if tree_models:
        return _candidate_from_trees(observations, evidence, tree_models, latest_only=latest_only)
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
    tree_models: Mapping[TailTreeModelKey, TailTreePredictor],
    *,
    latest_only: bool = True,
) -> pl.DataFrame:
    base = _candidate_observations(observations, latest_only=latest_only)
    if base.is_empty():
        return pl.DataFrame(schema=CANDIDATE_EVIDENCE_SCHEMA)

    frames = []
    for (outcome_horizon, direction), tree in sorted(tree_models.items()):
        dir_evidence = (
            evidence.filter(
                (pl.col("tree_direction") == direction)
                & (pl.col("outcome_horizon") == int(outcome_horizon))
            )
            if "tree_direction" in evidence.columns
            else evidence
        )
        if dir_evidence.is_empty():
            continue
        predict_score = getattr(tree, "predict_score", None)
        if "score_bucket" in dir_evidence.columns and callable(predict_score):
            try:
                with_score = predict_score(base).with_columns(
                    pl.lit(int(outcome_horizon)).alias("outcome_horizon")
                )
            except Exception:
                continue
            matched = with_score.join(dir_evidence, how="cross").filter(
                (pl.col("tailtree_score") >= pl.col("score_min"))
                & (pl.col("tailtree_score") <= pl.col("score_max"))
            )
            if matched.is_empty():
                continue
            matched = matched.with_columns(
                pl.lit(f"tree_{direction}").alias("matched_evidence_level"),
                pl.lit("matched_evidence").alias("candidate_status"),
            )
            frames.append(matched)
            continue
        try:
            with_leaf = tree.predict_leaf(base).with_columns(
                pl.lit(int(outcome_horizon)).alias("outcome_horizon")
            )
        except Exception:
            continue
        matched = with_leaf.join(
            dir_evidence,
            on=["outcome_horizon", "leaf_id"],
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
    if "outcome_horizon" not in candidates.columns:
        raise ValueError("candidate evidence pipe missing outcome_horizon")

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

    ranked = (
        ranked.with_columns(
            (
                pl.col("required_missing_source_count").fill_null(0).cast(pl.Float64) * 2.0
                + pl.col("required_stale_source_count").fill_null(0).cast(pl.Float64) * 0.3
                + pl.col("provider_bounded_source_count").fill_null(0).cast(pl.Float64) * 0.0
            ).alias("source_penalty_score")
        )
        .with_columns(
            (
                pl.col("source_penalty_score")
                + pl.when(pl.col("source_freshness") == "stale")
                .then(pl.lit(1.0))
                .otherwise(pl.lit(0.0))
                + pl.when(pl.col("candidate_status") != "matched_evidence")
                .then(pl.lit(2.0))
                .otherwise(pl.lit(0.0))
            ).alias("rank_penalty_component"),
        )
        .with_columns(
            (
                pl.col("tail_utility_mean").fill_null(0.0).fill_nan(0.0)
                if has_tail_lift
                else pl.lit(0.0)
            ).alias("profit_proxy_score"),
            (
                pl.col("tail_utility_mean").fill_null(0.0).fill_nan(0.0)
                if has_tail_lift
                else pl.lit(0.0)
            ).alias("profit_proxy_per_selected_obs"),
            (
                (pl.col("tail_utility_mean").fill_null(0.0).fill_nan(0.0) * 1000.0)
                / pl.max_horizontal(
                    pl.col("conditioned_observations").fill_null(0).cast(pl.Float64),
                    pl.lit(1.0),
                )
                if has_tail_lift
                else pl.lit(0.0)
            ).alias("profit_proxy_per_1k_observed"),
        )
        .with_columns(
            (
                pl.col("rank_information_component")
                + pl.col("rank_transition_component")
                + pl.col("rank_tail_component")
                + pl.col("rank_path_component")
                + pl.col("rank_stability_component")
                + pl.col("rank_quality_component")
                - pl.col("rank_penalty_component")
            ).alias("rank_score"),
            (
                pl.col("profit_proxy_score")
                + pl.col("rank_tail_component")
                + pl.col("rank_path_component")
                + pl.col("rank_stability_component")
            ).alias("promotion_score"),
            pl.when(pl.col("candidate_status") == "matched_evidence")
            .then(pl.lit("matched_selected_evidence"))
            .otherwise(pl.col("candidate_status"))
            .alias("rank_reason"),
        )
    )
    ranked = _select_schema(ranked.sort("rank_score", descending=True), CANDIDATE_RANK_SCHEMA)
    if has_tail_lift and "tree_direction" in ranked.columns:
        ranked = ranked.unique(
            subset=["symbol", "tree_direction"], keep="first", maintain_order=True
        )
    return ranked


def _select_schema(frame: pl.DataFrame, schema: dict[str, PolarsDtype]) -> pl.DataFrame:
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


__all__ = [
    "CANDIDATE_EVIDENCE_SCHEMA",
    "CANDIDATE_RANK_SCHEMA",
    "candidate_evidence_frame",
    "rank_candidate_evidence",
]
