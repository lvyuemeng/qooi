"""Tailtree evidence product builders."""

from __future__ import annotations

from math import ceil

import polars as pl

from qooi.scanner.tailtree.model import (
    TailTreeModel,
    _tailtree_outcome_by_decision,
)


def leaf_evidence_frame(
    tree: TailTreeModel,
    observations: pl.DataFrame,
    outcomes: pl.DataFrame,
    *,
    recent_window_days: int = 30,
) -> pl.DataFrame:
    """Per-leaf tail evidence: N_total, N_tail, xi, sigma, tail_lift, stability."""
    with_leaf = tree.predict_leaf(observations)

    global_tr = tree.metadata.global_baseline.tail_rate
    tail_col = "tail_up" if tree.metadata.direction == "up" else "tail_down"
    has_tail = tail_col in outcomes.columns

    if not has_tail:
        return pl.DataFrame(
            schema={
                "leaf_id": pl.Int32,
                "tree_direction": pl.String,
                "N_total": pl.UInt32,
                "N_tail_exceedances": pl.UInt32,
                "gpd_shape_xi": pl.Float64,
                "gpd_scale_sigma": pl.Float64,
                "tail_lift": pl.Float64,
                "tail_lift_stability": pl.Float64,
                "leaf_tail_rate": pl.Float64,
                "global_tail_rate": pl.Float64,
                "tail_utility_mean": pl.Float64,
                "tail_utility_p90": pl.Float64,
            }
        )

    outcome_by_decision = _tailtree_outcome_by_decision(outcomes)
    utility_col = f"tail_utility_{tree.metadata.direction}"
    outcome_columns = ["symbol", "decision_bar_close_ms", tail_col]
    if utility_col in outcome_by_decision.columns:
        outcome_columns.append(utility_col)
    joined = with_leaf.join(
        outcome_by_decision.select(outcome_columns),
        on=["symbol", "decision_bar_close_ms"],
        how="left",
    ).with_columns(
        pl.col(tail_col).fill_null(False).cast(pl.Boolean).alias(tail_col),
        (
            pl.col(utility_col).fill_null(0.0).cast(pl.Float64)
            if utility_col in outcome_by_decision.columns
            else pl.lit(0.0)
        ).alias(utility_col),
    )

    leaf_stats = joined.group_by("leaf_id").agg(
        pl.len().cast(pl.UInt32).alias("N_total"),
        pl.col(tail_col).cast(pl.UInt32).sum().alias("N_tail_exceedances"),
        pl.col(utility_col)
        .filter(pl.col(tail_col))
        .mean()
        .fill_null(0.0)
        .alias("tail_utility_mean"),
        pl.col(utility_col)
        .filter(pl.col(tail_col))
        .quantile(0.9)
        .fill_null(0.0)
        .alias("tail_utility_p90"),
    )

    # Recent window
    max_ts = observations.get_column("decision_bar_close_ms").max()
    recent_cutoff = max_ts - recent_window_days * 24 * 60 * 60 * 1000
    recent = joined.filter(pl.col("decision_bar_close_ms") >= recent_cutoff)
    if not recent.is_empty():
        recent_stats = recent.group_by("leaf_id").agg(
            pl.len().cast(pl.UInt32).alias("N_recent"),
            pl.col(tail_col).cast(pl.UInt32).sum().alias("N_tail_recent"),
        )
        leaf_stats = leaf_stats.join(recent_stats, on="leaf_id", how="left")
    else:
        leaf_stats = leaf_stats.with_columns(
            pl.lit(0, dtype=pl.UInt32).alias("N_recent"),
            pl.lit(0, dtype=pl.UInt32).alias("N_tail_recent"),
        )

    leaf_params_df = pl.DataFrame(
        [
            {"leaf_id": lid, "gpd_shape_xi": p.xi, "gpd_scale_sigma": p.sigma}
            for lid, p in tree.metadata.leaf_params.items()
        ],
        schema={
            "leaf_id": pl.Int32,
            "gpd_shape_xi": pl.Float64,
            "gpd_scale_sigma": pl.Float64,
        },
    )

    result = (
        leaf_stats.join(leaf_params_df, on="leaf_id", how="left")
        .with_columns(
            pl.lit(tree.metadata.direction).alias("tree_direction"),
            (
                pl.col("N_tail_exceedances").cast(pl.Float64)
                / pl.when(pl.col("N_total") > 0).then(pl.col("N_total")).otherwise(None)
            )
            .fill_null(0.0)
            .alias("leaf_tail_rate"),
            pl.lit(global_tr).alias("global_tail_rate"),
        )
        .with_columns(
            (
                pl.col("leaf_tail_rate")
                / pl.when(pl.col("global_tail_rate") > 0)
                .then(pl.col("global_tail_rate"))
                .otherwise(None)
            )
            .fill_null(0.0)
            .alias("tail_lift"),
        )
        .with_columns(
            (
                (
                    pl.col("N_tail_recent").cast(pl.Float64)
                    / pl.when(pl.col("N_recent") > 0).then(pl.col("N_recent")).otherwise(None)
                )
                / pl.when(pl.col("leaf_tail_rate") > 0)
                .then(pl.col("leaf_tail_rate"))
                .otherwise(None)
            )
            .clip(0, 2)
            .fill_null(0.0)
            .alias("tail_lift_stability"),
        )
    )

    return result.select(
        "leaf_id",
        "tree_direction",
        "N_total",
        "N_tail_exceedances",
        "gpd_shape_xi",
        "gpd_scale_sigma",
        "tail_lift",
        "tail_lift_stability",
        "leaf_tail_rate",
        "global_tail_rate",
        "tail_utility_mean",
        "tail_utility_p90",
    )


def score_bucket_evidence_frame(
    tree: TailTreeModel,
    observations: pl.DataFrame,
    outcomes: pl.DataFrame,
    *,
    score_quantiles: tuple[float, ...] = (0.99, 0.98, 0.95, 0.90),
) -> pl.DataFrame:
    """Evidence buckets for boosted utility models using full ensemble scores."""
    schema = {
        "score_bucket": pl.String,
        "score_quantile": pl.Float64,
        "score_min": pl.Float64,
        "score_max": pl.Float64,
        "tree_direction": pl.String,
        "N_total": pl.UInt32,
        "N_tail_exceedances": pl.UInt32,
        "gpd_shape_xi": pl.Float64,
        "gpd_scale_sigma": pl.Float64,
        "tail_lift": pl.Float64,
        "tail_lift_stability": pl.Float64,
        "leaf_tail_rate": pl.Float64,
        "global_tail_rate": pl.Float64,
        "tail_utility_mean": pl.Float64,
        "tail_utility_p90": pl.Float64,
        "information_gain_bits": pl.Float64,
        "statistical_direction": pl.String,
        "research_suggestion": pl.String,
        "selected_evidence_level": pl.Boolean,
    }
    if observations.is_empty():
        return pl.DataFrame(schema=schema)

    direction = tree.metadata.direction
    tail_col = "tail_up" if direction == "up" else "tail_down"
    utility_col = f"tail_utility_{direction}"
    if outcomes.is_empty() or tail_col not in outcomes.columns:
        return pl.DataFrame(schema=schema)

    scored = tree.predict_score(observations)
    outcome_aggs = [pl.col(tail_col).fill_null(False).cast(pl.Boolean).max().alias(tail_col)]
    if utility_col in outcomes.columns:
        outcome_aggs.append(
            pl.col(utility_col).fill_null(0.0).cast(pl.Float64).max().alias(utility_col)
        )
    outcome_by_decision = outcomes.group_by("symbol", "decision_bar_close_ms").agg(*outcome_aggs)
    joined = scored.join(
        outcome_by_decision,
        on=["symbol", "decision_bar_close_ms"],
        how="left",
    ).with_columns(
        pl.col(tail_col).fill_null(False).cast(pl.Boolean).alias(tail_col),
        (
            pl.col(utility_col).fill_null(0.0).cast(pl.Float64)
            if utility_col in outcome_by_decision.columns
            else pl.lit(0.0)
        ).alias(utility_col),
    )

    global_tail_rate = tree.metadata.global_baseline.tail_rate
    score_max = float(joined.get_column("tailtree_score").max() or 0.0)
    rows = []
    sorted_scores = joined.sort("tailtree_score", descending=True)
    for quantile in score_quantiles:
        bucket_size = max(1, ceil(len(sorted_scores) * (1.0 - quantile)))
        bucket = sorted_scores.head(bucket_size)
        if bucket.is_empty():
            continue
        threshold = float(bucket.get_column("tailtree_score").min() or 0.0)
        tail_count = int(bucket.get_column(tail_col).cast(pl.UInt32).sum())
        total = len(bucket)
        tail_rate = tail_count / total if total else 0.0
        utilities = bucket.filter(pl.col(tail_col)).get_column(utility_col)
        utility_mean = float(utilities.mean() or 0.0) if not utilities.is_empty() else 0.0
        utility_p90 = float(utilities.quantile(0.9) or 0.0) if not utilities.is_empty() else 0.0
        rows.append(
            {
                "score_bucket": f"top_{int(round((1.0 - quantile) * 100))}pct",
                "score_quantile": float(quantile),
                "score_min": threshold,
                "score_max": score_max,
                "tree_direction": direction,
                "N_total": total,
                "N_tail_exceedances": tail_count,
                "gpd_shape_xi": tree.metadata.global_baseline.xi,
                "gpd_scale_sigma": tree.metadata.global_baseline.sigma,
                "tail_lift": tail_rate / global_tail_rate if global_tail_rate > 0 else 0.0,
                "tail_lift_stability": 1.0,
                "leaf_tail_rate": tail_rate,
                "global_tail_rate": global_tail_rate,
                "tail_utility_mean": utility_mean,
                "tail_utility_p90": utility_p90,
                "information_gain_bits": 0.0,
                "statistical_direction": direction,
                "research_suggestion": "score_bucket_tail_utility",
                "selected_evidence_level": True,
            }
        )
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


def leaf_context_frame(
    tree: TailTreeModel,
    observations: pl.DataFrame,
    outcomes: pl.DataFrame,
    *,
    global_baseline: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Per-leaf context: directional probabilities, path diagnostics."""
    with_leaf = tree.predict_leaf(observations)

    if outcomes.is_empty():
        return pl.DataFrame(
            schema={
                "leaf_id": pl.Int32,
                "p_up": pl.Float64,
                "p_down": pl.Float64,
                "p_flat": pl.Float64,
                "conditioned_entropy_bits": pl.Float64,
                "information_gain_bits": pl.Float64,
                "tail_up_rate": pl.Float64,
                "tail_down_rate": pl.Float64,
                "path_skew": pl.Float64,
                "returned_to_origin_rate": pl.Float64,
                "statistical_direction": pl.String,
                "research_suggestion": pl.String,
            }
        )

    outcome_by_decision = _tailtree_outcome_by_decision(outcomes)
    joined = with_leaf.join(
        outcome_by_decision.select(
            [
                "symbol",
                "decision_bar_close_ms",
                "outcome_bucket",
                "tail_up",
                "tail_down",
                "direction_changed",
                "returned_to_origin",
            ]
        ),
        on=["symbol", "decision_bar_close_ms"],
        how="left",
    )

    leaf_agg = joined.group_by("leaf_id").agg(
        pl.len().cast(pl.UInt32).alias("N_leaf"),
        (pl.col("outcome_bucket") == "up").mean().alias("p_up"),
        (pl.col("outcome_bucket") == "down").mean().alias("p_down"),
        (pl.col("outcome_bucket") == "flat").mean().alias("p_flat"),
    )

    has_tails = "tail_up" in joined.columns and "tail_down" in joined.columns
    if has_tails:
        tail_agg = joined.group_by("leaf_id").agg(
            pl.col("tail_up").cast(pl.Float64).mean().alias("tail_up_rate"),
            pl.col("tail_down").cast(pl.Float64).mean().alias("tail_down_rate"),
        )
        leaf_agg = leaf_agg.join(tail_agg, on="leaf_id", how="left")

    path_agg = (
        joined.group_by("leaf_id").agg(
            (
                pl.col("tail_up").cast(pl.Float64).mean().fill_null(0.0)
                - pl.col("tail_down").cast(pl.Float64).mean().fill_null(0.0)
            ).alias("path_skew"),
            pl.col("returned_to_origin").cast(pl.Float64).mean().alias("returned_to_origin_rate"),
        )
        if "returned_to_origin" in joined.columns and has_tails
        else joined.group_by("leaf_id").agg(
            pl.lit(0.0).alias("path_skew"),
            pl.lit(0.0).alias("returned_to_origin_rate"),
        )
    )
    leaf_agg = leaf_agg.join(path_agg, on="leaf_id", how="left")

    # Entropy
    from qooi.scanner import entropy_expr

    leaf_agg = leaf_agg.with_columns(
        entropy_expr("p_up", "p_down", "p_flat").alias("conditioned_entropy_bits"),
        pl.lit(0.0).alias("information_gain_bits"),
    )

    # Statistical direction
    leaf_agg = leaf_agg.with_columns(
        pl.when(pl.col("p_up") > pl.max_horizontal("p_down", "p_flat"))
        .then(pl.lit("up"))
        .when(pl.col("p_down") > pl.max_horizontal("p_up", "p_flat"))
        .then(pl.lit("down"))
        .otherwise(pl.lit("flat"))
        .alias("statistical_direction"),
    )

    # Research suggestion
    leaf_agg = leaf_agg.with_columns(
        pl.when((pl.col("returned_to_origin_rate") >= 0.25) & (pl.col("path_skew").abs() <= 0.10))
        .then(pl.lit("chop_avoid"))
        .otherwise(pl.lit("insufficient_evidence"))
        .alias("research_suggestion"),
    )

    return leaf_agg.select(
        "leaf_id",
        "p_up",
        "p_down",
        "p_flat",
        "conditioned_entropy_bits",
        "information_gain_bits",
        "tail_up_rate",
        "tail_down_rate",
        "path_skew",
        "returned_to_origin_rate",
        "statistical_direction",
        "research_suggestion",
    )


def select_tail_leaves(
    leaf_evidence: pl.DataFrame,
    *,
    min_tail_exceedances: int = 30,
    min_tail_lift: float = 1.5,
    min_tail_lift_stability: float = 0.3,
    fallback_top_n: int = 10,
) -> pl.DataFrame:
    """Select tail leaves by hard gate, or top-ranked best available leaves.

    The fallback is deliberately labelled; it does not pretend weak leaves passed.
    It gives review/candidate ranking a quantitative surface when the strict gate
    selects zero leaves.
    """
    if leaf_evidence.is_empty():
        return leaf_evidence

    scored = leaf_evidence.with_columns(
        (pl.col("N_tail_exceedances") >= min_tail_exceedances).alias("passes_tail_count_gate"),
        (pl.col("tail_lift") >= min_tail_lift).alias("passes_tail_lift_gate"),
        (
            (pl.col("tail_lift_stability") >= min_tail_lift_stability) | (pl.col("N_total") < 200)
        ).alias("passes_stability_gate"),
    ).with_columns(
        (
            pl.col("passes_tail_count_gate")
            & pl.col("passes_tail_lift_gate")
            & pl.col("passes_stability_gate")
        ).alias("selected_evidence_level"),
        (
            pl.max_horizontal(pl.col("tail_lift").fill_null(0.0), pl.lit(0.0))
            * (pl.col("N_tail_exceedances").fill_null(0).cast(pl.Float64) + 1.0).log()
            * pl.max_horizontal(pl.col("tail_lift_stability").fill_null(0.0), pl.lit(0.05))
        ).alias("tail_evidence_score"),
    )

    hard = scored.filter(pl.col("selected_evidence_level"))
    if not hard.is_empty():
        return hard.with_columns(pl.lit("hard_gate").alias("selection_mode"))

    return (
        scored.sort(
            ["tail_evidence_score", "tail_lift", "N_tail_exceedances"],
            descending=[True, True, True],
        )
        .head(fallback_top_n)
        .with_columns(pl.lit("best_available").alias("selection_mode"))
    )
