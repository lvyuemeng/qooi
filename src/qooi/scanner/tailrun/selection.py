"""Tailtree scored-candidate replay and selection metric products."""

from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING

import polars as pl

from qooi.scanner.tailrun.types import (
    TailtreeArtifactTree,
    TailtreeDirection,
    TailtreePreparedFrames,
    TailtreeSelectionEfficiencyRow,
)

if TYPE_CHECKING:
    from qooi.scanner.tailrun.planning import TailtreeProfileRun


def score_bucket_candidate_frame(
    tree: TailtreeArtifactTree,
    observations: pl.DataFrame,
    outcomes: pl.DataFrame,
    outcome_horizon: int,
    *,
    score_quantiles: tuple[float, ...] = (0.99, 0.98, 0.95, 0.90),
) -> pl.DataFrame:
    direction = tree.metadata.direction
    tail_col = f"tail_{direction}"
    utility_col = f"tail_utility_{direction}"
    margin_col = f"tail_utility_margin_{direction}"
    if observations.is_empty() or outcomes.is_empty() or tail_col not in outcomes.columns:
        return pl.DataFrame()

    outcome_aggs = [pl.col(tail_col).fill_null(False).cast(pl.Boolean).max().alias(tail_col)]
    outcome_aggs.append(
        pl.col(utility_col).fill_null(0.0).cast(pl.Float64).max().alias(utility_col)
        if utility_col in outcomes.columns
        else pl.lit(0.0).alias(utility_col)
    )
    state_col = "path_state" if "path_state" in outcomes.columns else "tail_state"
    outcome_aggs.append(
        pl.col(state_col).drop_nulls().first().alias("tail_state")
        if state_col in outcomes.columns
        else pl.lit(None).alias("tail_state")
    )
    outcome_aggs.append(
        pl.col("tail_both").fill_null(False).cast(pl.Boolean).max().alias("tail_both")
        if "tail_both" in outcomes.columns
        else pl.lit(False).alias("tail_both")
    )
    outcome_aggs.append(
        pl.col(margin_col).fill_null(0.0).cast(pl.Float64).max().alias(margin_col)
        if margin_col in outcomes.columns
        else pl.lit(0.0).alias(margin_col)
    )
    outcome_by_decision = outcomes.group_by("symbol", "decision_bar_close_ms").agg(*outcome_aggs)
    joined = (
        tree.predict_score(observations)
        .join(
            outcome_by_decision,
            on=["symbol", "decision_bar_close_ms"],
            how="left",
        )
        .with_columns(
            pl.col(tail_col).fill_null(False).cast(pl.Boolean).alias("selected_tail"),
            pl.col(utility_col).fill_null(0.0).cast(pl.Float64).alias("selected_utility"),
            (
                pl.col("tail_state").is_in([direction, f"clean_{direction}"])
            ).alias("selected_side_only"),
            pl.col("tail_both").fill_null(False).cast(pl.Boolean).alias("selected_tail_both"),
            pl.col("tail_state").alias("selected_tail_state"),
            pl.col(margin_col).fill_null(0.0).cast(pl.Float64).alias("selected_utility_margin"),
            pl.lit(int(outcome_horizon)).alias("outcome_horizon"),
            pl.lit(direction).alias("direction"),
        )
    )
    if joined.is_empty():
        return joined.head(0)

    sorted_scores = joined.sort("tailtree_score", descending=True)
    frames: list[pl.DataFrame] = []
    for quantile in score_quantiles:
        bucket_value = round((1.0 - quantile) * 100)
        bucket = f"top_{int(bucket_value)}pct"
        frames.append(
            sorted_scores.head(max(1, ceil(len(sorted_scores) * (1.0 - quantile)))).select(
                "symbol",
                "decision_bar_close_ms",
                "outcome_horizon",
                "direction",
                "tailtree_score",
                "selected_tail",
                "selected_utility",
                "selected_side_only",
                "selected_tail_both",
                "selected_tail_state",
                "selected_utility_margin",
                pl.lit(bucket).alias("score_bucket"),
                pl.lit(float(bucket_value)).alias("budget_value"),
            )
        )
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def paired_candidate_replay_frame(scored_candidates: pl.DataFrame) -> pl.DataFrame:
    if scored_candidates.is_empty():
        return scored_candidates

    keys = ["symbol", "decision_bar_close_ms", "outcome_horizon", "score_bucket"]
    up = scored_candidates.filter(pl.col("direction") == "up").rename(
        {
            "tailtree_score": "score_up",
            "selected_tail": "tail_up",
            "selected_utility": "utility_up",
            "selected_side_only": "side_only_up",
            "selected_tail_both": "tail_both_up",
            "selected_tail_state": "tail_state_up",
            "selected_utility_margin": "utility_margin_up",
        }
    )
    down = scored_candidates.filter(pl.col("direction") == "down").rename(
        {
            "tailtree_score": "score_down",
            "selected_tail": "tail_down",
            "selected_utility": "utility_down",
            "selected_side_only": "side_only_down",
            "selected_tail_both": "tail_both_down",
            "selected_tail_state": "tail_state_down",
            "selected_utility_margin": "utility_margin_down",
        }
    )
    selected_up = up.join(
        down.select(
            *keys,
            "score_down",
            "tail_down",
            "utility_down",
            "side_only_down",
            "tail_both_down",
            "tail_state_down",
            "utility_margin_down",
        ),
        on=keys,
        how="left",
    ).with_columns(pl.lit("up").alias("selected_direction"))
    selected_down = down.join(
        up.select(
            *keys,
            "score_up",
            "tail_up",
            "utility_up",
            "side_only_up",
            "tail_both_up",
            "tail_state_up",
            "utility_margin_up",
        ),
        on=keys,
        how="left",
    ).with_columns(pl.lit("down").alias("selected_direction"))

    replay = pl.concat(
        [
            selected_up.select(
                *keys,
                "selected_direction",
                pl.col("score_up").alias("selected_score"),
                pl.col("score_down").alias("opposite_score"),
                pl.col("tail_up").alias("selected_tail"),
                pl.col("tail_down").fill_null(False).alias("opposite_tail"),
                pl.col("utility_up").alias("selected_utility"),
                pl.col("utility_down").fill_null(0.0).alias("opposite_utility"),
                pl.col("side_only_up").alias("selected_side_only"),
                pl.col("side_only_down").fill_null(False).alias("opposite_side_only"),
                pl.col("tail_both_up").alias("selected_tail_both"),
                pl.col("tail_both_down").fill_null(False).alias("opposite_tail_both"),
                pl.col("tail_state_up").alias("selected_tail_state"),
                pl.col("tail_state_down").fill_null("none").alias("opposite_tail_state"),
                pl.col("utility_margin_up").alias("selected_utility_margin"),
                pl.col("utility_margin_down").fill_null(0.0).alias("opposite_utility_margin"),
            ),
            selected_down.select(
                *keys,
                "selected_direction",
                pl.col("score_down").alias("selected_score"),
                pl.col("score_up").alias("opposite_score"),
                pl.col("tail_down").alias("selected_tail"),
                pl.col("tail_up").fill_null(False).alias("opposite_tail"),
                pl.col("utility_down").alias("selected_utility"),
                pl.col("utility_up").fill_null(0.0).alias("opposite_utility"),
                pl.col("side_only_down").alias("selected_side_only"),
                pl.col("side_only_up").fill_null(False).alias("opposite_side_only"),
                pl.col("tail_both_down").alias("selected_tail_both"),
                pl.col("tail_both_up").fill_null(False).alias("opposite_tail_both"),
                pl.col("tail_state_down").alias("selected_tail_state"),
                pl.col("tail_state_up").fill_null("none").alias("opposite_tail_state"),
                pl.col("utility_margin_down").alias("selected_utility_margin"),
                pl.col("utility_margin_up").fill_null(0.0).alias("opposite_utility_margin"),
            ),
        ],
        how="diagonal_relaxed",
    )
    return replay.with_columns(
        (pl.col("selected_score") - pl.col("opposite_score").fill_null(0.0)).alias(
            "directional_score_margin"
        ),
        (pl.col("selected_tail") & pl.col("opposite_tail")).cast(pl.Int64).alias("gray_zone_int"),
        ((~pl.col("selected_tail")) & pl.col("opposite_tail"))
        .cast(pl.Int64)
        .alias("false_direction_int"),
        (
            pl.col("selected_side_only")
            & ~pl.col("opposite_side_only")
            & ~pl.col("selected_tail_both")
        )
        .cast(pl.Int64)
        .alias("side_only_int"),
        (
            pl.col("selected_tail_both")
            | pl.col("opposite_tail_both")
            | (pl.col("selected_tail") & pl.col("opposite_tail"))
        )
        .cast(pl.Int64)
        .alias("tail_both_int"),
    )


def calibrated_candidate_replay_frame(replay: pl.DataFrame) -> pl.DataFrame:
    if replay.is_empty():
        return replay
    keys = ["outcome_horizon", "score_bucket", "selected_direction"]
    calibration = replay.group_by(keys).agg(
        pl.col("selected_tail").fill_null(False).mean().alias("selected_bucket_tail_rate"),
        pl.col("opposite_tail").fill_null(False).mean().alias("opposite_bucket_tail_rate"),
        pl.col("selected_side_only")
        .fill_null(False)
        .mean()
        .alias("selected_bucket_side_only_rate"),
        pl.col("opposite_side_only")
        .fill_null(False)
        .mean()
        .alias("opposite_bucket_side_only_rate"),
        pl.col("selected_tail_both")
        .fill_null(False)
        .mean()
        .alias("selected_bucket_tail_both_rate"),
    )
    return replay.join(calibration, on=keys, how="left").with_columns(
        (pl.col("selected_bucket_tail_rate") - pl.col("opposite_bucket_tail_rate")).alias(
            "calibrated_directional_margin"
        ),
        (
            pl.col("selected_bucket_side_only_rate")
            - pl.col("opposite_bucket_side_only_rate")
        ).alias("calibrated_side_margin"),
    )


def tailtree_action_surface_frame(replay: pl.DataFrame) -> pl.DataFrame:
    if replay.is_empty():
        return replay
    work = replay.with_columns(
        pl.col("selected_direction").alias("action_side"),
        pl.col("outcome_horizon").cast(pl.Int64).alias("entry_horizon"),
        pl.col("outcome_horizon").cast(pl.Int64).alias("max_valid_horizon"),
        pl.col("selected_tail_state").fill_null("none").alias("best_path_state"),
        pl.col("selected_tail_state").fill_null("none").alias("path_state_profile"),
        pl.col("selected_utility_margin").fill_null(0.0).alias("best_utility_margin"),
    )
    clean_side = (
        pl.col("selected_side_only").fill_null(False)
        & ~pl.col("selected_tail_both").fill_null(False)
        & (pl.col("calibrated_side_margin").fill_null(0.0) > 0.0)
        & (pl.col("selected_utility_margin").fill_null(0.0) > 0.0)
    )
    mixed_path = (
        pl.col("selected_tail_both").fill_null(False)
        | pl.col("opposite_tail_both").fill_null(False)
        | pl.col("best_path_state").is_in(["up_first_both", "down_first_both", "chop_both"])
    )
    opposite_dominates = pl.col("false_direction_int").fill_null(0).cast(pl.Int64) == 1
    actionability = (
        pl.when(clean_side)
        .then(pl.lit("trade_candidate"))
        .when(opposite_dominates)
        .then(pl.lit("reversal_watch"))
        .when(mixed_path)
        .then(pl.lit("gray_zone"))
        .otherwise(pl.lit("no_action"))
    )
    blocker = (
        pl.when(clean_side)
        .then(pl.lit(""))
        .when(opposite_dominates)
        .then(pl.lit("opposite_tail_dominates"))
        .when(mixed_path)
        .then(pl.lit("both_or_mixed_path"))
        .otherwise(pl.lit("no_clean_side"))
    )
    return work.with_columns(
        actionability.alias("actionability"),
        blocker.alias("blocker_reason"),
        pl.when(clean_side).then(pl.lit(1)).otherwise(pl.lit(0)).alias("clean_horizon_count"),
        pl.when(mixed_path).then(pl.lit(1)).otherwise(pl.lit(0)).alias("chop_horizon_count"),
        pl.when(opposite_dominates)
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("contradicting_horizon_count"),
        pl.when(actionability == "reversal_watch")
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("reversal_horizon_count"),
    ).select(
        "symbol",
        "decision_bar_close_ms",
        "action_side",
        "entry_horizon",
        "max_valid_horizon",
        "actionability",
        "path_state_profile",
        "best_path_state",
        "best_utility_margin",
        "clean_horizon_count",
        "chop_horizon_count",
        "reversal_horizon_count",
        "contradicting_horizon_count",
        "calibrated_side_margin",
        "false_direction_int",
        "blocker_reason",
        "score_bucket",
    )


def candidate_replay_metrics(
    replay: pl.DataFrame,
    *,
    outcome_horizon: int,
    direction: TailtreeDirection | str,
    score_bucket: str,
) -> dict[str, float]:
    empty = {
        "candidate_pair_count": 0.0,
        "paired_opposite_rate": 0.0,
        "paired_gray_zone_rate": 0.0,
        "paired_false_direction_rate": 0.0,
        "paired_false_direction_cost_mean": 0.0,
        "paired_directional_margin_mean": 0.0,
        "paired_side_only_rate": 0.0,
        "paired_tail_both_rate": 0.0,
        "paired_selected_utility_margin_mean": 0.0,
        "paired_calibrated_directional_margin_mean": 0.0,
        "paired_calibrated_side_margin_mean": 0.0,
    }
    if replay.is_empty():
        return empty

    selected = replay.filter(
        (pl.col("selected_direction") == str(direction))
        & (pl.col("outcome_horizon") == int(outcome_horizon))
        & (pl.col("score_bucket") == str(score_bucket))
    )
    if selected.is_empty():
        return empty

    false_cost = selected.filter(pl.col("false_direction_int") == 1).get_column("opposite_utility")

    def mean(values: pl.Series) -> float:
        if values.is_empty():
            return 0.0
        value = values.to_frame().select(pl.col(values.name).cast(pl.Float64).mean()).item()
        return float(value) if value is not None else 0.0

    metrics = {
        "candidate_pair_count": float(selected.height),
        "paired_opposite_rate": float(
            selected.select(pl.col("opposite_score").is_not_null().mean()).item() or 0.0
        ),
        "paired_gray_zone_rate": mean(selected.get_column("gray_zone_int")),
        "paired_false_direction_rate": mean(
            selected.get_column("false_direction_int")
        ),
        "paired_false_direction_cost_mean": mean(false_cost),
        "paired_directional_margin_mean": mean(
            selected.get_column("directional_score_margin")
        ),
        "paired_side_only_rate": mean(selected.get_column("side_only_int")),
        "paired_tail_both_rate": mean(selected.get_column("tail_both_int")),
        "paired_selected_utility_margin_mean": mean(
            selected.get_column("selected_utility_margin")
        ),
    }
    if "calibrated_directional_margin" in selected.columns:
        metrics["paired_calibrated_directional_margin_mean"] = mean(
            selected.get_column("calibrated_directional_margin")
        )
    if "calibrated_side_margin" in selected.columns:
        metrics["paired_calibrated_side_margin_mean"] = mean(
            selected.get_column("calibrated_side_margin")
        )
    return metrics


def directional_objective_score(
    base_score: float,
    metrics: dict[str, float],
    *,
    false_rate_weight: float = 10.0,
    false_cost_weight: float = 1.0,
    margin_weight: float = 5.0,
    gray_weight: float = 0.0,
) -> float:
    return (
        base_score
        + margin_weight * metrics["paired_directional_margin_mean"]
        - false_rate_weight * metrics["paired_false_direction_rate"]
        - false_cost_weight * metrics["paired_false_direction_cost_mean"]
        - gray_weight * metrics["paired_gray_zone_rate"]
    )


def side_objective_score(
    base_score: float,
    metrics: dict[str, float],
    *,
    false_rate_weight: float = 10.0,
    false_cost_weight: float = 1.0,
    margin_weight: float = 5.0,
    side_only_weight: float = 5.0,
    tail_both_weight: float = 5.0,
) -> float:
    return (
        base_score
        + margin_weight * metrics["paired_directional_margin_mean"]
        + side_only_weight * metrics["paired_side_only_rate"]
        - false_rate_weight * metrics["paired_false_direction_rate"]
        - false_cost_weight * metrics["paired_false_direction_cost_mean"]
        - tail_both_weight * metrics["paired_tail_both_rate"]
    )


def tailtree_selection_metrics_frame(
    run: TailtreeProfileRun,
    evidence: pl.DataFrame,
    prepared: TailtreePreparedFrames,
    models: dict[tuple[int, TailtreeDirection], TailtreeArtifactTree],
    seconds: float,
    candidate_replay: pl.DataFrame,
) -> pl.DataFrame:
    if evidence.is_empty():
        return pl.DataFrame()
    eligible_symbol_count = prepared.observations.get_column("symbol").n_unique()
    observation_row_count = prepared.observations.height
    feature_count = len(prepared.categorical_features) + len(prepared.continuous_features)
    rows: list[dict[str, object]] = []
    for row in evidence.to_dicts():
        selected_observations = int(row.get("N_total") or 0)
        selected_tails = int(row.get("N_tail_exceedances") or 0)
        valid_tail_lift = float(row.get("tail_lift") or 0.0)
        selected_tail_rate = (
            selected_tails / selected_observations if selected_observations else 0.0
        )
        valid_tail_rate = selected_tail_rate / valid_tail_lift if valid_tail_lift else 0.0
        selected_symbols = int(row.get("symbol_count") or eligible_symbol_count)
        utility_mean = float(row.get("tail_utility_mean") or 0.0)
        utility_p90 = float(row.get("tail_utility_p90") or 0.0)
        profit_per_1k = (utility_mean * 1000.0) / max(float(selected_observations), 1.0)
        base_hpo_score = valid_tail_lift + utility_mean + (selected_tails + 1.0) ** 0.5 / 10.0
        replay_metrics = candidate_replay_metrics(
            candidate_replay,
            outcome_horizon=int(row.get("outcome_horizon") or 0),
            direction=str(row.get("tree_direction") or ""),
            score_bucket=str(row.get("score_bucket") or ""),
        )
        row_values = TailtreeSelectionEfficiencyRow(
            universe_snapshot_id="active",
            model_tag=run.model_tag,
            objective=run.objective,
            training_profile=run.profile_id,
            trial_id=run.run_id.rsplit("-t", 1)[0] if run.run_source == "optuna" else run.run_id,
            trial_source=run.run_source,
            outcome_horizon=int(row.get("outcome_horizon") or 0),
            tree_direction=str(row.get("tree_direction") or ""),
            budget_family="score_bucket" if row.get("score_bucket") is not None else "leaf",
            budget_value=float(
                str(row.get("score_bucket") or row.get("leaf_id") or 0)
                .replace("top_", "")
                .replace("pct", "")
                or 0.0
            ),
            eligible_symbol_count=int(eligible_symbol_count),
            selected_symbol_count=selected_symbols,
            observation_row_count=int(observation_row_count),
            feature_count=int(feature_count),
            train_exceedance_count=selected_tails,
            valid_observation_count=int(observation_row_count),
            valid_tail_count=selected_tails,
            valid_tail_rate=float(valid_tail_rate),
            selected_observation_count=selected_observations,
            selected_observation_rate=selected_observations / observation_row_count
            if observation_row_count
            else 0.0,
            selected_tail_count=selected_tails,
            selected_tail_rate=float(selected_tail_rate),
            selected_tail_per_1k_obs=(selected_tails * 1000.0) / selected_observations
            if selected_observations
            else 0.0,
            valid_tail_lift=valid_tail_lift,
            selected_profit_proxy_mean=utility_mean,
            selected_profit_proxy_p90=utility_p90,
            selected_utility_mean=utility_mean,
            selected_utility_p90=utility_p90,
            profit_proxy_per_selected_obs=utility_mean,
            profit_proxy_per_1k_observed=profit_per_1k,
            hpo_score=base_hpo_score,
            promotion_threshold_pass_int=int(
                selected_observations >= 500 and valid_tail_lift >= 3.0
            ),
            trained_tree_count=len(models),
            selected_bucket_or_leaf_count=1,
            fit_seconds=seconds,
            score_seconds=0.0,
        ).__dict__
        row_values.update(
            {
                "base_hpo_score": base_hpo_score,
                "objective_hpo_score": directional_objective_score(base_hpo_score, replay_metrics),
                "side_hpo_score": side_objective_score(base_hpo_score, replay_metrics),
                **replay_metrics,
            }
        )
        rows.append(row_values)
    return pl.DataFrame(rows)


__all__ = [
    "candidate_replay_metrics",
    "calibrated_candidate_replay_frame",
    "directional_objective_score",
    "paired_candidate_replay_frame",
    "score_bucket_candidate_frame",
    "side_objective_score",
    "tailtree_action_surface_frame",
    "tailtree_selection_metrics_frame",
]
