"""Tailtree train/load_predict lifecycle boundary."""

from __future__ import annotations

import logging

import polars as pl

from qooi.scanner import PotentialScanConfig, ReportInputs
from qooi.scanner import outcome as outcome_eval
from qooi.scanner.tailrun.artifacts import (
    _cleanup_tailtree_artifacts,
    _load_tail_tree_evidence,
    _write_tailtree_artifacts,
)
from qooi.scanner.tailrun.types import (
    TAILTREE_RUN_SUMMARY_SCHEMA,
    TailtreeArtifactTree,
    TailtreeDirection,
    TailtreeDirectionQuality,
    TailtreeEvidenceResult,
    TailtreeResult,
)
from qooi.scanner.tailtree import TailtreeTrainingFrame

_TAILTREE_CATEGORICAL_TRAIN_FEATURES = (
    "background_regime",
    "swing_core",
    "decision_core",
    "decision_transition",
    "decision_direction",
)
_TAILTREE_CONTINUOUS_TRAIN_FEATURES = (
    "atr_percentile",
    "range_width_atr",
    "return_1bar",
    "return_4bar",
    "return_24bar",
    "vol_anomaly",
    "close_to_range_high_ratio",
    "funding_rate",
    "oi_delta",
    "taker_buy_sell_ratio",
    "long_short_ratio",
    "funding_age_ms",
    "oi_age_ms",
    "taker_age_ms",
    "lsr_age_ms",
)


def _tailtree_training_features(observations: pl.DataFrame) -> tuple[list[str], list[str]]:
    """Select persistent known-at-close features allowed for tailtree training.

    Ephemeral current-review/cost features may exist in the observation frame, but
    column presence alone does not make them historical model inputs.
    """
    categorical = [c for c in _TAILTREE_CATEGORICAL_TRAIN_FEATURES if c in observations.columns]
    continuous = [c for c in _TAILTREE_CONTINUOUS_TRAIN_FEATURES if c in observations.columns]
    return categorical, continuous


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _aggregate_quality(
    quality_by_direction: dict[TailtreeDirection, TailtreeDirectionQuality],
) -> TailtreeDirectionQuality:
    qualities = list(quality_by_direction.values())
    if not qualities:
        return TailtreeDirectionQuality.zero("up")
    train_tail_count = sum(q.train_tail_count for q in qualities)
    valid_observation_count = sum(q.valid_observation_count for q in qualities)
    valid_tail_count = sum(q.valid_tail_count for q in qualities)
    valid_selected_observation_count = sum(
        q.valid_selected_observation_count for q in qualities
    )
    valid_selected_tail_count = sum(q.valid_selected_tail_count for q in qualities)
    valid_selected_utility_mean = _rate(
        sum(q.valid_selected_utility_mean * q.valid_selected_tail_count for q in qualities),
        valid_selected_tail_count,
    )
    valid_selected_utility_p90 = max(q.valid_selected_utility_p90 for q in qualities)
    valid_tail_rate = _rate(valid_tail_count, valid_observation_count)
    valid_selected_tail_rate = _rate(
        valid_selected_tail_count, valid_selected_observation_count
    )
    return TailtreeDirectionQuality(
        direction="up",
        train_tail_count=train_tail_count,
        valid_observation_count=valid_observation_count,
        valid_tail_count=valid_tail_count,
        valid_tail_rate=valid_tail_rate,
        valid_selected_observation_count=valid_selected_observation_count,
        valid_selected_tail_count=valid_selected_tail_count,
        valid_selected_tail_rate=valid_selected_tail_rate,
        valid_tail_lift=valid_selected_tail_rate / valid_tail_rate
        if valid_tail_rate > 0
        else 0.0,
        valid_selected_utility_mean=valid_selected_utility_mean,
        valid_selected_utility_p90=valid_selected_utility_p90,
    )


def _tail_train_count(exceedance_count: int, validation_fraction: float) -> int:
    if exceedance_count <= 0:
        return 0
    valid_count = max(1, int(exceedance_count * validation_fraction))
    return max(0, exceedance_count - valid_count)


def _score_bucket_quality(
    direction: TailtreeDirection,
    train_tail_count: int,
    score_evidence: pl.DataFrame,
) -> TailtreeDirectionQuality:
    if score_evidence.is_empty():
        return TailtreeDirectionQuality.zero(direction)
    best = score_evidence.sort(
        ["tail_lift", "N_tail_exceedances"], descending=[True, True]
    ).row(0, named=True)
    valid_observation_count = int(best["N_total"] or 0)
    valid_tail_count = int(
        round(valid_observation_count * float(best["global_tail_rate"] or 0.0))
    )
    return TailtreeDirectionQuality(
        direction=direction,
        train_tail_count=train_tail_count,
        valid_observation_count=valid_observation_count,
        valid_tail_count=valid_tail_count,
        valid_tail_rate=float(best["global_tail_rate"] or 0.0),
        valid_selected_observation_count=valid_observation_count,
        valid_selected_tail_count=int(best["N_tail_exceedances"] or 0),
        valid_selected_tail_rate=float(best["leaf_tail_rate"] or 0.0),
        valid_tail_lift=float(best["tail_lift"] or 0.0),
        valid_selected_utility_mean=float(best["tail_utility_mean"] or 0.0),
        valid_selected_utility_p90=float(best["tail_utility_p90"] or 0.0),
    )


def _validation_leaf_frame(
    tree: TailtreeArtifactTree,
    training: TailtreeTrainingFrame,
    labeled_outcomes: pl.DataFrame,
    *,
    direction: TailtreeDirection,
    validation_fraction: float,
) -> pl.DataFrame:
    if training.all_observations.is_empty():
        return pl.DataFrame()
    valid_count = max(1, int(len(training.all_observations) * validation_fraction))
    validation_observations = training.all_observations.sort(
        "decision_bar_close_ms"
    ).tail(valid_count)
    tail_col = f"tail_{direction}"
    utility_col = f"tail_utility_{direction}"
    if tail_col not in labeled_outcomes.columns:
        return tree.predict_leaf(validation_observations)
    aggregations = [
        pl.col(tail_col).fill_null(False).cast(pl.Boolean).max().alias(tail_col)
    ]
    if utility_col in labeled_outcomes.columns:
        aggregations.append(pl.col(utility_col).fill_null(0.0).cast(pl.Float64).max().alias(utility_col))
    outcome_tail = labeled_outcomes.group_by("symbol", "decision_bar_close_ms").agg(
        *aggregations
    )
    return tree.predict_leaf(validation_observations).join(
        outcome_tail,
        on=["symbol", "decision_bar_close_ms"],
        how="left",
    )


def run(
    observations: pl.DataFrame,
    source_outcomes: pl.DataFrame,
    realized_transitions: pl.DataFrame,
    inputs: ReportInputs,
    *,
    source_event_row_count: int,
) -> TailtreeEvidenceResult:
    """Run the explicit tailtree lifecycle selected by config."""

    if inputs.config.evidence.tailtree.lifecycle == "load_predict":
        return load_predict(observations, inputs)
    return train_evaluate_predict(
        observations,
        source_outcomes,
        realized_transitions,
        inputs,
        source_event_row_count=source_event_row_count,
    )


def train_evaluate_predict(
    observations: pl.DataFrame,
    source_outcomes: pl.DataFrame,
    realized_transitions: pl.DataFrame,
    inputs: ReportInputs,
    *,
    source_event_row_count: int,
) -> TailtreeEvidenceResult:
    """Train tailtree evidence, persist artifacts, and score observations."""

    return _build_tail_tree_evidence(
        observations,
        source_outcomes,
        realized_transitions,
        inputs,
        source_event_row_count=source_event_row_count,
    )


def load_predict(
    observations: pl.DataFrame,
    inputs: ReportInputs,
) -> TailtreeEvidenceResult:
    """Load frozen tailtree artifacts and score current observations."""

    return _load_tail_tree_evidence(observations, inputs)






def _safe_nonnull_count(frame: pl.DataFrame, column: str) -> int:
    if frame.is_empty() or column not in frame.columns:
        return 0
    return len(frame.get_column(column).drop_nulls())


def _safe_true_count(frame: pl.DataFrame, column: str) -> int:
    if frame.is_empty() or column not in frame.columns:
        return 0
    return int(frame.get_column(column).fill_null(False).sum())


def _safe_utility_values(frame: pl.DataFrame, columns: tuple[str, ...]) -> pl.Series:
    values = [
        frame.get_column(column).fill_null(0.0).cast(pl.Float64)
        for column in columns
        if column in frame.columns
    ]
    if not values:
        return pl.Series("tail_utility", [], dtype=pl.Float64)
    combined = pl.concat(values)
    return combined.filter(combined > 0.0)


def _safe_utility_mean(frame: pl.DataFrame, columns: tuple[str, ...]) -> float:
    values = _safe_utility_values(frame, columns)
    return float(values.mean() or 0.0) if not values.is_empty() else 0.0


def _safe_utility_p90(frame: pl.DataFrame, columns: tuple[str, ...]) -> float:
    values = _safe_utility_values(frame, columns)
    return float(values.quantile(0.9) or 0.0) if not values.is_empty() else 0.0


def _filter_outcome_horizon(frame: pl.DataFrame, outcome_horizon: int) -> pl.DataFrame:
    if frame.is_empty() or "outcome_horizon" not in frame.columns:
        return frame
    return frame.filter(pl.col("outcome_horizon") == outcome_horizon)


def _tailtree_run_summary_frame(
    config: PotentialScanConfig,
    *,
    observations: pl.DataFrame,
    source_event_row_count: int,
    source_outcomes: pl.DataFrame,
    realized_transitions: pl.DataFrame,
    outcome_frame: pl.DataFrame,
    categorical_features: list[str],
    continuous_features: list[str],
    train_counts: dict[str, tuple[int, int]],
    selected_leaf_counts: dict[str, int],
    trained_tree_count: int,
    outcome_horizon: int | None = None,
    quality_by_direction: dict[TailtreeDirection, TailtreeDirectionQuality] | None = None,
) -> pl.DataFrame:
    min_exceedance_required = int(config.evidence.tailtree.min_data_in_leaf)
    feature_count = len(categorical_features) + len(continuous_features)
    outcome_rows = len(outcome_frame)
    run_tail_count = _safe_true_count(outcome_frame, "tail_up") + _safe_true_count(
        outcome_frame, "tail_down"
    )
    utility_cols = ("tail_utility_up", "tail_utility_down")
    common = {
        "observation_row_count": len(observations),
        "outcome_row_count": outcome_rows,
        "source_event_row_count": source_event_row_count,
        "source_outcome_row_count": len(source_outcomes),
        "realized_transition_row_count": len(realized_transitions),
        "feature_count": feature_count,
        "categorical_feature_count": len(categorical_features),
        "continuous_feature_count": len(continuous_features),
        "forward_return_nonnull_count": _safe_nonnull_count(outcome_frame, "forward_return_pct"),
        "forward_min_return_nonnull_count": _safe_nonnull_count(
            outcome_frame, "forward_min_return_pct"
        ),
        "forward_max_return_nonnull_count": _safe_nonnull_count(
            outcome_frame, "forward_max_return_pct"
        ),
        "path_range_nonnull_count": _safe_nonnull_count(outcome_frame, "path_range_pct"),
        "time_to_max_nonnull_count": _safe_nonnull_count(outcome_frame, "time_to_max_bar"),
        "time_to_min_nonnull_count": _safe_nonnull_count(outcome_frame, "time_to_min_bar"),
        "retention_nonnull_count": _safe_nonnull_count(
            outcome_frame, "close_retention_ratio"
        ),
        "path_efficiency_nonnull_count": _safe_nonnull_count(
            outcome_frame, "path_efficiency"
        ),
        "tail_utility_mean": _safe_utility_mean(outcome_frame, utility_cols),
        "tail_utility_p90": _safe_utility_p90(outcome_frame, utility_cols),
        "threshold_pct": float(config.evidence.tailtree.threshold_pct),
        "min_exceedance_required": min_exceedance_required,
        "trained_tree_count": trained_tree_count,
        "written_model_file_count": 0,
        "written_evidence_file_count": 0,
        "removed_stale_file_count": 0,
    }
    quality_by_direction = quality_by_direction or {}
    run_quality = _aggregate_quality(quality_by_direction)
    summary_horizon = int(outcome_horizon or config.evidence.tailtree.outcome_horizon[0])
    rows = [
        {
            "summary_scope": "run",
            "direction": "",
            "objective": config.evidence.tailtree.objective,
            "outcome_horizon": summary_horizon,
            **common,
            **run_quality.to_summary_fields(),
            "tail_count": run_tail_count,
            "tail_rate": _rate(run_tail_count, outcome_rows),
            "train_observation_count": outcome_rows,
            "train_exceedance_count": run_tail_count,
            "trainable_flag": int(
                feature_count > 0 and run_tail_count >= min_exceedance_required
            ),
            "selected_leaf_count": sum(selected_leaf_counts.values()),
        }
    ]
    for direction in ("up", "down"):
        train_observations, train_exceedances = train_counts.get(direction, (0, 0))
        quality = quality_by_direction.get(
            direction, TailtreeDirectionQuality.zero(direction, train_tail_count=train_exceedances)
        )
        rows.append(
            {
                "summary_scope": direction,
                "direction": direction,
                "objective": config.evidence.tailtree.objective,
                "outcome_horizon": summary_horizon,
                **common,
                "tail_utility_mean": _safe_utility_mean(
                    outcome_frame, (f"tail_utility_{direction}",)
                ),
                "tail_utility_p90": _safe_utility_p90(
                    outcome_frame, (f"tail_utility_{direction}",)
                ),
                **quality.to_summary_fields(),
                "tail_count": _safe_true_count(outcome_frame, f"tail_{direction}"),
                "tail_rate": _rate(train_exceedances, train_observations),
                "train_observation_count": train_observations,
                "train_exceedance_count": train_exceedances,
                "trainable_flag": int(
                    feature_count > 0 and train_exceedances >= min_exceedance_required
                ),
                "selected_leaf_count": selected_leaf_counts.get(direction, 0),
            }
        )
    return pl.DataFrame(rows, schema=TAILTREE_RUN_SUMMARY_SCHEMA)




def _build_tail_tree_evidence(
    observations: pl.DataFrame,
    source_outcomes: pl.DataFrame,
    realized_transitions: pl.DataFrame,
    inputs: ReportInputs,
    *,
    source_event_row_count: int,
) -> TailtreeEvidenceResult:
    from qooi.scanner.tailtree import (
        TailTreeModel,
        TrainConfig,
        label_tail_exceedances,
        leaf_context_frame,
        leaf_evidence_frame,
        score_bucket_evidence_frame,
        select_tail_leaves,
        tailtree_training_frame,
    )

    logger = logging.getLogger("qooi.scanner")
    inputs.artifacts.diagnostics_dir.mkdir(parents=True, exist_ok=True)

    outcome_frame = outcome_eval.potential_outcome_frame(
        observations,
        source_outcomes,
        realized_transitions,
        return_threshold_pct=inputs.config.transition.return_threshold_pct,
    )
    if outcome_frame.is_empty():
        return TailtreeEvidenceResult(pl.DataFrame(), {})

    labeled_outcome_frame = label_tail_exceedances(
        outcome_frame, threshold_pct=inputs.config.evidence.tailtree.threshold_pct
    )
    config = TrainConfig(
        objective=inputs.config.evidence.tailtree.objective,
        num_leaves=inputs.config.evidence.tailtree.num_leaves,
        min_data_in_leaf=inputs.config.evidence.tailtree.min_data_in_leaf,
        learning_rate=inputs.config.evidence.tailtree.learning_rate,
        num_iterations=inputs.config.evidence.tailtree.num_iterations,
        early_stopping_rounds=inputs.config.evidence.tailtree.early_stopping_rounds,
    )
    cat, con = _tailtree_training_features(observations)

    all_evidence: list[pl.DataFrame] = []
    summary_frames: list[pl.DataFrame] = []
    models: dict[tuple[int, TailtreeDirection], TailtreeArtifactTree] = {}
    removed_stale_file_count = _cleanup_tailtree_artifacts(inputs)
    for outcome_horizon in inputs.config.evidence.tailtree.outcome_horizon:
        horizon_outcome_frame = _filter_outcome_horizon(labeled_outcome_frame, outcome_horizon)
        if horizon_outcome_frame.is_empty():
            continue
        tail_up_count = (
            int(horizon_outcome_frame.get_column("tail_up").sum())
            if "tail_up" in horizon_outcome_frame.columns
            else 0
        )
        tail_down_count = (
            int(horizon_outcome_frame.get_column("tail_down").sum())
            if "tail_down" in horizon_outcome_frame.columns
            else 0
        )
        logger.info(
            "outcome_horizon=%d outcome rows=%d tail_up=%d tail_down=%d",
            outcome_horizon,
            len(horizon_outcome_frame),
            tail_up_count,
            tail_down_count,
        )
        evidence_by_direction: dict[TailtreeDirection, pl.DataFrame] = {}
        trees: dict[TailtreeDirection, TailtreeArtifactTree] = {}
        train_counts: dict[str, tuple[int, int]] = {}
        selected_leaf_counts: dict[str, int] = {}
        quality_by_direction: dict[TailtreeDirection, TailtreeDirectionQuality] = {}
        for direction in ("up", "down"):
            training = tailtree_training_frame(
                observations, horizon_outcome_frame, direction=direction
            )
            train_counts[direction] = (
                training.train_n_observations,
                training.train_n_exceedances,
            )
            if not training.has_min_exceedances(config.min_data_in_leaf):
                selected_leaf_counts[direction] = 0
                quality_by_direction[direction] = TailtreeDirectionQuality.zero(direction)
                continue

            tree = TailTreeModel.train(
                training.tail_observations,
                training.exceedance_values,
                config=config,
                categorical_features=cat,
                continuous_features=con,
                direction=direction,
                global_tail_rate=training.global_tail_rate,
                train_n_observations=training.train_n_observations,
                utility_values=training.utility_values,
            )
            trees[direction] = tree
            models[(outcome_horizon, direction)] = tree

            if config.objective == "tail_utility_quantile":
                merged = score_bucket_evidence_frame(
                    tree, observations, horizon_outcome_frame
                ).with_columns(pl.lit(outcome_horizon).alias("outcome_horizon"))
                selected = merged
            else:
                lev = leaf_evidence_frame(tree, observations, horizon_outcome_frame)
                if lev.is_empty():
                    continue
                lctx = leaf_context_frame(tree, observations, horizon_outcome_frame)
                merged = lev.join(lctx, on="leaf_id", how="left").with_columns(
                    pl.lit(outcome_horizon).alias("outcome_horizon")
                )
                selected = select_tail_leaves(merged)
            if merged.is_empty():
                continue
            selected_leaf_counts[direction] = len(selected)
            selected_leaf_ids = (
                set(selected.get_column("leaf_id").cast(pl.Int64).to_list())
                if "leaf_id" in selected.columns
                else set()
            )
            if config.objective == "tail_utility_quantile":
                quality_by_direction[direction] = _score_bucket_quality(
                    direction,
                    _tail_train_count(training.train_n_exceedances, config.validation_fraction),
                    selected,
                )
            else:
                validation_leaf_frame = _validation_leaf_frame(
                    tree,
                    training,
                    horizon_outcome_frame,
                    direction=direction,
                    validation_fraction=config.validation_fraction,
                )
                quality_by_direction[direction] = TailtreeDirectionQuality.from_labeled_leaf_frame(
                    direction=direction,
                    train_tail_count=_tail_train_count(
                        training.train_n_exceedances, config.validation_fraction
                    ),
                    validation_leaf_frame=validation_leaf_frame,
                    selected_leaf_ids=selected_leaf_ids,
                )
            suffix = f"h{int(outcome_horizon)}"
            selected.write_csv(
                inputs.artifacts.diagnostics_dir
                / f"potential-leaves-selected-{suffix}-{direction}.csv"
            )
            evidence_by_direction[direction] = merged
            all_evidence.append(merged)

        summary = _tailtree_run_summary_frame(
            inputs.config,
            observations=observations,
            source_event_row_count=source_event_row_count,
            source_outcomes=_filter_outcome_horizon(source_outcomes, outcome_horizon),
            realized_transitions=_filter_outcome_horizon(realized_transitions, outcome_horizon),
            outcome_frame=horizon_outcome_frame,
            categorical_features=cat,
            continuous_features=con,
            train_counts=train_counts,
            selected_leaf_counts=selected_leaf_counts,
            trained_tree_count=len(trees),
            outcome_horizon=outcome_horizon,
            quality_by_direction=quality_by_direction,
        )
        summary_frames.append(summary)
        _write_tailtree_artifacts(
            inputs,
            evidence_by_direction,
            trees,
            summary=pl.concat(summary_frames, how="diagonal_relaxed"),
            outcome_horizon=outcome_horizon,
            removed_stale_file_count=removed_stale_file_count,
            cleanup=False,
            categorical_features=cat,
            continuous_features=con,
        )
        removed_stale_file_count = 0

    ev = pl.concat(all_evidence, how="diagonal_relaxed") if all_evidence else pl.DataFrame()
    return TailtreeEvidenceResult(ev, models)


__all__ = [
    "TailtreeEvidenceResult",
    "TailtreeResult",
    "load_predict",
    "run",
    "train_evaluate_predict",
]
