"""Tailtree train/load_predict lifecycle boundary."""

from __future__ import annotations

import logging
from dataclasses import replace
from time import perf_counter
from typing import TYPE_CHECKING

import polars as pl

from qooi.profiling import ProfileContext
from qooi.scanner import PotentialScanConfig
from qooi.scanner import outcome as outcome_eval
from qooi.scanner.config import PotentialConfig, TailtreeOptunaTrainingConfig
from qooi.scanner.tailrun.artifacts import (
    _cleanup_tailtree_artifacts,
    _load_tail_tree_evidence,
    _write_tailtree_artifacts,
)
from qooi.scanner.tailrun.types import (
    TAILTREE_RUN_SUMMARY_SCHEMA,
    ReportInputs,
    TailtreeArtifactTree,
    TailtreeDirection,
    TailtreeDirectionQuality,
    TailtreeEvidenceResult,
    TailtreeFrameSplit,
    TailtreeInputFrames,
    TailtreePreparedFrames,
    TailtreeProfileFeedback,
    TailtreeResult,
    TailtreeRunOutput,
    TailtreeSelectionEfficiencyRow,
)

if TYPE_CHECKING:
    from qooi.scanner.tailrun.planning import TailtreeProfileRun

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
    valid_selected_observation_count = sum(q.valid_selected_observation_count for q in qualities)
    valid_selected_tail_count = sum(q.valid_selected_tail_count for q in qualities)
    valid_selected_utility_mean = _rate(
        sum(q.valid_selected_utility_mean * q.valid_selected_tail_count for q in qualities),
        valid_selected_tail_count,
    )
    valid_selected_utility_p90 = max(q.valid_selected_utility_p90 for q in qualities)
    valid_tail_rate = _rate(valid_tail_count, valid_observation_count)
    valid_selected_tail_rate = _rate(valid_selected_tail_count, valid_selected_observation_count)
    return TailtreeDirectionQuality(
        direction="up",
        train_tail_count=train_tail_count,
        valid_observation_count=valid_observation_count,
        valid_tail_count=valid_tail_count,
        valid_tail_rate=valid_tail_rate,
        valid_selected_observation_count=valid_selected_observation_count,
        valid_selected_tail_count=valid_selected_tail_count,
        valid_selected_tail_rate=valid_selected_tail_rate,
        valid_tail_lift=valid_selected_tail_rate / valid_tail_rate if valid_tail_rate > 0 else 0.0,
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
    best = score_evidence.sort(["tail_lift", "N_tail_exceedances"], descending=[True, True]).row(
        0, named=True
    )
    valid_observation_count = int(best["N_total"] or 0)
    valid_tail_count = int(round(valid_observation_count * float(best["global_tail_rate"] or 0.0)))
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
    validation_observations: pl.DataFrame,
    labeled_outcomes: pl.DataFrame,
    *,
    direction: TailtreeDirection,
) -> pl.DataFrame:
    if validation_observations.is_empty():
        return pl.DataFrame()
    tail_col = f"tail_{direction}"
    utility_col = f"tail_utility_{direction}"
    if tail_col not in labeled_outcomes.columns:
        return tree.predict_leaf(validation_observations)
    aggregations = [pl.col(tail_col).fill_null(False).cast(pl.Boolean).max().alias(tail_col)]
    if utility_col in labeled_outcomes.columns:
        aggregations.append(
            pl.col(utility_col).fill_null(0.0).cast(pl.Float64).max().alias(utility_col)
        )
    outcome_tail = labeled_outcomes.group_by("symbol", "decision_bar_close_ms").agg(*aggregations)
    return tree.predict_leaf(validation_observations).join(
        outcome_tail,
        on=["symbol", "decision_bar_close_ms"],
        how="left",
    )


def run_frame_split(
    frames: TailtreeFrameSplit,
    inputs: ReportInputs,
    *,
    source_event_row_count: int,
) -> TailtreeEvidenceResult:
    """Train on one frame split and score/evaluate its validation frame."""

    if inputs.config.evidence.tailtree.lifecycle == "load_predict":
        return load_predict(frames.valid_observations, inputs)
    return _build_tail_tree_evidence(
        frames,
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
    first_profile = config.evidence.tailtree.profiles[0]
    min_exceedance_required = int(first_profile.training.min_data_in_leaf)
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
        "retention_nonnull_count": _safe_nonnull_count(outcome_frame, "close_retention_ratio"),
        "path_efficiency_nonnull_count": _safe_nonnull_count(outcome_frame, "path_efficiency"),
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
            "objective": first_profile.objective,
            "outcome_horizon": summary_horizon,
            **common,
            **run_quality.to_summary_fields(),
            "tail_count": run_tail_count,
            "tail_rate": _rate(run_tail_count, outcome_rows),
            "train_observation_count": outcome_rows,
            "train_exceedance_count": run_tail_count,
            "trainable_flag": int(feature_count > 0 and run_tail_count >= min_exceedance_required),
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
                "objective": first_profile.objective,
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
    frames: TailtreeFrameSplit,
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

    train_outcome_frame = outcome_eval.potential_outcome_frame(
        frames.train_observations,
        frames.train_source_outcomes,
        frames.train_realized_transitions,
        return_threshold_pct=inputs.config.transition.return_threshold_pct,
    )
    valid_outcome_frame = outcome_eval.potential_outcome_frame(
        frames.valid_observations,
        frames.valid_source_outcomes,
        frames.valid_realized_transitions,
        return_threshold_pct=inputs.config.transition.return_threshold_pct,
    )
    if train_outcome_frame.is_empty() or valid_outcome_frame.is_empty():
        return TailtreeEvidenceResult(pl.DataFrame(), {})

    labeled_train_outcome_frame = label_tail_exceedances(
        train_outcome_frame, threshold_pct=inputs.config.evidence.tailtree.threshold_pct
    )
    labeled_valid_outcome_frame = label_tail_exceedances(
        valid_outcome_frame, threshold_pct=inputs.config.evidence.tailtree.threshold_pct
    )
    first_profile = inputs.config.evidence.tailtree.profiles[0]
    profile_training = first_profile.training
    config = TrainConfig(
        objective=first_profile.objective,
        num_leaves=profile_training.num_leaves,
        min_data_in_leaf=profile_training.min_data_in_leaf,
        learning_rate=profile_training.learning_rate,
        num_iterations=profile_training.num_iterations,
        early_stopping_rounds=profile_training.early_stopping_rounds,
    )
    cat, con = _tailtree_training_features(frames.train_observations)

    all_evidence: list[pl.DataFrame] = []
    summary_frames: list[pl.DataFrame] = []
    models: dict[tuple[int, TailtreeDirection], TailtreeArtifactTree] = {}
    removed_stale_file_count = _cleanup_tailtree_artifacts(inputs)
    for outcome_horizon in inputs.config.evidence.tailtree.outcome_horizon:
        train_horizon_outcome_frame = _filter_outcome_horizon(
            labeled_train_outcome_frame, outcome_horizon
        )
        valid_horizon_outcome_frame = _filter_outcome_horizon(
            labeled_valid_outcome_frame, outcome_horizon
        )
        if train_horizon_outcome_frame.is_empty() or valid_horizon_outcome_frame.is_empty():
            continue
        tail_up_count = (
            int(valid_horizon_outcome_frame.get_column("tail_up").sum())
            if "tail_up" in valid_horizon_outcome_frame.columns
            else 0
        )
        tail_down_count = (
            int(valid_horizon_outcome_frame.get_column("tail_down").sum())
            if "tail_down" in valid_horizon_outcome_frame.columns
            else 0
        )
        logger.info(
            "outcome_horizon=%d outcome rows=%d tail_up=%d tail_down=%d",
            outcome_horizon,
            len(valid_horizon_outcome_frame),
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
                frames.train_observations, train_horizon_outcome_frame, direction=direction
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
                    tree, frames.valid_observations, valid_horizon_outcome_frame
                ).with_columns(pl.lit(outcome_horizon).alias("outcome_horizon"))
                selected = merged
            else:
                lev = leaf_evidence_frame(
                    tree, frames.valid_observations, valid_horizon_outcome_frame
                )
                if lev.is_empty():
                    continue
                lctx = leaf_context_frame(
                    tree, frames.valid_observations, valid_horizon_outcome_frame
                )
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
                    frames.valid_observations,
                    valid_horizon_outcome_frame,
                    direction=direction,
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
            observations=frames.valid_observations,
            source_event_row_count=source_event_row_count,
            source_outcomes=_filter_outcome_horizon(frames.valid_source_outcomes, outcome_horizon),
            realized_transitions=_filter_outcome_horizon(
                frames.valid_realized_transitions, outcome_horizon
            ),
            outcome_frame=valid_horizon_outcome_frame,
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


def run_tailtree(
    frames: TailtreeInputFrames,
    *,
    config: PotentialConfig,
    profile: ProfileContext,
) -> TailtreeRunOutput:
    from qooi.scanner.outcome import potential_outcome_frame
    from qooi.scanner.tailrun import planning
    from qooi.scanner.tailrun.search import optuna_module, suggest_tailtree_trial_params
    from qooi.scanner.tailtree.model import label_tail_exceedances

    tailtree = config.evidence.tailtree
    with profile.stage("scanner", "tailtree", "potential_outcome_frame"):
        outcome_frame = potential_outcome_frame(
            frames.observations,
            frames.source_outcomes,
            frames.realized,
            return_threshold_pct=config.transition.return_threshold_pct,
        )
    profile.frame("scanner", "tailtree", "tailtree_outcomes", outcome_frame)

    with profile.stage("scanner", "tailtree", "label_tail_exceedances"):
        labeled = label_tail_exceedances(outcome_frame, threshold_pct=tailtree.threshold_pct)
    profile.frame("scanner", "tailtree", "labeled_tailtree_outcomes", labeled)

    categorical, continuous = _tailtree_training_features(frames.observations)
    prepared = TailtreePreparedFrames(
        observations=frames.observations,
        source_outcomes=frames.source_outcomes,
        realized=frames.realized,
        histories=frames.histories,
        outcomes=outcome_frame,
        labeled_outcomes=labeled,
        categorical_features=categorical,
        continuous_features=continuous,
    )
    tailtree.model_dir.mkdir(parents=True, exist_ok=True)

    all_evidence: list[pl.DataFrame] = []
    models: dict[tuple[int, TailtreeDirection], TailtreeArtifactTree] = {}
    feedback: list[TailtreeProfileFeedback] = []
    efficiency_frames: list[pl.DataFrame] = []

    for profile_config in config.evidence.tailtree.profiles:
        if isinstance(profile_config.training, TailtreeOptunaTrainingConfig):
            continue
        run = planning.tailtree_fixed_run(profile_config)
        for fold_id, fold_prepared in _profile_prepared_frames(
            prepared,
            profile_config,
            config=config,
            profile=profile,
            potential_outcome_frame=potential_outcome_frame,
            label_tail_exceedances=label_tail_exceedances,
        ):
            fold_run = _fold_run(run, fold_id)
            started = perf_counter()
            evidence, run_models, score = _train_profile_run(
                fold_run, fold_prepared, config=config, profile=profile
            )
            seconds = perf_counter() - started
            feedback.append(_profile_feedback(fold_run, score, evidence, run_models, seconds))
            efficiency_frames.append(
                _selection_efficiency_frame(fold_run, evidence, fold_prepared, run_models, seconds)
            )
            if not evidence.is_empty():
                all_evidence.append(evidence)
            models.update(run_models)

    if tailtree.lifecycle == "train":
        for profile_config in planning.tailtree_optuna_profiles(config):
            training_config = profile_config.training
            if not isinstance(training_config, TailtreeOptunaTrainingConfig):
                continue
            optuna = optuna_module()
            max_trials = max(1, int(training_config.max_trials))
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(
                    seed=training_config.seed,
                    n_startup_trials=min(10, max_trials),
                ),
            )
            fold_prepared = _profile_prepared_frames(
                prepared,
                profile_config,
                config=config,
                profile=profile,
                potential_outcome_frame=potential_outcome_frame,
                label_tail_exceedances=label_tail_exceedances,
            )
            for trial_index in range(max_trials):
                optuna_trial = study.ask()
                params = suggest_tailtree_trial_params(optuna_trial, training_config)
                run = planning.tailtree_trial_run(profile_config, params, trial_number=trial_index)
                trial_scores: list[float] = []
                for fold_id, fold_frame in fold_prepared:
                    fold_run = _fold_run(run, fold_id)
                    started = perf_counter()
                    with profile.stage("scanner", "tailtree", f"optuna_trial.{fold_run.run_id}"):
                        evidence, run_models, score = _train_profile_run(
                            fold_run, fold_frame, config=config, profile=profile
                        )
                    seconds = perf_counter() - started
                    trial_scores.append(score)
                    feedback.append(
                        _profile_feedback(fold_run, score, evidence, run_models, seconds)
                    )
                    efficiency_frames.append(
                        _selection_efficiency_frame(
                            fold_run, evidence, fold_frame, run_models, seconds
                        )
                    )
                    if not evidence.is_empty():
                        all_evidence.append(evidence)
                    models.update(run_models)
                trial_score = (
                    sum(trial_scores) / len(trial_scores) if trial_scores else -1_000_000_000.0
                )
                study.tell(optuna_trial, trial_score)

    evidence = pl.concat(all_evidence, how="diagonal_relaxed") if all_evidence else pl.DataFrame()
    efficiency = (
        pl.concat(efficiency_frames, how="diagonal_relaxed")
        if efficiency_frames
        else pl.DataFrame()
    )
    profile.frame("scanner", "tailtree", "tailtree_evidence", evidence)
    profile.frame("scanner", "tailtree", "tailtree_selection_efficiency", efficiency)
    return TailtreeRunOutput(
        evidence=evidence,
        models=models,
        profile_runs=tuple(feedback),
        selection_efficiency=efficiency,
    )


def _profile_prepared_frames(
    prepared: TailtreePreparedFrames,
    profile_config,
    *,
    config: PotentialConfig,
    profile: ProfileContext,
    potential_outcome_frame,
    label_tail_exceedances,
) -> tuple[tuple[int, TailtreePreparedFrames], ...]:
    from qooi.scanner.tailrun import planning

    evaluation = profile_config.evaluation
    if evaluation.protocol == "single_split":
        return ((0, prepared),)
    if config.bars is None:
        return ((0, prepared),)
    folds = planning.tailtree_fold_specs(
        planning.TailtreeWalkforwardSpec(
            train_days=evaluation.train_days,
            valid_days=evaluation.valid_days,
            step_days=evaluation.step_days,
            max_folds=evaluation.max_folds,
            embargo_bars=evaluation.embargo_bars,
        ),
        observations=prepared.observations,
        bar=config.bars.timeframes[0],
    )
    fold_frames: list[tuple[int, TailtreePreparedFrames]] = []
    for fold in folds:
        split = planning.tailtree_frame_split(
            prepared.observations,
            prepared.source_outcomes,
            prepared.realized,
            fold,
        )
        with profile.stage(
            "scanner", "tailtree", f"walkforward_train_outcomes.f{fold.fold_id:02d}"
        ):
            train_outcomes = potential_outcome_frame(
                split.train_observations,
                split.train_source_outcomes,
                split.train_realized_transitions,
                return_threshold_pct=config.transition.return_threshold_pct,
            )
            train_labeled = label_tail_exceedances(
                train_outcomes, threshold_pct=config.evidence.tailtree.threshold_pct
            )
        with profile.stage(
            "scanner", "tailtree", f"walkforward_valid_outcomes.f{fold.fold_id:02d}"
        ):
            valid_outcomes = potential_outcome_frame(
                split.valid_observations,
                split.valid_source_outcomes,
                split.valid_realized_transitions,
                return_threshold_pct=config.transition.return_threshold_pct,
            )
            valid_labeled = label_tail_exceedances(
                valid_outcomes, threshold_pct=config.evidence.tailtree.threshold_pct
            )
        fold_frames.append(
            (
                fold.fold_id,
                TailtreePreparedFrames(
                    observations=split.train_observations,
                    source_outcomes=split.train_source_outcomes,
                    realized=split.train_realized_transitions,
                    histories=prepared.histories,
                    outcomes=train_outcomes,
                    labeled_outcomes=train_labeled,
                    categorical_features=prepared.categorical_features,
                    continuous_features=prepared.continuous_features,
                    score_observations=split.valid_observations,
                    score_labeled_outcomes=valid_labeled,
                ),
            )
        )
    return tuple(fold_frames) or ((0, prepared),)


def _fold_run(run: TailtreeProfileRun, fold_id: int) -> TailtreeProfileRun:
    if fold_id == 0:
        return run
    suffix = f"-f{fold_id:02d}"
    return replace(run, run_id=f"{run.run_id}{suffix}", model_tag=f"{run.model_tag}{suffix}")


def _train_profile_run(
    run: TailtreeProfileRun,
    prepared: TailtreePreparedFrames,
    *,
    config: PotentialConfig,
    profile: ProfileContext,
) -> tuple[pl.DataFrame, dict[tuple[int, TailtreeDirection], TailtreeArtifactTree], float]:
    from qooi.scanner.tailtree.evidence import leaf_evidence_frame, score_bucket_evidence_frame
    from qooi.scanner.tailtree.model import TailTreeModel, TrainConfig, tailtree_training_frame

    tailtree = config.evidence.tailtree
    train_config = TrainConfig(
        objective=run.objective,
        num_leaves=run.training.num_leaves,
        min_data_in_leaf=run.training.min_data_in_leaf,
        learning_rate=run.training.learning_rate,
        num_iterations=run.training.num_iterations,
        early_stopping_rounds=run.training.early_stopping_rounds,
    )
    evidence_frames: list[pl.DataFrame] = []
    models: dict[tuple[int, TailtreeDirection], TailtreeArtifactTree] = {}
    score_observations = (
        prepared.score_observations
        if prepared.score_observations is not None
        else prepared.observations
    )
    score_labeled = (
        prepared.score_labeled_outcomes
        if prepared.score_labeled_outcomes is not None
        else prepared.labeled_outcomes
    )
    for outcome_horizon in tailtree.outcome_horizon:
        horizon_labeled = _filter_outcome_horizon(prepared.labeled_outcomes, int(outcome_horizon))
        horizon_score_labeled = _filter_outcome_horizon(score_labeled, int(outcome_horizon))
        for direction in ("up", "down"):
            label = f"{run.run_id}.h{int(outcome_horizon)}.{direction}"
            model_path = tailtree.model_dir / f"{run.model_tag}_{outcome_horizon}_{direction}.json"
            tree: TailtreeArtifactTree | None = None
            if tailtree.lifecycle == "load_predict" and model_path.exists():
                with profile.stage("scanner", "tailtree", f"load.{label}"):
                    tree = TailTreeModel.from_json(model_path)
            elif tailtree.lifecycle == "train":
                with profile.stage("scanner", "tailtree", f"training_frame.{label}"):
                    training = tailtree_training_frame(
                        prepared.observations, horizon_labeled, direction=direction
                    )
                profile.frame(
                    "scanner", "tailtree", f"training_{label}", training.tail_observations
                )
                if training.has_min_exceedances(train_config.min_data_in_leaf):

                    def train_tree() -> TailtreeArtifactTree:
                        return TailTreeModel.train(
                            training.tail_observations,
                            training.exceedance_values,
                            config=train_config,
                            categorical_features=prepared.categorical_features,
                            continuous_features=prepared.continuous_features,
                            direction=direction,
                            global_tail_rate=training.global_tail_rate,
                            train_n_observations=training.train_n_observations,
                            utility_values=training.utility_values,
                        )

                    with profile.stage("scanner", "tailtree", f"train.{label}"):
                        tree = profile.native(f"tailtree_train.{label}", train_tree)
                    tree.to_json(model_path)
            if tree is None:
                continue
            models[(int(outcome_horizon), direction)] = tree
            with profile.stage("scanner", "tailtree", f"evidence.{label}"):
                evidence = (
                    score_bucket_evidence_frame(tree, score_observations, horizon_score_labeled)
                    if run.objective == "tail_utility_quantile"
                    else leaf_evidence_frame(tree, score_observations, horizon_score_labeled)
                )
            if evidence.is_empty():
                continue
            evidence = evidence.with_columns(
                pl.lit(int(outcome_horizon)).alias("outcome_horizon"),
                pl.lit(run.run_id).alias("trial_id"),
                pl.lit(run.run_source).alias("trial_source"),
                pl.lit(run.model_tag).alias("model_tag"),
                pl.lit(run.objective).alias("objective"),
                pl.lit(run.profile_id).alias("training_profile"),
                pl.lit(run.training.num_leaves).alias("num_leaves"),
                pl.lit(run.training.min_data_in_leaf).alias("min_data_in_leaf"),
                pl.lit(run.training.learning_rate).alias("learning_rate"),
                pl.lit(run.training.num_iterations).alias("num_iterations"),
                pl.lit(run.training.early_stopping_rounds).alias("early_stopping_rounds"),
            )
            profile.frame("scanner", "tailtree", f"evidence_{label}", evidence)
            evidence_frames.append(evidence)
    evidence = (
        pl.concat(evidence_frames, how="diagonal_relaxed") if evidence_frames else pl.DataFrame()
    )
    return evidence, models, _tailtree_score(evidence)


def _tailtree_score(evidence: pl.DataFrame) -> float:
    if evidence.is_empty():
        return -1_000_000_000.0
    score_columns = [
        column
        for column in ("tail_utility_mean", "tail_lift", "N_tail_exceedances")
        if column in evidence.columns
    ]
    if not score_columns:
        return float(evidence.height)
    values = [
        float(evidence.get_column(column).cast(pl.Float64).fill_null(0.0).max() or 0.0)
        for column in score_columns
    ]
    return sum(values)


def _selection_efficiency_frame(
    run: TailtreeProfileRun,
    evidence: pl.DataFrame,
    prepared: TailtreePreparedFrames,
    models: dict[tuple[int, TailtreeDirection], TailtreeArtifactTree],
    seconds: float,
) -> pl.DataFrame:
    if evidence.is_empty():
        return pl.DataFrame()
    eligible_symbol_count = prepared.observations.get_column("symbol").n_unique()
    observation_row_count = prepared.observations.height
    feature_count = len(prepared.categorical_features) + len(prepared.continuous_features)
    rows: list[TailtreeSelectionEfficiencyRow] = []
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
        hpo_score = valid_tail_lift + utility_mean + (selected_tails + 1.0) ** 0.5 / 10.0
        rows.append(
            TailtreeSelectionEfficiencyRow(
                universe_snapshot_id="active",
                model_tag=run.model_tag,
                objective=run.objective,
                training_profile=run.profile_id,
                trial_id=run.run_id.rsplit("-t", 1)[0]
                if run.run_source == "optuna"
                else run.run_id,
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
                hpo_score=hpo_score,
                promotion_threshold_pass_int=int(
                    selected_observations >= 500 and valid_tail_lift >= 3.0
                ),
                trained_tree_count=len(models),
                selected_bucket_or_leaf_count=1,
                fit_seconds=seconds,
                score_seconds=0.0,
            )
        )
    return pl.DataFrame([row.__dict__ for row in rows])


def _profile_feedback(
    run: TailtreeProfileRun,
    score: float,
    evidence: pl.DataFrame,
    models: dict[tuple[int, TailtreeDirection], TailtreeArtifactTree],
    seconds: float,
) -> TailtreeProfileFeedback:
    return TailtreeProfileFeedback(
        run_id=run.run_id,
        trial_id=run.run_id.rsplit("-t", 1)[0] if run.run_source == "optuna" else run.run_id,
        trial_source=run.run_source,
        objective=run.objective,
        training_profile=run.profile_id,
        model_tag=run.model_tag,
        num_leaves=run.training.num_leaves,
        min_data_in_leaf=run.training.min_data_in_leaf,
        learning_rate=run.training.learning_rate,
        num_iterations=run.training.num_iterations,
        early_stopping_rounds=run.training.early_stopping_rounds,
        score=score,
        evidence_rows=evidence.height,
        model_count=len(models),
        seconds=seconds,
    )


__all__ = [
    "TailtreeEvidenceResult",
    "TailtreeResult",
    "load_predict",
    "run_frame_split",
    "run_tailtree",
]
