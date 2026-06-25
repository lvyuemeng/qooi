"""Tailtree train/load_predict lifecycle boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict

from qooi.profiling import ProfileContext
from qooi.scanner.config import (
    ExtremeTailConfig,
    PotentialConfig,
    TailtreeOptunaTrainingConfig,
)
from qooi.scanner.tailrun.types import (
    TailtreeArtifactTree,
    TailtreeCandidateGateSpec,
    TailtreeCandidateLocalModelRef,
    TailtreeDirection,
    TailtreeFeatureSelection,
    TailtreeFeatureSet,
    TailtreeInputFrames,
    TailtreeJobResult,
    TailtreeObjectiveJob,
    TailtreePreparedFrames,
    TailtreeProfileFeedback,
    TailtreeRunOutput,
)

if TYPE_CHECKING:
    from qooi.scanner.tailrun.planning import TailtreeProfileRun, TailtreeTrialParams

_TAILTREE_CATEGORICAL_TRAIN_FEATURES = (
    "background_regime",
    "swing_core",
    "decision_core",
    "decision_transition",
    "decision_direction",
)
_TAILTREE_BASE_TRAIN_FEATURES = (
    "atr_percentile",
    "range_width_atr",
    "bar_return_1h_pct",
    "bar_return_4h_pct",
    "bar_return_24h_pct",
    "bar_return_4h_per_vol_7d",
    "bar_return_24h_per_vol_7d",
    "bar_volume_1h_to_ma_20h",
    "bar_close_position_48h",
    "funding_rate_raw",
    "funding_rate_bps",
    "oi_change_raw",
    "oi_change_pct",
    "taker_buy_sell_ratio_raw",
    "taker_buy_pressure",
    "lsr_ratio_raw",
    "lsr_log_ratio",
    "funding_age_ms",
    "oi_age_ms",
    "taker_age_ms",
    "lsr_age_ms",
)

_TAILTREE_PROMOTER_EXTRA_FEATURES = (
    "return_sign_flip_rate_6h",
    "return_sign_flip_rate_24h",
    "body_to_range_mean_24h",
    "range_expansion_24h_to_7d",
    "close_position_24h",
    "prior_runup_6h",
    "prior_drawdown_6h",
    "return_efficiency_24h",
    "market_return_1h_median",
    "market_return_4h_median",
    "market_return_24h_median",
    "market_abs_return_24h_median",
    "market_dispersion_24h",
    "market_positive_return_24h_share",
    "symbol_vs_market_return_24h",
    "symbol_vs_market_return_4h",
    "symbol_abs_return_vs_market_24h",
)

_TAILTREE_PROMOTER_TRAIN_FEATURES = (
    *_TAILTREE_BASE_TRAIN_FEATURES,
    *_TAILTREE_PROMOTER_EXTRA_FEATURES,
)

_SOURCE_CONTEXT_CATEGORICAL_FEATURES = (
    "funding_level_state",
    "funding_level_transition",
    "funding_price_divergence_24h",
    "lsr_level_state",
    "lsr_level_transition",
    "lsr_price_divergence_24h",
    "oi_flow_state",
    "oi_flow_transition",
    "taker_pressure_state",
    "taker_pressure_transition",
)
_SOURCE_CONTEXT_CONTINUOUS_FEATURES = (
    "funding_direction_run_length",
    "lsr_direction_run_length",
    "lsr_log_ratio_change_24h",
    "oi_flow_run_length",
    "oi_change_pct_24h",
    "taker_pressure_run_length",
    "taker_buy_pressure_24h_mean",
)

TailtreeFeatureRole = Literal["opportunity", "candidate"]

_BASE_TAILTREE_FEATURE_SET = TailtreeFeatureSet(
    categorical=(*_TAILTREE_CATEGORICAL_TRAIN_FEATURES, *_SOURCE_CONTEXT_CATEGORICAL_FEATURES),
    continuous=(*_TAILTREE_BASE_TRAIN_FEATURES, *_SOURCE_CONTEXT_CONTINUOUS_FEATURES),
)
_PROMOTER_FEATURE_SET = TailtreeFeatureSet(
    categorical=(*_TAILTREE_CATEGORICAL_TRAIN_FEATURES, *_SOURCE_CONTEXT_CATEGORICAL_FEATURES),
    continuous=(*_TAILTREE_PROMOTER_TRAIN_FEATURES, *_SOURCE_CONTEXT_CONTINUOUS_FEATURES),
)
_CANDIDATE_PROMOTER_GATES = (
    TailtreeCandidateGateSpec("score_pct", 0.5),
    TailtreeCandidateGateSpec("score_pct", 1.0),
    TailtreeCandidateGateSpec("score_pct", 2.0),
    TailtreeCandidateGateSpec("top_k", 200.0),
    TailtreeCandidateGateSpec("top_k", 500.0),
    TailtreeCandidateGateSpec("top_k", 1000.0),
)


def train_features(
    observations: pl.DataFrame,
    *,
    role: TailtreeFeatureRole,
) -> TailtreeFeatureSelection:
    """Select known-at-close tailtree training features by training role."""
    return (
        _PROMOTER_FEATURE_SET if role == "candidate" else _BASE_TAILTREE_FEATURE_SET
    ).select(observations)


class LocalModelSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["promoter", "opposite_guard", "weak_path_guard"]
    label_column: str
    weight_column: str
    score_column: str
    objective: Literal["tail_event_lift", "path_guard"]


@dataclass(frozen=True)
class CandidateLocalProduct:
    efficiency: pl.DataFrame
    selection_error_anatomy: pl.DataFrame
    boundary_anatomy: pl.DataFrame
    contradiction_audit: pl.DataFrame


def _model_id_slug(value: object) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    return "_".join(part for part in text.split("_") if part)


def _local_model_ref(
    *,
    model_dir: str | Path,
    parent_model_id: str,
    role: Literal["promoter", "opposite_guard", "weak_path_guard"],
    gate_id: object,
) -> TailtreeCandidateLocalModelRef:
    gate_slug = _model_id_slug(gate_id)
    return TailtreeCandidateLocalModelRef(
        parent_model_id=parent_model_id,
        role=role,
        gate_id=str(gate_id),
        model_path=Path(model_dir) / f"{parent_model_id}_{role}_{gate_slug}.json",
    )


def _fit_local_model_scores(
    spec: LocalModelSpec,
    *,
    gate_id: object,
    train_targets: pl.DataFrame,
    score_features: pl.DataFrame,
    observations: pl.DataFrame,
    training: TailtreeTrialParams,
    join_keys: list[str],
    lifecycle: Literal["train", "load_predict"],
    model_dir: str | Path,
    parent_model_id: str,
) -> pl.DataFrame:
    from qooi.scanner.tailtree.model import TailTreeModel, TrainConfig

    model_ref = _local_model_ref(
        model_dir=model_dir,
        parent_model_id=parent_model_id,
        role=spec.role,
        gate_id=gate_id,
    )

    if lifecycle == "load_predict":
        if not model_ref.model_path.exists():
            raise FileNotFoundError(
                f"tailtree candidate-local model missing: {model_ref.model_path}"
            )
        model = TailTreeModel.from_json(model_ref.model_path)
        return model.predict_score(score_features).select(
            *join_keys, pl.col("tailtree_score").alias(spec.score_column)
        ).unique(subset=join_keys, maintain_order=True)

    gate_train = train_targets.filter(
        (pl.col("candidate_gate_id") == gate_id)
        & pl.col("in_candidate_gate")
        & pl.col(spec.label_column).is_not_null()
    )
    if gate_train.height < 30 or gate_train.get_column(spec.label_column).n_unique() < 2:
        return pl.DataFrame()
    train_frame = observations.join(
        gate_train.select(*join_keys, spec.label_column, spec.weight_column),
        on=join_keys,
        how="inner",
    ).sort("decision_bar_close_ms")
    if train_frame.height < 30:
        return pl.DataFrame()
    labels = train_frame.get_column(spec.label_column).to_numpy().astype(float)
    weights = train_frame.get_column(spec.weight_column).fill_null(1.0).to_numpy().astype(float)
    features = train_features(train_frame, role="candidate")
    min_leaf = min(training.min_data_in_leaf, max(10, train_frame.height // 5))
    try:
        model = TailTreeModel.train(
            train_frame,
            labels,
            config=TrainConfig(
                objective=spec.objective,
                num_leaves=min(training.num_leaves, 64),
                min_data_in_leaf=min_leaf,
                learning_rate=training.learning_rate,
                num_iterations=training.num_iterations,
                early_stopping_rounds=training.early_stopping_rounds,
            ),
            categorical_features=features.categorical_list(),
            continuous_features=features.continuous_list(),
            direction="up",
            global_tail_rate=float(np.mean(labels)),
            train_n_observations=train_frame.height,
            utility_values=weights,
        )
    except ValueError:
        return pl.DataFrame()
    model_ref.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.to_json(model_ref.model_path)
    return model.predict_score(score_features).select(
        *join_keys, pl.col("tailtree_score").alias(spec.score_column)
    ).unique(subset=join_keys, maintain_order=True)


@dataclass(frozen=True)
class TailtreeFoldRunResult:
    evidence: pl.DataFrame
    models: dict[tuple[int, TailtreeDirection], TailtreeArtifactTree]
    feedback: TailtreeProfileFeedback
    selection_efficiency: pl.DataFrame
    action_surface: pl.DataFrame
    selection_error_anatomy: pl.DataFrame
    boundary_anatomy: pl.DataFrame
    contradiction_audit: pl.DataFrame
    candidate_replay: pl.DataFrame
    candidate_population: pl.DataFrame
    score: float


def run_tailtree(
    frames: TailtreeInputFrames,
    *,
    config: PotentialConfig,
    profile: ProfileContext,
) -> TailtreeRunOutput:
    from qooi.scanner.outcome import potential_outcome_frame
    from qooi.scanner.tailrun import planning
    from qooi.scanner.tailrun.search import optuna_module, suggest_tailtree_trial_params
    from qooi.scanner.tailtree.labels import TailEventPolicy, tailtree_label_distribution_frame

    tailtree = config.evidence.tailtree
    tail_policy = TailEventPolicy(
        tailtree.extreme
        if tailtree.extreme is not None
        else ExtremeTailConfig(method="fixed_pct", material_floor_pct=tailtree.threshold_pct)
    )
    with profile.stage("scanner", "tailtree", "potential_outcome_frame"):
        outcome_frame = potential_outcome_frame(
            frames.observations,
            frames.source_outcomes,
            frames.realized,
            return_threshold_pct=config.transition.return_threshold_pct,
        )
    profile.frame("scanner", "tailtree", "tailtree_outcomes", outcome_frame)

    with profile.stage("scanner", "tailtree", "label_tail_paths"):
        labeled = tail_policy.label_paths(outcome_frame)
    profile.frame("scanner", "tailtree", "labeled_tailtree_outcomes", labeled)
    label_distribution = tailtree_label_distribution_frame(labeled)
    profile.frame("scanner", "tailtree", "tailtree_label_distribution", label_distribution)

    base_features = train_features(frames.observations, role="opportunity")
    prepared = TailtreePreparedFrames(
        observations=frames.observations,
        source_outcomes=frames.source_outcomes,
        realized=frames.realized,
        histories=frames.histories,
        outcomes=outcome_frame,
        labeled_outcomes=labeled,
        categorical_features=base_features.categorical_list(),
        continuous_features=base_features.continuous_list(),
    )
    tailtree.model_dir.mkdir(parents=True, exist_ok=True)

    all_evidence: list[pl.DataFrame] = []
    models: dict[tuple[int, TailtreeDirection], TailtreeArtifactTree] = {}
    feedback: list[TailtreeProfileFeedback] = []
    efficiency_frames: list[pl.DataFrame] = []
    action_frames: list[pl.DataFrame] = []
    anatomy_frames: list[pl.DataFrame] = []
    boundary_frames: list[pl.DataFrame] = []
    audit_frames: list[pl.DataFrame] = []

    if tailtree.lifecycle == "load_predict":
        result = run_tailtree_fold(
            planning.tailtree_predict_run(tailtree), 0, prepared, config=config, profile=profile
        )
        feedback.append(result.feedback)
        efficiency_frames.append(result.selection_efficiency)
        if not result.action_surface.is_empty():
            action_frames.append(result.action_surface)
        if not result.selection_error_anatomy.is_empty():
            anatomy_frames.append(result.selection_error_anatomy)
        if not result.boundary_anatomy.is_empty():
            boundary_frames.append(result.boundary_anatomy)
        if not result.contradiction_audit.is_empty():
            audit_frames.append(result.contradiction_audit)
        if not result.evidence.is_empty():
            all_evidence.append(result.evidence)
        models.update(result.models)

    if tailtree.lifecycle == "train":
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
                tail_policy=tail_policy,
            ):
                result = run_tailtree_fold(
                    run, fold_id, fold_prepared, config=config, profile=profile
                )
                feedback.append(result.feedback)
                efficiency_frames.append(result.selection_efficiency)
                if not result.action_surface.is_empty():
                    action_frames.append(result.action_surface)
                if not result.selection_error_anatomy.is_empty():
                    anatomy_frames.append(result.selection_error_anatomy)
                if not result.boundary_anatomy.is_empty():
                    boundary_frames.append(result.boundary_anatomy)
                if not result.contradiction_audit.is_empty():
                    audit_frames.append(result.contradiction_audit)
                if not result.evidence.is_empty():
                    all_evidence.append(result.evidence)
                models.update(result.models)

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
                tail_policy=tail_policy,
            )
            for trial_index in range(max_trials):
                optuna_trial = study.ask()
                params = suggest_tailtree_trial_params(optuna_trial, training_config)
                run = planning.tailtree_trial_run(profile_config, params, trial_number=trial_index)
                trial_scores: list[float] = []
                for fold_id, fold_frame in fold_prepared:
                    fold_run = run.for_fold(fold_id)
                    with profile.stage("scanner", "tailtree", f"optuna_trial.{fold_run.run_id}"):
                        result = run_tailtree_fold(
                            run, fold_id, fold_frame, config=config, profile=profile
                        )
                    trial_scores.append(result.score)
                    feedback.append(result.feedback)
                    efficiency_frames.append(result.selection_efficiency)
                    if not result.action_surface.is_empty():
                        action_frames.append(result.action_surface)
                    if not result.selection_error_anatomy.is_empty():
                        anatomy_frames.append(result.selection_error_anatomy)
                    if not result.boundary_anatomy.is_empty():
                        boundary_frames.append(result.boundary_anatomy)
                    if not result.contradiction_audit.is_empty():
                        audit_frames.append(result.contradiction_audit)
                    if not result.evidence.is_empty():
                        all_evidence.append(result.evidence)
                    models.update(result.models)
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
    action_surface = (
        pl.concat(action_frames, how="diagonal_relaxed") if action_frames else pl.DataFrame()
    )
    error_anatomy = (
        pl.concat(anatomy_frames, how="diagonal_relaxed") if anatomy_frames else pl.DataFrame()
    )
    boundary_anatomy = (
        pl.concat(boundary_frames, how="diagonal_relaxed") if boundary_frames else pl.DataFrame()
    )
    contradiction_audit = (
        pl.concat(audit_frames, how="diagonal_relaxed") if audit_frames else pl.DataFrame()
    )

    profile.frame("scanner", "tailtree", "tailtree_evidence", evidence)
    profile.frame("scanner", "tailtree", "tailtree_selection_efficiency", efficiency)
    if not action_surface.is_empty():
        profile.frame("scanner", "tailtree", "tailtree_action_surface", action_surface)
    if not error_anatomy.is_empty():
        profile.frame("scanner", "tailtree", "tailtree_selection_error_anatomy", error_anatomy)
    if not boundary_anatomy.is_empty():
        profile.frame("scanner", "tailtree", "tailtree_boundary_anatomy", boundary_anatomy)
    if not contradiction_audit.is_empty():
        profile.frame("scanner", "tailtree", "tailtree_contradiction_audit", contradiction_audit)
    return TailtreeRunOutput(
        evidence=evidence,
        models=models,
        profile_runs=tuple(feedback),
        selection_efficiency=efficiency,
        label_distribution=label_distribution,
        action_surface=action_surface,
        selection_error_anatomy=error_anatomy,
        boundary_anatomy=boundary_anatomy,
        contradiction_audit=contradiction_audit,
    )


def _profile_prepared_frames(
    prepared: TailtreePreparedFrames,
    profile_config,
    *,
    config: PotentialConfig,
    profile: ProfileContext,
    potential_outcome_frame,
    tail_policy,
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
            tail_reference = tail_policy.reference_frame(train_outcomes)
            train_labeled = tail_policy.label_paths(train_outcomes, reference=tail_reference)
        with profile.stage(
            "scanner", "tailtree", f"walkforward_valid_outcomes.f{fold.fold_id:02d}"
        ):
            valid_outcomes = potential_outcome_frame(
                split.valid_observations,
                split.valid_source_outcomes,
                split.valid_realized_transitions,
                return_threshold_pct=config.transition.return_threshold_pct,
            )
            valid_labeled = tail_policy.label_paths(valid_outcomes, reference=tail_reference)
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


def run_tailtree_job(
    job: TailtreeObjectiveJob,
    prepared: TailtreePreparedFrames,
    *,
    config: PotentialConfig,
    profile: ProfileContext,
) -> TailtreeJobResult:
    from qooi.scanner.tailrun.selection import (
        score_bucket_candidate_frame,
        score_bucket_population_frame,
    )
    from qooi.scanner.tailtree.evidence import leaf_evidence_frame, score_bucket_evidence_frame
    from qooi.scanner.tailtree.labels import TailEventPolicy
    from qooi.scanner.tailtree.model import (
        TailTreeModel,
        TrainConfig,
        tailtree_target_training_values,
        tailtree_training_frame,
    )

    run = job.run
    tailtree = config.evidence.tailtree
    train_config = TrainConfig(
        objective=run.objective,
        num_leaves=run.training.num_leaves,
        min_data_in_leaf=run.training.min_data_in_leaf,
        learning_rate=run.training.learning_rate,
        num_iterations=run.training.num_iterations,
        early_stopping_rounds=run.training.early_stopping_rounds,
    )
    horizon_labeled = (
        prepared.labeled_outcomes.filter(pl.col("outcome_horizon") == job.outcome_horizon)
        if "outcome_horizon" in prepared.labeled_outcomes.columns
        else prepared.labeled_outcomes
    )
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
    horizon_score_labeled = (
        score_labeled.filter(pl.col("outcome_horizon") == job.outcome_horizon)
        if "outcome_horizon" in score_labeled.columns
        else score_labeled
    )
    tail_policy = TailEventPolicy(
        tailtree.extreme
        if tailtree.extreme is not None
        else ExtremeTailConfig(method="fixed_pct", material_floor_pct=tailtree.threshold_pct)
    )
    training_behavior_targets = (
        tail_policy.behavior_target_frame(horizon_labeled, direction=job.direction)
        if job.direction == "up"
        else None
    )
    score_behavior_targets = (
        tail_policy.behavior_target_frame(horizon_score_labeled, direction=job.direction)
        if job.direction == "up"
        else None
    )
    tree: TailTreeModel | None = None

    if tailtree.lifecycle == "load_predict" and job.model_path.exists():
        with profile.stage("scanner", "tailtree", f"load.{job.label}"):
            tree = TailTreeModel.from_json(job.model_path)
    elif tailtree.lifecycle == "train":
        with profile.stage("scanner", "tailtree", f"training_frame.{job.label}"):
            training = tailtree_training_frame(
                prepared.observations, horizon_labeled, direction=job.direction
            )
        profile.frame("scanner", "tailtree", f"training_{job.label}", training.tail_observations)
        if training.has_min_exceedances(train_config.min_data_in_leaf):

            def train_tree() -> TailTreeModel:
                train_features = training.tail_observations
                train_values = training.exceedance_values
                train_utilities = training.utility_values
                model_global_tail_rate = training.global_tail_rate
                model_train_n_observations = training.train_n_observations
                if run.objective in {
                    "tail_event_lift",
                    "tail_any_event",
                    "tail_side_only",
                    "path_guard",
                    "path_guard_blocker",
                    "path_guard_tradability",
                    "path_guard_full",
                }:
                    train_features, train_values, train_utilities = tailtree_target_training_values(
                        prepared.observations,
                        horizon_labeled,
                        target=run.objective,
                        direction=job.direction,
                        behavior_targets=training_behavior_targets,
                    )
                    model_global_tail_rate = None
                    model_train_n_observations = len(train_features)
                return TailTreeModel.train(
                    train_features,
                    train_values,
                    config=train_config,
                    categorical_features=prepared.categorical_features,
                    continuous_features=prepared.continuous_features,
                    direction=job.direction,
                    global_tail_rate=model_global_tail_rate,
                    train_n_observations=model_train_n_observations,
                    utility_values=train_utilities,
                )

            with profile.stage("scanner", "tailtree", f"train.{job.label}"):
                tree = profile.native(f"tailtree_train.{job.label}", train_tree)
            tree.to_json(job.model_path)

    if tree is None:
        return TailtreeJobResult(job, pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), None)

    with profile.stage("scanner", "tailtree", f"score.{job.label}"):
        scored = score_bucket_candidate_frame(
            tree,
            score_observations,
            horizon_score_labeled,
            job.outcome_horizon,
            behavior_targets=score_behavior_targets,
            objective=run.objective,
        )
        scored_population = score_bucket_population_frame(
            tree,
            score_observations,
            horizon_score_labeled,
            job.outcome_horizon,
            behavior_targets=score_behavior_targets,
            objective=run.objective,
        )
    if not scored.is_empty():
        profile.frame("scanner", "tailtree", f"scores_{job.label}", scored)

    with profile.stage("scanner", "tailtree", f"evidence.{job.label}"):
        evidence = (
            score_bucket_evidence_frame(tree, score_observations, horizon_score_labeled)
            if run.objective
            in {
                "tail_utility_quantile",
                "tail_event_lift",
                "tail_any_event",
                "tail_side_only",
                "path_guard",
                "path_guard_blocker",
                "path_guard_tradability",
                "path_guard_full",
            }
            else leaf_evidence_frame(tree, score_observations, horizon_score_labeled)
        )
    if evidence.is_empty():
        return TailtreeJobResult(job, pl.DataFrame(), scored, scored_population, tree)

    evidence = evidence.with_columns(
        pl.lit(job.outcome_horizon).alias("outcome_horizon"),
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
    profile.frame("scanner", "tailtree", f"evidence_{job.label}", evidence)
    return TailtreeJobResult(job, evidence, scored, scored_population, tree)


def candidate_conditional_promoter_efficiency_frame(
    run: TailtreeProfileRun,
    opportunity_tree: TailtreeArtifactTree,
    prepared: TailtreePreparedFrames,
    *,
    config: PotentialConfig,
) -> CandidateLocalProduct:
    """Train candidate-gated promoter/guard products from a tail_event_lift tree."""
    empty = CandidateLocalProduct(
        efficiency=pl.DataFrame(),
        selection_error_anatomy=pl.DataFrame(),
        boundary_anatomy=pl.DataFrame(),
        contradiction_audit=pl.DataFrame(),
    )
    if run.objective != "tail_event_lift" or 24 not in config.evidence.tailtree.outcome_horizon:
        return empty

    from qooi.scanner.tailrun.selection import (
        actionability_contradiction_audit_frame,
        candidate_gate_frame,
        dual_guard_boundary_anatomy_frame,
        dual_guarded_promotion_selection_metrics_frame,
        guarded_selection_error_anatomy_frame,
        opposite_guard_target_frame,
        promoter_target_frame,
        score_bucket_population_frame,
        selection_error_anatomy_frame,
        weak_path_guard_target_frame,
    )
    from qooi.scanner.tailtree.labels import TailEventPolicy

    train_labeled = prepared.labeled_outcomes.filter(pl.col("outcome_horizon") == 24)
    score_labeled_source = (
        prepared.score_labeled_outcomes
        if prepared.score_labeled_outcomes is not None
        else prepared.labeled_outcomes
    )
    score_labeled = score_labeled_source.filter(pl.col("outcome_horizon") == 24)
    score_observations = (
        prepared.score_observations
        if prepared.score_observations is not None
        else prepared.observations
    )
    tail_policy = TailEventPolicy(
        config.evidence.tailtree.extreme
        if config.evidence.tailtree.extreme is not None
        else ExtremeTailConfig(
            method="fixed_pct", material_floor_pct=config.evidence.tailtree.threshold_pct
        )
    )
    train_behavior = tail_policy.behavior_target_frame(train_labeled, direction="up")
    score_behavior = tail_policy.behavior_target_frame(score_labeled, direction="up")
    train_population = score_bucket_population_frame(
        opportunity_tree,
        prepared.observations,
        train_labeled,
        24,
        behavior_targets=train_behavior,
        objective="tail_event_lift",
        score_quantiles=(0.0,),
    )
    score_population = score_bucket_population_frame(
        opportunity_tree,
        score_observations,
        score_labeled,
        24,
        behavior_targets=score_behavior,
        objective="tail_event_lift",
        score_quantiles=(0.0,),
    )
    train_targets = weak_path_guard_target_frame(
        opposite_guard_target_frame(
            promoter_target_frame(candidate_gate_frame(train_population, _CANDIDATE_PROMOTER_GATES))
        )
    )
    score_targets = weak_path_guard_target_frame(
        opposite_guard_target_frame(
            promoter_target_frame(candidate_gate_frame(score_population, _CANDIDATE_PROMOTER_GATES))
        )
    )
    if score_targets.is_empty() or (
        config.evidence.tailtree.lifecycle == "train" and train_targets.is_empty()
    ):
        return empty

    rows: list[pl.DataFrame] = []
    guarded_rows: list[pl.DataFrame] = []
    dual_guarded_rows: list[pl.DataFrame] = []
    join_keys = ["symbol", "decision_bar_close_ms"]
    parent_model_id = f"{run.model_tag}_24_up"
    for gate_id in score_targets.get_column("candidate_gate_id").unique().to_list():
        gate_score = score_targets.filter(
            (pl.col("candidate_gate_id") == gate_id) & pl.col("in_candidate_gate")
        )
        if gate_score.is_empty():
            continue
        gate_score = gate_score.unique(subset=join_keys, maintain_order=True)
        score_features = score_observations.join(
            gate_score.select(*join_keys).unique(subset=join_keys, maintain_order=True),
            on=join_keys,
            how="inner",
        )
        promotion_scores = _fit_local_model_scores(
            LocalModelSpec(
                role="promoter",
                label_column="promoter_label",
                weight_column="promoter_weight",
                score_column="promotion_score",
                objective="tail_event_lift",
            ),
            gate_id=gate_id,
            train_targets=train_targets,
            score_features=score_features,
            observations=prepared.observations,
            training=run.training,
            join_keys=join_keys,
            lifecycle=config.evidence.tailtree.lifecycle,
            model_dir=config.evidence.tailtree.model_dir,
            parent_model_id=parent_model_id,
        )
        if promotion_scores.is_empty():
            continue
        scored_gate = gate_score.join(promotion_scores, on=join_keys, how="inner")
        rows.append(scored_gate)

        guard_scores = _fit_local_model_scores(
            LocalModelSpec(
                role="opposite_guard",
                label_column="opposite_guard_label",
                weight_column="opposite_guard_weight",
                score_column="opposite_guard_score",
                objective="path_guard",
            ),
            gate_id=gate_id,
            train_targets=train_targets,
            score_features=score_features,
            observations=prepared.observations,
            training=run.training,
            join_keys=join_keys,
            lifecycle=config.evidence.tailtree.lifecycle,
            model_dir=config.evidence.tailtree.model_dir,
            parent_model_id=parent_model_id,
        )
        if guard_scores.is_empty():
            continue
        guarded_gate = scored_gate.join(guard_scores, on=join_keys, how="inner")
        guarded_rows.append(guarded_gate)

        weak_scores = _fit_local_model_scores(
            LocalModelSpec(
                role="weak_path_guard",
                label_column="weak_path_guard_label",
                weight_column="weak_path_guard_weight",
                score_column="weak_path_guard_score",
                objective="path_guard",
            ),
            gate_id=gate_id,
            train_targets=train_targets,
            score_features=score_features,
            observations=prepared.observations,
            training=run.training,
            join_keys=join_keys,
            lifecycle=config.evidence.tailtree.lifecycle,
            model_dir=config.evidence.tailtree.model_dir,
            parent_model_id=parent_model_id,
        )
        if weak_scores.is_empty():
            continue
        dual_guarded_rows.append(guarded_gate.join(weak_scores, on=join_keys, how="inner"))
    if not rows:
        return empty
    promoter_selection = pl.concat(rows, how="diagonal_relaxed")
    if guarded_rows:
        guarded_selection = pl.concat(guarded_rows, how="diagonal_relaxed")
        anatomy_frames = [
            selection_error_anatomy_frame(promoter_selection),
            guarded_selection_error_anatomy_frame(guarded_selection),
        ]
    else:
        anatomy_frames = [selection_error_anatomy_frame(promoter_selection)]
    if dual_guarded_rows:
        dual_guarded_selection = pl.concat(dual_guarded_rows, how="diagonal_relaxed")
        efficiency = dual_guarded_promotion_selection_metrics_frame(dual_guarded_selection)
        boundary_anatomy = dual_guard_boundary_anatomy_frame(dual_guarded_selection)
        contradiction_audit = actionability_contradiction_audit_frame(dual_guarded_selection)
    else:
        efficiency = pl.DataFrame()
        boundary_anatomy = pl.DataFrame()
        contradiction_audit = pl.DataFrame()
    anatomy = pl.concat(
        [frame for frame in anatomy_frames if not frame.is_empty()], how="diagonal_relaxed"
    )
    return CandidateLocalProduct(
        efficiency=efficiency,
        selection_error_anatomy=anatomy,
        boundary_anatomy=boundary_anatomy,
        contradiction_audit=contradiction_audit,
    )


def run_tailtree_fold(
    run: TailtreeProfileRun,
    fold_id: int,
    prepared: TailtreePreparedFrames,
    *,
    config: PotentialConfig,
    profile: ProfileContext,
) -> TailtreeFoldRunResult:
    from qooi.scanner.tailrun import planning
    from qooi.scanner.tailrun.selection import (
        calibrated_candidate_replay_frame,
        paired_candidate_replay_frame,
        tailtree_action_surface_frame,
        tailtree_selection_metrics_frame,
    )

    fold_run = run.for_fold(fold_id)
    started = perf_counter()
    results = [
        run_tailtree_job(job, prepared, config=config, profile=profile)
        for job in planning.tailtree_objective_jobs(
            fold_run, fold_id=fold_id, tailtree=config.evidence.tailtree
        )
    ]
    evidence_frames = [result.evidence for result in results if not result.evidence.is_empty()]
    score_frames = [
        result.scored_candidates
        for result in results
        if not result.scored_candidates.is_empty()
    ]
    population_frames = [
        result.scored_population
        for result in results
        if not result.scored_population.is_empty()
    ]
    evidence = (
        pl.concat(evidence_frames, how="diagonal_relaxed")
        if evidence_frames
        else pl.DataFrame()
    )
    scored = pl.concat(score_frames, how="diagonal_relaxed") if score_frames else pl.DataFrame()
    population = (
        pl.concat(population_frames, how="diagonal_relaxed")
        if population_frames
        else pl.DataFrame()
    )
    models = {
        (result.job.outcome_horizon, result.job.direction): result.model
        for result in results
        if result.model is not None
    }
    candidate_replay = calibrated_candidate_replay_frame(paired_candidate_replay_frame(scored))
    if not candidate_replay.is_empty():
        profile.frame(
            "scanner",
            "tailtree",
            f"candidate_replay_{fold_run.run_id}",
            candidate_replay,
        )
    action_surface = tailtree_action_surface_frame(candidate_replay)
    if not action_surface.is_empty():
        action_surface = action_surface.with_columns(
            pl.lit(fold_run.run_id).alias("run_id"),
            pl.lit(fold_run.trial_id).alias("trial_id"),
            pl.lit(fold_run.model_tag).alias("model_tag"),
            pl.lit(fold_id).alias("fold_id"),
        )
        profile.frame(
            "scanner",
            "tailtree",
            f"action_surface_{fold_run.run_id}",
            action_surface,
        )
    seconds = perf_counter() - started
    selection_efficiency = tailtree_selection_metrics_frame(
        fold_run, evidence, prepared, models, seconds, candidate_replay, population
    )
    selection_anatomy = pl.DataFrame()
    boundary_anatomy = pl.DataFrame()
    contradiction_audit = pl.DataFrame()
    opportunity_tree = models.get((24, "up"))
    if opportunity_tree is not None:
        candidate_local = candidate_conditional_promoter_efficiency_frame(
            fold_run, opportunity_tree, prepared, config=config
        )
        selection_anatomy = candidate_local.selection_error_anatomy
        boundary_anatomy = candidate_local.boundary_anatomy
        contradiction_audit = candidate_local.contradiction_audit
        if not candidate_local.efficiency.is_empty():
            selection_efficiency = pl.concat(
                [selection_efficiency, candidate_local.efficiency], how="diagonal_relaxed"
            )
    selection_efficiency = selection_efficiency.with_columns(pl.lit("base").alias("feature_set"))

    if (
        not selection_efficiency.is_empty()
        and "behavior_hpo_score" in selection_efficiency.columns
    ):
        score_value = selection_efficiency.select(
            pl.col("behavior_hpo_score").cast(pl.Float64).max()
        ).item()
        score = float(score_value) if score_value is not None else -1_000_000_000.0
    elif evidence.is_empty():
        score = -1_000_000_000.0
    else:
        score_columns = [
            column
            for column in ("tail_utility_mean", "tail_lift", "N_tail_exceedances")
            if column in evidence.columns
        ]
        score = (
            sum(
                float(
                    evidence.select(pl.col(column).cast(pl.Float64).fill_null(0.0).max()).item()
                    or 0.0
                )
                for column in score_columns
            )
            if score_columns
            else float(evidence.height)
        )
    feedback = TailtreeProfileFeedback(
        run_id=fold_run.run_id,
        trial_id=fold_run.trial_id,
        trial_source=fold_run.run_source,
        objective=fold_run.objective,
        training_profile=fold_run.profile_id,
        model_tag=fold_run.model_tag,
        num_leaves=fold_run.training.num_leaves,
        min_data_in_leaf=fold_run.training.min_data_in_leaf,
        learning_rate=fold_run.training.learning_rate,
        num_iterations=fold_run.training.num_iterations,
        early_stopping_rounds=fold_run.training.early_stopping_rounds,
        score=score,
        evidence_rows=evidence.height,
        model_count=len(models),
        seconds=seconds,
    )
    return TailtreeFoldRunResult(
        evidence=evidence,
        models=models,
        feedback=feedback,
        selection_efficiency=selection_efficiency,
        action_surface=action_surface,
        selection_error_anatomy=selection_anatomy,
        boundary_anatomy=boundary_anatomy,
        contradiction_audit=contradiction_audit,
        candidate_replay=candidate_replay,
        candidate_population=population,
        score=score,
    )


__all__ = ["run_tailtree"]
