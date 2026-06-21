"""Tailtree train/load_predict lifecycle boundary."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING

import polars as pl

from qooi.profiling import ProfileContext
from qooi.scanner.config import PotentialConfig, TailtreeOptunaTrainingConfig
from qooi.scanner.tailrun.types import (
    TailtreeArtifactTree,
    TailtreeDirection,
    TailtreeInputFrames,
    TailtreeJobResult,
    TailtreeObjectiveJob,
    TailtreePreparedFrames,
    TailtreeProfileFeedback,
    TailtreeRunOutput,
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


def _tailtree_training_features(observations: pl.DataFrame) -> tuple[list[str], list[str]]:
    """Select persistent known-at-close features allowed for tailtree training.

    Ephemeral current-review/cost features may exist in the observation frame, but
    column presence alone does not make them historical model inputs.
    """
    categorical = [c for c in _TAILTREE_CATEGORICAL_TRAIN_FEATURES if c in observations.columns]
    continuous = [c for c in _TAILTREE_CONTINUOUS_TRAIN_FEATURES if c in observations.columns]
    return categorical, continuous


@dataclass(frozen=True)
class TailtreeFoldRunResult:
    evidence: pl.DataFrame
    models: dict[tuple[int, TailtreeDirection], TailtreeArtifactTree]
    feedback: TailtreeProfileFeedback
    selection_efficiency: pl.DataFrame
    action_surface: pl.DataFrame
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
    from qooi.scanner.tailtree.model import (
        label_tail_paths,
        tailtree_label_distribution_frame,
    )

    tailtree = config.evidence.tailtree
    with profile.stage("scanner", "tailtree", "potential_outcome_frame"):
        outcome_frame = potential_outcome_frame(
            frames.observations,
            frames.source_outcomes,
            frames.realized,
            return_threshold_pct=config.transition.return_threshold_pct,
        )
    profile.frame("scanner", "tailtree", "tailtree_outcomes", outcome_frame)

    with profile.stage("scanner", "tailtree", "label_tail_paths"):
        labeled = label_tail_paths(outcome_frame, threshold_pct=tailtree.threshold_pct)
    profile.frame("scanner", "tailtree", "labeled_tailtree_outcomes", labeled)
    label_distribution = tailtree_label_distribution_frame(labeled)
    profile.frame("scanner", "tailtree", "tailtree_label_distribution", label_distribution)

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
    action_frames: list[pl.DataFrame] = []

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
            label_tail_paths=label_tail_paths,
        ):
            result = run_tailtree_fold(
                run, fold_id, fold_prepared, config=config, profile=profile
            )
            feedback.append(result.feedback)
            efficiency_frames.append(result.selection_efficiency)
            if not result.action_surface.is_empty():
                action_frames.append(result.action_surface)
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
                label_tail_paths=label_tail_paths,
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
    profile.frame("scanner", "tailtree", "tailtree_evidence", evidence)
    profile.frame("scanner", "tailtree", "tailtree_selection_efficiency", efficiency)
    if not action_surface.is_empty():
        profile.frame("scanner", "tailtree", "tailtree_action_surface", action_surface)
    return TailtreeRunOutput(
        evidence=evidence,
        models=models,
        profile_runs=tuple(feedback),
        selection_efficiency=efficiency,
        label_distribution=label_distribution,
        action_surface=action_surface,
    )


def _profile_prepared_frames(
    prepared: TailtreePreparedFrames,
    profile_config,
    *,
    config: PotentialConfig,
    profile: ProfileContext,
    potential_outcome_frame,
    label_tail_paths,
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
            train_labeled = label_tail_paths(
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
            valid_labeled = label_tail_paths(
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


def run_tailtree_job(
    job: TailtreeObjectiveJob,
    prepared: TailtreePreparedFrames,
    *,
    config: PotentialConfig,
    profile: ProfileContext,
) -> TailtreeJobResult:
    from qooi.scanner.tailrun.selection import score_bucket_candidate_frame
    from qooi.scanner.tailtree.evidence import leaf_evidence_frame, score_bucket_evidence_frame
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
                if run.objective == "tail_event_lift":
                    train_features, train_values, train_utilities = tailtree_target_training_values(
                        prepared.observations,
                        horizon_labeled,
                        target="tail_event_lift",
                        direction=job.direction,
                    )
                elif run.objective == "tail_any_event":
                    train_features, train_values, train_utilities = tailtree_target_training_values(
                        prepared.observations,
                        horizon_labeled,
                        target="tail_any_event",
                        direction=job.direction,
                    )
                elif run.objective == "tail_side_only":
                    train_features, train_values, train_utilities = tailtree_target_training_values(
                        prepared.observations,
                        horizon_labeled,
                        target="tail_side_only",
                        direction=job.direction,
                    )
                return TailTreeModel.train(
                    train_features,
                    train_values,
                    config=train_config,
                    categorical_features=prepared.categorical_features,
                    continuous_features=prepared.continuous_features,
                    direction=job.direction,
                    global_tail_rate=training.global_tail_rate,
                    train_n_observations=training.train_n_observations,
                    utility_values=train_utilities,
                )

            with profile.stage("scanner", "tailtree", f"train.{job.label}"):
                tree = profile.native(f"tailtree_train.{job.label}", train_tree)
            tree.to_json(job.model_path)

    if tree is None:
        return TailtreeJobResult(job, pl.DataFrame(), pl.DataFrame(), None)

    with profile.stage("scanner", "tailtree", f"score.{job.label}"):
        scored = score_bucket_candidate_frame(
            tree, score_observations, horizon_score_labeled, job.outcome_horizon
        )
    if not scored.is_empty():
        profile.frame("scanner", "tailtree", f"scores_{job.label}", scored)

    with profile.stage("scanner", "tailtree", f"evidence.{job.label}"):
        evidence = (
            score_bucket_evidence_frame(tree, score_observations, horizon_score_labeled)
            if run.objective
            in {"tail_utility_quantile", "tail_event_lift", "tail_any_event", "tail_side_only"}
            else leaf_evidence_frame(tree, score_observations, horizon_score_labeled)
        )
    if evidence.is_empty():
        return TailtreeJobResult(job, pl.DataFrame(), scored, tree)

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
    return TailtreeJobResult(job, evidence, scored, tree)


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
    evidence = (
        pl.concat(evidence_frames, how="diagonal_relaxed")
        if evidence_frames
        else pl.DataFrame()
    )
    scored = pl.concat(score_frames, how="diagonal_relaxed") if score_frames else pl.DataFrame()
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
        fold_run, evidence, prepared, models, seconds, candidate_replay
    )
    if (
        not selection_efficiency.is_empty()
        and "objective_hpo_score" in selection_efficiency.columns
    ):
        score_value = selection_efficiency.select(
            pl.col("objective_hpo_score").cast(pl.Float64).max()
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
        score=score,
    )


__all__ = ["run_tailtree"]
