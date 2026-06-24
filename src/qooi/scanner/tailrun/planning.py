"""Tailtree runtime planning: profile runs, folds, frame splits."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Literal

import polars as pl

from qooi.scanner.config import (
    PotentialConfig,
    TailtreeConfig,
    TailtreeFixedTrainingConfig,
    TailtreeObjective,
    TailtreeOptunaTrainingConfig,
    TailtreeProfileConfig,
)
from qooi.scanner.tailrun.types import (
    TailtreeDirection,
    TailtreeFrameSplit,
    TailtreeObjectiveJob,
    TailtreeSingleSplitFold,
    TailtreeTimeWindow,
    TailtreeWalkforwardFold,
)

_MS_PER_DAY = 24 * 60 * 60 * 1000


@dataclass(frozen=True)
class TailtreeProfileRun:
    profile_id: str
    run_id: str
    run_source: Literal["fixed", "optuna", "loaded"]
    model_tag: str
    objective: TailtreeObjective
    training: TailtreeTrialParams

    @property
    def trial_id(self) -> str:
        return self.run_id.rsplit("-t", 1)[0] if self.run_source == "optuna" else self.run_id

    def for_fold(self, fold_id: int) -> TailtreeProfileRun:
        if fold_id == 0:
            return self
        suffix = f"-f{fold_id:02d}"
        return replace(self, run_id=f"{self.run_id}{suffix}", model_tag=f"{self.model_tag}{suffix}")


@dataclass(frozen=True)
class TailtreeTrialParams:
    num_leaves: int
    min_data_in_leaf: int
    learning_rate: float
    num_iterations: int
    early_stopping_rounds: int


@dataclass(frozen=True)
class TailtreeSingleSplitSpec:
    protocol: Literal["single_split"] = "single_split"
    embargo_bars: int = 0


@dataclass(frozen=True)
class TailtreeWalkforwardSpec:
    train_days: int
    valid_days: int
    step_days: int
    max_folds: int
    embargo_bars: int = 0
    protocol: Literal["walkforward"] = "walkforward"


def _bar_ms(bar: str) -> int:
    match = re.fullmatch(r"(\d+)([mhdMHD])", bar.strip())
    if not match:
        raise ValueError(f"unsupported bar duration for walkforward: {bar!r}")
    value = int(match.group(1))
    unit = match.group(2).lower()
    if unit == "m":
        return value * 60 * 1000
    if unit == "h":
        return value * 60 * 60 * 1000
    return value * _MS_PER_DAY


def _time_filter(frame: pl.DataFrame, window: TailtreeTimeWindow) -> pl.DataFrame:
    if frame.is_empty() or "decision_bar_close_ms" not in frame.columns:
        return frame
    return frame.filter(
        (pl.col("decision_bar_close_ms") >= window.start_ms)
        & (pl.col("decision_bar_close_ms") < window.end_ms)
    )


def tailtree_trial_run(
    profile: TailtreeProfileConfig,
    params: TailtreeTrialParams,
    *,
    trial_number: int,
) -> TailtreeProfileRun:
    run_id = f"{profile.profile_id}-t{trial_number:04d}"
    return TailtreeProfileRun(
        profile_id=profile.profile_id,
        run_id=run_id,
        run_source="optuna",
        model_tag=f"{profile.model_tag}-t{trial_number:04d}",
        objective=profile.objective,
        training=params,
    )


def tailtree_fixed_run(profile: TailtreeProfileConfig) -> TailtreeProfileRun:
    training = profile.training
    if not isinstance(training, TailtreeFixedTrainingConfig):
        raise ValueError("tailtree_fixed_run requires fixed training")
    return TailtreeProfileRun(
        profile_id=profile.profile_id,
        run_id=profile.profile_id,
        run_source="fixed",
        model_tag=profile.model_tag,
        objective=profile.objective,
        training=TailtreeTrialParams(
            num_leaves=training.num_leaves,
            min_data_in_leaf=training.min_data_in_leaf,
            learning_rate=training.learning_rate,
            num_iterations=training.num_iterations,
            early_stopping_rounds=training.early_stopping_rounds,
        ),
    )


def _loaded_trial_params(path) -> TailtreeTrialParams:
    data = json.loads(path.read_text(encoding="utf-8"))
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"tailtree model metadata missing: {path}")
    train_config = metadata.get("train_config")
    if not isinstance(train_config, dict):
        raise ValueError(f"tailtree model train_config missing: {path}")
    return TailtreeTrialParams(
        num_leaves=int(train_config["num_leaves"]),
        min_data_in_leaf=int(train_config["min_data_in_leaf"]),
        learning_rate=float(train_config["learning_rate"]),
        num_iterations=int(train_config["num_iterations"]),
        early_stopping_rounds=int(train_config["early_stopping_rounds"]),
    )


def tailtree_predict_run(tailtree: TailtreeConfig) -> TailtreeProfileRun:
    if not tailtree.models:
        raise ValueError("tailtree load_predict requires explicit model ids")
    first = tailtree.models[0]
    return TailtreeProfileRun(
        profile_id="loaded-tailtree-models",
        run_id="loaded-tailtree-models",
        run_source="loaded",
        model_tag=first.model_tag,
        objective=first.objective,
        training=_loaded_trial_params(tailtree.model_dir / f"{first.model_id}.json"),
    )


def tailtree_profile_runs(config: PotentialConfig) -> tuple[TailtreeProfileRun, ...]:
    runs = []
    for profile in config.evidence.tailtree.profiles:
        if isinstance(profile.training, TailtreeFixedTrainingConfig):
            runs.append(tailtree_fixed_run(profile))
    return tuple(runs)


def tailtree_optuna_profiles(config: PotentialConfig) -> tuple[TailtreeProfileConfig, ...]:
    return tuple(
        profile
        for profile in config.evidence.tailtree.profiles
        if isinstance(profile.training, TailtreeOptunaTrainingConfig)
    )


def tailtree_objective_jobs(
    run: TailtreeProfileRun,
    *,
    fold_id: int,
    tailtree: TailtreeConfig,
) -> tuple[TailtreeObjectiveJob, ...]:
    jobs: list[TailtreeObjectiveJob] = []
    if tailtree.lifecycle == "load_predict":
        for model in tailtree.models:
            jobs.append(
                TailtreeObjectiveJob(
                    run=run,
                    fold_id=int(fold_id),
                    outcome_horizon=model.outcome_horizon,
                    direction=model.direction,
                    model_path=tailtree.model_dir / f"{model.model_id}.json",
                    label=f"{model.model_id}.loaded",
                )
            )
        return tuple(jobs)
    directions = ("up",) if run.objective.startswith("path_guard") else ("up", "down")
    for outcome_horizon in tailtree.outcome_horizon:
        for direction in directions:
            horizon = int(outcome_horizon)
            direction_value: TailtreeDirection = direction
            jobs.append(
                TailtreeObjectiveJob(
                    run=run,
                    fold_id=int(fold_id),
                    outcome_horizon=horizon,
                    direction=direction_value,
                    model_path=tailtree.model_dir / f"{run.model_tag}_{horizon}_{direction}.json",
                    label=f"{run.run_id}.h{horizon}.{direction}",
                )
            )
    return tuple(jobs)


def tailtree_fold_specs(
    evaluation: TailtreeSingleSplitSpec | TailtreeWalkforwardSpec,
    *,
    observations: pl.DataFrame,
    bar: str,
) -> tuple[TailtreeSingleSplitFold | TailtreeWalkforwardFold, ...]:
    if isinstance(evaluation, TailtreeSingleSplitSpec):
        return (
            TailtreeSingleSplitFold(
                fold_id=0,
                validation_fraction=0.0,
                embargo_bars=evaluation.embargo_bars,
            ),
        )
    if observations.is_empty() or "decision_bar_close_ms" not in observations.columns:
        raise ValueError(
            "tailtree walkforward requires non-empty decision_bar_close_ms observations"
        )
    start_value = observations.select(pl.col("decision_bar_close_ms").cast(pl.Int64).min()).item()
    end_value = observations.select(pl.col("decision_bar_close_ms").cast(pl.Int64).max()).item()
    if start_value is None or end_value is None:
        raise ValueError("tailtree walkforward requires non-null decision_bar_close_ms")
    start = int(start_value)
    end = int(end_value)
    train_ms = int(evaluation.train_days) * _MS_PER_DAY
    valid_ms = int(evaluation.valid_days) * _MS_PER_DAY
    step_ms = int(evaluation.step_days) * _MS_PER_DAY
    embargo_ms = int(evaluation.embargo_bars) * _bar_ms(bar)
    generated: list[TailtreeWalkforwardFold] = []
    cursor = start
    while True:
        train_window = TailtreeTimeWindow(cursor, cursor + train_ms)
        valid_window = TailtreeTimeWindow(
            train_window.end_ms + embargo_ms,
            train_window.end_ms + embargo_ms + valid_ms,
        )
        if valid_window.end_ms > end:
            break
        generated.append(
            TailtreeWalkforwardFold(
                fold_id=len(generated),
                train_window=train_window,
                valid_window=valid_window,
                embargo_bars=evaluation.embargo_bars,
            )
        )
        cursor += step_ms
    if not generated:
        raise ValueError("tailtree walkforward produced no valid folds for observation span")
    newest = generated[-int(evaluation.max_folds) :]
    return tuple(
        TailtreeWalkforwardFold(
            fold_id=index,
            train_window=fold.train_window,
            valid_window=fold.valid_window,
            embargo_bars=fold.embargo_bars,
        )
        for index, fold in enumerate(newest)
    )


def tailtree_frame_split(
    observations: pl.DataFrame,
    source_outcomes: pl.DataFrame,
    realized_transitions: pl.DataFrame,
    fold: TailtreeSingleSplitFold | TailtreeWalkforwardFold,
) -> TailtreeFrameSplit:
    if isinstance(fold, TailtreeSingleSplitFold):
        return TailtreeFrameSplit(
            train_observations=observations,
            valid_observations=observations,
            train_source_outcomes=source_outcomes,
            valid_source_outcomes=source_outcomes,
            train_realized_transitions=realized_transitions,
            valid_realized_transitions=realized_transitions,
        )
    return TailtreeFrameSplit(
        train_observations=_time_filter(observations, fold.train_window),
        valid_observations=_time_filter(observations, fold.valid_window),
        train_source_outcomes=_time_filter(source_outcomes, fold.train_window),
        valid_source_outcomes=_time_filter(source_outcomes, fold.valid_window),
        train_realized_transitions=_time_filter(realized_transitions, fold.train_window),
        valid_realized_transitions=_time_filter(realized_transitions, fold.valid_window),
    )
