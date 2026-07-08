"""Tailtree runtime planning: profile runs, folds, frame splits."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Literal

import polars as pl

from qooi.scanner.config import (
    Config,
    FixedTrainingConfig,
    Objective,
)
from qooi.scanner.tailrun.types import (
    ObjectiveJob,
    TailtreeFrameSplit,
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
    objective: Objective
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
    tailtree: Config,
    params: TailtreeTrialParams,
    *,
    trial_number: int,
) -> TailtreeProfileRun:
    run_id = f"path-prototype-fixed-t{trial_number:04d}"
    return TailtreeProfileRun(
        profile_id="path-prototype-fixed",
        run_id=run_id,
        run_source="optuna",
        model_tag=f"tailtree-path-t{trial_number:04d}",
        objective="path_prototype",
        training=params,
    )


def tailtree_fixed_run(tailtree: Config) -> TailtreeProfileRun:
    training = tailtree.training
    if not isinstance(training, FixedTrainingConfig):
        raise ValueError("tailtree_fixed_run requires fixed training")
    return TailtreeProfileRun(
        profile_id="path-prototype-fixed",
        run_id="path-prototype-fixed",
        run_source="fixed",
        model_tag="tailtree-path",
        objective="path_prototype",
        training=TailtreeTrialParams(
            num_leaves=training.num_leaves,
            min_data_in_leaf=training.min_data_in_leaf,
            learning_rate=training.learning_rate,
            num_iterations=training.num_iterations,
            early_stopping_rounds=training.early_stopping_rounds,
        ),
    )


def _loaded_trial_params(path) -> TailtreeTrialParams:
    from qooi.scanner.path_model import TailTreeModel

    train_config = TailTreeModel.from_json(path).metadata.train_config
    return TailtreeTrialParams(
        num_leaves=int(train_config.num_leaves),
        min_data_in_leaf=int(train_config.min_data_in_leaf),
        learning_rate=float(train_config.learning_rate),
        num_iterations=int(train_config.num_iterations),
        early_stopping_rounds=int(train_config.early_stopping_rounds),
    )


def tailtree_predict_run(tailtree: Config) -> TailtreeProfileRun:
    return TailtreeProfileRun(
        profile_id="loaded-tailtree-models",
        run_id="loaded-tailtree-models",
        run_source="loaded",
        model_tag=tailtree.model_id.removesuffix("_path"),
        objective="path_prototype",
        training=_loaded_trial_params(tailtree.model_dir / f"{tailtree.model_id}.json"),
    )


def tailtree_objective_jobs(
    run: TailtreeProfileRun,
    *,
    fold_id: int,
    tailtree: Config,
) -> tuple[ObjectiveJob, ...]:
    if tailtree.lifecycle == "load_predict":
        return (
            ObjectiveJob(
                run=run,
                fold_id=int(fold_id),
                outcome_horizon=0,
                direction="path",
                model_path=tailtree.model_dir / f"{tailtree.model_id}.json",
                label=f"{tailtree.model_id}.loaded",
            ),
        )
    return (
        ObjectiveJob(
            run=run,
            fold_id=int(fold_id),
            outcome_horizon=0,
            direction="path",
            model_path=tailtree.model_dir / f"{run.model_tag}_path.json",
            label=f"{run.run_id}.path",
        ),
    )


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


def _require_columns(frame: pl.DataFrame, *, name: str, columns: set[str]) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {', '.join(missing)}")


def make_horizon_samples(
    observations: pl.DataFrame,
    path_labels: pl.DataFrame,
    *,
    horizons: tuple[int, ...],
) -> pl.DataFrame:
    """Join observations to path labels at symbol × decision × horizon grain."""
    observation_keys = {"symbol", "decision_bar_close_ms"}
    label_keys = observation_keys | {"horizon_hours"}
    _require_columns(observations, name="observations", columns=observation_keys)
    _require_columns(path_labels, name="path_labels", columns=label_keys)
    horizon_values = [int(horizon) for horizon in horizons]
    if not horizon_values or observations.is_empty() or path_labels.is_empty():
        return observations.head(0).join(
            path_labels.head(0),
            on=["symbol", "decision_bar_close_ms"],
            how="inner",
        )
    labels = path_labels.filter(pl.col("horizon_hours").is_in(horizon_values))
    duplicate_count = (
        labels.group_by("symbol", "decision_bar_close_ms", "horizon_hours")
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    if duplicate_count:
        raise ValueError(
            "duplicate path label grain: symbol × decision_bar_close_ms × horizon_hours"
        )
    return observations.join(labels, on=["symbol", "decision_bar_close_ms"], how="inner").sort(
        "decision_bar_close_ms", "symbol", "horizon_hours"
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
