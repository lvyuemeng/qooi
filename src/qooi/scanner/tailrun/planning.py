"""Tailtree runtime planning: profile runs, folds, frame splits, selection context."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl

from qooi.scanner.config import (
    PotentialConfig,
    TailtreeConfig,
    TailtreeFixedTrainingConfig,
    TailtreeOptunaTrainingConfig,
    TailtreeProfileConfig,
)
from qooi.scanner.tailrun.selection import (
    TailtreeSelectionBudgets,
    TailtreeSelectionContext,
    TailtreeSelectionFeasibilityPolicy,
    TailtreeSelectionPolicy,
)
from qooi.scanner.tailrun.types import (
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
    run_source: Literal["fixed", "optuna"]
    model_tag: str
    objective: str
    training: TailtreeTrialParams


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


@dataclass(frozen=True)
class TailtreeExecutionContext:
    run: TailtreeProfileRun
    fold: TailtreeSingleSplitFold | TailtreeWalkforwardFold
    tailtree: TailtreeConfig
    selection: TailtreeSelectionPolicy
    universe_snapshot_id: str

    def selection_context(self) -> TailtreeSelectionContext:
        if isinstance(self.fold, TailtreeSingleSplitFold):
            return TailtreeSelectionContext.from_strings(
                trial_id=self.run.run_id,
                trial_source=self.run.run_source,
                fold_id=self.fold.fold_id,
                evaluation_protocol=self.fold.protocol,
                embargo_bars=self.fold.embargo_bars,
                universe_snapshot_id=self.universe_snapshot_id,
                model_tag=self.run.model_tag,
                objective=self.run.objective,
                training_profile=self.run.profile_id,
            )
        return TailtreeSelectionContext.from_strings(
            trial_id=self.run.run_id,
            trial_source=self.run.run_source,
            fold_id=self.fold.fold_id,
            evaluation_protocol=self.fold.protocol,
            train_start_ms=self.fold.train_window.start_ms,
            train_end_ms=self.fold.train_window.end_ms,
            valid_start_ms=self.fold.valid_window.start_ms,
            valid_end_ms=self.fold.valid_window.end_ms,
            embargo_bars=self.fold.embargo_bars,
            universe_snapshot_id=self.universe_snapshot_id,
            model_tag=self.run.model_tag,
            objective=self.run.objective,
            training_profile=self.run.profile_id,
        )


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


def tailtree_selection_policy(tailtree: TailtreeConfig) -> TailtreeSelectionPolicy:
    selection = tailtree.selection
    return TailtreeSelectionPolicy(
        budgets=TailtreeSelectionBudgets(
            top_k=selection.top_k,
            top_pct=selection.top_pct,
            score_gate=selection.score_gate,
        ),
        feasibility=TailtreeSelectionFeasibilityPolicy(
            min_selected_observation_count=selection.min_selected_observation_count,
            min_selected_symbol_count=selection.min_selected_symbol_count,
            min_selected_tail_count=selection.min_selected_tail_count,
            min_valid_tail_lift=selection.min_valid_tail_lift,
            min_profit_proxy_per_selected_obs=selection.min_profit_proxy_per_selected_obs,
        ),
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


def tailtree_fold_specs(
    evaluation: TailtreeSingleSplitSpec | TailtreeWalkforwardSpec,
    *,
    observations: pl.DataFrame,
    bar: str,
) -> tuple[TailtreeSingleSplitFold | TailtreeWalkforwardFold, ...]:
    if evaluation.protocol == "single_split":
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
    timestamps = observations.get_column("decision_bar_close_ms")
    start = int(timestamps.min())
    end = int(timestamps.max())
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


def tailtree_execution_contexts(
    config: PotentialConfig,
    *,
    observations: pl.DataFrame,
    universe_snapshot_id: str,
) -> tuple[TailtreeExecutionContext, ...]:
    if config.bars is None:
        return ()
    bar = config.bars.timeframes[0]
    fold = TailtreeSingleSplitFold(fold_id=0, validation_fraction=0.0, embargo_bars=0)
    return tuple(
        tailtree_execution_context(
            run, fold, tailtree=config.evidence.tailtree, universe_snapshot_id=universe_snapshot_id
        )
        for run in tailtree_profile_runs(config)
        if not observations.is_empty() or bar
    )


def tailtree_execution_context(
    run: TailtreeProfileRun,
    fold: TailtreeSingleSplitFold | TailtreeWalkforwardFold,
    *,
    tailtree: TailtreeConfig,
    universe_snapshot_id: str,
) -> TailtreeExecutionContext:
    context_tailtree = tailtree
    if isinstance(fold, TailtreeWalkforwardFold):
        context_tailtree = tailtree.model_copy(
            update={"model_dir": Path(tailtree.model_dir) / run.run_id / f"fold-{fold.fold_id:02d}"}
        )
    return TailtreeExecutionContext(
        run=run,
        fold=fold,
        tailtree=context_tailtree,
        selection=tailtree_selection_policy(tailtree),
        universe_snapshot_id=universe_snapshot_id,
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
