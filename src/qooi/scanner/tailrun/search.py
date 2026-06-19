"""Adaptive tailtree HPO study orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol

import polars as pl

from qooi.scanner.tailrun import planning
from qooi.scanner.tailrun.selection import _winner_score_expr
from qooi.scanner.tailrun.types import TailtreeSingleSplitFold, TailtreeWalkforwardFold


class _OptunaTrial(Protocol):
    number: int

    def suggest_int(
        self, name: str, low: int, high: int, *, step: int = 1, log: bool = False
    ) -> int: ...

    def suggest_float(self, name: str, low: float, high: float, *, log: bool = False) -> float: ...

    def report(self, value: float, step: int) -> None: ...

    def should_prune(self) -> bool: ...


@dataclass(frozen=True)
class TailtreeContextExecution:
    selection_efficiency: pl.DataFrame


@dataclass(frozen=True)
class TailtreeHpoStudyResult:
    selection_efficiency: pl.DataFrame
    trial_feedback: pl.DataFrame


def tailtree_trial_feedback_frame(selection_efficiency: pl.DataFrame) -> pl.DataFrame:
    fold_group_cols = [
        "universe_snapshot_id",
        "evaluation_protocol",
        "fold_id",
        "outcome_label_family",
        "outcome_horizon",
        "tree_direction",
    ]
    trial_group_cols = [
        "universe_snapshot_id",
        "evaluation_protocol",
        "outcome_label_family",
        "outcome_horizon",
        "tree_direction",
        "trial_id",
        "trial_source",
        "objective",
        "training_profile",
        "budget_family",
        "budget_value",
    ]
    required = {
        *fold_group_cols,
        *trial_group_cols,
        "selected_observation_count",
        "selected_tail_count",
        "selected_observation_rate",
        "valid_tail_lift",
        "profit_proxy_per_selected_obs",
        "profit_proxy_per_1k_observed",
        "promotion_threshold_pass_int",
        "feasibility_pass_int",
    }
    if selection_efficiency.is_empty() or not required.issubset(selection_efficiency.columns):
        return selection_efficiency.head(0).with_columns(
            pl.lit(None, dtype=pl.Int64).alias("hpo_fold_count"),
            pl.lit(None, dtype=pl.Int64).alias("hpo_feasible_fold_count"),
            pl.lit(None, dtype=pl.Float64).alias("hpo_trial_score"),
            pl.lit(None, dtype=pl.Int64).alias("hpo_trial_selected_int"),
        )
    scored = selection_efficiency.with_columns(
        _winner_score_expr(fold_group_cols, has_feasibility=True).alias("_fold_feedback_score")
    )
    aggregated = scored.group_by(trial_group_cols, maintain_order=True).agg(
        pl.col("fold_id").n_unique().cast(pl.Int64).alias("hpo_fold_count"),
        pl.col("feasibility_pass_int").sum().cast(pl.Int64).alias("hpo_feasible_fold_count"),
        pl.col("feasibility_pass_int").min().cast(pl.Int64).alias("hpo_all_folds_feasible_int"),
        pl.col("_fold_feedback_score").mean().alias("hpo_mean_fold_score"),
        pl.col("_fold_feedback_score").min().alias("hpo_min_fold_score"),
        pl.col("valid_tail_lift").mean().alias("hpo_mean_valid_tail_lift"),
        pl.col("selected_observation_rate").mean().alias("hpo_mean_selected_observation_rate"),
        pl.col("profit_proxy_per_selected_obs")
        .mean()
        .alias("hpo_mean_profit_proxy_per_selected_obs"),
        pl.col("selected_observation_count")
        .sum()
        .cast(pl.Int64)
        .alias("hpo_total_selected_observation_count"),
        pl.col("selected_tail_count").sum().cast(pl.Int64).alias("hpo_total_selected_tail_count"),
    )
    scored_trials = aggregated.with_columns(
        pl.when(pl.col("hpo_all_folds_feasible_int") == 1)
        .then((pl.col("hpo_min_fold_score") + pl.col("hpo_mean_fold_score")) / 2.0)
        .otherwise(-1_000_000_000.0 + pl.col("hpo_mean_fold_score"))
        .alias("hpo_trial_score"),
        pl.concat_str(
            [
                pl.col("objective"),
                pl.col("training_profile"),
                pl.col("trial_id"),
                pl.col("budget_family"),
                pl.col("budget_value").cast(pl.String),
            ],
            separator="|",
        ).alias("hpo_trial_setting_id"),
    )
    rank_group_cols = [
        "universe_snapshot_id",
        "evaluation_protocol",
        "outcome_label_family",
        "outcome_horizon",
        "tree_direction",
    ]
    sorted_trials = scored_trials.sort(
        [
            *rank_group_cols,
            "hpo_trial_score",
            "hpo_all_folds_feasible_int",
            "hpo_mean_profit_proxy_per_selected_obs",
            "hpo_mean_valid_tail_lift",
            "hpo_total_selected_tail_count",
            "hpo_mean_selected_observation_rate",
            "objective",
            "training_profile",
            "trial_id",
            "budget_family",
            "budget_value",
        ],
        descending=[
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
    )
    return sorted_trials.with_columns(
        pl.cum_count("hpo_trial_score")
        .over(rank_group_cols)
        .cast(pl.Int64)
        .alias("hpo_trial_rank"),
        (pl.col("hpo_trial_score").max().over(rank_group_cols) - pl.col("hpo_trial_score"))
        .cast(pl.Float64)
        .alias("hpo_trial_margin_to_best"),
    ).with_columns((pl.col("hpo_trial_rank") == 1).cast(pl.Int64).alias("hpo_trial_selected_int"))


def optuna_module():
    try:
        return import_module("optuna")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "tailtree optuna training requires the optional optuna dependency; "
            "install the tailtree dependency group before running kind='optuna' profiles"
        ) from exc


def suggest_tailtree_trial_params(
    trial: _OptunaTrial, training: Any
) -> planning.TailtreeTrialParams:
    return planning.TailtreeTrialParams(
        num_leaves=_suggest_int_range(
            trial,
            "num_leaves",
            int(training.num_leaves),
            training.num_leaves_range,
            default_low=16,
            default_high=max(128, int(training.num_leaves) * 2),
            log=True,
        ),
        min_data_in_leaf=_suggest_int_range(
            trial,
            "min_data_in_leaf",
            int(training.min_data_in_leaf),
            training.min_data_in_leaf_range,
            default_low=max(10, int(training.min_data_in_leaf) // 2),
            default_high=max(int(training.min_data_in_leaf), int(training.min_data_in_leaf) * 4),
            log=True,
        ),
        learning_rate=_suggest_float_range(
            trial,
            "learning_rate",
            float(training.learning_rate),
            training.learning_rate_range,
            default_low=max(0.005, float(training.learning_rate) / 3.0),
            default_high=min(0.20, float(training.learning_rate) * 3.0),
            log=True,
        ),
        num_iterations=_suggest_int_range(
            trial,
            "num_iterations",
            int(training.num_iterations),
            training.num_iterations_range,
            default_low=max(40, int(training.num_iterations) // 2),
            default_high=max(int(training.num_iterations) + 80, int(training.num_iterations) * 2),
            step=20,
        ),
        early_stopping_rounds=_suggest_int_range(
            trial,
            "early_stopping_rounds",
            int(training.early_stopping_rounds),
            training.early_stopping_rounds_range,
            default_low=max(5, int(training.early_stopping_rounds) // 2),
            default_high=max(
                int(training.early_stopping_rounds) + 10,
                int(training.early_stopping_rounds) * 2,
            ),
            step=5,
        ),
    )


def _suggest_int_range(
    trial: _OptunaTrial,
    name: str,
    seed: int,
    configured: tuple[int, int] | None,
    *,
    default_low: int,
    default_high: int,
    log: bool = False,
    step: int = 1,
) -> int:
    low, high = configured or (default_low, default_high)
    low = max(1, int(low))
    high = max(low, int(high))
    if log:
        return int(trial.suggest_int(name, low, high, log=True))
    return int(trial.suggest_int(name, low, high, step=step))


def _suggest_float_range(
    trial: _OptunaTrial,
    name: str,
    seed: float,
    configured: tuple[float, float] | None,
    *,
    default_low: float,
    default_high: float,
    log: bool = False,
) -> float:
    low, high = configured or (default_low, default_high)
    low = max(1e-9, float(low))
    high = max(low, float(high))
    return float(trial.suggest_float(name, low, high, log=log))


def trial_objective_score(selection_efficiency: pl.DataFrame) -> float:
    feedback = tailtree_trial_feedback_frame(selection_efficiency)
    if feedback.is_empty() or "hpo_trial_score" not in feedback.columns:
        return -1_000_000_000.0
    value = feedback.get_column("hpo_trial_score").max()
    return float(value) if value is not None else -1_000_000_000.0


def run_tailtree_hpo_study(
    *,
    base: Any,
    profile: Any,
    folds: tuple[TailtreeSingleSplitFold | TailtreeWalkforwardFold, ...],
    universe_snapshot_id: str,
    execute_context: Callable[[planning.TailtreeExecutionContext], TailtreeContextExecution],
) -> TailtreeHpoStudyResult:
    optuna = optuna_module()
    training = profile.training
    startup_trials = max(1, min(10, int(training.max_trials) // 4 or 1))
    sampler = optuna.samplers.TPESampler(
        seed=training.seed,
        n_startup_trials=startup_trials,
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    selection_frames: list[pl.DataFrame] = []
    for _ in range(int(training.max_trials)):
        trial = study.ask()
        params = suggest_tailtree_trial_params(trial, training)
        run = planning.tailtree_trial_run(profile, params, trial_number=trial.number)
        trial_frames: list[pl.DataFrame] = []
        pruned = False
        for fold_index, fold in enumerate(folds):
            context = planning.tailtree_execution_context(
                run,
                fold,
                tailtree=base,
                universe_snapshot_id=universe_snapshot_id,
            )
            executed = execute_context(context)
            selection_frames.append(executed.selection_efficiency)
            trial_frames.append(executed.selection_efficiency)
            score = trial_objective_score(pl.concat(trial_frames, how="diagonal_relaxed"))
            trial.report(score, step=fold_index)
            if trial.should_prune():
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                pruned = True
                break
        if not pruned:
            trial_score = trial_objective_score(pl.concat(trial_frames, how="diagonal_relaxed"))
            study.tell(trial, trial_score)
    selection_efficiency = (
        pl.concat(selection_frames, how="diagonal_relaxed") if selection_frames else pl.DataFrame()
    )
    return TailtreeHpoStudyResult(
        selection_efficiency=selection_efficiency,
        trial_feedback=tailtree_trial_feedback_frame(selection_efficiency),
    )
