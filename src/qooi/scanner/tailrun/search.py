"""Tailtree Optuna dependency and trial parameter suggestion."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Protocol

from qooi.scanner.tailrun import planning


class _OptunaTrial(Protocol):
    number: int

    def suggest_int(
        self, name: str, low: int, high: int, *, step: int = 1, log: bool = False
    ) -> int: ...

    def suggest_float(self, name: str, low: float, high: float, *, log: bool = False) -> float: ...


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
