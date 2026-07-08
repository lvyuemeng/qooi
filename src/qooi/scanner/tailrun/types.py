"""Tailtree lifecycle shared types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict

from qooi.scanner.path_model import TailTreeModel

if TYPE_CHECKING:
    from qooi.scanner.tailrun.planning import TailtreeProfileRun

TailtreeDirection = Literal["up", "down", "path"]
TailtreeArtifactTree = TailTreeModel



@dataclass(frozen=True)
class TailtreeTimeWindow:
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class TailtreeSingleSplitFold:
    protocol: Literal["single_split"] = "single_split"
    fold_id: int = 0
    validation_fraction: float = 0.0
    embargo_bars: int = 0


@dataclass(frozen=True)
class TailtreeWalkforwardFold:
    train_window: TailtreeTimeWindow
    valid_window: TailtreeTimeWindow
    embargo_bars: int
    protocol: Literal["walkforward"] = "walkforward"
    fold_id: int = 0


@dataclass(frozen=True)
class TailtreeFrameSplit:
    train_observations: pl.DataFrame
    valid_observations: pl.DataFrame
    train_source_outcomes: pl.DataFrame
    valid_source_outcomes: pl.DataFrame
    train_realized_transitions: pl.DataFrame
    valid_realized_transitions: pl.DataFrame


@dataclass(frozen=True)
class TailtreeInputFrames:
    observations: pl.DataFrame
    source_outcomes: pl.DataFrame
    realized: pl.DataFrame
    histories: pl.DataFrame


@dataclass(frozen=True)
class TailtreePreparedFrames:
    observations: pl.DataFrame
    source_outcomes: pl.DataFrame
    realized: pl.DataFrame
    histories: pl.DataFrame
    outcomes: pl.DataFrame
    labeled_outcomes: pl.DataFrame
    categorical_features: list[str]
    continuous_features: list[str]
    score_observations: pl.DataFrame | None = None
    score_labeled_outcomes: pl.DataFrame | None = None


@dataclass(frozen=True)
class ObjectiveJob:
    run: TailtreeProfileRun
    fold_id: int
    outcome_horizon: int
    direction: TailtreeDirection
    model_path: Path
    label: str


@dataclass(frozen=True)
class TailtreeJobResult:
    job: ObjectiveJob
    evidence: pl.DataFrame
    scored_candidates: pl.DataFrame
    scored_population: pl.DataFrame
    model: TailtreeArtifactTree | None


@dataclass(frozen=True)
class TailtreeProfileFeedback:
    run_id: str
    trial_id: str
    trial_source: str
    objective: str
    training_profile: str
    model_tag: str
    num_leaves: int
    min_data_in_leaf: int
    learning_rate: float
    num_iterations: int
    early_stopping_rounds: int
    score: float
    evidence_rows: int
    model_count: int
    seconds: float


class TailtreeSelectionEfficiencyRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    universe_snapshot_id: str
    model_tag: str
    objective: str
    training_profile: str
    trial_id: str
    trial_source: str
    outcome_horizon: int
    tree_direction: str
    budget_family: str
    budget_value: float
    eligible_symbol_count: int
    selected_symbol_count: int
    observation_row_count: int
    feature_count: int
    train_exceedance_count: int
    valid_observation_count: int
    valid_tail_count: int
    valid_tail_rate: float
    selected_observation_count: int
    selected_observation_rate: float
    selected_tail_count: int
    selected_tail_rate: float
    selected_tail_per_1k_obs: float
    valid_tail_lift: float
    selected_profit_proxy_mean: float
    selected_profit_proxy_p90: float
    selected_utility_mean: float
    selected_utility_p90: float
    profit_proxy_per_selected_obs: float
    profit_proxy_per_1k_observed: float
    hpo_score: float
    promotion_threshold_pass_int: int
    trained_tree_count: int
    selected_bucket_or_leaf_count: int
    fit_seconds: float
    score_seconds: float


@dataclass(frozen=True)
class TailtreeRunOutput:
    evidence: pl.DataFrame
    board: pl.DataFrame
    models: dict[tuple[int, TailtreeDirection], TailtreeArtifactTree]
    profile_runs: tuple[TailtreeProfileFeedback, ...]
    selection_efficiency: pl.DataFrame
    label_distribution: pl.DataFrame
    action_surface: pl.DataFrame
