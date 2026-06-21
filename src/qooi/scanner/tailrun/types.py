"""Tailtree lifecycle shared types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, TypedDict

import polars as pl

from qooi.pipeline.types import FrameHealth, ProductResult
from qooi.scanner import (
    PotentialScanConfig,
    SourceStateRow,
    TransitionAnalysis,
    TransitionPattern,
)

if TYPE_CHECKING:
    from qooi.scanner.tailrun.planning import TailtreeProfileRun


@dataclass(frozen=True)
class PotentialUniverse:
    discovery: pl.DataFrame
    selection_note: str
    missing_reason: str
    eligible_count: int = 0


@dataclass(frozen=True)
class PotentialArtifacts:
    report: Path
    diagnostics_dir: Path
    states_dir: Path


@dataclass(frozen=True)
class BarFetchResult:
    frames: dict[tuple[str, str], pl.DataFrame]
    state_frames: dict[tuple[str, str], pl.DataFrame]
    coverage: tuple[FrameHealth, ...]


@dataclass(frozen=True)
class SymbolStateBundle:
    symbol: str
    kline: SourceStateRow
    transition: SourceStateRow
    books: SourceStateRow
    trades: SourceStateRow
    derivatives: SourceStateRow
    context: SourceStateRow
    coverage_notes: tuple[str, ...]
    transition_patterns: tuple[TransitionPattern, ...] = ()


@dataclass(frozen=True)
class ScanDecision:
    symbol: str
    group: str
    direction: str
    confidence: str
    transition_evidence: str
    structure_evidence: str
    flow_evidence: str
    liquidity_evidence: str
    derivatives_evidence: str
    context_evidence: str
    missing_evidence: tuple[str, ...]
    contradictory_evidence: tuple[str, ...]
    block_reason: str
    review_caveat: str


@dataclass
class ReportInputs:
    config: PotentialScanConfig
    artifacts: PotentialArtifacts
    universe: PotentialUniverse
    bars: BarFetchResult
    context: dict[str, ProductResult]
    transitions: TransitionAnalysis
    bundles: tuple[SymbolStateBundle, ...]
    decisions: tuple[ScanDecision, ...]


TailtreeDirection = Literal["up", "down"]


class TailtreeModelMetadata(Protocol):
    @property
    def direction(self) -> TailtreeDirection: ...
    @property
    def categorical_features(self) -> list[str]: ...
    @property
    def continuous_features(self) -> list[str]: ...


class TailtreeArtifactTree(Protocol):
    @property
    def metadata(self) -> TailtreeModelMetadata: ...

    def predict_leaf(self, features: pl.DataFrame) -> pl.DataFrame: ...
    def predict_score(self, features: pl.DataFrame) -> pl.DataFrame: ...
    def to_json(self, path: Path) -> None: ...


class TailtreeArtifactMetadata(TypedDict):
    bar: str
    outcome_horizon: int
    threshold_pct: float
    categorical_features: list[str]
    continuous_features: list[str]
    feature_schema_hash: str
    model_tag: str
    trained_tree_count: int


TAILTREE_RUN_SUMMARY_SCHEMA = {
    "summary_scope": pl.String,
    "direction": pl.String,
    "objective": pl.String,
    "outcome_horizon": pl.Int64,
    "observation_row_count": pl.Int64,
    "outcome_row_count": pl.Int64,
    "source_event_row_count": pl.Int64,
    "source_outcome_row_count": pl.Int64,
    "realized_transition_row_count": pl.Int64,
    "feature_count": pl.Int64,
    "categorical_feature_count": pl.Int64,
    "continuous_feature_count": pl.Int64,
    "forward_return_nonnull_count": pl.Int64,
    "forward_min_return_nonnull_count": pl.Int64,
    "forward_max_return_nonnull_count": pl.Int64,
    "path_range_nonnull_count": pl.Int64,
    "time_to_max_nonnull_count": pl.Int64,
    "time_to_min_nonnull_count": pl.Int64,
    "retention_nonnull_count": pl.Int64,
    "path_efficiency_nonnull_count": pl.Int64,
    "tail_utility_mean": pl.Float64,
    "tail_utility_p90": pl.Float64,
    "train_tail_count": pl.Int64,
    "valid_observation_count": pl.Int64,
    "valid_tail_count": pl.Int64,
    "valid_tail_rate": pl.Float64,
    "valid_selected_observation_count": pl.Int64,
    "valid_selected_tail_count": pl.Int64,
    "valid_selected_tail_rate": pl.Float64,
    "valid_tail_lift": pl.Float64,
    "valid_selected_utility_mean": pl.Float64,
    "valid_selected_utility_p90": pl.Float64,
    "threshold_pct": pl.Float64,
    "tail_count": pl.Int64,
    "tail_rate": pl.Float64,
    "train_observation_count": pl.Int64,
    "train_exceedance_count": pl.Int64,
    "min_exceedance_required": pl.Int64,
    "trainable_flag": pl.Int64,
    "trained_tree_count": pl.Int64,
    "selected_leaf_count": pl.Int64,
    "written_model_file_count": pl.Int64,
    "written_evidence_file_count": pl.Int64,
    "removed_stale_file_count": pl.Int64,
}


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


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


@dataclass(frozen=True)
class TailtreeEvidenceResult:
    """Tailtree evidence/model build result before candidate matching."""

    evidence: pl.DataFrame
    models: dict[tuple[int, TailtreeDirection], TailtreeArtifactTree]


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
class TailtreeObjectiveJob:
    run: TailtreeProfileRun
    fold_id: int
    outcome_horizon: int
    direction: TailtreeDirection
    model_path: Path
    label: str


@dataclass(frozen=True)
class TailtreeJobResult:
    job: TailtreeObjectiveJob
    evidence: pl.DataFrame
    scored_candidates: pl.DataFrame
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


@dataclass(frozen=True)
class TailtreeSelectionEfficiencyRow:
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
    models: dict[tuple[int, TailtreeDirection], TailtreeArtifactTree]
    profile_runs: tuple[TailtreeProfileFeedback, ...]
    selection_efficiency: pl.DataFrame
    label_distribution: pl.DataFrame
    action_surface: pl.DataFrame


@dataclass(frozen=True)
class TailtreeDirectionQuality:
    """Validation-quality counts for one tailtree direction."""

    direction: TailtreeDirection
    train_tail_count: int
    valid_observation_count: int
    valid_tail_count: int
    valid_tail_rate: float
    valid_selected_observation_count: int
    valid_selected_tail_count: int
    valid_selected_tail_rate: float
    valid_tail_lift: float
    valid_selected_utility_mean: float
    valid_selected_utility_p90: float

    @classmethod
    def zero(
        cls, direction: TailtreeDirection, *, train_tail_count: int = 0
    ) -> TailtreeDirectionQuality:
        return cls(
            direction=direction,
            train_tail_count=train_tail_count,
            valid_observation_count=0,
            valid_tail_count=0,
            valid_tail_rate=0.0,
            valid_selected_observation_count=0,
            valid_selected_tail_count=0,
            valid_selected_tail_rate=0.0,
            valid_tail_lift=0.0,
            valid_selected_utility_mean=0.0,
            valid_selected_utility_p90=0.0,
        )

    @classmethod
    def from_labeled_leaf_frame(
        cls,
        *,
        direction: TailtreeDirection,
        train_tail_count: int,
        validation_leaf_frame: pl.DataFrame,
        selected_leaf_ids: set[int],
    ) -> TailtreeDirectionQuality:
        tail_col = f"tail_{direction}"
        if validation_leaf_frame.is_empty() or tail_col not in validation_leaf_frame.columns:
            return cls.zero(direction, train_tail_count=train_tail_count)
        valid_observation_count = len(validation_leaf_frame)
        valid_tail_count = int(validation_leaf_frame.get_column(tail_col).fill_null(False).sum())
        selected = (
            validation_leaf_frame.filter(pl.col("leaf_id").is_in(selected_leaf_ids))
            if selected_leaf_ids and "leaf_id" in validation_leaf_frame.columns
            else validation_leaf_frame.head(0)
        )
        valid_selected_observation_count = len(selected)
        valid_selected_tail_count = (
            int(selected.get_column(tail_col).fill_null(False).sum())
            if not selected.is_empty()
            else 0
        )
        valid_tail_rate = _rate(valid_tail_count, valid_observation_count)
        valid_selected_tail_rate = _rate(
            valid_selected_tail_count, valid_selected_observation_count
        )
        utility_col = f"tail_utility_{direction}"
        if not selected.is_empty() and utility_col in selected.columns:
            utilities = selected.filter(pl.col(tail_col).fill_null(False)).get_column(utility_col)
            value = utilities.to_frame().select(pl.col(utility_col).cast(pl.Float64).mean()).item()
            utility_mean = float(value) if value is not None else 0.0
            p90 = (
                utilities.to_frame()
                .select(pl.col(utility_col).cast(pl.Float64).quantile(0.9))
                .item()
            )
            utility_p90 = float(p90) if p90 is not None else 0.0
        else:
            utility_mean = 0.0
            utility_p90 = 0.0
        return cls(
            direction=direction,
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
            valid_selected_utility_mean=utility_mean,
            valid_selected_utility_p90=utility_p90,
        )

    def to_summary_fields(self) -> dict[str, int | float]:
        return {
            "train_tail_count": self.train_tail_count,
            "valid_observation_count": self.valid_observation_count,
            "valid_tail_count": self.valid_tail_count,
            "valid_tail_rate": self.valid_tail_rate,
            "valid_selected_observation_count": self.valid_selected_observation_count,
            "valid_selected_tail_count": self.valid_selected_tail_count,
            "valid_selected_tail_rate": self.valid_selected_tail_rate,
            "valid_tail_lift": self.valid_tail_lift,
            "valid_selected_utility_mean": self.valid_selected_utility_mean,
            "valid_selected_utility_p90": self.valid_selected_utility_p90,
        }
