"""Tailtree lifecycle shared types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, TypedDict

import polars as pl


class TailtreeModelMetadata(Protocol):
    categorical_features: list[str]
    continuous_features: list[str]


class TailtreeArtifactTree(Protocol):
    metadata: TailtreeModelMetadata

    def predict_leaf(self, features: pl.DataFrame) -> pl.DataFrame: ...
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


TailtreeDirection = Literal["up", "down"]


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


@dataclass(frozen=True)
class TailtreeResult:
    """Tailtree path pipeline result. Every field has a concrete type."""

    evidence: pl.DataFrame
    candidates: pl.DataFrame
    ranked: pl.DataFrame
    models: dict[tuple[int, TailtreeDirection], TailtreeArtifactTree]
    sections: tuple


@dataclass(frozen=True)
class TailtreeEvidenceResult:
    """Tailtree evidence/model build result before candidate matching."""

    evidence: pl.DataFrame
    models: dict[tuple[int, TailtreeDirection], TailtreeArtifactTree]


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
            utility_mean = float(utilities.mean() or 0.0) if not utilities.is_empty() else 0.0
            utility_p90 = float(utilities.quantile(0.9) or 0.0) if not utilities.is_empty() else 0.0
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
