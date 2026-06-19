"""Tailtree profit-selection efficiency artifact."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log1p
from pathlib import Path
from typing import NewType

import polars as pl
from pydantic import BaseModel, ConfigDict, field_validator

UniverseSnapshotId = NewType("UniverseSnapshotId", str)
TailtreeModelTag = NewType("TailtreeModelTag", str)
TailtreeTrainingProfile = NewType("TailtreeTrainingProfile", str)

TAILTREE_SELECTION_EFFICIENCY_SCHEMA = {
    "trial_id": pl.String,
    "trial_source": pl.String,
    "fold_id": pl.Int64,
    "evaluation_protocol": pl.String,
    "train_start_ms": pl.Int64,
    "train_end_ms": pl.Int64,
    "valid_start_ms": pl.Int64,
    "valid_end_ms": pl.Int64,
    "embargo_bars": pl.Int64,
    "universe_snapshot_id": pl.String,
    "model_tag": pl.String,
    "objective": pl.String,
    "training_profile": pl.String,
    "outcome_label_family": pl.String,
    "comparison_surface": pl.String,
    "objective_score_comparable_int": pl.Int64,
    "outcome_horizon": pl.Int64,
    "tree_direction": pl.String,
    "budget_family": pl.String,
    "budget_value": pl.Float64,
    "eligible_symbol_count": pl.Int64,
    "selected_symbol_count": pl.Int64,
    "observation_row_count": pl.Int64,
    "feature_count": pl.Int64,
    "train_exceedance_count": pl.Int64,
    "valid_observation_count": pl.Int64,
    "valid_tail_count": pl.Int64,
    "valid_tail_rate": pl.Float64,
    "selected_observation_count": pl.Int64,
    "selected_observation_rate": pl.Float64,
    "selected_tail_count": pl.Int64,
    "selected_tail_rate": pl.Float64,
    "selected_tail_per_1k_obs": pl.Float64,
    "valid_tail_lift": pl.Float64,
    "selected_profit_proxy_mean": pl.Float64,
    "selected_profit_proxy_p90": pl.Float64,
    "selected_utility_mean": pl.Float64,
    "selected_utility_p90": pl.Float64,
    "profit_proxy_per_selected_obs": pl.Float64,
    "profit_proxy_per_1k_observed": pl.Float64,
    "hpo_score": pl.Float64,
    "promotion_threshold_pass_int": pl.Int64,
    "feasibility_support_pass_int": pl.Int64,
    "feasibility_concentration_pass_int": pl.Int64,
    "feasibility_utility_pass_int": pl.Int64,
    "feasibility_pass_int": pl.Int64,
    "trained_tree_count": pl.Int64,
    "selected_bucket_or_leaf_count": pl.Int64,
    "fit_seconds": pl.Float64,
    "score_seconds": pl.Float64,
}


class TailtreeSelectionBudgets(BaseModel):
    """Comparable candidate budgets for objective/HPO replay."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    top_k: tuple[int, ...] = (1, 3, 5, 10)
    top_pct: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20)
    score_gate: tuple[float, ...] = ()

    @field_validator("top_k")
    @classmethod
    def _positive_top_k(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(value for value in values if value > 0)

    @field_validator("top_pct")
    @classmethod
    def _positive_top_pct(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(value for value in values if value > 0.0)

    def iter_rows(self) -> tuple[tuple[str, float], ...]:
        return (
            *(("top_k", float(value)) for value in self.top_k),
            *(("top_pct", float(value)) for value in self.top_pct),
            *(("score_gate", float(value)) for value in self.score_gate),
        )


class TailtreeSelectionFeasibilityPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    min_selected_observation_count: int = 10
    min_selected_symbol_count: int = 1
    min_selected_tail_count: int = 1
    min_valid_tail_lift: float = 1.0
    min_profit_proxy_per_selected_obs: float = 0.0


@dataclass(frozen=True)
class TailtreeSelectionPolicy:
    """Compact replay policy for one tailtree selection-efficiency pass."""

    budgets: TailtreeSelectionBudgets = TailtreeSelectionBudgets()
    feasibility: TailtreeSelectionFeasibilityPolicy = TailtreeSelectionFeasibilityPolicy()


@dataclass(frozen=True)
class TailtreeSelectionContext:
    """Opaque replay identity for one tailtree selection-efficiency pass."""

    trial_id: str
    trial_source: str
    fold_id: int
    evaluation_protocol: str
    train_start_ms: int | None
    train_end_ms: int | None
    valid_start_ms: int | None
    valid_end_ms: int | None
    embargo_bars: int
    universe_snapshot_id: UniverseSnapshotId
    model_tag: TailtreeModelTag
    objective: str
    training_profile: TailtreeTrainingProfile = TailtreeTrainingProfile("balanced_baseline")

    @classmethod
    def from_strings(
        cls,
        *,
        trial_id: str = "primary",
        trial_source: str = "primary",
        fold_id: int = 0,
        evaluation_protocol: str = "single_split",
        train_start_ms: int | None = None,
        train_end_ms: int | None = None,
        valid_start_ms: int | None = None,
        valid_end_ms: int | None = None,
        embargo_bars: int = 0,
        universe_snapshot_id: str,
        model_tag: str,
        objective: str,
        training_profile: str,
    ) -> TailtreeSelectionContext:
        return cls(
            trial_id=trial_id,
            trial_source=trial_source,
            fold_id=int(fold_id),
            evaluation_protocol=evaluation_protocol,
            train_start_ms=train_start_ms,
            train_end_ms=train_end_ms,
            valid_start_ms=valid_start_ms,
            valid_end_ms=valid_end_ms,
            embargo_bars=int(embargo_bars),
            universe_snapshot_id=UniverseSnapshotId(universe_snapshot_id),
            model_tag=TailtreeModelTag(model_tag),
            objective=objective,
            training_profile=TailtreeTrainingProfile(training_profile),
        )


def with_tailtree_selection_identity(
    frame: pl.DataFrame, context: TailtreeSelectionContext
) -> pl.DataFrame:
    """Attach one run/fold identity to a tailtree artifact frame."""
    if frame.is_empty():
        return frame
    return frame.with_columns(
        pl.lit(context.trial_id).alias("trial_id"),
        pl.lit(context.trial_source).alias("trial_source"),
        pl.lit(context.fold_id).alias("fold_id"),
        pl.lit(context.evaluation_protocol).alias("evaluation_protocol"),
        pl.lit(context.train_start_ms, dtype=pl.Int64).alias("train_start_ms"),
        pl.lit(context.train_end_ms, dtype=pl.Int64).alias("train_end_ms"),
        pl.lit(context.valid_start_ms, dtype=pl.Int64).alias("valid_start_ms"),
        pl.lit(context.valid_end_ms, dtype=pl.Int64).alias("valid_end_ms"),
        pl.lit(context.embargo_bars).alias("embargo_bars"),
        pl.lit(str(context.model_tag)).alias("model_tag"),
        pl.lit(str(context.training_profile)).alias("training_profile"),
        pl.lit(context.objective).alias("objective"),
    )


@dataclass(frozen=True)
class TailtreeSummaryView:
    """Typed access to one run-summary row; avoids untyped helper access."""

    row: dict[str, object]
    fallback_observation_count: int

    @classmethod
    def from_frame(
        cls,
        run_summary: pl.DataFrame,
        *,
        outcome_horizon: int,
        direction: str,
        fallback_observation_count: int,
    ) -> TailtreeSummaryView:
        if run_summary.is_empty():
            return cls({}, fallback_observation_count)
        summary = run_summary
        if "outcome_horizon" in summary.columns:
            summary = summary.filter(pl.col("outcome_horizon") == int(outcome_horizon))
        if "summary_scope" in summary.columns:
            scoped = summary.filter(pl.col("summary_scope") == direction)
            summary = (
                scoped
                if not scoped.is_empty()
                else summary.filter(pl.col("summary_scope") == "run")
            )
        return cls(
            summary.row(0, named=True) if not summary.is_empty() else {},
            fallback_observation_count,
        )

    def integer(self, column: str, default: int = 0) -> int:
        value = self.row.get(column, default)
        return int(value) if value is not None else default

    def number(self, column: str, default: float = 0.0) -> float:
        value = self.row.get(column, default)
        return float(value) if value is not None else default

    @property
    def observation_row_count(self) -> int:
        return self.integer("observation_row_count", self.fallback_observation_count)

    @property
    def valid_observation_count(self) -> int:
        return self.integer("valid_observation_count", self.fallback_observation_count)


@dataclass(frozen=True)
class TailtreeCandidateReplay:
    """Budget replay over one horizon/direction candidate group."""

    eligible: pl.DataFrame
    summary: TailtreeSummaryView
    context: TailtreeSelectionContext
    outcome_horizon: int
    direction: str
    feasibility: TailtreeSelectionFeasibilityPolicy

    def row_for_budget(self, family: str, value: float) -> dict[str, object]:
        selected = self.selected_for_budget(family, value)
        selected_count = self.observation_count(selected)
        selected_symbol_count = self.symbol_count(selected)
        selected_tail_count = self.tail_count(selected)
        profit_values = self.profit_values(selected)
        utility_p90_values = self.utility_p90_values(selected)
        profit_sum = float(profit_values.sum() or 0.0) if not profit_values.is_empty() else 0.0
        profit_mean = float(profit_values.mean() or 0.0) if not profit_values.is_empty() else 0.0
        profit_p90 = (
            float(profit_values.quantile(0.9) or 0.0) if not profit_values.is_empty() else 0.0
        )
        utility_p90 = (
            float(utility_p90_values.quantile(0.9) or 0.0)
            if not utility_p90_values.is_empty()
            else 0.0
        )
        valid_count = self.summary.valid_observation_count
        valid_tail_rate = self.summary.number("valid_tail_rate")
        selected_rate = selected_count / valid_count if valid_count else 0.0
        selected_tail_rate = selected_tail_count / selected_count if selected_count else 0.0
        lift = selected_tail_rate / valid_tail_rate if valid_tail_rate > 0.0 else 0.0
        profit_per_selected = profit_mean
        profit_per_1k = profit_sum / valid_count * 1000.0 if valid_count else 0.0
        hpo_score = profit_per_selected + lift + log1p(max(selected_tail_count, 0)) - selected_rate
        support_pass = int(
            selected_count >= self.feasibility.min_selected_observation_count
            and selected_symbol_count >= self.feasibility.min_selected_symbol_count
            and selected_tail_count >= self.feasibility.min_selected_tail_count
        )
        concentration_pass = int(lift >= self.feasibility.min_valid_tail_lift)
        utility_pass = int(
            profit_per_selected >= self.feasibility.min_profit_proxy_per_selected_obs
        )
        return {
            "trial_id": self.context.trial_id,
            "trial_source": self.context.trial_source,
            "fold_id": self.context.fold_id,
            "evaluation_protocol": self.context.evaluation_protocol,
            "train_start_ms": self.context.train_start_ms,
            "train_end_ms": self.context.train_end_ms,
            "valid_start_ms": self.context.valid_start_ms,
            "valid_end_ms": self.context.valid_end_ms,
            "embargo_bars": self.context.embargo_bars,
            "universe_snapshot_id": str(self.context.universe_snapshot_id),
            "model_tag": str(self.context.model_tag),
            "objective": self.context.objective,
            "training_profile": str(self.context.training_profile),
            "outcome_label_family": "path_extreme_return",
            "comparison_surface": "selection_efficiency",
            "objective_score_comparable_int": 0,
            "outcome_horizon": int(self.outcome_horizon),
            "tree_direction": self.direction,
            "budget_family": family,
            "budget_value": float(value),
            "eligible_symbol_count": self.symbol_count(self.eligible),
            "selected_symbol_count": selected_symbol_count,
            "observation_row_count": self.summary.observation_row_count,
            "feature_count": self.summary.integer("feature_count"),
            "train_exceedance_count": self.summary.integer("train_exceedance_count"),
            "valid_observation_count": valid_count,
            "valid_tail_count": self.summary.integer("valid_tail_count"),
            "valid_tail_rate": valid_tail_rate,
            "selected_observation_count": selected_count,
            "selected_observation_rate": selected_rate,
            "selected_tail_count": selected_tail_count,
            "selected_tail_rate": selected_tail_rate,
            "selected_tail_per_1k_obs": selected_tail_rate * 1000.0,
            "valid_tail_lift": lift,
            "selected_profit_proxy_mean": profit_mean,
            "selected_profit_proxy_p90": profit_p90,
            "selected_utility_mean": profit_mean,
            "selected_utility_p90": utility_p90,
            "profit_proxy_per_selected_obs": profit_per_selected,
            "profit_proxy_per_1k_observed": profit_per_1k,
            "hpo_score": hpo_score,
            "promotion_threshold_pass_int": int(profit_per_selected > 0.0 and lift >= 1.0),
            "feasibility_support_pass_int": support_pass,
            "feasibility_concentration_pass_int": concentration_pass,
            "feasibility_utility_pass_int": utility_pass,
            "feasibility_pass_int": int(support_pass and concentration_pass and utility_pass),
            "trained_tree_count": self.summary.integer("trained_tree_count"),
            "selected_bucket_or_leaf_count": self.summary.integer("selected_leaf_count"),
            "fit_seconds": self.summary.number("fit_seconds"),
            "score_seconds": self.summary.number("score_seconds"),
        }

    def selected_for_budget(self, family: str, value: float) -> pl.DataFrame:
        if self.eligible.is_empty():
            return self.eligible
        sorted_candidates = self.eligible.with_columns(self.rank_expr().alias("_rank")).sort(
            "_rank", descending=True
        )
        if family == "top_k":
            return sorted_candidates.head(max(0, int(value)))
        if family == "top_pct":
            return sorted_candidates.head(max(1, ceil(len(sorted_candidates) * value)))
        if family == "score_gate":
            return sorted_candidates.filter(pl.col("_rank") >= value)
        return sorted_candidates.head(0)

    def rank_expr(self) -> pl.Expr:
        for column in ("rank_score", "tailtree_score", "tail_lift"):
            if column in self.eligible.columns:
                return pl.col(column).fill_null(0.0).fill_nan(0.0)
        return pl.lit(0.0)

    def profit_expr(self) -> pl.Expr:
        for column in ("selected_profit_proxy_mean", "tail_utility_mean", "rank_score"):
            if column in self.eligible.columns:
                return pl.col(column).fill_null(0.0).fill_nan(0.0)
        return pl.lit(0.0)

    def profit_values(self, selected: pl.DataFrame) -> pl.Series:
        if selected.is_empty():
            return pl.Series("profit_proxy", [], dtype=pl.Float64)
        return selected.select(self.profit_expr().alias("profit_proxy")).get_column("profit_proxy")

    def utility_p90_values(self, selected: pl.DataFrame) -> pl.Series:
        if selected.is_empty():
            return pl.Series("tail_utility_p90", [], dtype=pl.Float64)
        expr = (
            pl.col("tail_utility_p90").fill_null(0.0).fill_nan(0.0)
            if "tail_utility_p90" in selected.columns
            else self.profit_expr()
        )
        return selected.select(expr.alias("tail_utility_p90")).get_column("tail_utility_p90")

    def tail_count(self, selected: pl.DataFrame) -> int:
        if selected.is_empty() or "N_tail_exceedances" not in selected.columns:
            return 0
        return int(selected.get_column("N_tail_exceedances").fill_null(0).sum())

    def observation_count(self, selected: pl.DataFrame) -> int:
        if selected.is_empty():
            return 0
        if "N_total" in selected.columns:
            return int(selected.get_column("N_total").fill_null(0).sum())
        return len(selected)

    def symbol_count(self, frame: pl.DataFrame) -> int:
        return int(frame.get_column("symbol").n_unique()) if "symbol" in frame.columns else 0


def tailtree_selection_efficiency_frame(
    candidates: pl.DataFrame,
    *,
    context: TailtreeSelectionContext,
    run_summary: pl.DataFrame,
    policy: TailtreeSelectionPolicy = TailtreeSelectionPolicy(),
) -> pl.DataFrame:
    """Replay candidate budgets into canonical profit-selection efficiency rows."""
    if candidates.is_empty() or not {"outcome_horizon", "tree_direction"}.issubset(
        candidates.columns
    ):
        return pl.DataFrame(schema=TAILTREE_SELECTION_EFFICIENCY_SCHEMA)
    eligible = candidates
    if "candidate_status" in eligible.columns:
        eligible = eligible.filter(pl.col("candidate_status") == "matched_evidence")
    if eligible.is_empty():
        return pl.DataFrame(schema=TAILTREE_SELECTION_EFFICIENCY_SCHEMA)
    rows: list[dict[str, object]] = []
    for key, group in eligible.group_by(["outcome_horizon", "tree_direction"], maintain_order=True):
        outcome_horizon, direction = key if isinstance(key, tuple) else (key, "")
        replay = TailtreeCandidateReplay(
            eligible=group,
            summary=TailtreeSummaryView.from_frame(
                run_summary,
                outcome_horizon=int(outcome_horizon),
                direction=str(direction),
                fallback_observation_count=len(group),
            ),
            context=context,
            outcome_horizon=int(outcome_horizon),
            direction=str(direction),
            feasibility=policy.feasibility,
        )
        rows.extend(
            replay.row_for_budget(family, value) for family, value in policy.budgets.iter_rows()
        )
    return pl.DataFrame(rows, schema=TAILTREE_SELECTION_EFFICIENCY_SCHEMA)


def _normalized_component(column: str, group_cols: list[str]) -> pl.Expr:
    value = pl.col(column).fill_null(0.0).fill_nan(0.0)
    maximum = value.max().over(group_cols)
    return pl.when(maximum > 0.0).then(value / maximum).otherwise(0.0)


def _winner_score_expr(group_cols: list[str], *, has_feasibility: bool) -> pl.Expr:
    feasibility = (
        pl.col("feasibility_pass_int").fill_null(1).cast(pl.Float64)
        if has_feasibility
        else pl.lit(1.0)
    )
    return (
        feasibility
        + pl.col("promotion_threshold_pass_int").fill_null(0).cast(pl.Float64)
        + _normalized_component("profit_proxy_per_selected_obs", group_cols)
        + _normalized_component("profit_proxy_per_1k_observed", group_cols)
        + _normalized_component("valid_tail_lift", group_cols)
        + _normalized_component("selected_tail_count", group_cols)
        - pl.col("selected_observation_rate").fill_null(0.0).fill_nan(0.0)
    )


def select_tailtree_budget_winners(selection_efficiency: pl.DataFrame) -> pl.DataFrame:
    """Select one normalized opportunity-efficiency budget winner per model grain.

    The selector deliberately ignores raw ``hpo_score`` so one unbounded component cannot
    dominate winner choice. It compares only scanner opportunity metrics; execution,
    liquidity, cost, funding, and sizing are not part of this surface.
    """
    group_cols = [
        "trial_id",
        "fold_id",
        "model_tag",
        "objective",
        "training_profile",
        "outcome_horizon",
        "tree_direction",
    ]
    required = {
        *group_cols,
        "budget_family",
        "budget_value",
        "selected_observation_rate",
        "selected_tail_count",
        "valid_tail_lift",
        "profit_proxy_per_selected_obs",
        "profit_proxy_per_1k_observed",
        "promotion_threshold_pass_int",
    }
    if selection_efficiency.is_empty() or not required.issubset(selection_efficiency.columns):
        return selection_efficiency.head(0).with_columns(
            pl.lit(None, dtype=pl.Float64).alias("winner_score"),
            pl.lit(None, dtype=pl.Int64).alias("winner_rank"),
        )

    scored = selection_efficiency.with_columns(
        _winner_score_expr(
            group_cols,
            has_feasibility="feasibility_pass_int" in selection_efficiency.columns,
        ).alias("winner_score")
    )
    return (
        scored.sort(
            [
                *group_cols,
                "winner_score",
                "profit_proxy_per_selected_obs",
                "profit_proxy_per_1k_observed",
                "valid_tail_lift",
                "selected_observation_rate",
                "selected_symbol_count",
                "budget_family",
                "budget_value",
            ],
            descending=[
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                False,
                False,
                False,
                False,
            ],
        )
        .unique(subset=group_cols, keep="first", maintain_order=True)
        .with_columns(pl.lit(1, dtype=pl.Int64).alias("winner_rank"))
    )


def select_tailtree_objective_winners(selection_efficiency: pl.DataFrame) -> pl.DataFrame:
    group_cols = [
        "universe_snapshot_id",
        "evaluation_protocol",
        "fold_id",
        "outcome_label_family",
        "outcome_horizon",
        "tree_direction",
    ]
    required = {
        *group_cols,
        "model_tag",
        "objective",
        "training_profile",
        "budget_family",
        "budget_value",
        "selected_observation_rate",
        "selected_tail_count",
        "valid_tail_lift",
        "profit_proxy_per_selected_obs",
        "profit_proxy_per_1k_observed",
        "promotion_threshold_pass_int",
        "feasibility_pass_int",
    }
    if selection_efficiency.is_empty() or not required.issubset(selection_efficiency.columns):
        return selection_efficiency.head(0).with_columns(
            pl.lit(None, dtype=pl.Float64).alias("objective_winner_score"),
            pl.lit(None, dtype=pl.Int64).alias("objective_winner_rank"),
        )
    scored = selection_efficiency.with_columns(
        _winner_score_expr(group_cols, has_feasibility=True).alias("objective_winner_score")
    )
    return (
        scored.sort(
            [
                *group_cols,
                "feasibility_pass_int",
                "objective_winner_score",
                "profit_proxy_per_selected_obs",
                "profit_proxy_per_1k_observed",
                "valid_tail_lift",
                "selected_observation_rate",
                "objective",
                "training_profile",
                "budget_family",
                "budget_value",
            ],
            descending=[
                False,
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
            ],
        )
        .unique(subset=group_cols, keep="first", maintain_order=True)
        .with_columns(pl.lit(1, dtype=pl.Int64).alias("objective_winner_rank"))
    )


def tailtree_hpo_feedback_frame(selection_efficiency: pl.DataFrame) -> pl.DataFrame:
    group_cols = [
        "universe_snapshot_id",
        "evaluation_protocol",
        "fold_id",
        "outcome_label_family",
        "outcome_horizon",
        "tree_direction",
    ]
    required = {
        *group_cols,
        "objective",
        "training_profile",
        "budget_family",
        "budget_value",
        "selected_observation_rate",
        "selected_tail_count",
        "valid_tail_lift",
        "profit_proxy_per_selected_obs",
        "profit_proxy_per_1k_observed",
        "promotion_threshold_pass_int",
        "feasibility_pass_int",
    }
    if selection_efficiency.is_empty() or not required.issubset(selection_efficiency.columns):
        return selection_efficiency.head(0).with_columns(
            pl.lit(None, dtype=pl.String).alias("hpo_setting_id"),
            pl.lit(None, dtype=pl.Float64).alias("hpo_feedback_score"),
            pl.lit(None, dtype=pl.Float64).alias("hpo_feedback_margin_to_best"),
            pl.lit(None, dtype=pl.Int64).alias("hpo_feedback_rank"),
            pl.lit(None, dtype=pl.Int64).alias("hpo_feedback_selected_int"),
        )
    sorted_feedback = selection_efficiency.with_columns(
        _winner_score_expr(group_cols, has_feasibility=True).alias("hpo_feedback_score"),
        pl.concat_str(
            [
                pl.col("objective"),
                pl.col("training_profile"),
                pl.col("trial_id"),
                pl.col("budget_family"),
                pl.col("budget_value").cast(pl.String),
            ],
            separator="|",
        ).alias("hpo_setting_id"),
    ).sort(
        [
            *group_cols,
            "feasibility_pass_int",
            "hpo_feedback_score",
            "profit_proxy_per_selected_obs",
            "profit_proxy_per_1k_observed",
            "valid_tail_lift",
            "selected_observation_rate",
            "objective",
            "training_profile",
            "budget_family",
            "budget_value",
        ],
        descending=[
            False,
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
        ],
    )
    return sorted_feedback.with_columns(
        (pl.cum_count("hpo_feedback_score").over(group_cols))
        .cast(pl.Int64)
        .alias("hpo_feedback_rank"),
        (pl.col("hpo_feedback_score").max().over(group_cols) - pl.col("hpo_feedback_score"))
        .cast(pl.Float64)
        .alias("hpo_feedback_margin_to_best"),
    ).with_columns(
        (pl.col("hpo_feedback_rank") == 1).cast(pl.Int64).alias("hpo_feedback_selected_int")
    )


def write_tailtree_selection_efficiency(
    frame: pl.DataFrame,
    diagnostics_dir: Path | str,
    model_dir: Path | str,
) -> None:
    """Replace the canonical selection-efficiency artifact in diagnostics/model dirs."""
    diagnostics = Path(diagnostics_dir)
    model = Path(model_dir)
    diagnostics.mkdir(parents=True, exist_ok=True)
    model.mkdir(parents=True, exist_ok=True)
    selected = frame.select(
        [column for column in TAILTREE_SELECTION_EFFICIENCY_SCHEMA if column in frame.columns]
    )
    selected.write_csv(diagnostics / "tailtree-selection-efficiency.csv")
    selected.write_csv(model / "tailtree-selection-efficiency.csv")


__all__ = [
    "TAILTREE_SELECTION_EFFICIENCY_SCHEMA",
    "TailtreeSelectionBudgets",
    "TailtreeSelectionContext",
    "TailtreeSelectionFeasibilityPolicy",
    "TailtreeTrainingProfile",
    "TailtreeModelTag",
    "UniverseSnapshotId",
    "select_tailtree_budget_winners",
    "select_tailtree_objective_winners",
    "tailtree_hpo_feedback_frame",
    "tailtree_selection_efficiency_frame",
    "write_tailtree_selection_efficiency",
]
