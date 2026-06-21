"""Composable scanner configuration models."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from qooi.profiling import ProfileConfig

RefreshMode = Literal["incremental", "cache_only", "force"]


# ── Product configs (3 shapes, 7 products, None = disabled) ──


class BarsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    timeframes: tuple[str, ...] = ("1H",)
    days: int = 60
    refresh_mode: RefreshMode = "incremental"
    latest_staleness_hours: int = 2


class SnapshotConfig(BaseModel):
    """Books, trades, funding. limit sets fetch depth."""

    model_config = ConfigDict(frozen=True)
    limit: int = 100
    max_staleness_hours: int | None = None


class RubikConfig(BaseModel):
    """Open interest, taker volume, long/short."""

    model_config = ConfigDict(frozen=True)
    period: str = "1H"
    limit: int = 100
    unit: Literal["0", "1", "2"] = "2"
    max_staleness_hours: int = 2


# ── Tailtree (unchanged below) ──

EvidenceKind = Literal["ladder", "tailtree"]
TailtreeLifecycle = Literal["train", "load_predict"]
TailtreeObjective = Literal[
    "tail_severity_gpd",
    "tail_utility_quantile",
    "tail_event_lift",
    "tail_any_event",
    "tail_side_only",
]
TailtreeTrainingKind = Literal["fixed", "optuna"]


class TransitionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    horizon: int = 12
    history_days: int = 0
    ngram_length: int = 3
    min_count: int = 20
    return_threshold_pct: float = 0.0
    min_information_bits: float = 0.001
    min_probability: float = 0.05
    min_directional_probability: float = 0.55
    min_reward_risk: float = 1.0
    max_tail_loss_pct: float = 20.0
    recent_window: int = 240
    long_window: int = 1440
    min_probability_delta: float = -0.10
    mae_mfe_horizon: int = 12
    context_scope: Literal["candidates", "all_scanned"] = "candidates"
    context_limit: int = 20
    scan_budget: int = 80

    @model_validator(mode="after")
    def normalize_transition_windows(self) -> TransitionConfig:
        history_days = max(0, self.history_days)
        ngram_length = max(2, self.ngram_length)
        if history_days == self.history_days and ngram_length == self.ngram_length:
            return self
        return self.model_copy(update={"history_days": history_days, "ngram_length": ngram_length})


class TailtreeSelectionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    top_k: tuple[int, ...] = (1, 3, 5, 10)
    top_pct: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20)
    score_gate: tuple[float, ...] = ()
    min_selected_observation_count: int = 10
    min_selected_symbol_count: int = 1
    min_selected_tail_count: int = 1
    min_valid_tail_lift: float = 1.0
    min_profit_proxy_per_selected_obs: float = 0.0

    @field_validator("top_k")
    @classmethod
    def positive_top_k(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(value for value in values if value > 0)

    @field_validator("top_pct")
    @classmethod
    def positive_top_pct(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(value for value in values if value > 0.0)


class TailtreeFixedTrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["fixed"] = "fixed"
    num_leaves: int = 64
    min_data_in_leaf: int = 80
    learning_rate: float = 0.03
    num_iterations: int = 240
    early_stopping_rounds: int = 30


class TailtreeOptunaTrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["optuna"] = "optuna"
    max_trials: int = 24
    seed: int = 42
    num_leaves: int = 64
    min_data_in_leaf: int = 80
    learning_rate: float = 0.03
    num_iterations: int = 240
    early_stopping_rounds: int = 30
    num_leaves_range: tuple[int, int] | None = None
    min_data_in_leaf_range: tuple[int, int] | None = None
    learning_rate_range: tuple[float, float] | None = None
    num_iterations_range: tuple[int, int] | None = None
    early_stopping_rounds_range: tuple[int, int] | None = None

    @field_validator(
        "num_leaves_range",
        "min_data_in_leaf_range",
        "num_iterations_range",
        "early_stopping_rounds_range",
    )
    @classmethod
    def positive_int_range(cls, value: tuple[int, int] | None) -> tuple[int, int] | None:
        if value is None:
            return None
        low, high = int(value[0]), int(value[1])
        if low <= 0 or high < low:
            raise ValueError("tailtree optuna int ranges must be positive [low, high]")
        return (low, high)

    @field_validator("learning_rate_range")
    @classmethod
    def positive_float_range(cls, value: tuple[float, float] | None) -> tuple[float, float] | None:
        if value is None:
            return None
        low, high = float(value[0]), float(value[1])
        if low <= 0.0 or high < low:
            raise ValueError("tailtree optuna float ranges must be positive [low, high]")
        return (low, high)


TailtreeTrainingConfig = TailtreeFixedTrainingConfig | TailtreeOptunaTrainingConfig


class TailtreeSingleSplitEvaluationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    protocol: Literal["single_split"] = "single_split"
    validation_fraction: float = 0.0
    embargo_bars: int = 0


class TailtreeWalkforwardEvaluationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    protocol: Literal["walkforward"] = "walkforward"
    train_days: int
    valid_days: int
    step_days: int
    max_folds: int
    embargo_bars: int = 0


TailtreeEvaluationConfig = TailtreeSingleSplitEvaluationConfig | TailtreeWalkforwardEvaluationConfig


class TailtreeProfileConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    profile_id: str
    model_tag: str
    objective: TailtreeObjective
    training: TailtreeTrainingConfig = TailtreeFixedTrainingConfig()
    evaluation: TailtreeEvaluationConfig = TailtreeSingleSplitEvaluationConfig()

    @field_validator("profile_id", "model_tag")
    @classmethod
    def nonempty_profile_field(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("tailtree profile_id/model_tag must not be empty")
        return cleaned


class TailtreeConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    lifecycle: TailtreeLifecycle = "train"
    model_dir: Path = Path("data/output/potential/models")
    threshold_pct: float = 5.0
    outcome_horizon: tuple[int, ...] = (12,)
    selection: TailtreeSelectionConfig = TailtreeSelectionConfig()
    profiles: tuple[TailtreeProfileConfig, ...] = (
        TailtreeProfileConfig(
            profile_id="gpd-balanced-fixed",
            model_tag="tailtree-current",
            objective="tail_severity_gpd",
            training=TailtreeFixedTrainingConfig(
                num_leaves=64,
                min_data_in_leaf=30,
                learning_rate=0.05,
                num_iterations=200,
                early_stopping_rounds=20,
            ),
        ),
    )

    @field_validator("outcome_horizon", mode="before")
    @classmethod
    def normalize_outcome_horizon(cls, value: object) -> tuple[int, ...]:
        if value is None:
            return (12,)
        if isinstance(value, int):
            values = (value,)
        elif isinstance(value, Iterable) and not isinstance(value, str | bytes):
            values = tuple(value)
        else:
            values = (value,)
        parsed_horizons = []
        for horizon in values:
            if not isinstance(horizon, int | float | str):
                continue
            horizon_int = int(horizon)
            if horizon_int > 0:
                parsed_horizons.append(horizon_int)
        horizons = tuple(dict.fromkeys(parsed_horizons))
        return horizons or (12,)

    @model_validator(mode="after")
    def require_unique_profile_ids(self) -> TailtreeConfig:
        profile_ids = [profile.profile_id for profile in self.profiles]
        if len(profile_ids) != len(set(profile_ids)):
            raise ValueError("tailtree profile_id values must be unique")
        return self


class EvidenceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: EvidenceKind = "ladder"
    tailtree: TailtreeConfig = TailtreeConfig()


class PotentialConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    output: Path = Path("data/output/potential/report.md")
    universe: str = "research"
    max_symbols: int = 25
    max_staleness_hours: int = 24
    fetch_concurrency: int = 3

    # 7 products, 3 shapes, None = off
    bars: BarsConfig | None = BarsConfig()
    books: SnapshotConfig | None = None
    trades: SnapshotConfig | None = None
    funding: SnapshotConfig | None = None
    open_interest: RubikConfig | None = None
    taker_volume: RubikConfig | None = None
    long_short: RubikConfig | None = None

    transition: TransitionConfig = TransitionConfig()
    evidence: EvidenceConfig = EvidenceConfig()
    profile: ProfileConfig = ProfileConfig()

    @model_validator(mode="after")
    def normalize_paths(self) -> PotentialConfig:
        output = (
            self.output
            if self.output.name == "report.md"
            else self.output / "potential" / "report.md"
        )
        if output == self.output:
            return self
        return self.model_copy(update={"output": output})
