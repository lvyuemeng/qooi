"""Composable scanner configuration models."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from qooi.sources.collect import BookMode

RefreshMode = Literal["incremental", "cache_only", "force"]
SourceRefreshMode = Literal["inherit", "incremental", "cache_only", "force"]
EvidenceKind = Literal["ladder", "tailtree"]
TailtreeLifecycle = Literal["train", "load_predict"]


class SourceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    refresh_mode: SourceRefreshMode = "inherit"
    disabled_sources: tuple[str, ...] = ()
    disabled_symbols: tuple[str, ...] = ()
    book_mode: BookMode = "snapshot"
    book_depth: int = 25
    max_staleness_hours: int = 24
    trade_limit: int = 100
    funding_limit: int = 100
    rubik_period: str = "1H"
    rubik_limit: int = 100
    rubik_taker_unit: Literal["0", "1", "2"] = "2"


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


class ReviewConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    require_context: bool = True


class TailtreeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    lifecycle: TailtreeLifecycle = "train"
    model_dir: Path = Path("data/output/potential/models")
    model_tag: str = "tailtree-current"
    threshold_pct: float = 5.0
    num_leaves: int = 64
    min_data_in_leaf: int = 30
    learning_rate: float = 0.05
    num_iterations: int = 200
    early_stopping_rounds: int = 20


class EvidenceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: EvidenceKind = "ladder"
    tailtree: TailtreeConfig = TailtreeConfig()


class PotentialConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    output: Path = Path("data/output/potential/report.md")
    symbols: tuple[str, ...] = ()
    universe: str = "research"
    bar: str = "1H"
    timeframes: tuple[str, ...] = ("1H", "4H", "1D")
    days: int = 60
    refresh_mode: RefreshMode = "incremental"
    fetch_concurrency: int = 3
    source: SourceConfig = SourceConfig()
    transition: TransitionConfig = TransitionConfig()
    review: ReviewConfig = ReviewConfig()
    evidence: EvidenceConfig = EvidenceConfig()

    @model_validator(mode="after")
    def normalize_paths_and_timeframes(self) -> PotentialConfig:
        output = (
            self.output
            if self.output.name == "report.md"
            else self.output / "potential" / "report.md"
        )
        timeframes = tuple(dict.fromkeys((*self.timeframes, self.bar)))
        if output == self.output and timeframes == self.timeframes:
            return self
        return self.model_copy(update={"output": output, "timeframes": timeframes})
