"""Learned-state contracts, VQ-RSSM config, and state preparation."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import polars as pl
from pydantic import Field, field_validator, model_validator

from qooi.core.config import StrictConfigModel
from qooi.dynamic import vq_rssm
from qooi.dynamic.contracts import (
    SPLIT_NAMES,
    AssetFeatureSequence,
    CodeSequence,
    SequenceDataset,
    SplitName,
    WindowDataset,
)
from qooi.dynamic.training import TrainingConfig

LEARNED_STATE_FEATURE_COLUMNS = (
    "open_rel",
    "high_rel",
    "low_rel",
    "close_rel",
    "volume_log_rel",
)
VolatilityScalingMethod = Literal["ewm_std"]
LearnedStateReturnSplit = Literal["train", "valid", "test", "all"]
LearnedStatePhase = Literal["train", "predict", "evaluate"]
LearnedObjectiveTerm = Literal["reconstruct", "future_infonce"]
_ROW_INDEX = "__behavior_row_index"
_FREEZE_ENCODER_BLOCKS = {
    "encoder_input",
    "latent_projection",
    "rssm",
    "decoder",
    "future_head",
}

class FeatureColumns(StrictConfigModel):
    open: str = "open"
    high: str = "high"
    low: str = "low"
    close: str = "close"
    volume: str = "vol"
    timestamp: str = "timestamp"
    symbol: str = "symbol"
    split: str = "split"

    def required(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((self.open, self.high, self.low, self.close, self.volume)))


class VolatilityScalingConfig(StrictConfigModel):
    on: bool = False
    method: VolatilityScalingMethod = "ewm_std"
    ret: str = "close_rel"
    cols: tuple[str, ...] = ("open_rel", "high_rel", "low_rel", "close_rel")
    half: float = 10.0
    min_periods: int = 2
    floor: float = 1e-4
    cap: float = 0.10
    output_column: str = "volatility_scale"

    @property
    def enabled(self) -> bool:
        return self.on

    @property
    def return_column(self) -> str:
        return self.ret

    @property
    def columns(self) -> tuple[str, ...]:
        return self.cols

    @property
    def half_life(self) -> float:
        return self.half

    @model_validator(mode="after")
    def _valid_scaling(self) -> Self:
        if self.half_life <= 0.0 or not math.isfinite(self.half_life):
            raise ValueError("half_life must be a finite positive float")
        if self.min_periods <= 0:
            raise ValueError("min_periods must be positive")
        if self.floor <= 0.0 or not math.isfinite(self.floor):
            raise ValueError("floor must be a finite positive float")
        if self.cap <= 0.0 or not math.isfinite(self.cap):
            raise ValueError("cap must be a finite positive float")
        if self.floor > self.cap:
            raise ValueError("floor must be <= cap")
        if not self.output_column:
            raise ValueError("output_column must be non-empty")
        if not self.columns:
            raise ValueError("columns must be non-empty")
        return self


class LearnedStateEvaluationConfig(StrictConfigModel):
    h: tuple[int, ...] = ()
    split: LearnedStateReturnSplit = "test"
    cost: float = 5.0
    min: int = 1

    @property
    def horizons(self) -> tuple[int, ...]:
        return self.h

    @property
    def returns_split(self) -> LearnedStateReturnSplit:
        return self.split

    @property
    def transaction_cost_bps(self) -> float:
        return self.cost

    @property
    def min_rows(self) -> int:
        return self.min

    @field_validator("h")
    @classmethod
    def _horizons_positive(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(horizon <= 0 for horizon in value):
            raise ValueError("learned state evaluation horizons must be positive")
        return tuple(dict.fromkeys(value))

    @model_validator(mode="after")
    def _valid_evaluation(self) -> Self:
        if self.cost < 0.0 or not math.isfinite(self.cost):
            raise ValueError("transaction_cost_bps must be a finite non-negative float")
        if self.min <= 0:
            raise ValueError("min_rows must be positive")
        return self


class LearnedStateRunConfig(StrictConfigModel):
    phases: tuple[LearnedStatePhase, ...] = ("train", "predict", "evaluate")
    checkpoint: Path = Path("")
    states: Path = Path("")

    @field_validator("phases")
    @classmethod
    def _phases_non_empty(
        cls, value: tuple[LearnedStatePhase, ...]
    ) -> tuple[LearnedStatePhase, ...]:
        phases = tuple(dict.fromkeys(value))
        if not phases:
            raise ValueError("learn.run.phases must contain at least one phase")
        return phases


class FutureObjectiveConfig(StrictConfigModel):
    kind: Literal["infonce"] = "infonce"
    min_len: int = 4
    max_len: int = 32
    samples: int = 1
    dim: int = 32
    temperature: float = 0.1
    source: Literal["features"] = "features"
    length_policy: Literal["cycle"] = "cycle"
    warmup_epochs: int = 0
    detrend: bool = False
    detrend_half_life: float = 20.0

    @model_validator(mode="after")
    def _valid_future(self) -> Self:
        if self.min_len <= 0:
            raise ValueError("future min_len must be positive")
        if self.max_len < self.min_len:
            raise ValueError("future max_len must be >= min_len")
        if self.samples <= 0:
            raise ValueError("future samples must be positive")
        if self.dim <= 0:
            raise ValueError("future dim must be positive")
        if self.temperature <= 0.0 or not math.isfinite(self.temperature):
            raise ValueError("future temperature must be a finite positive float")
        if self.warmup_epochs < 0:
            raise ValueError("future warmup_epochs must be non-negative")
        if self.detrend_half_life <= 0.0 or not math.isfinite(self.detrend_half_life):
            raise ValueError("future detrend_half_life must be a finite positive float")
        return self


class FutureContrastConfig(StrictConfigModel):
    standard_weight: float = 1.0
    future_similarity_weight: float = 0.0
    future_similarity_top_k: int = 3
    future_similarity_mse_weight: float = 1.0
    future_similarity_cosine_weight: float = 0.0
    future_similarity_max_distance: float = 0.0
    temperature: float | None = None

    @model_validator(mode="after")
    def _valid_future_contrast(self) -> Self:
        for name in (
            "standard_weight",
            "future_similarity_weight",
            "future_similarity_mse_weight",
            "future_similarity_cosine_weight",
            "future_similarity_max_distance",
        ):
            value = getattr(self, name)
            if value < 0.0 or not math.isfinite(value):
                raise ValueError(f"future_contrast {name} must be finite and non-negative")
        if self.future_similarity_top_k <= 0:
            raise ValueError("future_contrast future_similarity_top_k must be positive")
        if self.temperature is not None and (
            self.temperature <= 0.0 or not math.isfinite(self.temperature)
        ):
            raise ValueError("future_contrast temperature must be a finite positive float")
        if self.future_similarity_weight > 0.0 and (
            self.future_similarity_mse_weight + self.future_similarity_cosine_weight <= 0.0
        ):
            raise ValueError(
                "future_contrast future_similarity_weight requires a positive metric weight"
            )
        return self


class LearnedObjectiveConfig(StrictConfigModel):
    terms: tuple[LearnedObjectiveTerm, ...] = ("reconstruct",)
    reconstruct: float = 1.0
    future: float = 0.0
    future_config: FutureObjectiveConfig = Field(default_factory=FutureObjectiveConfig)

    @model_validator(mode="before")
    @classmethod
    def _reject_kl_term(cls, data):
        if not isinstance(data, dict):
            return data
        terms = tuple(data.get("terms") or ())
        if "kl" in terms:
            raise ValueError(
                "learn.objective.terms no longer supports 'kl'; "
                "pure-discrete RSSM has no Gaussian prior"
            )
        if "kl" in data:
            raise ValueError("learn.objective.kl is removed; pure-discrete RSSM has no KL term")
        return data

    @field_validator("terms")
    @classmethod
    def _terms_non_empty(
        cls, value: tuple[LearnedObjectiveTerm, ...]
    ) -> tuple[LearnedObjectiveTerm, ...]:
        terms = tuple(dict.fromkeys(value))
        if not terms:
            raise ValueError("learn.objective.terms must contain at least one term")
        return terms

    @model_validator(mode="after")
    def _valid_weights(self) -> Self:
        for name in ("reconstruct", "future"):
            value = getattr(self, name)
            if value < 0.0 or not math.isfinite(value):
                raise ValueError(f"objective {name} weight must be finite and non-negative")
        return self


class LearnedScheduleStageConfig(StrictConfigModel):
    name: str = ""
    start_epoch: int = 1
    epochs: int = 1
    reconstruct: float | None = None
    future: float | None = None
    diversity_weight: float | None = None
    reset_fraction: float | None = None
    reset_dead_codes: bool | None = None
    freeze_encoder_blocks: tuple[str, ...] = ()
    lr: float | None = None

    @field_validator("freeze_encoder_blocks")
    @classmethod
    def _freeze_blocks_supported(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        blocks = tuple(dict.fromkeys(value))
        removed = sorted(set(blocks) & {"posterior", "prior"})
        if removed:
            raise ValueError(
                "freeze block '"
                + removed[0]
                + "' is removed; use 'latent_projection' if you intend to freeze state assignment"
            )
        invalid = sorted(set(blocks) - _FREEZE_ENCODER_BLOCKS)
        if invalid:
            raise ValueError("unknown freeze encoder blocks: " + ", ".join(invalid))
        return blocks

    @model_validator(mode="after")
    def _valid_schedule_stage(self) -> Self:
        if self.start_epoch <= 0:
            raise ValueError("schedule stage start_epoch must be positive")
        if self.epochs <= 0:
            raise ValueError("schedule stage epochs must be positive")
        for name in ("reconstruct", "future", "diversity_weight", "reset_fraction"):
            value = getattr(self, name)
            if value is not None and (value < 0.0 or not math.isfinite(value)):
                raise ValueError(f"schedule stage {name} must be finite and non-negative")
        if self.reset_fraction is not None and not 0.0 < self.reset_fraction <= 1.0:
            raise ValueError("schedule stage reset_fraction must satisfy 0 < fraction <= 1")
        if self.lr is not None and (self.lr <= 0.0 or not math.isfinite(self.lr)):
            raise ValueError("schedule stage lr must be a finite positive float")
        return self


class LearnedScheduleConfig(StrictConfigModel):
    enabled: bool = False
    stage: tuple[LearnedScheduleStageConfig, ...] = ()

    @property
    def stages(self) -> tuple[LearnedScheduleStageConfig, ...]:
        return self.stage if self.enabled else ()


class StabilityConfig(StrictConfigModel):
    latent_smoothness_weight: float = 0.0
    latent_smoothness_margin: float = 0.0

    @model_validator(mode="after")
    def _valid_latent_smoothness(self) -> Self:
        for name in ("latent_smoothness_weight", "latent_smoothness_margin"):
            value = getattr(self, name)
            if value < 0.0 or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and non-negative")
            if value > 0.0:
                raise ValueError(
                    f"learn.stability.{name} is removed; pure-discrete RSSM has no "
                    "continuous latent smoothing path"
                )
        return self


class TemporalConsistencyConfig(StrictConfigModel):
    weight: float = 0.0
    temperature: float = 1.0

    @model_validator(mode="after")
    def _valid_temporal_consistency(self) -> Self:
        if not 0.0 <= self.weight <= 0.01 or not math.isfinite(self.weight):
            raise ValueError("temporal_consistency weight must satisfy 0 <= weight <= 0.01")
        if self.temperature <= 0.0 or not math.isfinite(self.temperature):
            raise ValueError("temporal_consistency temperature must be a finite positive float")
        return self


class LearnedStatePostprocessConfig(StrictConfigModel):
    min_state_duration: int = 0

    @model_validator(mode="after")
    def _valid_min_state_duration(self) -> Self:
        if self.min_state_duration < 0:
            raise ValueError("postprocess min_state_duration must be non-negative")
        return self


class WindowConfig(StrictConfigModel):
    len: int = 100
    stride: int = 1
    eps: float = 1e-8
    use_global_zscore: bool = False
    train: float = 0.7
    valid: float = 0.2

    @property
    def seq_len(self) -> int:
        return self.len

    @property
    def train_pct(self) -> float:
        return self.train

    @property
    def valid_pct(self) -> float:
        return self.valid

    @model_validator(mode="after")
    def _valid_window(self) -> WindowConfig:
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.eps <= 0.0 or not math.isfinite(self.eps):
            raise ValueError("eps must be a finite positive float")
        if not 0.0 < self.train_pct < 1.0:
            raise ValueError("train_pct must be between 0 and 1")
        if not 0.0 <= self.valid_pct < 1.0:
            raise ValueError("valid_pct must be between 0 and 1")
        if self.train_pct + self.valid_pct >= 1.0:
            raise ValueError("train_pct + valid_pct must be less than 1")
        return self

    def required_columns(self, columns: FeatureColumns) -> tuple[str, ...]:
        return columns.required()

    def features(
        self,
        frame: pl.DataFrame,
        columns: FeatureColumns,
        volatility_scaling: VolatilityScalingConfig | None = None,
    ) -> pl.DataFrame:
        if frame.is_empty():
            return frame
        _require_columns(frame, columns.required())
        work = frame.with_row_index(_ROW_INDEX) if _ROW_INDEX not in frame.columns else frame
        prev_close = pl.col(columns.close).shift(1)
        prev_volume = pl.col(columns.volume).shift(1)
        features = (
            work.with_columns(
                (pl.col(columns.open) / prev_close - 1.0).alias("open_rel"),
                (pl.col(columns.high) / prev_close - 1.0).alias("high_rel"),
                (pl.col(columns.low) / prev_close - 1.0).alias("low_rel"),
                (pl.col(columns.close) / prev_close - 1.0).alias("close_rel"),
                ((pl.col(columns.volume) + self.eps) / (prev_volume + self.eps))
                .log()
                .alias("volume_log_rel"),
            )
            .filter(
                pl.all_horizontal(
                    [pl.col(column).is_finite() for column in LEARNED_STATE_FEATURE_COLUMNS]
                )
            )
            .with_columns(pl.col(_ROW_INDEX).cast(pl.Int64))
        )
        if volatility_scaling is None or not volatility_scaling.enabled:
            return features
        return _apply_volatility_scaling(
            features,
            columns=columns,
            scaling=volatility_scaling,
        )

    def split(self, row_count: int) -> Split:
        if row_count < 0:
            raise ValueError("row_count must be non-negative")
        train_end = int(row_count * self.train_pct)
        valid_end = train_end + int(row_count * self.valid_pct)
        return Split(train_end=train_end, valid_end=valid_end)

    def assign_split(
        self,
        frame: pl.DataFrame,
        split: Split,
        columns: FeatureColumns,
    ) -> pl.DataFrame:
        if split.valid_end > frame.height:
            raise ValueError("valid_end must be <= frame height")
        return frame.with_row_index("__split_row").with_columns(
            pl.when(pl.col("__split_row") < split.train_end)
            .then(pl.lit("train"))
            .when(pl.col("__split_row") < split.valid_end)
            .then(pl.lit("valid"))
            .otherwise(pl.lit("test"))
            .alias(columns.split)
        ).drop("__split_row")

    def windows(
        self,
        frame: pl.DataFrame,
        columns: FeatureColumns,
        feature_columns: tuple[str, ...] = LEARNED_STATE_FEATURE_COLUMNS,
        volatility_scale_column: str = "volatility_scale",
    ) -> tuple[PreparedWindows, WindowProvenance]:
        _require_columns(frame, feature_columns)
        if columns.split not in frame.columns:
            raise ValueError("frame must contain a split column")
        work = frame.with_row_index(_ROW_INDEX) if _ROW_INDEX not in frame.columns else frame
        feature_rows = work.select(feature_columns).rows()
        timestamps = _timestamp_values(work, columns.timestamp)
        symbols = (
            work.get_column(columns.symbol).cast(pl.Utf8).to_list()
            if columns.symbol in work.columns
            else [""] * work.height
        )
        splits = tuple(work.get_column(columns.split).cast(pl.Utf8).to_list())
        _validate_splits(splits)
        source_row_index = work.get_column(_ROW_INDEX).to_list()
        windows = []
        row_index = []
        window_timestamps = []
        window_symbols = []
        window_splits = []
        window_scales = []
        scale_values = (
            work.get_column(volatility_scale_column).to_list()
            if volatility_scale_column in work.columns
            else None
        )
        for end in range(self.seq_len - 1, len(feature_rows), self.stride):
            start = end - self.seq_len + 1
            window = tuple(
                tuple(float(value) for value in row) for row in feature_rows[start : end + 1]
            )
            windows.append(window)
            row_index.append(int(source_row_index[end]))
            window_timestamps.append(int(timestamps[end]))
            window_symbols.append(str(symbols[end]))
            window_splits.append(_split_name(splits[end]))
            if scale_values is not None:
                scale = scale_values[end]
                window_scales.append(float(scale) if scale is not None else None)
        prepared = PreparedWindows(
            features=tuple(windows),
            splits=tuple(window_splits),
            feature_columns=feature_columns,
            seq_len=self.seq_len,
            stride=self.stride,
        )
        provenance = WindowProvenance(
            row_index=tuple(row_index),
            timestamps=tuple(window_timestamps),
            symbols=tuple(window_symbols),
            splits=tuple(window_splits),
            volatility_scale=tuple(window_scales),
        )
        if len(prepared.features) != len(provenance.row_index):
            raise ValueError("prepared window and provenance counts must match")
        return prepared, provenance

    def sequences(
        self,
        frame: pl.DataFrame,
        columns: FeatureColumns,
        seq_config: SequenceConfig,
        feature_columns: tuple[str, ...] = LEARNED_STATE_FEATURE_COLUMNS,
        volatility_scale_column: str = "volatility_scale",
    ) -> tuple[PreparedSequences, WindowProvenance]:
        _require_columns(frame, feature_columns)
        if columns.split not in frame.columns:
            raise ValueError("frame must contain a split column")
        work = frame.with_row_index(_ROW_INDEX) if _ROW_INDEX not in frame.columns else frame
        scale_values = (
            work.get_column(volatility_scale_column).to_list()
            if volatility_scale_column in work.columns
            else None
        )
        sequences = []
        emit_row_index = []
        emit_timestamps = []
        emit_symbols = []
        emit_splits = []
        emit_scales = []
        for split in SPLIT_NAMES:
            part = work.filter(pl.col(columns.split) == split)
            if part.is_empty():
                continue
            feature_rows = tuple(
                tuple(float(value) for value in row)
                for row in part.select(feature_columns).rows()
            )
            source_row_index = tuple(int(value) for value in part.get_column(_ROW_INDEX).to_list())
            timestamps = tuple(_timestamp_values(part, columns.timestamp))
            symbols = (
                tuple(part.get_column(columns.symbol).cast(pl.Utf8).to_list())
                if columns.symbol in part.columns
                else ("",) * part.height
            )
            symbol = str(symbols[0]) if symbols else ""
            scales = (
                tuple(
                    float(value) if value is not None else None
                    for value in part.get_column(volatility_scale_column).to_list()
                )
                if scale_values is not None
                else ()
            )
            sequences.append(
                AssetFeatureSequence(
                    symbol=symbol,
                    split=split,
                    features=feature_rows,
                    row_index=source_row_index,
                    timestamps=timestamps,
                    volatility_scale=scales,
                )
            )
            for offset in range(seq_config.warmup - 1, part.height, seq_config.stride):
                emit_row_index.append(source_row_index[offset])
                emit_timestamps.append(timestamps[offset])
                emit_symbols.append(symbols[offset])
                emit_splits.append(split)
                if scales:
                    emit_scales.append(scales[offset])
        prepared = PreparedSequences(
            sequences=tuple(sequences),
            feature_columns=feature_columns,
            chunk=seq_config.chunk,
            warmup=seq_config.warmup,
            stride=seq_config.stride,
            carry=seq_config.carry,
        )
        provenance = WindowProvenance(
            row_index=tuple(emit_row_index),
            timestamps=tuple(emit_timestamps),
            symbols=tuple(str(symbol) for symbol in emit_symbols),
            splits=tuple(_split_name(split) for split in emit_splits),
            volatility_scale=tuple(emit_scales),
        )
        return prepared, provenance


class SequenceConfig(StrictConfigModel):
    chunk: int = 256
    warmup: int = 64
    stride: int = 1
    carry: bool = True
    reset: Literal["asset_split"] = "asset_split"

    @model_validator(mode="after")
    def _valid_sequence(self) -> SequenceConfig:
        if self.chunk <= 0:
            raise ValueError("chunk must be positive")
        if self.warmup <= 0:
            raise ValueError("warmup must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        return self


class VqRssmConfig(StrictConfigModel):
    input_dim: int | None = None
    hidden: int = 128
    latent: int = 16
    codes: int = 128
    ema_decay: float = 0.99
    ema_eps: float = 1e-5
    normalize: bool = False
    reset_dead_codes: bool = False
    reset_interval: int = 500
    reset_threshold: float = 0.1
    reset_fraction: float = 0.1
    reset_warmup_epochs: int = 5
    reset_candidate_similarity_max: float = 0.9
    diversity_weight: float = 0.0
    diversity_margin: float = 0.5
    eps: float = 1e-8

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_kl_steps(cls, data):
        if isinstance(data, dict) and "kl_steps" in data:
            raise ValueError("learn.vq.kl_steps is removed; pure-discrete RSSM has no KL annealing")
        return data

    @property
    def hidden_dim(self) -> int:
        return self.hidden

    @property
    def latent_dim(self) -> int:
        return self.latent

    @property
    def num_codes(self) -> int:
        return self.codes

    @model_validator(mode="after")
    def _valid_ema(self) -> Self:
        if self.codes <= 0:
            raise ValueError("codes must be positive")
        if not 0.0 <= self.ema_decay < 1.0 or not math.isfinite(self.ema_decay):
            raise ValueError("ema_decay must be finite and satisfy 0 <= ema_decay < 1")
        if self.ema_eps <= 0.0 or not math.isfinite(self.ema_eps):
            raise ValueError("ema_eps must be a finite positive float")
        if self.reset_interval <= 0:
            raise ValueError("reset_interval must be positive")
        if not 0.0 < self.reset_threshold < 1.0 or not math.isfinite(self.reset_threshold):
            raise ValueError("reset_threshold must be finite and satisfy 0 < threshold < 1")
        if not 0.0 < self.reset_fraction <= 1.0 or not math.isfinite(self.reset_fraction):
            raise ValueError("reset_fraction must be finite and satisfy 0 < fraction <= 1")
        if self.reset_warmup_epochs < 0:
            raise ValueError("reset_warmup_epochs must be non-negative")
        if not -1.0 <= self.reset_candidate_similarity_max <= 1.0 or not math.isfinite(
            self.reset_candidate_similarity_max
        ):
            raise ValueError(
                "reset_candidate_similarity_max must be finite and satisfy -1 <= value <= 1"
            )
        if self.reset_dead_codes and not self.normalize:
            raise ValueError("reset_dead_codes requires normalize=true")
        if self.diversity_weight < 0.0 or not math.isfinite(self.diversity_weight):
            raise ValueError("diversity_weight must be a finite non-negative float")
        if not -1.0 <= self.diversity_margin < 1.0 or not math.isfinite(
            self.diversity_margin
        ):
            raise ValueError("diversity_margin must be finite and satisfy -1 <= margin < 1")
        if self.diversity_weight > 0.0 and not self.normalize:
            raise ValueError("diversity_weight requires normalize=true")
        return self


class LearnedStateConfig(StrictConfigModel):
    input: Literal["window", "sequence"] = "window"
    model: Literal["vq-rssm"] = "vq-rssm"
    state: str = "behavior_state_id"
    timeframe: str = "1H"
    out: Path = Path("data/output/learned-states/vq-rssm")
    run: LearnedStateRunConfig = Field(default_factory=LearnedStateRunConfig)
    objective: LearnedObjectiveConfig = Field(default_factory=LearnedObjectiveConfig)
    future_contrast: FutureContrastConfig = Field(default_factory=FutureContrastConfig)
    temporal_consistency: TemporalConsistencyConfig = Field(
        default_factory=TemporalConsistencyConfig
    )
    stability: StabilityConfig = Field(default_factory=StabilityConfig)
    postprocess: LearnedStatePostprocessConfig = Field(
        default_factory=LearnedStatePostprocessConfig
    )
    schedule: LearnedScheduleConfig = Field(default_factory=LearnedScheduleConfig)
    feature_columns: tuple[str, ...] = LEARNED_STATE_FEATURE_COLUMNS
    columns: FeatureColumns = Field(default_factory=FeatureColumns)
    scale: VolatilityScalingConfig = Field(default_factory=VolatilityScalingConfig)
    win: WindowConfig = Field(default_factory=WindowConfig)
    seq: SequenceConfig = Field(default_factory=SequenceConfig)
    vq: VqRssmConfig = Field(default_factory=VqRssmConfig)
    train: TrainingConfig = Field(default_factory=TrainingConfig)
    eval: LearnedStateEvaluationConfig = Field(default_factory=LearnedStateEvaluationConfig)

    @model_validator(mode="after")
    def _valid_future_contrast_with_objective(self) -> Self:
        if (
            "future_infonce" in self.objective.terms
            and self.objective.future > 0.0
            and self.future_contrast.standard_weight
            + self.future_contrast.future_similarity_weight
            <= 0.0
        ):
            raise ValueError(
                "learn.future_contrast requires a positive component weight "
                "when future objective is enabled"
            )
        return self

    @field_validator("feature_columns")
    @classmethod
    def _feature_columns_non_empty(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(value)) or LEARNED_STATE_FEATURE_COLUMNS

    @property
    def state_column(self) -> str:
        return self.state

    @property
    def checkpoint_dir(self) -> Path:
        return self.out

    @property
    def volatility_scaling(self) -> VolatilityScalingConfig:
        return self.scale

    @property
    def window(self) -> WindowConfig:
        return self.win

    @property
    def sequence(self) -> SequenceConfig:
        return self.seq

    @property
    def vq_rssm(self) -> VqRssmConfig:
        return self.vq

    @property
    def evaluation(self) -> LearnedStateEvaluationConfig:
        return self.eval

    def required_columns(self) -> tuple[str, ...]:
        return self.window.required_columns(self.columns)

    def checkpoint_path(self) -> Path:
        return self.out / "behavior-state-model.pt"

    def run_checkpoint_path(self) -> Path:
        return self.run.checkpoint if self.run.checkpoint != Path(".") else self.checkpoint_path()

    def run_states_path(self) -> Path:
        if self.run.states != Path("."):
            return self.run.states
        return self.out / "behavior-state-sequence.csv"

    def prepare(self, frame: pl.DataFrame):
        from qooi.dynamic.state import PreparedStateDiscovery

        return PreparedStateDiscovery.from_frame(frame, self)

    def prepare_many(self, frames: Iterable[pl.DataFrame]):
        from qooi.dynamic.state import PreparedStateDiscovery

        return PreparedStateDiscovery.from_frames(tuple(frames), self)

def summarize_hidden(
    states: StateSequence,
    diagnostics: vq_rssm.InferenceDiagnostics,
) -> pl.DataFrame:
    if not diagnostics.hidden_states:
        return pl.DataFrame()
    rows = []
    for index, state in enumerate(states.frame.get_column(states.state_column).to_list()):
        row = {states.state_column: int(state), "rows": 1}
        for dim, value in enumerate(diagnostics.hidden_states[index]):
            row[f"h_{dim}_mean"] = value
        rows.append(row)
    frame = pl.DataFrame(rows)
    return frame.group_by(states.state_column).agg(
        pl.len().alias("rows"),
        *[
            pl.col(column).mean().alias(column)
            for column in frame.columns
            if column.endswith("_mean")
        ],
    )


def mean_hidden(diagnostics: vq_rssm.InferenceDiagnostics) -> tuple[float, ...]:
    if not diagnostics.hidden_states:
        return ()
    width = len(diagnostics.hidden_states[0])
    return tuple(
        sum(row[index] for row in diagnostics.hidden_states) / len(diagnostics.hidden_states)
        for index in range(width)
    )


def project_codebook(
    checkpoint: vq_rssm.VqRssmCheckpoint,
    feature_cols: tuple[str, ...],
    hidden_refs: tuple[tuple[str, tuple[float, ...] | None], ...],
) -> pl.DataFrame:
    rows = []
    for label, hidden in hidden_refs:
        for state_id, values in enumerate(vq_rssm.decode_codebook(checkpoint, hidden_state=hidden)):
            row: dict[str, object] = {
                "behavior_state_id": state_id,
                "hidden_reference": label,
                "shape_note": "normalized feature reconstruction, not absolute OHLC",
            }
            row.update({column: value for column, value in zip(feature_cols, values, strict=False)})
            rows.append(row)
    return pl.DataFrame(rows)


def summarize_morph(prepared, states: StateSequence) -> pl.DataFrame:
    if getattr(prepared.config, "input", "window") == "sequence":
        return _summarize_sequence_morph(prepared, states)
    rows = []
    state_values = states.frame.get_column(states.state_column).to_list()
    for window_index, window in enumerate(prepared.windows.features):
        state = int(state_values[window_index])
        for step, values in enumerate(window):
            row = {states.state_column: state, "step": step, "rows": 1}
            row.update(
                {
                    column: value
                    for column, value in zip(prepared.windows.feature_columns, values, strict=False)
                }
            )
            rows.append(row)
    if not rows:
        return pl.DataFrame()
    frame = pl.DataFrame(rows)
    value_columns = list(prepared.windows.feature_columns)
    return frame.group_by(states.state_column, "step").agg(
        pl.len().alias("rows"),
        *[pl.col(column).mean().alias(column) for column in value_columns],
    )


def _summarize_sequence_morph(prepared, states: StateSequence) -> pl.DataFrame:
    rows = []
    state_values = states.frame.get_column(states.state_column).to_list()
    state_index = 0
    warmup = prepared.sequences.warmup
    stride = prepared.sequences.stride
    for sequence in prepared.sequences.sequences:
        for end in range(warmup - 1, len(sequence.features), stride):
            if state_index >= len(state_values):
                break
            state = int(state_values[state_index])
            state_index += 1
            start = end - warmup + 1
            for step, values in enumerate(sequence.features[start : end + 1]):
                row = {states.state_column: state, "step": step, "rows": 1}
                row.update(
                    {
                        column: value
                        for column, value in zip(
                            prepared.sequences.feature_columns, values, strict=False
                        )
                    }
                )
                rows.append(row)
    if not rows:
        return pl.DataFrame()
    frame = pl.DataFrame(rows)
    value_columns = list(prepared.sequences.feature_columns)
    return frame.group_by(states.state_column, "step").agg(
        pl.len().alias("rows"),
        *[pl.col(column).mean().alias(column) for column in value_columns],
    )


@dataclass(frozen=True)
class Split:
    train_end: int
    valid_end: int

    def __post_init__(self) -> None:
        if self.train_end < 0 or self.valid_end < 0:
            raise ValueError("split boundaries must be non-negative")
        if self.train_end > self.valid_end:
            raise ValueError("train_end must be <= valid_end")


@dataclass(frozen=True)
class PreparedWindows:
    features: tuple[tuple[tuple[float, ...], ...], ...]
    splits: tuple[SplitName, ...]
    feature_columns: tuple[str, ...]
    seq_len: int
    stride: int

    def __post_init__(self) -> None:
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if not self.feature_columns:
            raise ValueError("feature_columns must be non-empty")
        if not self.features:
            raise ValueError("features must contain at least one window")
        if len(self.features) != len(self.splits):
            raise ValueError("splits length must match feature window count")
        _validate_splits(self.splits)
        feature_width = len(self.feature_columns)
        for window in self.features:
            if len(window) != self.seq_len:
                raise ValueError("each feature window must match seq_len")
            for row in window:
                if len(row) != feature_width:
                    raise ValueError("each feature row must match feature_columns width")
                if any(not math.isfinite(value) for value in row):
                    raise ValueError("feature values must be finite floats")

    def to_dataset(self) -> WindowDataset:
        return WindowDataset(
            features=self.features,
            feature_columns=self.feature_columns,
            splits=self.splits,
            seq_len=self.seq_len,
            stride=self.stride,
        )

    @classmethod
    def concat(cls, windows: Iterable[PreparedWindows]) -> PreparedWindows:
        items = tuple(windows)
        non_empty = [item for item in items if item.features]
        if not non_empty:
            raise ValueError("windows must contain at least one prepared window set")
        first = non_empty[0]
        if any(item.feature_columns != first.feature_columns for item in non_empty):
            raise ValueError("all prepared window sets must use identical feature columns")
        if any(item.seq_len != first.seq_len for item in non_empty):
            raise ValueError("all prepared window sets must use identical seq_len")
        if any(item.stride != first.stride for item in non_empty):
            raise ValueError("all prepared window sets must use identical stride")
        return cls(
            features=tuple(window for item in non_empty for window in item.features),
            splits=tuple(split for item in non_empty for split in item.splits),
            feature_columns=first.feature_columns,
            seq_len=first.seq_len,
            stride=first.stride,
        )


@dataclass(frozen=True)
class PreparedSequences:
    sequences: tuple[AssetFeatureSequence, ...]
    feature_columns: tuple[str, ...]
    chunk: int
    warmup: int
    stride: int
    carry: bool

    def __post_init__(self) -> None:
        if not self.sequences:
            raise ValueError("sequences must contain at least one asset/split sequence")
        if not self.feature_columns:
            raise ValueError("feature_columns must be non-empty")
        if self.chunk <= 0:
            raise ValueError("chunk must be positive")
        if self.warmup <= 0:
            raise ValueError("warmup must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")

    def to_dataset(self) -> SequenceDataset:
        return SequenceDataset(
            sequences=self.sequences,
            feature_columns=self.feature_columns,
        )

    @classmethod
    def concat(cls, sequences: Iterable[PreparedSequences]) -> PreparedSequences:
        items = tuple(sequences)
        non_empty = [item for item in items if item.sequences]
        if not non_empty:
            raise ValueError("sequences must contain at least one prepared sequence set")
        first = non_empty[0]
        if any(item.feature_columns != first.feature_columns for item in non_empty):
            raise ValueError("all prepared sequence sets must use identical feature columns")
        if any(item.chunk != first.chunk for item in non_empty):
            raise ValueError("all prepared sequence sets must use identical chunk")
        if any(item.warmup != first.warmup for item in non_empty):
            raise ValueError("all prepared sequence sets must use identical warmup")
        if any(item.stride != first.stride for item in non_empty):
            raise ValueError("all prepared sequence sets must use identical stride")
        if any(item.carry != first.carry for item in non_empty):
            raise ValueError("all prepared sequence sets must use identical carry policy")
        return cls(
            sequences=tuple(sequence for item in non_empty for sequence in item.sequences),
            feature_columns=first.feature_columns,
            chunk=first.chunk,
            warmup=first.warmup,
            stride=first.stride,
            carry=first.carry,
        )


@dataclass(frozen=True)
class WindowProvenance:
    row_index: tuple[int, ...]
    timestamps: tuple[int, ...]
    symbols: tuple[str, ...]
    splits: tuple[SplitName, ...]
    volatility_scale: tuple[float | None, ...] = ()

    def __post_init__(self) -> None:
        lengths = {len(self.row_index), len(self.timestamps), len(self.symbols), len(self.splits)}
        if len(lengths) != 1:
            raise ValueError("row_index, timestamps, symbols, and splits must have equal lengths")
        if self.volatility_scale and len(self.volatility_scale) != len(self.row_index):
            raise ValueError("volatility_scale length must match window provenance count")
        _validate_splits(self.splits)

    @classmethod
    def concat(cls, provenances: Iterable[WindowProvenance]) -> WindowProvenance:
        items = tuple(provenances)
        non_empty = [item for item in items if item.row_index]
        if not non_empty:
            return cls(row_index=(), timestamps=(), symbols=(), splits=())
        include_scale = any(item.volatility_scale for item in non_empty)
        return cls(
            row_index=tuple(value for item in non_empty for value in item.row_index),
            timestamps=tuple(value for item in non_empty for value in item.timestamps),
            symbols=tuple(value for item in non_empty for value in item.symbols),
            splits=tuple(value for item in non_empty for value in item.splits),
            volatility_scale=tuple(
                value
                for item in non_empty
                for value in (
                    item.volatility_scale
                    if item.volatility_scale
                    else (None,) * len(item.row_index)
                )
            )
            if include_scale
            else (),
        )

    def states_from_codes(
        self,
        codes: CodeSequence,
        *,
        state_column: str = "behavior_state_id",
    ) -> StateSequence:
        if len(codes.codes) != len(self.row_index):
            raise ValueError("code sequence length must match window provenance count")
        if codes.row_index != tuple(range(len(codes.codes))):
            raise ValueError("code sequence row_index must be contiguous dataset positions")
        if codes.splits != self.splits:
            raise ValueError("code sequence splits do not match window provenance")
        return StateSequence(
            pl.DataFrame(
                {
                    "row_index": self.row_index,
                    "timestamp": self.timestamps,
                    "symbol": self.symbols,
                    "split": self.splits,
                    state_column: codes.codes,
                    "code_distance": codes.distances,
                    **(
                        {"volatility_scale": self.volatility_scale}
                        if self.volatility_scale
                        else {}
                    ),
                }
            ),
            state_column=state_column,
        )


@dataclass(frozen=True)
class StateSequence:
    frame: pl.DataFrame
    state_column: str = "behavior_state_id"

    def __post_init__(self) -> None:
        required = {"row_index", "timestamp", "symbol", "split", self.state_column, "code_distance"}
        missing = sorted(required - set(self.frame.columns))
        if missing:
            raise ValueError("state sequence missing required columns: " + ", ".join(missing))
        if self.frame.get_column(self.state_column).null_count() > 0:
            raise ValueError(f"{self.state_column} must not contain null values")
        if self.frame.get_column("code_distance").null_count() > 0:
            raise ValueError("code_distance must not contain null values")
        _validate_splits(tuple(self.frame.get_column("split").cast(pl.Utf8).to_list()))

    def attach_to(self, market_frame: pl.DataFrame) -> pl.DataFrame:
        work = _with_state_row_index(market_frame)
        if "symbol" in work.columns and "symbol" in self.frame.columns:
            return work.join(
                self.frame.select("symbol", "row_index", self.state_column),
                on=["symbol", "row_index"],
                how="left",
            )
        return work.join(
            self.frame.select("row_index", self.state_column),
            on="row_index",
            how="left",
        )

    def event_frame(self, market_frame: pl.DataFrame) -> pl.DataFrame:
        work = _with_state_row_index(market_frame)
        keys = ["row_index"]
        if "symbol" in work.columns and "symbol" in self.frame.columns:
            keys.insert(0, "symbol")
        market_columns = [
            column
            for column in work.columns
            if column in set(keys)
            or column not in set(self.frame.columns)
            or column in {"open", "high", "low", "close", "vol", "volume", "liquidity_event_type"}
        ]
        return self.frame.join(
            work.select(list(dict.fromkeys(market_columns))),
            on=keys,
            how="inner",
        )



def summarize_state_stability(states: StateSequence) -> pl.DataFrame:
    if states.frame.is_empty():
        return pl.DataFrame()
    frame = states.frame.sort("symbol", "split", "timestamp", "row_index")
    train_codes_by_symbol: dict[str, set[int]] = {}
    for part in frame.filter(pl.col("split") == "train").partition_by("symbol"):
        symbol = str(part.get_column("symbol").item(0))
        train_codes_by_symbol[symbol] = set(_state_values(part, states.state_column))
    rows = []
    for part in frame.partition_by("symbol", "split", maintain_order=True):
        symbol = str(part.get_column("symbol").item(0))
        split = str(part.get_column("split").item(0))
        rows.append(
            _state_stability_row(
                part,
                states.state_column,
                symbol=symbol,
                split=split,
                train_codes=train_codes_by_symbol.get(symbol, set()),
            )
        )
    rows.append(
        _state_stability_row(
            frame,
            states.state_column,
            symbol="__all__",
            split="all",
            train_codes=set(
                _state_values(frame.filter(pl.col("split") == "train"), states.state_column)
            ),
        )
    )
    return pl.DataFrame(rows)


def _require_columns(frame: pl.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError("missing required columns: " + ", ".join(missing))


def _state_values(frame: pl.DataFrame, state_column: str) -> list[int]:
    if frame.is_empty():
        return []
    return [int(value) for value in frame.get_column(state_column).to_list()]


def _state_stability_row(
    frame: pl.DataFrame,
    state_column: str,
    *,
    symbol: str,
    split: str,
    train_codes: set[int],
) -> dict[str, object]:
    values = _state_values(frame, state_column)
    rows = len(values)
    active = set(values)
    transitions = max(0, rows - 1)
    self_transitions = sum(
        1 for left, right in zip(values, values[1:], strict=False) if left == right
    )
    dwell_lengths = _dwell_lengths(values)
    transition_entropy = _transition_entropy(values)
    valid_train_overlap = None
    if split in {"valid", "test", "all"} and active:
        valid_train_overlap = len(active & train_codes) / len(active) * 100.0
    return {
        "symbol": symbol,
        "split": split,
        "rows": rows,
        "active_codes": len(active),
        "transitions": transitions,
        "self_transitions": self_transitions,
        "state_self_transition_rate": self_transitions / transitions if transitions else None,
        "state_transition_entropy": transition_entropy,
        "state_mean_dwell": sum(dwell_lengths) / len(dwell_lengths) if dwell_lengths else None,
        "state_median_dwell": _median(dwell_lengths),
        "state_singleton_segment_pct": (
            sum(1 for length in dwell_lengths if length == 1) / len(dwell_lengths) * 100.0
            if dwell_lengths
            else None
        ),
        "state_valid_train_overlap_pct": valid_train_overlap,
        "code_distance_mean": float(frame.get_column("code_distance").mean()) if rows else None,
        "code_distance_p95": float(frame.get_column("code_distance").quantile(0.95))
        if rows
        else None,
    }


def _dwell_lengths(values: list[int]) -> list[int]:
    if not values:
        return []
    lengths = []
    current = 1
    for left, right in zip(values, values[1:], strict=False):
        if left == right:
            current += 1
        else:
            lengths.append(current)
            current = 1
    lengths.append(current)
    return lengths


def filter_short_state_runs(
    codes: CodeSequence,
    *,
    symbols: Iterable[str],
    splits: Iterable[str],
    min_duration: int,
) -> CodeSequence:
    if min_duration < 0:
        raise ValueError("min_duration must be non-negative")
    symbol_values = tuple(str(symbol) for symbol in symbols)
    split_values = tuple(_split_name(str(split)) for split in splits)
    if len(symbol_values) != len(codes.codes) or len(split_values) != len(codes.codes):
        raise ValueError("symbols and splits must match code sequence length")
    if min_duration <= 1 or not codes.codes:
        return codes
    filtered = list(codes.codes)
    start = 0
    while start < len(filtered):
        end = start + 1
        while (
            end < len(filtered)
            and symbol_values[end] == symbol_values[start]
            and split_values[end] == split_values[start]
        ):
            end += 1
        filtered[start:end] = _filter_short_state_run_segment(
            filtered[start:end], min_duration=min_duration
        )
        start = end
    return CodeSequence(
        codes=tuple(filtered),
        distances=codes.distances,
        row_index=codes.row_index,
        splits=codes.splits,
    )


def _filter_short_state_run_segment(values: list[int], *, min_duration: int) -> list[int]:
    if len(values) <= 1:
        return values
    runs: list[tuple[int, int, int]] = []
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            runs.append((start, index, values[start]))
            start = index
    filtered = list(values)
    for run_index, (start, end, value) in enumerate(runs):
        if end - start >= min_duration:
            continue
        previous_value = runs[run_index - 1][2] if run_index > 0 else None
        next_value = runs[run_index + 1][2] if run_index + 1 < len(runs) else None
        if previous_value is not None and next_value is not None:
            replacement = previous_value
        elif previous_value is not None:
            replacement = previous_value
        elif next_value is not None:
            replacement = next_value
        else:
            replacement = value
        filtered[start:end] = [replacement] * (end - start)
    return filtered


def _transition_entropy(values: list[int]) -> float | None:
    if len(values) < 2:
        return None
    transitions: dict[int, dict[int, int]] = {}
    for left, right in zip(values, values[1:], strict=False):
        transitions.setdefault(left, {})[right] = transitions.setdefault(left, {}).get(right, 0) + 1
    entropies = []
    for counts in transitions.values():
        total = sum(counts.values())
        if total <= 0:
            continue
        entropy = 0.0
        for count in counts.values():
            probability = count / total
            entropy -= probability * math.log(probability)
        entropies.append(entropy)
    return sum(entropies) / len(entropies) if entropies else None


def _median(values: list[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _timestamp_values(frame: pl.DataFrame, column: str) -> list[int]:
    if column not in frame.columns:
        return list(range(frame.height))
    series = frame.get_column(column)
    if series.dtype == pl.Datetime:
        return [int(value.timestamp() * 1000) for value in series.to_list()]
    return [int(value) for value in series.to_list()]


def _validate_splits(splits: Iterable[str]) -> None:
    invalid = [split for split in splits if split not in SPLIT_NAMES]
    if invalid:
        raise ValueError("splits must be one of: train, valid, test")


def _split_name(value: str) -> Literal["train", "valid", "test"]:
    if value not in SPLIT_NAMES:
        raise ValueError("splits must be one of: train, valid, test")
    return value  # type: ignore[return-value]


def _with_state_row_index(frame: pl.DataFrame) -> pl.DataFrame:
    if "row_index" in frame.columns:
        return frame
    if "symbol" not in frame.columns:
        return frame.with_row_index("row_index")
    parts = [
        part.with_row_index("row_index")
        for part in frame.partition_by("symbol", maintain_order=True)
    ]
    return pl.concat(parts, how="diagonal_relaxed") if parts else frame.with_row_index("row_index")


def _apply_volatility_scaling(
    frame: pl.DataFrame,
    *,
    columns: FeatureColumns,
    scaling: VolatilityScalingConfig,
) -> pl.DataFrame:
    if scaling.return_column not in frame.columns:
        raise ValueError(f"volatility return column missing: {scaling.return_column}")
    if columns.symbol in frame.columns:
        parts = frame.partition_by(columns.symbol, maintain_order=True)
        scaled = [
            _apply_volatility_scaling_to_symbol(part, scaling)
            for part in parts
        ]
        return pl.concat(scaled, how="diagonal_relaxed") if scaled else frame
    return _apply_volatility_scaling_to_symbol(frame, scaling)


def _apply_volatility_scaling_to_symbol(
    frame: pl.DataFrame,
    scaling: VolatilityScalingConfig,
) -> pl.DataFrame:
    returns = [float(value) for value in frame.get_column(scaling.return_column).to_list()]
    scales = _causal_ewm_std(
        returns,
        half_life=scaling.half_life,
        min_periods=scaling.min_periods,
        floor=scaling.floor,
        cap=scaling.cap,
    )
    work = frame.with_columns(pl.Series(scaling.output_column, scales))
    return work.with_columns(
        [
            (pl.col(column) / pl.col(scaling.output_column)).alias(column)
            for column in scaling.columns
            if column in work.columns
        ]
    )


def _causal_ewm_std(
    values: Iterable[float],
    *,
    half_life: float,
    min_periods: int,
    floor: float,
    cap: float,
) -> tuple[float, ...]:
    alpha = 1.0 - math.exp(math.log(0.5) / half_life)
    mean = 0.0
    variance = 0.0
    count = 0
    scales = []
    for raw in values:
        value = float(raw)
        count += 1
        if count == 1:
            mean = value
            variance = 0.0
        else:
            delta = value - mean
            mean = mean + alpha * delta
            variance = (1.0 - alpha) * (variance + alpha * delta * delta)
        scale = math.sqrt(max(variance, 0.0)) if count >= min_periods else floor
        scales.append(min(max(scale, floor), cap))
    return tuple(scales)

