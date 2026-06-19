"""Cross-cutting native profiling context and artifacts."""

from __future__ import annotations

import cProfile
import io
import pstats
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, TypeVar

import polars as pl
from pydantic import BaseModel, ConfigDict

ProfileMode = Literal["off", "stage", "hotpath", "native"]
T = TypeVar("T")


class ProfileConfig(BaseModel):
    """Workflow-embedded profile configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: ProfileMode = "off"
    top_n: int = 30


@dataclass(frozen=True)
class StageRecord:
    run_id: str
    layer: str
    component: str
    stage: str
    seconds: float
    status: str


@dataclass(frozen=True)
class FrameRecord:
    run_id: str
    layer: str
    component: str
    frame: str
    rows: int
    cols: int
    symbol_count: int
    timeframe_count: int
    horizon_count: int
    source_family_count: int
    decision_timeframe_count: int


@dataclass(frozen=True)
class NativeProfileRecord:
    run_id: str
    rank: int
    function: str
    file: str
    line: int
    ncalls: str
    tottime_s: float
    cumtime_s: float


@dataclass
class ProfileContext:
    """Injected profiling collector for workflow/module stages."""

    config: ProfileConfig
    root: Path
    run_id: str = field(default_factory=lambda: time.strftime("%Y%m%dT%H%M%S"))
    stages: list[StageRecord] = field(default_factory=list)
    frames: list[FrameRecord] = field(default_factory=list)
    native_records: list[NativeProfileRecord] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: ProfileConfig | None, root: Path) -> ProfileContext:
        return cls(config or ProfileConfig(), root)

    @property
    def enabled(self) -> bool:
        return self.config.mode != "off"

    def stage(self, layer: str, component: str, stage: str) -> _ProfileStage:
        return _ProfileStage(self, layer, component, stage)

    def frame(self, layer: str, component: str, frame: str, data: pl.DataFrame) -> None:
        if not self.enabled:
            return
        self.frames.append(
            FrameRecord(
                run_id=self.run_id,
                layer=layer,
                component=component,
                frame=frame,
                rows=data.height,
                cols=len(data.columns),
                symbol_count=_n_unique(data, "symbol"),
                timeframe_count=_n_unique(data, "timeframe"),
                horizon_count=_n_unique(data, "outcome_horizon"),
                source_family_count=_n_unique(data, "source_family"),
                decision_timeframe_count=_n_unique(data, "decision_timeframe"),
            )
        )

    def native(self, label: str, func: Callable[[], T]) -> T:
        if self.config.mode not in {"hotpath", "native"}:
            return func()
        profiler = cProfile.Profile()
        result = profiler.runcall(func)
        self._record_native(label, profiler)
        return result

    def write(self) -> None:
        if not self.enabled:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        _stage_frame(self.stages).write_csv(self.root / "stages.csv")
        _frame_frame(self.frames).write_csv(self.root / "frames.csv")
        if self.native_records:
            _native_frame(self.native_records).write_csv(self.root / "native.csv")
        self.root.joinpath("summary.md").write_text(self._summary(), encoding="utf-8")

    def _record_native(self, label: str, profiler: cProfile.Profile) -> None:
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumtime")
        rows = sorted(
            stats.stats.items(),
            key=lambda item: item[1][3],
            reverse=True,
        )[: self.config.top_n]
        for rank, ((filename, line, function), values) in enumerate(rows, start=1):
            primitive_calls, total_calls, tottime, cumtime, _callers = values
            ncalls = (
                str(total_calls)
                if primitive_calls == total_calls
                else f"{primitive_calls}/{total_calls}"
            )
            self.native_records.append(
                NativeProfileRecord(
                    run_id=self.run_id,
                    rank=rank,
                    function=f"{label}:{function}",
                    file=filename,
                    line=line,
                    ncalls=ncalls,
                    tottime_s=float(tottime),
                    cumtime_s=float(cumtime),
                )
            )

    def _summary(self) -> str:
        lines = ["# Profile Summary", ""]
        if self.stages:
            lines.extend(["## Stages", "", "| stage | seconds | status |", "|---|---:|---|"])
            for record in sorted(self.stages, key=lambda row: row.seconds, reverse=True):
                lines.append(
                    f"| {record.layer}.{record.component}.{record.stage} | "
                    f"{record.seconds:.6f} | {record.status} |"
                )
            lines.append("")
        if self.frames:
            lines.extend(["## Frames", "", "| frame | rows | cols |", "|---|---:|---:|"])
            for record in sorted(self.frames, key=lambda row: row.rows, reverse=True):
                lines.append(
                    f"| {record.layer}.{record.component}.{record.frame} | "
                    f"{record.rows} | {record.cols} |"
                )
            lines.append("")
        return "\n".join(lines)


@dataclass
class _ProfileStage(AbstractContextManager[None]):
    profile: ProfileContext
    layer: str
    component: str
    name: str
    started: float = 0.0

    def __enter__(self) -> None:
        self.started = time.perf_counter()
        return None

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self.profile.enabled:
            self.profile.stages.append(
                StageRecord(
                    run_id=self.profile.run_id,
                    layer=self.layer,
                    component=self.component,
                    stage=self.name,
                    seconds=time.perf_counter() - self.started,
                    status="error" if exc_type else "ok",
                )
            )
            self.profile.write()
        return False


def _n_unique(frame: pl.DataFrame, column: str) -> int:
    if column not in frame.columns or frame.is_empty():
        return 0
    return frame.get_column(column).drop_nulls().n_unique()


def _stage_frame(records: list[StageRecord]) -> pl.DataFrame:
    return pl.DataFrame(
        [asdict(record) for record in records],
        schema={
            "run_id": pl.String,
            "layer": pl.String,
            "component": pl.String,
            "stage": pl.String,
            "seconds": pl.Float64,
            "status": pl.String,
        },
    )


def _frame_frame(records: list[FrameRecord]) -> pl.DataFrame:
    return pl.DataFrame(
        [asdict(record) for record in records],
        schema={
            "run_id": pl.String,
            "layer": pl.String,
            "component": pl.String,
            "frame": pl.String,
            "rows": pl.Int64,
            "cols": pl.Int64,
            "symbol_count": pl.Int64,
            "timeframe_count": pl.Int64,
            "horizon_count": pl.Int64,
            "source_family_count": pl.Int64,
            "decision_timeframe_count": pl.Int64,
        },
    )


def _native_frame(records: list[NativeProfileRecord]) -> pl.DataFrame:
    return pl.DataFrame(
        [asdict(record) for record in records],
        schema={
            "run_id": pl.String,
            "rank": pl.Int64,
            "function": pl.String,
            "file": pl.String,
            "line": pl.Int64,
            "ncalls": pl.String,
            "tottime_s": pl.Float64,
            "cumtime_s": pl.Float64,
        },
    )
