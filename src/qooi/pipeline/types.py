"""Pipeline result types."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import polars as pl


@dataclass(frozen=True)
class FrameHealth:
    product: str
    key: str
    actual_rows: int = 0
    target_rows: int = 0
    coverage_pct: float = 0.0
    latest_ts: int | None = None
    age_hours: float = 0.0
    status: str = "missing"
    gaps: int = 0
    duplicates: int = 0
    notes: tuple[str, ...] = ()

    @classmethod
    def from_frame(
        cls,
        frame: pl.DataFrame,
        *,
        product: str,
        key: str,
        target_rows: int = 0,
        threshold_hours: float = 0.0,
        timestamp_col: str = "timestamp",
        expected_interval_ms: int = 0,
    ) -> FrameHealth:
        if frame.is_empty() or timestamp_col not in frame.columns:
            return cls(
                product=product, key=key, status="missing", notes=("empty_or_missing_timestamp",)
            )

        ordered = frame.sort(timestamp_col) if timestamp_col in frame.columns else frame
        ts = [int(v) for v in ordered[timestamp_col].to_list() if v is not None]
        unique_ts = len(set(ts))
        duplicates = len(ts) - unique_ts
        gaps = sum(
            1
            for i in range(1, len(ts))
            if expected_interval_ms > 0 and ts[i] - ts[i - 1] != expected_interval_ms
        )
        latest = ts[-1] if ts else None
        age_hours = (
            max(0.0, (datetime.now(UTC).timestamp() * 1000 - latest) / 3_600_000) if latest else 0.0
        )
        fresh = threshold_hours > 0 and age_hours <= threshold_hours
        coverage = min(100.0, len(ts) / target_rows * 100.0) if target_rows > 0 else 0.0
        notes = []
        if len(ts) < target_rows:
            notes.append("below_target_rows")
        if duplicates:
            notes.append("duplicate_timestamps")
        if gaps:
            notes.append("timeframe_gaps")
        return cls(
            product=product,
            key=key,
            actual_rows=frame.height,
            target_rows=target_rows,
            coverage_pct=coverage,
            latest_ts=latest,
            age_hours=age_hours,
            status="fresh" if fresh else ("stale" if ts else "missing"),
            gaps=gaps,
            duplicates=duplicates,
            notes=tuple(notes),
        )


@dataclass(frozen=True)
class ProductResult:
    product: str
    frame: pl.DataFrame
    health: FrameHealth
