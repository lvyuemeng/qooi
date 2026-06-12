"""Source manifest and coverage scoring helpers."""

from __future__ import annotations

from typing import Protocol

import polars as pl

from qooi.sources.manifest import SourceManifestRow, manifest_frame, now_ms, source_manifest_row

__all__ = [
    "compute_source_coverage_score",
    "eligible_backfill_symbols",
    "eligible_fetch_symbols",
    "latest_manifest_rows",
    "latest_manifest_status",
    "manifest_frame",
    "manifest_row_from_history_coverage",
    "missing_evidence_for_symbol",
    "now_ms",
    "source_manifest_row",
    "stale_symbols",
]


class HistoryCoverageLike(Protocol):
    inst_id: str
    bar: str
    source: str
    actual_bars: int
    actual_start_ms: int | None
    actual_end_ms: int | None
    coverage_pct: float
    notes: tuple[str, ...]


def manifest_row_from_history_coverage(
    coverage: HistoryCoverageLike,
    *,
    phase: str = "collect-market",
    status: str | None = None,
    error: str | None = None,
) -> SourceManifestRow:
    warning = ";".join(coverage.notes)
    row_status = status or ("ok" if coverage.actual_bars else "missing")
    if error:
        row_status = "failed"
        warning = ";".join(part for part in (warning, error) if part)
    return source_manifest_row(
        symbol=coverage.inst_id,
        source="bars",
        phase=phase,
        status=row_status,
        backend="cache",
        endpoint=f"bars:{coverage.bar}:{coverage.source}",
        rows=coverage.actual_bars,
        range_start=coverage.actual_start_ms,
        range_end=coverage.actual_end_ms,
        coverage_pct=coverage.coverage_pct,
        warning=warning,
        stop_reason=_note_value(coverage.notes, "fetch_stop"),
    )


def compute_source_coverage_score(coverage: pl.DataFrame, symbol: str) -> float:
    if coverage.is_empty() or "symbol" not in coverage.columns:
        return 0.0
    rows = coverage.filter(pl.col("symbol") == symbol)
    if rows.is_empty():
        return 0.0
    required = ("bars", "books", "trades", "funding", "open_interest")
    weights = {"bars": 0.40, "books": 0.15, "trades": 0.20, "funding": 0.15, "open_interest": 0.10}
    score = 0.0
    for source in required:
        source_rows = rows.filter(pl.col("source") == source)
        if source_rows.is_empty():
            continue
        statuses = set(source_rows.get_column("status").cast(pl.String).to_list())
        if "ok" in statuses:
            score += weights[source]
        elif "partial" in statuses:
            score += weights[source] * 0.5
    return round(score, 4)


def missing_evidence_for_symbol(coverage: pl.DataFrame, symbol: str) -> str:
    required = ("bars", "books", "trades", "funding", "open_interest")
    if coverage.is_empty() or "symbol" not in coverage.columns:
        return ";".join(f"{source}_missing" for source in required)
    rows = coverage.filter(pl.col("symbol") == symbol)
    missing = []
    for source in required:
        source_rows = rows.filter(pl.col("source") == source)
        if source_rows.is_empty():
            missing.append(f"{source}_missing")
            continue
        statuses = set(source_rows.get_column("status").cast(pl.String).to_list())
        if not ({"ok", "partial"} & statuses):
            missing.append(f"{source}_missing")
    return ";".join(missing)


def latest_manifest_rows(manifest: pl.DataFrame) -> pl.DataFrame:
    if manifest.is_empty() or not {"symbol", "source", "timestamp"}.issubset(manifest.columns):
        return manifest
    return manifest.sort("timestamp").unique(subset=["symbol", "source"], keep="last")


def latest_manifest_status(manifest: pl.DataFrame, *, source: str, symbol: str) -> str | None:
    rows = latest_manifest_rows(manifest)
    if rows.is_empty() or not {"symbol", "source", "status"}.issubset(rows.columns):
        return None
    match = rows.filter((pl.col("symbol") == symbol) & (pl.col("source") == source)).head(1)
    if match.is_empty():
        return None
    value = match.get_column("status")[0]
    return str(value) if value is not None else None


def stale_symbols(
    frame: pl.DataFrame,
    symbols: tuple[str, ...],
    *,
    now_ms: int,
    max_age_ms: int,
    timestamp_col: str = "timestamp",
) -> tuple[str, ...]:
    latest_by_symbol = _latest_timestamps(frame, timestamp_col=timestamp_col)
    return tuple(
        symbol
        for symbol in symbols
        if (latest := latest_by_symbol.get(symbol)) is None or now_ms - latest > max_age_ms
    )


def eligible_backfill_symbols(
    frame: pl.DataFrame,
    symbols: tuple[str, ...],
    *,
    target_start_ms: int,
    now_ms: int,
    max_age_ms: int,
    min_rows: int,
    refresh: bool,
    timestamp_col: str = "timestamp",
) -> tuple[str, ...]:
    if refresh:
        return symbols
    if frame.is_empty() or "symbol" not in frame.columns or timestamp_col not in frame.columns:
        return symbols
    ranges = frame.group_by("symbol").agg(
        pl.col(timestamp_col).min().alias("earliest"),
        pl.col(timestamp_col).max().alias("latest"),
        pl.len().alias("rows"),
    )
    by_symbol = {str(row["symbol"]): row for row in ranges.iter_rows(named=True)}
    eligible: list[str] = []
    for symbol in symbols:
        row = by_symbol.get(symbol)
        if row is None:
            eligible.append(symbol)
            continue
        earliest = row["earliest"]
        latest = row["latest"]
        rows = row["rows"]
        if earliest is None or latest is None:
            eligible.append(symbol)
            continue
        if int(latest) < now_ms - max_age_ms:
            eligible.append(symbol)
            continue
        if int(earliest) > target_start_ms:
            eligible.append(symbol)
            continue
        if int(rows) < min_rows:
            eligible.append(symbol)
    return tuple(eligible)


def eligible_fetch_symbols(
    frame: pl.DataFrame,
    symbols: tuple[str, ...],
    *,
    now_ms: int,
    max_age_ms: int,
    refresh: bool,
    timestamp_col: str = "timestamp",
) -> tuple[str, ...]:
    if refresh:
        return symbols
    latest_by_symbol = _latest_timestamps(frame, timestamp_col=timestamp_col)
    return tuple(
        symbol
        for symbol in symbols
        if (latest := latest_by_symbol.get(symbol)) is None or now_ms - latest > max_age_ms
    )


def _latest_timestamps(frame: pl.DataFrame, *, timestamp_col: str) -> dict[str, int]:
    if frame.is_empty() or "symbol" not in frame.columns or timestamp_col not in frame.columns:
        return {}
    rows = frame.group_by("symbol").agg(pl.col(timestamp_col).max().alias("latest"))
    return {
        str(row["symbol"]): int(row["latest"])
        for row in rows.iter_rows(named=True)
        if row["latest"] is not None
    }


def _note_value(notes: tuple[str, ...], key: str) -> str:
    prefix = f"{key}="
    for note in notes:
        if note.startswith(prefix):
            return note.removeprefix(prefix)
    return ""
