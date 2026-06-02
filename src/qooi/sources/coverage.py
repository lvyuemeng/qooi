"""Source manifest and coverage scoring helpers."""

from __future__ import annotations

from typing import Any

import polars as pl

from qooi.exchange.store import HistoryCoverage
from qooi.sources.manifest import manifest_frame, now_ms, source_manifest_row

__all__ = [
    "compute_source_coverage_score",
    "manifest_frame",
    "manifest_row_from_history_coverage",
    "missing_evidence_for_symbol",
    "now_ms",
    "source_manifest_row",
]


def manifest_row_from_history_coverage(
    coverage: HistoryCoverage,
    *,
    phase: str = "collect-market",
    status: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
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


def _note_value(notes: tuple[str, ...], key: str) -> str:
    prefix = f"{key}="
    for note in notes:
        if note.startswith(prefix):
            return note.removeprefix(prefix)
    return ""
