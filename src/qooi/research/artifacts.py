"""Artifact projections and export writing for the research pipe."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class ArtifactBundle:
    name: str
    tables: dict[str, pl.DataFrame]
    summary: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)


def project_transition_graph(patterns: pl.DataFrame) -> pl.DataFrame:
    if patterns.is_empty():
        return pl.DataFrame()
    transitions = patterns.filter(pl.col("pattern_family") == "transition")
    if transitions.is_empty():
        return pl.DataFrame()
    split = transitions.with_columns(
        pl.col("pattern_value")
        .str.split_exact("->", 1)
        .struct.rename_fields(["source_state", "target_state"])
        .alias("_edge")
    ).unnest("_edge")
    counts = split.group_by(
        "symbol",
        "timeframe",
        "state_column",
        "source_state",
        "target_state",
        "invalid_state_present",
    ).agg(pl.len().alias("rows"))
    source_totals = counts.group_by("symbol", "timeframe", "state_column", "source_state").agg(
        pl.col("rows").sum().alias("source_rows")
    )
    return counts.join(
        source_totals, on=["symbol", "timeframe", "state_column", "source_state"]
    ).with_columns(
        pl.lit("state-transition-graph").alias("artifact"),
        (pl.col("rows") / pl.col("source_rows")).alias("transition_probability"),
    )


def project_transition_information(scored: pl.DataFrame) -> pl.DataFrame:
    if scored.is_empty() or "pattern_family" not in scored.columns:
        return pl.DataFrame()
    return scored.filter(pl.col("pattern_family") == "transition_information")


def project_pattern_quality(scored: pl.DataFrame, families: tuple[str, ...] = ()) -> pl.DataFrame:
    if scored.is_empty() or not families:
        return scored
    return scored.filter(pl.col("pattern_family").is_in(families))


def project_promotion_candidates(scored: pl.DataFrame) -> pl.DataFrame:
    if scored.is_empty() or "passes_promotion_gate" not in scored.columns:
        return scored.head(0)
    return scored.filter(pl.col("passes_promotion_gate").fill_null(False))


def write_bundle(bundle: ArtifactBundle, export_dir: str | Path) -> list[str]:
    root = Path(export_dir)
    root.mkdir(parents=True, exist_ok=True)
    written = []
    for name, table in bundle.tables.items():
        path = root / name
        table.write_csv(path)
        written.append(str(path))
    return written
