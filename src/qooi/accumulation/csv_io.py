"""CSV-only IO for accumulation scanner artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from qooi.accumulation.artifacts import ARTIFACT_SPECS, ArtifactName, artifact_spec


@dataclass(frozen=True)
class SourceBundle:
    discovery: pl.DataFrame
    bars: pl.DataFrame
    books: pl.DataFrame
    trades: pl.DataFrame
    funding: pl.DataFrame
    open_interest: pl.DataFrame
    onchain_flows: pl.DataFrame
    messages: pl.DataFrame
    polymarket_events: pl.DataFrame
    polymarket_markets: pl.DataFrame
    message_classifications: pl.DataFrame
    manifest: pl.DataFrame


def artifact_path(output_dir: Path, name: ArtifactName) -> Path:
    return output_dir / artifact_spec(name).relative_path


def coerce_frame(frame: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if not schema:
        return frame
    for col, dtype in schema.items():
        if col not in frame.columns:
            frame = frame.with_columns(pl.lit(None).cast(dtype).alias(col))
        else:
            frame = frame.with_columns(pl.col(col).cast(dtype, strict=False).alias(col))
    return frame.select(schema.keys())


def read_artifact(output_dir: Path, name: ArtifactName) -> pl.DataFrame:
    spec = artifact_spec(name)
    path = output_dir / spec.relative_path
    if not path.exists():
        return pl.DataFrame(schema=spec.schema)
    return coerce_frame(pl.read_csv(path), spec.schema)


def write_artifact(output_dir: Path, name: ArtifactName, frame: pl.DataFrame) -> None:
    spec = artifact_spec(name)
    path = output_dir / spec.relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    coerce_frame(frame, spec.schema).write_csv(path)


def write_text_artifact(output_dir: Path, name: str, text: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / name).write_text(text, encoding="utf-8")


def write_csv_artifacts(
    output_dir: Path,
    *,
    features: pl.DataFrame | None = None,
    scores: pl.DataFrame | None = None,
    alerts: pl.DataFrame | None = None,
    backtest_events: pl.DataFrame | None = None,
    backtest_summary: pl.DataFrame | None = None,
    data_coverage: pl.DataFrame | None = None,
    discovery: pl.DataFrame | None = None,
    source_manifest: pl.DataFrame | None = None,
    candidate_detail: pl.DataFrame | None = None,
    candidate_summary: pl.DataFrame | None = None,
    next_fetch_actions: pl.DataFrame | None = None,
) -> None:
    artifacts: dict[ArtifactName, pl.DataFrame | None] = {
        "features": features,
        "scores": scores,
        "alerts": alerts,
        "backtest_events": backtest_events,
        "backtest_summary": backtest_summary,
        "data_coverage": data_coverage,
        "candidate_discovery": discovery,
        "source_manifest": source_manifest,
        "candidate_detail": candidate_detail,
        "candidate_summary": candidate_summary,
        "next_fetch_actions": next_fetch_actions,
    }
    for name, frame in artifacts.items():
        if frame is not None:
            write_artifact(output_dir, name, frame)


def write_source_bundle(
    output_dir: Path,
    *,
    bars: pl.DataFrame | None = None,
    books: pl.DataFrame | None = None,
    trades: pl.DataFrame | None = None,
    funding: pl.DataFrame | None = None,
    open_interest: pl.DataFrame | None = None,
    onchain_flows: pl.DataFrame | None = None,
    messages: pl.DataFrame | None = None,
    polymarket_events: pl.DataFrame | None = None,
    polymarket_markets: pl.DataFrame | None = None,
    message_classifications: pl.DataFrame | None = None,
) -> None:
    artifacts: dict[ArtifactName, pl.DataFrame | None] = {
        "source_bars": bars,
        "source_books": books,
        "source_trades": trades,
        "source_funding": funding,
        "source_open_interest": open_interest,
        "source_onchain_flows": onchain_flows,
        "source_messages": messages,
        "source_polymarket_events": polymarket_events,
        "source_polymarket_markets": polymarket_markets,
        "message_classifications": message_classifications,
    }
    for name, frame in artifacts.items():
        should_write = frame is not None and (
            not frame.is_empty() or not artifact_path(output_dir, name).exists()
        )
        if should_write:
            write_artifact(output_dir, name, frame)


def read_source_bundle(output_dir: Path) -> SourceBundle:
    return SourceBundle(
        discovery=read_artifact(output_dir, "candidate_discovery"),
        bars=read_artifact(output_dir, "source_bars"),
        books=read_artifact(output_dir, "source_books"),
        trades=read_artifact(output_dir, "source_trades"),
        funding=read_artifact(output_dir, "source_funding"),
        open_interest=read_artifact(output_dir, "source_open_interest"),
        onchain_flows=read_artifact(output_dir, "source_onchain_flows"),
        messages=read_artifact(output_dir, "source_messages"),
        polymarket_events=read_artifact(output_dir, "source_polymarket_events"),
        polymarket_markets=read_artifact(output_dir, "source_polymarket_markets"),
        message_classifications=read_artifact(output_dir, "message_classifications"),
        manifest=read_artifact(output_dir, "source_manifest"),
    )


def assert_csv_catalog() -> None:
    paths = [spec.relative_path for spec in ARTIFACT_SPECS.values()]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate accumulation artifact paths")
    non_csv = [path for path in paths if not path.endswith(".csv")]
    if non_csv:
        raise ValueError(f"non-CSV DataFrame artifacts: {non_csv}")
