"""Generic CSV artifact helpers for source-backed workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from qooi.sources.schema import (
    MESSAGE_CLASSIFICATION_SCHEMA,
    SOURCE_BARS_SCHEMA,
    SOURCE_BOOKS_SCHEMA,
    SOURCE_FUNDING_SCHEMA,
    SOURCE_LONG_SHORT_RATIO_SCHEMA,
    SOURCE_MANIFEST_SCHEMA,
    SOURCE_MESSAGES_SCHEMA,
    SOURCE_ONCHAIN_FLOWS_SCHEMA,
    SOURCE_OPEN_INTEREST_SCHEMA,
    SOURCE_POLYMARKET_EVENTS_SCHEMA,
    SOURCE_POLYMARKET_MARKETS_SCHEMA,
    SOURCE_TAKER_VOLUME_SCHEMA,
    SOURCE_TRADES_SCHEMA,
)


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    relative_path: str
    schema: dict[str, pl.DataType]
    required: bool = True


SOURCE_ARTIFACT_SPECS: dict[str, ArtifactSpec] = {
    "source_manifest": ArtifactSpec(
        "source_manifest", "source-manifest.csv", SOURCE_MANIFEST_SCHEMA
    ),
    "source_bars": ArtifactSpec("source_bars", "sources/bars.csv", SOURCE_BARS_SCHEMA),
    "source_books": ArtifactSpec("source_books", "sources/books.csv", SOURCE_BOOKS_SCHEMA),
    "source_trades": ArtifactSpec("source_trades", "sources/trades.csv", SOURCE_TRADES_SCHEMA),
    "source_funding": ArtifactSpec("source_funding", "sources/funding.csv", SOURCE_FUNDING_SCHEMA),
    "source_open_interest": ArtifactSpec(
        "source_open_interest", "sources/open-interest.csv", SOURCE_OPEN_INTEREST_SCHEMA
    ),
    "source_taker_volume": ArtifactSpec(
        "source_taker_volume", "sources/taker-volume-contract.csv", SOURCE_TAKER_VOLUME_SCHEMA
    ),
    "source_long_short_ratios": ArtifactSpec(
        "source_long_short_ratios", "sources/long-short-ratios.csv", SOURCE_LONG_SHORT_RATIO_SCHEMA
    ),
    "source_onchain_flows": ArtifactSpec(
        "source_onchain_flows", "sources/onchain-flows.csv", SOURCE_ONCHAIN_FLOWS_SCHEMA
    ),
    "source_messages": ArtifactSpec(
        "source_messages", "sources/messages-normalized.csv", SOURCE_MESSAGES_SCHEMA
    ),
    "source_polymarket_events": ArtifactSpec(
        "source_polymarket_events",
        "sources/polymarket-events.csv",
        SOURCE_POLYMARKET_EVENTS_SCHEMA,
    ),
    "source_polymarket_markets": ArtifactSpec(
        "source_polymarket_markets",
        "sources/polymarket-markets.csv",
        SOURCE_POLYMARKET_MARKETS_SCHEMA,
    ),
    "message_classifications": ArtifactSpec(
        "message_classifications",
        "sources/message-classifications.csv",
        MESSAGE_CLASSIFICATION_SCHEMA,
    ),
}


def artifact_path(output_dir: Path, spec: ArtifactSpec) -> Path:
    return output_dir / spec.relative_path


def coerce_frame(frame: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if not schema:
        return frame
    for col, dtype in schema.items():
        if col not in frame.columns:
            frame = frame.with_columns(pl.lit(None).cast(dtype).alias(col))
        elif dtype == pl.Boolean:
            frame = frame.with_columns(_boolean_expr(col).alias(col))
        else:
            frame = frame.with_columns(pl.col(col).cast(dtype, strict=False).alias(col))
    return frame.select(schema.keys())


def read_frame_artifact(output_dir: Path, spec: ArtifactSpec) -> pl.DataFrame:
    path = artifact_path(output_dir, spec)
    if not path.exists():
        return pl.DataFrame(schema=spec.schema)
    return coerce_frame(pl.read_csv(path), spec.schema)


def write_frame_artifact(output_dir: Path, spec: ArtifactSpec, frame: pl.DataFrame) -> None:
    path = artifact_path(output_dir, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    coerce_frame(frame, spec.schema).write_csv(path)


def write_text_artifact(output_dir: Path, name: str, text: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _boolean_expr(col: str) -> pl.Expr:
    text = pl.col(col).cast(pl.String, strict=False).str.to_lowercase().str.strip_chars()
    return (
        pl.when(text.is_in(["true", "1", "yes"]))
        .then(True)
        .when(text.is_in(["false", "0", "no"]))
        .then(False)
        .otherwise(None)
        .cast(pl.Boolean)
    )

