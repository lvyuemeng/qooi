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


@dataclass(frozen=True)
class SourceFamily:
    name: str
    artifact: str
    timestamp_col: str
    raw_sources: tuple[str, ...]
    merge_keys: tuple[tuple[str, ...], ...]
    row_kind_col: str | None = None
    history_kind: str | None = None
    known_at_col: str | None = None


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


SOURCE_FAMILIES: dict[str, SourceFamily] = {
    "books": SourceFamily(
        "books", "source_books", "timestamp", ("books",), (("symbol", "timestamp"),)
    ),
    "trades": SourceFamily(
        "trades",
        "source_trades",
        "timestamp",
        ("trades",),
        (("symbol", "trade_id"), ("symbol", "timestamp", "price", "size", "side")),
    ),
    "funding": SourceFamily(
        "funding",
        "source_funding",
        "known_at_ms",
        ("funding", "funding_rate"),
        (("symbol", "funding_time"), ("symbol", "timestamp")),
        row_kind_col="funding_source_kind",
        history_kind="history",
        known_at_col="known_at_ms",
    ),
    "open_interest": SourceFamily(
        "open_interest",
        "source_open_interest",
        "timestamp",
        ("open_interest_history",),
        (("symbol", "timestamp"),),
    ),
    "taker_volume": SourceFamily(
        "taker_volume",
        "source_taker_volume",
        "timestamp",
        ("taker_volume_contract",),
        (("symbol", "timestamp"),),
    ),
    "long_short_ratios": SourceFamily(
        "long_short_ratios",
        "source_long_short_ratios",
        "timestamp",
        (
            "long_short_ratio_contract",
            "top_trader_long_short_account_ratio_contract",
            "top_trader_long_short_position_ratio_contract",
        ),
        (("symbol", "timestamp"),),
    ),
    "messages": SourceFamily(
        "messages",
        "source_messages",
        "timestamp",
        ("messages",),
        (("source", "source_id"), ("symbol", "timestamp", "text_hash")),
    ),
}

_RAW_SOURCE_FAMILIES = {
    raw_source: family.name
    for family in SOURCE_FAMILIES.values()
    for raw_source in family.raw_sources
}


def source_family(name: str) -> SourceFamily:
    return SOURCE_FAMILIES[name]


def source_manifest_family(raw_source: str) -> str:
    return _RAW_SOURCE_FAMILIES.get(raw_source, raw_source if raw_source in SOURCE_FAMILIES else "")


_EXTRA_ARTIFACT_KEYS: dict[str, tuple[tuple[str, ...], ...]] = {
    "source_bars": (("symbol", "timestamp"),),
    "source_onchain_flows": (("symbol", "timestamp"),),
    "source_polymarket_events": (("symbol", "event_id"),),
    "source_polymarket_markets": (("symbol", "market_id"),),
    "message_classifications": (("symbol", "message_id"),),
}


def artifact_merge_keys(artifact_name: str) -> tuple[tuple[str, ...], ...]:
    for family in SOURCE_FAMILIES.values():
        if family.artifact == artifact_name:
            return family.merge_keys
    return _EXTRA_ARTIFACT_KEYS.get(artifact_name, ())


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
