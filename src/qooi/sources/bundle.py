"""Generic source bundle IO and merge behavior."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from qooi.sources.artifacts import (
    ArtifactSpec,
    artifact_merge_keys,
    artifact_path,
    coerce_frame,
    read_frame_artifact,
    write_frame_artifact,
)


@dataclass(frozen=True)
class SourceBundle:
    discovery: pl.DataFrame
    bars: pl.DataFrame
    books: pl.DataFrame
    trades: pl.DataFrame
    funding: pl.DataFrame
    open_interest: pl.DataFrame
    taker_volume: pl.DataFrame
    long_short_ratios: pl.DataFrame
    onchain_flows: pl.DataFrame
    messages: pl.DataFrame
    polymarket_events: pl.DataFrame
    polymarket_markets: pl.DataFrame
    message_classifications: pl.DataFrame
    manifest: pl.DataFrame


SOURCE_BUNDLE_FIELDS: dict[str, str] = {
    "source_bars": "bars",
    "source_books": "books",
    "source_trades": "trades",
    "source_funding": "funding",
    "source_open_interest": "open_interest",
    "source_taker_volume": "taker_volume",
    "source_long_short_ratios": "long_short_ratios",
    "source_onchain_flows": "onchain_flows",
    "source_messages": "messages",
    "source_polymarket_events": "polymarket_events",
    "source_polymarket_markets": "polymarket_markets",
    "message_classifications": "message_classifications",
}

SOURCE_FRAME_ARTIFACTS: dict[str, str] = {value: key for key, value in SOURCE_BUNDLE_FIELDS.items()}


def read_source_bundle(output_dir: Path, catalog: Mapping[str, ArtifactSpec]) -> SourceBundle:
    return SourceBundle(
        discovery=_read_optional_frame(output_dir, catalog, "candidate_discovery"),
        bars=read_frame_artifact(output_dir, catalog["source_bars"]),
        books=read_frame_artifact(output_dir, catalog["source_books"]),
        trades=read_frame_artifact(output_dir, catalog["source_trades"]),
        funding=read_frame_artifact(output_dir, catalog["source_funding"]),
        open_interest=read_frame_artifact(output_dir, catalog["source_open_interest"]),
        taker_volume=read_frame_artifact(output_dir, catalog["source_taker_volume"]),
        long_short_ratios=read_frame_artifact(output_dir, catalog["source_long_short_ratios"]),
        onchain_flows=read_frame_artifact(output_dir, catalog["source_onchain_flows"]),
        messages=read_frame_artifact(output_dir, catalog["source_messages"]),
        polymarket_events=read_frame_artifact(output_dir, catalog["source_polymarket_events"]),
        polymarket_markets=read_frame_artifact(output_dir, catalog["source_polymarket_markets"]),
        message_classifications=read_frame_artifact(output_dir, catalog["message_classifications"]),
        manifest=read_frame_artifact(output_dir, catalog["source_manifest"]),
    )


def _read_optional_frame(
    output_dir: Path, catalog: Mapping[str, ArtifactSpec], artifact_name: str
) -> pl.DataFrame:
    spec = catalog.get(artifact_name)
    return read_frame_artifact(output_dir, spec) if spec is not None else pl.DataFrame()


def source_frame(bundle: SourceBundle, field: str) -> pl.DataFrame:
    value = getattr(bundle, field, None)
    return value if isinstance(value, pl.DataFrame) else pl.DataFrame()


def source_symbols(frame: pl.DataFrame, *, symbol_col: str = "symbol") -> set[str]:
    if frame.is_empty() or symbol_col not in frame.columns:
        return set()
    return {str(symbol) for symbol in frame.get_column(symbol_col).drop_nulls().unique().to_list()}


def missing_symbols(
    frame: pl.DataFrame, symbols: tuple[str, ...], *, symbol_col: str = "symbol"
) -> tuple[str, ...]:
    existing = source_symbols(frame, symbol_col=symbol_col)
    return tuple(symbol for symbol in symbols if symbol not in existing)


def latest_timestamp(
    frame: pl.DataFrame, *, symbol: str | None = None, timestamp_col: str = "timestamp"
) -> int | None:
    if frame.is_empty() or timestamp_col not in frame.columns:
        return None
    rows = frame
    if symbol is not None:
        if "symbol" not in rows.columns:
            return None
        rows = rows.filter(pl.col("symbol") == symbol)
    if rows.is_empty():
        return None
    value = rows.get_column(timestamp_col).drop_nulls().max()
    return int(value) if value is not None else None


def replace_symbol_rows(
    existing: pl.DataFrame, incoming: pl.DataFrame, *, symbol_col: str = "symbol"
) -> pl.DataFrame:
    if incoming.is_empty():
        return existing
    if (
        existing.is_empty()
        or symbol_col not in existing.columns
        or symbol_col not in incoming.columns
    ):
        return incoming
    symbols = incoming.get_column(symbol_col).drop_nulls().unique().to_list()
    kept = existing.filter(~pl.col(symbol_col).is_in(symbols))
    return pl.concat([kept, incoming], how="vertical_relaxed") if not kept.is_empty() else incoming


def write_source_bundle(
    output_dir: Path,
    catalog: Mapping[str, ArtifactSpec],
    **frames: pl.DataFrame | None,
) -> None:
    for frame_name, frame in frames.items():
        artifact_name = SOURCE_FRAME_ARTIFACTS.get(frame_name)
        if artifact_name is None:
            continue
        spec = catalog[artifact_name]
        should_write = frame is not None and (
            not frame.is_empty() or not artifact_path(output_dir, spec).exists()
        )
        if not should_write:
            continue
        merged = merge_source_artifact(output_dir, artifact_name, frame, spec)
        write_frame_artifact(output_dir, spec, merged)


def merge_source_artifact(
    output_dir: Path,
    artifact_name: str,
    frame: pl.DataFrame | None,
    spec: ArtifactSpec,
) -> pl.DataFrame:
    if frame is None or frame.is_empty():
        return frame if frame is not None else pl.DataFrame()
    frame = coerce_frame(frame, spec.schema)
    path = artifact_path(output_dir, spec)
    if not path.exists():
        return frame
    existing = read_frame_artifact(output_dir, spec)
    return merge_source_frames(artifact_name, existing, frame, spec)


def merge_source_frames(
    artifact_name: str,
    existing: pl.DataFrame,
    incoming: pl.DataFrame | None,
    spec: ArtifactSpec,
) -> pl.DataFrame:
    if incoming is None or incoming.is_empty():
        return existing
    frame = coerce_frame(incoming, spec.schema)
    if existing.is_empty():
        return frame
    existing = coerce_frame(existing, spec.schema)
    key = _merge_key(artifact_name, existing, frame)
    if key is None:
        return _replace_symbols(existing, frame)
    return _merge_by_key(existing, frame, key)


def _merge_key(
    artifact_name: str, existing: pl.DataFrame, incoming: pl.DataFrame
) -> tuple[str, ...] | None:
    for key in artifact_merge_keys(artifact_name):
        if set(key).issubset(existing.columns) and set(key).issubset(incoming.columns):
            return key
    if "symbol" in existing.columns and "symbol" in incoming.columns:
        return None
    return ()


def _merge_by_key(
    existing: pl.DataFrame, incoming: pl.DataFrame, key: tuple[str, ...]
) -> pl.DataFrame:
    if not key:
        return incoming
    merged = pl.concat([existing, incoming], how="vertical_relaxed")
    sort_cols = [col for col in ("symbol", "timestamp") if col in merged.columns]
    merged = merged.unique(subset=list(key), keep="last", maintain_order=True)
    return merged.sort(sort_cols) if sort_cols else merged


def _replace_symbols(existing: pl.DataFrame, incoming: pl.DataFrame) -> pl.DataFrame:
    # Fallback preserves the old source-bundle behavior when a resource key is unavailable.
    return replace_symbol_rows(existing, incoming)
