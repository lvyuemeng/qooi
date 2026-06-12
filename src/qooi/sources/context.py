"""Source context loading and availability for scanner workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import polars as pl

from qooi.sources import (
    DAY_MS,
    funding_min_rows,
    normalize_funding_artifact_frame,
    period_min_rows,
)
from qooi.sources.artifacts import (
    SOURCE_ARTIFACT_SPECS,
    source_manifest_family,
    write_frame_artifact,
)
from qooi.sources.bundle import (
    SOURCE_FRAME_ARTIFACTS,
    SourceBundle,
    merge_source_frames,
    read_source_bundle,
)
from qooi.sources.collect import BookMode, SourceCollectRequest, collect_source_context
from qooi.sources.manifest import now_ms

CONTEXT_FAMILIES = (
    "books",
    "trades",
    "funding",
    "open_interest",
    "taker_volume",
    "long_short_ratios",
    "messages",
)


class PotentialSourceConfig(Protocol):
    class SourceSection(Protocol):
        refresh_mode: Literal["inherit", "incremental", "cache_only", "force"]
        book_mode: BookMode
        book_depth: int
        max_staleness_hours: int
        trade_limit: int
        funding_limit: int
        rubik_period: str
        rubik_limit: int
        rubik_taker_unit: Literal["0", "1", "2"]
        disabled_sources: tuple[str, ...]
        disabled_symbols: tuple[str, ...]

    class TransitionSection(Protocol):
        history_days: int

    output: Path
    days: int
    fetch_concurrency: int
    refresh_mode: Literal["incremental", "cache_only", "force"]
    source: SourceSection
    transition: TransitionSection


@dataclass(frozen=True)
class SourceAvailability:
    family: str
    symbol: str
    rows: int
    latest_timestamp: int | None
    status: str
    warning: str


@dataclass(frozen=True)
class SourceContextResult:
    manifest: pl.DataFrame
    frames: dict[str, pl.DataFrame]
    availability: tuple[SourceAvailability, ...]


async def load_source_context(
    config: PotentialSourceConfig,
    *,
    symbols: tuple[str, ...],
    context_symbols: tuple[str, ...],
    discovery: pl.DataFrame,
) -> SourceContextResult:
    bundle = read_source_bundle(config.output.parent, SOURCE_ARTIFACT_SPECS)
    refresh_mode = (
        config.refresh_mode
        if config.source.refresh_mode == "inherit"
        else config.source.refresh_mode
    )
    if refresh_mode == "cache_only":
        frames = context_frames(bundle, {})
        return SourceContextResult(
            bundle.manifest,
            frames,
            tuple(source_availability(frames, bundle.manifest, symbols, config)),
        )
    if context_symbols:
        force_refresh = refresh_mode == "force"
        target_end_ms = now_ms()
        target_days = max(config.days, config.transition.history_days)
        target_start_ms = target_end_ms - target_days * DAY_MS
        request = SourceCollectRequest(
            output_dir=config.output.parent,
            symbols=context_symbols,
            discovery=discovery,
            concurrency=config.fetch_concurrency,
            book_mode=config.source.book_mode,
            book_depth=config.source.book_depth,
            max_source_staleness_hours=config.source.max_staleness_hours,
            trade_limit=config.source.trade_limit,
            funding_limit=config.source.funding_limit,
            rubik_period=config.source.rubik_period,
            rubik_limit=config.source.rubik_limit,
            rubik_taker_unit=config.source.rubik_taker_unit,
            disabled_sources=config.source.disabled_sources,
            disabled_symbols=config.source.disabled_symbols,
            refresh_trades=force_refresh,
            refresh_context=force_refresh,
            target_source_start_ms=target_start_ms,
            target_source_end_ms=target_end_ms,
            funding_min_rows=funding_min_rows(target_days),
            rubik_min_rows=period_min_rows(target_days, config.source.rubik_period),
            existing_frames=context_frames(bundle, {}),
        )
        try:
            result = await collect_source_context(request)
            frames = merge_context_frames(bundle, result.frames)
            write_context_frames(config.output.parent, frames)
            write_frame_artifact(
                config.output.parent,
                SOURCE_ARTIFACT_SPECS["source_manifest"],
                concat_manifest(bundle.manifest, result.manifest),
            )
            manifest = concat_manifest(bundle.manifest, result.manifest)
            availability = tuple(source_availability(frames, manifest, symbols, config))
            return SourceContextResult(manifest, frames, availability)
        except Exception as exc:
            frames = context_frames(bundle, {})
            availability = tuple(source_availability(frames, bundle.manifest, symbols, config))
            failed = SourceAvailability(
                "context", "*", 0, None, "failed", f"{type(exc).__name__}: {exc}"
            )
            return SourceContextResult(bundle.manifest, frames, (*availability, failed))
    frames = context_frames(bundle, {})
    return SourceContextResult(
        bundle.manifest,
        frames,
        tuple(source_availability(frames, bundle.manifest, symbols, config)),
    )


def context_frames(
    bundle: SourceBundle, collected: dict[str, pl.DataFrame]
) -> dict[str, pl.DataFrame]:
    return {
        "books": _prefer_frame(collected.get("books"), bundle.books),
        "trades": _prefer_frame(collected.get("trades"), bundle.trades),
        "funding": _prefer_frame(collected.get("funding"), bundle.funding),
        "open_interest": _prefer_frame(collected.get("open_interest"), bundle.open_interest),
        "taker_volume": _prefer_frame(collected.get("taker_volume"), bundle.taker_volume),
        "long_short_ratios": _prefer_frame(
            collected.get("long_short_ratios"), bundle.long_short_ratios
        ),
        "messages": bundle.messages,
        "message_classifications": bundle.message_classifications,
    }


def merge_context_frames(
    bundle: SourceBundle, collected: dict[str, pl.DataFrame]
) -> dict[str, pl.DataFrame]:
    frames = context_frames(bundle, {})
    merged: dict[str, pl.DataFrame] = {}
    for family, cached in frames.items():
        artifact_name = SOURCE_FRAME_ARTIFACTS.get(family)
        spec = SOURCE_ARTIFACT_SPECS.get(artifact_name or "")
        incoming = collected.get(family)
        if artifact_name is None or spec is None:
            merged[family] = _prefer_frame(incoming, cached)
            continue
        merged_frame = merge_source_frames(artifact_name, cached, incoming, spec)
        merged[family] = (
            normalize_funding_artifact_frame(merged_frame) if family == "funding" else merged_frame
        )
    return merged


def write_context_frames(output_dir: Path, frames: dict[str, pl.DataFrame]) -> None:
    for family in (
        "books",
        "trades",
        "funding",
        "open_interest",
        "taker_volume",
        "long_short_ratios",
    ):
        frame = frames.get(family, pl.DataFrame())
        if frame.is_empty():
            continue
        artifact_name = SOURCE_FRAME_ARTIFACTS[family]
        write_frame_artifact(output_dir, SOURCE_ARTIFACT_SPECS[artifact_name], frame)


def concat_manifest(left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
    frames = [frame for frame in (left, right) if not frame.is_empty()]
    return pl.concat(frames, how="vertical_relaxed") if frames else pl.DataFrame()


def source_availability(
    frames: dict[str, pl.DataFrame],
    manifest: pl.DataFrame,
    symbols: tuple[str, ...],
    config: PotentialSourceConfig,
) -> list[SourceAvailability]:
    rows: list[SourceAvailability] = []
    symbol_frame = pl.DataFrame({"symbol": list(symbols)}, schema={"symbol": pl.String})
    status_by_family_symbol, warning_by_family_symbol = manifest_latest_maps(manifest)
    for family in CONTEXT_FAMILIES:
        frame = frames.get(family, pl.DataFrame())
        if frame.is_empty() or "symbol" not in frame.columns:
            availability = symbol_frame.with_columns(
                pl.lit(0, dtype=pl.Int64).alias("rows"),
                pl.lit(None, dtype=pl.Int64).alias("latest_timestamp"),
            )
        else:
            timestamp_col = "timestamp" if "timestamp" in frame.columns else "known_at_ms"
            grouped = frame.group_by("symbol").agg(
                pl.len().cast(pl.Int64).alias("rows"),
                pl.col(timestamp_col).max().cast(pl.Int64).alias("latest_timestamp")
                if timestamp_col in frame.columns
                else pl.lit(None, dtype=pl.Int64).alias("latest_timestamp"),
            )
            availability = symbol_frame.join(grouped, on="symbol", how="left").with_columns(
                pl.col("rows").fill_null(0),
            )
        availability_rows = availability.select("symbol", "rows", "latest_timestamp").iter_rows(
            named=True
        )
        for current in availability_rows:
            symbol = str(current["symbol"])
            if family in config.source.disabled_sources or symbol in config.source.disabled_symbols:
                rows.append(SourceAvailability(family, symbol, 0, None, "disabled", "disabled"))
                continue
            latest = current["latest_timestamp"]
            status = status_by_family_symbol.get((family, symbol))
            warning = warning_by_family_symbol.get((family, symbol), "")
            if status is None:
                status = "available" if int(current["rows"] or 0) else "missing"
            rows.append(
                SourceAvailability(
                    family,
                    symbol,
                    int(current["rows"] or 0),
                    int(latest) if latest is not None else None,
                    status,
                    warning,
                )
            )
    return rows


def manifest_latest_maps(
    manifest: pl.DataFrame,
) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], str]]:
    if manifest.is_empty() or not {"symbol", "source", "status", "warning"}.issubset(
        manifest.columns
    ):
        return {}, {}
    family_rows = []
    for row in manifest.select("timestamp", "symbol", "source", "status", "warning").iter_rows(
        named=True
    ):
        source = str(row["source"] or "")
        family = source_manifest_family(source)
        if not family:
            continue
        family_rows.append(
            {
                "timestamp": int(row["timestamp"] or 0),
                "symbol": str(row["symbol"] or ""),
                "source_family": family,
                "status": str(row["status"] or ""),
                "warning": str(row["warning"] or ""),
            }
        )
    if not family_rows:
        return {}, {}
    rows = (
        pl.DataFrame(family_rows)
        .sort("timestamp")
        .unique(subset=["symbol", "source_family"], keep="last")
    )
    status: dict[tuple[str, str], str] = {}
    warning: dict[tuple[str, str], str] = {}
    for row in rows.select("symbol", "source_family", "status", "warning").iter_rows(named=True):
        key = (str(row["source_family"] or ""), str(row["symbol"] or ""))
        status[key] = str(row["status"] or "")
        warning[key] = str(row["warning"] or "")
    return status, warning


def _prefer_frame(collected: pl.DataFrame | None, cached: pl.DataFrame) -> pl.DataFrame:
    return collected if collected is not None and not collected.is_empty() else cached
