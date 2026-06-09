"""OKX public context collection for scanner orchestration."""

from __future__ import annotations

import asyncio
import fnmatch
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
import polars as pl

from qooi.sources.artifacts import SOURCE_ARTIFACT_SPECS, artifact_path
from qooi.sources.coverage import eligible_fetch_symbols
from qooi.sources.manifest import manifest_frame, now_ms, source_manifest_row
from qooi.sources.models import SourceResult
from qooi.sources.okx import (
    OKX_BASE_URL,
    fetch_okx_book_snapshot,
    fetch_okx_funding_history,
    fetch_okx_long_short_account_ratio_contract,
    fetch_okx_open_interest_history,
    fetch_okx_recent_trades,
    fetch_okx_taker_volume_contract,
    fetch_okx_top_trader_long_short_account_ratio_contract,
    fetch_okx_top_trader_long_short_position_ratio_contract,
)

HOUR_MS = 60 * 60 * 1000
BookMode = Literal["snapshot", "sample", "off"]


@dataclass(frozen=True)
class MarketContextRequest:
    output_dir: Path
    symbols: tuple[str, ...]
    discovery: pl.DataFrame
    concurrency: int
    book_mode: BookMode
    book_depth: int
    max_source_staleness_hours: int
    trade_limit: int
    funding_limit: int
    rubik_period: str
    rubik_limit: int
    rubik_taker_unit: Literal["0", "1", "2"]
    disabled_sources: tuple[str, ...]
    disabled_symbols: tuple[str, ...]
    refresh_bars: bool = False
    refresh_trades: bool = False
    refresh_context: bool = False
    existing_frames: dict[str, pl.DataFrame] | None = None


@dataclass(frozen=True)
class MarketContextResult:
    manifest: pl.DataFrame
    frames: dict[str, pl.DataFrame]


async def collect_market_context(request: MarketContextRequest) -> MarketContextResult:
    existing_frames = request.existing_frames or {}
    request_budget = asyncio.Semaphore(max(1, request.concurrency))
    async with httpx.AsyncClient(base_url=OKX_BASE_URL, timeout=20.0) as client:
        collectors = [
            (
                "books",
                _collect_books_source(
                    client,
                    request,
                    existing_frames.get("books", pl.DataFrame()),
                    request_budget,
                ),
            ),
            (
                "trades",
                _collect_trades_source(
                    client,
                    request,
                    existing_frames.get("trades", pl.DataFrame()),
                    request_budget,
                ),
            ),
            (
                "funding",
                _collect_funding_source(
                    client,
                    request,
                    existing_frames.get("funding", pl.DataFrame()),
                    request_budget,
                ),
            ),
            (
                "open_interest",
                _collect_open_interest_source(
                    client,
                    request,
                    existing_frames.get("open_interest", pl.DataFrame()),
                    request_budget,
                ),
            ),
            (
                "taker_volume",
                _collect_taker_volume_source(
                    client,
                    request,
                    existing_frames.get("taker_volume", pl.DataFrame()),
                    request_budget,
                ),
            ),
            (
                "long_short_ratios",
                _collect_long_short_ratios_source(
                    client,
                    request,
                    existing_frames.get("long_short_ratios", pl.DataFrame()),
                    request_budget,
                ),
            ),
        ]
        gathered = await asyncio.gather(*(collector for _name, collector in collectors))
    frames = {
        name: result.frame
        for (name, _collector), result in zip(collectors, gathered, strict=True)
    }
    manifests = [result.manifest for result in gathered if not result.manifest.is_empty()]
    manifest = pl.concat(manifests, how="diagonal_relaxed") if manifests else manifest_frame([])
    return MarketContextResult(manifest=manifest, frames=frames)


async def _collect_books_source(
    client: httpx.AsyncClient,
    request: MarketContextRequest,
    existing: pl.DataFrame,
    request_budget: asyncio.Semaphore,
) -> SourceResult:
    if request.book_mode == "off":
        return SourceResult(
            pl.DataFrame(),
            _market_manifest_rows(request.symbols, source="books", status="skipped", warning=""),
        )
    disabled_symbols = tuple(
        symbol for symbol in request.symbols if _source_disabled(request, "books", symbol)
    )
    active_symbols = tuple(
        symbol for symbol in request.symbols if symbol not in set(disabled_symbols)
    )
    disabled_manifest = _market_manifest_rows(
        disabled_symbols, source="books", status="skipped", warning="books_disabled"
    )
    eligible = _eligible_symbols(
        existing, active_symbols, refresh=request.refresh_context, request=request
    )
    local = _local_frame_manifest(
        tuple(symbol for symbol in active_symbols if symbol not in set(eligible)),
        source="books",
        existing=existing,
        endpoint=_source_endpoint(request, "source_books"),
    )
    local_symbols = tuple(symbol for symbol in active_symbols if symbol not in set(eligible))

    async def fetch(symbol: str) -> SourceResult:
        result = await fetch_okx_book_snapshot(client, symbol, limit=request.book_depth)
        return SourceResult(_with_symbol(result.frame, symbol), result.manifest)

    fetched = await _collect_symbol_results(eligible, request_budget, fetch)
    return _combine_source_results(
        fetched,
        local_frame=_local_frame(existing, local_symbols),
        local_manifest=_concat_frames(disabled_manifest, local),
    )


async def _collect_trades_source(
    client: httpx.AsyncClient,
    request: MarketContextRequest,
    existing: pl.DataFrame,
    request_budget: asyncio.Semaphore,
) -> SourceResult:
    disabled_symbols = tuple(
        symbol for symbol in request.symbols if _source_disabled(request, "trades", symbol)
    )
    active_symbols = tuple(
        symbol for symbol in request.symbols if symbol not in set(disabled_symbols)
    )
    disabled_manifest = _market_manifest_rows(
        disabled_symbols, source="trades", status="skipped", warning="trades_disabled"
    )
    eligible = _eligible_symbols(
        existing, active_symbols, refresh=request.refresh_trades, request=request
    )
    local = _local_frame_manifest(
        tuple(symbol for symbol in active_symbols if symbol not in set(eligible)),
        source="trades",
        existing=existing,
        endpoint=_source_endpoint(request, "source_trades"),
    )
    local_symbols = tuple(symbol for symbol in active_symbols if symbol not in set(eligible))
    contract_metadata = _discovery_contract_metadata(request.discovery)

    async def fetch(symbol: str) -> SourceResult:
        metadata = contract_metadata.get(symbol, {})
        result = await fetch_okx_recent_trades(
            client,
            symbol,
            limit=request.trade_limit,
            contract_value=metadata.get("ct_val"),
            contract_value_currency=metadata.get("ct_val_ccy"),
            contract_base_currency=metadata.get("base_ccy"),
        )
        return SourceResult(_with_symbol(result.frame, symbol), result.manifest)

    fetched = await _collect_symbol_results(eligible, request_budget, fetch)
    return _combine_source_results(
        fetched,
        local_frame=_local_frame(existing, local_symbols),
        local_manifest=_concat_frames(disabled_manifest, local),
    )


async def _collect_funding_source(
    client: httpx.AsyncClient,
    request: MarketContextRequest,
    existing: pl.DataFrame,
    request_budget: asyncio.Semaphore,
) -> SourceResult:
    timestamp_col = "funding_time" if "funding_time" in existing.columns else "timestamp"
    disabled_symbols = tuple(
        symbol for symbol in request.symbols if _source_disabled(request, "funding", symbol)
    )
    active_symbols = tuple(
        symbol for symbol in request.symbols if symbol not in set(disabled_symbols)
    )
    disabled_manifest = _market_manifest_rows(
        disabled_symbols, source="funding", status="skipped", warning="funding_disabled"
    )
    eligible = _eligible_symbols(
        existing,
        active_symbols,
        refresh=request.refresh_context,
        request=request,
        timestamp_col=timestamp_col,
    )
    local = _local_frame_manifest(
        tuple(symbol for symbol in active_symbols if symbol not in set(eligible)),
        source="funding",
        existing=existing,
        endpoint=_source_endpoint(request, "source_funding"),
        timestamp_col=timestamp_col,
    )
    local_symbols = tuple(symbol for symbol in active_symbols if symbol not in set(eligible))

    async def fetch(symbol: str) -> SourceResult:
        result = await fetch_okx_funding_history(client, symbol, limit=request.funding_limit)
        return SourceResult(_with_symbol(result.frame, symbol), result.manifest)

    fetched = await _collect_symbol_results(eligible, request_budget, fetch)
    return _combine_source_results(
        fetched,
        local_frame=_local_frame(existing, local_symbols),
        local_manifest=_concat_frames(disabled_manifest, local),
    )


async def _collect_open_interest_source(
    client: httpx.AsyncClient,
    request: MarketContextRequest,
    existing: pl.DataFrame,
    request_budget: asyncio.Semaphore,
) -> SourceResult:
    return await _collect_rubik_source(
        client,
        request,
        existing,
        frame_source="open_interest_history",
        artifact_name="source_open_interest",
        request_budget=request_budget,
        fetch=lambda symbol, begin, end: fetch_okx_open_interest_history(
            client,
            symbol,
            period=request.rubik_period,
            limit=request.rubik_limit,
            begin=begin,
            end=end,
        ),
    )


async def _collect_taker_volume_source(
    client: httpx.AsyncClient,
    request: MarketContextRequest,
    existing: pl.DataFrame,
    request_budget: asyncio.Semaphore,
) -> SourceResult:
    return await _collect_rubik_source(
        client,
        request,
        existing,
        frame_source="taker_volume_contract",
        artifact_name="source_taker_volume",
        request_budget=request_budget,
        fetch=lambda symbol, begin, end: fetch_okx_taker_volume_contract(
            client,
            symbol,
            period=request.rubik_period,
            unit=request.rubik_taker_unit,
            limit=request.rubik_limit,
            begin=begin,
            end=end,
        ),
    )


async def _collect_long_short_ratios_source(
    client: httpx.AsyncClient,
    request: MarketContextRequest,
    existing: pl.DataFrame,
    request_budget: asyncio.Semaphore,
) -> SourceResult:
    async def fetch(symbol: str, begin: str | None, end: str | None) -> SourceResult:
        frame, manifest = await _fetch_long_short_ratios(client, request, symbol, begin, end)
        return SourceResult(_with_symbol(frame, symbol), manifest)

    return await _collect_rubik_source(
        client,
        request,
        existing,
        frame_source="long_short_ratio_contract",
        artifact_name="source_long_short_ratios",
        request_budget=request_budget,
        fetch=fetch,
    )


async def _collect_rubik_source(
    client: httpx.AsyncClient,
    request: MarketContextRequest,
    existing: pl.DataFrame,
    *,
    frame_source: str,
    artifact_name: str,
    request_budget: asyncio.Semaphore,
    fetch: Callable[[str, str | None, str | None], Awaitable[SourceResult]],
) -> SourceResult:
    _ = client
    family = artifact_name.removeprefix("source_")
    disabled_symbols = tuple(
        symbol
        for symbol in request.symbols
        if _source_disabled(request, family, symbol) or _source_disabled(request, "rubik", symbol)
    )
    active_symbols = tuple(
        symbol for symbol in request.symbols if symbol not in set(disabled_symbols)
    )
    disabled_manifest = (
        _market_manifest_rows(
            disabled_symbols,
            source=frame_source,
            status="skipped",
            warning=f"{family}_disabled",
        )
        if disabled_symbols
        else pl.DataFrame()
    )
    eligible = _eligible_symbols(
        existing, active_symbols, refresh=request.refresh_context, request=request
    )
    local = _local_frame_manifest(
        tuple(symbol for symbol in active_symbols if symbol not in set(eligible)),
        source=frame_source,
        existing=existing,
        endpoint=_source_endpoint(request, artifact_name),
    )
    local_symbols = tuple(symbol for symbol in active_symbols if symbol not in set(eligible))

    async def run(symbol: str) -> SourceResult:
        begin, end = _incremental_rubik_window(existing, symbol, request)
        result = await fetch(symbol, begin, end)
        return SourceResult(_with_symbol(result.frame, symbol), result.manifest)

    fetched = await _collect_symbol_results(eligible, request_budget, run)
    return _combine_source_results(
        fetched,
        local_frame=_local_frame(existing, local_symbols),
        local_manifest=_concat_frames(disabled_manifest, local),
    )


def _incremental_rubik_window(
    existing: pl.DataFrame, symbol: str, request: MarketContextRequest
) -> tuple[str | None, str | None]:
    if request.refresh_context:
        return None, None
    latest = _symbol_latest_timestamp(existing, symbol)
    if latest is None:
        return None, None
    return str(latest + 1), str(now_ms())


def _eligible_symbols(
    existing: pl.DataFrame,
    symbols: tuple[str, ...],
    *,
    refresh: bool,
    request: MarketContextRequest,
    timestamp_col: str = "timestamp",
) -> tuple[str, ...]:
    return eligible_fetch_symbols(
        existing,
        symbols,
        now_ms=now_ms(),
        max_age_ms=request.max_source_staleness_hours * HOUR_MS,
        refresh=refresh,
        timestamp_col=timestamp_col,
    )


async def _collect_symbol_results(
    symbols: tuple[str, ...], request_budget: asyncio.Semaphore, fetch
) -> list[SourceResult]:
    async def run(symbol: str) -> SourceResult:
        async with request_budget:
            return await fetch(symbol)

    return list(await asyncio.gather(*(run(symbol) for symbol in symbols))) if symbols else []


def _combine_source_results(
    results: list[SourceResult], *, local_frame: pl.DataFrame, local_manifest: pl.DataFrame
) -> SourceResult:
    frames = [local_frame, *(result.frame for result in results if not result.frame.is_empty())]
    frames = [frame for frame in frames if not frame.is_empty()]
    manifests = [local_manifest, *(result.manifest for result in results)]
    return SourceResult(
        pl.concat(frames, how="vertical_relaxed") if frames else pl.DataFrame(),
        _concat_frames(*manifests),
    )


def _local_frame_manifest(
    symbols: tuple[str, ...],
    *,
    source: str,
    existing: pl.DataFrame,
    endpoint: str,
    timestamp_col: str = "timestamp",
) -> pl.DataFrame:
    summary = _source_symbol_summary(existing, timestamp_col=timestamp_col)
    rows = []
    for symbol in symbols:
        item = summary.get(symbol, (0, None, None))
        rows.append(
            source_manifest_row(
                symbol=symbol,
                source=source,
                phase="collect-market",
                status="ok" if item[0] else "missing",
                backend="local",
                endpoint=endpoint,
                rows=item[0],
                range_start=item[1],
                range_end=item[2],
                warning="" if item[0] else f"{source}_missing",
            )
        )
    return manifest_frame(rows)


def _local_frame(existing: pl.DataFrame, symbols: tuple[str, ...]) -> pl.DataFrame:
    if existing.is_empty() or "symbol" not in existing.columns or not symbols:
        return pl.DataFrame()
    return existing.filter(pl.col("symbol").is_in(symbols))


def _source_endpoint(request: MarketContextRequest, artifact_name: str) -> str:
    return str(artifact_path(request.output_dir, SOURCE_ARTIFACT_SPECS[artifact_name]))


def _concat_frames(*frames: pl.DataFrame) -> pl.DataFrame:
    nonempty = [frame for frame in frames if not frame.is_empty()]
    return pl.concat(nonempty, how="diagonal_relaxed") if nonempty else manifest_frame([])


def _min_frame_value(frame: pl.DataFrame, col: str) -> int | None:
    return int(frame[col].min()) if col in frame.columns and not frame.is_empty() else None


def _max_frame_value(frame: pl.DataFrame, col: str) -> int | None:
    return int(frame[col].max()) if col in frame.columns and not frame.is_empty() else None


def _source_symbol_summary(
    frame: pl.DataFrame, *, timestamp_col: str = "timestamp"
) -> dict[str, tuple[int, int | None, int | None]]:
    if frame.is_empty() or "symbol" not in frame.columns:
        return {}
    if timestamp_col not in frame.columns:
        grouped = frame.group_by("symbol").agg(pl.len().alias("rows"))
        return {str(row["symbol"]): (int(row["rows"]), None, None) for row in grouped.to_dicts()}
    grouped = frame.group_by("symbol").agg(
        pl.len().alias("rows"),
        pl.col(timestamp_col).min().alias("range_start"),
        pl.col(timestamp_col).max().alias("range_end"),
    )
    return {
        str(row["symbol"]): (
            int(row["rows"]),
            int(row["range_start"]) if row["range_start"] is not None else None,
            int(row["range_end"]) if row["range_end"] is not None else None,
        )
        for row in grouped.to_dicts()
    }


def _symbol_latest_timestamp(frame: pl.DataFrame, symbol: str) -> int | None:
    return _source_symbol_summary(frame).get(symbol, (0, None, None))[2]


def _discovery_contract_metadata(discovery: pl.DataFrame) -> dict[str, dict[str, object]]:
    required = {"symbol", "ct_val", "ct_val_ccy", "base_ccy"}
    if discovery.is_empty() or "symbol" not in discovery.columns:
        return {}
    for column in required - set(discovery.columns):
        discovery = discovery.with_columns(pl.lit(None).alias(column))
    rows = discovery.select("symbol", "ct_val", "ct_val_ccy", "base_ccy").iter_rows(named=True)
    metadata: dict[str, dict[str, object]] = {}
    for row in rows:
        ct_val = row["ct_val"]
        metadata[str(row["symbol"])] = {
            "ct_val": float(ct_val) if ct_val is not None else None,
            "ct_val_ccy": str(row["ct_val_ccy"]) if row["ct_val_ccy"] is not None else None,
            "base_ccy": str(row["base_ccy"]) if row["base_ccy"] is not None else None,
        }
    return metadata


async def _fetch_long_short_ratios(
    client: httpx.AsyncClient,
    request: MarketContextRequest,
    symbol: str,
    begin: str | None,
    end: str | None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    calls = [
        fetch_okx_long_short_account_ratio_contract(
            client,
            symbol,
            period=request.rubik_period,
            limit=request.rubik_limit,
            begin=begin,
            end=end,
        ),
        fetch_okx_top_trader_long_short_account_ratio_contract(
            client,
            symbol,
            period=request.rubik_period,
            limit=request.rubik_limit,
            begin=begin,
            end=end,
        ),
        fetch_okx_top_trader_long_short_position_ratio_contract(
            client,
            symbol,
            period=request.rubik_period,
            limit=request.rubik_limit,
            begin=begin,
            end=end,
        ),
    ]
    results = await asyncio.gather(*calls)
    frames = [result.frame for result in results if not result.frame.is_empty()]
    combined = _join_ratio_frames(frames)
    endpoints = ";".join(
        dict.fromkeys(
            str(row.get("endpoint") or "")
            for result in results
            for row in result.manifest.to_dicts()
            if str(row.get("endpoint") or "")
        )
    )
    warnings = ";".join(
        dict.fromkeys(
            str(row.get("warning") or "")
            for result in results
            for row in result.manifest.to_dicts()
            if str(row.get("warning") or "")
        )
    )
    timestamp = (
        max(
            int(row.get("timestamp") or 0)
            for result in results
            for row in result.manifest.to_dicts()
        )
        + 1
    )
    manifest = manifest_frame(
        [
            source_manifest_row(
                symbol=symbol,
                source="long_short_ratio_contract",
                phase="collect-market",
                status="ok" if not combined.is_empty() else "missing",
                backend="okx_rest",
                endpoint=endpoints,
                rows=combined.height,
                range_start=_min_timestamp(combined),
                range_end=_max_timestamp(combined),
                warning=warnings
                if warnings
                else ("" if not combined.is_empty() else "long_short_ratio_contract_missing"),
                timestamp=timestamp,
            )
        ]
    )
    return combined, manifest


def _join_ratio_frames(frames: list[pl.DataFrame]) -> pl.DataFrame:
    frames = [frame for frame in frames if not frame.is_empty()]
    if not frames:
        return pl.DataFrame()
    out = frames[0]
    for frame in frames[1:]:
        out = out.join(frame, on="timestamp", how="full", coalesce=True)
    return out.sort("timestamp")


def _market_manifest_rows(
    symbols: tuple[str, ...], *, source: str, status: str, warning: str
) -> pl.DataFrame:
    return manifest_frame(
        [
            source_manifest_row(
                symbol=symbol,
                source=source,
                phase="collect-market",
                status=status,
                warning=warning,
            )
            for symbol in symbols
        ]
    )


def _min_timestamp(frame: pl.DataFrame) -> int | None:
    return (
        int(frame["timestamp"].min())
        if "timestamp" in frame.columns and not frame.is_empty()
        else None
    )


def _max_timestamp(frame: pl.DataFrame) -> int | None:
    return (
        int(frame["timestamp"].max())
        if "timestamp" in frame.columns and not frame.is_empty()
        else None
    )


def _family_disabled(request: MarketContextRequest, family: str) -> bool:
    return _matches_any(family, request.disabled_sources)


def _source_disabled(request: MarketContextRequest, family: str, symbol: str = "") -> bool:
    return _family_disabled(request, family) or bool(
        symbol and _matches_any(symbol, request.disabled_symbols)
    )


def _matches_any(value: str, patterns: tuple[str, ...]) -> bool:
    normalized = value.strip().lower()
    for pattern in patterns:
        candidate = pattern.strip().lower()
        if candidate and fnmatch.fnmatchcase(normalized, candidate):
            return True
    return False


def _with_symbol(frame: pl.DataFrame, symbol: str) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    return frame.with_columns(pl.lit(symbol).alias("symbol"))

