"""Demand-first source collection planning and execution."""

from __future__ import annotations

import asyncio
import fnmatch
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
import polars as pl

from qooi.sources import (
    HOUR_MS,
    funding_min_rows,
    normalize_funding_artifact_frame,
    period_min_rows,
)
from qooi.sources.artifacts import SOURCE_ARTIFACT_SPECS, artifact_path
from qooi.sources.coverage import eligible_backfill_symbols, eligible_fetch_symbols
from qooi.sources.manifest import manifest_frame, now_ms, source_manifest_row
from qooi.sources.models import SourceResult
from qooi.sources.okx import (
    OKX_BASE_URL,
    fetch_okx_book_snapshot,
    fetch_okx_funding_history,
    fetch_okx_funding_rate,
    fetch_okx_long_short_account_ratio_contract,
    fetch_okx_open_interest_history,
    fetch_okx_recent_trades,
    fetch_okx_taker_volume_contract,
    fetch_okx_top_trader_long_short_account_ratio_contract,
    fetch_okx_top_trader_long_short_position_ratio_contract,
)

BookMode = Literal["snapshot", "sample", "off"]


@dataclass(frozen=True)
class SourceNeed:
    family: str
    symbols: tuple[str, ...]
    start_ms: int | None
    end_ms: int | None
    min_rows: int
    freshness_ms: int | None
    mode: Literal["snapshot", "history", "both", "local"]


@dataclass(frozen=True)
class SourceFetchPlan:
    family: str
    raw_source: str
    symbol: str
    start_ms: int | None
    end_ms: int | None
    limit: int
    reason: str


def source_needs_from_config(
    config: object,
    *,
    symbols: tuple[str, ...],
    context_symbols: tuple[str, ...],
    start_ms: int | None,
    end_ms: int | None,
) -> tuple[SourceNeed, ...]:
    target_symbols = context_symbols or symbols
    disabled = set(getattr(config, "disabled_sources", ()))
    freshness_ms = int(getattr(config, "max_source_staleness_hours", 0)) * HOUR_MS
    days = max(
        int(getattr(config, "days", 0)),
        int(getattr(config, "transition_history_days", 0)),
    )
    rubik_rows = period_min_rows(days, str(getattr(config, "rubik_period", "1H")))
    candidates = (
        SourceNeed("books", target_symbols, None, end_ms, 1, freshness_ms, "snapshot"),
        SourceNeed("trades", target_symbols, None, end_ms, 1, freshness_ms, "snapshot"),
        SourceNeed(
            "funding",
            target_symbols,
            start_ms,
            end_ms,
            funding_min_rows(days),
            freshness_ms,
            "both",
        ),
        SourceNeed(
            "open_interest", target_symbols, start_ms, end_ms, rubik_rows, freshness_ms, "history"
        ),
        SourceNeed(
            "taker_volume", target_symbols, start_ms, end_ms, rubik_rows, freshness_ms, "history"
        ),
        SourceNeed(
            "long_short_ratios",
            target_symbols,
            start_ms,
            end_ms,
            rubik_rows,
            freshness_ms,
            "history",
        ),
    )
    return tuple(need for need in candidates if need.family not in disabled)


@dataclass(frozen=True)
class SourceCollectRequest:
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
    target_source_start_ms: int | None = None
    target_source_end_ms: int | None = None
    funding_min_rows: int = 0
    rubik_min_rows: int = 0
    existing_frames: dict[str, pl.DataFrame] | None = None


@dataclass(frozen=True)
class SourceCollectResult:
    manifest: pl.DataFrame
    frames: dict[str, pl.DataFrame]


async def collect_source_context(request: SourceCollectRequest) -> SourceCollectResult:
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
        name: result.frame for (name, _collector), result in zip(collectors, gathered, strict=True)
    }
    manifests = [result.manifest for result in gathered if not result.manifest.is_empty()]
    manifest = pl.concat(manifests, how="diagonal_relaxed") if manifests else manifest_frame([])
    return SourceCollectResult(manifest=manifest, frames=frames)


async def _collect_books_source(
    client: httpx.AsyncClient,
    request: SourceCollectRequest,
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
    request: SourceCollectRequest,
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
    request: SourceCollectRequest,
    existing: pl.DataFrame,
    request_budget: asyncio.Semaphore,
    fetch: Callable[[str, str | None, str | None], Awaitable[SourceResult]] | None = None,
    current_fetch: Callable[[str], Awaitable[SourceResult]] | None = None,
) -> SourceResult:
    existing = normalize_funding_artifact_frame(existing)
    timestamp_col = "funding_time" if "funding_time" in existing.columns else "timestamp"
    historical_existing = _funding_history_frame(existing)
    disabled_symbols = tuple(
        symbol for symbol in request.symbols if _source_disabled(request, "funding", symbol)
    )
    active_symbols = tuple(
        symbol for symbol in request.symbols if symbol not in set(disabled_symbols)
    )
    disabled_manifest = _market_manifest_rows(
        disabled_symbols, source="funding", status="skipped", warning="funding_disabled"
    )
    eligible = _eligible_historical_symbols(
        historical_existing,
        active_symbols,
        refresh=request.refresh_context,
        request=request,
        min_rows=request.funding_min_rows,
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

    if fetch is None:

        async def fetch(symbol: str, after: str | None, before: str | None) -> SourceResult:
            result = await fetch_okx_funding_history(
                client,
                symbol,
                limit=request.funding_limit,
                after=after,
                before=before,
            )
            return SourceResult(_with_symbol(result.frame, symbol), result.manifest)

    if current_fetch is None:

        async def current_fetch(symbol: str) -> SourceResult:
            result = await fetch_okx_funding_rate(client, symbol)
            return SourceResult(_with_symbol(result.frame, symbol), result.manifest)

    async def run(symbol: str) -> SourceResult:
        return await _collect_historical_pages(
            symbol,
            historical_existing,
            request,
            fetch=fetch,
            cursor_fn=_incremental_funding_window,
            min_rows=request.funding_min_rows,
            page_limit=request.funding_limit,
            timestamp_col=timestamp_col,
        )

    fetched = await _collect_symbol_results(eligible, request_budget, run)
    current = await _collect_symbol_results(active_symbols, request_budget, current_fetch)
    return _combine_source_results(
        [*fetched, *current],
        local_frame=_local_frame(existing, local_symbols),
        local_manifest=_concat_frames(disabled_manifest, local),
    )


async def _collect_open_interest_source(
    client: httpx.AsyncClient,
    request: SourceCollectRequest,
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
    request: SourceCollectRequest,
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
    request: SourceCollectRequest,
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
    request: SourceCollectRequest,
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
    eligible = _eligible_historical_symbols(
        existing,
        active_symbols,
        refresh=request.refresh_context,
        request=request,
        min_rows=request.rubik_min_rows,
    )
    local = _local_frame_manifest(
        tuple(symbol for symbol in active_symbols if symbol not in set(eligible)),
        source=frame_source,
        existing=existing,
        endpoint=_source_endpoint(request, artifact_name),
    )
    local_symbols = tuple(symbol for symbol in active_symbols if symbol not in set(eligible))

    async def run(symbol: str) -> SourceResult:
        return await _collect_historical_pages(
            symbol,
            existing,
            request,
            fetch=fetch,
            cursor_fn=_incremental_rubik_window,
            min_rows=request.rubik_min_rows,
            page_limit=request.rubik_limit,
            timestamp_col="timestamp",
        )

    fetched = await _collect_symbol_results(eligible, request_budget, run)
    return _combine_source_results(
        fetched,
        local_frame=_local_frame(existing, local_symbols),
        local_manifest=_concat_frames(disabled_manifest, local),
    )


async def _collect_historical_pages(
    symbol: str,
    existing: pl.DataFrame,
    request: SourceCollectRequest,
    *,
    fetch: Callable[[str, str | None, str | None], Awaitable[SourceResult]],
    cursor_fn: Callable[[pl.DataFrame, str, SourceCollectRequest], tuple[str | None, str | None]],
    min_rows: int,
    page_limit: int,
    timestamp_col: str,
) -> SourceResult:
    merged = _local_frame(existing, (symbol,)) if not request.refresh_context else pl.DataFrame()
    pages: list[SourceResult] = []
    max_pages = _historical_max_pages(min_rows, page_limit)
    for _page_index in range(max_pages):
        if _historical_depth_satisfied(
            merged, symbol, request, min_rows, timestamp_col=timestamp_col
        ):
            break
        begin, end = cursor_fn(merged, symbol, request)
        if begin is None and end is None and not merged.is_empty():
            break
        before_earliest = _source_symbol_summary(merged, timestamp_col=timestamp_col).get(
            symbol, (0, None, None)
        )[1]
        result = await fetch(symbol, begin, end)
        page = _with_symbol(result.frame, symbol)
        pages.append(SourceResult(page, result.manifest))
        if page.is_empty():
            break
        page_earliest = _source_symbol_summary(page, timestamp_col=timestamp_col).get(
            symbol, (0, None, None)
        )[1]
        if page_earliest is None:
            break
        if before_earliest is not None and page_earliest >= before_earliest:
            break
        merged = _merge_historical_symbol_frame(merged, page, timestamp_col=timestamp_col)
    return _combine_source_results(
        pages,
        local_frame=_local_frame(existing, (symbol,))
        if not request.refresh_context
        else pl.DataFrame(),
        local_manifest=manifest_frame([]),
    )


def _historical_max_pages(min_rows: int, page_limit: int) -> int:
    if min_rows <= 0:
        return 1
    limit = max(1, page_limit)
    return max(1, min(25, (min_rows + limit - 1) // limit + 2))


def _funding_history_frame(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty() or "funding_source_kind" not in frame.columns:
        return frame
    return frame.filter(
        pl.col("funding_source_kind").is_null() | (pl.col("funding_source_kind") == "history")
    )


def _historical_depth_satisfied(
    frame: pl.DataFrame,
    symbol: str,
    request: SourceCollectRequest,
    min_rows: int,
    *,
    timestamp_col: str,
) -> bool:
    rows, earliest, latest = _source_symbol_summary(frame, timestamp_col=timestamp_col).get(
        symbol, (0, None, None)
    )
    if rows == 0:
        return False
    has_rows = min_rows <= 0 or rows >= min_rows
    has_start = request.target_source_start_ms is None or (
        earliest is not None and earliest <= request.target_source_start_ms
    )
    has_recent = request.target_source_end_ms is None or (
        latest is not None
        and latest >= request.target_source_end_ms - request.max_source_staleness_hours * HOUR_MS
    )
    return has_rows and has_start and has_recent


def _merge_historical_symbol_frame(
    existing: pl.DataFrame, page: pl.DataFrame, *, timestamp_col: str
) -> pl.DataFrame:
    frames = [frame for frame in (existing, page) if not frame.is_empty()]
    if not frames:
        return pl.DataFrame()
    merged = pl.concat(frames, how="diagonal_relaxed")
    if "symbol" in merged.columns and timestamp_col in merged.columns:
        merged = merged.unique(subset=["symbol", timestamp_col], keep="first")
    return merged.sort(timestamp_col) if timestamp_col in merged.columns else merged


def _incremental_rubik_window(
    existing: pl.DataFrame, symbol: str, request: SourceCollectRequest
) -> tuple[str | None, str | None]:
    if request.refresh_context:
        return None, None
    summary = _source_symbol_summary(existing).get(symbol)
    if summary is None:
        return None, None
    rows, earliest, latest = summary
    if earliest is None or latest is None:
        return None, None
    if request.target_source_start_ms is not None and earliest > request.target_source_start_ms:
        return None, str(earliest - 1)
    if request.rubik_min_rows and rows < request.rubik_min_rows:
        return None, str(earliest - 1)
    return str(latest + 1), str(request.target_source_end_ms or now_ms())


def _incremental_funding_window(
    existing: pl.DataFrame, symbol: str, request: SourceCollectRequest
) -> tuple[str | None, str | None]:
    timestamp_col = "funding_time" if "funding_time" in existing.columns else "timestamp"
    if request.refresh_context:
        return None, None
    summary = _source_symbol_summary(existing, timestamp_col=timestamp_col).get(symbol)
    if summary is None:
        return None, None
    rows, earliest, latest = summary
    if earliest is None or latest is None:
        return None, None
    if request.target_source_start_ms is not None and earliest > request.target_source_start_ms:
        return str(earliest), None
    if request.funding_min_rows and rows < request.funding_min_rows:
        return str(earliest), None
    return None, str(latest)


def _eligible_symbols(
    existing: pl.DataFrame,
    symbols: tuple[str, ...],
    *,
    refresh: bool,
    request: SourceCollectRequest,
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


def _eligible_historical_symbols(
    existing: pl.DataFrame,
    symbols: tuple[str, ...],
    *,
    refresh: bool,
    request: SourceCollectRequest,
    min_rows: int,
    timestamp_col: str = "timestamp",
) -> tuple[str, ...]:
    if request.target_source_start_ms is None:
        return _eligible_symbols(
            existing,
            symbols,
            refresh=refresh,
            request=request,
            timestamp_col=timestamp_col,
        )
    return eligible_backfill_symbols(
        existing,
        symbols,
        target_start_ms=request.target_source_start_ms,
        now_ms=request.target_source_end_ms or now_ms(),
        max_age_ms=request.max_source_staleness_hours * HOUR_MS,
        min_rows=min_rows,
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
        pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame(),
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


def _source_endpoint(request: SourceCollectRequest, artifact_name: str) -> str:
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
    request: SourceCollectRequest,
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


def _family_disabled(request: SourceCollectRequest, family: str) -> bool:
    return _matches_any(family, request.disabled_sources)


def _source_disabled(request: SourceCollectRequest, family: str, symbol: str = "") -> bool:
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
