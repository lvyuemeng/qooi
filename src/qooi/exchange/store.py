"""OHLCV data cache — fetch, store as Parquet, load locally.

No API key needed. Caches repeated API calls and enables fast local
backtesting by avoiding network I/O on every run.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import polars as pl

from qooi.exchange.market import (
    AsyncExchange,
    CandleSource,
    CcxtBooksStream,
    CcxtSyncExchange,
    OkxAsyncExchange,
    OkxSyncExchange,
    SyncExchange,
)

CACHE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "cache"


@dataclass(frozen=True)
class HistoryRequest:
    inst_id: str
    bar: str = "1H"
    days: int = 30
    min_bars: int = 400
    refresh: bool = False
    source: CandleSource = "trade"
    max_newest_age_hours: float | None = None
    cache_only: bool = False


@dataclass(frozen=True)
class HistoryRefreshRequest(HistoryRequest):
    incremental: bool = True
    recent_limit: int = 300


@dataclass(frozen=True)
class FundingRequest:
    inst_id: str
    limit: int = 400
    refresh: bool = False


@dataclass(frozen=True)
class BooksRequest:
    inst_id: str
    samples: int = 0
    limit: int = 25
    refresh: bool = False
    append: bool = True
    transport: Literal["ws", "rest"] = "ws"
    params: dict | None = None
    every_seconds: float = 5.0


@dataclass(frozen=True)
class HistoryRefreshResult:
    request: HistoryRefreshRequest
    coverage: HistoryCoverage
    rows_written: int
    refreshed: bool
    path: Path
    error: str | None = None


def bar_freshness_threshold_hours(bar: str) -> float:
    unit = bar[-1].lower()
    value = float(bar[:-1]) if bar[:-1].isdigit() else 1.0
    if unit == "m":
        return max(value / 60.0 * 2.0, 0.5)
    if unit == "h":
        return value * 2.0
    if unit == "d":
        return value * 24.0 * 2.0
    return 2.0


@dataclass(frozen=True)
class CacheRefreshEvent:
    kind: Literal["started", "completed", "failed", "summary"]
    request: HistoryRefreshRequest | None
    result: HistoryRefreshResult | None
    completed: int
    total: int
    message: str


@dataclass(frozen=True)
class HistoryTarget:
    inst_id: str
    bar: str
    source: CandleSource
    requested_days: int
    requested_min_bars: int
    target_days: int
    target_bars: int
    target_since_ms: int


@dataclass(frozen=True)
class HistoryCoverage:
    inst_id: str
    bar: str
    source: CandleSource
    target: HistoryTarget
    actual_bars: int
    actual_start_ms: int | None
    actual_end_ms: int | None
    duplicate_timestamps: int
    gap_count: int
    newest_age_hours: float
    coverage_pct: float
    refreshed: bool = False
    notes: tuple[str, ...] = ()

    def note(self) -> str:
        return (
            f"{self.inst_id} {self.bar} source={self.source}: "
            f"target_bars={self.target.target_bars} "
            f"target_days={self.target.target_days} actual_bars={self.actual_bars} "
            f"coverage={self.coverage_pct:.1f}% "
            f"start={self.actual_start_ms or 'n/a'} end={self.actual_end_ms or 'n/a'} "
            f"dups={self.duplicate_timestamps} gaps={self.gap_count} "
            f"age_h={self.newest_age_hours:.1f} refreshed={'yes' if self.refreshed else 'no'}"
        )


def _safe_inst_id(inst_id: str) -> str:
    return inst_id.replace("-", "_").replace("/", "_").upper()


def _resource_path(
    inst_id: str,
    *,
    resource: Literal["bars", "funding", "books"] = "bars",
    bar: str = "1H",
    source: CandleSource = "trade",
) -> Path:
    safe = _safe_inst_id(inst_id)
    if resource == "funding":
        return CACHE_DIR / f"{safe}_FUNDING.parquet"
    if resource == "books":
        return CACHE_DIR / f"{safe}_BOOKS.parquet"
    timeframe = bar.replace(" ", "").upper()
    if source != "trade":
        return CACHE_DIR / f"{safe}_{source.upper()}_{timeframe}.parquet"
    return CACHE_DIR / f"{safe}_{timeframe}.parquet"


def _read_frame(path: Path, normalizer: Callable[[pl.DataFrame], pl.DataFrame]) -> pl.DataFrame:
    return normalizer(pl.read_parquet(path))


def _write_frame(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{id(df)}.tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


def _normalize_bars(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    if "volume" not in df.columns and "vol" in df.columns:
        df = df.rename({"vol": "volume"})
    cols = ("timestamp", "datetime", "open", "high", "low", "close", "volume")
    keep = [col for col in cols if col in df.columns]
    return df.select(keep).sort("timestamp")


def _merge_bars(left: pl.DataFrame, right: pl.DataFrame, target_bars: int) -> pl.DataFrame:
    frames = [frame for frame in (left, right) if not frame.is_empty()]
    if not frames:
        return pl.DataFrame()
    common_cols = set(frames[0].columns)
    for frame in frames[1:]:
        common_cols &= set(frame.columns)
    ordered_cols = [
        col
        for col in ("timestamp", "datetime", "open", "high", "low", "close", "volume")
        if col in common_cols
    ]
    merged = pl.concat([frame.select(ordered_cols) for frame in frames])
    merged = merged.unique(subset=["timestamp"]).sort("timestamp")
    if merged.height > target_bars:
        merged = merged.tail(target_bars)
    return merged


def _empty_funding() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "timestamp": pl.Int64,
            "funding_rate": pl.Float64,
            "funding_time": pl.Int64,
        }
    )


def _normalize_funding(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return _empty_funding()
    normalized = df
    if "funding_time" not in normalized.columns:
        normalized = normalized.with_columns(pl.col("timestamp").alias("funding_time"))
    return (
        normalized.select(["timestamp", "funding_rate", "funding_time"])
        .unique(subset=["timestamp"])
        .sort("timestamp")
    )


def _empty_books() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "timestamp": pl.Int64,
            "ob_bid_price": pl.Float64,
            "ob_ask_price": pl.Float64,
            "ob_bid_vol_5": pl.Float64,
            "ob_ask_vol_5": pl.Float64,
            "ob_bid_vol_25": pl.Float64,
            "ob_ask_vol_25": pl.Float64,
            "ob_bid_vol": pl.Float64,
            "ob_ask_vol": pl.Float64,
            "ob_imbalance_5": pl.Float64,
            "ob_imbalance_25": pl.Float64,
        }
    )


def _normalize_books(df: pl.DataFrame) -> pl.DataFrame:
    required = list(_empty_books().schema.keys())
    if df.is_empty():
        return _empty_books()
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Books frame missing columns: {missing}")
    return df.select(required).unique(subset=["timestamp"]).sort("timestamp")


class CacheStore:
    """Synchronous cache store with uniform resource methods."""

    def __init__(self, exchange: SyncExchange | None = None) -> None:
        self._exchange = exchange or OkxSyncExchange(book_fallback=CcxtSyncExchange("okx"))
        self._owns_exchange = exchange is None
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> CacheStore:
        if self._owns_exchange:
            self._exchange.__enter__()
        return self

    def __exit__(self, *_: object) -> None:
        if self._owns_exchange:
            self._exchange.__exit__(*_)  # type: ignore[misc]

    def bars(self, request: HistoryRequest) -> tuple[pl.DataFrame, HistoryCoverage]:
        target = plan_history(
            request.inst_id,
            request.bar,
            days=request.days,
            min_bars=request.min_bars,
            source=request.source,
        )
        path = _resource_path(
            request.inst_id, resource="bars", bar=request.bar, source=request.source
        )
        if request.cache_only and not path.exists():
            coverage = validate_history(
                pl.DataFrame(),
                target,
                extra_notes=("cache_only=yes", "cache_missing", f"source={request.source}"),
            )
            return pl.DataFrame(), coverage
        refreshed = request.refresh or not path.exists()
        if refreshed:
            frame = self._update_bars(request, path, target)
        else:
            frame = _read_frame(path, _normalize_bars)
        notes = (f"source={request.source}", *(self._exchange.last_bars_audit if refreshed else ()))
        return frame, validate_history(frame, target, refreshed=refreshed, extra_notes=notes)

    def audit_bars(
        self,
        requests: Iterable[HistoryRequest],
        *,
        min_coverage_pct: float = 0.0,
    ) -> pl.DataFrame:
        rows: list[dict[str, object]] = []
        for request in requests:
            try:
                _df, coverage = self.bars(request)
                rows.append(history_coverage_row(coverage, min_coverage_pct=min_coverage_pct))
            except Exception as exc:
                rows.append(history_coverage_error_row(request, exc))
        return pl.DataFrame(rows, schema=HISTORY_COVERAGE_SCHEMA)

    def funding(self, request: FundingRequest) -> pl.DataFrame:
        path = _resource_path(request.inst_id, resource="funding")
        if path.exists() and not request.refresh:
            return _read_frame(path, _normalize_funding)
        frame = _normalize_funding(self._exchange.funding(request.inst_id, request.limit))
        _write_frame(frame, _resource_path(request.inst_id, resource="funding"))
        return frame

    def books(self, request: BooksRequest) -> pl.DataFrame:
        path = _resource_path(request.inst_id, resource="books")
        if path.exists() and not request.refresh and request.samples <= 0:
            return _read_frame(path, _normalize_books)
        if request.transport == "ws":
            raise RuntimeError("Use AsyncCacheStore.books() for websocket books collection")
        current = _normalize_books(pl.DataFrame(self._collect_books_rest(request)))
        return self._write_books(request.inst_id, current, append=request.append)

    def _update_bars(
        self, request: HistoryRequest, path: Path, target: HistoryTarget
    ) -> pl.DataFrame:
        existing = _read_frame(path, _normalize_bars) if path.exists() else pl.DataFrame()
        since = datetime.fromtimestamp(target.target_since_ms / 1000, tz=UTC).strftime("%Y-%m-%d")
        frame = self._exchange.bars_since(
            request.inst_id,
            bar=request.bar,
            since=since,
            limit=target.target_bars,
            source=request.source,
        )
        frame = _merge_bars(existing, _normalize_bars(frame), target.target_bars)
        try:
            recent = self._exchange.bars(
                request.inst_id, bar=request.bar, limit=300, source=request.source
            )
        except Exception:
            recent = pl.DataFrame()
        frame = _merge_bars(frame, _normalize_bars(recent), target.target_bars)
        _write_frame(frame, path)
        return _normalize_bars(frame)

    def _collect_books_rest(self, request: BooksRequest) -> list[dict]:
        rows: list[dict] = []
        for idx in range(max(0, request.samples)):
            snap = self._exchange.book(request.inst_id, limit=request.limit)
            row = snap.to_row()
            if row["timestamp"] <= 0:
                row["timestamp"] = int(datetime.now(UTC).timestamp() * 1000)
            rows.append(row)
            if idx + 1 < request.samples and request.every_seconds > 0:
                time.sleep(request.every_seconds)
        return rows

    def _write_books(self, inst_id: str, frame: pl.DataFrame, *, append: bool) -> pl.DataFrame:
        path = _resource_path(inst_id, resource="books")
        current = _normalize_books(frame)
        existing_path = _resource_path(inst_id, resource="books")
        if append and existing_path.exists():
            existing = _read_frame(existing_path, _normalize_books)
            current = _normalize_books(pl.concat([existing, current], how="vertical"))
        _write_frame(current, path)
        return current


class AsyncCacheStore:
    """Asynchronous cache store with uniform resource methods."""

    def __init__(self, exchange: AsyncExchange | None = None) -> None:
        self._exchange = exchange or OkxAsyncExchange(stream=CcxtBooksStream("okx"))
        self._owns_exchange = exchange is None
        self._path_locks: dict[Path, asyncio.Lock] = {}
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    async def __aenter__(self) -> AsyncCacheStore:
        if self._owns_exchange:
            await self._exchange.__aenter__()
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._owns_exchange:
            await self._exchange.__aexit__(*_)  # type: ignore[misc]

    async def bars(self, request: HistoryRequest) -> tuple[pl.DataFrame, HistoryCoverage]:
        target = plan_history(
            request.inst_id,
            request.bar,
            days=request.days,
            min_bars=request.min_bars,
            source=request.source,
        )
        path = _resource_path(
            request.inst_id, resource="bars", bar=request.bar, source=request.source
        )
        refreshed = request.refresh or not path.exists()
        if refreshed:
            refresh_request = (
                request
                if isinstance(request, HistoryRefreshRequest)
                else HistoryRefreshRequest(
                    request.inst_id,
                    request.bar,
                    request.days,
                    request.min_bars,
                    True,
                    request.source,
                    request.max_newest_age_hours,
                )
            )
            result = await self._update_bars(refresh_request, path)
            if result.error:
                return pl.DataFrame(), result.coverage
            frame = _read_frame(path, _normalize_bars)
            return frame, result.coverage
        frame = await asyncio.to_thread(_read_frame, path, _normalize_bars)
        coverage = validate_history(frame, target, extra_notes=(f"source={request.source}",))
        max_newest_age_hours = (
            request.max_newest_age_hours
            if request.max_newest_age_hours is not None
            else bar_freshness_threshold_hours(request.bar)
        )
        if request.cache_only:
            return frame, validate_history(
                frame,
                target,
                extra_notes=("cache_only=yes", f"source={request.source}"),
            )
        if not frame.is_empty() and coverage.newest_age_hours > max_newest_age_hours:
            refresh_request = (
                request
                if isinstance(request, HistoryRefreshRequest)
                else HistoryRefreshRequest(
                    request.inst_id,
                    request.bar,
                    request.days,
                    request.min_bars,
                    True,
                    request.source,
                    request.max_newest_age_hours,
                )
            )
            result = await self._update_bars(replace(refresh_request, refresh=True), path)
            if result.error:
                return pl.DataFrame(), result.coverage
            frame = _read_frame(path, _normalize_bars)
            return frame, result.coverage
        return frame, coverage

    async def funding(self, request: FundingRequest) -> pl.DataFrame:
        path = _resource_path(request.inst_id, resource="funding")
        if path.exists() and not request.refresh:
            return await asyncio.to_thread(_read_frame, path, _normalize_funding)
        frame = _normalize_funding(await self._exchange.funding(request.inst_id, request.limit))
        await asyncio.to_thread(
            _write_frame,
            frame,
            _resource_path(request.inst_id, resource="funding"),
        )
        return frame

    async def books(self, request: BooksRequest) -> pl.DataFrame:
        path = _resource_path(request.inst_id, resource="books")
        if path.exists() and not request.refresh and request.samples <= 0:
            return await asyncio.to_thread(_read_frame, path, _normalize_books)
        current = _normalize_books(pl.DataFrame(await self._collect_books(request)))
        return await asyncio.to_thread(
            self._write_books,
            request.inst_id,
            current,
            append=request.append,
        )

    async def many(
        self,
        requests: list[HistoryRefreshRequest] | tuple[HistoryRefreshRequest, ...],
        *,
        concurrency: int = 3,
        fail_fast: bool = False,
    ) -> tuple[HistoryRefreshResult, ...]:
        results = []
        async for event in self.stream_many(
            requests,
            concurrency=concurrency,
            fail_fast=fail_fast,
        ):
            if event.result is not None:
                results.append(event.result)
        return tuple(results)

    async def audit_bars_many(
        self,
        requests: Iterable[HistoryRefreshRequest],
        *,
        concurrency: int = 3,
        min_coverage_pct: float = 0.0,
    ) -> pl.DataFrame:
        results = await self.many(tuple(requests), concurrency=concurrency, fail_fast=False)
        return history_coverage_frame(
            (result.coverage for result in results), min_coverage_pct=min_coverage_pct
        )

    async def stream_many(
        self,
        requests: list[HistoryRefreshRequest] | tuple[HistoryRefreshRequest, ...],
        *,
        concurrency: int = 3,
        fail_fast: bool = False,
    ) -> AsyncIterator[CacheRefreshEvent]:
        unique = tuple(dict.fromkeys(requests))
        total = len(unique)
        if not unique:
            yield CacheRefreshEvent("summary", None, None, 0, 0, "cache refresh: no requests")
            return
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def _run(request: HistoryRefreshRequest) -> HistoryRefreshResult:
            async with semaphore:
                path = _resource_path(
                    request.inst_id, resource="bars", bar=request.bar, source=request.source
                )
                result = await self._update_bars(request, path)
                if fail_fast and result.error:
                    raise RuntimeError(result.error)
                return result

        for request in unique:
            yield CacheRefreshEvent(
                "started",
                request,
                None,
                0,
                total,
                f"cache refresh queued {request.inst_id} {request.bar} source={request.source}",
            )
        completed = 0
        tasks = [asyncio.create_task(_run(request)) for request in unique]
        for task in asyncio.as_completed(tasks):
            result = await task
            completed += 1
            kind = "failed" if result.error else "completed"
            yield CacheRefreshEvent(
                kind,
                result.request,
                result,
                completed,
                total,
                _refresh_event_message(result, completed, total),
            )
        yield CacheRefreshEvent(
            "summary",
            None,
            None,
            completed,
            total,
            f"cache refresh complete {completed}/{total}",
        )

    async def _update_bars(
        self, request: HistoryRefreshRequest, path: Path
    ) -> HistoryRefreshResult:
        lock = self._path_locks.setdefault(path, asyncio.Lock())
        async with lock:
            try:
                frame, coverage, notes = await self._refresh_bars_frame(request, path)
                await asyncio.to_thread(_write_frame, frame, path)
                coverage = validate_history(
                    frame,
                    coverage.target,
                    refreshed=True,
                    extra_notes=(*notes, *self._exchange.last_bars_audit),
                )
                return HistoryRefreshResult(request, coverage, frame.height, True, path)
            except Exception as exc:
                target = plan_history(
                    request.inst_id,
                    request.bar,
                    days=request.days,
                    min_bars=request.min_bars,
                    source=request.source,
                )
                coverage = validate_history(
                    pl.DataFrame(),
                    target,
                    refreshed=False,
                    extra_notes=(f"refresh_error={type(exc).__name__}", str(exc)),
                )
                return HistoryRefreshResult(request, coverage, 0, False, path, error=str(exc))

    async def _refresh_bars_frame(
        self, request: HistoryRefreshRequest, path: Path
    ) -> tuple[pl.DataFrame, HistoryCoverage, tuple[str, ...]]:
        target = plan_history(
            request.inst_id,
            request.bar,
            days=request.days,
            min_bars=request.min_bars,
            source=request.source,
        )
        existing = pl.DataFrame()
        if path.exists():
            try:
                existing = await asyncio.to_thread(_read_frame, path, _normalize_bars)
            except Exception:
                existing = pl.DataFrame()
        existing_coverage = validate_history(existing, target) if not existing.is_empty() else None
        starts_before_target = bool(
            existing_coverage
            and existing_coverage.actual_start_ms is not None
            and existing_coverage.actual_start_ms
            <= target.target_since_ms + _bar_interval_ms(target.bar)
        )
        skip_since = bool(
            request.incremental
            and existing_coverage is not None
            and existing_coverage.actual_bars >= target.target_bars
            and starts_before_target
            and existing_coverage.duplicate_timestamps == 0
            and existing_coverage.gap_count == 0
        )

        frame = existing if skip_since else pl.DataFrame()
        since = datetime.fromtimestamp(target.target_since_ms / 1000, tz=UTC).strftime("%Y-%m-%d")
        if not skip_since:
            frame = await self._exchange.bars_since(
                request.inst_id,
                bar=request.bar,
                since=since,
                limit=target.target_bars,
                source=request.source,
            )
            frame = _merge_bars(existing, _normalize_bars(frame), target.target_bars)

        try:
            recent = await self._exchange.bars(
                request.inst_id,
                bar=request.bar,
                limit=request.recent_limit,
                source=request.source,
            )
        except Exception:
            recent = pl.DataFrame()
        frame = _merge_bars(frame, _normalize_bars(recent), target.target_bars)
        notes = (
            f"source={request.source}",
            "refresh_transport=async",
            f"refresh_mode={'incremental' if request.incremental else 'full'}",
            f"refresh_skipped_history={'yes' if skip_since else 'no'}",
            f"refresh_existing_bars={existing.height}",
            f"refresh_recent_bars={recent.height}",
            f"refresh_older_bars={0 if skip_since else frame.height}",
        )
        return _normalize_bars(frame), validate_history(frame, target, refreshed=True), notes

    async def _collect_books(self, request: BooksRequest) -> list[dict]:
        if request.samples <= 0:
            return []
        rows: list[dict] = []
        if request.transport == "rest":
            for idx in range(request.samples):
                snap = await self._exchange.book(request.inst_id, limit=request.limit)
                row = snap.to_row()
                if row["timestamp"] <= 0:
                    row["timestamp"] = int(datetime.now(UTC).timestamp() * 1000)
                rows.append(row)
                if idx + 1 < request.samples and request.every_seconds > 0:
                    await asyncio.sleep(request.every_seconds)
            return rows
        if request.transport != "ws":
            raise ValueError(f"Unsupported books transport: {request.transport}")
        async for snap in self._exchange.books(
            request.inst_id,
            limit=request.limit,
            params=request.params,
        ):
            row = snap.to_row()
            if row["timestamp"] <= 0:
                row["timestamp"] = int(datetime.now(UTC).timestamp() * 1000)
            rows.append(row)
            if len(rows) >= request.samples:
                break
        return rows

    def _write_books(self, inst_id: str, frame: pl.DataFrame, *, append: bool) -> pl.DataFrame:
        path = _resource_path(inst_id, resource="books")
        current = _normalize_books(frame)
        existing_path = _resource_path(inst_id, resource="books")
        if append and existing_path.exists():
            existing = _read_frame(existing_path, _normalize_books)
            current = _normalize_books(pl.concat([existing, current], how="vertical"))
        _write_frame(current, path)
        return current


def plan_history(
    inst_id: str,
    bar: str,
    *,
    days: int,
    min_bars: int,
    source: CandleSource = "trade",
) -> HistoryTarget:
    interval_ms = _bar_interval_ms(bar) or 86_400_000
    days_from_bars = max(1, int((min_bars * interval_ms + 86_400_000 - 1) / 86_400_000))
    target_bars = max(1, min_bars)
    target_days = max(days, days_from_bars)
    target_since_ms = int((datetime.now(UTC) - timedelta(days=target_days)).timestamp() * 1000)
    return HistoryTarget(
        inst_id=inst_id,
        bar=bar,
        source=source,
        requested_days=days,
        requested_min_bars=min_bars,
        target_days=target_days,
        target_bars=target_bars,
        target_since_ms=target_since_ms,
    )


def validate_history(
    df: pl.DataFrame,
    target: HistoryTarget,
    *,
    refreshed: bool = False,
    extra_notes: tuple[str, ...] = (),
) -> HistoryCoverage:
    if df.is_empty() or "timestamp" not in df.columns:
        return HistoryCoverage(
            inst_id=target.inst_id,
            bar=target.bar,
            source=target.source,
            target=target,
            actual_bars=0,
            actual_start_ms=None,
            actual_end_ms=None,
            duplicate_timestamps=0,
            gap_count=0,
            newest_age_hours=0.0,
            coverage_pct=0.0,
            refreshed=refreshed,
            notes=("empty_or_missing_timestamp", *extra_notes),
        )
    ts = [int(v) for v in df["timestamp"].to_list() if v is not None]
    unique_ts = len(set(ts))
    duplicate_timestamps = len(ts) - unique_ts
    expected_ms = _bar_interval_ms(target.bar)
    gap_count = sum(
        1 for i in range(1, len(ts)) if expected_ms > 0 and ts[i] - ts[i - 1] != expected_ms
    )
    newest_age_hours = max(0.0, (datetime.now(UTC).timestamp() * 1000 - ts[-1]) / 3_600_000.0)
    coverage_pct = len(ts) / max(target.target_bars, 1) * 100.0
    notes = []
    if len(ts) < target.target_bars:
        notes.append("below_target_bars")
    if ts[0] > target.target_since_ms + expected_ms:
        notes.append("starts_after_target_since")
    if duplicate_timestamps:
        notes.append("duplicate_timestamps")
    if gap_count:
        notes.append("timeframe_gaps")
    return HistoryCoverage(
        inst_id=target.inst_id,
        bar=target.bar,
        source=target.source,
        target=target,
        actual_bars=df.height,
        actual_start_ms=ts[0] if ts else None,
        actual_end_ms=ts[-1] if ts else None,
        duplicate_timestamps=duplicate_timestamps,
        gap_count=gap_count,
        newest_age_hours=newest_age_hours,
        coverage_pct=coverage_pct,
        refreshed=refreshed,
        notes=(*notes, *extra_notes),
    )


HISTORY_COVERAGE_SCHEMA = {
    "status": pl.Utf8,
    "instrument": pl.Utf8,
    "bar": pl.Utf8,
    "actual_bars": pl.Int64,
    "target_bars": pl.Int64,
    "coverage_pct": pl.Float64,
    "start_ms": pl.Int64,
    "end_ms": pl.Int64,
    "notes": pl.Utf8,
}


def history_coverage_frame(
    coverages: Iterable[HistoryCoverage],
    *,
    min_coverage_pct: float = 0.0,
) -> pl.DataFrame:
    rows = [history_coverage_row(item, min_coverage_pct=min_coverage_pct) for item in coverages]
    return pl.DataFrame(rows, schema=HISTORY_COVERAGE_SCHEMA)


def history_coverage_row(
    coverage: HistoryCoverage,
    *,
    min_coverage_pct: float = 0.0,
) -> dict[str, object]:
    return {
        "status": "PASS" if coverage.coverage_pct >= min_coverage_pct else "LOW",
        "instrument": coverage.inst_id,
        "bar": coverage.bar,
        "actual_bars": coverage.actual_bars,
        "target_bars": coverage.target.target_bars,
        "coverage_pct": coverage.coverage_pct,
        "start_ms": coverage.actual_start_ms,
        "end_ms": coverage.actual_end_ms,
        "notes": ",".join(coverage.notes),
    }


def history_coverage_error_row(
    request: HistoryRequest,
    error: Exception,
) -> dict[str, object]:
    return {
        "status": "ERROR",
        "instrument": request.inst_id,
        "bar": request.bar,
        "actual_bars": 0,
        "target_bars": 0,
        "coverage_pct": 0.0,
        "start_ms": None,
        "end_ms": None,
        "notes": str(error),
    }


def _refresh_event_message(result: HistoryRefreshResult, completed: int, total: int) -> str:
    coverage = result.coverage
    fetch_pages = _note_value(coverage.notes, "fetch_pages")
    fetch_stop = _note_value(coverage.notes, "fetch_stop")
    oldest_ts = _note_value(coverage.notes, "fetch_oldest_ts")
    since_ms = _note_value(coverage.notes, "fetch_since_ms")
    status = "failed" if result.error else "done"
    parts = [
        f"cache refresh {completed}/{total} {status}",
        f"{result.request.inst_id} {result.request.bar} source={result.request.source}",
        f"rows={result.rows_written}",
        f"coverage={coverage.coverage_pct:.1f}%",
    ]
    if fetch_pages:
        parts.append(f"pages={fetch_pages}")
    if fetch_stop:
        parts.append(f"stop={fetch_stop}")
    if oldest_ts:
        parts.append(f"oldest={oldest_ts}")
    if since_ms:
        parts.append(f"since={since_ms}")
    if result.error:
        parts.append(f"error={result.error}")
    return " ".join(parts)


def _note_value(notes: tuple[str, ...], key: str) -> str:
    prefix = f"{key}="
    for note in notes:
        if note.startswith(prefix):
            return note.removeprefix(prefix)
    return ""


def _bar_interval_ms(bar: str) -> int:
    normalized = bar.replace(" ", "")
    unit = normalized[-1:].upper()
    try:
        value = int(normalized[:-1])
    except ValueError:
        return 0
    if unit == "M":
        return value * 60_000
    if unit == "H":
        return value * 3_600_000
    if unit == "D":
        return value * 86_400_000
    if unit == "W":
        return value * 7 * 86_400_000
    return 0
