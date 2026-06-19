"""Market data load boundary — cache -> plan -> page -> merge -> cache -> health."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import polars as pl

from qooi.pipeline import HOUR_MS, now_ms
from qooi.pipeline.coverage import (
    CoveragePlan,
    CoverageRunPolicy,
    allocate_coverage,
    bar_spec,
    coin_list_times,
    plan_product_coverage,
    source_spec,
)
from qooi.pipeline.io import load_frame, merge_frames, save_frame
from qooi.pipeline.types import FrameHealth, ProductResult
from qooi.transport.core import sanitize_error
from qooi.transport.okx import OkxClient

LoadMode = Literal["incremental", "cache_only", "force"]
SourceName = Literal[
    "books",
    "trades",
    "funding",
    "open_interest",
    "taker_volume",
    "long_short_ratios",
]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "data" / "cache"


@dataclass(frozen=True)
class BarLoadRequest:
    symbols: tuple[str, ...]
    timeframes: tuple[str, ...]
    target_days: int
    max_staleness_hours: int
    latest_staleness_hours: int | None = None
    refresh_mode: LoadMode = "incremental"


@dataclass(frozen=True)
class SourceProductLoadRequest:
    name: SourceName
    limit: int
    period: str = "1H"
    unit: str = "2"
    max_staleness_hours: int | None = None


@dataclass(frozen=True)
class SourceLoadRequest:
    symbols: tuple[str, ...]
    products: tuple[SourceProductLoadRequest, ...]
    target_days: int
    max_staleness_hours: int


@dataclass(frozen=True)
class MarketLoadRequest:
    bars: BarLoadRequest
    sources: SourceLoadRequest


@dataclass(frozen=True)
class MarketLoadPolicy:
    cache_root: Path = DEFAULT_CACHE_ROOT
    coverage: CoverageRunPolicy = field(default_factory=CoverageRunPolicy)

    @property
    def concurrency(self) -> int:
        return self.coverage.concurrency

    @property
    def allow_partial(self) -> bool:
        return self.coverage.allow_partial


@dataclass(frozen=True)
class LoadStats:
    bar_pages: int = 0
    source_pages: dict[str, int] = field(default_factory=dict)
    provider_bounded: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class LoadedMarketFrames:
    bar_frames: dict[tuple[str, str], pl.DataFrame]
    source_frames: dict[str, pl.DataFrame]
    products: dict[str, ProductResult]
    stats: LoadStats
    coverage_before: dict[str, CoveragePlan] = field(default_factory=dict)
    coverage_after: dict[str, CoveragePlan] = field(default_factory=dict)


async def load_market(
    okx: OkxClient,
    request: MarketLoadRequest,
    policy: MarketLoadPolicy,
    instrument_frame: pl.DataFrame | None = None,
) -> LoadedMarketFrames:
    listed = coin_list_times(instrument_frame if instrument_frame is not None else pl.DataFrame())
    bounded: set[tuple[str, str, str]] = _load_provider_bounds(policy.cache_root)
    bar_cache = {
        timeframe: _bar_cache_frames(policy.cache_root, request.bars.symbols, timeframe)
        for timeframe in request.bars.timeframes
    }
    if request.bars.refresh_mode == "force":
        bar_cache = {
            timeframe: {symbol: pl.DataFrame() for symbol in request.bars.symbols}
            for timeframe in request.bars.timeframes
        }
    source_cache = {
        product.name: _load_source_cache(policy.cache_root, product.name)
        for product in request.sources.products
    }
    before_feasible = _plan_market_coverage(
        request, policy, listed, bounded, bar_cache, source_cache
    )
    before = allocate_coverage(before_feasible, policy.coverage)

    bar_pages = 0
    if request.bars.refresh_mode != "cache_only":
        bar_pages = await _execute_bar_jobs(
            okx, before.get("bars"), request.bars, policy, bar_cache, bounded
        )
    source_frames, source_pages = await _execute_source_jobs(
        okx, before, request.sources, policy, source_cache, bounded
    )

    _save_provider_bounds(policy.cache_root, bounded)
    after_feasible = _plan_market_coverage(
        request, policy, listed, bounded, bar_cache, source_cache
    )
    after = allocate_coverage(
        after_feasible, CoverageRunPolicy(max_requests=0, allow_partial=policy.allow_partial)
    )
    bar_frames = _bar_output_frames(bar_cache, request.bars)
    bar_product = _bars_product(bar_frames, request.bars)
    source_products = _source_products(source_frames, request.sources)
    return LoadedMarketFrames(
        bar_frames=bar_frames,
        source_frames=source_frames,
        products={"bars": bar_product, **source_products},
        stats=LoadStats(
            bar_pages=bar_pages,
            source_pages=source_pages,
            provider_bounded=_request_bound_counts(request, bounded),
        ),
        coverage_before=before,
        coverage_after=after,
    )


def _plan_market_coverage(
    request: MarketLoadRequest,
    policy: MarketLoadPolicy,
    listed: dict[str, int],
    bounded: set[tuple[str, str, str]],
    bar_cache: dict[str, dict[str, pl.DataFrame]],
    source_cache: dict[str, pl.DataFrame],
) -> dict[str, CoveragePlan]:
    plans: dict[str, CoveragePlan] = {}
    bar_plans = []
    for timeframe, by_symbol in bar_cache.items():
        spec = bar_spec(
            timeframe=timeframe,
            target_days=request.bars.target_days,
            max_staleness_hours=request.bars.latest_staleness_hours
            or request.bars.max_staleness_hours,
        )
        bar_plans.append(
            plan_product_coverage(
                spec=spec,
                symbols=request.bars.symbols,
                frame=_concat(by_symbol.values()),
                coin_listed_ms=listed,
                policy=policy.coverage,
                provider_bounded=bounded,
            )
        )
    plans["bars"] = _combine_coverage(bar_plans)
    for product in request.sources.products:
        spec = source_spec(
            product_name=product.name,
            target_days=request.sources.target_days,
            max_staleness_hours=product.max_staleness_hours or request.sources.max_staleness_hours,
            page_limit=product.limit,
        )
        plans[product.name] = plan_product_coverage(
            spec=spec,
            symbols=request.sources.symbols,
            frame=source_cache.get(product.name, pl.DataFrame()),
            coin_listed_ms=listed,
            policy=policy.coverage,
            provider_bounded=bounded,
        )
    return plans


async def _execute_bar_jobs(
    okx: OkxClient,
    plan: CoveragePlan | None,
    request: BarLoadRequest,
    policy: MarketLoadPolicy,
    bar_cache: dict[str, dict[str, pl.DataFrame]],
    bounded: set[tuple[str, str, str]],
) -> int:
    if plan is None:
        return 0
    pages_total = 0
    for job in plan.jobs:
        by_symbol = bar_cache.setdefault(job.timeframe, {})
        existing = by_symbol.get(job.symbol, pl.DataFrame())
        if job.kind == "latest_refresh":
            fetched, pages, is_bounded = await _refresh_latest_bars(
                okx, job.symbol, job.timeframe, existing, policy
            )
        else:
            fetched, pages, is_bounded = await _backfill_bars(
                okx, job.symbol, job.timeframe, existing, request, policy, job.max_pages
            )
        pages_total += pages
        if is_bounded:
            bounded.add((job.symbol, "bars", job.timeframe))
        cached = _cache_bar_frame(
            policy.cache_root, job.symbol, job.timeframe, fetched, request.target_days
        )
        by_symbol[job.symbol] = cached
    return pages_total


async def _execute_source_jobs(
    okx: OkxClient,
    plans: dict[str, CoveragePlan],
    request: SourceLoadRequest,
    policy: MarketLoadPolicy,
    source_cache: dict[str, pl.DataFrame],
    bounded: set[tuple[str, str, str]],
) -> tuple[dict[str, pl.DataFrame], dict[str, int]]:
    pages_by_product: dict[str, int] = {}
    product_by_name = {product.name: product for product in request.products}
    for name, plan in plans.items():
        if name == "bars" or name not in product_by_name:
            continue
        product = product_by_name[name]
        path = _source_cache_path(policy.cache_root, name)
        existing = source_cache.get(name, pl.DataFrame())
        pages = 0
        current_symbols = tuple(job.symbol for job in plan.jobs if job.kind == "current_snapshot")
        if current_symbols:
            fetched = await _fetch_current_sources(okx, product, current_symbols, policy)
            existing = merge_frames(existing, fetched, ("symbol", "timestamp"))
            pages += len(current_symbols)
        for job in (job for job in plan.jobs if job.kind != "current_snapshot"):
            local = _symbol_frame(existing, job.symbol)
            if job.kind == "latest_refresh":
                fetched, symbol_pages, is_bounded = await _refresh_latest_source(
                    okx, product, job.symbol, local, policy
                )
            else:
                fetched, symbol_pages, is_bounded = await _backfill_source(
                    okx, product, job.symbol, local, policy, job.max_pages
                )
            if is_bounded:
                bounded.add((job.symbol, product.name, job.timeframe))
            if not fetched.is_empty():
                existing = merge_frames(existing, fetched, ("symbol", "timestamp"))
            pages += symbol_pages
        if not existing.is_empty() or path.exists():
            save_frame(path, existing, {}, fmt="parquet")
        source_cache[name] = existing
        pages_by_product[name] = pages
    return source_cache, pages_by_product


def _bar_output_frames(
    bar_cache: dict[str, dict[str, pl.DataFrame]], request: BarLoadRequest
) -> dict[tuple[str, str], pl.DataFrame]:
    frames: dict[tuple[str, str], pl.DataFrame] = {}
    for timeframe, by_symbol in bar_cache.items():
        for symbol in request.symbols:
            frames[(symbol, timeframe)] = _bar_window(
                by_symbol.get(symbol, pl.DataFrame()), request.target_days
            )
    return frames


def _source_products(
    frames: dict[str, pl.DataFrame], request: SourceLoadRequest
) -> dict[str, ProductResult]:
    products = {}
    for product in request.products:
        frame = frames.get(product.name, pl.DataFrame())
        threshold_hours = product.max_staleness_hours or request.max_staleness_hours
        health = FrameHealth.from_frame(
            frame,
            product=product.name,
            key="",
            threshold_hours=threshold_hours,
        )
        products[product.name] = ProductResult(product.name, frame, health)
    return products


def _load_source_cache(root: Path, product: str) -> pl.DataFrame:
    path = _source_cache_path(root, product)
    return load_frame(path, {}, fmt="parquet") if path.exists() else pl.DataFrame()


def _source_cache_path(root: Path, product: str) -> Path:
    return root.parent / "sources" / f"{product}.parquet"


async def _backfill_bars(
    okx: OkxClient,
    symbol: str,
    timeframe: str,
    existing: pl.DataFrame,
    request: BarLoadRequest,
    policy: MarketLoadPolicy,
    max_pages: int,
) -> tuple[pl.DataFrame, int, bool]:
    merged = _drop_loader_columns(existing)
    pages = 0
    target_rows = request.target_days * 24
    target_start = now_ms() - request.target_days * 24 * HOUR_MS
    for _ in range(max_pages):
        if _depth_satisfied(merged, target_rows=target_rows, target_start_ms=target_start):
            break
        before = _earliest(merged)
        page = await _safe_fetch_bar_page(okx, symbol, timeframe, before, policy)
        if page.is_empty():
            return _bar_result(symbol, timeframe, merged, pages, True)
        page_earliest = _earliest(page)
        if before is not None and page_earliest is not None and page_earliest >= before:
            return _bar_result(symbol, timeframe, merged, pages, True)
        merged = _merge_symbol(merged, page)
        pages += 1
        _save_bar_frame(policy.cache_root, symbol, timeframe, merged)
    return _bar_result(symbol, timeframe, merged, pages, False)


async def _refresh_latest_bars(
    okx: OkxClient,
    symbol: str,
    timeframe: str,
    existing: pl.DataFrame,
    policy: MarketLoadPolicy,
) -> tuple[pl.DataFrame, int, bool]:
    merged = _drop_loader_columns(existing)
    page = await _safe_fetch_bar_page(okx, symbol, timeframe, None, policy)
    if page.is_empty():
        return _bar_result(symbol, timeframe, merged, 0, False)
    next_frame = _merge_symbol(merged, page)
    pages = 1 if next_frame.height > merged.height or _latest(next_frame) != _latest(merged) else 0
    if pages:
        _save_bar_frame(policy.cache_root, symbol, timeframe, next_frame)
    return _bar_result(symbol, timeframe, next_frame, pages, False)


async def _backfill_source(
    okx: OkxClient,
    product: SourceProductLoadRequest,
    symbol: str,
    existing: pl.DataFrame,
    policy: MarketLoadPolicy,
    max_pages: int,
) -> tuple[pl.DataFrame, int, bool]:
    merged = existing
    pages = 0
    for _ in range(max_pages):
        before = _earliest(merged)
        page = await _safe_fetch_source_page(okx, product, symbol, before, policy)
        if page.is_empty():
            return merged, pages, True
        page_earliest = _earliest(page)
        if before is not None and page_earliest is not None and page_earliest >= before:
            return merged, pages, True
        next_frame = _merge_source(merged, page)
        if next_frame.height <= merged.height:
            return merged, pages, True
        merged = next_frame
        pages += 1
    return merged, pages, False


async def _refresh_latest_source(
    okx: OkxClient,
    product: SourceProductLoadRequest,
    symbol: str,
    existing: pl.DataFrame,
    policy: MarketLoadPolicy,
) -> tuple[pl.DataFrame, int, bool]:
    page = await _safe_fetch_source_page(okx, product, symbol, None, policy)
    if page.is_empty():
        return existing, 0, False
    next_frame = _merge_source(existing, page)
    pages = (
        1 if next_frame.height > existing.height or _latest(next_frame) != _latest(existing) else 0
    )
    return next_frame, pages, False


async def _safe_fetch_bar_page(
    okx: OkxClient,
    symbol: str,
    timeframe: str,
    earliest_ms: int | None,
    policy: MarketLoadPolicy,
) -> pl.DataFrame:
    try:
        page = await okx.history_candles(
            symbol,
            bar=timeframe,
            limit=100,
            after=str(earliest_ms) if earliest_ms is not None else None,
        )
    except Exception as exc:
        if not policy.allow_partial:
            raise
        return _failed_frame(symbol, timeframe, sanitize_error(exc))
    return _source_identity(page, symbol).with_columns(pl.lit(timeframe).alias("timeframe"))


async def _safe_fetch_source_page(
    okx: OkxClient,
    product: SourceProductLoadRequest,
    symbol: str,
    earliest_ms: int | None,
    policy: MarketLoadPolicy,
) -> pl.DataFrame:
    try:
        if product.name == "funding":
            frame = (
                await okx.funding_history(
                    symbol,
                    after=str(earliest_ms) if earliest_ms is not None else None,
                    limit=product.limit,
                )
            ).frame
        elif product.name == "open_interest":
            frame = (
                await okx.open_interest(
                    symbol,
                    period=product.period,
                    limit=product.limit,
                    end=str(earliest_ms - 1) if earliest_ms is not None else None,
                )
            ).frame
        elif product.name == "taker_volume":
            frame = (
                await okx.taker_volume(
                    symbol,
                    period=product.period,
                    unit=product.unit,
                    limit=product.limit,
                    end=str(earliest_ms - 1) if earliest_ms is not None else None,
                )
            ).frame
        elif product.name == "long_short_ratios":
            frame = (
                await okx.long_short_ratio(
                    symbol,
                    period=product.period,
                    limit=product.limit,
                    end=str(earliest_ms - 1) if earliest_ms is not None else None,
                )
            ).frame
        else:
            frame = pl.DataFrame()
    except Exception as exc:
        if not policy.allow_partial:
            raise
        return _failed_frame(symbol, "", sanitize_error(exc))
    return _source_identity(frame, symbol)


async def _fetch_current_sources(
    okx: OkxClient,
    product: SourceProductLoadRequest,
    symbols: tuple[str, ...],
    policy: MarketLoadPolicy,
) -> pl.DataFrame:
    semaphore = asyncio.Semaphore(policy.concurrency)

    async def one(symbol: str) -> pl.DataFrame:
        async with semaphore:
            try:
                if product.name == "books":
                    frame = (await okx.book_snapshot(symbol, limit=product.limit)).frame
                elif product.name == "trades":
                    frame = (await okx.recent_trades(symbol, limit=product.limit)).frame
                else:
                    frame = pl.DataFrame()
            except Exception as exc:
                if not policy.allow_partial:
                    raise
                return _failed_frame(symbol, "", sanitize_error(exc))
        return _source_identity(frame, symbol)

    return _concat(await asyncio.gather(*(one(symbol) for symbol in symbols)))


def _bar_cache_frames(
    root: Path, symbols: tuple[str, ...], timeframe: str
) -> dict[str, pl.DataFrame]:
    frames = {}
    for symbol in symbols:
        path = root / symbol / f"bars_{timeframe}.parquet"
        frame = load_frame(path, {}, fmt="parquet") if path.exists() else pl.DataFrame()
        if not frame.is_empty():
            frame = frame.with_columns(
                pl.lit(symbol).alias("symbol"), pl.lit(timeframe).alias("timeframe")
            )
        frames[symbol] = frame
    return frames


def _cache_bar_frame(
    root: Path, symbol: str, timeframe: str, frame: pl.DataFrame, days: int
) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    bare = _drop_loader_columns(frame)
    _save_bar_frame(root, symbol, timeframe, bare)
    return _bar_window(
        bare.with_columns(pl.lit(symbol).alias("symbol"), pl.lit(timeframe).alias("timeframe")),
        days,
    )


def _bar_window(frame: pl.DataFrame, days: int) -> pl.DataFrame:
    if frame.is_empty() or "timestamp" not in frame.columns:
        return frame
    return frame.sort("timestamp").tail(days * 24)


def _save_bar_frame(root: Path, symbol: str, timeframe: str, frame: pl.DataFrame) -> None:
    if frame.is_empty():
        return
    save_frame(
        root / symbol / f"bars_{timeframe}.parquet",
        frame,
        {},
        fmt="parquet",
    )


def _bars_product(
    frames: dict[tuple[str, str], pl.DataFrame], request: BarLoadRequest
) -> ProductResult:
    frame = _concat(frames.values())
    if frame.is_empty():
        return ProductResult("bars", frame, FrameHealth(product="bars", key="", status="missing"))
    grouped = frame.group_by("symbol", "timeframe").agg(
        pl.len().alias("rows"),
        pl.col("timestamp").n_unique().alias("unique_rows"),
        pl.col("timestamp").max().alias("latest_ts"),
    )
    target_rows = request.target_days * 24 * len(request.symbols) * len(request.timeframes)
    latest_ts = int(grouped.get_column("latest_ts").max())
    age_hours = max(0.0, (now_ms() - latest_ts) / HOUR_MS)
    return ProductResult(
        "bars",
        frame,
        FrameHealth(
            product="bars",
            key="",
            actual_rows=frame.height,
            target_rows=target_rows,
            coverage_pct=min(100.0, frame.height / target_rows * 100.0) if target_rows else 0.0,
            latest_ts=latest_ts,
            age_hours=age_hours,
            status="fresh" if age_hours <= request.max_staleness_hours else "stale",
            duplicates=int((grouped.get_column("rows") - grouped.get_column("unique_rows")).sum()),
        ),
    )


def _source_identity(frame: pl.DataFrame, symbol: str) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    if "instId" in frame.columns and "symbol" not in frame.columns:
        frame = frame.rename({"instId": "symbol"})
    if "inst_id" in frame.columns and "symbol" not in frame.columns:
        frame = frame.rename({"inst_id": "symbol"})
    if "ts" in frame.columns and "timestamp" not in frame.columns:
        frame = frame.rename({"ts": "timestamp"})
    if "funding_time" in frame.columns and "timestamp" not in frame.columns:
        frame = frame.with_columns(pl.col("funding_time").alias("timestamp"))
    if "symbol" not in frame.columns:
        frame = frame.with_columns(pl.lit(symbol).alias("symbol"))
    if "timestamp" not in frame.columns:
        frame = frame.with_columns(pl.lit(now_ms()).alias("timestamp"))
    return frame.with_columns(pl.col("timestamp").cast(pl.Int64, strict=False))


def _symbol_frame(frame: pl.DataFrame, symbol: str) -> pl.DataFrame:
    if frame.is_empty() or "symbol" not in frame.columns:
        return pl.DataFrame()
    return frame.filter(pl.col("symbol") == symbol)


def _depth_satisfied(frame: pl.DataFrame, *, target_rows: int, target_start_ms: int) -> bool:
    if frame.is_empty() or "timestamp" not in frame.columns:
        return False
    earliest = _earliest(frame)
    return frame.height >= target_rows and earliest is not None and earliest <= target_start_ms


def _earliest(frame: pl.DataFrame) -> int | None:
    if frame.is_empty() or "timestamp" not in frame.columns:
        return None
    value = frame.get_column("timestamp").min()
    return int(value) if value is not None else None


def _latest(frame: pl.DataFrame) -> int | None:
    if frame.is_empty() or "timestamp" not in frame.columns:
        return None
    value = frame.get_column("timestamp").max()
    return int(value) if value is not None else None


def _merge_symbol(existing: pl.DataFrame, page: pl.DataFrame) -> pl.DataFrame:
    if page.is_empty():
        return existing
    existing = _drop_loader_columns(existing)
    page = _drop_loader_columns(page)
    merged = merge_frames(existing, page, ("timestamp",))
    return merged.sort("timestamp") if "timestamp" in merged.columns else merged


def _merge_source(existing: pl.DataFrame, page: pl.DataFrame) -> pl.DataFrame:
    if page.is_empty():
        return existing
    if existing.is_empty():
        return page.sort("timestamp") if "timestamp" in page.columns else page
    merged = merge_frames(existing, page, ("symbol", "timestamp"))
    return merged.sort(["symbol", "timestamp"]) if "timestamp" in merged.columns else merged


def _bar_result(
    symbol: str, timeframe: str, frame: pl.DataFrame, pages: int, bounded: bool
) -> tuple[pl.DataFrame, int, bool]:
    return (
        frame.with_columns(pl.lit(symbol).alias("symbol"), pl.lit(timeframe).alias("timeframe")),
        pages,
        bounded,
    )


def _combine_coverage(plans: list[CoveragePlan]) -> CoveragePlan:
    states = tuple(state for plan in plans for state in plan.states)
    jobs = tuple(job for plan in plans for job in plan.jobs)
    return CoveragePlan(
        states=states,
        jobs=jobs,
        estimated_pages=sum(plan.estimated_pages for plan in plans),
    )


def _request_bound_counts(
    request: MarketLoadRequest, bounded: set[tuple[str, str, str]]
) -> dict[str, int]:
    allowed: dict[str, set[tuple[str, str]]] = {
        "bars": {
            (symbol, timeframe)
            for symbol in request.bars.symbols
            for timeframe in request.bars.timeframes
        }
    }
    for product in request.sources.products:
        allowed[product.name] = {
            (
                symbol,
                "1H"
                if product.name in {"open_interest", "taker_volume", "long_short_ratios"}
                else "",
            )
            for symbol in request.sources.symbols
        }
    counts: dict[str, set[str]] = {}
    for symbol, product, timeframe in bounded:
        if (symbol, timeframe) in allowed.get(product, set()):
            counts.setdefault(product, set()).add(symbol)
    return {product: len(symbols) for product, symbols in counts.items()}


def _provider_bounds_path(root: Path) -> Path:
    return root / "_coverage" / "provider_bounds.parquet"


def _load_provider_bounds(root: Path) -> set[tuple[str, str, str]]:
    path = _provider_bounds_path(root)
    if not path.exists():
        return set()
    frame = load_frame(path, {}, fmt="parquet")
    if frame.is_empty():
        return set()
    return {
        (str(row["symbol"]), str(row["product"]), str(row.get("timeframe") or ""))
        for row in frame.select("symbol", "product", "timeframe").drop_nulls("symbol").to_dicts()
    }


def _save_provider_bounds(root: Path, bounded: set[tuple[str, str, str]]) -> None:
    if not bounded:
        return
    path = _provider_bounds_path(root)
    rows = [
        {
            "symbol": symbol,
            "product": product,
            "timeframe": timeframe,
            "observed_at_ms": now_ms(),
            "reason": "older_page_empty_or_repeated",
        }
        for symbol, product, timeframe in sorted(bounded)
    ]
    save_frame(path, pl.DataFrame(rows), {}, fmt="parquet")


def _failed_frame(symbol: str, timeframe: str, warning: str) -> pl.DataFrame:
    return pl.DataFrame(
        {"symbol": [symbol], "timeframe": [timeframe], "timestamp": [0], "fetch_warning": [warning]}
    ).filter(pl.col("timestamp") > 0)


def _drop_loader_columns(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.drop([c for c in ("symbol", "timeframe") if c in frame.columns])


def _concat(frames) -> pl.DataFrame:
    live = [frame for frame in frames if not frame.is_empty()]
    return pl.concat(live, how="diagonal_relaxed") if live else pl.DataFrame()
