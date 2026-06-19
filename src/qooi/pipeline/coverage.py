"""Coverage planning: cache depth + coin life + endpoint capability -> jobs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Literal

import polars as pl

from qooi.pipeline import HOUR_MS, now_ms

EndpointKind = Literal["historical", "current_only"]
CursorKind = Literal["bars_after", "funding_after", "rubik_end", "none"]
CoverageStatus = Literal[
    "complete",
    "fetch_more",
    "allocated",
    "current_refresh",
    "deferred_by_budget",
    "provider_bounded",
    "coin_too_new",
    "current_only",
    "missing",
    "stale",
]
NeedKind = Literal["latest_refresh", "older_backfill", "current_snapshot"]


@dataclass(frozen=True)
class Product:
    name: str
    merge_keys: tuple[str, ...]
    endpoint_kind: EndpointKind
    cursor_kind: CursorKind
    interval_ms: int | None
    page_limit: int = 100
    cache_format: str = "parquet"

    def cache_path(self, root: Path, *, symbol: str = "", timeframe: str = "") -> Path:
        if self.name == "bars":
            return root / symbol / f"bars_{timeframe}.parquet"
        return root.parent / "sources" / f"{self.name}.parquet"

    def target_rows(self, target_days: int) -> int:
        if self.interval_ms is None:
            return 1
        return max(1, target_days * 24 * HOUR_MS // self.interval_ms)

    def timeframe(self, default: str = "") -> str:
        if self.name == "bars":
            return default
        return "1H" if self.cursor_kind == "rubik_end" else ""


PRODUCTS: dict[str, Product] = {
    "bars": Product("bars", ("timestamp",), "historical", "bars_after", HOUR_MS),
    "books": Product("books", ("symbol", "timestamp"), "current_only", "none", None),
    "trades": Product("trades", ("symbol", "timestamp"), "current_only", "none", None),
    "funding": Product(
        "funding", ("symbol", "timestamp"), "historical", "funding_after", 8 * HOUR_MS
    ),
    "open_interest": Product(
        "open_interest", ("symbol", "timestamp"), "historical", "rubik_end", HOUR_MS
    ),
    "taker_volume": Product(
        "taker_volume", ("symbol", "timestamp"), "historical", "rubik_end", HOUR_MS
    ),
    "long_short_ratios": Product(
        "long_short_ratios", ("symbol", "timestamp"), "historical", "rubik_end", HOUR_MS
    ),
}


def product(name: str) -> Product:
    return PRODUCTS[name]


def product_frame(frame: pl.DataFrame, symbol: str) -> pl.DataFrame:
    if frame.is_empty() or "symbol" not in frame.columns:
        return pl.DataFrame()
    return frame.filter(pl.col("symbol") == symbol)


@dataclass(frozen=True)
class CoverageRunPolicy:
    max_requests: int = 1000
    max_seconds: int = 900
    max_requests_per_symbol_product: int = 24
    concurrency: int = 1
    allow_partial: bool = True


@dataclass(frozen=True)
class ProductCoverageSpec:
    product: str
    timeframe: str
    target_days: int
    target_rows: int
    interval_ms: int | None
    endpoint_kind: EndpointKind
    cursor_kind: CursorKind
    page_limit: int
    max_staleness_hours: int


@dataclass(frozen=True)
class CoverageState:
    symbol: str
    product: str
    timeframe: str
    status: CoverageStatus
    reason: str
    target_rows: int
    target_start_ms: int | None
    cache_rows: int
    cache_earliest_ms: int | None
    cache_latest_ms: int | None
    cache_span_days: float
    coin_listed_ms: int | None
    max_possible_rows: int
    missing_rows: int
    missing_pages: int
    fresh: bool
    deep_enough: bool
    allocated_pages: int = 0

    def allocate(self, pages: int, status: CoverageStatus) -> CoverageState:
        return CoverageState(
            self.symbol,
            self.product,
            self.timeframe,
            status,
            "allocated_this_run" if pages else "deferred_by_budget",
            self.target_rows,
            self.target_start_ms,
            self.cache_rows,
            self.cache_earliest_ms,
            self.cache_latest_ms,
            self.cache_span_days,
            self.coin_listed_ms,
            self.max_possible_rows,
            self.missing_rows,
            self.missing_pages,
            self.fresh,
            self.deep_enough,
            pages,
        )


@dataclass(frozen=True)
class CoverageJob:
    symbol: str
    product: str
    timeframe: str
    kind: NeedKind
    max_pages: int
    cursor_kind: CursorKind
    first_cursor: str | None
    limit: int
    priority: int
    reason: str


@dataclass(frozen=True)
class CoveragePlan:
    states: tuple[CoverageState, ...]
    jobs: tuple[CoverageJob, ...]
    estimated_pages: int

    def with_allocation(self, jobs: tuple[CoverageJob, ...]) -> CoveragePlan:
        pages_by_key = {(job.symbol, job.product, job.timeframe): job.max_pages for job in jobs}
        return CoveragePlan(
            states=tuple(_allocated_state(state, pages_by_key) for state in self.states),
            jobs=jobs,
            estimated_pages=self.estimated_pages,
        )

    def allocated_pages(self) -> int:
        return sum(job.max_pages for job in self.jobs)


def bar_spec(*, timeframe: str, target_days: int, max_staleness_hours: int) -> ProductCoverageSpec:
    return coverage_spec(
        product("bars"),
        timeframe=timeframe,
        target_days=target_days,
        max_staleness_hours=max_staleness_hours,
    )


def source_spec(
    *, product_name: str, target_days: int, max_staleness_hours: int, page_limit: int
) -> ProductCoverageSpec:
    meta = product(product_name)
    return ProductCoverageSpec(
        product=meta.name,
        timeframe=meta.timeframe(),
        target_days=target_days if meta.endpoint_kind == "historical" else 0,
        target_rows=meta.target_rows(target_days),
        interval_ms=meta.interval_ms,
        endpoint_kind=meta.endpoint_kind,
        cursor_kind=meta.cursor_kind,
        page_limit=page_limit,
        max_staleness_hours=max_staleness_hours,
    )


def coverage_spec(
    meta: Product, *, timeframe: str, target_days: int, max_staleness_hours: int
) -> ProductCoverageSpec:
    return ProductCoverageSpec(
        product=meta.name,
        timeframe=meta.timeframe(timeframe),
        target_days=target_days,
        target_rows=meta.target_rows(target_days),
        interval_ms=meta.interval_ms,
        endpoint_kind=meta.endpoint_kind,
        cursor_kind=meta.cursor_kind,
        page_limit=meta.page_limit,
        max_staleness_hours=max_staleness_hours,
    )


def plan_product_coverage(
    *,
    spec: ProductCoverageSpec,
    symbols: tuple[str, ...],
    frame: pl.DataFrame,
    coin_listed_ms: dict[str, int],
    policy: CoverageRunPolicy = CoverageRunPolicy(),
    provider_bounded: set[tuple[str, str, str]] | None = None,
) -> CoveragePlan:
    bounded = provider_bounded or set()
    states = tuple(
        coverage_state(
            spec=spec,
            symbol=symbol,
            frame=product_frame(frame, symbol),
            coin_listed_ms=coin_listed_ms.get(symbol),
            provider_bounded=(symbol, spec.product, spec.timeframe) in bounded,
        )
        for symbol in symbols
    )
    jobs = _candidate_jobs_for_states(spec, states, policy)
    return CoveragePlan(
        states=states,
        jobs=jobs,
        estimated_pages=sum(state.missing_pages for state in states),
    )


def allocate_coverage(
    plans: dict[str, CoveragePlan], policy: CoverageRunPolicy
) -> dict[str, CoveragePlan]:
    historical = sorted(
        (job for plan in plans.values() for job in plan.jobs if job.kind != "current_snapshot"),
        key=lambda job: (job.priority, job.product, job.symbol),
    )
    current = sorted(
        (job for plan in plans.values() for job in plan.jobs if job.kind == "current_snapshot"),
        key=lambda job: (job.priority, job.product, job.symbol),
    )
    allocated = _allocate_jobs(historical, policy.max_requests)
    current_budget = (
        min(len(current), max(1, policy.max_requests // 10))
        if current and policy.max_requests > 0
        else 0
    )
    allocated += _allocate_jobs(current, current_budget)
    by_product: dict[str, list[CoverageJob]] = {name: [] for name in plans}
    for job in allocated:
        by_product.setdefault(job.product, []).append(job)
    return {
        name: plan.with_allocation(tuple(by_product.get(name, ()))) for name, plan in plans.items()
    }


def coverage_state(
    *,
    spec: ProductCoverageSpec,
    symbol: str,
    frame: pl.DataFrame,
    coin_listed_ms: int | None,
    provider_bounded: bool = False,
) -> CoverageState:
    rows = frame.height
    earliest = _timestamp_min(frame)
    latest = _timestamp_max(frame)
    target_start = now_ms() - spec.target_days * 24 * HOUR_MS if spec.interval_ms else None
    effective_start = (
        max(v for v in (target_start, coin_listed_ms) if v is not None)
        if (target_start or coin_listed_ms)
        else target_start
    )
    fresh = _fresh(latest, spec.max_staleness_hours)
    latest_possible_end = (
        latest
        if fresh
        and coin_listed_ms is not None
        and target_start is not None
        and coin_listed_ms > target_start
        else None
    )
    max_possible_rows = _max_possible_rows(spec, effective_start, latest_ms=latest_possible_end)
    required_rows = (
        min(spec.target_rows, max_possible_rows) if spec.endpoint_kind == "historical" else 1
    )
    span_days = _span_days(earliest, latest)
    deep_enough = rows >= required_rows and (
        effective_start is None or (earliest is not None and earliest <= effective_start)
    )
    missing_rows = max(0, required_rows - rows)
    missing_pages = ceil(missing_rows / spec.page_limit) if spec.page_limit > 0 else 0

    if spec.endpoint_kind == "current_only":
        return _state(
            spec,
            symbol,
            "current_only",
            "current_only",
            rows,
            target_start,
            earliest,
            latest,
            span_days,
            coin_listed_ms,
            max_possible_rows,
            missing_rows,
            1 if rows == 0 or not fresh else 0,
            fresh,
            rows > 0,
        )
    if provider_bounded:
        return _state(
            spec,
            symbol,
            "provider_bounded",
            "older_page_empty_or_repeated",
            rows,
            effective_start,
            earliest,
            latest,
            span_days,
            coin_listed_ms,
            max_possible_rows,
            0,
            0,
            fresh,
            deep_enough,
        )
    if (
        coin_listed_ms is not None
        and max_possible_rows < spec.target_rows
        and _covers_coin_life(
            rows=rows,
            earliest=earliest,
            effective_start=effective_start,
            interval_ms=spec.interval_ms,
            max_possible_rows=max_possible_rows,
            fresh=fresh,
        )
    ):
        return _state(
            spec,
            symbol,
            "coin_too_new",
            "coin_life_shorter_than_target",
            rows,
            effective_start,
            earliest,
            latest,
            span_days,
            coin_listed_ms,
            max_possible_rows,
            0,
            0,
            fresh,
            deep_enough,
        )
    if rows == 0:
        return _state(
            spec,
            symbol,
            "missing",
            "missing_cache",
            rows,
            effective_start,
            earliest,
            latest,
            span_days,
            coin_listed_ms,
            max_possible_rows,
            missing_rows or required_rows,
            max(1, missing_pages),
            fresh,
            False,
        )
    if not fresh:
        return _state(
            spec,
            symbol,
            "stale",
            "latest_older_than_staleness",
            rows,
            effective_start,
            earliest,
            latest,
            span_days,
            coin_listed_ms,
            max_possible_rows,
            missing_rows,
            max(1, missing_pages),
            fresh,
            deep_enough,
        )
    if deep_enough:
        return _state(
            spec,
            symbol,
            "complete",
            "target_depth_satisfied",
            rows,
            effective_start,
            earliest,
            latest,
            span_days,
            coin_listed_ms,
            max_possible_rows,
            0,
            0,
            fresh,
            True,
        )
    return _state(
        spec,
        symbol,
        "fetch_more",
        "fresh_but_shallow",
        rows,
        effective_start,
        earliest,
        latest,
        span_days,
        coin_listed_ms,
        max_possible_rows,
        missing_rows,
        max(1, missing_pages),
        fresh,
        False,
    )


def coverage_summary(plan: CoveragePlan) -> dict[str, int]:
    return dict(Counter(state.status for state in plan.states))


def median_span_days(plan: CoveragePlan) -> float:
    spans = sorted(state.cache_span_days for state in plan.states if state.cache_rows > 0)
    return spans[len(spans) // 2] if spans else 0.0


def coin_list_times(frame: pl.DataFrame) -> dict[str, int]:
    if frame.is_empty() or "symbol" not in frame.columns or "list_time" not in frame.columns:
        return {}
    return {
        str(row["symbol"]): int(row["list_time"])
        for row in frame.select("symbol", "list_time").drop_nulls().to_dicts()
    }


def _candidate_jobs_for_states(
    spec: ProductCoverageSpec,
    states: tuple[CoverageState, ...],
    policy: CoverageRunPolicy,
) -> tuple[CoverageJob, ...]:
    jobs: list[CoverageJob] = []
    candidates = sorted(
        (
            state
            for state in states
            if state.status in {"fetch_more", "missing", "stale"}
            or (state.status == "current_only" and state.missing_pages > 0)
        ),
        key=lambda s: (_priority(s), s.cache_span_days, s.symbol),
    )
    for state in candidates:
        pages = min(max(1, state.missing_pages), policy.max_requests_per_symbol_product)
        if pages <= 0:
            continue
        kind: NeedKind = "current_snapshot" if state.status == "current_only" else "older_backfill"
        if state.status == "stale":
            kind = "latest_refresh"
        jobs.append(
            CoverageJob(
                symbol=state.symbol,
                product=state.product,
                timeframe=state.timeframe,
                kind=kind,
                max_pages=pages,
                cursor_kind=spec.cursor_kind,
                first_cursor=_cursor(spec, state.cache_earliest_ms),
                limit=spec.page_limit,
                priority=_priority(state),
                reason=state.reason,
            )
        )
    return tuple(jobs)


def _allocate_jobs(jobs: list[CoverageJob], budget: int) -> list[CoverageJob]:
    remaining = max(0, budget)
    allocated: list[CoverageJob] = []
    for job in jobs:
        if remaining <= 0:
            break
        pages = min(job.max_pages, remaining)
        if pages <= 0:
            continue
        allocated.append(
            CoverageJob(
                job.symbol,
                job.product,
                job.timeframe,
                job.kind,
                pages,
                job.cursor_kind,
                job.first_cursor,
                job.limit,
                job.priority,
                job.reason,
            )
        )
        remaining -= pages
    return allocated


def _allocated_state(
    state: CoverageState, pages_by_key: dict[tuple[str, str, str], int]
) -> CoverageState:
    pages = pages_by_key.get((state.symbol, state.product, state.timeframe), 0)
    if pages > 0:
        status: CoverageStatus = (
            "current_refresh" if state.status == "current_only" else "allocated"
        )
        return state.allocate(pages, status)
    if state.status in {"fetch_more", "missing", "stale"} or (
        state.status == "current_only" and state.missing_pages > 0
    ):
        return state.allocate(0, "deferred_by_budget")
    return state


def _priority(state: CoverageState) -> int:
    if state.status == "stale":
        return 0
    if state.product == "bars":
        return 1
    if state.product in {"open_interest", "taker_volume", "long_short_ratios"}:
        return 2
    return 3


def _cursor(spec: ProductCoverageSpec, earliest_ms: int | None) -> str | None:
    if earliest_ms is None:
        return None
    if spec.cursor_kind in {"bars_after", "funding_after"}:
        return str(earliest_ms)
    if spec.cursor_kind == "rubik_end":
        return str(earliest_ms - 1)
    return None


def _state(
    spec: ProductCoverageSpec,
    symbol: str,
    status: CoverageStatus,
    reason: str,
    rows: int,
    target_start: int | None,
    earliest: int | None,
    latest: int | None,
    span_days: float,
    coin_listed_ms: int | None,
    max_possible_rows: int,
    missing_rows: int,
    missing_pages: int,
    fresh: bool,
    deep_enough: bool,
) -> CoverageState:
    return CoverageState(
        symbol,
        spec.product,
        spec.timeframe,
        status,
        reason,
        spec.target_rows,
        target_start,
        rows,
        earliest,
        latest,
        span_days,
        coin_listed_ms,
        max_possible_rows,
        missing_rows,
        missing_pages,
        fresh,
        deep_enough,
    )


def _timestamp_min(frame: pl.DataFrame) -> int | None:
    if frame.is_empty() or "timestamp" not in frame.columns:
        return None
    value = frame.get_column("timestamp").min()
    return int(value) if value is not None else None


def _timestamp_max(frame: pl.DataFrame) -> int | None:
    if frame.is_empty() or "timestamp" not in frame.columns:
        return None
    value = frame.get_column("timestamp").max()
    return int(value) if value is not None else None


def _span_days(earliest: int | None, latest: int | None) -> float:
    if earliest is None or latest is None or latest < earliest:
        return 0.0
    return (latest - earliest) / (24 * HOUR_MS)


def _fresh(latest: int | None, max_staleness_hours: int) -> bool:
    return latest is not None and (now_ms() - latest) <= max_staleness_hours * HOUR_MS


def _covers_coin_life(
    *,
    rows: int,
    earliest: int | None,
    effective_start: int | None,
    interval_ms: int | None,
    max_possible_rows: int,
    fresh: bool,
) -> bool:
    if rows >= max_possible_rows:
        return True
    if not fresh or earliest is None or effective_start is None or interval_ms is None:
        return False
    return rows >= max(0, max_possible_rows - 1) and earliest <= effective_start + interval_ms


def _max_possible_rows(
    spec: ProductCoverageSpec, effective_start_ms: int | None, *, latest_ms: int | None = None
) -> int:
    if spec.interval_ms is None:
        return spec.target_rows
    start = (
        effective_start_ms
        if effective_start_ms is not None
        else now_ms() - spec.target_days * 24 * HOUR_MS
    )
    completed_end = (now_ms() // spec.interval_ms) * spec.interval_ms
    possible_end = min(completed_end, latest_ms) if latest_ms is not None else completed_end
    aligned_start = ((start + spec.interval_ms - 1) // spec.interval_ms) * spec.interval_ms
    if possible_end < aligned_start:
        return 0
    return max(0, (possible_end - aligned_start) // spec.interval_ms + 1)
