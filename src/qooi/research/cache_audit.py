"""Cache audit request planning for research commands."""

from __future__ import annotations

import asyncio
from typing import Any

from qooi.core.config import PairConfig
from qooi.exchange.store import (
    AsyncCacheStore,
    CacheStore,
    HistoryRefreshRequest,
    HistoryRequest,
)
from qooi.research.config import ResolvedBacktestConfig
from qooi.strategies.preprocessing import _context_min_bars, source_inst_ids

CONTEXT_BARS: tuple[str, ...] = ("4H", "1D")


def history_requests_for_pairs(
    pairs: tuple[PairConfig, ...],
    args: Any,
    config: ResolvedBacktestConfig,
    *,
    refresh: bool = False,
) -> tuple[HistoryRefreshRequest, ...]:
    requests: list[HistoryRefreshRequest] = []
    incremental = not bool(getattr(args, "refresh_full", False))
    for pair in pairs:
        signal_inst_id, execution_inst_id = source_inst_ids(pair, config.data_source)
        for inst_id in dict.fromkeys((signal_inst_id, execution_inst_id)):
            requests.append(
                HistoryRefreshRequest(
                    inst_id=inst_id,
                    bar=pair.asset.timeframe,
                    days=config.days,
                    min_bars=config.min_bars,
                    refresh=refresh,
                    source="trade",
                    incremental=incremental,
                )
            )
        for bar in CONTEXT_BARS:
            requests.append(
                HistoryRefreshRequest(
                    inst_id=signal_inst_id,
                    bar=bar,
                    days=config.days,
                    min_bars=_context_min_bars(bar, config.days, role="higher_context"),
                    refresh=refresh,
                    source="trade",
                    incremental=incremental,
                )
            )
    return tuple(dict.fromkeys(requests))


def cache_audit_rows(
    pairs: tuple[PairConfig, ...],
    args: Any,
    config: ResolvedBacktestConfig,
) -> list[list[str]]:
    store = CacheStore()
    if bool(getattr(args, "refresh_cache", False)) and bool(getattr(args, "async_refresh", False)):
        requests = history_requests_for_pairs(pairs, args, config, refresh=True)
        concurrency = int(getattr(args, "refresh_concurrency", 3) or 3)
        asyncio_store = AsyncCacheStore()
        asyncio.run(asyncio_store.many(requests, concurrency=concurrency))
    rows: list[list[str]] = []
    refresh_local = bool(getattr(args, "refresh_cache", False)) and not bool(
        getattr(args, "async_refresh", False)
    )
    for request in history_requests_for_pairs(pairs, args, config, refresh=refresh_local):
        try:
            _df, coverage = store.bars(
                HistoryRequest(
                    inst_id=request.inst_id,
                    bar=request.bar,
                    days=request.days,
                    min_bars=request.min_bars,
                    refresh=refresh_local,
                    source=request.source,
                )
            )
            status = (
                "PASS" if coverage.coverage_pct >= config.risk_gates.min_coverage_pct else "LOW"
            )
            rows.append(
                [
                    status,
                    request.inst_id,
                    request.bar,
                    str(coverage.actual_bars),
                    str(coverage.target.target_bars),
                    f"{coverage.coverage_pct:.1f}",
                    str(coverage.actual_start_ms or "n/a"),
                    str(coverage.actual_end_ms or "n/a"),
                    ",".join(coverage.notes),
                ]
            )
        except Exception as exc:
            rows.append(
                [
                    "ERROR",
                    request.inst_id,
                    request.bar,
                    "0",
                    "0",
                    "0.0",
                    "n/a",
                    "n/a",
                    str(exc),
                ]
            )
    return rows
