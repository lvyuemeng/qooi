"""Research command to cache history request planning."""

from __future__ import annotations

from dataclasses import dataclass

from qooi.core.instruments import PairConfig
from qooi.exchange.store import HistoryRefreshRequest
from qooi.research.context_frames import DEFAULT_CONTEXTS, _context_min_bars, source_inst_ids


@dataclass(frozen=True)
class CacheAuditRequest:
    pairs: tuple[PairConfig, ...]
    data_source: str
    days: int
    min_bars: int
    min_coverage_pct: float
    bars: tuple[str, ...] = ()
    refresh: bool = False
    async_refresh: bool = False
    refresh_concurrency: int = 3
    incremental: bool = True


def build_history_refresh_requests(
    request: CacheAuditRequest,
) -> tuple[HistoryRefreshRequest, ...]:
    requests: list[HistoryRefreshRequest] = []
    contexts = tuple(context for context in DEFAULT_CONTEXTS if context.role == "higher_context")
    for pair in request.pairs:
        signal_inst_id, execution_inst_id = source_inst_ids(pair, request.data_source)
        for bar in request.bars or (pair.asset.timeframe,):
            for inst_id in dict.fromkeys((signal_inst_id, execution_inst_id)):
                requests.append(
                    HistoryRefreshRequest(
                        inst_id=inst_id,
                        bar=bar,
                        days=request.days,
                        min_bars=request.min_bars,
                        refresh=request.refresh,
                        source="trade",
                        incremental=request.incremental,
                    )
                )
        for context in contexts:
            requests.append(
                HistoryRefreshRequest(
                    inst_id=signal_inst_id,
                    bar=context.bar,
                    days=request.days,
                    min_bars=_context_min_bars(
                        context.bar,
                        request.days,
                        role="higher_context",
                    ),
                    refresh=request.refresh,
                    source=context.source,
                    incremental=request.incremental,
                )
            )
    return tuple(dict.fromkeys(requests))
