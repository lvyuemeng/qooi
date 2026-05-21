"""Research-owned frame preparation workflows."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
from typing import Literal

import polars as pl

from qooi.core.instruments import PairConfig
from qooi.exchange.market import CandleSource
from qooi.exchange.store import (
    AsyncCacheStore,
    CacheStore,
    HistoryCoverage,
    HistoryRefreshRequest,
    HistoryRequest,
    validate_history,
)
from qooi.research.config import SignalDebugFilterConfig
from qooi.strategies.features import (
    StructureClassifierConfig,
    add_price_structure_stage_features,
)
from qooi.strategies.indicators import add_indicators, add_macd_histogram
from qooi.strategies.specs import StrategyBehavior, compute_signal_frame

logger = logging.getLogger(__name__)


class DataCoverageError(RuntimeError):
    def __init__(self, coverage: HistoryCoverage, required_pct: float) -> None:
        self.coverage = coverage
        self.required_pct = required_pct
        super().__init__(
            f"cache coverage below threshold for {coverage.inst_id}: "
            f"{coverage.coverage_pct:.1f}% < {required_pct:.1f}%"
        )


@dataclass(frozen=True)
class FrameRequest:
    pair: PairConfig
    data_source: str
    bar: str
    days: int
    min_bars: int
    refresh: bool = False
    min_coverage_pct: float = 0.0
    allow_swap_signal_fallback: bool = False


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


@dataclass(frozen=True)
class BacktestFrameOptions:
    signal_filters: SignalDebugFilterConfig
    metadata: tuple[str, ...]


@dataclass(frozen=True)
class FrameResult:
    pair: PairConfig
    frame: pl.DataFrame
    signal_inst_id: str
    execution_inst_id: str | None
    coverage: tuple[HistoryCoverage, ...]
    metadata: tuple[str, ...]


@dataclass(frozen=True)
class PreparedBacktestFrame(FrameResult):
    precomputed_signal: bool
    signal_coverage: HistoryCoverage
    execution_coverage: HistoryCoverage


PreparedClassifierFrame = FrameResult


@dataclass(frozen=True)
class CacheAuditResult:
    frame: pl.DataFrame


CACHE_AUDIT_SCHEMA = {
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


@dataclass(frozen=True)
class ContextSpec:
    bar: str
    prefix: str
    source: CandleSource = "trade"
    role: Literal["base", "higher_context"] = "higher_context"
    required: bool = True


DEFAULT_CONTEXTS: tuple[ContextSpec, ...] = (
    ContextSpec("1H", "h1", role="base"),
    ContextSpec("4H", "h4", role="higher_context"),
    ContextSpec("1D", "d1", role="higher_context"),
)


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


def run_cache_audit_workflow(request: CacheAuditRequest) -> CacheAuditResult:
    store = CacheStore()
    if request.refresh and request.async_refresh:
        requests = build_history_refresh_requests(request)
        asyncio.run(_stream_cache_refresh(requests, request.refresh_concurrency))
    refresh_local = request.refresh and not request.async_refresh
    local_request = replace(request, refresh=refresh_local)
    rows: list[dict[str, object]] = []
    for history_request in build_history_refresh_requests(local_request):
        try:
            _df, coverage = store.bars(
                HistoryRequest(
                    inst_id=history_request.inst_id,
                    bar=history_request.bar,
                    days=history_request.days,
                    min_bars=history_request.min_bars,
                    refresh=refresh_local,
                    source=history_request.source,
                )
            )
            rows.append(
                {
                    "status": "PASS"
                    if coverage.coverage_pct >= request.min_coverage_pct
                    else "LOW",
                    "instrument": history_request.inst_id,
                    "bar": history_request.bar,
                    "actual_bars": coverage.actual_bars,
                    "target_bars": coverage.target.target_bars,
                    "coverage_pct": coverage.coverage_pct,
                    "start_ms": coverage.actual_start_ms,
                    "end_ms": coverage.actual_end_ms,
                    "notes": ",".join(coverage.notes),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "status": "ERROR",
                    "instrument": history_request.inst_id,
                    "bar": history_request.bar,
                    "actual_bars": 0,
                    "target_bars": 0,
                    "coverage_pct": 0.0,
                    "start_ms": None,
                    "end_ms": None,
                    "notes": str(exc),
                }
            )
    return CacheAuditResult(pl.DataFrame(rows, schema=CACHE_AUDIT_SCHEMA))


async def _stream_cache_refresh(
    requests: tuple[HistoryRefreshRequest, ...], concurrency: int
) -> None:
    async with AsyncCacheStore() as store:
        async for event in store.stream_many(requests, concurrency=concurrency):
            if event.kind in {"completed", "failed", "summary"}:
                logger.info("%s", event.message)


SIGNAL_CONTEXT_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "raw_entry_signal",
    "entry_signal",
    "position_signal",
    "exit_signal",
    "signal_strength",
    "signal_id",
    "signal",
    "h1_close",
    "h1_trend_state",
    "h1_structure_trend_state",
    "h1_market_stage",
    "h1_structure_reason",
    "h1_market_stage_reason",
    "h1_stage_unknown_reason",
    "h1_range_width_atr",
    "h1_range_compression",
    "h1_near_range_high",
    "h1_near_range_low",
    "h1_context_available",
    "h4_close",
    "h4_trend_state",
    "h4_structure_trend_state",
    "h4_market_stage",
    "h4_structure_reason",
    "h4_market_stage_reason",
    "h4_stage_unknown_reason",
    "h4_range_width_atr",
    "h4_range_compression",
    "h4_near_range_high",
    "h4_near_range_low",
    "h4_context_available",
    "d1_close",
    "d1_trend_state",
    "d1_structure_trend_state",
    "d1_market_stage",
    "d1_structure_reason",
    "d1_market_stage_reason",
    "d1_stage_unknown_reason",
    "d1_range_width_atr",
    "d1_range_compression",
    "d1_near_range_high",
    "d1_near_range_low",
    "d1_context_available",
    "mtf_state_key",
    "mtf_structure_key",
    "mtf_stage_key",
    "mtf_event_state_key",
)


def source_inst_ids(pair: PairConfig, data_source: str) -> tuple[str, str]:
    if data_source == "spot_signal_swap_exec":
        return pair.asset.sig_symbol, pair.asset.symbol
    if data_source == "spot":
        return pair.asset.sig_symbol, pair.asset.sig_symbol
    return pair.asset.symbol, pair.asset.symbol


def apply_signal_debug_filters(
    frame: pl.DataFrame,
    filters: SignalDebugFilterConfig,
) -> pl.DataFrame:
    """Suppress diagnostic entry buckets without mutating cached strategy output."""
    if not filters.active:
        return frame
    if "entry_signal" not in frame.columns:
        return frame

    keep_entry = pl.lit(True)
    if filters.side == "long":
        keep_entry = keep_entry & (pl.col("entry_signal").cast(pl.Float64) >= 0)
    elif filters.side == "short":
        keep_entry = keep_entry & (pl.col("entry_signal").cast(pl.Float64) <= 0)
    if filters.include_signal_ids and "signal_id" in frame.columns:
        keep_entry = keep_entry & pl.col("signal_id").is_in(list(filters.include_signal_ids))
    if filters.exclude_signal_ids and "signal_id" in frame.columns:
        keep_entry = keep_entry & ~pl.col("signal_id").is_in(list(filters.exclude_signal_ids))

    expressions = [
        pl.when(keep_entry).then(pl.col("entry_signal")).otherwise(0.0).alias("entry_signal")
    ]
    if "raw_entry_signal" in frame.columns:
        expressions.append(
            pl.when(keep_entry)
            .then(pl.col("raw_entry_signal"))
            .otherwise(0.0)
            .alias("raw_entry_signal")
        )
    if filters.side == "long" and "position_signal" in frame.columns:
        expressions.append(
            pl.when(pl.col("position_signal").cast(pl.Float64) < 0)
            .then(0.0)
            .otherwise(pl.col("position_signal"))
            .alias("position_signal")
        )
    elif filters.side == "short" and "position_signal" in frame.columns:
        expressions.append(
            pl.when(pl.col("position_signal").cast(pl.Float64) > 0)
            .then(0.0)
            .otherwise(pl.col("position_signal"))
            .alias("position_signal")
        )
    if filters.side != "both" and "signal" in frame.columns:
        blocked_signal = (
            pl.col("signal").cast(pl.Float64) < 0
            if filters.side == "long"
            else pl.col("signal").cast(pl.Float64) > 0
        )
        expressions.append(
            pl.when(blocked_signal).then(0.0).otherwise(pl.col("signal")).alias("signal")
        )
    return frame.with_columns(expressions)


def load_cache_for_request(
    store: CacheStore,
    inst_id: str,
    timeframe: str,
    request: FrameRequest,
    *,
    trim_to_target: bool = True,
    source: CandleSource = "trade",
) -> tuple[pl.DataFrame, HistoryCoverage]:
    df, coverage = store.bars(
        HistoryRequest(
            inst_id=inst_id,
            bar=timeframe,
            days=request.days,
            min_bars=request.min_bars,
            refresh=request.refresh,
            source=source,
        )
    )
    if trim_to_target and df.height > coverage.target.target_bars:
        df = df.tail(coverage.target.target_bars)
        coverage = validate_history(df, coverage.target, refreshed=coverage.refreshed)
    if request.min_coverage_pct > 0 and coverage.coverage_pct < request.min_coverage_pct:
        raise DataCoverageError(coverage, request.min_coverage_pct)
    return df, coverage


def load_context_for_request(
    store: CacheStore,
    inst_id: str,
    context: ContextSpec,
    request: FrameRequest,
) -> tuple[pl.DataFrame, HistoryCoverage]:
    context_request = replace(
        request,
        bar=context.bar,
        min_bars=_context_min_bars(context.bar, request.days, role="higher_context"),
    )
    return load_cache_for_request(
        store,
        inst_id,
        context.bar,
        context_request,
        source=context.source,
    )


def _timeframe_interval_ms(timeframe: str) -> int:
    normalized = timeframe.replace(" ", "")
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
    return 0


def _expected_bars_for_days(timeframe: str, days: int) -> int:
    interval_ms = _timeframe_interval_ms(timeframe)
    if interval_ms <= 0:
        return 0
    return max(1, int((days * 86_400_000 + interval_ms - 1) / interval_ms))


def _context_min_bars(
    timeframe: str,
    days: int,
    *,
    role: Literal["higher_context"],
) -> int:
    expected = _expected_bars_for_days(timeframe, days)
    return expected + 250


def _infer_step_ms(df: pl.DataFrame, fallback: int) -> int:
    if df.height < 2 or "timestamp" not in df.columns:
        return fallback
    diffs = df.select(pl.col("timestamp").cast(pl.Int64).diff().median()).item()
    return int(diffs or fallback)


def attach_higher_timeframe_context(
    base_df: pl.DataFrame,
    htf_df: pl.DataFrame,
    *,
    prefix: str,
) -> pl.DataFrame:
    """Attach last fully closed higher-timeframe bar known at the H1 row timestamp."""
    if base_df.is_empty() or htf_df.is_empty() or "timestamp" not in htf_df.columns:
        return base_df
    htf_step_ms = _infer_step_ms(htf_df, 4 * 60 * 60 * 1000)
    context_cols = [column for column in htf_df.columns if column != "timestamp"]
    if not context_cols:
        return base_df
    context = (
        htf_df.sort("timestamp")
        .with_columns((pl.col("timestamp").cast(pl.Int64) + htf_step_ms).alias("_known_ts"))
        .select(
            "_known_ts",
            *(pl.col(column).alias(f"{prefix}_{column}") for column in context_cols),
        )
    )
    joined = base_df.sort("timestamp").join_asof(
        context,
        left_on="timestamp",
        right_on="_known_ts",
        strategy="backward",
    )
    return joined.drop("_known_ts") if "_known_ts" in joined.columns else joined


def _trend_state_expr(prefix: str = "") -> pl.Expr:
    close = pl.col(f"{prefix}close") if prefix else pl.col("close")
    ema20 = pl.col(f"{prefix}ema_20") if prefix else pl.col("ema_20")
    ema50 = pl.col(f"{prefix}ema_50") if prefix else pl.col("ema_50")
    return (
        pl.when(close.is_null() | ema20.is_null() | ema50.is_null())
        .then(pl.lit(None, dtype=pl.Utf8))
        .when((close > ema20) & (ema20 > ema50))
        .then(pl.lit("bullish"))
        .when((close < ema20) & (ema20 < ema50))
        .then(pl.lit("bearish"))
        .otherwise(pl.lit("mixed"))
    )


STRUCTURE_CONTEXT_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "structure_trend_state",
    "market_stage",
    "structure_reason",
    "market_stage_reason",
    "stage_unknown_reason",
    "range_width_atr",
    "range_width_atr_threshold",
    "range_width_threshold_mode",
    "range_width_threshold_ready",
    "range_width_threshold_source",
    "range_compression",
    "near_range_high",
    "near_range_low",
    "last_swing_high",
    "last_swing_low",
)


def _structure_context_frame(
    df: pl.DataFrame,
    classifier_config: StructureClassifierConfig | None = None,
) -> pl.DataFrame:
    if df.is_empty() or "timestamp" not in df.columns:
        return df
    enriched = df
    if not {"ema_20", "ema_50", "ema_200", "atr_14"} <= set(enriched.columns):
        enriched = add_indicators(enriched)
    if "structure_trend_state" not in enriched.columns or "market_stage" not in enriched.columns:
        enriched = add_price_structure_stage_features(config=classifier_config)(enriched)
    keep = [column for column in STRUCTURE_CONTEXT_COLUMNS if column in enriched.columns]
    return enriched.select(keep)


def _compact_higher_timeframe_context(
    htf_df: pl.DataFrame,
    classifier_config: StructureClassifierConfig | None = None,
) -> pl.DataFrame:
    if htf_df.is_empty() or "timestamp" not in htf_df.columns:
        return htf_df
    enriched = add_macd_histogram()(add_indicators(htf_df))
    enriched = add_price_structure_stage_features(config=classifier_config)(enriched)
    keep = [
        column
        for column in (
            "timestamp",
            "close",
            "ema_20",
            "ema_50",
            "ema_200",
            "atr_14",
            "macd_hist",
            *STRUCTURE_CONTEXT_COLUMNS[1:],
        )
        if column in enriched.columns
    ]
    compact = enriched.select(keep)
    if {"close", "ema_20", "ema_50"} <= set(compact.columns):
        compact = compact.with_columns(_trend_state_expr().alias("trend_state"))
    return compact


def _add_missing_context_columns(base_df: pl.DataFrame, prefix: str) -> pl.DataFrame:
    columns = {
        f"{prefix}_close": pl.Float64,
        f"{prefix}_ema_20": pl.Float64,
        f"{prefix}_ema_50": pl.Float64,
        f"{prefix}_ema_200": pl.Float64,
        f"{prefix}_atr_14": pl.Float64,
        f"{prefix}_macd_hist": pl.Float64,
        f"{prefix}_trend_state": pl.Utf8,
        f"{prefix}_structure_trend_state": pl.Utf8,
        f"{prefix}_market_stage": pl.Utf8,
        f"{prefix}_structure_reason": pl.Utf8,
        f"{prefix}_market_stage_reason": pl.Utf8,
        f"{prefix}_stage_unknown_reason": pl.Utf8,
        f"{prefix}_range_width_atr": pl.Float64,
        f"{prefix}_range_width_atr_threshold": pl.Float64,
        f"{prefix}_range_width_threshold_mode": pl.Utf8,
        f"{prefix}_range_width_threshold_ready": pl.Boolean,
        f"{prefix}_range_width_threshold_source": pl.Utf8,
        f"{prefix}_range_compression": pl.Boolean,
        f"{prefix}_near_range_high": pl.Boolean,
        f"{prefix}_near_range_low": pl.Boolean,
        f"{prefix}_last_swing_high": pl.Float64,
        f"{prefix}_last_swing_low": pl.Float64,
        f"{prefix}_context_available": pl.Boolean,
    }
    expressions = []
    for column, dtype in columns.items():
        if column in base_df.columns:
            continue
        if column.endswith("_context_available"):
            expressions.append(pl.lit(False).alias(column))
        else:
            expressions.append(pl.lit(None, dtype=dtype).alias(column))
    if not expressions:
        return base_df
    return base_df.with_columns(expressions)


def _mark_higher_context_available(base_df: pl.DataFrame, prefix: str) -> pl.DataFrame:
    available_col = f"{prefix}_context_available"
    close_col = f"{prefix}_close"
    if close_col not in base_df.columns:
        return _add_missing_context_columns(base_df, prefix)
    return base_df.with_columns(pl.col(close_col).is_not_null().alias(available_col))


def _attach_higher_context(
    base_df: pl.DataFrame,
    context_df: pl.DataFrame,
    prefix: str,
    classifier_config: StructureClassifierConfig | None = None,
) -> pl.DataFrame:
    compact = _compact_higher_timeframe_context(context_df, classifier_config)
    attached = attach_higher_timeframe_context(base_df, compact, prefix=prefix)
    return _mark_higher_context_available(attached, prefix)


def _attach_base_context_aliases(
    base_df: pl.DataFrame,
    prefix: str,
    classifier_config: StructureClassifierConfig | None = None,
) -> pl.DataFrame:
    alias_map = {
        "close": f"{prefix}_close",
        "ema_20": f"{prefix}_ema_20",
        "ema_50": f"{prefix}_ema_50",
        "ema_200": f"{prefix}_ema_200",
        "atr_14": f"{prefix}_atr_14",
        "macd_hist": f"{prefix}_macd_hist",
    }
    work = base_df
    if not {"ema_20", "ema_50", "ema_200", "atr_14"} <= set(work.columns):
        work = add_indicators(work)
    if "macd_hist" not in work.columns:
        work = add_macd_histogram()(work)
    expressions = [
        pl.col(source).alias(target)
        for source, target in alias_map.items()
        if source in work.columns and target not in work.columns
    ]
    has_trend_inputs = {"close", "ema_20", "ema_50"} <= set(work.columns)
    if has_trend_inputs and f"{prefix}_trend_state" not in work.columns:
        expressions.append(_trend_state_expr().alias(f"{prefix}_trend_state"))
    if "structure_trend_state" not in work.columns or "market_stage" not in work.columns:
        work = add_price_structure_stage_features(config=classifier_config)(work)
    structure_aliases = {
        column: f"{prefix}_{column}"
        for column in STRUCTURE_CONTEXT_COLUMNS
        if column != "timestamp"
    }
    expressions.extend(
        pl.col(source).alias(target)
        for source, target in structure_aliases.items()
        if source in work.columns and target not in work.columns
    )
    expressions.append(pl.lit(True).alias(f"{prefix}_context_available"))
    return work.with_columns(expressions)


def _state_component(frame: pl.DataFrame, column: str) -> pl.Expr:
    if column in frame.columns:
        return pl.col(column).cast(pl.Utf8).fill_null("data_error")
    return pl.lit("data_error")


def add_mtf_state_keys(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    # These readable keys compress already-closed multi-timeframe context;
    # they are diagnostic descriptions, not forecasts of the next state.
    expressions = [
        pl.concat_str(
            [
                _state_component(frame, "d1_structure_trend_state"),
                _state_component(frame, "h4_market_stage"),
                _state_component(frame, "h1_market_stage"),
            ],
            separator="|",
        ).alias("mtf_state_key"),
        pl.concat_str(
            [
                _state_component(frame, "d1_structure_trend_state"),
                _state_component(frame, "h4_structure_trend_state"),
                _state_component(frame, "h1_structure_trend_state"),
            ],
            separator="|",
        ).alias("mtf_structure_key"),
        pl.concat_str(
            [
                _state_component(frame, "d1_market_stage"),
                _state_component(frame, "h4_market_stage"),
                _state_component(frame, "h1_market_stage"),
            ],
            separator="|",
        ).alias("mtf_stage_key"),
    ]
    if "liquidity_event_type" in frame.columns:
        expressions.append(
            pl.concat_str(
                [
                    _state_component(frame, "d1_structure_trend_state"),
                    _state_component(frame, "h4_market_stage"),
                    _state_component(frame, "h1_market_stage"),
                    _state_component(frame, "liquidity_event_type"),
                ],
                separator="|",
            ).alias("mtf_event_state_key")
        )
    return frame.with_columns(expressions)


def _context_bundle_for_strategy(strategy: StrategyBehavior) -> tuple[ContextSpec, ...]:
    required_columns = getattr(strategy, "required_columns", ())
    strategy_name = str(getattr(strategy, "name", ""))
    if strategy_name.startswith("structure_event_") or any(
        column.startswith(("h1_", "h4_", "d1_")) for column in required_columns
    ):
        return DEFAULT_CONTEXTS
    return ()


def _attach_contexts(
    store: CacheStore,
    frame: pl.DataFrame,
    signal_inst_id: str,
    request: FrameRequest,
    contexts: tuple[ContextSpec, ...],
    classifier_config: StructureClassifierConfig | None = None,
) -> tuple[pl.DataFrame, tuple[HistoryCoverage, ...]]:
    context_summaries: list[HistoryCoverage] = []
    work = frame
    for spec in contexts:
        if spec.role == "base":
            work = _attach_base_context_aliases(work, spec.prefix, classifier_config)
            continue
        context_df, summary = load_context_for_request(store, signal_inst_id, spec, request)
        context_summaries.append(summary)
        work = _attach_higher_context(work, context_df, spec.prefix, classifier_config)
    return work, tuple(context_summaries)


def _attach_strategy_context(
    store: CacheStore,
    signal_df: pl.DataFrame,
    signal_inst_id: str,
    strategy: StrategyBehavior,
    request: FrameRequest,
    *,
    contexts: tuple[ContextSpec, ...] | None = None,
) -> tuple[pl.DataFrame, tuple[HistoryCoverage, ...]]:
    bundle = _context_bundle_for_strategy(strategy) if contexts is None else contexts
    if not bundle:
        return signal_df, ()
    return _attach_contexts(store, signal_df, signal_inst_id, request, bundle)


def coverage_metadata(
    coverage: HistoryCoverage,
    *,
    prefix: str = "data_coverage",
) -> tuple[str, ...]:
    notes = ",".join(coverage.notes) or "none"
    return (
        f"{prefix}_inst={coverage.inst_id}",
        f"{prefix}_bar={coverage.bar}",
        f"{prefix}_actual_bars={coverage.actual_bars}",
        f"{prefix}_target_bars={coverage.target.target_bars}",
        f"{prefix}_pct={coverage.coverage_pct:.1f}",
        f"{prefix}_start={coverage.actual_start_ms or 'n/a'}",
        f"{prefix}_end={coverage.actual_end_ms or 'n/a'}",
        f"{prefix}_notes={notes}",
    )


def prepare_classifier_frame(
    store: CacheStore,
    request: FrameRequest,
    classifier: StructureClassifierConfig | None = None,
    *,
    contexts: tuple[ContextSpec, ...] = (),
) -> PreparedClassifierFrame:
    classifier = classifier or StructureClassifierConfig.default()
    pair = request.pair
    signal_inst_id, _execution_inst_id = source_inst_ids(pair, request.data_source)
    try:
        base_df, base_coverage = load_cache_for_request(store, signal_inst_id, request.bar, request)
    except FileNotFoundError:
        if not request.allow_swap_signal_fallback or signal_inst_id == pair.asset.symbol:
            raise
        signal_inst_id = pair.asset.symbol
        base_df, base_coverage = load_cache_for_request(store, signal_inst_id, request.bar, request)

    work = add_macd_histogram()(add_indicators(base_df))
    work = add_price_structure_stage_features(config=classifier)(work)
    work, context_coverage = _attach_contexts(
        store, work, signal_inst_id, request, contexts, classifier
    )
    coverage = (base_coverage, *context_coverage)
    if contexts:
        work = add_mtf_state_keys(work)
    work = work.with_columns(pl.lit(request.bar).alias("timeframe"))
    metadata = (
        f"signal_inst={signal_inst_id}",
        f"classifier_bar={request.bar}",
        *(summary.note() for summary in coverage),
    )
    return PreparedClassifierFrame(
        pair=pair,
        frame=work,
        signal_inst_id=signal_inst_id,
        execution_inst_id=None,
        coverage=coverage,
        metadata=metadata,
    )


def prepare_backtest_frame(
    store: CacheStore,
    request: FrameRequest,
    strategy: StrategyBehavior,
    options: BacktestFrameOptions,
) -> PreparedBacktestFrame:
    return prepare_signal_frame(store, request, strategy, options=options)


def prepare_signal_frame(
    store: CacheStore,
    request: FrameRequest,
    strategy: StrategyBehavior,
    *,
    options: BacktestFrameOptions,
    contexts: tuple[ContextSpec, ...] | None = None,
) -> PreparedBacktestFrame:
    pair = request.pair
    signal_inst_id, execution_inst_id = source_inst_ids(pair, request.data_source)
    timeframe = request.bar
    try:
        signal_df, signal_summary = load_cache_for_request(
            store, signal_inst_id, timeframe, request
        )
    except FileNotFoundError:
        if not request.allow_swap_signal_fallback or signal_inst_id == pair.asset.symbol:
            raise
        signal_inst_id = pair.asset.symbol
        signal_df, signal_summary = load_cache_for_request(
            store, signal_inst_id, timeframe, request
        )

    signal_df, context_summaries = _attach_strategy_context(
        store,
        signal_df,
        signal_inst_id,
        strategy,
        request,
        contexts=contexts,
    )
    has_context = bool(context_summaries)
    if execution_inst_id == signal_inst_id:
        execution_summary = signal_summary
        if options.signal_filters.active or has_context:
            signal_frame = compute_signal_frame(signal_df, strategy)
            frame = add_mtf_state_keys(
                apply_signal_debug_filters(signal_frame, options.signal_filters)
            )
            precomputed_signal = True
        else:
            frame = signal_df
            precomputed_signal = False
    else:
        execution_df, execution_summary = load_cache_for_request(
            store, execution_inst_id, timeframe, request
        )
        signal_frame = add_mtf_state_keys(
            apply_signal_debug_filters(
                compute_signal_frame(signal_df, strategy),
                options.signal_filters,
            )
        )
        signal_frame = signal_frame.select(
            *(column for column in SIGNAL_CONTEXT_COLUMNS if column in signal_frame.columns)
        )
        frame = add_indicators(execution_df).join(signal_frame, on="timestamp", how="inner")
        precomputed_signal = True

    coverage = (signal_summary, execution_summary, *context_summaries)
    metadata = (
        *options.metadata,
        f"signal_inst={signal_inst_id}",
        f"execution_inst={execution_inst_id}",
        *(summary.note() for summary in coverage),
    )
    log_cache_warnings(coverage, signal_inst_id, execution_inst_id)
    return PreparedBacktestFrame(
        pair=pair,
        frame=frame,
        signal_inst_id=signal_inst_id,
        execution_inst_id=execution_inst_id,
        coverage=coverage,
        metadata=metadata,
        precomputed_signal=precomputed_signal,
        signal_coverage=signal_summary,
        execution_coverage=execution_summary,
    )


def log_cache_warnings(
    coverage: tuple[HistoryCoverage, ...], signal_inst_id: str, execution_inst_id: str
) -> None:
    for summary in coverage:
        if not summary.notes:
            continue
        label = (
            summary.inst_id
            if summary.inst_id in {signal_inst_id, execution_inst_id}
            else f"{summary.inst_id} {summary.bar}"
        )
        logger.warning("cache warning %s: %s", label, ", ".join(summary.notes))
