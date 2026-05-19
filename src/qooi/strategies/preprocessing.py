"""Strategy-owned frame preprocessing for research backtests and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

import polars as pl

from qooi.core.config import PairConfig
from qooi.exchange.market import CandleSource
from qooi.exchange.store import (
    CacheStore,
    HistoryCoverage,
    HistoryRequest,
    validate_history,
)
from qooi.research.config import ResolvedBacktestConfig
from qooi.strategies.features import (
    StructureClassifierConfig,
    add_price_structure_stage_features,
)
from qooi.strategies.indicators import add_indicators, add_macd_histogram
from qooi.strategies.specs import StrategyBehavior, compute_signal_frame


class DataCoverageError(RuntimeError):
    def __init__(self, coverage: HistoryCoverage, required_pct: float) -> None:
        self.coverage = coverage
        self.required_pct = required_pct
        super().__init__(
            f"cache coverage below threshold for {coverage.inst_id}: "
            f"{coverage.coverage_pct:.1f}% < {required_pct:.1f}%"
        )


@dataclass(frozen=True)
class PreparedBacktestFrame:
    pair: PairConfig
    frame: pl.DataFrame
    precomputed_signal: bool
    signal_inst_id: str
    execution_inst_id: str
    signal_coverage: HistoryCoverage
    execution_coverage: HistoryCoverage
    metadata: tuple[str, ...]


@dataclass(frozen=True)
class PreparedClassifierFrame:
    pair: PairConfig
    frame: pl.DataFrame
    signal_inst_id: str
    metadata: tuple[str, ...]


@dataclass(frozen=True)
class TimeframeContextSpec:
    timeframe: str
    prefix: str
    source: CandleSource = "trade"
    role: Literal["base", "higher_context"] = "higher_context"
    required: bool = True


@dataclass(frozen=True)
class ClassifierContextConfig:
    base_timeframe: str = "1H"
    higher_timeframes: tuple[TimeframeContextSpec, ...] = (
        TimeframeContextSpec("1H", "h1", role="base"),
        TimeframeContextSpec("4H", "h4", role="higher_context"),
        TimeframeContextSpec("1D", "d1", role="higher_context"),
    )
    classifier: StructureClassifierConfig = field(default_factory=StructureClassifierConfig.default)


DEFAULT_MTF_CONTEXT_BUNDLE: tuple[TimeframeContextSpec, ...] = (
    TimeframeContextSpec("1H", "h1", role="base"),
    TimeframeContextSpec("4H", "h4", role="higher_context"),
    TimeframeContextSpec("1D", "d1", role="higher_context"),
)


def source_inst_ids(pair: PairConfig, data_source: str) -> tuple[str, str]:
    if data_source == "spot_signal_swap_exec":
        return pair.asset.sig_symbol, pair.asset.symbol
    if data_source == "spot":
        return pair.asset.sig_symbol, pair.asset.sig_symbol
    return pair.asset.symbol, pair.asset.symbol


def apply_signal_debug_filters(
    frame: pl.DataFrame,
    config: ResolvedBacktestConfig,
) -> pl.DataFrame:
    """Suppress diagnostic entry buckets without mutating cached strategy output."""
    filters = config.signal_filters
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
        pl.when(keep_entry)
        .then(pl.col("entry_signal"))
        .otherwise(0.0)
        .alias("entry_signal")
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


def load_cache(
    store: CacheStore,
    inst_id: str,
    timeframe: str,
    config: ResolvedBacktestConfig,
    *,
    refresh: bool,
    trim_to_target: bool = True,
    source: CandleSource = "trade",
) -> tuple[pl.DataFrame, HistoryCoverage]:
    df, coverage = store.bars(
        HistoryRequest(
            inst_id=inst_id,
            bar=timeframe,
            days=config.days,
            min_bars=config.min_bars,
            refresh=refresh,
            source=source,
        )
    )
    if trim_to_target and df.height > coverage.target.target_bars:
        df = df.tail(coverage.target.target_bars)
        coverage = validate_history(df, coverage.target, refreshed=coverage.refreshed)
    if config.risk_gates.min_coverage_pct > 0 and (
        coverage.coverage_pct < config.risk_gates.min_coverage_pct
    ):
        raise DataCoverageError(coverage, config.risk_gates.min_coverage_pct)
    return df, coverage


def load_timeframe_context(
    store: CacheStore,
    inst_id: str,
    timeframe: str,
    config: ResolvedBacktestConfig,
    args: Any,
    source: CandleSource = "trade",
    role: Literal["higher_context"] = "higher_context",
) -> tuple[pl.DataFrame, HistoryCoverage]:
    """Load a research-only context timeframe through the normal cache policy."""
    context_config = replace(
        config,
        min_bars=_context_min_bars(timeframe, config.days, role=role),
    )
    return load_cache(
        store,
        inst_id,
        timeframe,
        context_config,
        refresh=bool(getattr(args, "refresh_cache", False)),
        source=source,
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
    return base_df.sort("timestamp").join_asof(
        context,
        left_on="timestamp",
        right_on="_known_ts",
        strategy="backward",
    )


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


def _context_bundle_for_strategy(strategy: StrategyBehavior) -> tuple[TimeframeContextSpec, ...]:
    required_columns = getattr(strategy, "required_columns", ())
    strategy_name = str(getattr(strategy, "name", ""))
    if strategy_name.startswith("structure_event_") or any(
        column.startswith(("h1_", "h4_", "d1_")) for column in required_columns
    ):
        return DEFAULT_MTF_CONTEXT_BUNDLE
    return ()


def _attach_strategy_context(
    store: CacheStore,
    signal_df: pl.DataFrame,
    signal_inst_id: str,
    strategy: StrategyBehavior,
    args: Any,
    config: ResolvedBacktestConfig,
) -> tuple[pl.DataFrame, tuple[HistoryCoverage, ...]]:
    bundle = _context_bundle_for_strategy(strategy)
    if not bundle:
        return signal_df, ()
    context_summaries: list[HistoryCoverage] = []
    work = signal_df
    for spec in bundle:
        if spec.role == "base":
            work = _attach_base_context_aliases(work, spec.prefix)
            continue
        context_df, summary = load_timeframe_context(
            store,
            signal_inst_id,
            spec.timeframe,
            config,
            args,
            source=spec.source,
            role=spec.role,
        )
        context_summaries.append(summary)
        compact = _compact_higher_timeframe_context(context_df)
        work = attach_higher_timeframe_context(work, compact, prefix=spec.prefix)
        work = _mark_higher_context_available(work, spec.prefix)
    return work, tuple(context_summaries)


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


class ClassifierFramePipeline:
    def __init__(
        self,
        store: CacheStore,
        args: Any,
        config: ResolvedBacktestConfig,
        classifier_context: ClassifierContextConfig | None = None,
    ) -> None:
        self.store = store
        self.args = args
        self.config = config
        self.classifier_context = classifier_context or ClassifierContextConfig()

    def load_base(self, pair: PairConfig) -> tuple[pl.DataFrame, HistoryCoverage, str]:
        signal_inst_id, _execution_inst_id = source_inst_ids(pair, self.config.data_source)
        try:
            frame, coverage = load_cache(
                self.store,
                signal_inst_id,
                self.classifier_context.base_timeframe or pair.asset.timeframe,
                self.config,
                refresh=bool(getattr(self.args, "refresh_cache", False)),
            )
        except FileNotFoundError:
            if not bool(getattr(self.args, "allow_swap_signal_fallback", False)) or (
                signal_inst_id == pair.asset.symbol
            ):
                raise
            signal_inst_id = pair.asset.symbol
            frame, coverage = load_cache(
                self.store,
                signal_inst_id,
                self.classifier_context.base_timeframe or pair.asset.timeframe,
                self.config,
                refresh=bool(getattr(self.args, "refresh_cache", False)),
            )
        return frame, coverage, signal_inst_id

    def load_contexts(
        self,
        signal_inst_id: str,
    ) -> tuple[tuple[TimeframeContextSpec, pl.DataFrame, HistoryCoverage], ...]:
        contexts = []
        for spec in self.classifier_context.higher_timeframes:
            if spec.role != "higher_context":
                continue
            context_df, coverage = load_timeframe_context(
                self.store,
                signal_inst_id,
                spec.timeframe,
                self.config,
                self.args,
                source=spec.source,
                role=spec.role,
            )
            contexts.append((spec, context_df, coverage))
        return tuple(contexts)

    def attach_contexts(
        self,
        base_df: pl.DataFrame,
        contexts: tuple[tuple[TimeframeContextSpec, pl.DataFrame, HistoryCoverage], ...],
    ) -> pl.DataFrame:
        work = base_df
        for spec in self.classifier_context.higher_timeframes:
            if spec.role == "base":
                work = _attach_base_context_aliases(
                    work,
                    spec.prefix,
                    self.classifier_context.classifier,
                )
        for spec, context_df, _coverage in contexts:
            compact = _compact_higher_timeframe_context(
                context_df,
                self.classifier_context.classifier,
            )
            work = attach_higher_timeframe_context(work, compact, prefix=spec.prefix)
            work = _mark_higher_context_available(work, spec.prefix)
        return work

    def add_state_keys(self, frame: pl.DataFrame) -> pl.DataFrame:
        return add_mtf_state_keys(frame)

    def prepare(self, pair: PairConfig) -> PreparedClassifierFrame:
        base_df, base_coverage, signal_inst_id = self.load_base(pair)
        work = add_macd_histogram()(add_indicators(base_df))
        work = add_price_structure_stage_features(config=self.classifier_context.classifier)(work)
        contexts = self.load_contexts(signal_inst_id)
        work = self.add_state_keys(self.attach_contexts(work, contexts))
        metadata = (
            *self.config.metadata(),
            f"signal_inst={signal_inst_id}",
            base_coverage.note(),
            *(coverage.note() for _spec, _frame, coverage in contexts),
        )
        return PreparedClassifierFrame(
            pair=pair,
            frame=work,
            signal_inst_id=signal_inst_id,
            metadata=metadata,
        )


def prepare_classifier_frame(
    store: CacheStore,
    pair: PairConfig,
    args: Any,
    config: ResolvedBacktestConfig,
    classifier_context: ClassifierContextConfig | None = None,
) -> PreparedClassifierFrame:
    return ClassifierFramePipeline(store, args, config, classifier_context).prepare(pair)


class BacktestFramePipeline:
    def __init__(
        self,
        store: CacheStore,
        args: Any,
        config: ResolvedBacktestConfig,
    ) -> None:
        self.store = store
        self.args = args
        self.config = config

    def prepare(self, pair: PairConfig, strategy: StrategyBehavior) -> PreparedBacktestFrame:
        signal_inst_id, execution_inst_id = source_inst_ids(pair, self.config.data_source)
        timeframe = pair.asset.timeframe
        refresh = bool(getattr(self.args, "refresh_cache", False))
        try:
            signal_df, signal_summary = load_cache(
                self.store, signal_inst_id, timeframe, self.config, refresh=refresh
            )
        except FileNotFoundError:
            if not bool(getattr(self.args, "allow_swap_signal_fallback", False)) or (
                signal_inst_id == pair.asset.symbol
            ):
                raise
            signal_inst_id = pair.asset.symbol
            signal_df, signal_summary = load_cache(
                self.store, signal_inst_id, timeframe, self.config, refresh=refresh
            )

        signal_df, context_summaries = _attach_strategy_context(
            self.store,
            signal_df,
            signal_inst_id,
            strategy,
            self.args,
            self.config,
        )
        has_context = bool(context_summaries)

        if execution_inst_id == signal_inst_id:
            execution_summary = signal_summary
            if self.config.signal_filters.active or has_context:
                signal_frame = compute_signal_frame(signal_df, strategy)
                frame = add_mtf_state_keys(
                    apply_signal_debug_filters(signal_frame, self.config)
                )
                precomputed_signal = True
            else:
                frame = signal_df
                precomputed_signal = False
        else:
            execution_df, execution_summary = load_cache(
                self.store, execution_inst_id, timeframe, self.config, refresh=refresh
            )
            signal_frame = add_mtf_state_keys(
                apply_signal_debug_filters(compute_signal_frame(signal_df, strategy), self.config)
            )
            signal_cols = [
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
            ]
            signal_frame = signal_frame.select(
                *(column for column in signal_cols if column in signal_frame.columns)
            )
            frame = add_indicators(execution_df).join(signal_frame, on="timestamp", how="inner")
            precomputed_signal = True

        metadata = (
            *self.config.metadata(),
            f"signal_inst={signal_inst_id}",
            f"execution_inst={execution_inst_id}",
            signal_summary.note(),
            execution_summary.note(),
            *(summary.note() for summary in context_summaries),
        )
        if signal_summary.notes:
            print(f"cache warning {signal_inst_id}: {', '.join(signal_summary.notes)}")
        if execution_summary.notes and execution_inst_id != signal_inst_id:
            print(f"cache warning {execution_inst_id}: {', '.join(execution_summary.notes)}")
        for summary in context_summaries:
            if summary.notes:
                print(f"cache warning {summary.inst_id} {summary.bar}: {', '.join(summary.notes)}")

        return PreparedBacktestFrame(
            pair=pair,
            frame=frame,
            precomputed_signal=precomputed_signal,
            signal_inst_id=signal_inst_id,
            execution_inst_id=execution_inst_id,
            signal_coverage=signal_summary,
            execution_coverage=execution_summary,
            metadata=metadata,
        )


def prepare_backtest_frame(
    store: CacheStore,
    pair: PairConfig,
    strategy: StrategyBehavior,
    args: Any,
    config: ResolvedBacktestConfig,
) -> PreparedBacktestFrame:
    return BacktestFramePipeline(store, args, config).prepare(pair, strategy)
