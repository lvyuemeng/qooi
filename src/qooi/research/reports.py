"""Research evaluation and backtest report helpers."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import polars as pl

from qooi.core.basket import ExitConfig
from qooi.core.evaluate import (
    BacktestDiagnostics,
    EngineDataAudit,
    FeatureDiagnostics,
    Report,
    format_backtest_report,
    format_benchmark_report,
    format_table,
)
from qooi.core.executor import BacktestExecutor
from qooi.core.recovery import (
    GridRecovery,
    HedgeRecovery,
    MartingaleRecovery,
    NoRecovery,
    RecoveryPolicy,
    ReverseRecovery,
    ZScoreReversionRecovery,
)
from qooi.core.styles import cross_validate, rolling_window, walk_forward
from qooi.exchange.store import AsyncCacheStore, CacheStore, HistoryRequest
from qooi.research.config import (
    ResearchCommandConfig,
    ResearchOutputName,
    resolve_research_outputs,
    risk_gate_metadata,
)
from qooi.research.data import (
    DEFAULT_CONTEXTS,
    BacktestFrameOptions,
    CacheAuditRequest,
    DataCoverageError,
    add_mtf_state_keys,
    build_history_refresh_requests,
    coverage_metadata,
    prepare_backtest_frame,
    prepare_classifier_frame,
)
from qooi.research.states import classifier_health
from qooi.research.tables import ArtifactBundle, build_transition_bundle, write_bundle
from qooi.strategies import StrategyBehavior, compute_signal_frame, strategy_signal_diagnostics
from qooi.strategies.catalog import (
    strategy_metadata,
    strategy_selection,
)
from qooi.strategies.features import add_liquidity_sweep_features, add_none_context_diagnostics

logger = logging.getLogger(__name__)

EVIDENCE_ROWS = (
    ["timeframe-classifier", "peer timeframe cache -> independent classifier health"],
    ["dynamic-transition-discovery", "classifier frame -> transition pattern artifacts"],
    ["pattern-quality", "pattern table -> scored pattern candidates"],
    ["trade-record-modulation", "strategy signal/backtest branch -> trades"],
)

CONTROL_SCHEMA = {
    "artifact": pl.Utf8,
    "base_feature": pl.Utf8,
    "base_value": pl.Utf8,
    "modulator_feature": pl.Utf8,
    "modulator_value": pl.Utf8,
    "base_trades": pl.Int64,
    "conditional_trades": pl.Int64,
    "base_expectancy": pl.Float64,
    "conditional_expectancy": pl.Float64,
    "delta_expectancy": pl.Float64,
    "classification": pl.Utf8,
    "sufficient_base": pl.Boolean,
    "sufficient_cell": pl.Boolean,
    "significant": pl.Boolean,
}


def add_market_state_reductions(frame: pl.DataFrame) -> pl.DataFrame:
    work = frame
    if "market_stage" in work.columns and "market_stage_reduced" not in work.columns:
        work = work.with_columns(pl.col("market_stage").cast(pl.Utf8).alias("market_stage_reduced"))
    if "h4_market_stage" in work.columns and "h4_market_stage_reduced" not in work.columns:
        work = work.with_columns(
            pl.col("h4_market_stage").cast(pl.Utf8).alias("h4_market_stage_reduced")
        )
    if "d1_market_stage" in work.columns and "d1_market_stage_reduced" not in work.columns:
        work = work.with_columns(
            pl.col("d1_market_stage").cast(pl.Utf8).alias("d1_market_stage_reduced")
        )
    return work


@dataclass(frozen=True)
class TradeRecordControlResult:
    frame: pl.DataFrame
    text: str


def trade_record_control(
    trades: pl.DataFrame,
    *,
    min_base_trades: int,
    min_cell_trades: int,
    practical_delta_threshold: float,
) -> TradeRecordControlResult:
    out = _trade_record_control_frame(
        trades,
        min_base_trades=min_base_trades,
        min_cell_trades=min_cell_trades,
        practical_delta_threshold=practical_delta_threshold,
    )
    text = "Trade-record control\n" + format_table(
        ["Metric", "Value"],
        [["rows", str(out.height)], ["significant", str(_bool_sum(out, "significant"))]],
    )
    return TradeRecordControlResult(out, text)


def mode_config(mode: str) -> RecoveryPolicy:
    if mode == "grid":
        return GridRecovery(zone_atr=1.0, multiplier=2.0, max_levels=3)
    if mode == "martingale":
        return MartingaleRecovery(zone_atr=1.0, max_levels=3)
    if mode == "reverse":
        return ReverseRecovery(zone_atr=1.0, max_levels=3)
    if mode == "hedge":
        return HedgeRecovery(zone_atr=1.0)
    if mode == "zscore_recovery":
        return ZScoreReversionRecovery(zone_atr=1.0, multiplier=1.0, max_levels=1)
    return NoRecovery()


def exit_config_from_command(command: ResearchCommandConfig) -> ExitConfig:
    return ExitConfig(
        stop_mult=command.exit.stop_mult,
        target_mult=command.exit.target_mult,
        trail_mult=command.exit.trail_mult,
        max_bars=command.exit.max_bars,
        breakeven_after_target=command.exit.breakeven_after_target,
    )


def backtest_frame_options_from_command(command: ResearchCommandConfig) -> BacktestFrameOptions:
    return BacktestFrameOptions(
        signal_filters=command.signal_filters,
        metadata=command.metadata(),
    )


def strategy_selection_from_config(command: ResearchCommandConfig):
    labels = command.strategy.strategies
    return strategy_selection(
        labels,
        benchmark=command.strategy.benchmark,
        benchmark_group=command.strategy.benchmark_group,
        default=command.strategy.strategy,
    )


def run_backtest_workflow(command: ResearchCommandConfig) -> str:
    pairs = command.pairs()
    selection = strategy_selection_from_config(command)
    recovery_cfg = mode_config(command.strategy.mode)
    exit_cfg = exit_config_from_command(command)

    if len(selection.strategies) > 1:
        benchmark_results = []
        all_reports = []
        for strategy in selection.strategies:
            reports = run_reports(pairs, strategy, recovery_cfg, exit_cfg, command)
            all_reports.extend(reports)
            benchmark_results.append((strategy.name, reports))
        _assert_reports_pass(all_reports, command)
        output = format_benchmark_report(
            mode=command.strategy.mode,
            benchmark_results=benchmark_results,
            diagnostics=command.strategy.diagnostics,
        )
        return output

    strategy = selection.strategies[0]
    if command.strategy.style != "single":
        return run_style(pairs, strategy, recovery_cfg, exit_cfg, command)

    reports = run_reports(pairs, strategy, recovery_cfg, exit_cfg, command)
    _assert_reports_pass(reports, command)
    signal_diagnostics = []
    wants_diagnostics = command.strategy.diagnostics
    if wants_diagnostics:
        for pair, report in zip(pairs, reports, strict=False):
            try:
                prepared = prepare_backtest_frame(
                    CacheStore(),
                    command.frame_request(pair),
                    strategy,
                    backtest_frame_options_from_command(command),
                )
            except DataCoverageError:
                continue
            signal_frame = (
                prepared.frame
                if prepared.precomputed_signal
                else compute_signal_frame(prepared.frame, strategy)
            )
            values = strategy_signal_diagnostics(signal_frame, strategy)
            label = f"{pair.asset.symbol} {strategy.name}"
            signal_diagnostics.append((label, values))

    output = format_backtest_report(
        mode=command.strategy.mode,
        strategy=strategy.name,
        reports=reports,
        detail=command.strategy.detail,
        diagnostics=command.strategy.diagnostics,
        signal_diagnostics=signal_diagnostics,
    )
    return output


def run_reports(
    pairs,
    strategy: StrategyBehavior,
    recovery_cfg: RecoveryPolicy,
    exit_cfg: ExitConfig,
    command: ResearchCommandConfig,
):
    reports = []
    store = CacheStore()
    options = backtest_frame_options_from_command(command)
    for pair in pairs:
        try:
            prepared = prepare_backtest_frame(
                store, command.frame_request(pair), strategy, options
            )
        except FileNotFoundError as exc:
            logger.warning("skip %s: %s", pair.asset.symbol, exc)
            continue
        except DataCoverageError as exc:
            logger.warning("data incomplete %s: %s", pair.asset.symbol, exc)
            reports.append(_data_incomplete_report(pair, strategy, exc, command))
            continue
        bt = BacktestExecutor(
            initial_capital=pair.asset.capital,
            cost_pct=0.00005,
            drawdown_stop_pct=None
            if command.exit.no_drawdown_stop
            else command.exit.drawdown_stop_pct,
            max_per_strategy_symbol=command.max_per_strategy_symbol,
            loss_cooldown_bars=command.exit.loss_cooldown_bars,
        )
        reports.append(
            bt.run_report(
                prepared.frame,
                pair,
                exit_cfg=exit_cfg,
                recovery_cfg=recovery_cfg,
                strategy=strategy,
                precomputed_signal=prepared.precomputed_signal,
                metadata=(
                    *prepared.metadata,
                    f"recovery_policy={command.strategy.mode}",
                    strategy_metadata(strategy),
                    *risk_gate_metadata(command.risk_gates),
                ),
            )
        )
    return reports


def _prepare_market_state_frame(
    frame: pl.DataFrame, *, liquidity_lookback: int = 20, include_mtf_keys: bool = True
) -> pl.DataFrame:
    work = frame
    if "liquidity_event_type" not in work.columns:
        work = add_liquidity_sweep_features(lookback=liquidity_lookback)(work)
    if (
        "atr_percentile_bucket" not in work.columns
        or "key_level_proximity_bucket" not in work.columns
    ):
        work = add_none_context_diagnostics()(work)
    if include_mtf_keys:
        work = add_mtf_state_keys(work)
    return add_market_state_reductions(work)


def run_research_evaluation(command: ResearchCommandConfig) -> str:
    outputs = resolve_research_outputs(command.research_evaluation.outputs)
    sections = ["Layered research evaluation", _research_graph_text(outputs)]
    summary_rows: list[list[str]] = []
    export_messages: list[str] = []
    reports: list[Report] = []

    timeframe_frames = (
        _prepare_timeframe_frames(command) if "timeframe-classifier" in outputs else {}
    )
    prepared_market = (
        _prepare_market_frames(command)
        if {"dynamic-transition-discovery", "pattern-quality"} & set(outputs)
        else []
    )

    if "timeframe-classifier" in outputs:
        frame, text = _timeframe_classifier_artifact(timeframe_frames)
        summary_rows.append(["timeframe-classifier", f"rows={frame.height}"])
        if text:
            sections.append("Timeframe classifier diagnostics\n" + text)
        export_messages.extend(_write_tables(command, {"timeframe-classifier.csv": frame}))

    if {"dynamic-transition-discovery", "pattern-quality"} & set(outputs):
        bundle = _dynamic_transition_bundle(command, prepared_market)
        scored = bundle.tables.get("scored-patterns.csv", pl.DataFrame())
        summary_rows.append(
            [
                "pattern-quality",
                f"rows={scored.height} candidates={_bool_count(scored, 'passes_candidate_gate')}",
            ]
        )
        sections.append("Pattern quality\n" + "\n".join(bundle.summary))
        export_messages.extend(_write_tables(command, _filter_bundle_tables(bundle, outputs)))

    if command.research_evaluation.include_backtest_report or "trade-record-modulation" in outputs:
        reports, text = _research_backtest_branch(command)
        if text:
            sections.append(text)
    if "trade-record-modulation" in outputs:
        frame = _trade_record_modulation_frame(command, reports)
        summary_rows.extend(_trade_modulation_summary_rows(frame))
        sections.append("Trade-record control\nrows=" + str(frame.height))
        export_messages.extend(_write_tables(command, {"trade-record-modulation.csv": frame}))

    if export_messages:
        sections.append("Research evaluation exports\n" + "\n".join(export_messages))
    summary = "Evidence graph summary\n" + format_table(["Layer", "Summary"], summary_rows)
    return summary + "\n\n" + "\n\n".join(section for section in sections if section.strip())


def _research_graph_text(outputs: tuple[ResearchOutputName, ...]) -> str:
    rows = pl.DataFrame(EVIDENCE_ROWS, schema=["output", "upstream_evidence"], orient="row")
    return (
        "Requested evidence graph\n"
        + format_table(["Output", "Upstream evidence"], _frame_table_rows(rows))
        + f"\nactive_outputs={','.join(outputs)}"
    )


def _prepare_market_frames(command: ResearchCommandConfig):
    store = CacheStore()
    classifier_config = command.classifier.to_structure_config()
    frames = []
    for pair in command.pairs():
        try:
            prepared = prepare_classifier_frame(
                store,
                command.frame_request(pair),
                classifier_config,
                contexts=DEFAULT_CONTEXTS,
            )
        except FileNotFoundError as exc:
            logger.warning("skip %s: %s", pair.asset.symbol, exc)
            continue
        except DataCoverageError as exc:
            logger.warning("data incomplete %s: %s", pair.asset.symbol, exc)
            continue
        market_frame = _prepare_market_state_frame(prepared.frame).with_columns(
            pl.lit(pair.asset.symbol).alias("symbol")
        )
        frames.append((pair.asset.symbol, prepared.frame, market_frame))
    return frames


def _prepare_timeframe_frames(
    command: ResearchCommandConfig,
) -> dict[tuple[str, str], pl.DataFrame]:
    store = CacheStore()
    frames = {}
    for pair in command.pairs():
        for spec in command.timeframes.resolved_specs(command):
            try:
                prepared = prepare_classifier_frame(
                    store,
                    command.frame_request(
                        pair,
                        bar=spec.bar,
                        days=spec.days,
                        min_bars=spec.min_bars,
                    ),
                    spec.classifier,
                    contexts=(),
                )
            except FileNotFoundError as exc:
                logger.warning("skip %s %s: %s", pair.asset.symbol, spec.bar, exc)
                continue
            except DataCoverageError as exc:
                logger.warning("data incomplete %s %s: %s", pair.asset.symbol, spec.bar, exc)
                continue
            frames[(pair.asset.symbol, spec.bar)] = _prepare_market_state_frame(
                prepared.frame,
                liquidity_lookback=spec.liquidity_lookback,
                include_mtf_keys=False,
            ).with_columns(
                pl.lit(pair.asset.symbol).alias("symbol"),
                pl.lit(spec.bar).alias("timeframe"),
            )
    return frames


def _timeframe_classifier_artifact(
    timeframe_frames: dict[tuple[str, str], pl.DataFrame],
) -> tuple[pl.DataFrame, str]:
    texts = []
    frames = []
    for (symbol, timeframe), frame in timeframe_frames.items():
        result = classifier_health(frame, label=f"{symbol} {timeframe}")
        texts.append(result.text)
        frames.append(
            result.frame.with_columns(
                pl.lit(symbol).alias("symbol"), pl.lit(timeframe).alias("timeframe")
            )
        )
    return _concat_frames(frames), "\n\n".join(texts)


def _dynamic_transition_bundle(command: ResearchCommandConfig, prepared_market) -> ArtifactBundle:
    config = command.research_evaluation.dynamic_transition_discovery
    thresholds = {
        "ngram_lengths": config.ngram_lengths,
        "none_context_columns": config.none_context_columns if config.include_none_context else (),
        "min_rows": config.min_rows,
        "omega_threshold": config.omega_threshold,
        "pwpr_threshold": config.pwpr_threshold,
        "promotion_min_rows": config.promotion_min_rows,
        "promotion_min_symbols": config.promotion_min_symbols,
        "promotion_min_time_splits": config.promotion_min_time_splits,
        "promotion_symbol_agreement_pct": config.promotion_symbol_agreement_pct,
        "promotion_time_agreement_pct": config.promotion_time_agreement_pct,
    }
    return build_transition_bundle(
        [market_frame for _symbol, _classifier_frame, market_frame in prepared_market],
        frame_specs=[
            {
                "symbol": symbol,
                "timeframe": "1H",
                "state_columns": config.state_columns,
                "event_column": config.event_column,
                "context_columns": config.none_context_columns,
            }
            for symbol, _classifier_frame, _market_frame in prepared_market
        ],
        horizons=command.market_state.horizons,
        thresholds=thresholds,
    )


def _filter_bundle_tables(
    bundle: ArtifactBundle, outputs: tuple[ResearchOutputName, ...]
) -> dict[str, pl.DataFrame]:
    tables = {}
    for name, frame in bundle.tables.items():
        if name.startswith("state-") and "dynamic-transition-discovery" not in outputs:
            continue
        if (
            name in {"scored-patterns.csv", "promotion-candidates.csv"}
            and "pattern-quality" not in outputs
        ):
            continue
        tables[name] = frame
    return tables


def _research_backtest_branch(command: ResearchCommandConfig) -> tuple[list[Report], str]:
    selection = strategy_selection_from_config(command)
    if len(selection.strategies) != 1 or command.strategy.style != "single":
        return [], (
            "Backtest branch\n"
            "skipped: research-evaluation backtest summary supports one single strategy"
            if command.research_evaluation.include_backtest_report
            else ""
        )
    strategy = selection.strategies[0]
    reports = run_reports(
        command.pairs(),
        strategy,
        mode_config(command.strategy.mode),
        exit_config_from_command(command),
        command,
    )
    if command.research_evaluation.fail_fast:
        _assert_reports_pass(reports, command)
    if not command.research_evaluation.include_backtest_report:
        return reports, ""
    return reports, format_backtest_report(
        mode=command.strategy.mode,
        strategy=strategy.name,
        reports=reports,
        detail=command.strategy.detail,
        diagnostics=command.strategy.diagnostics,
        signal_diagnostics=[],
    )


def _trade_record_modulation_frame(
    command: ResearchCommandConfig, reports: list[Report]
) -> pl.DataFrame:
    frames = [
        report.trades.with_columns(pl.lit(str(report.label).split(" ", 1)[0]).alias("symbol"))
        for report in reports
        if not report.trades.is_empty()
    ]
    if not frames:
        return pl.DataFrame(schema=CONTROL_SCHEMA)
    config = command.research_evaluation.modulation_effect
    return trade_record_control(
        pl.concat(frames, how="diagonal_relaxed"),
        min_base_trades=config.min_base_trades,
        min_cell_trades=config.min_cell_trades,
        practical_delta_threshold=config.practical_delta_threshold,
    ).frame


def _write_tables(command: ResearchCommandConfig, tables: dict[str, pl.DataFrame]) -> list[str]:
    export_root = command.diagnostics.export_dir or command.diagnostics.export
    if not command.research_evaluation.write_exports or not export_root:
        return []
    non_empty = {
        name: frame
        for name, frame in tables.items()
        if not frame.is_empty() or name == "promotion-candidates.csv"
    }
    written = write_bundle(ArtifactBundle("research-evaluation", non_empty), export_root)
    messages = []
    for path in written:
        name = path.replace("\\", "/").rsplit("/", 1)[-1]
        messages.append(f"{name}: {path}")
    return messages


def _trade_modulation_summary_rows(frame: pl.DataFrame) -> list[list[str]]:
    if frame.is_empty():
        return [["trade-record-modulation", "rows=0"]]
    return [
        [
            "trade-record-modulation",
            f"rows={frame.height} classification={_counts_text(frame, 'classification')} "
            f"significant={_bool_count(frame, 'significant')} "
            f"stable={_bool_count(frame, 'stable_across_symbols')} "
            f"below_base={_bool_count(frame, 'sufficient_base', value=False)} "
            f"below_cell={_bool_count(frame, 'sufficient_cell', value=False)}",
        ]
    ]


def _counts_text(frame: pl.DataFrame, column: str) -> str:
    if frame.is_empty() or column not in frame.columns:
        return "none"
    rows = (
        frame.group_by(column)
        .agg(pl.len().alias("count"))
        .sort(column)
        .select(
            pl.concat_str(
                [
                    pl.col(column).cast(pl.Utf8).fill_null("null"),
                    pl.lit("="),
                    pl.col("count").cast(pl.Utf8),
                ]
            ).alias("item")
        )
    )
    return ",".join(rows.get_column("item").to_list())


def _bool_count(frame: pl.DataFrame, column: str, *, value: bool = True) -> int:
    if frame.is_empty() or column not in frame.columns:
        return 0
    predicate = pl.col(column).fill_null(False)
    if not value:
        predicate = ~predicate
    return int(frame.select(predicate.cast(pl.Int64).sum()).item() or 0)


def _concat_frames(frames: list[pl.DataFrame]) -> pl.DataFrame:
    frames = [frame for frame in frames if not frame.is_empty()]
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def _frame_table_rows(frame: pl.DataFrame) -> list[list[str]]:
    if frame.is_empty():
        return []
    return [list(row) for row in frame.rows()]


def _trade_record_control_frame(
    trades: pl.DataFrame,
    *,
    min_base_trades: int,
    min_cell_trades: int,
    practical_delta_threshold: float,
) -> pl.DataFrame:
    if trades.is_empty():
        return pl.DataFrame(schema=CONTROL_SCHEMA)
    value_col = "net_pnl_usd" if "net_pnl_usd" in trades.columns else "pnl_usd"
    work = _normalize_trade_aliases(trades)
    bases = [column for column in ("entry_market_stage_bucket", "side") if column in work.columns]
    mods = [
        column
        for column in ("entry_d1_structure_trend_state", "entry_d1_market_stage")
        if column in work.columns
    ]
    frames = []
    for base in bases:
        base_stats = work.group_by(base).agg(
            pl.len().alias("base_trades"),
            pl.col(value_col).cast(pl.Float64).mean().alias("base_expectancy"),
        )
        for mod in mods:
            frame = (
                work.group_by(base, mod)
                .agg(
                    pl.len().alias("conditional_trades"),
                    pl.col(value_col).cast(pl.Float64).mean().alias("conditional_expectancy"),
                )
                .join(base_stats, on=base)
                .with_columns(
                    pl.lit("trade-record-modulation").alias("artifact"),
                    pl.lit(base).alias("base_feature"),
                    pl.col(base).cast(pl.Utf8).alias("base_value"),
                    pl.lit(mod).alias("modulator_feature"),
                    pl.col(mod).cast(pl.Utf8).alias("modulator_value"),
                    (pl.col("conditional_expectancy") - pl.col("base_expectancy")).alias(
                        "delta_expectancy"
                    ),
                )
                .with_columns(
                    (pl.col("base_trades") >= min_base_trades).alias("sufficient_base"),
                    (pl.col("conditional_trades") >= min_cell_trades).alias("sufficient_cell"),
                )
                .with_columns(
                    (
                        pl.col("sufficient_base")
                        & pl.col("sufficient_cell")
                        & (pl.col("delta_expectancy").abs() >= practical_delta_threshold)
                    ).alias("significant"),
                    pl.when(pl.col("sufficient_base") & pl.col("sufficient_cell"))
                    .then(pl.lit("control"))
                    .otherwise(pl.lit("insufficient"))
                    .alias("classification"),
                )
            )
            frames.append(_select_schema(frame, CONTROL_SCHEMA))
    return (
        pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame(schema=CONTROL_SCHEMA)
    )


def _normalize_trade_aliases(trades: pl.DataFrame) -> pl.DataFrame:
    work = trades
    if "entry_market_stage_bucket" not in work.columns and "entry_market_stage" in work.columns:
        work = work.with_columns(pl.col("entry_market_stage").alias("entry_market_stage_bucket"))
    if "side" in work.columns:
        work = work.with_columns(
            pl.when(pl.col("side").is_in(["buy", "long"]))
            .then(pl.lit("long"))
            .when(pl.col("side").is_in(["sell", "short"]))
            .then(pl.lit("short"))
            .otherwise(pl.col("side").cast(pl.Utf8))
            .alias("side")
        )
    return work


def _select_schema(frame: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if frame.is_empty() and not frame.columns:
        return pl.DataFrame(schema=schema)
    additions = [
        pl.lit(None).cast(dtype).alias(column)
        for column, dtype in schema.items()
        if column not in frame.columns
    ]
    work = frame.with_columns(additions) if additions else frame
    return work.select([pl.col(column).cast(dtype) for column, dtype in schema])


def _bool_sum(frame: pl.DataFrame, column: str) -> int:
    if frame.is_empty() or column not in frame.columns:
        return 0
    return int(frame.select(pl.col(column).fill_null(False).cast(pl.Int64).sum()).item() or 0)


def _data_incomplete_report(
    pair,
    strategy: StrategyBehavior,
    error: DataCoverageError,
    command: ResearchCommandConfig,
) -> Report:
    coverage = error.coverage
    return Report.from_raw(
        [],
        [pair.asset.capital, pair.asset.capital],
        pair,
        label=f"{pair.asset.symbol} {strategy.name}",
        diagnostics=BacktestDiagnostics(
            feature=FeatureDiagnostics(
                bars=coverage.target.target_bars,
                usable_bars=coverage.actual_bars,
                warmup_bars=max(0, coverage.target.target_bars - coverage.actual_bars),
            ),
            audit=EngineDataAudit(
                bars=coverage.target.target_bars,
                bars_processed=coverage.actual_bars,
                data_start=coverage.actual_start_ms,
                data_end=coverage.actual_end_ms,
            ),
            bars=coverage.target.target_bars,
            bars_processed=coverage.actual_bars,
            stopped_early=True,
            data_start=coverage.actual_start_ms,
            data_end=coverage.actual_end_ms,
        ),
        metadata=(
            *command.metadata(),
            "data_quality=data_incomplete",
            "data_incomplete_reason=listing_age"
            if "starts_after_target_since" in coverage.notes
            else "data_incomplete_reason=coverage_low",
            f"required_coverage_pct={error.required_pct:.1f}",
            *coverage_metadata(coverage),
            strategy_metadata(strategy),
        ),
    )


def _assert_reports_pass(reports, command: ResearchCommandConfig) -> None:
    gates = command.risk_gates
    if not gates.fail_on_risk:
        return
    failed = []
    for report in reports:
        metrics = report.metrics
        reasons = []
        if metrics.num_trades < gates.min_trades:
            reasons.append("SPARSE")
        if gates.min_pf > 0 and metrics.profit_factor < gates.min_pf:
            reasons.append("PF_LOW")
        if (
            gates.min_expectancy_pct is not None
            and report.trade_expectancy_pct <= gates.min_expectancy_pct
        ):
            reasons.append("EXP_LOW")
        if gates.max_dd_pct is not None and metrics.max_drawdown_pct > gates.max_dd_pct:
            reasons.append("DD_HIGH")
        diagnostics = report.diagnostics
        if (
            gates.max_notional_exposure_pct is not None
            and diagnostics is not None
            and diagnostics.max_notional_exposure_pct > gates.max_notional_exposure_pct
        ):
            reasons.append("NOTIONAL_HIGH")
        if reasons:
            failed.append(f"{report.label}:{','.join(reasons)}")
    if failed:
        raise SystemExit("risk gates failed: " + "; ".join(failed))


def run_style(
    pairs,
    strategy: StrategyBehavior,
    recovery_cfg: RecoveryPolicy,
    exit_cfg: ExitConfig,
    command: ResearchCommandConfig,
) -> str:
    lines = []
    store = CacheStore()
    options = backtest_frame_options_from_command(command)
    for pair in pairs:
        prepared = prepare_backtest_frame(
            store, command.frame_request(pair), strategy, options
        )

        def _run_window(seg: pl.DataFrame):
            bt = BacktestExecutor(
                initial_capital=pair.asset.capital,
                cost_pct=0.00005,
                drawdown_stop_pct=None
                if command.exit.no_drawdown_stop
                else command.exit.drawdown_stop_pct,
                max_per_strategy_symbol=command.max_per_strategy_symbol,
                loss_cooldown_bars=command.exit.loss_cooldown_bars,
            )
            return bt.run_result(
                seg,
                pair,
                exit_cfg=exit_cfg,
                recovery_cfg=recovery_cfg,
                strategy=strategy,
                precomputed_signal=prepared.precomputed_signal,
            )

        if command.strategy.style == "rolling":
            result = rolling_window(
                _run_window,
                prepared.frame,
                lookback_bars=command.strategy.train_bars,
                step_bars=command.strategy.step_bars,
            )
        elif command.strategy.style == "walk-forward":
            result = walk_forward(
                _run_window,
                prepared.frame,
                train_bars=command.strategy.train_bars,
                test_bars=command.strategy.test_bars,
                step_bars=command.strategy.step_bars,
            )
        else:
            result = cross_validate(
                _run_window,
                prepared.frame,
                folds=command.strategy.folds,
            )
        lines.append(result.summary())
    return "\n".join(lines)


def run_cache_audit(command: ResearchCommandConfig) -> str:
    request = CacheAuditRequest(
        pairs=command.pairs(),
        data_source=command.run.data_source,
        days=command.days,
        min_bars=command.min_bars,
        min_coverage_pct=command.min_coverage_pct,
        bars=command.timeframes.bars if command.diagnostics.mode == "research-evaluation" else (),
        refresh=command.cache.refresh,
        async_refresh=command.cache.async_refresh,
        refresh_concurrency=command.cache.refresh_concurrency,
        incremental=not command.cache.refresh_full,
    )
    refresh_requests = build_history_refresh_requests(request)
    if request.refresh and request.async_refresh:
        asyncio.run(_stream_cache_refresh(refresh_requests, request.refresh_concurrency))
    refresh_local = request.refresh and not request.async_refresh
    local_requests = [
        HistoryRequest(
            inst_id=item.inst_id,
            bar=item.bar,
            days=item.days,
            min_bars=item.min_bars,
            refresh=refresh_local,
            source=item.source,
        )
        for item in refresh_requests
    ]
    frame = CacheStore().audit_bars(local_requests, min_coverage_pct=request.min_coverage_pct)
    rows = _frame_table_rows(
        frame.select(
            pl.col("status").cast(pl.Utf8),
            pl.col("instrument").cast(pl.Utf8),
            pl.col("bar").cast(pl.Utf8),
            pl.col("actual_bars").cast(pl.Utf8),
            pl.col("target_bars").cast(pl.Utf8),
            pl.col("coverage_pct").fill_null(0.0).round(1).cast(pl.Utf8),
            pl.col("start_ms").cast(pl.Utf8).fill_null("n/a"),
            pl.col("end_ms").cast(pl.Utf8).fill_null("n/a"),
            pl.col("notes").cast(pl.Utf8),
        )
    )
    return "\n".join(
        [
            "Cache audit",
            format_table(
                [
                    "Status",
                    "Instrument",
                    "Bar",
                    "Bars",
                    "Target",
                    "Coverage%",
                    "Start",
                    "End",
                    "Notes",
                ],
                rows,
            ),
        ]
    )


async def _stream_cache_refresh(requests, concurrency: int) -> None:
    async with AsyncCacheStore() as store:
        async for event in store.stream_many(requests, concurrency=concurrency):
            if event.kind in {"completed", "failed", "summary"}:
                logger.info("%s", event.message)
