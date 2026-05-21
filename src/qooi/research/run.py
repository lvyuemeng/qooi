"""Research workflow execution helpers."""

from __future__ import annotations

import logging
from pathlib import Path

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
from qooi.exchange.store import CacheStore
from qooi.research.config import (
    ResearchCommandConfig,
    ResearchOutputName,
    risk_gate_metadata,
)
from qooi.research.diagnostics import (
    CONTROL_SCHEMA,
    add_forward_outcomes,
    add_market_state_reductions,
    classifier_health,
    joint_forward_quality,
    trade_record_control,
)
from qooi.research.workflows import (
    DEFAULT_CONTEXTS,
    BacktestFrameOptions,
    CacheAuditRequest,
    DataCoverageError,
    FrameRequest,
    add_mtf_state_keys,
    coverage_metadata,
    prepare_backtest_frame,
    prepare_classifier_frame,
    run_cache_audit_workflow,
)
from qooi.strategies import StrategyBehavior, compute_signal_frame, strategy_signal_diagnostics
from qooi.strategies.catalog import (
    strategy_metadata,
    strategy_selection,
)
from qooi.strategies.features import add_liquidity_sweep_features, add_none_context_diagnostics

logger = logging.getLogger(__name__)


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


def frame_request_from_command(
    pair,
    command: ResearchCommandConfig,
    *,
    bar: str | None = None,
    days: int | None = None,
    min_bars: int | None = None,
) -> FrameRequest:
    return FrameRequest(
        pair=pair,
        data_source=command.run.data_source,
        bar=bar or pair.asset.timeframe,
        days=days or command.days,
        min_bars=min_bars or command.min_bars,
        refresh=command.cache.refresh,
        min_coverage_pct=command.min_coverage_pct,
        allow_swap_signal_fallback=command.run.allow_swap_signal_fallback,
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
                    frame_request_from_command(pair, command),
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
                store, frame_request_from_command(pair, command), strategy, options
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
    return _ResearchEvaluationGraph(command).run()


def _resolve_research_outputs(
    requested: tuple[ResearchOutputName, ...],
) -> tuple[ResearchOutputName, ...]:
    return _ResearchEvaluationGraph.resolve_outputs(requested)


class _ResearchEvaluationGraph:
    OUTPUT_ORDER: tuple[ResearchOutputName, ...] = (
        "timeframe-classifier",
        "joint-forward-quality",
        "trade-record-modulation",
    )
    MARKET_OUTPUTS = {
        "timeframe-classifier",
        "joint-forward-quality",
    }
    EVIDENCE_ROWS = (
        ["timeframe-classifier", "peer timeframe cache -> independent classifier health"],
        ["joint-forward-quality", "classifier frame -> side-normalized joint buckets"],
        ["trade-record-modulation", "strategy signal/backtest branch -> trades"],
    )

    def __init__(self, command: ResearchCommandConfig) -> None:
        self.command = command
        self.outputs = self.resolve_outputs(command.research_evaluation.outputs)
        self.classifier_config = command.classifier.to_structure_config()
        export_root = command.diagnostics.export_dir or command.diagnostics.export
        self.export_dir = Path(export_root or ".")
        self.should_write = command.research_evaluation.write_exports and bool(export_root)
        self.sections = ["Layered research evaluation", self.graph_text()]
        self.summary_rows: list[list[str]] = []
        self.export_messages: list[str] = []
        self.prepared_market = []
        self.timeframe_specs = command.timeframes.resolved_specs(command)
        self.timeframe_frames: dict[tuple[str, str], pl.DataFrame] = {}
        self.timeframe_classifier_exports: list[pl.DataFrame] = []
        self.reports: list[Report] = []

    @classmethod
    def resolve_outputs(
        cls, requested: tuple[ResearchOutputName, ...]
    ) -> tuple[ResearchOutputName, ...]:
        outputs = set(requested)
        return tuple(output for output in cls.OUTPUT_ORDER if output in outputs)

    def run(self) -> str:
        if "timeframe-classifier" in self.outputs:
            self.prepare_timeframe_frames()
        if "joint-forward-quality" in self.outputs:
            self.prepared_market = self.prepare_market_frames()
        if "timeframe-classifier" in self.outputs:
            self.add_timeframe_classifier()
        if "joint-forward-quality" in self.outputs:
            self.add_joint_forward_quality()
        if (
            self.command.research_evaluation.include_backtest_report
            or "trade-record-modulation" in self.outputs
        ):
            self.add_backtest_branch()
        if "trade-record-modulation" in self.outputs:
            self.add_trade_record_modulation()
        if self.export_messages:
            self.sections.append("Research evaluation exports\n" + "\n".join(self.export_messages))
        summary = "Evidence graph summary\n" + format_table(["Layer", "Summary"], self.summary_rows)
        return (
            summary + "\n\n" + "\n\n".join(section for section in self.sections if section.strip())
        )

    def graph_text(self) -> str:
        rows = pl.DataFrame(
            self.EVIDENCE_ROWS, schema=["output", "upstream_evidence"], orient="row"
        )
        return (
            "Requested evidence graph\n"
            + format_table(["Output", "Upstream evidence"], _frame_table_rows(rows))
            + f"\nactive_outputs={','.join(self.outputs)}"
        )

    def prepare_market_frames(self):
        store = CacheStore()
        frames = []
        for pair in self.command.pairs():
            try:
                prepared = prepare_classifier_frame(
                    store,
                    frame_request_from_command(pair, self.command),
                    self.classifier_config,
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

    def prepare_timeframe_frames(self) -> None:
        store = CacheStore()
        for pair in self.command.pairs():
            for spec in self.timeframe_specs:
                try:
                    prepared = prepare_classifier_frame(
                        store,
                        frame_request_from_command(
                            pair,
                            self.command,
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
                market_frame = _prepare_market_state_frame(
                    prepared.frame,
                    liquidity_lookback=spec.liquidity_lookback,
                    include_mtf_keys=False,
                ).with_columns(
                    pl.lit(pair.asset.symbol).alias("symbol"),
                    pl.lit(spec.bar).alias("timeframe"),
                )
                self.timeframe_frames[(pair.asset.symbol, spec.bar)] = market_frame

    def add_timeframe_classifier(self) -> None:
        texts = []
        for (symbol, timeframe), frame in self.timeframe_frames.items():
            result = classifier_health(frame, label=f"{symbol} {timeframe}")
            texts.append(result.text)
            self.timeframe_classifier_exports.append(
                result.frame.with_columns(
                    pl.lit(symbol).alias("symbol"),
                    pl.lit(timeframe).alias("timeframe"),
                )
            )
        export_frame = self.concat_frames(self.timeframe_classifier_exports)
        self.summary_rows.extend([["timeframe-classifier", f"rows={export_frame.height}"]])
        if texts:
            self.sections.append("Timeframe classifier diagnostics\n" + "\n\n".join(texts))
        self.write_export("timeframe-classifier.csv", export_frame)

    def add_joint_forward_quality(self) -> None:
        frames = []
        market = self.command.market_state
        config = self.command.research_evaluation.joint_forward_quality
        for symbol, _classifier_frame, market_frame in self.prepared_market:
            forward = add_forward_outcomes(market_frame, symbol=symbol, horizons=market.horizons)
            result = joint_forward_quality(
                forward,
                horizons=market.horizons,
                min_rows=config.min_rows,
                omega_threshold=config.omega_threshold,
                pwpr_threshold=config.pwpr_threshold,
                transition_min_rows=config.transition_min_rows,
                prior_strength=config.prior_strength,
                invalid_values=config.invalid_values,
            )
            if not result.frame.is_empty():
                frames.append(result.frame)
        if frames:
            all_forward = pl.concat(
                [
                    add_forward_outcomes(frame, symbol=symbol, horizons=market.horizons)
                    for symbol, _classifier_frame, frame in self.prepared_market
                ],
                how="diagonal_relaxed",
            )
            aggregate = joint_forward_quality(
                all_forward,
                horizons=market.horizons,
                min_rows=config.min_rows,
                omega_threshold=config.omega_threshold,
                pwpr_threshold=config.pwpr_threshold,
                transition_min_rows=config.transition_min_rows,
                prior_strength=config.prior_strength,
                invalid_values=config.invalid_values,
            )
            if not aggregate.frame.is_empty():
                frames.append(aggregate.frame)
        export_frame = self.concat_frames(frames)
        self.summary_rows.extend(self.joint_forward_summary_rows(export_frame))
        self.sections.append(self.joint_forward_text(export_frame))
        self.write_export("joint-forward-quality.csv", export_frame)

    def add_backtest_branch(self) -> None:
        selection = strategy_selection_from_config(self.command)
        include_report = self.command.research_evaluation.include_backtest_report
        if len(selection.strategies) != 1 or self.command.strategy.style != "single":
            if include_report:
                self.sections.append(
                    "Backtest branch\n"
                    "skipped: research-evaluation backtest summary supports one single strategy"
                )
            return
        strategy = selection.strategies[0]
        self.reports = run_reports(
            self.command.pairs(),
            strategy,
            mode_config(self.command.strategy.mode),
            exit_config_from_command(self.command),
            self.command,
        )
        if self.command.research_evaluation.fail_fast:
            _assert_reports_pass(self.reports, self.command)
        if include_report:
            self.sections.append(
                format_backtest_report(
                    mode=self.command.strategy.mode,
                    strategy=strategy.name,
                    reports=self.reports,
                    detail=self.command.strategy.detail,
                    diagnostics=self.command.strategy.diagnostics,
                    signal_diagnostics=[],
                )
            )

    def add_trade_record_modulation(self) -> None:
        modulation_frame = self.trade_record_modulation_frame()
        self.summary_rows.extend(self.trade_modulation_summary_rows(modulation_frame))
        config = self.command.research_evaluation.modulation_effect
        result = trade_record_control(
            pl.DataFrame(),
            min_base_trades=config.min_base_trades,
            min_cell_trades=config.min_cell_trades,
            practical_delta_threshold=config.practical_delta_threshold,
        )
        self.sections.append(
            result.text
            if modulation_frame.is_empty()
            else "Trade-record control\nrows=" + str(modulation_frame.height)
        )
        self.write_export("trade-record-modulation.csv", modulation_frame)

    def trade_record_modulation_frame(self) -> pl.DataFrame:
        frames = []
        for report in self.reports:
            if report.trades.is_empty():
                continue
            symbol = str(report.label).split(" ", 1)[0]
            frames.append(report.trades.with_columns(pl.lit(symbol).alias("symbol")))
        if not frames:
            return pl.DataFrame(schema=CONTROL_SCHEMA)
        config = self.command.research_evaluation.modulation_effect
        return trade_record_control(
            pl.concat(frames, how="diagonal_relaxed"),
            min_base_trades=config.min_base_trades,
            min_cell_trades=config.min_cell_trades,
            practical_delta_threshold=config.practical_delta_threshold,
        ).frame

    def write_export(self, name: str, frame: pl.DataFrame) -> None:
        if not self.should_write or frame.is_empty():
            return
        self.export_dir.mkdir(parents=True, exist_ok=True)
        path = self.export_dir / name
        frame.write_csv(path)
        self.export_messages.append(f"{name}: {path}")

    def trade_modulation_summary_rows(self, frame: pl.DataFrame) -> list[list[str]]:
        if frame.is_empty():
            return [["trade-record-modulation", "rows=0"]]
        return [
            [
                "trade-record-modulation",
                f"rows={frame.height} classification={self.counts_text(frame, 'classification')} "
                f"significant={self.bool_count(frame, 'significant')} "
                f"stable={self.bool_count(frame, 'stable_across_symbols')} "
                f"below_base={self.bool_count(frame, 'sufficient_base', value=False)} "
                f"below_cell={self.bool_count(frame, 'sufficient_cell', value=False)}",
            ]
        ]

    def joint_forward_summary_rows(self, frame: pl.DataFrame) -> list[list[str]]:
        if frame.is_empty():
            return [["joint-forward-quality", "rows=0"]]
        candidates = self.count_if(frame, pl.col("passes_candidate_gate"))
        transitions = self.artifact_filter(frame, "transition-event-quality")
        intrinsic = self.artifact_filter(frame, "configuration-intrinsic-quality")
        return [
            [
                "joint-forward-quality",
                f"rows={frame.height} candidates={candidates} "
                f"artifacts={self.counts_text(frame, 'artifact')} "
                f"configs={self.counts_text(intrinsic, 'intrinsic_quality_bucket')} "
                f"transition_rows={transitions.height}",
            ]
        ]

    def joint_forward_text(self, frame: pl.DataFrame) -> str:
        if frame.is_empty():
            return "Joint forward quality\nno joint-forward-quality rows"
        top = (
            frame.filter(
                (pl.col("artifact").is_in(["joint-forward-quality", "transition-event-quality"]))
                & pl.col("passes_candidate_gate").fill_null(False)
            )
            .sort("omega_ratio", descending=True)
            .head(8)
        )
        rows = [
            ["Artifact counts", self.counts_text(frame, "artifact")],
            ["Configuration quality", self.counts_text(frame, "intrinsic_quality_bucket")],
            ["Candidate gates", f"passed={self.bool_count(frame, 'passes_candidate_gate')}"],
        ]
        if top.is_empty():
            rows.append(["Top candidates", "none"])
        else:
            rows.append(
                [
                    "Top candidates",
                    "; ".join(
                        f"h{row['horizon']} {row['configuration_name']} "
                        f"{row['joint_group']} {row['liquidity_event_type']} {row['side']} "
                        f"n={row['rows']} omega={float(row['omega_ratio'] or 0.0):.2f}"
                        for row in top.iter_rows(named=True)
                    ),
                ]
            )
        return "Joint forward quality\n" + format_table(["Diagnostic", "Summary"], rows)

    def artifact_filter(
        self, frame: pl.DataFrame, artifact: str, *, exclude: bool = False
    ) -> pl.DataFrame:
        if frame.is_empty() or "artifact" not in frame.columns:
            return frame
        predicate = pl.col("artifact") != artifact if exclude else pl.col("artifact") == artifact
        return frame.filter(predicate)

    def counts_text(self, frame: pl.DataFrame, column: str) -> str:
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

    def bool_count(self, frame: pl.DataFrame, column: str, *, value: bool = True) -> int:
        if frame.is_empty() or column not in frame.columns:
            return 0
        predicate = pl.col(column).fill_null(False)
        if not value:
            predicate = ~predicate
        return int(frame.select(predicate.cast(pl.Int64).sum()).item() or 0)

    def count_if(self, frame: pl.DataFrame, predicate: pl.Expr) -> int:
        if frame.is_empty():
            return 0
        return int(frame.select(predicate.fill_null(False).cast(pl.Int64).sum()).item() or 0)

    @staticmethod
    def concat_frames(frames: list[pl.DataFrame]) -> pl.DataFrame:
        frames = [frame for frame in frames if not frame.is_empty()]
        return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def _frame_table_rows(frame: pl.DataFrame) -> list[list[str]]:
    if frame.is_empty():
        return []
    return [list(row) for row in frame.rows()]


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
            store, frame_request_from_command(pair, command), strategy, options
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
    result = run_cache_audit_workflow(
        CacheAuditRequest(
            pairs=command.pairs(),
            data_source=command.run.data_source,
            days=command.days,
            min_bars=command.min_bars,
            min_coverage_pct=command.min_coverage_pct,
            bars=command.timeframes.bars
            if command.diagnostics.mode == "research-evaluation"
            else (),
            refresh=command.cache.refresh,
            async_refresh=command.cache.async_refresh,
            refresh_concurrency=command.cache.refresh_concurrency,
            incremental=not command.cache.refresh_full,
        )
    )
    rows = _frame_table_rows(
        result.frame.select(
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
