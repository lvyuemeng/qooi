"""Research workflow execution helpers."""

from __future__ import annotations

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
from qooi.core.plot import (
    plot_market_state_horizon_decay,
    plot_market_state_modulation_heatmap,
)
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
    MARKET_STATE_BASE_COLUMNS,
    MARKET_STATE_MODULATOR_COLUMNS,
    MODULATION_BASE_COLUMNS,
    MODULATION_EFFECT_SCHEMA,
    MODULATION_MODULATOR_COLUMNS,
    MarketStateAnalysisResult,
    ModulationRobustnessConfig,
    TradabilityConfig,
    TradabilityResult,
    add_market_state_reductions,
    assert_reports_pass,
    classifier_validity_frame,
    evaluate_classifier_frame,
    format_candidate_status_table,
    format_cross_run_consistency,
    format_layer_summary,
    format_modulation_effect_summary,
    format_state_diagnostics_summary,
    format_state_filter_delta,
    format_state_profitability_summary,
    format_status_table,
    format_strategy_development_summary,
    market_state_forward_frame,
    market_state_forward_summary,
    market_state_modulation_matrix,
    modulation_effect_matrix,
    state_diagnostics_export_frame,
    state_profitability_export_frame,
    state_tradability_frame,
)
from qooi.research.workflows import (
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
) -> FrameRequest:
    return FrameRequest(
        pair=pair,
        data_source=command.run.data_source,
        days=command.days,
        min_bars=command.min_bars,
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
    diagnostic_mode = command.diagnostics.mode
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
        assert_reports_pass(all_reports, command.risk_gates)
        output = format_benchmark_report(
            mode=command.strategy.mode,
            benchmark_results=benchmark_results,
            diagnostics=command.strategy.diagnostics,
        )
        return output + _status_suffix(all_reports, command)

    strategy = selection.strategies[0]
    if command.strategy.style != "single":
        return run_style(pairs, strategy, recovery_cfg, exit_cfg, command)

    reports = run_reports(pairs, strategy, recovery_cfg, exit_cfg, command)
    assert_reports_pass(reports, command.risk_gates)
    signal_diagnostics = []
    layer_summaries = []
    development_summaries = []
    state_summaries = []
    state_profitability_summaries = []
    state_export_frames = []
    modulation_frame = pl.DataFrame(schema=MODULATION_EFFECT_SCHEMA)
    wants_diagnostics = command.strategy.diagnostics
    wants_explain_layers = command.strategy.explain_layers
    wants_state_diagnostics = diagnostic_mode == "state"
    wants_state_profitability = diagnostic_mode == "state-profitability"
    wants_modulation_effect = diagnostic_mode == "modulation-effect"
    state_export_path = command.diagnostics.export
    if (
        wants_diagnostics
        or wants_explain_layers
        or wants_state_diagnostics
        or wants_state_profitability
        or (state_export_path and not wants_modulation_effect)
    ):
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
            if wants_diagnostics or wants_explain_layers:
                signal_diagnostics.append((label, values))
            if wants_explain_layers:
                layer_summaries.append(
                    format_layer_summary(report.label, report, values, signal_frame)
                )
            if wants_diagnostics:
                development_summaries.append(
                    format_strategy_development_summary(
                        report.label,
                        report,
                        signal_frame,
                        strategy,
                    )
                )
            if wants_state_diagnostics:
                state_summaries.append(
                    format_state_diagnostics_summary(report.label, report, signal_frame)
                )
            if wants_state_profitability:
                state_profitability_summaries.append(
                    format_state_profitability_summary(report.label, report, signal_frame)
                )
            if state_export_path:
                if wants_state_profitability:
                    state_export_frames.append(
                        state_profitability_export_frame(report.label, report, signal_frame)
                    )
                else:
                    state_export_frames.append(
                        state_diagnostics_export_frame(report.label, report, signal_frame)
                    )

    if wants_modulation_effect:
        modulation_trades = []
        for pair, report in zip(pairs, reports, strict=False):
            if report.trades.is_empty():
                continue
            modulation_trades.append(
                report.trades.with_columns(pl.lit(pair.asset.symbol).alias("symbol"))
            )
        if modulation_trades:
            modulation_config = command.research_evaluation.modulation_effect
            modulation_frame = modulation_effect_matrix(
                pl.concat(modulation_trades, how="diagonal_relaxed"),
                base_columns=modulation_config.base_columns or MODULATION_BASE_COLUMNS,
                modulator_columns=modulation_config.modulator_columns
                or MODULATION_MODULATOR_COLUMNS,
                min_base_trades=modulation_config.min_base_trades,
                min_cell_trades=modulation_config.min_cell_trades,
                practical_delta_threshold=modulation_config.practical_delta_threshold,
            )

    output = format_backtest_report(
        mode=command.strategy.mode,
        strategy=strategy.name,
        reports=reports,
        detail=command.strategy.detail,
        diagnostics=command.strategy.diagnostics,
        signal_diagnostics=signal_diagnostics,
    )
    output += _status_suffix(reports, command)
    if layer_summaries:
        output += "\n\nLayer behavior summary\n" + "\n\n".join(layer_summaries)
    if development_summaries:
        output += "\n\nStrategy development diagnosis\n" + "\n\n".join(development_summaries)
    if state_summaries:
        output += "\n\nState diagnostics\n" + "\n\n".join(state_summaries)
    if state_profitability_summaries:
        output += "\n\nState profitability diagnostics\n" + "\n\n".join(
            state_profitability_summaries
        )
    if wants_modulation_effect:
        output += "\n\n" + format_modulation_effect_summary(modulation_frame)
    if state_export_path and wants_modulation_effect:
        path = Path(state_export_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        modulation_frame.write_csv(path)
        output += f"\n\nModulation effect export written: {path}"
    elif state_export_path and state_export_frames:
        path = Path(state_export_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pl.concat(state_export_frames, how="diagonal_relaxed").write_csv(path)
        export_label = "State profitability" if wants_state_profitability else "State diagnostics"
        output += f"\n\n{export_label} export written: {path}"
    return output


def run_state_filter_delta(command: ResearchCommandConfig) -> str:
    baseline_path = Path(command.diagnostics.baseline_export)
    variant_path = Path(command.diagnostics.variant_export)
    if not str(baseline_path) or not str(variant_path):
        raise ValueError(
            "state-filter-delta requires --baseline-diagnostics-export and "
            "--variant-diagnostics-export"
        )
    return format_state_filter_delta(
        pl.read_csv(baseline_path),
        pl.read_csv(variant_path),
        baseline_label=baseline_path.stem,
        variant_label=variant_path.stem,
    )


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
            print(f"skip {pair.asset.symbol}: {exc}")
            continue
        except DataCoverageError as exc:
            print(f"data incomplete {pair.asset.symbol}: {exc}")
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


def run_classifier_diagnostics(
    command: ResearchCommandConfig,
) -> str:
    pairs = command.pairs()
    store = CacheStore()
    classifier_config = command.classifier.to_structure_config()
    summaries = []
    export_frames = []
    export_path = command.diagnostics.export
    for pair in pairs:
        try:
            prepared = prepare_classifier_frame(
                store, frame_request_from_command(pair, command), classifier_config
            )
        except FileNotFoundError as exc:
            print(f"skip {pair.asset.symbol}: {exc}")
            continue
        except DataCoverageError as exc:
            print(f"data incomplete {pair.asset.symbol}: {exc}")
            continue
        diagnostics = evaluate_classifier_frame(pair.asset.symbol, prepared.frame)
        summaries.append(diagnostics.to_text())
        if export_path:
            export_frames.append(diagnostics.to_export_frame())
    output = "Classifier diagnostics"
    output += "\n" + command.classifier.summary(classifier_config)
    if summaries:
        output += "\n\n" + "\n\n".join(summaries)
    if export_path and export_frames:
        path = Path(export_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pl.concat(export_frames, how="diagonal_relaxed").write_csv(path)
        output += f"\n\nClassifier diagnostics export written: {path}"
    return output


def run_market_state_forward(
    command: ResearchCommandConfig,
) -> str:
    pairs = command.pairs()
    store = CacheStore()
    classifier_config = command.classifier.to_structure_config()
    market = command.market_state
    horizons = market.horizons
    outcomes = market.outcomes
    min_rows = market.min_rows
    min_base_rows = market.min_base_rows
    min_cell_rows = market.min_cell_rows
    delta_threshold, delta_mode_summary = market.delta_threshold()
    base_columns = market.base_columns or MARKET_STATE_BASE_COLUMNS
    modulator_columns = market.modulator_columns or MARKET_STATE_MODULATOR_COLUMNS
    time_splits = market.time_splits
    min_segment_base_rows = market.min_segment_base_rows
    min_segment_cell_rows = market.min_segment_cell_rows
    robustness = ModulationRobustnessConfig(
        se_method=market.robustness.se_method,
        fdr=market.robustness.fdr,
        fdr_alpha=market.robustness.fdr_alpha,
        cohens_d_threshold=market.robustness.cohens_d_threshold,
        n_eff_min=market.robustness.n_eff_min,
    )
    export_path = command.diagnostics.export
    plot_dir = market.plot_dir
    forward_frames = []
    summary_frames = []
    modulation_frames = []
    group_specs = (
        ("market_stage_reduced",),
        ("liquidity_event_type",),
        ("structure_trend_state",),
        ("atr_percentile_bucket",),
        ("market_stage_reduced", "liquidity_event_type"),
        ("structure_trend_state", "liquidity_event_type"),
        ("mtf_stage_key",),
        ("mtf_structure_key",),
        ("mtf_event_state_key",),
    )
    for pair in pairs:
        try:
            prepared = prepare_classifier_frame(
                store, frame_request_from_command(pair, command), classifier_config
            )
        except FileNotFoundError as exc:
            print(f"skip {pair.asset.symbol}: {exc}")
            continue
        except DataCoverageError as exc:
            print(f"data incomplete {pair.asset.symbol}: {exc}")
            continue
        market_frame = _prepare_market_state_frame(prepared.frame)
        forward = market_state_forward_frame(
            market_frame,
            symbol=pair.asset.symbol,
            horizons=horizons,
        )
        forward_frames.append(forward)
        for horizon in horizons:
            for group_columns in group_specs:
                summary = market_state_forward_summary(
                    forward,
                    group_columns=group_columns,
                    horizon=horizon,
                    min_rows=min_rows,
                )
                if not summary.is_empty():
                    summary_frames.append(summary)
    if forward_frames:
        all_forward = pl.concat(forward_frames, how="diagonal_relaxed")
        for horizon in horizons:
            for group_columns in group_specs:
                summary = market_state_forward_summary(
                    all_forward,
                    group_columns=group_columns,
                    horizon=horizon,
                    min_rows=min_rows,
                )
                if not summary.is_empty():
                    summary_frames.append(summary.with_columns(pl.lit("ALL").alias("symbol")))
            for outcome in outcomes:
                modulation = market_state_modulation_matrix(
                    all_forward,
                    outcome_column=_market_state_outcome_column(horizon, outcome),
                    min_base_rows=min_base_rows,
                    min_cell_rows=min_cell_rows,
                    practical_delta_threshold=delta_threshold,
                    base_columns=base_columns,
                    modulator_columns=modulator_columns,
                    time_splits=time_splits,
                    min_segment_base_rows=min_segment_base_rows,
                    min_segment_cell_rows=min_segment_cell_rows,
                    robustness=robustness,
                )
                if not modulation.is_empty():
                    modulation_frames.append(modulation)
    result = MarketStateAnalysisResult(tuple(summary_frames), tuple(modulation_frames))
    export_frame = result.export_frame()
    output = result.to_text()
    output += (
        f"\n\nThresholds: horizons={','.join(str(h) for h in horizons)} "
        f"outcomes={','.join(outcomes)} "
        f"min_rows={min_rows} min_base_rows={min_base_rows} "
        f"min_cell_rows={min_cell_rows} delta_threshold_pct={delta_threshold:.2f} "
        f"delta_mode={delta_mode_summary} time_splits={time_splits} "
        f"se_method={robustness.se_method} fdr={robustness.fdr} "
        f"fdr_alpha={robustness.fdr_alpha:.2f} cohens_d_threshold="
        f"{robustness.cohens_d_threshold:.2f} "
        f"base_columns={','.join(base_columns)} modulator_columns={','.join(modulator_columns)}"
    )
    output += "\n" + command.classifier.summary(classifier_config)
    if plot_dir and modulation_frames:
        plot_path = Path(plot_dir)
        plot_path.mkdir(parents=True, exist_ok=True)
        modulation_frame = pl.concat(modulation_frames, how="diagonal_relaxed")
        heatmap = plot_market_state_modulation_heatmap(
            modulation_frame,
            output_path=plot_path / "market-state-modulation-heatmap.svg",
        )
        decay = plot_market_state_horizon_decay(
            modulation_frame,
            output_path=plot_path / "market-state-horizon-decay.svg",
        )
        output += f"\n\nMarket-state plots written: {heatmap}, {decay}"
    if export_path and not export_frame.is_empty():
        path = Path(export_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        export_frame.write_csv(path)
        output += f"\n\nMarket-state forward export written: {path}"
    return output


def _prepare_market_state_frame(frame: pl.DataFrame) -> pl.DataFrame:
    work = frame
    if "liquidity_event_type" not in work.columns:
        work = add_liquidity_sweep_features()(work)
    if (
        "atr_percentile_bucket" not in work.columns
        or "key_level_proximity_bucket" not in work.columns
    ):
        work = add_none_context_diagnostics()(work)
    return add_market_state_reductions(add_mtf_state_keys(work))


def run_tradability_diagnostics(
    command: ResearchCommandConfig,
) -> str:
    pairs = command.pairs()
    store = CacheStore()
    classifier_config = command.classifier.to_structure_config()
    frames = []
    for pair in pairs:
        try:
            prepared = prepare_classifier_frame(
                store, frame_request_from_command(pair, command), classifier_config
            )
        except FileNotFoundError as exc:
            print(f"skip {pair.asset.symbol}: {exc}")
            continue
        except DataCoverageError as exc:
            print(f"data incomplete {pair.asset.symbol}: {exc}")
            continue
        market_frame = _prepare_market_state_frame(prepared.frame).with_columns(
            pl.lit(pair.asset.symbol).alias("symbol")
        )
        tradability = state_tradability_frame(
            market_frame,
            TradabilityConfig(min_rows=command.market_state.min_rows),
        )
        validity = classifier_validity_frame(market_frame)
        frames.extend(frame for frame in (tradability, validity) if not frame.is_empty())
    result = TradabilityResult(tuple(frames))
    export_frame = result.export_frame()
    output = result.to_text()
    output += "\n" + command.classifier.summary(classifier_config)
    export_path = command.diagnostics.export
    if export_path and not export_frame.is_empty():
        path = Path(export_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        export_frame.write_csv(path)
        output += f"\n\nTradability export written: {path}"
    return output


def run_research_evaluation(command: ResearchCommandConfig) -> str:
    return _ResearchEvaluationGraph(command).run()


def _resolve_research_outputs(
    requested: tuple[ResearchOutputName, ...]
) -> tuple[ResearchOutputName, ...]:
    return _ResearchEvaluationGraph.resolve_outputs(requested)


class _ResearchEvaluationGraph:
    OUTPUT_ORDER: tuple[ResearchOutputName, ...] = (
        "classifier",
        "tradability",
        "market-state-forward",
        "market-state-modulation",
        "trade-record-modulation",
    )
    MARKET_OUTPUTS = {
        "classifier",
        "tradability",
        "market-state-forward",
        "market-state-modulation",
    }
    EVIDENCE_ROWS = (
        ["classifier", "cache history -> prepared classifier frame"],
        ["tradability", "classifier frame -> market-state reductions"],
        ["market-state-forward", "classifier frame -> forward market outcomes"],
        ["market-state-modulation", "market-state-forward outcomes"],
        ["trade-record-modulation", "strategy signal/backtest branch -> trades"],
    )
    MARKET_GROUP_SPECS = (
        ("market_stage_reduced",),
        ("liquidity_event_type",),
        ("structure_trend_state",),
        ("atr_percentile_bucket",),
        ("market_stage_reduced", "liquidity_event_type"),
        ("structure_trend_state", "liquidity_event_type"),
        ("mtf_stage_key",),
        ("mtf_structure_key",),
        ("mtf_event_state_key",),
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
        self.reports: list[Report] = []

    @classmethod
    def resolve_outputs(
        cls, requested: tuple[ResearchOutputName, ...]
    ) -> tuple[ResearchOutputName, ...]:
        outputs = set(requested)
        if "market-state-modulation" in outputs:
            outputs.add("market-state-forward")
        if outputs & {"tradability", "market-state-forward", "market-state-modulation"}:
            outputs.add("classifier")
        return tuple(output for output in cls.OUTPUT_ORDER if output in outputs)

    def run(self) -> str:
        if any(output in self.MARKET_OUTPUTS for output in self.outputs):
            self.prepared_market = self.prepare_market_frames()
        if "classifier" in self.outputs:
            self.add_classifier()
        if "tradability" in self.outputs:
            self.add_tradability()
        if "market-state-forward" in self.outputs or "market-state-modulation" in self.outputs:
            self.add_market_state()
        if (
            self.command.research_evaluation.include_backtest_report
            or "trade-record-modulation" in self.outputs
        ):
            self.add_backtest_branch()
        if "trade-record-modulation" in self.outputs:
            self.add_trade_record_modulation()
        if self.export_messages:
            self.sections.append("Research evaluation exports\n" + "\n".join(self.export_messages))
        summary = "Evidence graph summary\n" + format_table(
            ["Layer", "Summary"], self.summary_rows
        )
        return summary + "\n\n" + "\n\n".join(
            section for section in self.sections if section.strip()
        )

    def graph_text(self) -> str:
        rows = pl.DataFrame(
            self.EVIDENCE_ROWS, schema=["output", "upstream_evidence"], orient="row"
        )
        return "Requested evidence graph\n" + format_table(
            ["Output", "Upstream evidence"], _frame_table_rows(rows)
        ) + f"\nactive_outputs={','.join(self.outputs)}"

    def prepare_market_frames(self):
        store = CacheStore()
        frames = []
        for pair in self.command.pairs():
            try:
                prepared = prepare_classifier_frame(
                    store,
                    frame_request_from_command(pair, self.command),
                    self.classifier_config,
                )
            except FileNotFoundError as exc:
                print(f"skip {pair.asset.symbol}: {exc}")
                continue
            except DataCoverageError as exc:
                print(f"data incomplete {pair.asset.symbol}: {exc}")
                continue
            market_frame = _prepare_market_state_frame(prepared.frame).with_columns(
                pl.lit(pair.asset.symbol).alias("symbol")
            )
            frames.append((pair.asset.symbol, prepared.frame, market_frame))
        return frames

    def add_classifier(self) -> None:
        classifier_frames = []
        classifier_texts = []
        for symbol, classifier_frame, _market_frame in self.prepared_market:
            diagnostics = evaluate_classifier_frame(symbol, classifier_frame)
            classifier_texts.append(diagnostics.to_text())
            classifier_frames.append(diagnostics.to_export_frame())
        export_frame = self.concat_frames(classifier_frames)
        self.summary_rows.extend(self.classifier_summary_rows(export_frame))
        self.sections.append("Classifier diagnostics\n" + "\n\n".join(classifier_texts))
        self.write_export("classifier.csv", export_frame)

    def add_tradability(self) -> None:
        frames = []
        for _symbol, _classifier_frame, market_frame in self.prepared_market:
            tradability = state_tradability_frame(
                market_frame,
                TradabilityConfig(min_rows=self.command.market_state.min_rows),
            )
            validity = classifier_validity_frame(market_frame)
            frames.extend(frame for frame in (tradability, validity) if not frame.is_empty())
        result = TradabilityResult(tuple(frames))
        export_frame = result.export_frame()
        self.summary_rows.extend(self.tradability_summary_rows(export_frame))
        self.sections.append(
            result.to_text() + "\n" + self.command.classifier.summary(self.classifier_config)
        )
        self.write_export("tradability.csv", export_frame)

    def add_market_state(self) -> None:
        result = self.market_state_result(
            include_modulation="market-state-modulation" in self.outputs
        )
        export_frame = result.export_frame()
        self.summary_rows.extend(self.market_state_summary_rows(export_frame))
        self.sections.append(self.market_state_text(result))
        if "market-state-forward" in self.outputs:
            self.write_export(
                "market-state-forward.csv",
                self.artifact_filter(export_frame, "market-state-modulation", exclude=True),
            )
        if "market-state-modulation" in self.outputs:
            self.write_export(
                "market-state-modulation.csv",
                self.artifact_filter(export_frame, "market-state-modulation"),
            )

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
            assert_reports_pass(self.reports, self.command.risk_gates)
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
        self.sections.append(format_modulation_effect_summary(modulation_frame))
        self.write_export("trade-record-modulation.csv", modulation_frame)

    def market_state_result(self, *, include_modulation: bool) -> MarketStateAnalysisResult:
        market = self.command.market_state
        forward_frames = []
        summary_frames = []
        modulation_frames = []
        for symbol, _classifier_frame, market_frame in self.prepared_market:
            forward = market_state_forward_frame(
                market_frame, symbol=symbol, horizons=market.horizons
            )
            forward_frames.append(forward)
            summary_frames.extend(self.forward_summaries(forward))
        if not forward_frames:
            return MarketStateAnalysisResult(tuple(summary_frames), tuple(modulation_frames))
        all_forward = pl.concat(forward_frames, how="diagonal_relaxed")
        for horizon in market.horizons:
            summary_frames.extend(
                frame.with_columns(pl.lit("ALL").alias("symbol"))
                for frame in self.forward_summaries(all_forward, horizons=(horizon,))
            )
            if include_modulation:
                modulation_frames.extend(self.market_state_modulation_frames(all_forward, horizon))
        return MarketStateAnalysisResult(tuple(summary_frames), tuple(modulation_frames))

    def forward_summaries(
        self, forward: pl.DataFrame, *, horizons: tuple[int, ...] | None = None
    ) -> list[pl.DataFrame]:
        market = self.command.market_state
        frames = []
        for horizon in horizons or market.horizons:
            for group_columns in self.MARKET_GROUP_SPECS:
                summary = market_state_forward_summary(
                    forward,
                    group_columns=group_columns,
                    horizon=horizon,
                    min_rows=market.min_rows,
                )
                if not summary.is_empty():
                    frames.append(summary)
        return frames

    def market_state_modulation_frames(
        self, all_forward: pl.DataFrame, horizon: int
    ) -> list[pl.DataFrame]:
        market = self.command.market_state
        delta_threshold, _delta_mode_summary = market.delta_threshold()
        robustness = ModulationRobustnessConfig(
            se_method=market.robustness.se_method,
            fdr=market.robustness.fdr,
            fdr_alpha=market.robustness.fdr_alpha,
            cohens_d_threshold=market.robustness.cohens_d_threshold,
            n_eff_min=market.robustness.n_eff_min,
        )
        frames = []
        for outcome in market.outcomes:
            modulation = market_state_modulation_matrix(
                all_forward,
                outcome_column=_market_state_outcome_column(horizon, outcome),
                min_base_rows=market.min_base_rows,
                min_cell_rows=market.min_cell_rows,
                practical_delta_threshold=delta_threshold,
                base_columns=market.base_columns or MARKET_STATE_BASE_COLUMNS,
                modulator_columns=market.modulator_columns or MARKET_STATE_MODULATOR_COLUMNS,
                time_splits=market.time_splits,
                min_segment_base_rows=market.min_segment_base_rows,
                min_segment_cell_rows=market.min_segment_cell_rows,
                robustness=robustness,
            )
            if not modulation.is_empty():
                frames.append(modulation)
        return frames

    def market_state_text(self, result: MarketStateAnalysisResult) -> str:
        market = self.command.market_state
        delta_threshold, delta_mode_summary = market.delta_threshold()
        return result.to_text() + (
            f"\n\nThresholds: horizons={','.join(str(h) for h in market.horizons)} "
            f"outcomes={','.join(market.outcomes)} min_rows={market.min_rows} "
            f"min_base_rows={market.min_base_rows} min_cell_rows={market.min_cell_rows} "
            f"delta_threshold_pct={delta_threshold:.2f} delta_mode={delta_mode_summary}"
            f"\n{self.command.classifier.summary(self.classifier_config)}"
        )

    def trade_record_modulation_frame(self) -> pl.DataFrame:
        frames = []
        for report in self.reports:
            if report.trades.is_empty():
                continue
            symbol = str(report.label).split(" ", 1)[0]
            frames.append(report.trades.with_columns(pl.lit(symbol).alias("symbol")))
        if not frames:
            return pl.DataFrame(schema=MODULATION_EFFECT_SCHEMA)
        config = self.command.research_evaluation.modulation_effect
        return modulation_effect_matrix(
            pl.concat(frames, how="diagonal_relaxed"),
            base_columns=config.base_columns or MODULATION_BASE_COLUMNS,
            modulator_columns=config.modulator_columns or MODULATION_MODULATOR_COLUMNS,
            min_base_trades=config.min_base_trades,
            min_cell_trades=config.min_cell_trades,
            practical_delta_threshold=config.practical_delta_threshold,
        )

    def write_export(self, name: str, frame: pl.DataFrame) -> None:
        if not self.should_write or frame.is_empty():
            return
        self.export_dir.mkdir(parents=True, exist_ok=True)
        path = self.export_dir / name
        frame.write_csv(path)
        self.export_messages.append(f"{name}: {path}")

    def classifier_summary_rows(self, frame: pl.DataFrame) -> list[list[str]]:
        if frame.is_empty():
            return [["classifier", "rows=0"]]
        rows = self.artifact_filter(frame, "row") if "artifact" in frame.columns else frame
        severity = self.counts_text(rows, "severity") if "severity" in rows.columns else "n/a"
        return [["classifier", f"rows={rows.height} severity={severity}"]]

    def tradability_summary_rows(self, frame: pl.DataFrame) -> list[list[str]]:
        if frame.is_empty():
            return [["tradability", "rows=0"]]
        tradability = self.artifact_filter(frame, "state-tradability")
        validity = self.artifact_filter(frame, "classifier-validity")
        return [
            [
                "tradability",
                f"state_rows={tradability.height} "
                f"buckets={self.counts_text(tradability, 'tradability_bucket')}",
            ],
            [
                "classifier-validity",
                f"rows={validity.height} status={self.counts_text(validity, 'status')}",
            ],
        ]

    def market_state_summary_rows(self, frame: pl.DataFrame) -> list[list[str]]:
        if frame.is_empty():
            return [["market-state", "rows=0"]]
        summary = self.artifact_filter(frame, "forward-summary")
        modulation = self.artifact_filter(frame, "market-state-modulation")
        directional = (
            self.count_if(summary, pl.col("directional_bias").is_in(["up", "down"]))
            if "directional_bias" in summary.columns
            else 0
        )
        global_fdr = (
            self.count_if(
                modulation,
                (pl.col("classification") == "global") & pl.col("fdr_significant"),
            )
            if {"classification", "fdr_significant"} <= set(modulation.columns)
            else 0
        )
        return [
            [
                "market-state-forward",
                f"summary_rows={summary.height} "
                f"sufficient={self.bool_count(summary, 'sufficient_rows')} "
                f"directional={directional}",
            ],
            [
                "market-state-modulation",
                f"rows={modulation.height} "
                f"robust={self.bool_count(modulation, 'robust_significant')} "
                f"fdr={self.bool_count(modulation, 'fdr_significant')} "
                f"global_fdr={global_fdr}",
            ],
        ]

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
        rows = frame.group_by(column).agg(pl.len().alias("count")).sort(column).select(
            pl.concat_str(
                [pl.col(column).cast(pl.Utf8), pl.lit("="), pl.col("count").cast(pl.Utf8)]
            ).alias("item")
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


def _market_state_outcome_column(horizon: int, outcome: str) -> str:
    return f"fwd_{horizon}_{outcome}"


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


def _status_suffix(
    reports,
    command: ResearchCommandConfig,
) -> str:
    if not reports:
        return ""
    show_status = command.run.show_status
    explain_layers = command.strategy.explain_layers
    if not (explain_layers or show_status):
        return ""
    report_list = list(reports)
    return (
        "\n\nRisk gate status\n"
        + format_status_table(report_list, command.risk_gates)
        + "\n\nCandidate-grade status\n"
        + format_candidate_status_table(report_list, command.risk_gates)
        + "\n\nCross-run consistency\n"
        + format_cross_run_consistency(report_list, command.risk_gates)
    )
