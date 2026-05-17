"""Research backtest command dispatch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
    ResolvedBacktestConfig,
    resolve_config,
    risk_gate_metadata,
    selected_pairs,
)
from qooi.research.data import (
    ClassifierContextConfig,
    DataCoverageError,
    cache_audit_rows,
    coverage_metadata,
    prepare_backtest_frame,
    prepare_classifier_frame,
)
from qooi.research.diagnostics import (
    assert_reports_pass,
    classifier_diagnostics_export_frame,
    evaluate_classifier_frame,
    format_candidate_status_table,
    format_classifier_diagnostics,
    format_cross_run_consistency,
    format_layer_summary,
    format_state_diagnostics_summary,
    format_status_table,
    format_strategy_development_summary,
    state_diagnostics_export_frame,
)
from qooi.research.strategies import (
    build_strategies,
    selected_strategy_names,
    strategy_args_metadata,
)
from qooi.strategies import StrategyBehavior, compute_signal_frame, strategy_signal_diagnostics
from qooi.strategies.features import StructureClassifierConfig


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


def exit_config_from_args(args: Any) -> ExitConfig:
    return ExitConfig(
        stop_mult=float(getattr(args, "stop_mult", 1.5)),
        target_mult=float(getattr(args, "target_mult", 1.3)),
        trail_mult=float(getattr(args, "trail_mult", 2.0)),
        max_bars=int(getattr(args, "max_bars", 10)),
        breakeven_after_target=bool(getattr(args, "breakeven_after_target", False)),
    )


def run_command(args: Any) -> str:
    config = resolve_config(args)
    pairs = selected_pairs(args, config)
    if bool(getattr(args, "cache_audit", False)):
        return run_cache_audit(args, config, pairs)

    diagnostic_mode = str(getattr(args, "diagnostic_mode", "backtest") or "backtest")
    if bool(getattr(args, "state_diagnostics", False)):
        diagnostic_mode = "state"
    if diagnostic_mode == "classifier":
        return run_classifier_diagnostics(args, config, pairs)

    names = selected_strategy_names(args)
    selection = build_strategies(names, args)
    recovery_cfg = mode_config(str(getattr(args, "mode", "base")))
    exit_cfg = exit_config_from_args(args)

    if len(selection.strategies) > 1:
        benchmark_results = []
        all_reports = []
        for strategy in selection.strategies:
            reports = run_reports(pairs, strategy, recovery_cfg, exit_cfg, args, config)
            all_reports.extend(reports)
            benchmark_results.append((strategy.name, reports))
        assert_reports_pass(all_reports, config.risk_gates)
        output = format_benchmark_report(
            mode=str(getattr(args, "mode", "base")),
            benchmark_results=benchmark_results,
            diagnostics=bool(getattr(args, "diagnostics", False)),
        )
        return output + _status_suffix(all_reports, args, config)

    strategy = selection.strategies[0]
    if config.style != "single":
        return run_style(pairs, strategy, recovery_cfg, exit_cfg, args, config)

    reports = run_reports(pairs, strategy, recovery_cfg, exit_cfg, args, config)
    assert_reports_pass(reports, config.risk_gates)
    signal_diagnostics = []
    layer_summaries = []
    development_summaries = []
    state_summaries = []
    state_export_frames = []
    wants_diagnostics = bool(getattr(args, "diagnostics", False))
    wants_explain_layers = bool(getattr(args, "explain_layers", False))
    wants_state_diagnostics = diagnostic_mode == "state" or bool(
        getattr(args, "state_diagnostics", False)
    )
    state_export_path = str(
        getattr(args, "diagnostics_export", "")
        or getattr(args, "state_diagnostics_export", "")
        or ""
    )
    if wants_diagnostics or wants_explain_layers or wants_state_diagnostics or state_export_path:
        for pair, report in zip(pairs, reports, strict=False):
            try:
                prepared = prepare_backtest_frame(CacheStore(), pair, strategy, args, config)
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
            if state_export_path:
                state_export_frames.append(
                    state_diagnostics_export_frame(report.label, report, signal_frame)
                )

    output = format_backtest_report(
        mode=str(getattr(args, "mode", "base")),
        strategy=strategy.name,
        reports=reports,
        detail=bool(getattr(args, "detail", True)),
        diagnostics=bool(getattr(args, "diagnostics", False)),
        signal_diagnostics=signal_diagnostics,
    )
    output += _status_suffix(reports, args, config)
    if layer_summaries:
        output += "\n\nLayer behavior summary\n" + "\n\n".join(layer_summaries)
    if development_summaries:
        output += "\n\nStrategy development diagnosis\n" + "\n\n".join(development_summaries)
    if state_summaries:
        output += "\n\nState diagnostics\n" + "\n\n".join(state_summaries)
    if state_export_path and state_export_frames:
        path = Path(state_export_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pl.concat(state_export_frames, how="diagonal_relaxed").write_csv(path)
        output += f"\n\nState diagnostics export written: {path}"
    return output


def run_reports(
    pairs,
    strategy: StrategyBehavior,
    recovery_cfg: RecoveryPolicy,
    exit_cfg: ExitConfig,
    args: Any,
    config: ResolvedBacktestConfig,
):
    reports = []
    store = CacheStore()
    for pair in pairs:
        try:
            prepared = prepare_backtest_frame(store, pair, strategy, args, config)
        except FileNotFoundError as exc:
            print(f"skip {pair.asset.symbol}: {exc}")
            continue
        except DataCoverageError as exc:
            print(f"data incomplete {pair.asset.symbol}: {exc}")
            reports.append(_data_incomplete_report(pair, strategy, exc, config))
            continue
        bt = BacktestExecutor(
            initial_capital=pair.asset.capital,
            cost_pct=0.00005,
            drawdown_stop_pct=getattr(args, "drawdown_stop_pct", None),
            max_per_strategy_symbol=config.max_per_strategy_symbol,
            loss_cooldown_bars=int(getattr(args, "loss_cooldown_bars", 0) or 0),
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
                    f"recovery_policy={getattr(args, 'mode', 'base')}",
                    f"strategy_args={strategy_args_metadata(args, strategy.name)}",
                    *risk_gate_metadata(config.risk_gates),
                ),
            )
        )
    return reports


def classifier_config_from_args(args: Any) -> StructureClassifierConfig:
    profile = str(getattr(args, "classifier_profile", "default") or "default")
    if profile == "fixed":
        return StructureClassifierConfig.fixed()
    if profile == "rolling":
        return StructureClassifierConfig.rolling_quantile()
    return StructureClassifierConfig.default()


def run_classifier_diagnostics(args: Any, config: ResolvedBacktestConfig, pairs) -> str:
    store = CacheStore()
    context = ClassifierContextConfig(classifier=classifier_config_from_args(args))
    summaries = []
    export_frames = []
    export_path = str(getattr(args, "diagnostics_export", "") or "")
    for pair in pairs:
        try:
            prepared = prepare_classifier_frame(store, pair, args, config, context)
        except FileNotFoundError as exc:
            print(f"skip {pair.asset.symbol}: {exc}")
            continue
        except DataCoverageError as exc:
            print(f"data incomplete {pair.asset.symbol}: {exc}")
            continue
        diagnostics = evaluate_classifier_frame(pair.asset.symbol, prepared.frame)
        summaries.append(format_classifier_diagnostics(diagnostics))
        if export_path:
            export_frames.append(classifier_diagnostics_export_frame(diagnostics))
    output = "Classifier diagnostics"
    if summaries:
        output += "\n\n" + "\n\n".join(summaries)
    if export_path and export_frames:
        path = Path(export_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pl.concat(export_frames, how="diagonal_relaxed").write_csv(path)
        output += f"\n\nClassifier diagnostics export written: {path}"
    return output


def _data_incomplete_report(
    pair,
    strategy: StrategyBehavior,
    error: DataCoverageError,
    config: ResolvedBacktestConfig,
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
            *config.metadata(),
            "data_quality=data_incomplete",
            "data_incomplete_reason=listing_age"
            if "starts_after_target_since" in coverage.notes
            else "data_incomplete_reason=coverage_low",
            f"required_coverage_pct={error.required_pct:.1f}",
            *coverage_metadata(coverage),
            f"strategy_args={strategy_args_metadata(None, strategy.name)}",
        ),
    )


def run_style(
    pairs,
    strategy: StrategyBehavior,
    recovery_cfg: RecoveryPolicy,
    exit_cfg: ExitConfig,
    args: Any,
    config: ResolvedBacktestConfig,
) -> str:
    lines = []
    store = CacheStore()
    for pair in pairs:
        prepared = prepare_backtest_frame(store, pair, strategy, args, config)

        def _run_window(seg: pl.DataFrame):
            bt = BacktestExecutor(
                initial_capital=pair.asset.capital,
                cost_pct=0.00005,
                drawdown_stop_pct=getattr(args, "drawdown_stop_pct", None),
                max_per_strategy_symbol=config.max_per_strategy_symbol,
                loss_cooldown_bars=int(getattr(args, "loss_cooldown_bars", 0) or 0),
            )
            return bt.run_result(
                seg,
                pair,
                exit_cfg=exit_cfg,
                recovery_cfg=recovery_cfg,
                strategy=strategy,
                precomputed_signal=prepared.precomputed_signal,
            )

        if config.style == "rolling":
            result = rolling_window(
                _run_window,
                prepared.frame,
                lookback_bars=int(getattr(args, "train_bars", 500)),
                step_bars=int(getattr(args, "step_bars", 100)),
            )
        elif config.style == "walk-forward":
            result = walk_forward(
                _run_window,
                prepared.frame,
                train_bars=int(getattr(args, "train_bars", 500)),
                test_bars=int(getattr(args, "test_bars", 100)),
                step_bars=int(getattr(args, "step_bars", 100)),
            )
        else:
            result = cross_validate(
                _run_window,
                prepared.frame,
                folds=int(getattr(args, "folds", 5)),
            )
        lines.append(result.summary())
    return "\n".join(lines)


def run_cache_audit(args: Any, config: ResolvedBacktestConfig, pairs) -> str:
    rows = cache_audit_rows(pairs, args, config)
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


def _status_suffix(reports, args: Any, config: ResolvedBacktestConfig) -> str:
    if not reports:
        return ""
    show_status = bool(getattr(args, "show_status", False))
    explain_layers = bool(getattr(args, "explain_layers", False))
    if not (explain_layers or show_status):
        return ""
    report_list = list(reports)
    return (
        "\n\nRisk gate status\n"
        + format_status_table(report_list, config.risk_gates)
        + "\n\nCandidate-grade status\n"
        + format_candidate_status_table(report_list, config.risk_gates)
        + "\n\nCross-run consistency\n"
        + format_cross_run_consistency(report_list, config.risk_gates)
    )
