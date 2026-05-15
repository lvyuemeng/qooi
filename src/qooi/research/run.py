"""Research backtest command dispatch."""

from __future__ import annotations

from typing import Any

import polars as pl

from qooi.core.basket import ExitConfig
from qooi.core.evaluate import format_backtest_report, format_benchmark_report, format_table
from qooi.core.executor import BacktestExecutor
from qooi.core.recovery import (
    GridRecovery,
    HedgeRecovery,
    MartingaleRecovery,
    NoRecovery,
    RecoveryPolicy,
    ReverseRecovery,
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
    cache_audit_rows,
    load_cache,
    prepare_backtest_frame,
    source_inst_ids,
)
from qooi.research.diagnostics import (
    assert_reports_pass,
    format_layer_summary,
    format_status_table,
)
from qooi.research.strategies import (
    build_strategies,
    selected_strategy_names,
    strategy_args_metadata,
)
from qooi.strategies import StrategyBehavior, compute_signal_frame, strategy_signal_diagnostics


def mode_config(mode: str) -> RecoveryPolicy:
    if mode == "grid":
        return GridRecovery(zone_atr=1.0, multiplier=2.0, max_levels=3)
    if mode == "martingale":
        return MartingaleRecovery(zone_atr=1.0, max_levels=3)
    if mode == "reverse":
        return ReverseRecovery(zone_atr=1.0, max_levels=3)
    if mode == "hedge":
        return HedgeRecovery(zone_atr=1.0)
    return NoRecovery()


def exit_config_from_args(args: Any) -> ExitConfig:
    return ExitConfig(
        stop_mult=float(getattr(args, "stop_mult", 1.5)),
        target_mult=float(getattr(args, "target_mult", 1.3)),
        trail_mult=float(getattr(args, "trail_mult", 2.0)),
        max_bars=int(getattr(args, "max_bars", 10)),
    )


def run_command(args: Any) -> str:
    config = resolve_config(args)
    pairs = selected_pairs(args, config)
    if bool(getattr(args, "cache_audit", False)):
        return run_cache_audit(args, config, pairs)

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
    if bool(getattr(args, "diagnostics", False)) or bool(getattr(args, "explain_layers", False)):
        store = CacheStore()
        diagnostic_config = config
        for pair, report in zip(pairs, reports, strict=False):
            signal_inst_id, _ = source_inst_ids(pair, config.data_source)
            df, _coverage = load_cache(
                store,
                signal_inst_id,
                pair.asset.timeframe,
                diagnostic_config,
                refresh=False,
            )
            values = strategy_signal_diagnostics(df, strategy)
            signal_diagnostics.append((f"{pair.asset.symbol} {strategy.name}", values))
            if bool(getattr(args, "explain_layers", False)):
                signal_frame = compute_signal_frame(df, strategy)
                layer_summaries.append(
                    format_layer_summary(report.label, report, values, signal_frame)
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
        bt = BacktestExecutor(
            initial_capital=pair.asset.capital,
            cost_pct=0.00005,
            drawdown_stop_pct=getattr(args, "drawdown_stop_pct", None),
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
                    f"strategy_args={strategy_args_metadata(args, strategy.name)}",
                    *risk_gate_metadata(config.risk_gates),
                ),
            )
        )
    return reports


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
    return "\n\nRisk gate status\n" + format_status_table(list(reports), config.risk_gates)
