"""Research backtest orchestration tests."""

import polars as pl
import pytest

from qooi.core.evaluate import (
    BacktestDiagnostics,
    BasketLifecycleDiagnostics,
    PortfolioRiskDiagnostics,
    Report,
)
from qooi.core.plot import plot_market_state_horizon_decay, plot_market_state_modulation_heatmap
from qooi.research.config import (
    ClassifierConfigRequest,
    MarketStateConfig,
    ResearchCommandConfig,
    apply_sizing_overrides,
)
from qooi.research.diagnostics import (
    ModulationRobustnessConfig,
    TradabilityConfig,
    add_market_state_reductions,
    adjusted_standard_error,
    candidate_report_status,
    classifier_health_frame,
    classifier_validity_frame,
    effective_sample_size,
    format_candidate_status_table,
    format_cross_run_consistency,
    format_market_state_forward_summary,
    format_modulation_effect_summary,
    format_state_diagnostics_summary,
    format_state_filter_delta,
    format_state_profitability_summary,
    format_strategy_development_summary,
    market_state_forward_frame,
    market_state_forward_summary,
    market_state_modulation_matrix,
    modulation_effect_matrix,
    report_status,
    sizing_layer,
    state_diagnostics_export_frame,
    state_profitability_export_frame,
    state_tradability_frame,
)
from qooi.research.instruments import CORE_UNIVERSE
from qooi.research.run import strategy_selection_from_config
from qooi.research.workflows import (
    CacheAuditRequest,
    apply_signal_debug_filters,
    build_history_refresh_requests,
)
from qooi.strategies import HoldPolicy, SignalRule, StrategySpec
from qooi.strategies.catalog import (
    BENCHMARK_GROUPS,
)
from qooi.strategies.specs import (
    structure_event_reversal_v1_spec,
    structure_event_trend_aligned_v1_spec,
)


def _args(**overrides):
    defaults = {
        "profile": "research",
        "days": 730,
        "min_bars": 12000,
        "min_coverage_pct": 0.0,
        "max_dd_pct": None,
        "max_notional_exposure_pct": None,
        "min_trades": 0,
        "min_pf": 0.0,
        "min_expectancy_pct": None,
        "min_execution_acceptance_pct": 0.0,
        "fail_on_risk": False,
        "normalize_sizing": False,
        "risk_pct": None,
        "max_notional_pct": None,
        "leverage": None,
        "capital": None,
        "min_contracts": None,
        "max_per_strategy_symbol": 0,
        "universe": "core",
        "data_source": "swap",
        "style": "single",
        "symbol": "",
        "exclude_symbol": "",
        "strategy": "ema_trend_baseline",
        "strategies": "",
        "benchmark": False,
        "benchmark_group": "baselines",
        "long_only": False,
        "short_only": False,
        "include_signal_id": "",
        "exclude_signal_id": "",
        "refresh_full": False,
    }
    defaults.update(overrides)
    return defaults


def _command(**overrides):
    config = ResearchCommandConfig()
    run_updates = {}
    cache_updates = {}
    risk_updates = {}
    sizing_updates = {}
    strategy_updates = {}
    for key in ("profile", "universe", "data_source", "symbol", "exclude_symbol"):
        if key in overrides:
            value = overrides[key]
            run_updates[key] = (
                tuple(item.strip() for item in value.split(",") if item.strip())
                if key == "exclude_symbol"
                else value
            )
    for key in ("days", "min_bars", "min_coverage_pct"):
        if key in overrides:
            cache_updates[key] = overrides[key]
    for source, target in (("normalize_sizing", "normalize"),):
        if source in overrides:
            sizing_updates[target] = overrides[source]
    for key in (
        "risk_pct",
        "max_notional_pct",
        "leverage",
        "capital",
        "min_contracts",
        "max_per_strategy_symbol",
    ):
        if key in overrides:
            sizing_updates[key] = overrides[key]
    for key in (
        "max_dd_pct",
        "max_notional_exposure_pct",
        "min_trades",
        "min_pf",
        "min_expectancy_pct",
        "min_execution_acceptance_pct",
        "fail_on_risk",
    ):
        if key in overrides:
            risk_updates[key] = overrides[key]
    for key in ("strategy", "benchmark", "benchmark_group", "long_only", "short_only", "style"):
        if key in overrides:
            strategy_updates[key] = overrides[key]
    for source, target in (
        ("include_signal_id", "include_signal_id"),
        ("exclude_signal_id", "exclude_signal_id"),
    ):
        if source in overrides:
            value = overrides[source]
            strategy_updates[target] = tuple(
                item.strip() for item in value.split(",") if item.strip()
            )
    if "strategies" in overrides:
        strategy_updates["strategies"] = tuple(
            item.strip() for item in overrides["strategies"].split(",") if item.strip()
        )
    if run_updates:
        config = config.model_copy(update={"run": config.run.model_copy(update=run_updates)})
    if cache_updates:
        config = config.model_copy(update={"cache": config.cache.model_copy(update=cache_updates)})
    if risk_updates:
        config = config.model_copy(update={"risk": config.risk.model_copy(update=risk_updates)})
    if sizing_updates:
        config = config.model_copy(
            update={"sizing": config.sizing.model_copy(update=sizing_updates)}
        )
    if strategy_updates:
        config = config.model_copy(
            update={"strategy": config.strategy.model_copy(update=strategy_updates)}
        )
    return ResearchCommandConfig.model_validate(config.model_dump())


def _report(
    trades=None,
    equity=None,
    *,
    pair=None,
    label=None,
    diagnostics=None,
    metadata=(),
    **kwargs,
):
    return Report.from_raw(
        trades or [],
        equity or [100.0, 100.0],
        pair or CORE_UNIVERSE[0],
        label=label,
        diagnostics=diagnostics,
        metadata=metadata,
        **kwargs,
    )


def _trade(**overrides):
    trade = {
        "pnl": 0.01,
        "pnl_usd": 1.0,
        "net_pnl_usd": 1.0,
    }
    trade.update(overrides)
    return trade


def _diagnostic_strategy(*, required_columns=None, filters=(), max_bars=2):
    return StrategySpec(
        name="diagnostic_spec",
        required_columns=required_columns or ("timestamp", "close", "high", "low", "x"),
        features=(),
        entries=(SignalRule("long_x", 1, pl.col("x") > 0),),
        filters=filters,
        hold=HoldPolicy(max_bars=max_bars),
    )


def test_strategy_registry_builds_baselines():
    selection = strategy_selection_from_config(_command(benchmark=True))

    assert [strategy.name for strategy in selection.strategies] == [
        "ema_trend_baseline",
        "rsi_bounce_reversion",
        "rsi_macd_trend",
        "momentum_burst",
    ]


def test_strategy_registry_builds_structural_event_reversal():
    strategy = structure_event_reversal_v1_spec()

    assert strategy.name == "structure_event_reversal_v1"


def test_strategy_registry_builds_structural_event_trend_aligned():
    strategy = structure_event_trend_aligned_v1_spec()

    assert strategy.name == "structure_event_trend_aligned_v1"


def test_cache_audit_history_requests_include_base_and_higher_context_targets():
    command = _command(strategy="structure_event_trend_aligned_v1", days=730, min_bars=12000)
    pair = CORE_UNIVERSE[0]

    requests = build_history_refresh_requests(
        CacheAuditRequest(
            pairs=(pair,),
            data_source=command.run.data_source,
            days=command.days,
            min_bars=command.min_bars,
            min_coverage_pct=command.min_coverage_pct,
            refresh=True,
            incremental=True,
        )
    )

    by_bar = {(request.inst_id, request.bar): request for request in requests}

    assert (pair.asset.symbol, "1H") in by_bar
    assert (pair.asset.symbol, "15m") not in by_bar
    assert by_bar[(pair.asset.symbol, "4H")].min_bars == 4_630
    assert by_bar[(pair.asset.symbol, "1D")].min_bars == 980


def test_strategy_registry_builds_no_volume_structural_variants():
    selection = strategy_selection_from_config(
        _command(
            strategies="structure_event_reversal_no_vol_v1,structure_event_trend_aligned_no_vol_v1"
        )
    )
    reversal, trend_aligned = selection.strategies

    assert reversal.name == "structure_event_reversal_no_vol_v1"
    assert trend_aligned.name == "structure_event_trend_aligned_no_vol_v1"


def test_strategy_registry_rejects_unknown_name():
    try:
        strategy_selection_from_config(_command(strategy="missing"))
    except ValueError as exc:
        assert "strategy.strategy='missing'" in str(exc)
    else:
        raise AssertionError("missing strategy did not raise")


def test_strategy_selection_supports_comma_list_and_benchmark_group():
    assert strategy_selection_from_config(
        _command(strategies="ema_trend_baseline,rsi_macd_trend")
    ).names == ("ema_trend_baseline", "rsi_macd_trend")
    assert (
        strategy_selection_from_config(_command(benchmark=True)).names
        == BENCHMARK_GROUPS["baselines"]
    )


def test_structure_development_group_uses_retained_comparisons_only():
    assert BENCHMARK_GROUPS["structure-development"] == (
        "structure_event_reversal_v1",
        "structure_event_trend_aligned_v1",
        "structure_event_trend_aligned_no_range_v1",
        "structure_event_trend_aligned_no_range_longs_v1",
        "structure_event_reversal_no_vol_v1",
        "structure_event_trend_aligned_no_vol_v1",
        "ema_trend_baseline",
        "momentum_burst",
        "rsi_macd_trend",
    )
    assert not any("zscore" in name for name in BENCHMARK_GROUPS["structure-development"])
    assert "structure_event_reversal_v1" in BENCHMARK_GROUPS["candidate"]
    assert "structure_event_trend_aligned_v1" in BENCHMARK_GROUPS["candidate"]


def test_safe_profile_enables_normalized_sizing_and_gates():
    config = _command(profile="safe")

    assert config.sizing_overrides.normalize is True
    assert config.sizing_overrides.risk_pct == 0.02
    assert config.sizing_overrides.max_notional_pct == 1.0
    assert config.risk_gates.min_pf == 1.10
    assert config.risk_gates.max_notional_exposure_pct == 200.0
    assert config.risk_gates.min_execution_acceptance_pct == 30.0
    assert config.max_per_strategy_symbol == 1


def test_research_profile_keeps_default_basket_stacking_cap():
    config = _command(profile="research")

    assert config.max_per_strategy_symbol == 3


def test_sizing_override_does_not_mutate_global_pair():
    pair = CORE_UNIVERSE[0]
    config = _command(normalize_sizing=True, risk_pct=0.03)

    updated = apply_sizing_overrides(pair, config.sizing_overrides)

    assert updated is not pair
    assert updated.asset.max_risk_pct == 0.03
    assert pair.asset.max_risk_pct == CORE_UNIVERSE[0].asset.max_risk_pct


def test_selected_pairs_filters_symbol_and_exclusion():
    command = _command(symbol="ETH-USDT-SWAP", exclude_symbol="XAU-USDT-SWAP")

    pairs = command.pairs()

    assert [pair.asset.symbol for pair in pairs] == ["ETH-USDT-SWAP"]


def test_report_status_flags_layer_four_risk_failures():
    report = _report(
        [_trade(pnl=-0.01, pnl_usd=-1.0), _trade(pnl=-0.02, pnl_usd=-2.0)],
        [100.0, 90.0, 80.0],
        active_exposure=[0.0, 1.0, 1.0],
        timestamps=[1, 2, 3],
        signals=[0.0, 1.0, 1.0],
    )
    gates = _command(profile="safe").risk_gates

    status = report_status(report, gates)

    assert status.status == "FAIL"
    assert "PF_LOW" in status.reasons or "EXP_LOW" in status.reasons


def test_report_status_flags_yield_divergence():
    report = _report(
        [
            _trade(pnl=0.01, pnl_usd=-2.0, net_pnl_usd=-2.0, reason="stop"),
            _trade(pnl=0.02, pnl_usd=-1.0, net_pnl_usd=-1.0, reason="stop"),
        ],
        [100.0, 98.0],
    )
    gates = _command().risk_gates

    status = report_status(report, gates)

    assert status.status == "FAIL"
    assert "YIELD_DIVERGENCE" in status.reasons


def test_report_status_flags_oversized_losses():
    report = _report(
        [
            _trade(
                pnl=0.02,
                pnl_usd=2.0,
                net_pnl_usd=2.0,
                entry_notional_usd=100.0,
                notional_pct_capital=10.0,
                contracts=1.0,
            ),
            _trade(
                pnl=-0.01,
                pnl_usd=-5.0,
                net_pnl_usd=-5.0,
                entry_notional_usd=500.0,
                notional_pct_capital=50.0,
                contracts=5.0,
            ),
        ],
        [100.0, 97.0],
    )
    gates = _command().risk_gates

    status = report_status(report, gates)

    assert "LOSS_OVERSIZED" in status.reasons


def test_report_status_flags_stop_dominated_losses():
    report = _report(
        [
            _trade(
                side="buy",
                pnl=-0.02,
                pnl_usd=-20.0,
                net_pnl_usd=-20.0,
                gross_pnl_usd=-20.0,
                reason="stop",
                signal_id="bad_long",
            ),
            _trade(
                side="buy",
                pnl=0.01,
                pnl_usd=5.0,
                net_pnl_usd=5.0,
                gross_pnl_usd=5.0,
                reason="strategy_exit",
                signal_id="good_long",
            ),
        ],
        [100.0, 80.0, 85.0],
    )
    gates = _command().risk_gates

    status = report_status(report, gates)

    assert "STOP_DOMINATED_LOSSES" in status.reasons
    assert "STOP_SIGNAL_LEAK" in status.reasons
    assert "STOP_SIDE_LEAK" in status.reasons


def test_report_status_flags_execution_infeasible_min_contract_runs():
    report = _report(
        [_trade()],
        [100.0, 101.0],
        diagnostics=BacktestDiagnostics(
            lifecycle=BasketLifecycleDiagnostics(
                entry_signals=100,
                entry_actions=10,
                blocked_entry_signals=90,
                sizing_blocked_entries=80,
                entry_acceptance_rate_pct=10.0,
                min_contract_block_count=80,
            )
        ),
    )
    gates = _command(profile="safe", min_pf=0.0, min_trades=0).risk_gates

    status = report_status(report, gates)

    assert "EXECUTION_INFEASIBLE" in status.reasons


def test_candidate_gate_flags_drawdown_above_promotion_target():
    report = _report([_trade(pnl=-0.01, pnl_usd=-1.0, net_pnl_usd=-1.0)], [100.0, 94.0, 95.0])
    gates = _command(profile="safe", min_pf=0.0, min_trades=0).risk_gates

    status = candidate_report_status(report, gates)

    assert status.status == "FAIL"
    assert status.classification == "FEASIBLE"
    assert "DD_TARGET_HIGH" in status.reasons


@pytest.mark.parametrize(
    ("report", "expected_reason"),
    [
        (
            _report(
                [_trade()],
                [100.0, 101.0],
                diagnostics=BacktestDiagnostics(
                    lifecycle=BasketLifecycleDiagnostics(
                        entry_signals=100,
                        blocked_entry_signals=90,
                        sizing_blocked_entries=80,
                        entry_acceptance_rate_pct=10.0,
                        min_contract_block_count=80,
                    )
                ),
            ),
            "EXECUTION_INFEASIBLE",
        ),
        (
            _report(
                [],
                [100.0, 100.0],
                metadata=(
                    "data_quality=data_incomplete",
                    "data_incomplete_reason=listing_age",
                    "data_coverage_pct=27.1",
                ),
            ),
            "DATA_INCOMPLETE_LISTING_AGE",
        ),
    ],
)
def test_candidate_gate_classifies_diagnostic_only_failures(report, expected_reason):
    gates = _command(profile="safe", min_pf=0.0, min_trades=0).risk_gates

    operational = report_status(report, gates)
    status = candidate_report_status(report, gates)

    assert operational.status == "FAIL"
    assert expected_reason in operational.reasons
    assert status.status == "WARN"
    assert status.classification == "DIAGNOSTIC_ONLY"
    assert expected_reason in status.reasons


def test_candidate_and_cross_run_tables_are_formatted():
    reports = [
        _report(
            [
                _trade(
                    pnl=-0.01,
                    pnl_usd=-1.0,
                    net_pnl_usd=-1.0,
                    entry_trend_bucket="uptrend",
                    entry_volatility_bucket="expanded",
                ),
                _trade(
                    pnl=-0.01,
                    pnl_usd=-1.0,
                    net_pnl_usd=-1.0,
                    entry_trend_bucket="uptrend",
                    entry_volatility_bucket="expanded",
                ),
            ],
            [100.0, 99.0, 98.0],
            label="A",
        ),
        _report(
            [
                _trade(entry_trend_bucket="flat", entry_volatility_bucket="normal"),
                _trade(entry_trend_bucket="flat", entry_volatility_bucket="normal"),
            ],
            [100.0, 101.0, 102.0],
            pair=CORE_UNIVERSE[1],
            label="B",
        ),
    ]
    gates = _command().risk_gates

    candidate_table = format_candidate_status_table(reports, gates)
    consistency = format_cross_run_consistency(reports, gates)

    assert "TargetDD%" in candidate_table
    assert "Cross-run consistency: FAIL" in consistency
    assert "CROSS_ASSET_INCONSISTENT" in consistency


def test_sizing_layer_exposes_yield_and_risk_controls():
    report = _report(
        [_trade(pnl=-0.01, pnl_usd=-3.0, net_pnl_usd=-3.0, reason="stop")],
        [100.0, 97.0],
        diagnostics=BacktestDiagnostics(
            avg_active_exposure=1.0,
            max_active_exposure=2.0,
            avg_notional_exposure_pct=50.0,
            max_notional_exposure_pct=150.0,
            risk=PortfolioRiskDiagnostics(
                avg_active_exposure=1.0,
                max_active_exposure=2.0,
                avg_notional_exposure_pct=50.0,
                max_notional_exposure_pct=150.0,
                stop_exit_count=1,
                stop_exit_net_pnl_usd=-3.0,
            ),
        ),
    )

    layer, summary = sizing_layer(report)

    assert layer == "SIZING"
    assert "fee_drag" in summary
    assert "loss/win_notional" in summary
    assert "stops=1" in summary
    assert "stop_net=$-3.00" in summary


def test_strategy_development_summary_explains_sparse_signal_attrition_and_paths():
    strategy = _diagnostic_strategy(
        required_columns=("timestamp", "close", "high", "low", "atr_14", "x", "filter_ok"),
        filters=(pl.col("filter_ok"),),
    )
    signal_frame = pl.DataFrame(
        {
            "timestamp": [1, 2, 3, 4],
            "close": [100.0, 100.0, 102.0, 101.0],
            "high": [100.0, 101.0, 103.0, 102.0],
            "low": [100.0, 99.0, 100.0, 99.5],
            "atr_14": [1.0, 1.0, 1.0, 1.0],
            "x": [0, 1, 1, 0],
            "filter_ok": [False, False, True, False],
            "entry_signal": [0.0, 0.0, 1.0, 0.0],
        }
    )
    report = _report(
        [
            _trade(
                side="buy",
                entry_px=100.0,
                exit_px=101.0,
                entry_bar_index=1,
                exit_bar_index=3,
                bars_held=2,
                pnl=-0.01,
                pnl_usd=-1.0,
                net_pnl_usd=-1.0,
                reason="stop",
                entry_volatility_bucket="expanded",
            )
        ],
        [100.0, 99.0],
        diagnostics=BacktestDiagnostics(
            lifecycle=BasketLifecycleDiagnostics(entry_actions=1, entry_signals=1)
        ),
        metadata=("max_per_strategy_symbol=1",),
    )

    summary = format_strategy_development_summary(
        "TEST diagnostic_spec", report, signal_frame, strategy
    )

    assert "Signal opportunity" in summary
    assert "raw_opportunities=2" in summary
    assert "entry_events=1" in summary
    assert "Filter 0 attrition" in summary
    assert "removed=1" in summary
    assert "Trade path" in summary
    assert "avg_mfe=+3.00%" in summary
    assert "Loss path" in summary
    assert "volatility_expansion:1" in summary
    assert "single-thesis basket mode" in summary


def test_strategy_development_summary_separates_loss_taxonomy():
    strategy = _diagnostic_strategy()
    signal_frame = pl.DataFrame(
        {
            "timestamp": [1, 2, 3, 4, 5],
            "close": [100.0, 99.0, 98.0, 97.0, 96.0],
            "high": [101.0, 100.0, 99.0, 98.0, 97.0],
            "low": [99.0, 98.0, 97.0, 96.0, 95.0],
            "x": [1, 1, 1, 1, 1],
            "entry_signal": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    trades = [
        _trade(
            **{
                "side": "buy",
                "entry_px": 100.0,
                "exit_px": 99.0,
                "entry_bar_index": 0,
                "exit_bar_index": 1,
                "bars_held": 1,
                "pnl": -0.01,
                "pnl_usd": -1.0,
                "net_pnl_usd": -1.0,
                "reason": "stop",
                "entry_breakout_acceptance_low": 1.0,
                "entry_liquidity_event_type": "breakout_acceptance_low",
            }
        ),
        _trade(
            **{
                "side": "sell",
                "entry_px": 100.0,
                "exit_px": 101.0,
                "entry_bar_index": 1,
                "exit_bar_index": 2,
                "bars_held": 1,
                "pnl": -0.01,
                "pnl_usd": -1.0,
                "net_pnl_usd": -1.0,
                "reason": "stop",
                "entry_structure_trend_state": "uptrend",
                "entry_liquidity_event_type": "none",
            }
        ),
        _trade(
            **{
                "side": "buy",
                "entry_px": 100.0,
                "exit_px": 99.0,
                "entry_bar_index": 2,
                "exit_bar_index": 3,
                "bars_held": 1,
                "pnl": -0.01,
                "pnl_usd": -1.0,
                "net_pnl_usd": -1.0,
                "reason": "stop",
                "entry_liquidity_event_type": "bullish_reclaim",
            }
        ),
        _trade(
            **{
                "side": "buy",
                "entry_px": 100.0,
                "exit_px": 99.0,
                "entry_bar_index": 3,
                "exit_bar_index": 4,
                "bars_held": 1,
                "pnl": -0.01,
                "pnl_usd": -1.0,
                "net_pnl_usd": -1.0,
                "reason": "stop",
                "entry_liquidity_event_type": "none",
            }
        ),
    ]
    report = _report(
        trades,
        [100.0, 99.0, 98.0],
        diagnostics=BacktestDiagnostics(
            lifecycle=BasketLifecycleDiagnostics(entry_actions=4, entry_signals=4)
        ),
    )

    summary = format_strategy_development_summary(
        "TEST diagnostic_spec", report, signal_frame, strategy
    )

    assert "accepted_breakout_against_reversion:1" in summary
    assert "trend_continuation_against_reversion:1" in summary
    assert "reclaim_failed:1" in summary
    assert "unclassified_none_event:1" in summary
    assert "Loss confidence" in summary
    assert "low_confidence_residual=1" in summary


def test_strategy_development_summary_reports_none_coverage_and_structure_overlap():
    strategy = _diagnostic_strategy(max_bars=None)
    signal_frame = pl.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "close": [100.0, 101.0, 99.0],
            "high": [101.0, 102.0, 100.0],
            "low": [99.0, 100.0, 98.0],
            "x": [1, 1, 1],
            "entry_signal": [1.0, 1.0, 1.0],
            "market_stage": ["unknown", "range", "unknown"],
            "structure_trend_state": ["uptrend", "range", "uptrend"],
        }
    )
    trades = [
        _trade(
            **{
                "side": "sell",
                "entry_px": 100.0,
                "exit_px": 101.0,
                "entry_bar_index": 0,
                "exit_bar_index": 1,
                "pnl": -0.01,
                "pnl_usd": -1.0,
                "net_pnl_usd": -1.0,
                "reason": "stop",
                "entry_liquidity_event_type": "none",
                "entry_market_stage": "unknown",
                "entry_market_stage_reason": "trend_without_range_break",
                "entry_stage_unknown_reason": "trend_without_range_break",
                "entry_structure_trend_state": "uptrend",
            }
        ),
        _trade(
            **{
                "side": "buy",
                "entry_px": 100.0,
                "exit_px": 99.0,
                "entry_bar_index": 1,
                "exit_bar_index": 2,
                "pnl": -0.01,
                "pnl_usd": -1.0,
                "net_pnl_usd": -1.0,
                "reason": "stop",
                "entry_liquidity_event_type": "none",
                "entry_market_stage": "range",
                "entry_market_stage_reason": "compressed_mid_range",
                "entry_stage_unknown_reason": "none",
                "entry_structure_trend_state": "range",
            }
        ),
        _trade(
            **{
                "side": "buy",
                "entry_px": 100.0,
                "exit_px": 101.0,
                "entry_bar_index": 1,
                "exit_bar_index": 2,
                "pnl": 0.01,
                "pnl_usd": 1.0,
                "net_pnl_usd": 1.0,
                "reason": "strategy_exit",
                "entry_liquidity_event_type": "bullish_reclaim",
                "entry_market_stage": "range",
                "entry_market_stage_reason": "compressed_mid_range",
                "entry_stage_unknown_reason": "none",
                "entry_structure_trend_state": "range",
            }
        ),
    ]
    report = _report(trades, [100.0, 99.0, 100.0])

    summary = format_strategy_development_summary(
        "TEST diagnostic_spec", report, signal_frame, strategy
    )

    assert "None-event coverage" in summary
    assert "none_event_trades=2/3" in summary
    assert "residual=66.7%" in summary
    assert "none_by_stage_reason=" in summary
    assert "trend_without_range_break" in summary
    assert "Structural overlap audit" in summary
    assert "overlap_not_confirmation=true" in summary
    assert "structure_x_reason=" in summary
    assert "Interpretability guardrail" in summary
    assert "attribution identifies where PnL occurred, not why" in summary
    assert "not a validated trend edge" in summary
    assert "event_coverage=poor" in summary
    assert "none_is_residual_not_market_event=true" in summary


def test_strategy_development_summary_reports_structural_event_rows():
    strategy = _diagnostic_strategy()
    signal_frame = pl.DataFrame(
        {
            "timestamp": [1, 2, 3, 4],
            "close": [100.0, 99.0, 101.0, 102.0],
            "high": [101.0, 100.0, 102.0, 104.0],
            "low": [99.0, 97.0, 100.0, 101.0],
            "x": [1, 1, 1, 1],
            "entry_signal": [1.0, 0.0, -1.0, 0.0],
            "liquidity_event_type": [
                "failed_breakout_low",
                "bullish_reclaim",
                "failed_breakout_high",
                "breakout_acceptance_high",
            ],
            "failed_breakout_low": [True, False, False, False],
            "failed_breakout_high": [False, False, True, False],
            "prior_liquidity_low": [98.0, 98.0, 98.0, 98.0],
            "prior_liquidity_high": [102.0, 102.0, 102.0, 102.0],
            "volume_impulse": [True, False, True, False],
            "event_quality_score": [2.0, 1.0, 2.5, 0.0],
        }
    )
    report = _report(
        [
            _trade(
                side="buy",
                entry_bar_index=0,
                entry_liquidity_event_type="failed_breakout_low",
                entry_structure_trend_state="uptrend",
                entry_market_stage="markup",
                entry_event_quality_score=2.0,
                entry_volume_impulse=True,
                net_pnl_usd=2.0,
                pnl_usd=2.0,
            ),
            _trade(
                side="sell",
                entry_bar_index=2,
                entry_liquidity_event_type="failed_breakout_high",
                entry_structure_trend_state="downtrend",
                entry_market_stage="distribution_or_reversal",
                entry_event_quality_score=2.5,
                entry_volume_impulse=True,
                net_pnl_usd=-1.0,
                pnl_usd=-1.0,
            ),
        ],
        [100.0, 102.0, 101.0],
    )

    summary = format_strategy_development_summary(
        "TEST diagnostic_spec", report, signal_frame, strategy
    )

    assert "Structural event opportunity" in summary
    assert "failed_breakout=2" in summary
    assert "after_both=2" in summary
    assert "Structural event quality" in summary
    assert "Structural event volume impulse" in summary
    assert "Structural event attribution" in summary
    assert "Structural side x trend" in summary
    assert "sell/downtrend" in summary
    assert "loss_source" in summary
    assert "Structural side x event" in summary
    assert "Structural side x event x stage" in summary
    assert "distribution_or_reversal" in summary
    assert "Structural quality x result" in summary
    assert "Structural volume x result" in summary
    assert "Structural event time clustering" in summary
    assert "promotion_clustered=" in summary
    assert "Reclaim/sweep extension" in summary
    assert "disabled reclaim_extension_trades=0" in summary


def test_structural_event_opportunity_counts_executable_failed_breakouts_only():
    strategy = _diagnostic_strategy()
    signal_frame = pl.DataFrame(
        {
            "timestamp": [1, 2, 3, 4],
            "close": [100.0, 99.0, 101.0, 102.0],
            "high": [101.0, 100.0, 102.0, 104.0],
            "low": [99.0, 97.0, 100.0, 101.0],
            "x": [1, 1, 1, 1],
            "entry_signal": [1.0, 0.0, -1.0, 0.0],
            "liquidity_event_type": [
                "failed_breakout_low",
                "bullish_reclaim",
                "failed_breakout_high",
                "failed_breakout_high",
            ],
            "failed_breakout_low": [True, True, False, False],
            "failed_breakout_high": [False, False, True, True],
            "prior_liquidity_low": [98.0, 98.0, 98.0, 98.0],
            "prior_liquidity_high": [102.0, 102.0, 102.0, None],
            "volume_impulse": [True, True, False, True],
            "event_quality_score": [2.0, 2.0, 2.5, 2.5],
        }
    )

    summary = format_strategy_development_summary(
        "TEST diagnostic_spec", _report(), signal_frame, strategy
    )

    assert "failed_breakout=2" in summary
    assert "low=1 high=1" in summary
    assert "after_volume=1" in summary
    assert "after_quality=2" in summary
    assert "after_both=1" in summary


def test_strategy_development_summary_reports_mtf_confirmation_rows():
    strategy = _diagnostic_strategy()
    signal_frame = pl.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "close": [100.0, 99.0, 101.0],
            "high": [101.0, 100.0, 102.0],
            "low": [99.0, 97.0, 100.0],
            "x": [1, 1, 1],
            "entry_signal": [1.0, -1.0, 0.0],
            "liquidity_event_type": [
                "failed_breakout_low",
                "failed_breakout_high",
                "bullish_reclaim",
            ],
            "failed_breakout_low": [True, False, False],
            "failed_breakout_high": [False, True, False],
            "prior_liquidity_low": [98.0, 98.0, 98.0],
            "prior_liquidity_high": [102.0, 102.0, 102.0],
            "volume_impulse": [True, True, False],
            "event_quality_score": [2.0, 2.5, 1.0],
            "m15_confirm_long": [True, False, False],
            "m15_confirm_short": [False, True, False],
            "m15_confirm_reason": ["breakout", "macd", "none"],
            "m15_confirm_available": [True, True, False],
        }
    )
    report = _report(
        [
            _trade(
                side="buy",
                entry_liquidity_event_type="failed_breakout_low",
                entry_mtf_state_key="uptrend|markup|accumulation",
                entry_m15_confirm_available=True,
                entry_m15_confirm_long=True,
                entry_m15_confirm_short=False,
                net_pnl_usd=2.0,
                pnl_usd=2.0,
            ),
            _trade(
                side="sell",
                entry_liquidity_event_type="failed_breakout_high",
                entry_mtf_state_key="downtrend|range|range",
                entry_m15_confirm_available=True,
                entry_m15_confirm_long=False,
                entry_m15_confirm_short=True,
                net_pnl_usd=-1.0,
                pnl_usd=-1.0,
            ),
        ],
        [100.0, 102.0, 101.0],
    )

    summary = format_strategy_development_summary(
        "TEST diagnostic_spec", report, signal_frame, strategy
    )

    assert "MTF confirmation opportunity" in summary
    assert "raw_structural_candidates=2" in summary
    assert "confirm_pass_long=1" in summary
    assert "confirm_pass_short=1" in summary
    assert "MTF confirmation attribution" in summary
    assert "confirmed" in summary
    assert "MTF confirmation reason" in summary
    assert "breakout_confirm=1" in summary
    assert "macd_confirm=1" in summary
    assert "MTF state attribution" in summary
    assert "uptrend|markup|accumulation" in summary
    assert "MTF state x event" in summary
    assert "MTF state separation" in summary
    assert "MTF state cardinality" in summary
    assert "MTF state stability" in summary
    assert "MTF right-edge drift" in summary


def test_strategy_development_summary_reports_unknown_breakdown_and_state_stability():
    strategy = _diagnostic_strategy()
    states = ["uptrend|markup|accumulation"] * 8 + ["uptrend|wide_range|transition"] * 4
    signal_frame = pl.DataFrame(
        {
            "timestamp": [idx * 3_600_000 for idx in range(len(states))],
            "close": [100.0] * len(states),
            "high": [101.0] * len(states),
            "low": [99.0] * len(states),
            "x": [0] * len(states),
            "entry_signal": [0.0] * len(states),
            "market_stage": ["warmup", "wide_range", "trend_continuation", "transition"] * 3,
            "market_stage_reason": [
                "warmup_range_not_ready",
                "wide_range_no_stage",
                "trend_without_range_break",
                "ambiguous_transition",
            ]
            * 3,
            "stage_unknown_reason": ["warmup", "wide_range", "none", "transition"] * 3,
            "mtf_state_key": states,
            "mtf_structure_key": states,
            "mtf_stage_key": states,
        }
    )

    summary = format_strategy_development_summary(
        "TEST diagnostic_spec", _report(), signal_frame, strategy
    )

    assert "Structure stage unknown breakdown" in summary
    assert "warmup" in summary
    assert "wide_range" in summary
    assert "transition" in summary
    assert "MTF state cardinality" in summary
    assert "MTF state stability" in summary


def test_compact_state_diagnostics_summary_and_export_frame():
    states = ["uptrend|range|range"] * 10 + ["downtrend|transition|wide_range"] * 2
    signal_frame = pl.DataFrame(
        {
            "timestamp": [idx * 3_600_000 for idx in range(len(states))],
            "close": [100.0] * len(states),
            "high": [101.0] * len(states),
            "low": [99.0] * len(states),
            "entry_signal": [0.0] * len(states),
            "market_stage": ["range"] * 10 + ["transition", "wide_range"],
            "market_stage_reason": ["compressed_mid_range"] * 10
            + ["ambiguous_transition", "wide_range_no_stage"],
            "stage_unknown_reason": ["none"] * 10 + ["transition", "wide_range"],
            "mtf_state_key": states,
            "mtf_structure_key": states,
            "mtf_stage_key": states,
        }
    )
    report = _report(
        [
            _trade(
                side="buy",
                entry_liquidity_event_type="failed_breakout_low",
                entry_mtf_state_key="uptrend|range|range",
                net_pnl_usd=2.0,
                pnl_usd=2.0,
            )
        ],
        [100.0, 102.0],
        label="TEST compact",
    )

    summary = format_state_diagnostics_summary("TEST compact", report, signal_frame)
    export = state_diagnostics_export_frame("TEST compact", report, signal_frame)

    assert "State diagnostic" in summary
    assert "Structure stage unknown breakdown" in summary
    assert "MTF state x event" in summary
    assert "uptrend|range|range" in summary
    assert export.columns == ["label", "diagnostic", "summary"]
    assert export.filter(pl.col("diagnostic") == "MTF state x event").height == 1


def test_state_profitability_export_flags_actionable_and_high_risk_groups():
    signal_frame = pl.DataFrame({"timestamp": list(range(24)), "mtf_state_key": ["s"] * 24})
    trades = []
    for idx in range(12):
        trades.append(
            _trade(
                side="buy",
                entry_ts=idx,
                entry_liquidity_event_type="failed_breakout_low",
                entry_mtf_state_key="d1_up|h4_range|h1_accumulation",
                entry_mtf_structure_key="uptrend|range|uptrend",
                entry_mtf_stage_key="markup|range|accumulation",
                net_pnl_usd=1.0 if idx < 8 else -0.5,
                pnl_usd=1.0 if idx < 8 else -0.5,
            )
        )
    for idx in range(12):
        trades.append(
            _trade(
                side="sell",
                entry_ts=100 + idx,
                entry_liquidity_event_type="failed_breakout_high",
                entry_mtf_state_key="d1_down|h4_transition|h1_range",
                net_pnl_usd=-1.0,
                pnl_usd=-1.0,
            )
        )
    report = _report(trades, [100.0, 106.0, 94.0], label="TEST state profit")

    export = state_profitability_export_frame("TEST state profit", report, signal_frame)
    summary = format_state_profitability_summary("TEST state profit", report, signal_frame)

    actionable = export.filter(pl.col("actionable"))
    high_risk = export.filter(pl.col("high_risk"))
    assert actionable.height == 1
    assert high_risk.height == 1
    assert actionable["event_type"].to_list() == ["failed_breakout_low"]
    assert high_risk["event_type"].to_list() == ["failed_breakout_high"]
    assert "State profitability coverage" in summary
    assert "Actionable state-event-side" in summary


def test_state_profitability_flags_clustered_and_fragile_positive_groups():
    signal_frame = pl.DataFrame({"timestamp": list(range(12)), "mtf_state_key": ["s"] * 12})
    trades = []
    for idx, pnl in enumerate([10.0, *([-0.1] * 11)]):
        trades.append(
            _trade(
                side="buy",
                entry_ts=idx,
                entry_liquidity_event_type="failed_breakout_low",
                entry_mtf_state_key="d1_up|h4_markup|h1_range",
                net_pnl_usd=pnl,
                pnl_usd=pnl,
            )
        )
    report = _report(trades, [100.0, 108.9], label="TEST fragile")

    export = state_profitability_export_frame("TEST fragile", report, signal_frame)
    row = export.row(0, named=True)

    assert row["fragile"]
    assert row["clustered"]
    assert row["high_risk"]
    assert not row["actionable"]


def test_state_filter_delta_summarizes_removed_and_remaining_buckets():
    baseline = pl.DataFrame(
        [
            {
                "label": "ETH-USDT-SWAP baseline",
                "stage_key": "range|range|range",
                "market_stage": "range",
                "market_stage_reason": "compressed_mid_range",
                "event_type": "failed_breakout_low",
                "side": "long",
                "trades": 5,
                "gross_profit_usd": 1.0,
                "gross_loss_usd": 11.0,
                "net_pnl_usd": -10.0,
            },
            {
                "label": "ETH-USDT-SWAP baseline",
                "stage_key": "range|range|accumulation",
                "market_stage": "accumulation",
                "market_stage_reason": "compressed_near_low",
                "event_type": "failed_breakout_low",
                "side": "long",
                "trades": 3,
                "gross_profit_usd": 9.0,
                "gross_loss_usd": 0.0,
                "net_pnl_usd": 9.0,
            },
        ]
    )
    variant = baseline.filter(pl.col("market_stage") != "range")

    summary = format_state_filter_delta(baseline, variant)

    assert "State filter delta" in summary
    assert "removed bucket" in summary
    assert "ETH-USDT-SWAP range/compressed_mid_range" in summary
    assert "retained bucket" in summary
    assert "remaining loss buckets" in summary


def test_modulation_effect_matrix_flags_global_stable_effects_with_count_thresholds():
    trades = []
    for symbol in ("ETH-USDT-SWAP", "SOL-USDT-SWAP"):
        for _ in range(4):
            trades.append(
                _trade(
                    symbol=symbol,
                    side="buy",
                    entry_market_stage="range",
                    entry_d1_structure_trend_state="uptrend",
                    net_pnl_usd=1.0,
                    pnl_usd=1.0,
                )
            )
        for _ in range(4):
            trades.append(
                _trade(
                    symbol=symbol,
                    side="sell",
                    entry_market_stage="range",
                    entry_d1_structure_trend_state="downtrend",
                    net_pnl_usd=-1.0,
                    pnl_usd=-1.0,
                )
            )
    frame = modulation_effect_matrix(
        pl.DataFrame(trades),
        base_columns=("entry_market_stage_bucket", "side"),
        modulator_columns=("entry_d1_structure_trend_state",),
        min_base_trades=4,
        min_cell_trades=2,
        practical_delta_threshold=0.25,
    )

    global_uptrend = frame.filter(
        (pl.col("symbol") == "ALL")
        & (pl.col("base_feature") == "entry_market_stage_bucket")
        & (pl.col("base_value") == "range")
        & (pl.col("modulator_value") == "uptrend")
    ).row(0, named=True)
    summary = format_modulation_effect_summary(frame)

    assert global_uptrend["base_trades"] == 16
    assert global_uptrend["conditional_trades"] == 8
    assert global_uptrend["min_base_trades"] == 4
    assert global_uptrend["min_cell_trades"] == 2
    assert global_uptrend["significant"]
    assert global_uptrend["stable_across_symbols"]
    assert global_uptrend["classification"] == "global"
    assert "Trade-count sufficiency" in summary
    assert "Global modulation effects" in summary


def test_classifier_config_request_applies_profile_and_explicit_overrides():
    default = ClassifierConfigRequest().to_structure_config()
    assert default.range_width_threshold.mode == "rolling_quantile"

    fixed = ClassifierConfigRequest(
        profile="rolling",
        range_threshold_mode="fixed",
        fixed_range_width_atr=6.5,
        swing_lookback=7,
        range_lookback=60,
        trend_window=18,
        level_proximity_atr=0.75,
    )
    fixed = fixed.to_structure_config()

    assert fixed.range_width_threshold.mode == "fixed"
    assert fixed.range_width_threshold.fixed_atr_max == pytest.approx(6.5)
    assert fixed.swing_lookback == 7
    assert fixed.range_lookback == 60
    assert fixed.trend_window == 18
    assert fixed.level_proximity_atr == pytest.approx(0.75)


def test_market_state_config_cost_linked_threshold():
    fixed_threshold, fixed_mode = MarketStateConfig(delta_threshold_pct=0.25).delta_threshold()
    cost_threshold, cost_mode = MarketStateConfig(
        delta_mode="cost_multiple",
        cost_pct=0.10,
        cost_multiple=2.0,
    ).delta_threshold()

    assert fixed_threshold == pytest.approx(0.25)
    assert fixed_mode == "fixed"
    assert cost_threshold == pytest.approx(0.20)
    assert "cost_multiple" in cost_mode


def test_modulation_effect_matrix_marks_sparse_cells_insufficient_and_normalizes_side():
    frame = modulation_effect_matrix(
        pl.DataFrame(
            [
                _trade(
                    symbol="BTC-USDT-SWAP",
                    side="buy",
                    entry_market_stage_bucket="range",
                    entry_d1_market_stage="markup",
                    net_pnl_usd=1.0,
                    pnl_usd=1.0,
                ),
                _trade(
                    symbol="BTC-USDT-SWAP",
                    side="sell",
                    entry_market_stage_bucket="range",
                    entry_d1_market_stage="markdown",
                    net_pnl_usd=-1.0,
                    pnl_usd=-1.0,
                ),
            ]
        ),
        base_columns=("side",),
        modulator_columns=("entry_d1_market_stage",),
        min_base_trades=3,
        min_cell_trades=2,
    )

    assert set(frame["base_value"].to_list()) == {"long", "short"}
    assert set(frame["classification"].to_list()) == {"insufficient"}
    assert not any(frame["sufficient_base"].to_list())


def test_market_state_forward_frame_computes_forward_path_metrics():
    frame = pl.DataFrame(
        {
            "timestamp": list(range(6)),
            "open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "atr_14": [2.0] * 6,
        }
    )

    forward = market_state_forward_frame(frame, symbol="TEST", horizons=(3,))
    row = forward.row(0, named=True)

    assert row["symbol"] == "TEST"
    assert row["fwd_3_return_pct"] == pytest.approx(3.0)
    assert row["fwd_3_return_atr"] == pytest.approx(1.5)
    assert row["fwd_3_mfe_long_atr"] == pytest.approx(2.0)
    assert row["fwd_3_mae_long_atr"] == pytest.approx(0.0)
    assert row["fwd_3_mfe_short_atr"] == pytest.approx(0.0)
    assert row["fwd_3_mae_short_atr"] == pytest.approx(2.0)
    assert row["fwd_3_direction"] == "up"
    assert forward.tail(3)["fwd_3_return_pct"].null_count() == 3


def test_market_state_forward_summary_filters_incomplete_future_rows():
    forward = market_state_forward_frame(
        pl.DataFrame(
            {
                "timestamp": list(range(6)),
                "high": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
                "low": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0],
                "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
                "atr_14": [2.0] * 6,
                "market_stage": ["markup"] * 6,
            }
        ),
        symbol="TEST",
        horizons=(3,),
    )

    summary = market_state_forward_summary(
        forward,
        group_columns=("market_stage",),
        horizon=3,
        min_rows=3,
    )
    row = summary.row(0, named=True)

    assert row["artifact"] == "forward-summary"
    assert row["rows"] == 3
    assert row["sufficient_rows"]
    assert row["directional_bias"] == "up"
    assert row["effective_rows"] == pytest.approx(3.0)
    assert row["overlap_lag"] == 2
    assert row["overlap_warning"] == "overlapping_forward_windows"
    assert "Market-state forward diagnostics" in format_market_state_forward_summary(summary)


def test_market_state_reduction_preserves_raw_labels_and_projects_semantics():
    frame = add_market_state_reductions(
        pl.DataFrame(
            {
                "market_stage": ["wide_range", "trend_continuation", "range"],
                "market_stage_reason": [
                    "wide_range_no_stage",
                    "trend_without_range_break",
                    "compressed_mid_range",
                ],
            }
        )
    )

    assert frame["market_stage"].to_list() == ["wide_range", "trend_continuation", "range"]
    assert frame["market_stage_reduced"].to_list() == [
        "wide_range",
        "trend_continuation",
        "range",
    ]


def test_market_state_defaults_prefer_reduced_stage():
    from qooi.research.diagnostics import MARKET_STATE_BASE_COLUMNS

    assert "market_stage_reduced" in MARKET_STATE_BASE_COLUMNS
    assert "market_stage_reason" not in MARKET_STATE_BASE_COLUMNS


def test_state_tradability_scores_persistent_state_above_churning_state():
    states = ["stable"] * 12 + ["a", "b"] * 6
    close = [100.0 + idx for idx in range(len(states))]
    frame = pl.DataFrame(
        {
            "close": close,
            "market_stage_reduced": states,
            "symbol": ["TEST"] * len(states),
        }
    )

    out = state_tradability_frame(
        frame,
        TradabilityConfig(state_columns=("market_stage_reduced",), min_rows=4),
    )
    stable = out.filter(pl.col("state_value") == "stable").row(0, named=True)
    churning = out.filter(pl.col("state_value") == "a").row(0, named=True)

    assert stable["artifact"] == "state-tradability"
    assert 0.0 <= stable["eti"] <= 1.0
    assert stable["self_transition_pct"] > churning["self_transition_pct"]
    assert stable["median_dwell_bars"] > churning["median_dwell_bars"]


def test_classifier_validity_frame_reports_separation_and_enrichment():
    frame = pl.DataFrame(
        {
            "structure_trend_state": ["range"] * 8 + ["uptrend"] * 8,
            "market_stage": ["range"] * 8 + ["markup"] * 8,
            "market_stage_reason": ["compressed_mid_range"] * 8 + ["markup_breakout"] * 8,
            "structure_reason": ["compressed_range"] * 16,
            "stage_unknown_reason": ["none"] * 16,
            "atr_14": [1.0] * 8 + [4.0] * 8,
            "adx_14": [10.0] * 8 + [40.0] * 8,
            "liquidity_event_type": ["none"] * 8 + ["breakout_acceptance_high"] * 8,
        }
    )

    validity = classifier_validity_frame(frame)

    assert "classifier-validity" in set(validity["artifact"].to_list())
    assert not validity.filter(pl.col("check") == "atr_14_state_separation").is_empty()
    assert not validity.filter(pl.col("check") == "liquidity_event_stage_enrichment").is_empty()


def test_overlap_adjusted_standard_error_shrinks_effective_rows_for_positive_autocorrelation():
    values = [1.0, 1.1, 1.2, 1.1, 1.0, -1.0, -1.1, -1.2, -1.1, -1.0]

    assert effective_sample_size([1.0] * 5, max_lag=3) == pytest.approx(5.0)
    assert effective_sample_size(values, max_lag=3) < len(values)
    assert adjusted_standard_error(values, max_lag=3) >= adjusted_standard_error(values, max_lag=0)


def test_market_state_modulation_matrix_detects_deterministic_effect():
    rows = []
    for symbol in ("ETH-USDT-SWAP", "SOL-USDT-SWAP"):
        for _ in range(6):
            rows.append(
                {
                    "symbol": symbol,
                    "market_stage": "range",
                    "d1_market_stage": "markup",
                    "fwd_10_return_pct": 1.0,
                }
            )
        for _ in range(6):
            rows.append(
                {
                    "symbol": symbol,
                    "market_stage": "range",
                    "d1_market_stage": "markdown",
                    "fwd_10_return_pct": -1.0,
                }
            )
    matrix = market_state_modulation_matrix(
        pl.DataFrame(rows),
        base_columns=("market_stage",),
        modulator_columns=("d1_market_stage",),
        min_base_rows=4,
        min_cell_rows=2,
        practical_delta_threshold=0.25,
        min_segment_base_rows=4,
        min_segment_cell_rows=2,
    )

    markup = matrix.filter(
        (pl.col("symbol") == "ALL") & (pl.col("modulator_value") == "markup")
    ).row(0, named=True)

    assert markup["artifact"] == "market-state-modulation"
    assert markup["base_rows"] == 24
    assert markup["conditional_rows"] == 12
    assert markup["significant"]
    assert markup["stable_across_symbols"]
    assert markup["classification"] == "global"
    assert markup["effective_base_rows"] <= markup["base_rows"]
    assert markup["overlap_warning"] == "overlapping_forward_windows"
    assert markup["sufficient_time_splits"] >= 2
    assert markup["meta_stable"]
    assert markup["outcome_kind"] == "return_pct"
    assert markup["delta_cohens_d"] > 0.0
    assert markup["effect_size_material"]
    assert markup["time_stable"]
    assert markup["cross_asset_homogeneous"]


def test_market_state_modulation_matrix_adds_robust_fdr_fields():
    rows = []
    for idx in range(80):
        rows.append(
            {
                "timestamp": idx,
                "symbol": "BTC-USDT-SWAP" if idx < 40 else "ETH-USDT-SWAP",
                "market_stage": "range",
                "d1_market_stage": "markup" if idx % 2 == 0 else "markdown",
                "fwd_3_return_pct": 1.0 if idx % 2 == 0 else -1.0,
                "fwd_3_return_atr": 0.5 if idx % 2 == 0 else -0.5,
            }
        )

    matrix = market_state_modulation_matrix(
        pl.DataFrame(rows),
        base_columns=("market_stage",),
        modulator_columns=("d1_market_stage",),
        outcome_column="fwd_3_return_pct",
        min_base_rows=20,
        min_cell_rows=10,
        practical_delta_threshold=0.25,
        robustness=ModulationRobustnessConfig(se_method="newey_west", fdr=True),
    )
    markup = matrix.filter(
        (pl.col("symbol") == "ALL") & (pl.col("modulator_value") == "markup")
    ).row(0, named=True)

    assert markup["se_method"] == "newey_west"
    assert markup["robust_significant"]
    assert markup["fdr_significant"]
    assert markup["delta_return_atr"] > 0.0
    assert markup["delta_q10"] > 0.0


def test_market_state_modulation_matrix_marks_sparse_cells_insufficient():
    matrix = market_state_modulation_matrix(
        pl.DataFrame(
            {
                "symbol": ["BTC-USDT-SWAP"],
                "market_stage": ["range"],
                "h4_market_stage": ["markup"],
                "fwd_10_return_pct": [1.0],
            }
        ),
        base_columns=("market_stage",),
        modulator_columns=("h4_market_stage",),
        min_base_rows=3,
        min_cell_rows=2,
    )

    assert matrix.row(0, named=True)["classification"] == "insufficient"


def test_market_state_plot_helpers_write_artifacts(tmp_path):
    frame = pl.DataFrame(
        {
            "base_feature": ["market_stage", "market_stage"],
            "base_value": ["range", "range"],
            "modulator": ["d1_market_stage", "d1_market_stage"],
            "modulator_value": ["markup", "markdown"],
            "outcome_kind": ["return_pct", "return_pct"],
            "horizon": [3, 5],
            "conditional_rows": [30, 25],
            "delta_return_pct": [0.2, -0.1],
            "delta_cohens_d": [0.4, -0.2],
        }
    )

    heatmap = plot_market_state_modulation_heatmap(frame, output_path=tmp_path / "heatmap.svg")
    decay = plot_market_state_horizon_decay(frame, output_path=tmp_path / "decay.svg")

    assert heatmap.exists()
    assert decay.exists()


def test_classifier_health_frame_flags_missing_and_contradictory_columns():
    missing = classifier_health_frame(pl.DataFrame({"market_stage": ["range"]}))
    missing_required = missing.filter(pl.col("health_check") == "required_classifier_columns").row(
        0, named=True
    )
    assert missing_required["status"] == "fail"

    contradictory = classifier_health_frame(
        pl.DataFrame(
            {
                "structure_trend_state": ["range"],
                "market_stage": ["warmup"],
                "structure_reason": ["compressed_range"],
                "market_stage_reason": ["compressed_mid_range"],
                "stage_unknown_reason": ["none"],
            }
        )
    )

    assert (
        contradictory.filter(pl.col("health_check") == "contradiction_count").row(0, named=True)[
            "status"
        ]
        == "fail"
    )


def test_research_cli_accepts_market_state_forward_config(monkeypatch, tmp_path):
    import importlib.util
    from pathlib import Path

    config_path = tmp_path / "market-state.toml"
    config_path.write_text(
        """
[diagnostics]
mode = "market-state-forward"

[market_state]
horizons = [3, 5]
delta_mode = "cost_multiple"
base_columns = ["market_stage", "atr_percentile_bucket"]
outcomes = ["return_pct", "return_atr"]
plot_dir = "F:/Stratum/TEMP/kilo/plots"

[market_state.robustness]
se_method = "newey_west"
fdr = true

[classifier]
range_threshold_mode = "fixed"
""",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location(
        "research_cli", Path(__file__).parents[1] / "scripts" / "research.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(
        "sys.argv",
        [
            "research.py",
            "--config",
            str(config_path),
        ],
    )

    command = module._command_config(module._parse_args())

    assert command.diagnostics.mode == "market-state-forward"
    assert command.market_state.horizons == (3, 5)
    assert command.classifier.range_threshold_mode == "fixed"
    assert command.market_state.delta_mode == "cost_multiple"
    assert command.market_state.base_columns == ("market_stage", "atr_percentile_bucket")
    assert command.market_state.outcomes == ("return_pct", "return_atr")
    assert command.market_state.robustness.se_method == "newey_west"
    assert command.market_state.robustness.fdr
    assert command.market_state.plot_dir.endswith("plots")


def test_state_stability_distinguishes_right_edge_drift():
    strategy = _diagnostic_strategy()
    stable = ["uptrend|markup|accumulation"] * 20
    churning = ["a|b|c", "a|b|d"] * 6
    states = stable + churning
    signal_frame = pl.DataFrame(
        {
            "timestamp": [idx * 3_600_000 for idx in range(len(states))],
            "close": [100.0] * len(states),
            "high": [101.0] * len(states),
            "low": [99.0] * len(states),
            "x": [0] * len(states),
            "entry_signal": [0.0] * len(states),
            "mtf_state_key": states,
        }
    )

    summary = format_strategy_development_summary(
        "TEST diagnostic_spec", _report(), signal_frame, strategy
    )

    assert "MTF right-edge drift" in summary
    assert "right_edge_drift" in summary


def test_state_operability_flags_outlier_dominated_positive_expectancy():
    strategy = _diagnostic_strategy()
    trades = [
        _trade(
            side="buy",
            entry_liquidity_event_type="failed_breakout_low",
            entry_mtf_state_key="uptrend|markup|accumulation",
            pnl_usd=-1.0,
            net_pnl_usd=-1.0,
        )
        for _ in range(9)
    ]
    trades.append(
        _trade(
            side="buy",
            entry_liquidity_event_type="failed_breakout_low",
            entry_mtf_state_key="uptrend|markup|accumulation",
            pnl_usd=20.0,
            net_pnl_usd=20.0,
        )
    )
    signal_frame = pl.DataFrame(
        {
            "timestamp": [1],
            "close": [100.0],
            "high": [101.0],
            "low": [99.0],
            "x": [0],
            "entry_signal": [0.0],
        }
    )

    summary = format_strategy_development_summary(
        "TEST diagnostic_spec", _report(trades, [100.0, 111.0]), signal_frame, strategy
    )

    assert "MTF state operability" in summary
    assert "fragile_positive" in summary


def test_state_pnl_time_consistency_flags_time_concentration():
    strategy = _diagnostic_strategy()
    day_ms = 86_400_000
    trades = [
        _trade(
            side="buy",
            entry_ts=idx * day_ms,
            entry_liquidity_event_type="failed_breakout_low",
            entry_mtf_state_key="uptrend|markup|accumulation",
            pnl_usd=5.0 if idx < 2 else -0.5,
            net_pnl_usd=5.0 if idx < 2 else -0.5,
        )
        for idx in range(12)
    ]
    signal_frame = pl.DataFrame(
        {
            "timestamp": [1],
            "close": [100.0],
            "high": [101.0],
            "low": [99.0],
            "x": [0],
            "entry_signal": [0.0],
        }
    )

    summary = format_strategy_development_summary(
        "TEST diagnostic_spec", _report(trades, [100.0, 105.0]), signal_frame, strategy
    )

    assert "MTF state PnL time consistency" in summary
    assert "time_concentrated=1" in summary


def test_strategy_development_summary_structural_rows_degrade_without_event_columns():
    strategy = _diagnostic_strategy()
    signal_frame = pl.DataFrame(
        {
            "timestamp": [1],
            "close": [100.0],
            "high": [101.0],
            "low": [99.0],
            "x": [0],
            "entry_signal": [0.0],
        }
    )

    summary = format_strategy_development_summary(
        "TEST diagnostic_spec", _report(), signal_frame, strategy
    )

    assert "Structural event opportunity" in summary
    assert "failed_breakout=0" in summary
    assert "Structural event quality" in summary
    assert "n/a" in summary
    assert "Reclaim/sweep extension" in summary
    assert "disabled" in summary


def test_signal_debug_filters_suppress_side_without_mutating_frame():
    frame = pl.DataFrame(
        {
            "entry_signal": [1.0, -1.0, 0.0],
            "raw_entry_signal": [1.0, -1.0, 0.0],
            "position_signal": [1.0, -1.0, -1.0],
            "signal": [1.0, -1.0, -1.0],
            "exit_signal": [False, True, True],
            "signal_id": ["long_rule", "short_rule", "short_rule"],
        }
    )
    config = _command(long_only=True)

    filtered = apply_signal_debug_filters(frame, config.signal_filters)

    assert filtered["entry_signal"].to_list() == [1.0, 0.0, 0.0]
    assert filtered["position_signal"].to_list() == [1.0, 0.0, 0.0]
    assert filtered["exit_signal"].to_list() == [False, True, True]
    assert frame["entry_signal"].to_list() == [1.0, -1.0, 0.0]


def test_signal_debug_filters_include_and_exclude_signal_ids():
    frame = pl.DataFrame(
        {
            "entry_signal": [1.0, -1.0, 1.0],
            "raw_entry_signal": [1.0, -1.0, 1.0],
            "position_signal": [1.0, -1.0, 1.0],
            "signal": [1.0, -1.0, 1.0],
            "exit_signal": [False, False, False],
            "signal_id": ["a", "b", "c"],
        }
    )

    included = apply_signal_debug_filters(frame, _command(include_signal_id="a,c").signal_filters)
    excluded = apply_signal_debug_filters(frame, _command(exclude_signal_id="b").signal_filters)

    assert included["entry_signal"].to_list() == [1.0, 0.0, 1.0]
    assert included["position_signal"].to_list() == [1.0, -1.0, 1.0]
    assert excluded["entry_signal"].to_list() == [1.0, 0.0, 1.0]


def test_signal_debug_filter_metadata_is_recorded():
    config = _command(short_only=True, exclude_signal_id="long_rule")

    metadata = config.metadata()

    assert "signal_debug_filter=active" in metadata
    assert "signal_side=short" in metadata
    assert "exclude_signal_id=long_rule" in metadata
