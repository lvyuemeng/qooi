"""Reduced research orchestration tests."""

from pathlib import Path

from qooi.core.basket import ExitConfig
from qooi.core.config import CORE_UNIVERSE
from qooi.research.config import ResearchCommandConfig, apply_sizing_overrides
from qooi.research.data import CacheAuditRequest, build_history_refresh_requests
from qooi.strategies.catalog import BENCHMARK_GROUPS, strategy_selection
from qooi.strategies.specs import (
    structure_event_reversal_v1_spec,
    structure_event_trend_aligned_v1_spec,
)


def _command(**overrides):
    config = ResearchCommandConfig()
    run_updates = {
        key: overrides[key] for key in ("profile", "universe", "ds", "symbol") if key in overrides
    }
    req_updates = {key: overrides[key] for key in ("days", "min", "cov") if key in overrides}
    strategy_updates = {
        key: overrides[key]
        for key in ("strategy", "benchmark", "benchmark_group")
        if key in overrides
    }
    sizing_updates = {
        key: overrides[key]
        for key in ("risk_pct", "max_notional_pct", "leverage", "capital", "min_contracts")
        if key in overrides
    }
    if "normalize_sizing" in overrides:
        sizing_updates["normalize"] = overrides["normalize_sizing"]
    if run_updates:
        config = config.model_copy(update={"run": config.run.model_copy(update=run_updates)})
    if req_updates:
        config = config.model_copy(update={"req": config.req.model_copy(update=req_updates)})
    if strategy_updates:
        config = config.model_copy(
            update={"strategy": config.strategy.model_copy(update=strategy_updates)}
        )
    if sizing_updates:
        config = config.model_copy(
            update={"sizing": config.sizing.model_copy(update=sizing_updates)}
        )
    return ResearchCommandConfig.model_validate(config.model_dump())


def test_strategy_registry_builds_baselines():
    command = _command(benchmark=True)
    selection = strategy_selection(
        command.strategy.strategies,
        benchmark=command.strategy.benchmark,
        benchmark_group=command.strategy.benchmark_group,
        default=command.strategy.strategy,
    )

    assert [strategy.name for strategy in selection.strategies] == [
        "ema_trend_baseline",
        "rsi_bounce_reversion",
        "rsi_macd_trend",
        "momentum_burst",
    ]


def test_strategy_registry_builds_structural_variants():
    assert structure_event_reversal_v1_spec().name == "structure_event_reversal_v1"
    assert structure_event_trend_aligned_v1_spec().name == "structure_event_trend_aligned_v1"


def test_strategy_registry_rejects_unknown_name():
    try:
        _command(strategy="missing")
    except ValueError as exc:
        assert "strategy.strategy='missing'" in str(exc)
    else:
        raise AssertionError("missing strategy did not raise")


def test_strategy_selection_supports_benchmark_group():
    command = _command(benchmark=True)
    assert (
        strategy_selection(
            command.strategy.strategies,
            benchmark=command.strategy.benchmark,
            benchmark_group=command.strategy.benchmark_group,
            default=command.strategy.strategy,
        ).names
        == BENCHMARK_GROUPS["baselines"]
    )


def test_cache_audit_requests_include_base_and_higher_context_targets():
    command = _command(strategy="structure_event_trend_aligned_v1", days=730, min=12000)
    pair = CORE_UNIVERSE[0]

    requests = build_history_refresh_requests(
        CacheAuditRequest(
            pairs=(pair,),
            data_source=command.run.ds,
            days=command.days,
            min_bars=command.min_bars,
            min_coverage_pct=command.min_coverage_pct,
            refresh=True,
            incremental=True,
        )
    )
    by_bar = {(request.inst_id, request.bar): request for request in requests}

    assert (pair.asset.symbol, "1H") in by_bar
    assert by_bar[(pair.asset.symbol, "4H")].min_bars == 4_630
    assert by_bar[(pair.asset.symbol, "1D")].min_bars == 980


def test_safe_profile_enables_normalized_sizing_and_gates():
    config = _command(profile="safe")

    assert config.sizing_overrides.normalize is True
    assert config.risk_gates.min_pf == 1.10
    assert config.max_per_strategy_symbol == 1


def test_research_profile_keeps_default_basket_stacking_cap():
    assert _command(profile="research").max_per_strategy_symbol == 3


def test_sizing_override_does_not_mutate_global_pair():
    pair = CORE_UNIVERSE[0]
    updated = apply_sizing_overrides(
        pair, _command(normalize_sizing=True, risk_pct=0.03).sizing_overrides
    )

    assert updated is not pair
    assert updated.asset.max_risk_pct == 0.03
    assert pair.asset.max_risk_pct == CORE_UNIVERSE[0].asset.max_risk_pct


def test_exit_config_is_canonical_runtime_shape():
    assert isinstance(_command().exit, ExitConfig)


def test_removed_research_api_wrappers_are_absent():
    root = Path(__file__).resolve().parents[1]
    learned = (root / "scripts" / "learned_states.py").read_text(encoding="utf-8")
    reports = (root / "src" / "qooi" / "research" / "reports.py").read_text(encoding="utf-8")
    data = (root / "src" / "qooi" / "research" / "data.py").read_text(encoding="utf-8")
    config = (root / "src" / "qooi" / "research" / "config.py").read_text(encoding="utf-8")

    assert "RunCtx" not in learned
    assert "def pipe" not in learned
    assert "resolve_flow" not in learned
    assert "prepare_backtest_frame" not in data
    assert "_attach_strategy_context" not in data
    assert "_context_bundle_for_strategy" not in data
    assert "exit_config_from_command" not in reports
    assert "backtest_frame_options_from_command" not in reports
    assert "strategy_selection_from_config" not in reports
    assert "ExitConfigRequest" not in config
