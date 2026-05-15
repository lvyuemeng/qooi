"""Research backtest orchestration tests."""

from argparse import Namespace

from qooi.core.config import PAIRS
from qooi.core.evaluate import Report
from qooi.research.config import apply_sizing_overrides, resolve_config, selected_pairs
from qooi.research.diagnostics import report_status
from qooi.research.strategies import (
    BENCHMARK_GROUPS,
    build_strategies,
    selected_strategy_names,
    strategy_from_name,
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
        "fail_on_risk": False,
        "normalize_sizing": False,
        "risk_pct": None,
        "max_notional_pct": None,
        "leverage": None,
        "capital": None,
        "min_contracts": None,
        "universe": "core",
        "data_source": "swap",
        "style": "single",
        "symbol": "",
        "exclude_symbol": "",
        "strategy": "zscore_mean_reversion",
        "strategies": "",
        "benchmark": False,
        "benchmark_group": "zscore-family",
        "entry_z": 2.0,
        "exit_z": 0.25,
        "z_period": 20,
        "ewma_span": 48,
        "robust_period": 96,
        "volatility_ratio_max": 2.5,
        "adx_max": 25.0,
        "mom_threshold": 0.003,
        "trend_maturity": 12,
        "volume_mult": 1.1,
        "adx_threshold": 15.0,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def test_strategy_registry_builds_z_family():
    args = _args(entry_z=2.25, robust_period=120)

    selection = build_strategies(BENCHMARK_GROUPS["zscore-family"], args)

    assert [strategy.name for strategy in selection.strategies] == [
        "zscore_mean_reversion",
        "adaptive_zscore_mean_reversion",
        "robust_zscore_mean_reversion",
    ]


def test_strategy_registry_rejects_unknown_name():
    try:
        strategy_from_name("missing", _args())
    except ValueError as exc:
        assert "Unknown strategy" in str(exc)
    else:
        raise AssertionError("missing strategy did not raise")


def test_selected_strategy_names_supports_comma_list_and_benchmark_group():
    assert selected_strategy_names(
        _args(strategies="zscore_mean_reversion,robust_zscore_mean_reversion")
    ) == ("zscore_mean_reversion", "robust_zscore_mean_reversion")
    assert selected_strategy_names(_args(benchmark=True)) == BENCHMARK_GROUPS["zscore-family"]


def test_safe_profile_enables_normalized_sizing_and_gates():
    config = resolve_config(_args(profile="safe"))

    assert config.sizing.normalize is True
    assert config.sizing.risk_pct == 0.02
    assert config.sizing.max_notional_pct == 1.0
    assert config.risk_gates.min_pf == 1.10
    assert config.risk_gates.max_notional_exposure_pct == 200.0


def test_sizing_override_does_not_mutate_global_pair():
    pair = PAIRS[0]
    config = resolve_config(_args(normalize_sizing=True, risk_pct=0.03))

    updated = apply_sizing_overrides(pair, config.sizing)

    assert updated is not pair
    assert updated.asset.max_risk_pct == 0.03
    assert pair.asset.max_risk_pct == PAIRS[0].asset.max_risk_pct


def test_selected_pairs_filters_symbol_and_exclusion():
    config = resolve_config(_args(symbol="ETH-USDT-SWAP", exclude_symbol="XAU-USDT-SWAP"))

    pairs = selected_pairs(_args(symbol="ETH-USDT-SWAP", exclude_symbol="XAU-USDT-SWAP"), config)

    assert [pair.asset.symbol for pair in pairs] == ["ETH-USDT-SWAP"]


def test_report_status_flags_layer_four_risk_failures():
    trades = [
        {"pnl": -0.01, "pnl_usd": -1.0},
        {"pnl": -0.02, "pnl_usd": -2.0},
    ]
    report = Report.from_raw(
        trades,
        [100.0, 90.0, 80.0],
        PAIRS[0],
        active_exposure=[0.0, 1.0, 1.0],
        timestamps=[1, 2, 3],
        signals=[0.0, 1.0, 1.0],
    )
    gates = resolve_config(_args(profile="safe")).risk_gates

    status = report_status(report, gates)

    assert status.status == "FAIL"
    assert "PF_LOW" in status.reasons or "EXP_LOW" in status.reasons
