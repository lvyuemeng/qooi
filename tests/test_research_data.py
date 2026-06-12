from __future__ import annotations

import polars as pl
import pytest

from qooi.core.config import RESEARCH_UNIVERSE
from qooi.exchange.store import HistoryCoverage, HistoryTarget
from qooi.research.config import ResearchCommandConfig, resolve_research_outputs
from qooi.research.data import DEFAULT_CONTEXTS, FrameRequest, load_frame, prepare_classifier_frame
from qooi.strategies.structure import StructureClassifierConfig


def _frame(rows: int = 240, step_ms: int = 3_600_000) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [idx * step_ms for idx in range(rows)],
            "open": [100.0 + (idx % 3) for idx in range(rows)],
            "high": [102.0 + (idx % 5) for idx in range(rows)],
            "low": [98.0 - (idx % 4) for idx in range(rows)],
            "close": [100.5 + (idx % 2) for idx in range(rows)],
            "vol": [1000.0] * rows,
        }
    )


def _coverage(inst_id: str, bar: str, rows: int) -> HistoryCoverage:
    target = HistoryTarget(inst_id, bar, "trade", 10, 10, 10, rows, 0)
    return HistoryCoverage(inst_id, bar, "trade", target, rows, 0, rows, 0, 0, 0.0, 100.0)


class FakeStore:
    def __init__(self) -> None:
        self.requests = []

    def bars(self, request):
        self.requests.append(request)
        if request.bar == "4H":
            return _frame(90, 4 * 3_600_000), _coverage(request.inst_id, request.bar, 90)
        if request.bar == "1D":
            return _frame(60, 24 * 3_600_000), _coverage(request.inst_id, request.bar, 60)
        if request.bar == "15m":
            return _frame(240, 15 * 60_000), _coverage(request.inst_id, request.bar, 240)
        return _frame(), _coverage(request.inst_id, request.bar, 240)


def _command():
    config = ResearchCommandConfig()
    return config.model_copy(
        update={
            "run": config.run.model_copy(update={"profile": "smoke"}),
            "req": config.req.model_copy(update={"days": 10, "min": 10, "cov": 0.0}),
        }
    )


def _request(pair, command: ResearchCommandConfig) -> FrameRequest:
    return FrameRequest(
        pair=pair,
        data_source=command.run.ds,
        bar=pair.asset.timeframe,
        days=command.days,
        min_bars=command.min_bars,
        refresh=command.req.refresh,
        min_coverage_pct=command.min_coverage_pct,
        allow_swap_signal_fallback=command.run.allow_swap_signal_fallback,
    )


def test_prepare_classifier_frame_does_not_run_backtest():
    command = _command()

    prepared = prepare_classifier_frame(
        FakeStore(),
        _request(RESEARCH_UNIVERSE[0], command),
        StructureClassifierConfig.fixed(),
        contexts=DEFAULT_CONTEXTS,
    )

    assert "market_stage" in prepared.frame.columns
    assert "h4_market_stage" in prepared.frame.columns
    assert "d1_market_stage" in prepared.frame.columns
    assert "mtf_state_key" in prepared.frame.columns
    assert "range_width_atr_threshold" in prepared.frame.columns
    assert "h4_range_width_atr_threshold" in prepared.frame.columns


def test_prepare_classifier_frame_is_deterministic():
    command = _command()
    pair = RESEARCH_UNIVERSE[0]

    first = prepare_classifier_frame(
        FakeStore(),
        _request(pair, command),
        StructureClassifierConfig.fixed(),
        contexts=DEFAULT_CONTEXTS,
    )
    second = prepare_classifier_frame(
        FakeStore(),
        _request(pair, command),
        StructureClassifierConfig.fixed(),
        contexts=DEFAULT_CONTEXTS,
    )

    assert first.signal_inst_id == second.signal_inst_id
    assert first.frame.select("mtf_state_key", "h4_market_stage", "d1_market_stage").equals(
        second.frame.select("mtf_state_key", "h4_market_stage", "d1_market_stage")
    )


def test_prepare_classifier_frame_uses_requested_bar_without_default_context():
    command = _command()
    pair = RESEARCH_UNIVERSE[0]
    request = _request(pair, command)
    request = FrameRequest(
        pair=pair,
        data_source=request.data_source,
        bar="15m",
        days=request.days,
        min_bars=request.min_bars,
        refresh=request.refresh,
        min_coverage_pct=request.min_coverage_pct,
        allow_swap_signal_fallback=request.allow_swap_signal_fallback,
    )
    store = FakeStore()

    prepared = prepare_classifier_frame(store, request, StructureClassifierConfig.fixed())

    assert store.requests[0].bar == "15m"
    assert "timeframe" in prepared.frame.columns
    assert prepared.frame.select("timeframe").item(0, 0) == "15m"
    assert "h4_market_stage" not in prepared.frame.columns
    assert "d1_market_stage" not in prepared.frame.columns
    assert "mtf_state_key" not in prepared.frame.columns


def test_req_replaces_cache_and_rejects_old_cache_shape() -> None:
    command = ResearchCommandConfig.model_validate(
        {"req": {"days": 20, "min": 12, "cap": 0, "trim": False, "cov": 0}}
    )

    assert command.days == 20
    assert command.min_bars == 12
    with pytest.raises(ValueError):
        ResearchCommandConfig.model_validate({"cache": {"days": 20}})


def test_load_frame_treats_min_as_floor_and_cap_only_when_trimmed() -> None:
    command = ResearchCommandConfig.model_validate(
        {"req": {"days": 10, "min": 10, "cap": 100, "trim": False, "cov": 0}}
    )

    full, _coverage = load_frame(FakeStore(), "BTC-USDT-SWAP", "1H", command.req)
    trimmed, _coverage = load_frame(
        FakeStore(),
        "BTC-USDT-SWAP",
        "1H",
        command.req.model_copy(update={"trim": True}),
    )

    assert full.height == 240
    assert trimmed.height == 100


def test_timeframe_config_resolves_defaults_and_overrides():
    config = ResearchCommandConfig.model_validate(
        {
            "timeframes": {
                "bars": ["15m", "1H", "1H"],
                "specs": [
                    {"bar": "15m", "min_bars": 120, "liquidity_lookback": 44},
                    {"bar": "1H", "horizons": [2, 4, 4]},
                ],
            }
        }
    )

    specs = config.timeframes.resolved_specs(config)

    assert [spec.bar for spec in specs] == ["15m", "1H"]
    assert specs[0].horizons == (4, 8, 16)
    assert specs[0].min_bars == 120
    assert specs[0].liquidity_lookback == 44
    assert specs[1].horizons == (2, 4)


def test_diagnostic_mode_accepts_only_backtest_and_research_evaluation():
    assert ResearchCommandConfig.model_validate({"diagnostics": {"mode": "backtest"}})
    assert ResearchCommandConfig.model_validate({"diagnostics": {"mode": "research-evaluation"}})
    for mode in (
        "classifier",
        "state",
        "state-profitability",
        "state-filter-delta",
        "modulation-effect",
        "market-state-forward",
        "tradability",
    ):
        with pytest.raises(ValueError):
            ResearchCommandConfig.model_validate({"diagnostics": {"mode": mode}})


def test_pattern_quality_config_defaults_and_dynamic_transition_validates():
    default = ResearchCommandConfig()
    assert default.research_evaluation.outputs == (
        "timeframe-classifier",
        "dynamic-transition-discovery",
        "pattern-quality",
    )
    assert default.research_evaluation.include_backtest_report is False
    assert default.research_evaluation.pattern_quality.enabled is False
    assert default.research_evaluation.dynamic_transition_discovery.enabled is False

    config = ResearchCommandConfig.model_validate(
        {
            "research_evaluation": {
                "outputs": ["dynamic-transition-discovery", "pattern-quality"],
                "dynamic_transition_discovery": {
                    "enabled": True,
                    "min_rows": 12,
                    "ngram_lengths": [2, 3, 3],
                },
                "pattern_quality": {"enabled": True, "min_rows": 12},
            }
        }
    )

    assert config.research_evaluation.outputs == (
        "dynamic-transition-discovery",
        "pattern-quality",
    )
    assert config.research_evaluation.dynamic_transition_discovery.enabled is True
    assert config.research_evaluation.dynamic_transition_discovery.ngram_lengths == (2, 3, 3)
    assert config.research_evaluation.pattern_quality.enabled is True


def test_research_evaluation_rejects_removed_outputs():
    removed = [
        "classifier",
        "joint-forward-quality",
        "tradability",
        "market-state-forward",
        "market-state-modulation",
        "timeframe-tradability",
        "timeframe-forward-quality",
        "resonance-candidates",
    ]

    for output in removed:
        with pytest.raises(ValueError):
            ResearchCommandConfig.model_validate({"research_evaluation": {"outputs": [output]}})


def test_research_evaluation_resolves_reduced_outputs_only():
    assert resolve_research_outputs(
        (
            "pattern-quality",
            "timeframe-classifier",
            "dynamic-transition-discovery",
            "trade-record-modulation",
        )
    ) == (
        "timeframe-classifier",
        "dynamic-transition-discovery",
        "pattern-quality",
        "trade-record-modulation",
    )
