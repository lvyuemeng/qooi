from __future__ import annotations

import polars as pl

from qooi.exchange.store import HistoryCoverage, HistoryTarget
from qooi.research.config import ResearchCommandConfig
from qooi.research.instruments import RESEARCH_UNIVERSE
from qooi.research.workflows import FrameRequest, prepare_classifier_frame
from qooi.strategies.features import StructureClassifierConfig


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
    def bars(self, request):
        if request.bar == "4H":
            return _frame(90, 4 * 3_600_000), _coverage(request.inst_id, request.bar, 90)
        if request.bar == "1D":
            return _frame(60, 24 * 3_600_000), _coverage(request.inst_id, request.bar, 60)
        return _frame(), _coverage(request.inst_id, request.bar, 240)


def _command():
    config = ResearchCommandConfig()
    return config.model_copy(
        update={
            "run": config.run.model_copy(update={"profile": "smoke"}),
            "cache": config.cache.model_copy(
                update={"days": 10, "min_bars": 10, "min_coverage_pct": 0.0}
            ),
        }
    )


def _request(pair, command: ResearchCommandConfig) -> FrameRequest:
    return FrameRequest(
        pair=pair,
        data_source=command.run.data_source,
        days=command.days,
        min_bars=command.min_bars,
        refresh=command.cache.refresh,
        min_coverage_pct=command.min_coverage_pct,
        allow_swap_signal_fallback=command.run.allow_swap_signal_fallback,
    )


def test_prepare_classifier_frame_does_not_run_backtest():
    command = _command()

    prepared = prepare_classifier_frame(
        FakeStore(), _request(RESEARCH_UNIVERSE[0], command), StructureClassifierConfig.fixed()
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
        FakeStore(), _request(pair, command), StructureClassifierConfig.fixed()
    )
    second = prepare_classifier_frame(
        FakeStore(), _request(pair, command), StructureClassifierConfig.fixed()
    )

    assert first.signal_inst_id == second.signal_inst_id
    assert first.frame.select("mtf_state_key", "h4_market_stage", "d1_market_stage").equals(
        second.frame.select("mtf_state_key", "h4_market_stage", "d1_market_stage")
    )
