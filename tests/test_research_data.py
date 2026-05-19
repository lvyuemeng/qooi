from __future__ import annotations

from types import SimpleNamespace

import polars as pl

from qooi.core.config import RESEARCH_PAIRS
from qooi.exchange.store import HistoryCoverage, HistoryTarget
from qooi.research.config import resolve_config
from qooi.strategies.features import StructureClassifierConfig
from qooi.strategies.preprocessing import (
    ClassifierContextConfig,
    ClassifierFramePipeline,
    prepare_classifier_frame,
)


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


def _args():
    return SimpleNamespace(
        profile="smoke",
        days=10,
        min_bars=10,
        min_coverage_pct=0.0,
        universe="core",
        data_source="swap",
        style="single",
        refresh_cache=False,
        allow_swap_signal_fallback=False,
        max_per_strategy_symbol=0,
    )


def test_prepare_classifier_frame_does_not_run_backtest():
    args = _args()
    config = resolve_config(args)
    context = ClassifierContextConfig(classifier=StructureClassifierConfig.fixed())

    prepared = prepare_classifier_frame(FakeStore(), RESEARCH_PAIRS[0], args, config, context)

    assert "market_stage" in prepared.frame.columns
    assert "h4_market_stage" in prepared.frame.columns
    assert "d1_market_stage" in prepared.frame.columns
    assert "mtf_state_key" in prepared.frame.columns
    assert "range_width_atr_threshold" in prepared.frame.columns
    assert "h4_range_width_atr_threshold" in prepared.frame.columns


def test_classifier_frame_pipeline_prepare_matches_tail_composed_steps():
    args = _args()
    config = resolve_config(args)
    context = ClassifierContextConfig(classifier=StructureClassifierConfig.fixed())
    pair = RESEARCH_PAIRS[0]

    wrapper = prepare_classifier_frame(FakeStore(), pair, args, config, context)
    method = ClassifierFramePipeline(FakeStore(), args, config, context).prepare(pair)

    assert wrapper.signal_inst_id == method.signal_inst_id
    assert wrapper.frame.select("mtf_state_key", "h4_market_stage", "d1_market_stage").equals(
        method.frame.select("mtf_state_key", "h4_market_stage", "d1_market_stage")
    )
