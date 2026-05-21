from __future__ import annotations

import polars as pl

from qooi.core.evaluate import Report
from qooi.research.diagnostics import evaluate_state_attribution, format_state_attribution
from qooi.research.instruments import CORE_UNIVERSE


def _report(trades=None, equity=None):
    return Report.from_raw(trades or [], equity or [100.0, 100.0], CORE_UNIVERSE[0])


def _trade(**overrides):
    trade = {"pnl": 0.01, "pnl_usd": 1.0, "net_pnl_usd": 1.0}
    trade.update(overrides)
    return trade


def test_state_diagnostics_composes_classifier_and_trade_artifacts():
    frame = pl.DataFrame(
        {
            "timestamp": [0, 1],
            "mtf_state_key": ["uptrend|range|range", "uptrend|range|range"],
            "mtf_structure_key": ["uptrend|range|range", "uptrend|range|range"],
            "mtf_stage_key": ["range|range|range", "range|range|range"],
        }
    )
    report = _report(
        [
            _trade(
                side="buy",
                entry_liquidity_event_type="failed_breakout_low",
                entry_mtf_state_key="uptrend|range|range",
                net_pnl_usd=1.0,
                pnl_usd=1.0,
            )
        ],
        [100.0, 101.0],
    )

    diagnostics = evaluate_state_attribution("TEST", report, frame)
    summary = format_state_attribution(diagnostics)

    assert "MTF state attribution" in summary
    assert "MTF state x event" in summary
