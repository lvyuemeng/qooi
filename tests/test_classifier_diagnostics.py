from __future__ import annotations

import polars as pl

from qooi.research.states import classifier_health


def test_classifier_health_replaces_legacy_builder_surface():
    result = classifier_health(
        pl.DataFrame(
            {
                "structure_trend_state": ["uptrend", "range"],
                "market_stage": ["markup", "range"],
                "structure_reason": ["trend", "compressed"],
                "stage_unknown_reason": ["none", "none"],
            }
        ),
        label="TEST",
    )

    assert set(result.frame["artifact"].to_list()) == {"classifier-health"}
    assert "required_classifier_columns" in result.text


def test_no_legacy_classifier_builder_symbols_are_exported():
    import qooi.research.states as states

    assert not hasattr(states, "ClassifierDiagnosticsBuilder")
    assert not hasattr(states, "evaluate_classifier_frame")
