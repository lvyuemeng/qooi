from __future__ import annotations

import polars as pl

from qooi.scanner.classifiers import classifier_health


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
    import qooi.scanner.classifiers as classifiers

    assert not hasattr(classifiers, "ClassifierDiagnosticsBuilder")
    assert not hasattr(classifiers, "evaluate_classifier_frame")


