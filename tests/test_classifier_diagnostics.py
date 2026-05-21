from __future__ import annotations

import polars as pl

from qooi.research.diagnostics import (
    ClassifierDiagnosticsBuilder,
    evaluate_classifier_frame,
)


def _classifier_frame() -> pl.DataFrame:
    states = ["uptrend|range|range"] * 6 + ["downtrend|transition|wide_range"] * 4
    return pl.DataFrame(
        {
            "timestamp": [idx * 3_600_000 for idx in range(len(states))],
            "structure_trend_state": ["uptrend"] * 6 + ["downtrend"] * 4,
            "market_stage": ["range"] * 6
            + ["transition", "wide_range", "transition", "wide_range"],
            "structure_reason": ["compressed_range"] * 6 + ["ambiguous_transition"] * 4,
            "market_stage_reason": ["compressed_mid_range"] * 6
            + ["ambiguous_transition", "wide_range_no_stage"] * 2,
            "stage_unknown_reason": ["none"] * 6 + ["transition", "wide_range"] * 2,
            "range_width_atr": [4.0] * 10,
            "range_width_atr_threshold": [8.0] * 6 + [5.0] * 4,
            "range_width_threshold_mode": ["fixed"] * 10,
            "range_width_threshold_ready": [True] * 10,
            "range_width_threshold_source": ["fixed"] * 10,
            "mtf_state_key": states,
            "mtf_structure_key": states,
            "mtf_stage_key": states,
        }
    )


def test_classifier_diagnostics_artifact_contains_rows_and_tables():
    diagnostics = ClassifierDiagnosticsBuilder().evaluate("TEST", _classifier_frame())

    row_names = {row.name for row in diagnostics.rows}
    table_names = {table.name for table in diagnostics.tables}

    assert "Classifier coverage" in row_names
    assert "Structure x stage matrix" in row_names
    assert "Unknown reason consistency" in row_names
    assert "Resolved none audit" in row_names
    assert "MTF state transition matrix" in row_names
    assert {
        "distribution",
        "unknown_consistency",
        "matrix",
        "transition",
        "dwell",
        "time_distribution",
        "threshold",
    } <= table_names


def test_classifier_diagnostics_builder_matches_wrapper():
    frame = _classifier_frame()

    method = ClassifierDiagnosticsBuilder().evaluate("TEST", frame)
    wrapper = evaluate_classifier_frame("TEST", frame)

    assert method.rows == wrapper.rows
    assert [table.name for table in method.tables] == [table.name for table in wrapper.tables]


def test_classifier_diagnostics_export_is_long_form():
    diagnostics = evaluate_classifier_frame("TEST", _classifier_frame())
    export = diagnostics.to_export_frame()

    assert {"label", "artifact", "table", "layer", "name", "field", "value"} <= set(export.columns)
    assert export.filter(pl.col("artifact") == "table").height > 0
    assert export.filter(pl.col("table") == "matrix").height > 0
    assert export.filter(pl.col("table") == "transition").height > 0
    assert export.filter(pl.col("table") == "unknown_consistency").height > 0


def test_classifier_diagnostics_flags_unknown_none_contradictions():
    frame = _classifier_frame().with_columns(
        pl.when(pl.arange(0, pl.len()) == 0)
        .then(pl.lit("transition"))
        .otherwise(pl.col("market_stage"))
        .alias("market_stage"),
        pl.when(pl.arange(0, pl.len()) == 0)
        .then(pl.lit("none"))
        .otherwise(pl.col("stage_unknown_reason"))
        .alias("stage_unknown_reason"),
    )

    diagnostics = evaluate_classifier_frame("TEST", frame)
    consistency = next(row for row in diagnostics.rows if row.name == "Unknown reason consistency")

    assert consistency.severity == "fail"
    assert "contradictions=1/10" in consistency.summary
