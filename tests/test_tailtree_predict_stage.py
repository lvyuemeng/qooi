from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl
import pytest


def _load_predict_script():
    path = Path("scripts/03_predict.py")
    spec = importlib.util.spec_from_file_location("tailtree_predict_stage", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scored_at_ms() -> int:
    return 1_700_000_000_000 + 2 * 60 * 60 * 1000


def _scored_rows() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"],
            "decision_bar_close_ms": [1_700_000_000_000] * 3,
            "horizon_hours": [24, 24, 24],
            "path_prob_calm": [0.20, 0.40, 0.25],
            "path_prob_smooth_up": [0.62, 0.20, 0.15],
            "path_prob_smooth_down": [0.05, 0.12, 0.55],
            "path_prob_chop": [0.08, 0.18, 0.20],
            "path_prob_fake_breakout": [0.05, 0.10, 0.05],
            "path_pred_label": [1, 0, 2],
            "path_pred_label_name": ["smooth_up", "calm", "smooth_down"],
            "path_confidence": [0.62, 0.40, 0.55],
        }
    )


def _scored_horizon_rows() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["BTC-USDT-SWAP", "BTC-USDT-SWAP", "BTC-USDT-SWAP", "ETH-USDT-SWAP"],
            "decision_bar_close_ms": [1_700_000_000_000] * 4,
            "horizon_hours": [4, 12, 24, 4],
            "path_prob_calm": [0.15, 0.20, 0.18, 0.40],
            "path_prob_smooth_up": [0.70, 0.20, 0.62, 0.15],
            "path_prob_smooth_down": [0.05, 0.55, 0.10, 0.12],
            "path_prob_chop": [0.05, 0.10, 0.12, 0.20],
            "path_prob_fake_breakout": [0.05, 0.05, 0.08, 0.13],
            "path_pred_label": [1, 2, 1, 0],
            "path_pred_label_name": ["smooth_up", "smooth_down", "smooth_up", "calm"],
            "path_confidence": [0.70, 0.55, 0.62, 0.40],
        }
    )


def test_predict_script_has_no_output_or_live_workflow_dependency() -> None:
    predict = _load_predict_script()
    text = Path("scripts/03_predict.py").read_text(encoding="utf-8")

    assert predict.OUTPUT_DIR == Path("data/output/potential/path")
    assert predict.FEATURE_DIR == predict.OUTPUT_DIR
    assert predict.FEATURE_MATRIX_PATH.name == "predict_features.parquet"
    assert predict.RECENT_DECISION_MAX_AGE_HOURS == 2.0
    assert predict.BOARD_PATH.name == "path_probability_board.parquet"
    assert predict.REPORT_PATH.name == "prediction-report.md"
    assert "qooi.scanner.output" not in text
    assert "qooi.scanner.workflow" not in text
    assert "run_tailtree" not in text


def test_predict_stage_removes_retired_output_dirs(tmp_path, monkeypatch) -> None:
    predict = _load_predict_script()
    legacy = tmp_path / "path-predict"
    legacy.mkdir()
    (legacy / "prediction-report.md").write_text("stale", encoding="utf-8")
    monkeypatch.setattr(predict, "LEGACY_OUTPUT_DIRS", (legacy,))

    predict.remove_legacy_outputs()

    assert not legacy.exists()


def test_recent_latest_feature_rows_uses_recent_window_before_latest() -> None:
    predict = _load_predict_script()
    matrix = pl.DataFrame(
        {
            "symbol": ["BTC", "BTC", "BTC"],
            "horizon_hours": [24, 24, 12],
            "decision_bar_close_ms": [1_000, 30_000_000, 30_000_001],
        }
    )

    recent = predict.recent_latest_feature_rows(matrix, max_age_hours=6)

    assert recent.height == 2
    assert recent.get_column("decision_bar_close_ms").min() > 1_000


def test_predict_stage_rejects_stale_input_matrix() -> None:
    predict = _load_predict_script()
    matrix = pl.DataFrame(
        {
            "symbol": ["BTC"],
            "horizon_hours": [24],
            "decision_bar_close_ms": [1_700_000_000_000],
        }
    )

    assert predict.latest_decision_age_hours(matrix, scored_at_ms=_scored_at_ms()) == 2.0
    with pytest.raises(RuntimeError, match="predict_features.parquet is stale"):
        predict.require_recent_input(matrix)


def test_prediction_board_ranks_participation_score_and_reports_freshness_columns() -> None:
    predict = _load_predict_script()
    board = predict.prediction_board(_scored_rows(), scored_at_ms=_scored_at_ms())

    assert board.row(0, named=True)["symbol"] == "BTC-USDT-SWAP"
    assert board.row(0, named=True)["direction"] == "long"
    assert board.row(1, named=True)["direction"] == "short"
    assert {
        "participation_score",
        "trend_probability",
        "risk_probability",
        "decision_age_hours",
        "prediction_validity",
        "decision_time_utc",
        "scored_at_utc",
        "reason",
    } <= set(board.columns)
    assert board.get_column("promotion_score").is_sorted(descending=True)
    assert set(board.get_column("prediction_validity")) == {"valid"}


def test_prediction_report_is_user_readable_markdown() -> None:
    predict = _load_predict_script()
    board = predict.prediction_board(_scored_rows(), scored_at_ms=_scored_at_ms())
    report = predict.prediction_report(
        board, model_id="tailtree-path_path", selected_feature_count=46
    )

    assert "# Tailtree prediction report" in report
    assert "## Most worth participating" in report
    assert "BTC-USDT-SWAP" in report
    assert "participation_score" in report
    assert "trend_probability" in report
    assert "scored_at_utc" in report
    assert "valid_for_hours" in report
    assert "max_decision_age_hours" not in report
    assert "decision_age_h" in report
    assert "side selection uses model probabilities directly" in report
    assert "not financial advice" in report


def test_prediction_report_keeps_all_horizons_for_ranked_symbol() -> None:
    predict = _load_predict_script()
    board = predict.prediction_board(_scored_horizon_rows(), scored_at_ms=_scored_at_ms())
    report = predict.prediction_report(
        board, model_id="tailtree-path_path", selected_feature_count=46
    )

    assert "## Per-symbol horizon ranking" in report
    assert "horizon_direction_conflict_count" in report
    assert "BTC-USDT-SWAP" in report
    assert "| 4 |" in report
    assert "| 12 |" in report
    assert "| 24 |" in report
    assert "conflict/watch" in report


def test_prediction_board_uses_source_presence_calibrated_promotion_score() -> None:
    predict = _load_predict_script()
    scored = _scored_rows().with_columns(pl.Series("base__source_any_present", [0.0, 1.0, 1.0]))

    board = predict.prediction_board(scored, scored_at_ms=_scored_at_ms())

    assert {"source_any_present", "source_presence_calibrated_score", "promotion_score"} <= set(
        board.columns
    )
    btc = board.filter(pl.col("symbol") == "BTC-USDT-SWAP").row(0, named=True)
    assert btc["source_any_present"] == 0.0
    assert btc["source_presence_calibrated_score"] == btc["participation_score"] * 0.5
    assert board.get_column("promotion_score").is_sorted(descending=True)


def test_prediction_report_explains_promotion_score() -> None:
    predict = _load_predict_script()
    board = predict.prediction_board(
        _scored_rows().with_columns(pl.Series("base__source_any_present", [0.0, 1.0, 1.0])),
        scored_at_ms=_scored_at_ms(),
    )
    report = predict.prediction_report(
        board, model_id="tailtree-path_path", selected_feature_count=46
    )

    assert "promotion_score" in report
    assert "source_presence_calibrated_score" in report


def test_prediction_board_does_not_block_high_scoring_short_side() -> None:
    predict = _load_predict_script()

    board = predict.prediction_board(_scored_rows(), scored_at_ms=_scored_at_ms())

    short = board.filter(pl.col("direction") == "short").row(0, named=True)
    assert short["promotion_action"] == "promote"
    assert short["side_gate_pass"] is True
    assert short["side_gate_reason"] == "model side probabilities used directly"
