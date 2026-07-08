from __future__ import annotations

import polars as pl
import pytest
from qooi.scanner.rank import PATH_PROBABILITY_BOARD_SCHEMA, path_probability_board


def _scored_path_frame() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "symbol": "BTC-USDT-SWAP",
                "decision_bar_close_ms": 1_000,
                "horizon_hours": 4,
                "path_prob_calm": 0.10,
                "path_prob_smooth_up": 0.72,
                "path_prob_smooth_down": 0.04,
                "path_prob_chop": 0.07,
                "path_prob_fake_breakout": 0.07,
                "path_pred_label": 1,
                "path_pred_label_name": "smooth_up",
                "path_confidence": 0.72,
            },
            {
                "symbol": "ETH-USDT-SWAP",
                "decision_bar_close_ms": 1_000,
                "horizon_hours": 12,
                "path_prob_calm": 0.05,
                "path_prob_smooth_up": 0.20,
                "path_prob_smooth_down": 0.05,
                "path_prob_chop": 0.10,
                "path_prob_fake_breakout": 0.60,
                "path_pred_label": 4,
                "path_pred_label_name": "fake_breakout",
                "path_confidence": 0.60,
            },
            {
                "symbol": "SOL-USDT-SWAP",
                "decision_bar_close_ms": 1_000,
                "horizon_hours": 24,
                "path_prob_calm": 0.15,
                "path_prob_smooth_up": 0.22,
                "path_prob_smooth_down": 0.18,
                "path_prob_chop": 0.38,
                "path_prob_fake_breakout": 0.07,
                "path_pred_label": 3,
                "path_pred_label_name": "chop",
                "path_confidence": 0.38,
            },
        ]
    )


def test_path_probability_board_projects_probabilities_without_execution_columns() -> None:
    board = path_probability_board(_scored_path_frame())

    assert board.schema == PATH_PROBABILITY_BOARD_SCHEMA
    assert board.columns == list(PATH_PROBABILITY_BOARD_SCHEMA)
    forbidden = {
        "action",
        "action_side",
        "actionability",
        "order_side",
        "trigger_price",
        "position_size",
        "stop_loss",
        "take_profit",
    }
    assert forbidden.isdisjoint(set(board.columns))
    assert board.select("symbol").to_series().to_list() == [
        "BTC-USDT-SWAP",
        "SOL-USDT-SWAP",
        "ETH-USDT-SWAP",
    ]


def test_path_probability_board_marks_risk_and_veto_reasons() -> None:
    rows = {row["symbol"]: row for row in path_probability_board(_scored_path_frame()).to_dicts()}

    assert rows["BTC-USDT-SWAP"]["conflict_flag"] is False
    assert rows["BTC-USDT-SWAP"]["veto_reason"] == ""
    assert rows["BTC-USDT-SWAP"]["preferred_path"] == "smooth_up"
    assert rows["BTC-USDT-SWAP"]["direction_hint"] == "up"
    assert rows["BTC-USDT-SWAP"]["path_rank_score"] > rows["ETH-USDT-SWAP"]["path_rank_score"]

    assert rows["ETH-USDT-SWAP"]["conflict_flag"] is True
    assert rows["ETH-USDT-SWAP"]["risk_path"] == "fake_breakout"
    assert rows["ETH-USDT-SWAP"]["veto_reason"] == "fake_breakout_risk"
    assert rows["ETH-USDT-SWAP"]["direction_hint"] == "risk"

    assert rows["SOL-USDT-SWAP"]["conflict_flag"] is True
    assert rows["SOL-USDT-SWAP"]["risk_path"] == "chop"
    assert rows["SOL-USDT-SWAP"]["veto_reason"] == "chop_risk"
    assert rows["SOL-USDT-SWAP"]["direction_hint"] == "risk"


def test_path_probability_board_rejects_missing_probability_columns() -> None:
    with pytest.raises(ValueError, match="path_prob_fake_breakout"):
        path_probability_board(_scored_path_frame().drop("path_prob_fake_breakout"))


def test_path_probability_board_empty_schema() -> None:
    assert path_probability_board(pl.DataFrame()).schema == PATH_PROBABILITY_BOARD_SCHEMA
