import polars as pl

from qooi.scanner.tailrun.selection import (
    candidate_gate_frame,
    promoter_target_frame,
)
from qooi.scanner.tailrun.types import TailtreeCandidateGateSpec


def test_candidate_gate_frame_marks_top_pct_and_top_k_opportunity_rows() -> None:
    scored = pl.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "decision_bar_close_ms": [1, 1, 1, 1],
            "outcome_horizon": [24, 24, 24, 24],
            "objective": ["tail_event_lift"] * 4,
            "direction": ["up"] * 4,
            "tailtree_score": [0.9, 0.7, 0.3, 0.1],
            "selected_behavior_actionable": [True, False, True, False],
            "selected_behavior_false_direction": [False, True, False, False],
            "selected_behavior_utility_margin": [2.0, -1.0, 0.5, 0.0],
            "selected_behavior_blocker": ["", "opposite_clean_path", "", "no_clean_side"],
            "selected_behavior_actionability": [
                "tradable_up",
                "not_tradable",
                "tradable_up",
                "not_tradable",
            ],
        }
    )

    gates = candidate_gate_frame(
        scored,
        (TailtreeCandidateGateSpec("score_pct", 50.0), TailtreeCandidateGateSpec("top_k", 1.0)),
    )

    selected = gates.filter(pl.col("in_candidate_gate")).select(
        "candidate_gate_id", "symbol"
    )
    assert selected.to_dicts() == [
        {"candidate_gate_id": "score_pct_50", "symbol": "A"},
        {"candidate_gate_id": "score_pct_50", "symbol": "B"},
        {"candidate_gate_id": "top_k_1", "symbol": "A"},
    ]


def test_promoter_target_frame_separates_promotable_reject_and_gray_rows() -> None:
    gated = pl.DataFrame(
        {
            "symbol": ["GOOD", "BAD", "GRAY"],
            "decision_bar_close_ms": [1, 1, 1],
            "outcome_horizon": [24, 24, 24],
            "candidate_gate_id": ["top_k_3"] * 3,
            "in_candidate_gate": [True, True, True],
            "selected_behavior_actionable": [True, False, False],
            "selected_behavior_false_direction": [False, True, False],
            "selected_behavior_utility_margin": [1.5, -2.0, 0.4],
            "selected_behavior_blocker": ["", "opposite_clean_path", "no_clean_side"],
            "selected_behavior_actionability": [
                "tradable_up",
                "not_tradable",
                "watch",
            ],
        }
    )

    targets = promoter_target_frame(gated).select(
        "symbol", "promotable_up", "reject_up", "gray_up", "promoter_label"
    )

    assert targets.to_dicts() == [
        {
            "symbol": "GOOD",
            "promotable_up": True,
            "reject_up": False,
            "gray_up": False,
            "promoter_label": 1,
        },
        {
            "symbol": "BAD",
            "promotable_up": False,
            "reject_up": True,
            "gray_up": False,
            "promoter_label": 0,
        },
        {
            "symbol": "GRAY",
            "promotable_up": False,
            "reject_up": False,
            "gray_up": True,
            "promoter_label": None,
        },
    ]
