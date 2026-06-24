import polars as pl

from qooi.scanner.tailrun.selection import actionability_contradiction_audit_frame


def test_actionability_contradiction_audit_separates_no_tail_actionable() -> None:
    scored = pl.DataFrame(
        {
            "candidate_gate_id": ["top_k_200"] * 5,
            "promotion_score": [0.5, 0.4, 0.3, 0.2, 0.1],
            "opposite_guard_score": [0.1, 0.2, 0.3, 0.4, 0.5],
            "weak_path_guard_score": [0.2, 0.3, 0.4, 0.5, 0.6],
            "selected_behavior_actionable": [True, True, False, False, True],
            "selected_behavior_false_direction": [False, False, False, True, False],
            "selected_behavior_utility_margin": [5.0, 3.0, 0.0, -1.0, 2.0],
            "selected_behavior_path_state": [
                "none",
                "clean_up",
                "none",
                "clean_down",
                "up_first_both",
            ],
            "selected_behavior_actionability": [
                "no_action",
                "tradable_up",
                "no_action",
                "tradable_down",
                "reversal_watch",
            ],
            "selected_behavior_blocker": [
                "no_tail_touch",
                "",
                "no_tail_touch",
                "opposite_clean_path",
                "both_or_mixed_path",
            ],
        }
    )

    audit = actionability_contradiction_audit_frame(scored)
    counts = {
        row["audit_bucket"]: row["row_count"]
        for row in audit.group_by("audit_bucket").agg(pl.col("row_count").sum()).to_dicts()
    }

    assert counts == {
        "blocked_negative": 1,
        "clean_actionable": 1,
        "contradictory_actionable_no_tail": 1,
        "opposite_risk": 1,
        "other": 1,
    }
    contradiction = audit.filter(
        pl.col("audit_bucket") == "contradictory_actionable_no_tail"
    )
    assert contradiction.get_column("utility_margin_mean").item() == 5.0
