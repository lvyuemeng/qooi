import polars as pl

from qooi.scanner.tailrun.selection import dual_guard_boundary_anatomy_frame


def test_dual_guard_boundary_anatomy_splits_top25_next25_and_missed_actionable() -> None:
    rows = 60
    actionable = [idx < 10 or 25 <= idx < 35 or idx >= 55 for idx in range(rows)]
    scored = pl.DataFrame(
        {
            "candidate_gate_id": ["top_k_200"] * rows,
            "promotion_score": [float(rows - idx) for idx in range(rows)],
            "opposite_guard_score": [float(idx) / rows for idx in range(rows)],
            "weak_path_guard_score": [float(idx) / rows for idx in range(rows)],
            "selected_behavior_actionable": actionable,
            "selected_behavior_false_direction": [idx in {30, 56} for idx in range(rows)],
            "selected_behavior_utility_margin": [2.0 if value else 0.0 for value in actionable],
            "selected_behavior_path_state": [
                "clean_up" if value else "none" for value in actionable
            ],
            "selected_behavior_actionability": [
                "tradable_up" if value else "no_action" for value in actionable
            ],
            "selected_behavior_blocker": [
                "" if value else "no_tail_touch" for value in actionable
            ],
        }
    )

    anatomy = dual_guard_boundary_anatomy_frame(
        scored,
        opposite_keep_pct=100.0,
        weak_keep_pct=100.0,
        selected_count=50,
        high_confidence_count=25,
    )

    bucket_rows = {
        row["boundary_bucket"]: row["row_count"]
        for row in anatomy.group_by("boundary_bucket").agg(pl.col("row_count").sum()).to_dicts()
    }
    assert bucket_rows == {
        "expansion_selected": 25,
        "high_confidence_selected": 25,
        "missed_actionable": 5,
        "unselected_negative": 5,
    }

    missed = anatomy.filter(pl.col("boundary_bucket") == "missed_actionable")
    assert missed.get_column("actionable_count").sum() == 5
    assert missed.get_column("false_direction_count").sum() == 1
