import polars as pl

from qooi.scanner.tailrun.selection import opposite_guard_target_frame


def test_opposite_guard_target_frame_marks_clean_down_only() -> None:
    candidates = pl.DataFrame(
        {
            "selected_behavior_path_state": ["clean_down", "up_first_both", "none"],
            "selected_behavior_actionability": ["tradable_down", "reversal_watch", "no_action"],
            "selected_behavior_blocker": [
                "opposite_clean_path",
                "both_or_mixed_path",
                "no_tail_touch",
            ],
        }
    )

    guarded = opposite_guard_target_frame(candidates)

    assert guarded.get_column("opposite_guard_label").to_list() == [1, 0, 0]
    assert guarded.get_column("opposite_guard_weight").to_list() == [1.0, 1.0, 1.0]
