import polars as pl

from qooi.scanner.tailrun.selection import guarded_selection_error_anatomy_frame


def test_guarded_selection_error_anatomy_labels_guarded_buckets() -> None:
    scored = pl.DataFrame(
        {
            "candidate_gate_id": ["top_k_200"] * 4,
            "promotion_score": [0.99, 0.80, 0.70, 0.60],
            "opposite_guard_score": [0.99, 0.10, 0.20, 0.30],
            "selected_behavior_actionable": [False, True, False, True],
            "selected_behavior_false_direction": [True, False, False, False],
            "selected_behavior_path_state": ["clean_down", "clean_up", "none", "clean_up"],
            "selected_behavior_actionability": [
                "tradable_down",
                "tradable_up",
                "no_action",
                "tradable_up",
            ],
            "selected_behavior_blocker": ["opposite_clean_path", "", "no_tail_touch", ""],
            "selected_behavior_utility_margin": [-2.0, 3.0, 0.0, 1.0],
        }
    )

    anatomy = guarded_selection_error_anatomy_frame(
        scored,
        guard_keep_pcts=(50.0,),
        top_k_buckets=(2,),
        pct_buckets=(),
    )

    assert anatomy.get_column("objective").unique().to_list() == ["candidate_opposite_guard"]
    assert anatomy.get_column("guard_keep_pct").unique().to_list() == [50.0]
    assert anatomy.get_column("score_bucket").unique().to_list() == [
        "top_k_200_guard_keep_50_top_k_2"
    ]
    selected_fp = anatomy.filter(pl.col("error_family") == "selected_fp")
    assert selected_fp.get_column("path_state").to_list() == ["none"]
    assert selected_fp.get_column("blocker").to_list() == ["no_tail_touch"]
