import polars as pl

import qooi.scanner.tailrun.selection as selection
from qooi.scanner.tailrun.selection import (
    dual_guarded_promotion_selection_metrics_frame,
    weak_path_guard_target_frame,
)


def test_low_performance_candidate_metric_apis_are_removed() -> None:
    assert not hasattr(selection, "promotion_selection_metrics_frame")
    assert not hasattr(selection, "guarded_promotion_selection_metrics_frame")
    assert not hasattr(selection, "continuous_guard_curve_frame")
    assert not hasattr(selection, "two_model_guard_selection_metrics_frame")


def test_weak_path_guard_target_marks_no_tail_none_and_non_positive_utility() -> None:
    candidates = pl.DataFrame(
        {
            "selected_behavior_path_state": ["none", "clean_up", "clean_up", "up_first_both"],
            "selected_behavior_blocker": ["no_tail_touch", "", "", "both_or_mixed_path"],
            "selected_behavior_utility_margin": [0.0, 0.0, 2.0, -1.0],
        }
    )

    guarded = weak_path_guard_target_frame(candidates)

    assert guarded.get_column("weak_path_guard_label").to_list() == [1, 1, 0, 1]
    assert guarded.get_column("weak_path_guard_weight").to_list() == [1.0, 1.0, 1.0, 1.0]


def test_dual_guarded_selection_applies_opposite_and_weak_filters_before_ranking() -> None:
    scored = pl.DataFrame(
        {
            "candidate_gate_id": ["top_k_200"] * 6,
            "candidate_gate_family": ["top_k"] * 6,
            "candidate_gate_value": [200.0] * 6,
            "promotion_score": [0.99, 0.95, 0.90, 0.85, 0.80, 0.70],
            "opposite_guard_score": [0.95, 0.10, 0.20, 0.30, 0.40, 0.50],
            "weak_path_guard_score": [0.10, 0.95, 0.20, 0.30, 0.40, 0.50],
            "selected_tail": [True, True, True, False, True, True],
            "selected_behavior_actionable": [False, False, True, False, True, True],
            "selected_behavior_false_direction": [True, False, False, False, False, False],
            "selected_behavior_utility_margin": [-2.0, 0.0, 3.0, 0.0, 2.0, 1.0],
        }
    )

    metrics = dual_guarded_promotion_selection_metrics_frame(
        scored,
        opposite_keep_pcts=(50.0,),
        weak_keep_pcts=(50.0,),
        top_k_buckets=(2,),
        pct_buckets=(),
    )

    row = metrics.row(0, named=True)
    assert row["objective"] == "candidate_dual_guard"
    assert row["opposite_keep_pct"] == 50.0
    assert row["weak_keep_pct"] == 50.0
    assert row["selected_observation_count"] == 2.0
    assert row["behavior_tp_count"] == 1.0
    assert row["behavior_fp_count"] == 1.0
    assert row["paired_behavior_false_direction_rate"] == 0.0
