import polars as pl

from qooi.scanner.tailrun.selection import selection_error_anatomy_frame


def test_selection_error_anatomy_separates_false_direction_from_generic_fp() -> None:
    scored = pl.DataFrame(
        {
            "candidate_gate_id": ["top_k_200"] * 4,
            "candidate_gate_family": ["top_k"] * 4,
            "candidate_gate_value": [200.0] * 4,
            "symbol": ["A", "B", "C", "D"],
            "decision_bar_close_ms": [1, 2, 3, 4],
            "promotion_score": [0.9, 0.8, 0.7, 0.1],
            "selected_behavior_actionable": [True, False, False, False],
            "selected_behavior_false_direction": [False, True, False, True],
            "selected_behavior_path_state": [
                "clean_up",
                "down_first_both",
                "chop_both",
                "clean_down",
            ],
            "selected_behavior_actionability": [
                "tradable_up",
                "reversal_watch",
                "gray_zone",
                "tradable_down",
            ],
            "selected_behavior_blocker": ["", "both_or_mixed_path", "no_clean_side", ""],
            "selected_behavior_utility_margin": [2.0, 1.0, -0.5, -2.0],
        }
    )

    anatomy = selection_error_anatomy_frame(scored, top_k_buckets=(3,), pct_buckets=())

    false_rows = anatomy.filter(pl.col("error_family") == "false_direction")
    assert false_rows.get_column("row_count").sum() == 1
    assert false_rows.select("path_state").to_series().to_list() == ["down_first_both"]

    fp_rows = anatomy.filter(pl.col("error_family") == "selected_fp")
    assert fp_rows.get_column("row_count").sum() == 1
    assert fp_rows.select("path_state").to_series().to_list() == ["chop_both"]

    tp_rows = anatomy.filter(pl.col("error_family") == "selected_tp")
    assert tp_rows.get_column("row_count").sum() == 1
    assert tp_rows.select("precision").item() == 1.0


def test_selection_error_anatomy_empty_schema_for_missing_columns() -> None:
    anatomy = selection_error_anatomy_frame(pl.DataFrame({"candidate_gate_id": ["x"]}))

    assert anatomy.is_empty()
    assert {
        "candidate_gate_id",
        "score_bucket",
        "error_family",
        "path_state",
        "row_count",
        "false_direction_rate",
    }.issubset(set(anatomy.columns))
