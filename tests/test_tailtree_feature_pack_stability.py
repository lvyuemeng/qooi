from pathlib import Path

import polars as pl

from qooi.scanner.tailrun.artifacts import write_tailtree_feature_pack_stability
from qooi.scanner.tailrun.selection import (
    decision_key_action_surface_frame,
    feature_pack_stability_frame,
    frontier_benchmark_frame,
)


def test_feature_pack_stability_emits_explicit_improvement_actions() -> None:
    source_features = pl.DataFrame(
        {
            "symbol": ["A", "B", "C", "D", "E", "F"],
            "timestamp": [1, 1, 1, 1, 1, 1],
            "funding_level_state": [
                "funding_positive",
                "funding_positive",
                "funding_positive",
                "funding_positive",
                "funding_negative",
                "funding_negative",
            ],
            "lsr_level_state": [
                "lsr_short_crowding",
                "lsr_short_crowding",
                "lsr_short_crowding",
                "lsr_short_crowding",
                None,
                None,
            ],
        }
    )
    action_surface = pl.DataFrame(
        {
            "symbol": ["A", "B", "C", "D", "E", "F"],
            "decision_bar_close_ms": [1, 1, 1, 1, 1, 1],
            "actionability": [
                "trade_candidate",
                "trade_candidate",
                "trade_candidate",
                "no_action",
                "trade_candidate",
                "no_action",
            ],
            "false_direction_int": [0, 1, 0, 0, 0, 0],
            "best_utility_margin": [3.0, 2.0, 1.0, 0.5, -1.0, -0.5],
        }
    )

    frame = feature_pack_stability_frame(
        source_features,
        action_surface,
        min_support=2,
        feature_columns=("funding_level_state", "lsr_level_state"),
    )

    assert set(frame.get_column("improvement_action").to_list()) >= {
        "investigate_high_risk_opportunity",
        "reject_negative_utility",
    }
    high_risk = frame.filter(pl.col("feature_value") == "funding_positive").row(0, named=True)
    assert high_risk["precision"] == 0.75
    assert high_risk["false_direction_rate"] == 0.25
    assert high_risk["improvement_action"] == "investigate_high_risk_opportunity"
    rejected = frame.filter(pl.col("feature_value") == "funding_negative").row(0, named=True)
    assert rejected["improvement_action"] == "reject_negative_utility"
    assert rejected["improvement_reason"]


def test_feature_pack_stability_uses_decision_key_grain() -> None:
    source_features = pl.DataFrame(
        {
            "symbol": ["A"],
            "timestamp": [1],
            "lsr_level_state": ["lsr_short_crowding"],
        }
    )
    duplicated_surface = pl.DataFrame(
        {
            "symbol": ["A", "A", "A"],
            "decision_bar_close_ms": [1, 1, 1],
            "actionability": ["no_action", "trade_candidate", "trade_candidate"],
            "false_direction_int": [0, 0, 1],
            "best_utility_margin": [0.5, 2.0, 1.0],
        }
    )

    decision_keys = decision_key_action_surface_frame(duplicated_surface)
    frame = feature_pack_stability_frame(
        source_features,
        decision_keys,
        min_support=1,
        feature_columns=("lsr_level_state",),
    )

    row = frame.row(0, named=True)
    assert decision_keys.height == 1
    assert row["support"] == 1
    assert row["candidate_count"] == 1
    assert row["false_direction_count"] == 1
    assert row["surface_rows"] == 3


def test_feature_pack_stability_rejects_low_support() -> None:
    source_features = pl.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "timestamp": [1, 1, 1],
            "funding_level_transition": ["funding_flip", "funding_persistence", None],
        }
    )
    action_surface = pl.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "decision_bar_close_ms": [1, 1, 1],
            "actionability": ["trade_candidate", "trade_candidate", "no_action"],
            "false_direction_int": [0, 0, 0],
            "best_utility_margin": [2.0, 1.0, 0.0],
        }
    )

    frame = feature_pack_stability_frame(
        source_features,
        action_surface,
        min_support=2,
        feature_columns=("funding_level_transition",),
    )

    assert frame.filter(pl.col("feature_value") == "funding_flip").item(
        0, "improvement_action"
    ) == "reject_low_support"


def test_write_tailtree_feature_pack_stability_writes_csv(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        {
            "feature_pack": ["source_timeseries_context"],
            "feature_name": ["funding_level_state"],
            "feature_value": ["funding_positive"],
            "improvement_action": ["keep_diagnostic_only"],
        }
    )

    write_tailtree_feature_pack_stability(tmp_path, frame)

    assert (tmp_path / "tailtree-feature-pack-stability.csv").exists()


def test_frontier_benchmark_keeps_base_objectives_only() -> None:
    efficiency = pl.DataFrame(
        {
            "objective": [
                "candidate_opposite_guard",
                "candidate_dual_guard",
                "candidate_dual_guard",
            ],
            "feature_set": ["base", "base", "base"],
            "candidate_gate_id": ["top_k_200", "top_k_200", "top_k_200"],
            "budget_family": ["gate_pct", "top_k", "top_k"],
            "budget_value": [25.0, 100.0, 200.0],
            "guard_keep_pct": [100.0, None, None],
            "selected_observation_count": [50.0, 75.0, 75.0],
            "behavior_precision": [0.60, 0.613333, 0.613333],
            "paired_behavior_false_direction_rate": [0.14, 0.133333, 0.133333],
            "paired_behavior_utility_margin_mean": [2.991, 3.377, 3.377],
            "behavior_hpo_score": [1.56, -6.14, -6.14],
        }
    )

    benchmark = frontier_benchmark_frame(efficiency)

    assert benchmark.filter(pl.col("objective") == "candidate_dual_guard").height == 1
    assert set(benchmark.get_column("feature_set").to_list()) == {"base"}
    assert not any("source_blended" in objective for objective in benchmark["objective"].to_list())
