from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from qooi.scanner.output import _tailtree_action_surface_lines, _tailtree_promotion_gate_lines
from qooi.scanner.tailrun.selection import (
    calibrated_candidate_replay_frame,
    candidate_replay_metrics,
    paired_candidate_replay_frame,
    score_bucket_candidate_frame,
    tailtree_action_surface_frame,
)
from qooi.scanner.tailrun.types import TailtreeDirection
from qooi.scanner.tailtree.model import (
    TrainConfig,
    label_tail_paths,
    tailtree_label_distribution_frame,
    tailtree_target_training_values,
)


@dataclass(frozen=True)
class _ScoreMetadata:
    direction: TailtreeDirection
    categorical_features: list[str]
    continuous_features: list[str]


class _ScoreTree:
    def __init__(self, direction: TailtreeDirection, scores: list[float]) -> None:
        self.metadata = _ScoreMetadata(
            direction=direction,
            categorical_features=[],
            continuous_features=[],
        )
        self._scores = scores

    def predict_score(self, features: pl.DataFrame) -> pl.DataFrame:
        return features.with_columns(pl.Series("tailtree_score", self._scores))

    def predict_leaf(self, features: pl.DataFrame) -> pl.DataFrame:
        return features.with_columns(pl.lit(0).alias("leaf_id"))

    def to_json(self, path: str | Path) -> None:
        return None


def _outcomes() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "decision_bar_close_ms": [1, 2, 3, 4],
            "outcome_horizon": [24, 24, 24, 24],
            "forward_max_return_pct": [40.0, 10.0, 40.0, 10.0],
            "forward_min_return_pct": [-10.0, -40.0, -40.0, -10.0],
        }
    )


def test_label_tail_paths_adds_orthogonal_tail_state() -> None:
    labeled = label_tail_paths(_outcomes(), threshold_pct=30.0)

    label_columns = ["symbol", "tail_up", "tail_down", "tail_any", "tail_both", "tail_state"]
    assert labeled.select(label_columns).to_dicts() == [
        {
            "symbol": "A",
            "tail_up": True,
            "tail_down": False,
            "tail_any": True,
            "tail_both": False,
            "tail_state": "up",
        },
        {
            "symbol": "B",
            "tail_up": False,
            "tail_down": True,
            "tail_any": True,
            "tail_both": False,
            "tail_state": "down",
        },
        {
            "symbol": "C",
            "tail_up": True,
            "tail_down": True,
            "tail_any": True,
            "tail_both": True,
            "tail_state": "both",
        },
        {
            "symbol": "D",
            "tail_up": False,
            "tail_down": False,
            "tail_any": False,
            "tail_both": False,
            "tail_state": "none",
        },
    ]
    assert "tail_up_only" not in labeled.columns
    assert "tail_down_only" not in labeled.columns
    margins = labeled.select("tail_utility_margin_up", "tail_utility_margin_down")
    assert margins.get_column("tail_utility_margin_up").to_list() == [10.0, -10.0, 0.0, 0.0]
    assert margins.get_column("tail_utility_margin_down").to_list() == [-10.0, 10.0, 0.0, 0.0]


def test_tailtree_label_distribution_uses_tail_state_as_orthogonal_grain() -> None:
    labeled = label_tail_paths(_outcomes(), threshold_pct=30.0)

    distribution = tailtree_label_distribution_frame(labeled)

    assert distribution.select("tail_state").to_series().to_list() == ["both", "down", "none", "up"]
    assert distribution.select(pl.col("class_rate").sum().round(6)).item() == 1.0
    by_state = {row["tail_state"]: row for row in distribution.to_dicts()}
    assert by_state["up"]["tail_up_count"] == 1
    assert by_state["down"]["tail_down_count"] == 1
    assert by_state["both"]["tail_both_count"] == 1
    assert by_state["none"]["tail_any_count"] == 0


def test_score_bucket_replay_metrics_derive_side_only_from_tail_state() -> None:
    observations = pl.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "decision_bar_close_ms": [1, 2, 3, 4],
        }
    )
    labeled = label_tail_paths(_outcomes(), threshold_pct=30.0)

    up_candidates = score_bucket_candidate_frame(
        _ScoreTree("up", [0.9, 0.1, 0.8, 0.2]),
        observations,
        labeled,
        24,
        score_quantiles=(0.5,),
    )
    down_candidates = score_bucket_candidate_frame(
        _ScoreTree("down", [0.1, 0.9, 0.8, 0.2]),
        observations,
        labeled,
        24,
        score_quantiles=(0.5,),
    )

    replay = paired_candidate_replay_frame(pl.concat([up_candidates, down_candidates]))
    up_top = replay.filter(
        (pl.col("selected_direction") == "up") & (pl.col("symbol") == "A")
    ).row(0, named=True)
    both_top = replay.filter(
        (pl.col("selected_direction") == "up") & (pl.col("symbol") == "C")
    ).row(0, named=True)

    assert up_top["selected_side_only"] is True
    assert up_top["selected_tail_both"] is False
    assert up_top["side_only_int"] == 1
    assert both_top["selected_side_only"] is False
    assert both_top["selected_tail_both"] is True
    assert both_top["tail_both_int"] == 1

    metrics = candidate_replay_metrics(
        replay,
        outcome_horizon=24,
        direction="up",
        score_bucket="top_50pct",
    )
    assert metrics["paired_side_only_rate"] == 0.5
    assert metrics["paired_tail_both_rate"] == 0.5
    assert metrics["paired_selected_utility_margin_mean"] == 5.0


def test_calibrated_candidate_replay_frame_adds_bucket_side_margins() -> None:
    replay = pl.DataFrame(
        {
            "outcome_horizon": [24, 24, 24, 24],
            "score_bucket": ["top_50pct", "top_50pct", "top_50pct", "top_50pct"],
            "selected_direction": ["up", "up", "down", "down"],
            "selected_tail": [True, False, True, False],
            "opposite_tail": [False, True, True, False],
            "selected_side_only": [True, False, False, False],
            "opposite_side_only": [False, True, False, False],
            "selected_tail_both": [False, False, True, False],
        }
    )

    calibrated = calibrated_candidate_replay_frame(replay)
    up = calibrated.filter(pl.col("selected_direction") == "up").row(0, named=True)
    down = calibrated.filter(pl.col("selected_direction") == "down").row(0, named=True)

    assert up["selected_bucket_tail_rate"] == 0.5
    assert up["opposite_bucket_tail_rate"] == 0.5
    assert up["selected_bucket_side_only_rate"] == 0.5
    assert up["opposite_bucket_side_only_rate"] == 0.5
    assert up["selected_bucket_tail_both_rate"] == 0.0
    assert up["calibrated_directional_margin"] == 0.0
    assert up["calibrated_side_margin"] == 0.0
    assert down["selected_bucket_tail_both_rate"] == 0.5


def test_tailtree_target_training_values_use_any_event_and_side_only_labels() -> None:
    observations = pl.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "decision_bar_close_ms": [1, 2, 3, 4],
            "feature": [10.0, 20.0, 30.0, 40.0],
        }
    )
    labeled = label_tail_paths(_outcomes(), threshold_pct=30.0)

    event_features, event_labels, event_utilities = tailtree_target_training_values(
        observations,
        labeled,
        target="tail_event_lift",
        direction="up",
    )
    any_features, any_labels, any_utilities = tailtree_target_training_values(
        observations,
        labeled,
        target="tail_any_event",
        direction="up",
    )
    side_features, side_labels, side_utilities = tailtree_target_training_values(
        observations,
        labeled,
        target="tail_side_only",
        direction="up",
    )

    assert event_features.select("symbol").to_series().to_list() == ["A", "B", "C", "D"]
    assert event_labels.tolist() == [1.0, 0.0, 1.0, 0.0]
    assert event_utilities.tolist() == [10.0, 0.0, 10.0, 0.0]
    assert any_features.select("symbol").to_series().to_list() == ["A", "B", "C", "D"]
    assert any_labels.tolist() == [1.0, 1.0, 1.0, 0.0]
    assert any_utilities.tolist() == [10.0, 10.0, 10.0, 0.0]
    assert side_features.select("symbol").to_series().to_list() == ["A", "B", "C", "D"]
    assert side_labels.tolist() == [1.0, 0.0, 0.0, 0.0]
    assert side_utilities.tolist() == [10.0, 0.0, 0.0, 0.0]
    assert TrainConfig(objective="tail_any_event").objective == "tail_any_event"
    assert TrainConfig(objective="tail_side_only").objective == "tail_side_only"


def test_label_tail_paths_adds_behavior_and_actionability_columns() -> None:
    outcomes = pl.DataFrame(
        {
            "symbol": ["A", "B", "C", "D", "E"],
            "decision_bar_close_ms": [1, 2, 3, 4, 5],
            "outcome_horizon": [24, 24, 24, 24, 24],
            "forward_max_return_pct": [40.0, 10.0, 40.0, 40.0, 10.0],
            "forward_min_return_pct": [-10.0, -40.0, -40.0, -40.0, -10.0],
            "time_to_max_bar": [0, 0, 4, 8, 0],
            "time_to_min_bar": [0, 0, 8, 4, 0],
            "path_efficiency": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    labeled = label_tail_paths(outcomes, threshold_pct=30.0)
    rows = labeled.select(
        "symbol",
        "tail_touch_up",
        "tail_touch_down",
        "first_touch_side",
        "path_state",
        "path_actionability",
    ).to_dicts()

    assert rows == [
        {
            "symbol": "A",
            "tail_touch_up": True,
            "tail_touch_down": False,
            "first_touch_side": "up",
            "path_state": "clean_up",
            "path_actionability": "tradable_up",
        },
        {
            "symbol": "B",
            "tail_touch_up": False,
            "tail_touch_down": True,
            "first_touch_side": "down",
            "path_state": "clean_down",
            "path_actionability": "tradable_down",
        },
        {
            "symbol": "C",
            "tail_touch_up": True,
            "tail_touch_down": True,
            "first_touch_side": "up",
            "path_state": "up_first_both",
            "path_actionability": "reversal_watch",
        },
        {
            "symbol": "D",
            "tail_touch_up": True,
            "tail_touch_down": True,
            "first_touch_side": "down",
            "path_state": "down_first_both",
            "path_actionability": "reversal_watch",
        },
        {
            "symbol": "E",
            "tail_touch_up": False,
            "tail_touch_down": False,
            "first_touch_side": "none",
            "path_state": "none",
            "path_actionability": "no_action",
        },
    ]


def test_tailtree_action_surface_is_single_semantic_candidate_surface() -> None:
    replay = pl.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "decision_bar_close_ms": [1, 2, 3],
            "outcome_horizon": [6, 24, 24],
            "score_bucket": ["top_10pct", "top_10pct", "top_10pct"],
            "selected_direction": ["up", "down", "down"],
            "selected_score": [0.9, 0.8, 0.7],
            "opposite_score": [0.1, 0.6, 0.9],
            "selected_tail": [True, True, False],
            "opposite_tail": [False, False, True],
            "selected_side_only": [True, False, False],
            "opposite_side_only": [False, False, True],
            "selected_tail_both": [False, True, False],
            "opposite_tail_both": [False, False, False],
            "selected_tail_state": ["clean_up", "down_first_both", "none"],
            "selected_utility_margin": [5.0, 1.0, -2.0],
            "calibrated_side_margin": [0.2, -0.1, -0.3],
            "false_direction_int": [0, 0, 1],
        }
    )

    surface = tailtree_action_surface_frame(replay)

    assert surface.select(
        "symbol", "action_side", "entry_horizon", "actionability", "blocker_reason"
    ).to_dicts() == [
        {
            "symbol": "A",
            "action_side": "up",
            "entry_horizon": 6,
            "actionability": "trade_candidate",
            "blocker_reason": "",
        },
        {
            "symbol": "B",
            "action_side": "down",
            "entry_horizon": 24,
            "actionability": "gray_zone",
            "blocker_reason": "both_or_mixed_path",
        },
        {
            "symbol": "C",
            "action_side": "down",
            "entry_horizon": 24,
            "actionability": "reversal_watch",
            "blocker_reason": "opposite_tail_dominates",
        },
    ]


def test_tailtree_report_lines_keep_infeasible_down_as_market_state() -> None:
    surface = pl.DataFrame(
        {
            "action_side": ["up", "down", "down"],
            "entry_horizon": [6, 24, 24],
            "actionability": ["trade_candidate", "reversal_watch", "no_action"],
            "blocker_reason": ["", "opposite_tail_dominates", "no_clean_side"],
            "best_path_state": ["clean_up", "clean_up", "none"],
            "calibrated_side_margin": [0.2, -0.1, -0.2],
            "false_direction_int": [0, 1, 0],
        }
    )

    action_lines = "\n".join(_tailtree_action_surface_lines(surface))
    gate_lines = "\n".join(_tailtree_promotion_gate_lines(surface))

    assert "actionability trade_candidate: rows=1" in action_lines
    assert "down opposite_tail_dominates: rows=1" in action_lines
    assert "up: candidate annotation" in gate_lines
    assert "down: market-state only; do not promote short" in gate_lines
