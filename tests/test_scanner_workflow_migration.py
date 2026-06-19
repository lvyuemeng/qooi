from __future__ import annotations

import polars as pl

from qooi.scanner import rank
from qooi.scanner.config import PotentialConfig
from qooi.scanner.output import review_decisions


def test_rank_public_api_is_branch_explicit() -> None:
    assert hasattr(rank, "ladder_candidates")
    assert hasattr(rank, "tailtree_candidates")
    assert hasattr(rank, "rank_ladder_candidates")
    assert hasattr(rank, "rank_tailtree_candidates")
    assert hasattr(rank, "candidate_metric_surface")
    assert not hasattr(rank, "candidate_evidence")
    assert not hasattr(rank, "rank_candidate_evidence")


def test_candidate_metric_surface_keeps_branch_metrics_comparable() -> None:
    ladder = pl.DataFrame(
        {
            "symbol": ["BTC-USDT-SWAP"],
            "decision_timeframe": ["1H"],
            "decision_bar_close_ms": [1000],
            "outcome_horizon": [12],
            "statistical_direction": ["up"],
            "conditioned_observations": [50],
            "lift_up": [1.4],
            "lift_down": [0.8],
            "information_stability": [0.9],
            "transition_information_stability": [0.6],
            "avg_path_range_pct": [3.0],
            "rank_score": [8.0],
            "rank_reason": ["ladder"],
        }
    )
    tailtree = pl.DataFrame(
        {
            "symbol": ["ETH-USDT-SWAP"],
            "decision_timeframe": ["1H"],
            "decision_bar_close_ms": [1000],
            "outcome_horizon": [12],
            "tree_direction": ["down"],
            "N_total": [80],
            "tail_lift": [2.1],
            "tail_lift_stability": [1.1],
            "tail_utility_mean": [0.5],
            "tail_utility_p90": [0.9],
            "rank_score": [9.0],
            "rank_reason": ["tailtree"],
        }
    )

    surface = rank.rank_candidates(rank.candidate_metric_surface(ladder=ladder, tailtree=tailtree))

    assert surface.select("branch").to_series().to_list() == ["tailtree", "ladder"]
    assert set(surface.columns) == set(rank.CANDIDATE_METRIC_SURFACE_SCHEMA)
    assert surface.get_column("tail_lift").to_list() == [2.1, 1.4]


def test_review_freshness_skips_only_prediction_rows() -> None:
    config = PotentialConfig(max_staleness_hours=24)
    ranked = pl.DataFrame(
        {
            "symbol": ["BTC-USDT-SWAP"],
            "rank_score": [1.0],
        }
    )
    freshness = pl.DataFrame(
        {
            "symbol": ["BTC-USDT-SWAP"],
            "prediction_age_hours": [25.0],
            "prediction_freshness": ["stale"],
        }
    )

    decisions = review_decisions(ranked, freshness, {}, config)

    assert decisions[0].symbol == "BTC-USDT-SWAP"
    assert decisions[0].action == "skip"
    assert "stale prediction" in decisions[0].reason


def test_review_skips_zero_support_candidates() -> None:
    config = PotentialConfig(max_staleness_hours=24)
    ranked = pl.DataFrame(
        {
            "symbol": ["BTC-USDT-SWAP"],
            "rank_score": [0.0],
            "support_count": [0.0],
            "candidate_status": ["no_matching_evidence"],
        }
    )
    freshness = pl.DataFrame(
        {
            "symbol": ["BTC-USDT-SWAP"],
            "prediction_age_hours": [1.0],
            "prediction_freshness": ["fresh"],
        }
    )

    decisions = review_decisions(ranked, freshness, {}, config)

    assert decisions[0].symbol == "BTC-USDT-SWAP"
    assert decisions[0].action == "skip"
    assert decisions[0].reason == "no matching evidence"


def test_potential_config_has_nested_tailtree_only() -> None:
    config = PotentialConfig()

    assert hasattr(config.evidence, "tailtree")
    assert not hasattr(config, "tailtree")
