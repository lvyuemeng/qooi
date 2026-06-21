from __future__ import annotations

import polars as pl

from qooi.scanner import rank
from qooi.scanner.config import BarsConfig, PotentialConfig, RubikConfig, SnapshotConfig
from qooi.scanner.output import review_decisions
from qooi.scanner.workflow import scanner_market_request


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


def test_review_abstains_on_material_direction_conflict() -> None:
    config = PotentialConfig(max_staleness_hours=24)
    ranked = pl.DataFrame(
        {
            "symbol": ["RE-USDT-SWAP", "RE-USDT-SWAP"],
            "branch": ["tailtree", "tailtree"],
            "direction": ["down", "up"],
            "outcome_horizon": [24, 24],
            "rank_score": [53.9, 42.7],
            "support_count": [4378.0, 2189.0],
            "tail_lift": [44.2, 33.2],
            "utility_proxy": [3.1, 30.0],
            "source_freshness": ["fresh", "fresh"],
            "required_missing_source_count": [0, 0],
            "required_stale_source_count": [0, 0],
        }
    )
    freshness = pl.DataFrame(
        {
            "symbol": ["RE-USDT-SWAP"],
            "prediction_age_hours": [1.0],
            "prediction_freshness": ["fresh"],
        }
    )

    decisions = review_decisions(ranked, freshness, {}, config)

    assert len(decisions) == 1
    assert decisions[0].symbol == "RE-USDT-SWAP"
    assert decisions[0].direction == "up"
    assert decisions[0].action == "watch"
    assert decisions[0].reason == "direction conflict: down vs up; abstain from promotion"


def test_potential_config_has_nested_tailtree_only() -> None:
    config = PotentialConfig()

    assert hasattr(config.evidence, "tailtree")
    assert not hasattr(config, "tailtree")


def test_scanner_market_request_splits_fetch_freshness_from_review_tolerance() -> None:
    config = PotentialConfig(
        max_staleness_hours=24,
        bars=BarsConfig(latest_staleness_hours=2),
        books=SnapshotConfig(limit=25, max_staleness_hours=1),
        trades=SnapshotConfig(limit=100, max_staleness_hours=1),
        funding=SnapshotConfig(limit=100, max_staleness_hours=8),
        open_interest=RubikConfig(max_staleness_hours=2),
        taker_volume=RubikConfig(max_staleness_hours=2),
        long_short=RubikConfig(max_staleness_hours=2),
    )

    request = scanner_market_request(config, ("BTC-USDT-SWAP",))
    source_hours = {
        product.name: product.max_staleness_hours for product in request.sources.products
    }

    assert request.bars.max_staleness_hours == 24
    assert request.bars.latest_staleness_hours == 2
    assert request.sources.max_staleness_hours == 24
    assert source_hours == {
        "books": 1,
        "trades": 1,
        "funding": 8,
        "open_interest": 2,
        "taker_volume": 2,
        "long_short_ratios": 2,
    }
