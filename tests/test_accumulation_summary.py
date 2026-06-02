from __future__ import annotations

import polars as pl

from qooi.accumulation.summary import (
    CandidateReadoutSettings,
    build_candidate_detail,
    build_candidate_summary,
    build_next_fetch_actions,
    render_candidate_rationale,
)


def test_summary_keeps_high_score_low_confidence_separate() -> None:
    scores = pl.DataFrame(
        {
            "timestamp": [1],
            "symbol": ["BTC-USDT-SWAP"],
            "alert_level": ["red"],
            "score_total": [75],
            "source_coverage_score": [0.4],
            "confidence_level": ["low"],
            "positive_components": ["depth_support_on_down_day +20"],
            "negative_filters": [""],
            "missing_evidence": ["trades_missing;funding_missing"],
            "data_quality_warning": ["trades_missing;funding_missing"],
        }
    )

    summary = build_candidate_summary(scores, pl.DataFrame(), top_n=10)

    assert summary["alert_level"][0] == "red"
    assert summary["confidence_level"][0] == "low"
    assert "trades_missing" in summary["missing_evidence"][0]
    assert "collect-market trades" == summary["next_fetch_action"][0]


def test_next_fetch_actions_prioritize_missing_trades_for_supported_scores() -> None:
    scores = pl.DataFrame(
        {
            "timestamp": [1],
            "symbol": ["BTC-USDT-SWAP"],
            "score_total": [35],
            "missing_evidence": ["trades_missing;open_interest_missing"],
        }
    )

    actions = build_next_fetch_actions(scores, pl.DataFrame())

    assert actions["source"][0] == "trades"
    assert actions["priority"][0] == 1


def test_trade_notional_metadata_missing_requests_discovery() -> None:
    scores = pl.DataFrame(
        {
            "timestamp": [1],
            "symbol": ["BTC-USDT-SWAP"],
            "score_total": [0],
            "missing_evidence": ["trade_notional_metadata_missing"],
        }
    )

    summary = build_candidate_summary(scores, pl.DataFrame(), top_n=1)

    assert summary["next_fetch_action"][0] == "discover contract metadata"


def test_watchlist_onchain_gap_requests_verification() -> None:
    scores = pl.DataFrame(
        {
            "timestamp": [1],
            "symbol": ["MEME-USDT-SWAP"],
            "score_total": [0],
            "suggestion_type": ["prepare_watch"],
            "missing_evidence": ["onchain_missing;whale_missing"],
        }
    )

    summary = build_candidate_summary(scores, pl.DataFrame(), top_n=1)
    actions = build_next_fetch_actions(scores, pl.DataFrame())

    assert summary["next_fetch_action"][0] == "collect-onchain"
    assert actions["source"][0] == "onchain"


def test_candidate_detail_includes_latest_feature_readout() -> None:
    scores = pl.DataFrame(
        {
            "timestamp": [1, 2],
            "symbol": ["MEME-USDT-SWAP", "MEME-USDT-SWAP"],
            "score_total": [10, 35],
            "alert_level": ["none", "yellow"],
            "confidence_level": ["blocked", "medium"],
            "suggestion_type": ["reject_or_deprioritize", "prepare_watch"],
            "missing_evidence": ["", "onchain_missing"],
            "positive_components": ["", "depth_support_on_down_day +20"],
            "negative_filters": ["", ""],
        }
    )
    features = pl.DataFrame(
        {
            "timestamp": [1, 2],
            "symbol": ["MEME-USDT-SWAP", "MEME-USDT-SWAP"],
            "return_24h": [-0.03, 0.07],
            "range_position_pct": [0.2, 0.8],
            "depth_imbalance_25_mean": [0.1, 0.42],
        }
    )

    detail = build_candidate_detail(
        scores,
        features,
        pl.DataFrame(),
        settings=CandidateReadoutSettings(top_n=1),
    )

    assert detail["rank"][0] == 1
    assert detail["score_total"][0] == 35
    assert detail["return_24h"][0] == 0.07
    assert detail["depth_imbalance_25_mean"][0] == 0.42
    assert detail["next_fetch_action"][0] == "collect-onchain"


def test_rationale_renders_ranked_rows() -> None:
    summary = pl.DataFrame(
        {
            "rank": [1],
            "symbol": ["BTC-USDT-SWAP"],
            "alert_level": ["yellow"],
            "confidence_level": ["medium"],
            "score_total": [20],
            "rationale": ["score components: depth"],
            "next_fetch_action": [""],
        }
    )

    out = render_candidate_rationale(summary)

    assert "BTC-USDT-SWAP" in out
    assert "yellow / medium" in out
