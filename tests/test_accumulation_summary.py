from __future__ import annotations

import polars as pl

from qooi.accumulation.summary import (
    CandidateReadoutSettings,
    NextFetchPolicy,
    build_candidate_detail,
    build_candidate_summary,
    build_next_fetch_actions,
    render_candidate_rationale,
    render_scan_feedback,
)
from qooi.sources.coverage import manifest_frame, source_manifest_row


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
            "score_total": [25],
            "suggestion_type": ["prepare_watch"],
            "missing_evidence": ["onchain_missing;whale_missing"],
            "positive_components": ["depth_support_on_down_day +20"],
        }
    )
    policy = NextFetchPolicy(actionable_sources=("discovery", "onchain"))

    summary = build_candidate_summary(scores, pl.DataFrame(), top_n=1, policy=policy)
    actions = build_next_fetch_actions(scores, pl.DataFrame(), policy=policy)

    assert summary["next_fetch_action"][0] == "collect-onchain"
    assert actions["source"][0] == "onchain"


def test_disabled_policy_does_not_queue_optional_sources() -> None:
    scores = pl.DataFrame(
        {
            "timestamp": [1],
            "symbol": ["MEME-USDT-SWAP"],
            "score_total": [25],
            "suggestion_type": ["prepare_watch"],
            "missing_evidence": ["onchain_missing;messages_missing"],
            "positive_components": ["depth_support_on_down_day +20"],
        }
    )

    actions = build_next_fetch_actions(scores, pl.DataFrame())

    assert actions.is_empty()


def test_unimplemented_onchain_provider_does_not_queue_action() -> None:
    scores = pl.DataFrame(
        {
            "timestamp": [1],
            "symbol": ["BTC-USDT-SWAP"],
            "score_total": [25],
            "suggestion_type": ["prepare_watch"],
            "missing_evidence": ["onchain_missing"],
            "positive_components": ["depth_support_on_down_day +20"],
        }
    )
    coverage = manifest_frame(
        [
            source_manifest_row(
                symbol="BTC-USDT-SWAP",
                source="onchain",
                phase="collect-onchain",
                status="missing",
                warning="onchain_provider_not_implemented",
            )
        ]
    )
    policy = NextFetchPolicy(actionable_sources=("discovery", "onchain"))

    actions = build_next_fetch_actions(scores, coverage, policy=policy)
    summary = build_candidate_summary(scores, coverage, policy=policy)
    out = render_scan_feedback(summary, pl.DataFrame(), actions, coverage)

    assert actions.is_empty()
    assert "on-chain provider collector is not implemented" in out


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
        policy=NextFetchPolicy(actionable_sources=("discovery", "onchain")),
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


def test_scan_feedback_renders_run_readout_and_followups() -> None:
    summary = pl.DataFrame(
        {
            "rank": [1],
            "symbol": ["BTC-USDT-SWAP"],
            "alert_level": ["yellow"],
            "confidence_level": ["low"],
            "score_total": [35],
            "suggestion_type": ["prepare_watch"],
            "top_positive_components": ["depth_support_on_down_day +20"],
            "top_negative_filters": [""],
            "missing_evidence": ["trades_missing"],
            "next_fetch_action": ["collect-market trades"],
            "data_quality_warning": ["trades_missing"],
        }
    )
    next_fetch = pl.DataFrame(
        {
            "symbol": ["BTC-USDT-SWAP"],
            "priority": [1],
            "source": ["trades"],
            "phase": ["collect-market"],
            "reason": ["trade resilience missing"],
            "expected_confidence_delta": ["medium"],
            "requires_secret": [False],
        }
    )
    coverage = manifest_frame(
        [
            source_manifest_row(
                symbol="BTC-USDT-SWAP",
                source="trades",
                phase="collect-market",
                status="missing",
                warning="trades_missing",
            )
        ]
    )

    out = render_scan_feedback(summary, pl.DataFrame(), next_fetch, coverage)

    assert "# Accumulation Scan Feedback" in out
    assert "Research-only: scanner artifacts do not authorize live trading." in out
    assert "Candidates reviewed: 1" in out
    assert "BTC-USDT-SWAP" in out
    assert "collect-market trades" in out
    assert "trades/missing/trades_missing: 1" in out


def test_scan_feedback_renders_broad_selection_candidates() -> None:
    summary = pl.DataFrame(
        {
            "rank": [1],
            "symbol": ["EDGE-USDT-SWAP"],
            "alert_level": ["yellow"],
            "confidence_level": ["medium"],
            "score_total": [20],
            "source_coverage_score": [0.9],
            "suggestion_type": ["prepare_watch"],
            "preparation_state": ["prepare_watch"],
            "activation_state": ["early"],
            "structure_state": ["supportive"],
            "top_positive_components": ["depth_support_on_down_day +20"],
            "top_negative_filters": [""],
            "missing_evidence": ["onchain_missing;messages_missing"],
            "next_fetch_action": [""],
            "data_quality_warning": ["onchain_missing;messages_missing"],
        }
    )
    broad = pl.DataFrame(
        {
            "rank": [1, 2, 3, 4],
            "base_ccy": ["EDGE", "BILL", "USDT", "NOPE"],
            "okx_symbol": ["EDGE-USDT-SWAP", "BILL-USDT-SWAP", "USDT-USDT-SWAP", ""],
            "okx_mapped": [True, True, True, False],
            "broad_score": [12.0, 14.0, 25.0, 20.0],
            "broad_reasons": ["active_1h;active_24h", "active_24h", "active_1h", "active_1h"],
            "exclude_reason": ["", "", "stablecoin", ""],
        }
    )

    out = render_scan_feedback(summary, pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), broad)

    assert "## Selection Readout" in out
    assert "Review ranking only: broad_score is not merged" in out
    assert "## Best Structure Setups" in out
    assert "EDGE-USDT-SWAP: verdict=watch_orderbook, bucket=strict_alert" in out
    assert "## Data Blocked Broad Rows" in out
    assert "BILL-USDT-SWAP: verdict=data_blocked, bucket=data_blocked" in out
    assert "USDT-USDT-SWAP" not in out
    assert "Broad excluded/unmapped: excluded=1, unmapped=1" in out
    assert "flow_outflow_3sigma_2h unavailable/missing: 1" in out
    assert "message_not_overheated unavailable/missing: 1" in out


def test_selection_range_unknown_early_broad_row_is_data_blocked() -> None:
    summary = pl.DataFrame(
        {
            "rank": [1],
            "symbol": ["EARLY-USDT-SWAP"],
            "alert_level": ["none"],
            "confidence_level": ["medium"],
            "score_total": [0],
            "source_coverage_score": [1.0],
            "suggestion_type": ["reject_or_deprioritize"],
            "preparation_state": ["prepare_watch"],
            "activation_state": ["early"],
            "structure_state": ["range_mid"],
            "top_positive_components": [""],
            "top_negative_filters": [""],
            "missing_evidence": [""],
            "next_fetch_action": [""],
            "data_quality_warning": [""],
        }
    )
    broad = pl.DataFrame(
        {
            "rank": [1],
            "base_ccy": ["EARLY"],
            "okx_symbol": ["EARLY-USDT-SWAP"],
            "okx_mapped": [True],
            "broad_score": [10.0],
            "broad_reasons": ["active_1h;active_24h"],
            "exclude_reason": [""],
        }
    )

    out = render_scan_feedback(summary, pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), broad)

    assert "## Data Blocked Broad Rows" in out
    assert "EARLY-USDT-SWAP: verdict=data_blocked, bucket=data_blocked" in out
    assert "bucket=early_trend_review" not in out


def test_selection_strict_alert_with_unknown_range_remains_visible() -> None:
    summary = pl.DataFrame(
        {
            "rank": [1],
            "symbol": ["STRICT-USDT-SWAP"],
            "alert_level": ["yellow"],
            "confidence_level": ["medium"],
            "score_total": [20],
            "source_coverage_score": [1.0],
            "suggestion_type": ["prepare_watch"],
            "preparation_state": ["prepare_watch"],
            "activation_state": ["early"],
            "structure_state": ["range_mid"],
            "top_positive_components": ["depth_support_on_down_day +20"],
            "top_negative_filters": [""],
            "missing_evidence": [""],
            "next_fetch_action": [""],
            "data_quality_warning": [""],
        }
    )
    broad = pl.DataFrame(
        {
            "rank": [1],
            "base_ccy": ["STRICT"],
            "okx_symbol": ["STRICT-USDT-SWAP"],
            "okx_mapped": [True],
            "broad_score": [10.0],
            "broad_reasons": ["active_1h"],
            "exclude_reason": [""],
        }
    )

    out = render_scan_feedback(summary, pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), broad)

    assert "STRICT-USDT-SWAP:" in out
    assert "bucket=strict_alert" in out
    assert "px=n/a" in out


def test_selection_prefers_favorable_range_setup_over_near_resistance() -> None:
    summary = pl.DataFrame(
        {
            "rank": [1, 2],
            "symbol": ["LOW-USDT-SWAP", "HIGH-USDT-SWAP"],
            "alert_level": ["none", "none"],
            "confidence_level": ["medium", "medium"],
            "score_total": [0, 0],
            "source_coverage_score": [1.0, 1.0],
            "suggestion_type": ["reject_or_deprioritize", "reject_or_deprioritize"],
            "preparation_state": ["prepare_watch", "watchlist"],
            "activation_state": ["early", "overextended"],
            "structure_state": ["range_low", "extended"],
            "top_positive_components": ["", ""],
            "top_negative_filters": ["", ""],
            "missing_evidence": ["messages_missing", "messages_missing"],
            "next_fetch_action": ["", ""],
            "data_quality_warning": ["messages_missing", "messages_missing"],
        }
    )
    detail = pl.DataFrame(
        {
            "symbol": ["LOW-USDT-SWAP", "HIGH-USDT-SWAP"],
            "close": [12.0, 19.5],
            "range_low_px": [10.0, 10.0],
            "range_high_px": [20.0, 20.0],
            "range_position_pct": [0.2, 0.95],
            "upside_to_range_high_pct": [(20.0 / 12.0) - 1.0, (20.0 / 19.5) - 1.0],
            "downside_to_range_low_pct": [(12.0 / 10.0) - 1.0, (19.5 / 10.0) - 1.0],
            "range_reward_risk": [3.3333333333, 0.027],
        }
    )
    broad = pl.DataFrame(
        {
            "rank": [1, 2],
            "base_ccy": ["HIGH", "LOW"],
            "okx_symbol": ["HIGH-USDT-SWAP", "LOW-USDT-SWAP"],
            "okx_mapped": [True, True],
            "broad_score": [20.0, 12.0],
            "broad_reasons": ["active_1h;active_24h", "active_24h"],
            "exclude_reason": ["", ""],
        }
    )

    out = render_scan_feedback(summary, detail, pl.DataFrame(), pl.DataFrame(), broad)

    assert "## Best Structure Setups" in out
    assert "LOW-USDT-SWAP: verdict=review_now, bucket=accumulation_setup" in out
    assert "## Late Or Poor Structure" in out
    assert "HIGH-USDT-SWAP: verdict=avoid_late, bucket=late_trend_review" in out
    assert "range_quality=favorable_range_setup" in out
    assert "near_resistance" in out
    assert out.index("LOW-USDT-SWAP") < out.index("HIGH-USDT-SWAP")


def test_selection_derivative_context_does_not_upgrade_poor_structure() -> None:
    summary = pl.DataFrame(
        {
            "rank": [1],
            "symbol": ["LATE-USDT-SWAP"],
            "alert_level": ["none"],
            "confidence_level": ["medium"],
            "score_total": [0],
            "source_coverage_score": [1.0],
            "suggestion_type": ["reject_or_deprioritize"],
            "preparation_state": ["watchlist"],
            "activation_state": ["early"],
            "structure_state": ["range_mid"],
            "top_positive_components": [""],
            "top_negative_filters": [""],
            "missing_evidence": [""],
            "next_fetch_action": [""],
            "data_quality_warning": [""],
        }
    )
    detail = pl.DataFrame(
        {
            "symbol": ["LATE-USDT-SWAP"],
            "close": [19.5],
            "range_low_px": [10.0],
            "range_high_px": [20.0],
            "range_position_pct": [0.95],
            "upside_to_range_high_pct": [(20.0 / 19.5) - 1.0],
            "downside_to_range_low_pct": [(19.5 / 10.0) - 1.0],
            "range_reward_risk": [0.027],
            "taker_buy_ratio": [0.75],
            "open_interest_usd_change_24h": [0.12],
        }
    )
    broad = pl.DataFrame(
        {
            "rank": [1],
            "base_ccy": ["LATE"],
            "okx_symbol": ["LATE-USDT-SWAP"],
            "okx_mapped": [True],
            "broad_score": [30.0],
            "broad_reasons": ["active_1h;active_24h"],
            "exclude_reason": [""],
        }
    )

    out = render_scan_feedback(summary, detail, pl.DataFrame(), pl.DataFrame(), broad)

    assert "## Late Or Poor Structure" in out
    assert "LATE-USDT-SWAP" in out.split("## Late Or Poor Structure", maxsplit=1)[1]
    assert "taker=taker_buy_dominant" in out
    assert "derivatives=oi_expanding" in out


def test_selection_renders_orderbook_watch_without_broad_activity() -> None:
    summary = pl.DataFrame(
        {
            "rank": [1],
            "symbol": ["BOOK-USDT-SWAP"],
            "alert_level": ["none"],
            "confidence_level": ["medium"],
            "score_total": [0],
            "source_coverage_score": [1.0],
            "suggestion_type": ["reject_or_deprioritize"],
            "preparation_state": ["not_ready"],
            "activation_state": ["inactive"],
            "structure_state": ["range_mid"],
            "top_positive_components": [""],
            "top_negative_filters": [""],
            "missing_evidence": [""],
            "next_fetch_action": [""],
            "data_quality_warning": [""],
        }
    )
    detail = pl.DataFrame(
        {
            "symbol": ["BOOK-USDT-SWAP"],
            "close": [14.0],
            "range_low_px": [10.0],
            "range_high_px": [30.0],
            "range_position_pct": [0.6],
            "upside_to_range_high_pct": [(30.0 / 14.0) - 1.0],
            "downside_to_range_low_pct": [(14.0 / 10.0) - 1.0],
            "range_reward_risk": [2.85],
            "depth_imbalance_25_mean": [0.35],
        }
    )
    broad = pl.DataFrame(
        {
            "rank": [1],
            "base_ccy": ["BOOK"],
            "okx_symbol": ["BOOK-USDT-SWAP"],
            "okx_mapped": [True],
            "broad_score": [4.0],
            "broad_reasons": [""],
            "exclude_reason": [""],
        }
    )

    out = render_scan_feedback(summary, detail, pl.DataFrame(), pl.DataFrame(), broad)

    assert "BOOK-USDT-SWAP: verdict=watch_orderbook, bucket=orderbook_watch" in out
    assert "order=order_supported" in out


def test_selection_exact_range_low_uses_buffered_risk_readout() -> None:
    summary = pl.DataFrame(
        {
            "rank": [1],
            "symbol": ["LOW-USDT-SWAP"],
            "alert_level": ["none"],
            "confidence_level": ["medium"],
            "score_total": [0],
            "source_coverage_score": [1.0],
            "suggestion_type": ["reject_or_deprioritize"],
            "preparation_state": ["prepare_watch"],
            "activation_state": ["early"],
            "structure_state": ["range_low"],
            "top_positive_components": [""],
            "top_negative_filters": [""],
            "missing_evidence": [""],
            "next_fetch_action": [""],
            "data_quality_warning": [""],
        }
    )
    detail = pl.DataFrame(
        {
            "symbol": ["LOW-USDT-SWAP"],
            "close": [10.0],
            "range_low_px": [10.0],
            "range_high_px": [20.0],
            "range_position_pct": [0.0],
            "upside_to_range_high_pct": [1.0],
            "downside_to_range_low_pct": [0.0],
            "range_reward_risk": [None],
        }
    )
    broad = pl.DataFrame(
        {
            "rank": [1],
            "base_ccy": ["LOW"],
            "okx_symbol": ["LOW-USDT-SWAP"],
            "okx_mapped": [True],
            "broad_score": [8.0],
            "broad_reasons": ["active_24h"],
            "exclude_reason": [""],
        }
    )

    out = render_scan_feedback(summary, detail, pl.DataFrame(), pl.DataFrame(), broad)

    assert "LOW-USDT-SWAP: verdict=review_now, bucket=accumulation_setup" in out
    assert "risk=5.3%" in out
    assert "R:R=n/a" not in out


def test_scan_feedback_selection_is_not_limited_to_summary_rows() -> None:
    summary = pl.DataFrame(
        {
            "rank": [1],
            "symbol": ["A-USDT-SWAP"],
            "alert_level": ["none"],
            "confidence_level": ["low"],
            "score_total": [0],
            "suggestion_type": ["reject_or_deprioritize"],
            "top_positive_components": [""],
            "top_negative_filters": [""],
            "missing_evidence": [""],
            "next_fetch_action": [""],
            "data_quality_warning": [""],
        }
    )
    symbols = [f"S{i}-USDT-SWAP" for i in range(30)]
    broad = pl.DataFrame(
        {
            "rank": list(range(1, 31)),
            "base_ccy": [f"S{i}" for i in range(30)],
            "okx_symbol": symbols,
            "okx_mapped": [True] * 30,
            "broad_score": [float(100 - i) for i in range(30)],
            "broad_reasons": ["active_24h"] * 30,
            "exclude_reason": [""] * 30,
        }
    )

    out = render_scan_feedback(summary, pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), broad)

    assert "S0-USDT-SWAP" in out
    assert "S24-USDT-SWAP" in out
    assert "S25-USDT-SWAP" not in out


def test_scan_feedback_hides_weak_rows_from_top_candidates() -> None:
    summary = pl.DataFrame(
        {
            "rank": [1],
            "symbol": ["SATS-USDT-SWAP"],
            "alert_level": ["none"],
            "confidence_level": ["medium"],
            "score_total": [0],
            "suggestion_type": ["reject_or_deprioritize"],
            "top_positive_components": [""],
            "top_negative_filters": [""],
            "missing_evidence": ["messages_missing"],
            "next_fetch_action": [""],
            "data_quality_warning": ["messages_missing"],
        }
    )

    out = render_scan_feedback(summary, pl.DataFrame(), pl.DataFrame(), pl.DataFrame())

    assert "No positive-evidence candidates in this run." in out
    assert "## Rejected Or Neutral Rows" in out
    assert "no positive scanner rules fired" in out


def test_scan_feedback_renders_message_manifest_reasons() -> None:
    summary = pl.DataFrame(
        {
            "rank": [1, 2, 3, 4],
            "symbol": ["A", "B", "C", "D"],
            "alert_level": ["none", "none", "none", "none"],
            "confidence_level": ["low", "low", "low", "low"],
            "score_total": [0, 0, 0, 0],
            "suggestion_type": [
                "reject_or_deprioritize",
                "reject_or_deprioritize",
                "reject_or_deprioritize",
                "reject_or_deprioritize",
            ],
            "top_positive_components": ["", "", "", ""],
            "top_negative_filters": ["", "", "", ""],
            "missing_evidence": [
                "messages_missing",
                "messages_missing",
                "messages_missing",
                "messages_missing",
            ],
            "next_fetch_action": ["", "", "", ""],
            "data_quality_warning": [
                "messages_missing",
                "messages_missing",
                "messages_missing",
                "messages_missing",
            ],
        }
    )
    coverage = manifest_frame(
        [
            source_manifest_row(
                symbol="A",
                source="messages",
                phase="collect-context",
                status="skipped",
                warning="messages_disabled",
            ),
            source_manifest_row(
                symbol="B",
                source="messages",
                phase="collect-context",
                status="missing",
                warning="local_messages_path_missing",
            ),
            source_manifest_row(
                symbol="C",
                source="messages",
                phase="collect-context",
                status="missing",
                warning="local_messages_file_missing",
            ),
            source_manifest_row(
                symbol="D",
                source="messages",
                phase="collect-context",
                status="missing",
                warning="local_messages_missing",
            ),
        ]
    )

    out = render_scan_feedback(summary, pl.DataFrame(), pl.DataFrame(), coverage)

    assert "messages disabled in config" in out
    assert "local message CSV path is not configured" in out
    assert "configured local message CSV file does not exist" in out
    assert "local message CSV has no rows for this symbol" in out


def test_scan_feedback_renders_polymarket_manifest_reasons() -> None:
    coverage = manifest_frame(
        [
            source_manifest_row(
                symbol="A",
                source="polymarket_markets",
                phase="collect-context",
                status="skipped",
                warning="polymarket_disabled",
            ),
            source_manifest_row(
                symbol="B",
                source="polymarket_markets",
                phase="collect-context",
                status="skipped",
                warning="polymarket_query_disabled",
            ),
            source_manifest_row(
                symbol="C",
                source="polymarket_markets",
                phase="collect-context",
                status="missing",
                warning="polymarket_unmatched",
            ),
            source_manifest_row(
                symbol="D",
                source="polymarket_markets",
                phase="collect-context",
                status="missing",
                warning="polymarket_alias_missing",
            ),
        ]
    )

    out = render_scan_feedback(pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), coverage)

    assert "Polymarket context disabled by config" in out
    assert "Polymarket query disabled by config" in out
    assert "polymarket_unmatched" in out
    assert "polymarket_alias_missing" in out


def test_onchain_token_mapping_missing_does_not_queue_provider_fetch() -> None:
    scores = pl.DataFrame(
        {
            "timestamp": [1],
            "symbol": ["BTC-USDT-SWAP"],
            "score_total": [25],
            "suggestion_type": ["prepare_watch"],
            "missing_evidence": ["onchain_missing"],
            "positive_components": ["depth_support_on_down_day +20"],
        }
    )
    coverage = manifest_frame(
        [
            source_manifest_row(
                symbol="BTC-USDT-SWAP",
                source="onchain",
                phase="collect-onchain",
                status="missing",
                warning="onchain_token_mapping_missing",
            )
        ]
    )
    policy = NextFetchPolicy(actionable_sources=("discovery", "onchain"))

    actions = build_next_fetch_actions(scores, coverage, policy=policy)

    assert actions.is_empty()


def test_scan_feedback_suppresses_ok_cache_notes() -> None:
    coverage = manifest_frame(
        [
            source_manifest_row(
                symbol="BTC-USDT-SWAP",
                source="trades",
                phase="collect-market",
                status="ok",
                warning="source=trade;refresh_transport=rest",
            )
        ]
    )

    out = render_scan_feedback(pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), coverage)

    assert "source=trade" not in out
    assert "No actionable coverage warnings recorded." in out


def test_scan_feedback_uses_latest_manifest_rows_for_coverage() -> None:
    coverage = manifest_frame(
        [
            source_manifest_row(
                symbol="BTC-USDT-SWAP",
                source="funding",
                phase="collect-market",
                status="missing",
                warning="funding_missing",
                timestamp=1,
            ),
            source_manifest_row(
                symbol="BTC-USDT-SWAP",
                source="funding",
                phase="collect-market",
                status="ok",
                timestamp=2,
            ),
            source_manifest_row(
                symbol="BTC-USDT-SWAP",
                source="books",
                phase="collect-market",
                status="skipped",
                timestamp=1,
            ),
            source_manifest_row(
                symbol="BTC-USDT-SWAP",
                source="books",
                phase="collect-market",
                status="skipped",
                timestamp=2,
            ),
        ]
    )

    out = render_scan_feedback(pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), coverage)

    assert "funding_missing" not in out
    assert "books skipped: book collection skipped by run mode (1 symbols)" in out


def test_scan_feedback_renders_skipped_books_as_availability() -> None:
    summary = pl.DataFrame(
        {
            "rank": [1],
            "symbol": ["BTC-USDT-SWAP"],
            "alert_level": ["none"],
            "confidence_level": ["low"],
            "score_total": [0],
            "suggestion_type": ["reject_or_deprioritize"],
            "top_positive_components": [""],
            "top_negative_filters": [""],
            "missing_evidence": ["book_missing"],
            "next_fetch_action": [""],
            "data_quality_warning": ["book_missing"],
        }
    )
    coverage = manifest_frame(
        [
            source_manifest_row(
                symbol="BTC-USDT-SWAP",
                source="books",
                phase="collect-market",
                status="skipped",
            )
        ]
    )

    out = render_scan_feedback(summary, pl.DataFrame(), pl.DataFrame(), coverage)

    assert "book collection skipped by run mode" in out
    assert "book missing" not in out


def test_scan_feedback_handles_empty_summary() -> None:
    out = render_scan_feedback(pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), pl.DataFrame())

    assert "Candidates reviewed: 0" in out
    assert "No candidates were scored." in out
    assert "No follow-up fetch actions queued." in out

