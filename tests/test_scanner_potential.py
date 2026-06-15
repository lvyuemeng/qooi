from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import qooi.scanner as scan
import qooi.scanner.workflow as potential
from qooi.exchange.discovery import DiscoveryResult, empty_discovery_frame
from qooi.scanner import feasibility as potential_feasibility
from qooi.scanner import ladder as potential_ladder
from qooi.scanner import outcome as potential_outcome
from qooi.scanner import rank as potential_rank
from qooi.scanner import report as potential_report
from qooi.scanner import state as potential_state
from qooi.scanner import transitions
from qooi.scanner.config import TransitionConfig
from qooi.scanner.state import STATE_FRAME_SCHEMA
from qooi.scanner.tailtree import _tailtree_outcome_by_decision, select_tail_leaves
from qooi.scanner.workflow import run


class _FakeLeafTree:
    def __init__(self, leaf_id: int) -> None:
        self.leaf_id = leaf_id

    def predict_leaf(self, features: pl.DataFrame) -> pl.DataFrame:
        return features.with_columns(pl.lit(self.leaf_id).cast(pl.Int32).alias("leaf_id"))


class _FakeScoreTreeForRank:
    def predict_score(self, features: pl.DataFrame) -> pl.DataFrame:
        return features.with_columns(pl.lit(0.95).alias("tailtree_score"))

class _ReportInputsForTest:
    def __init__(self, artifacts: scan.PotentialArtifacts) -> None:
        self.artifacts = artifacts


def _bullish_pattern() -> scan.TransitionPattern:
    return scan.TransitionPattern(
        symbol="BTC-USDT-SWAP",
        timeframe="1H",
        path="accumulation -> markup -> trend_continuation",
        event="none_in_trend",
        count=30,
        transition_probability=0.6,
        win_rate=0.7,
        average_forward_return_pct=1.2,
        omega=1.8,
        pwpr=0.45,
        transition_information_bits=0.1,
        conditional_transition_information_bits=0.2,
        direction="bullish",
        p_up=0.7,
        p_down=0.2,
        median_forward_return_pct=0.8,
        q10_forward_return_pct=-0.6,
        q25_forward_return_pct=-0.2,
        q75_forward_return_pct=1.6,
        q90_forward_return_pct=2.4,
        loss_stop_pct=0.6,
        profit_stop_pct=1.6,
        reward_risk=2.67,
        suggestion="rapid_trend_watch",
    )


def _state_row(
    family: str,
    direction: str,
    *,
    state: str | None = None,
    score: float = 0.6,
    evidence: str | None = None,
    reason: str = "",
    timestamp: int | None = 1,
) -> scan.SourceStateRow:
    return scan.SourceStateRow(
        "BTC-USDT-SWAP",
        family,
        timestamp,
        state or direction,
        direction,
        score,
        evidence or direction,
        reason,
        False,
    )


def _decision_bundle(
    *,
    kline: scan.SourceStateRow | None = None,
    transition: scan.SourceStateRow | None = None,
    books: scan.SourceStateRow | None = None,
    trades: scan.SourceStateRow | None = None,
    derivatives: scan.SourceStateRow | None = None,
    context: scan.SourceStateRow | None = None,
    patterns: tuple[scan.TransitionPattern, ...] = (_bullish_pattern(),),
) -> scan.SymbolStateBundle:
    return scan.SymbolStateBundle(
        symbol="BTC-USDT-SWAP",
        kline=kline
        or _state_row("kline", "bullish", state="uptrend/markup", evidence="kline bullish"),
        transition=transition
        or _state_row(
            "transition",
            "bullish",
            state="accumulation -> markup -> trend_continuation",
            score=0.7,
            evidence="transition bullish",
        ),
        books=books or _state_row("books", "neutral", state="balanced_book", score=0.5),
        trades=trades or _state_row("trades", "neutral", state="balanced_trade_flow", score=0.5),
        derivatives=derivatives
        or _state_row("derivatives", "neutral", state="mixed_derivatives", score=0.5),
        context=context
        or _state_row("context", "missing", state="context_missing", score=0.0, reason="missing"),
        coverage_notes=(),
        transition_patterns=patterns,
    )


def _observation(symbol: str, index: int, *, changed: bool) -> dict[str, object]:
    return {
        "symbol": symbol,
        "decision_timeframe": "1H",
        "decision_bar_close_ms": index + 1,
        "background_regime": "trend_background" if changed else "range_background",
        "background_structure": "trend",
        "background_range": "range_normal",
        "background_vol": "vol_normal",
        "swing_regime": "range",
        "swing_core": "range|coil",
        "swing_range": "range_tight",
        "swing_transition": "range|coil|same_context",
        "decision_direction": "neutral",
        "decision_regime": "range",
        "decision_core": "range|coil",
        "decision_range": "range_tight",
        "decision_vol": "vol_normal",
        "decision_event": "none_in_accumulation",
        "decision_event_age_bucket": "old",
        "decision_transition": "range|coil|same_context",
        "source_family": "open_interest",
        "source_state": "oi_expansion" if changed else "oi_flat",
        "source_direction": "neutral",
        "source_known_at_ms": index + 1,
        "source_age_ms": 0,
        "source_freshness": "fresh",
        "market_alignment": "background_swing_conflict",
        "source_market_alignment": "source_neutral",
        "risk_context": "range_tight|vol_normal",
    }


def _source_outcome(symbol: str, index: int, *, changed: bool) -> dict[str, object]:
    return {
        "symbol": symbol,
        "source_family": "open_interest",
        "source_state": "oi_expansion" if changed else "oi_flat",
        "source_direction": "neutral",
        "provider_timestamp_ms": index + 1,
        "known_at_ms": index + 1,
        "aligned_bar": "1H",
        "aligned_bar_close_ms": index + 1,
        "serialization_status": "stored_source_row",
        "outcome_horizon": 4,
        "close_at_event": 100.0,
        "future_close": 103.0 if changed else 99.0,
        "forward_return_pct": 3.0 if changed else -1.0,
        "forward_min_return_pct": -1.0,
        "forward_max_return_pct": 4.0 if changed else 1.0,
        "path_range_pct": 5.0 if changed else 2.0,
        "tail_asymmetry_pct": 3.0 if changed else 0.0,
        "outcome_available": True,
        "outcome_reason": "available",
    }


def _realized_transition(symbol: str, index: int, *, changed: bool) -> dict[str, object]:
    return {
        "symbol": symbol,
        "timeframe": "1H",
        "bar_close_ms": index + 1,
        "outcome_horizon": 4,
        "terminal_direction": "bullish" if changed else "neutral",
        "terminal_regime_state": "markup" if changed else "range",
        "terminal_structure_state": "trend" if changed else "coil",
        "terminal_core_context": "markup|trend" if changed else "range|coil",
        "terminal_transition_kind": "state_transition" if changed else "same_context",
        "direction_changed": changed,
        "regime_changed": changed,
        "structure_changed": changed,
        "core_context_changed": changed,
        "event_fired": False,
        "returned_to_origin": False,
        "time_to_direction_change_bars": 1 if changed else None,
        "time_to_core_change_bars": 1 if changed else None,
        "transition_count": 1 if changed else 0,
        "forward_return_pct": 3.0 if changed else -1.0,
        "forward_min_return_pct": -1.0,
        "forward_max_return_pct": 4.0 if changed else 1.0,
        "path_range_pct": 5.0 if changed else 2.0,
        "tail_asymmetry_pct": 3.0 if changed else 0.0,
        "time_to_max_bar": 1 if changed else 1,
        "time_to_min_bar": 2 if changed else 1,
        "close_retention_ratio": 0.75 if changed else None,
        "post_max_drawdown_pct": 1.0 if changed else 2.0,
        "post_min_rebound_pct": 4.0 if changed else 0.0,
        "path_efficiency": 0.6 if changed else 0.5,
    }


def test_realized_transition_frame_preserves_future_path_metrics() -> None:
    history = pl.DataFrame(
        {
            "symbol": ["BTC", "BTC", "BTC", "BTC", "ETH", "ETH"],
            "timeframe": ["1H", "1H", "1H", "1H", "4H", "4H"],
            "bar_close_ms": [1, 2, 3, 4, 1, 2],
            "close": [100.0, 105.0, 110.0, 103.0, 50.0, 55.0],
            "high": [101.0, 108.0, 115.0, 104.0, 51.0, 58.0],
            "low": [99.0, 104.0, 109.0, 95.0, 49.0, 54.0],
            "direction_hint": ["flat", "flat", "up", "flat", "down", "up"],
            "regime_state": ["range", "range", "trend", "range", "bear", "bull"],
            "structure_state": ["coil", "coil", "break", "coil", "drop", "rise"],
            "core_context": [
                "range|coil",
                "range|coil",
                "trend|break",
                "range|coil",
                "bear|drop",
                "bull|rise",
            ],
            "transition_kind": ["same", "same", "state", "state", "same", "state"],
            "event_state": [
                "none_event",
                "shock",
                "none_event",
                "none_event",
                "none_event",
                "none_event",
            ],
        }
    )

    rows = potential_outcome.realized_transition_frame(history, (1, 2)).sort(
        "symbol", "timeframe", "bar_close_ms", "outcome_horizon"
    )

    btc_h2 = rows.filter(
        (pl.col("symbol") == "BTC")
        & (pl.col("bar_close_ms") == 1)
        & (pl.col("outcome_horizon") == 2)
    ).row(0, named=True)
    eth_h1 = rows.filter(
        (pl.col("symbol") == "ETH")
        & (pl.col("bar_close_ms") == 1)
        & (pl.col("outcome_horizon") == 1)
    ).row(0, named=True)

    assert rows.height == 12
    assert btc_h2["terminal_core_context"] == "trend|break"
    assert btc_h2["time_to_direction_change_bars"] == 2
    assert btc_h2["time_to_core_change_bars"] == 2
    assert btc_h2["transition_count"] == 1
    assert btc_h2["event_fired"] is True
    assert btc_h2["returned_to_origin"] is False
    assert btc_h2["forward_return_pct"] == 10.0
    assert btc_h2["forward_min_return_pct"] == 4.0
    assert btc_h2["forward_max_return_pct"] == 15.0
    assert btc_h2["path_range_pct"] == 11.0
    assert btc_h2["time_to_max_bar"] == 2
    assert btc_h2["time_to_min_bar"] == 1
    assert btc_h2["close_retention_ratio"] == 10.0 / 15.0
    assert btc_h2["post_max_drawdown_pct"] == 5.0
    assert btc_h2["post_min_rebound_pct"] == 6.0
    assert btc_h2["path_efficiency"] == 10.0 / 11.0
    assert eth_h1["direction_changed"] is True
    assert eth_h1["core_context_changed"] is True
    assert eth_h1["forward_return_pct"] == 10.0
    assert eth_h1["forward_max_return_pct"] == 16.0


def test_potential_outcome_frame_preserves_market_forward_excursions() -> None:
    outcome = potential_outcome.potential_outcome_frame(
        pl.DataFrame(
            [_observation("BTC-USDT-SWAP", 0, changed=True)],
            schema=potential_state.POTENTIAL_OBSERVATION_SCHEMA,
        ),
        pl.DataFrame(schema=potential_outcome.SOURCE_OUTCOME_SCHEMA),
        pl.DataFrame(
            [_realized_transition("BTC-USDT-SWAP", 0, changed=True)],
            schema=potential_outcome.REALIZED_TRANSITION_SCHEMA,
        ),
        return_threshold_pct=3.5,
    )

    row = outcome.row(0, named=True)
    assert row["forward_return_pct"] == 3.0
    assert row["forward_min_return_pct"] == -1.0
    assert row["forward_max_return_pct"] == 4.0
    assert row["path_range_pct"] == 5.0
    assert row["time_to_max_bar"] == 1
    assert row["time_to_min_bar"] == 2
    assert row["close_retention_ratio"] == 0.75
    assert row["post_max_drawdown_pct"] == 1.0
    assert row["post_min_rebound_pct"] == 4.0
    assert row["path_efficiency"] == 0.6
    assert row["tail_up"] is True
    assert row["tail_down"] is False


def _selected_evidence_for_test() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "evidence_level": ["market_decision_source_risk"],
            "outcome_horizon": [4],
            "background_regime": ["trend_background"],
            "swing_core": ["range|coil"],
            "decision_core": ["range|coil"],
            "decision_transition": ["range|coil|same_context"],
            "source_family": ["open_interest"],
            "source_state": ["oi_expansion"],
            "risk_context": ["range_tight|vol_normal"],
            "conditioned_observations": [120],
            "symbol_count": [24],
            "conditioned_p_up": [0.65],
            "conditioned_p_down": [0.20],
            "conditioned_p_flat": [0.15],
            "lift_up": [0.20],
            "lift_down": [-0.10],
            "lift_flat": [-0.10],
            "information_gain_bits": [0.12],
            "transition_information_gain_bits": [0.08],
            "tail_up_rate": [0.30],
            "tail_down_rate": [0.08],
            "avg_forward_max_return_pct": [4.0],
            "avg_forward_min_return_pct": [-1.0],
            "avg_path_range_pct": [5.0],
            "path_skew": [0.22],
            "returned_to_origin_rate": [0.10],
            "information_stability": [0.80],
            "transition_information_stability": [0.70],
            "selected_evidence_level": [True],
            "statistical_direction": ["up"],
            "research_suggestion": ["rapid_trend_watch"],
            "evidence_status": ["usable_stable_information"],
            "transition_status": ["usable_stable_transition_information"],
        }
    )


def test_candidate_evidence_matches_latest_observation_to_selected_evidence() -> None:
    observations = pl.DataFrame(
        [
            _observation("BTC-USDT-SWAP", 1, changed=True),
            _observation("BTC-USDT-SWAP", 2, changed=True),
        ],
        schema=potential_state.POTENTIAL_OBSERVATION_SCHEMA,
    )

    candidates = potential_rank.candidate_evidence_frame(
        observations, _selected_evidence_for_test()
    )

    assert candidates.height == 1
    row = candidates.row(0, named=True)
    assert row["symbol"] == "BTC-USDT-SWAP"
    assert row["decision_bar_close_ms"] == 3
    assert row["matched_evidence_level"] == "market_decision_source_risk"
    assert row["research_suggestion"] == "rapid_trend_watch"
    assert row["candidate_status"] == "matched_evidence"


def test_candidate_evidence_scores_all_tailtree_horizon_models() -> None:
    observations = pl.DataFrame(
        [_observation("BTC-USDT-SWAP", 1, changed=True)],
        schema=potential_state.POTENTIAL_OBSERVATION_SCHEMA,
    )
    evidence = pl.DataFrame(
        {
            "outcome_horizon": [6, 12],
            "tree_direction": ["up", "up"],
            "leaf_id": [6, 12],
            "selected_evidence_level": [True, True],
            "tail_lift": [1.6, 2.4],
        }
    )

    candidates = potential_rank.candidate_evidence_frame(
        observations,
        evidence,
        tree_models={(6, "up"): _FakeLeafTree(6), (12, "up"): _FakeLeafTree(12)},
    )

    assert candidates.height == 2
    rows = {row["outcome_horizon"]: row for row in candidates.iter_rows(named=True)}
    assert set(rows) == {6, 12}
    assert rows[6]["matched_evidence_level"] == "tree_up"
    assert rows[12]["tail_lift"] == 2.4


def test_candidate_evidence_matches_tailtree_score_bucket_models() -> None:
    observations = pl.DataFrame(
        [_observation("BTC-USDT-SWAP", 1, changed=True)],
        schema=potential_state.POTENTIAL_OBSERVATION_SCHEMA,
    )
    evidence = pl.DataFrame(
        {
            "outcome_horizon": [6],
            "tree_direction": ["up"],
            "score_bucket": ["top_5pct"],
            "score_min": [0.90],
            "score_max": [1.0],
            "selected_evidence_level": [True],
            "tail_lift": [2.7],
            "N_total": [100],
            "N_tail_exceedances": [50],
        }
    )

    candidates = potential_rank.candidate_evidence_frame(
        observations,
        evidence,
        tree_models={(6, "up"): _FakeScoreTreeForRank()},
    )

    assert candidates.height == 1
    row = candidates.row(0, named=True)
    assert row["score_bucket"] == "top_5pct"
    assert row["matched_evidence_level"] == "tree_up"
    assert row["tail_lift"] == pytest.approx(2.7)
    assert row["N_total"] == 100
    assert row["candidate_status"] == "matched_evidence"


def test_candidate_horizon_consistency_counts_agreement_without_mean_cancellation() -> None:
    ranked = pl.DataFrame(
        {
            "symbol": ["BTC", "BTC", "BTC", "BTC"],
            "decision_timeframe": ["1H", "1H", "1H", "1H"],
            "tree_direction": ["up", "up", "down", "down"],
            "outcome_horizon": [6, 12, 6, 24],
            "rank_score": [8.0, 7.0, 9.0, 2.0],
            "tail_lift": [2.4, 2.1, 2.5, 1.1],
            "N_tail_exceedances": [60, 40, 70, 10],
            "candidate_status": [
                "matched_evidence",
                "matched_evidence",
                "matched_evidence",
                "matched_evidence",
            ],
        }
    )

    panel = potential_rank.candidate_horizon_consistency_frame(ranked)

    assert set(panel.get_column("tree_direction")) == {"up", "down"}
    rows = {row["tree_direction"]: row for row in panel.iter_rows(named=True)}
    assert rows["up"]["horizon_count"] == 2
    assert rows["up"]["strong_horizon_count"] == 2
    assert rows["up"]["best_outcome_horizon"] == 6
    assert rows["up"]["best_rank_score"] == pytest.approx(8.0)
    assert rows["up"]["opposite_direction_count"] == 2
    assert rows["up"]["opposite_direction_best_rank_score"] == pytest.approx(9.0)
    assert rows["up"]["conflict_penalty_score"] > 0.0
    assert "mean_rank_score" not in panel.columns
    assert "mean_tailtree_score" not in panel.columns


def test_candidate_horizon_consistency_counts_each_horizon_once() -> None:
    ranked = pl.DataFrame(
        {
            "symbol": ["BTC", "BTC", "BTC"],
            "decision_timeframe": ["1H", "1H", "1H"],
            "tree_direction": ["down", "down", "down"],
            "outcome_horizon": [6, 6, 12],
            "rank_score": [3.0, 8.0, 4.0],
            "tail_lift": [2.0, 2.5, 1.6],
            "N_tail_exceedances": [10, 20, 30],
            "candidate_status": ["matched_evidence", "matched_evidence", "matched_evidence"],
        }
    )

    panel = potential_rank.candidate_horizon_consistency_frame(ranked)

    row = panel.row(0, named=True)
    assert row["horizon_count"] == 2
    assert row["strong_horizon_count"] == 2
    assert row["best_outcome_horizon"] == 6
    assert row["best_rank_score"] == pytest.approx(8.0)


def test_candidate_evidence_combines_matched_and_unmatched_latest_observations() -> None:
    observations = pl.DataFrame(
        [
            _observation("BTC-USDT-SWAP", 1, changed=True),
            _observation("ETH-USDT-SWAP", 1, changed=False),
        ],
        schema=potential_state.POTENTIAL_OBSERVATION_SCHEMA,
    )

    candidates = potential_rank.candidate_evidence_frame(
        observations, _selected_evidence_for_test()
    )

    assert candidates.height == 2
    rows = {row["symbol"]: row for row in candidates.iter_rows(named=True)}
    assert rows["BTC-USDT-SWAP"]["candidate_status"] == "matched_evidence"
    assert rows["ETH-USDT-SWAP"]["candidate_status"] == "no_matching_evidence"


def test_candidate_evidence_emits_unmatched_latest_observation_caveat() -> None:
    observations = pl.DataFrame(
        [_observation("BTC-USDT-SWAP", 1, changed=True)],
        schema=potential_state.POTENTIAL_OBSERVATION_SCHEMA,
    )
    evidence = pl.DataFrame(schema={"selected_evidence_level": pl.Boolean})

    candidates = potential_rank.candidate_evidence_frame(observations, evidence)

    assert candidates.height == 1
    row = candidates.row(0, named=True)
    assert row["symbol"] == "BTC-USDT-SWAP"
    assert row["candidate_status"] == "no_selected_evidence"
    assert row["matched_evidence_level"] is None


def test_rank_candidate_evidence_exposes_components_without_trading_signal() -> None:
    observations = pl.DataFrame(
        [_observation("BTC-USDT-SWAP", 1, changed=True)],
        schema=potential_state.POTENTIAL_OBSERVATION_SCHEMA,
    )
    candidates = potential_rank.candidate_evidence_frame(
        observations, _selected_evidence_for_test()
    )

    ranked = potential_rank.rank_candidate_evidence(candidates)

    assert "rank_score" in ranked.columns
    assert "rank_information_component" in ranked.columns
    assert "rank_transition_component" in ranked.columns
    assert "rank_tail_component" in ranked.columns
    assert "rank_path_component" in ranked.columns
    assert "rank_stability_component" in ranked.columns
    assert "rank_quality_component" in ranked.columns
    assert "rank_penalty_component" in ranked.columns
    assert "entry_signal" not in ranked.columns
    assert "position_signal" not in ranked.columns


def test_rank_candidate_evidence_penalizes_required_source_gaps_not_optional_absence() -> None:
    candidates = pl.DataFrame(
        [
            {
                "symbol": "BTC-USDT-SWAP",
                "decision_timeframe": "1H",
                "decision_bar_close_ms": 1,
                "outcome_horizon": 12,
                "matched_evidence_level": "market_decision_source",
                "candidate_status": "matched_evidence",
                "statistical_direction": "bullish",
                "research_suggestion": "review",
                "conditioned_observations": 100,
                "symbol_count": 20,
                "conditioned_p_up": 0.4,
                "conditioned_p_down": 0.2,
                "conditioned_p_flat": 0.4,
                "lift_up": 1.4,
                "lift_down": 0.7,
                "lift_flat": 1.0,
                "information_gain_bits": 0.5,
                "transition_information_gain_bits": 0.2,
                "tail_up_rate": 0.1,
                "tail_down_rate": 0.05,
                "avg_forward_max_return_pct": 4.0,
                "avg_forward_min_return_pct": -2.0,
                "avg_path_range_pct": 6.0,
                "path_skew": 0.3,
                "returned_to_origin_rate": 0.2,
                "information_stability": 1.0,
                "transition_information_stability": 1.0,
                "evidence_status": "selected",
                "transition_status": "supported",
                "source_freshness": "fresh",
                "source_age_ms": 0,
                "market_alignment": "aligned",
                "source_market_alignment": "source_aligned",
                "required_missing_source_count": 1,
                "required_stale_source_count": 2,
                "provider_bounded_source_count": 3,
                "optional_absent_source_count": 1,
            }
        ]
    )

    ranked = potential_rank.rank_candidate_evidence(candidates)
    row = ranked.row(0, named=True)

    assert "required_missing_source_count" in ranked.columns
    assert "provider_bounded_source_count" in ranked.columns
    assert "optional_absent_source_count" in ranked.columns
    assert row["source_penalty_score"] == pytest.approx(2.6)
    assert row["rank_penalty_component"] == pytest.approx(2.6)


def test_rank_candidate_evidence_exposes_profit_proxy_and_promotion_score() -> None:
    candidates = pl.DataFrame(
        [
            {
                "symbol": "BTC-USDT-SWAP",
                "decision_timeframe": "1H",
                "decision_bar_close_ms": 1,
                "outcome_horizon": 12,
                "matched_evidence_level": "score_bucket",
                "candidate_status": "matched_evidence",
                "statistical_direction": "up",
                "research_suggestion": "score_bucket_tail_utility",
                "conditioned_observations": 100,
                "symbol_count": 20,
                "conditioned_p_up": 0.4,
                "conditioned_p_down": 0.2,
                "conditioned_p_flat": 0.4,
                "lift_up": 1.4,
                "lift_down": 0.7,
                "lift_flat": 1.0,
                "information_gain_bits": 0.5,
                "transition_information_gain_bits": 0.2,
                "tail_up_rate": 0.1,
                "tail_down_rate": 0.05,
                "avg_forward_max_return_pct": 4.0,
                "avg_forward_min_return_pct": -2.0,
                "avg_path_range_pct": 6.0,
                "path_skew": 0.3,
                "returned_to_origin_rate": 0.2,
                "information_stability": 1.0,
                "transition_information_stability": 1.0,
                "evidence_status": "selected",
                "transition_status": "supported",
                "source_freshness": "fresh",
                "source_age_ms": 0,
                "market_alignment": "aligned",
                "source_market_alignment": "source_aligned",
                "required_missing_source_count": 0,
                "required_stale_source_count": 0,
                "provider_bounded_source_count": 0,
                "optional_absent_source_count": 0,
                "tail_lift": 2.0,
                "N_tail_exceedances": 50,
                "tail_utility_mean": 8.0,
                "tail_utility_p90": 10.0,
            }
        ]
    )

    ranked = potential_rank.rank_candidate_evidence(candidates)
    row = ranked.row(0, named=True)

    assert "tail_utility_mean" in ranked.columns
    assert "profit_proxy_score" in ranked.columns
    assert "promotion_score" in ranked.columns
    assert row["profit_proxy_score"] == pytest.approx(8.0)
    assert row["profit_proxy_per_selected_obs"] == pytest.approx(8.0)
    assert row["profit_proxy_per_1k_observed"] == pytest.approx(80.0)
    assert row["promotion_score"] > row["rank_score"]


def test_promotion_score_does_not_subtract_source_or_external_penalties() -> None:
    candidates = pl.DataFrame(
        [
            {
                "symbol": "BTC-USDT-SWAP",
                "decision_timeframe": "1H",
                "decision_bar_close_ms": 1,
                "outcome_horizon": 12,
                "matched_evidence_level": "score_bucket",
                "candidate_status": "matched_evidence",
                "statistical_direction": "up",
                "research_suggestion": "score_bucket_tail_utility",
                "conditioned_observations": 100,
                "symbol_count": 20,
                "conditioned_p_up": 0.4,
                "conditioned_p_down": 0.2,
                "conditioned_p_flat": 0.4,
                "lift_up": 1.4,
                "lift_down": 0.7,
                "lift_flat": 1.0,
                "information_gain_bits": 0.5,
                "transition_information_gain_bits": 0.2,
                "tail_up_rate": 0.1,
                "tail_down_rate": 0.05,
                "avg_forward_max_return_pct": 4.0,
                "avg_forward_min_return_pct": -2.0,
                "avg_path_range_pct": 6.0,
                "path_skew": 0.3,
                "returned_to_origin_rate": 0.2,
                "information_stability": 1.0,
                "transition_information_stability": 1.0,
                "evidence_status": "selected",
                "transition_status": "supported",
                "source_freshness": "fresh",
                "source_age_ms": 0,
                "market_alignment": "aligned",
                "source_market_alignment": "source_aligned",
                "required_missing_source_count": 2,
                "required_stale_source_count": 3,
                "provider_bounded_source_count": 0,
                "optional_absent_source_count": 0,
                "tail_lift": 2.0,
                "N_tail_exceedances": 50,
                "tail_utility_mean": 8.0,
                "tail_utility_p90": 10.0,
            }
        ]
    )

    ranked = potential_rank.rank_candidate_evidence(candidates)
    row = ranked.row(0, named=True)

    opportunity_score = (
        row["profit_proxy_score"]
        + row["rank_tail_component"]
        + row["rank_path_component"]
        + row["rank_stability_component"]
    )
    assert row["source_penalty_score"] == pytest.approx(4.9)
    assert row["rank_score"] < opportunity_score
    assert row["promotion_score"] == pytest.approx(opportunity_score)


def test_scanner_architecture_excludes_execution_cost_from_internal_promotion() -> None:
    architecture = Path("docs/architecture/scanner.md").read_text(encoding="utf-8")
    graph = Path("docs/graph/scanner.md").read_text(encoding="utf-8")
    acceptance = architecture.split("Acceptance rule:", 1)[1].split(
        "Universe reproducibility", 1
    )[0]

    assert "estimated_cost_slippage_penalty" not in acceptance
    assert "data_cost_penalty" not in acceptance
    assert "cost_adjusted_score" not in graph
    assert "cost_penalty" not in graph


def test_candidate_feasibility_frame_selects_best_ranked_row_per_symbol() -> None:
    candidate_rank = pl.DataFrame(
        [
            {
                "symbol": "AAA-USDT-SWAP",
                "outcome_horizon": 6,
                "rank_score": 20.0,
                "promotion_score": 1.0,
                "profit_proxy_score": 0.5,
                "profit_proxy_per_selected_obs": 0.5,
                "profit_proxy_per_1k_observed": 50.0,
                "tail_utility_mean": 0.5,
                "tail_utility_p90": 1.0,
                "source_penalty_score": 0.3,
                "required_missing_source_count": 0,
                "required_stale_source_count": 1,
                "provider_bounded_source_count": 3,
                "optional_absent_source_count": 1,
                "tree_direction": "up",
                "matched_evidence_level": "tree_up",
                "tail_lift": 1.4,
                "gpd_shape_xi": 0.10,
                "N_tail_exceedances": 40,
                "rank_reason": "weaker",
            },
            {
                "symbol": "AAA-USDT-SWAP",
                "outcome_horizon": 12,
                "rank_score": 10.0,
                "promotion_score": 5.0,
                "profit_proxy_score": 2.0,
                "profit_proxy_per_selected_obs": 2.0,
                "profit_proxy_per_1k_observed": 200.0,
                "tail_utility_mean": 2.0,
                "tail_utility_p90": 3.0,
                "source_penalty_score": 0.1,
                "required_missing_source_count": 0,
                "required_stale_source_count": 0,
                "provider_bounded_source_count": 3,
                "optional_absent_source_count": 1,
                "tree_direction": "down",
                "matched_evidence_level": "tree_down",
                "tail_lift": 2.5,
                "gpd_shape_xi": 0.20,
                "N_tail_exceedances": 60,
                "rank_reason": "best",
            },
        ]
    )
    watchlist = pl.DataFrame(
        [
            {
                "symbol": "AAA-USDT-SWAP",
                "watchlist_feasibility": "reviewable",
                "min_history_coverage_pct": 100.0,
                "min_source_capability_coverage_pct": 99.0,
                "source_status": "source_context_available",
                "history_status": "reviewable_history",
            }
        ]
    )

    frame = potential_feasibility.candidate_feasibility_frame(candidate_rank, watchlist)

    assert frame.height == 1
    row = frame.row(0, named=True)
    assert row["symbol"] == "AAA-USDT-SWAP"
    assert row["outcome_horizon"] == 12
    assert row["rank_score"] == pytest.approx(10.0)
    assert row["promotion_score"] == pytest.approx(5.0)
    assert row["profit_proxy_score"] == pytest.approx(2.0)
    assert row["tree_direction"] == "down"
    assert row["rank_tier"] == "1"
    assert row["candidate_reason"] == "reviewable"
    assert row["watchlist_feasibility"] == "reviewable"


def test_candidate_rank_pipe_requires_outcome_horizon() -> None:
    candidate_rank = pl.DataFrame(
        {
            "symbol": ["AAA-USDT-SWAP"],
            "rank_score": [1.0],
            "source_penalty_score": [0.0],
            "required_missing_source_count": [0],
            "required_stale_source_count": [0],
            "provider_bounded_source_count": [0],
            "optional_absent_source_count": [0],
            "tree_direction": ["up"],
            "matched_evidence_level": ["tree_up"],
            "tail_lift": [1.0],
            "gpd_shape_xi": [0.1],
            "N_tail_exceedances": [30],
            "rank_reason": ["missing_horizon"],
        }
    )

    with pytest.raises(ValueError, match="candidate evidence pipe missing outcome_horizon"):
        potential_rank.rank_candidate_evidence(candidate_rank)


def test_tailtree_report_summary_uses_horizon_run_summary(tmp_path: Path) -> None:
    diagnostics_dir = tmp_path / "diagnostics"
    diagnostics_dir.mkdir()
    pl.DataFrame(
        {
            "summary_scope": ["run", "up", "down", "run", "up", "down"],
            "objective": ["tail_utility_quantile"] * 6,
            "outcome_horizon": [6, 6, 6, 12, 12, 12],
            "trained_tree_count": [2, 2, 2, 1, 1, 1],
            "train_tail_count": [30, 20, 10, 40, 40, 0],
            "valid_tail_lift": [1.5, 2.0, 1.2, 2.5, 2.5, 0.0],
            "tail_utility_mean": [0.7, 0.8, 0.5, 0.6, 0.6, 0.0],
            "valid_selected_utility_mean": [0.9, 1.1, 0.4, 1.0, 1.0, 0.0],
            "written_model_file_count": [4, 4, 4, 2, 2, 2],
            "written_evidence_file_count": [4, 4, 4, 2, 2, 2],
        }
    ).write_csv(diagnostics_dir / "tailtree-run-summary.csv")
    inputs = _ReportInputsForTest(
        artifacts=scan.PotentialArtifacts(
            report=tmp_path / "report.md",
            diagnostics_dir=diagnostics_dir,
            states_dir=tmp_path / "states",
        )
    )

    rendered = potential_report._TreeSummarySection().render(inputs, object())

    assert "| H | Scope | Obj | Trees | TrainTail | ValidLift | UtilMean |" in rendered
    assert (
        "| 6 | up | tail_utility_quantile | 2 | 20 | 2.0000 | "
        "0.8000 | 1.1000 | 4 | 4 |" in rendered
    )
    assert "Tree_UP: not trained" not in rendered


def test_tailtree_selection_efficiency_report_projects_budget_winners() -> None:
    class _Frames:
        tailtree_selection_efficiency = pl.DataFrame(
            {
                "universe_snapshot_id": ["u", "u"],
                "model_tag": ["tag", "tag"],
                "objective": ["tail_utility_quantile", "tail_utility_quantile"],
                "training_profile": ["balanced_baseline", "balanced_baseline"],
                "outcome_horizon": [12, 12],
                "tree_direction": ["up", "up"],
                "budget_family": ["top_k", "top_k"],
                "budget_value": [1.0, 3.0],
                "selected_symbol_count": [1, 3],
                "selected_observation_count": [1, 3],
                "selected_observation_rate": [0.01, 0.03],
                "selected_tail_count": [4, 6],
                "selected_tail_per_1k_obs": [900.0, 120.0],
                "valid_tail_lift": [1.2, 2.6],
                "selected_profit_proxy_mean": [0.3, 1.0],
                "selected_profit_proxy_p90": [0.4, 1.4],
                "selected_utility_mean": [0.3, 1.0],
                "selected_utility_p90": [0.4, 1.4],
                "profit_proxy_per_selected_obs": [0.3, 1.0],
                "profit_proxy_per_1k_observed": [0.03, 0.30],
                "hpo_score": [9999.0, 20.0],
                "promotion_threshold_pass_int": [1, 1],
                "feasibility_pass_int": [0, 1],
            }
        )

    rendered = potential_report._TreeSelectionEfficiencySection().render(object(), _Frames())

    assert "## Tail Tree Selection Efficiency" in rendered
    assert "Winner=normalized opportunity score" in rendered
    assert (
        "| H | Dir | Obj | Profile | Budget | Feas | Win | Proxy/Obs | Proxy/1k | "
        "Lift | Obs | Sel | Tail |" in rendered
    )
    assert (
        "| 12 | up | tail_utility_quantile | balanced_baseline | top_k=3.0000 | 1 |"
        in rendered
    )


def test_report_candidate_selection_uses_typed_rows_not_opaque_dicts() -> None:
    source = Path("src/qooi/scanner/report.py").read_text(encoding="utf-8")
    assert "class CandidateSelectionRow" in source
    assert "class CandidateSelectionSection" in source
    candidate_section = source.split("class CandidateSelectionSection", 1)[1].split(
        "class _CaveatsSection", 1
    )[0]
    assert "row.get(" not in candidate_section
    assert "float(str(" not in candidate_section
    assert "dict[str, object]" not in candidate_section


def test_tailtree_report_summary_does_not_read_legacy_artifact_names() -> None:
    source = Path("src/qooi/scanner/report.py").read_text(encoding="utf-8")
    section = source.split("class _TreeSummarySection", 1)[1].split(
        "class _TreeImportanceSection", 1
    )[0]

    assert "tail-tree-up.json" not in section
    assert "potential-leaf-evidence-h" not in section
    assert "legacy_model_paths" not in section


def test_potential_run_writes_report_and_diagnostics_without_trading_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    report_path = tmp_path / "potential" / "report.md"
    config = tmp_path / "potential.toml"
    config.write_text(
        f'''
[potential]
output = "{report_path.as_posix()}"
transition_context_limit = 0

[potential.profile]
mode = "stage"
''',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        potential,
        "discover_candidates",
        lambda *_args, **_kwargs: DiscoveryResult((), empty_discovery_frame(), pl.DataFrame()),
    )

    written = run(config)

    report = report_path.read_text(encoding="utf-8")
    diagnostics = report_path.parent / "diagnostics"
    states = report_path.parent / "states"
    assert written == report_path
    assert "# Potential Altcoin Diagnostics Report" in report
    assert "research-only evidence report" in report
    assert "place orders" in report
    assert "mutate baskets" in report
    assert "## Unified Evidence Surface" in report
    assert "## Data Health Summary" in report
    assert "## Candidate Selection" in report
    assert "## Horizon Consistency" in report
    assert "## Decision Rule Audit" not in report
    assert "Tiers: 1=top-decile" not in report
    assert (
        "| Symbol | H | Feas | Promo | Proxy | P/Obs | P/1k | Util | "
        "Rank | SrcPen | Miss | Stale | Bound | Opt | "
        "Hist% | Cap% | Tree | TailLift | ξ | Reason |" in report
    )
    assert (diagnostics / "coverage.csv").exists()
    assert (diagnostics / "source-freshness.csv").exists()
    assert (diagnostics / "source-capability.csv").exists()
    assert (diagnostics / "potential-observation-summary.csv").exists()
    assert (diagnostics / "potential-evidence-summary.csv").exists()
    assert (diagnostics / "potential-evidence-selected.csv").exists()
    assert (diagnostics / "candidate-inspection.csv").exists()
    assert (diagnostics / "candidate-rank.csv").exists()
    assert (diagnostics / "candidate-horizon-consistency.csv").exists()
    assert (diagnostics / "candidate-feasibility.csv").exists()
    profile = report_path.parent / "profile"
    assert (profile / "stages.csv").exists()
    assert (profile / "frames.csv").exists()
    assert (profile / "summary.md").exists()
    profile_stages = pl.read_csv(profile / "stages.csv")
    profile_frames = pl.read_csv(profile / "frames.csv")
    assert "write_diagnostics" in profile_stages.get_column("stage").to_list()
    assert "realized_transitions" in profile_frames.get_column("frame").to_list()
    assert not (diagnostics / "candidate-evidence.csv").exists()
    assert (states / "kline-state.csv").exists()
    assert not (diagnostics / "potential-observation.csv").exists()
    assert not (diagnostics / "potential-evidence.csv").exists()
    assert not (diagnostics / "evidence-backtest.csv").exists()
    assert not (diagnostics / "evidence-backtest-summary.csv").exists()
    assert not (diagnostics / "evidence-baselines.csv").exists()
    assert not (diagnostics / "kline-path-history.csv").exists()
    assert not (diagnostics / "realized-transition.csv").exists()
    (diagnostics / "candidate-rank.parquet").write_text("stale", encoding="utf-8")
    (states / "kline-state.parquet").write_text("stale", encoding="utf-8")
    (diagnostics / "potential-observation.csv").write_text("stale", encoding="utf-8")
    (diagnostics / "evidence-backtest.csv").write_text("stale", encoding="utf-8")
    (diagnostics / "evidence-baselines.csv").write_text("stale", encoding="utf-8")

    second_written = run(config)

    assert second_written == report_path
    assert (diagnostics / "candidate-rank.csv").exists()
    assert (states / "kline-state.csv").exists()
    assert not (diagnostics / "candidate-rank.parquet").exists()
    assert not (states / "kline-state.parquet").exists()
    assert not (diagnostics / "potential-observation.csv").exists()
    assert not (diagnostics / "evidence-backtest.csv").exists()
    assert not (diagnostics / "evidence-baselines.csv").exists()
    assert not (report_path.parent / "research-board.csv").exists()


def test_potential_config_rejects_legacy_aliases(
    tmp_path: Path,
) -> None:
    legacy_path = tmp_path / "legacy.toml"
    legacy_path.write_text(
        """
[run]
universe = "legacy-research"
out = "data/output/accumulation/mvp"

[market]
bar = "4H"
days = 30

[sources.disabled]
families = ["messages"]
""",
        encoding="utf-8",
    )
    current_path = tmp_path / "current.toml"
    current_path.write_text(
        """
[potential]
output = "data/output/potential/report.md"
universe = "research"
bar = "4H"
days = 30
refresh_mode = "incremental"
fetch_concurrency = 8

[potential.transition]
scan_budget = 80
context_scope = "all_scanned"
context_limit = 80
history_days = 365
ngram_length = 4
horizon = 8
min_information_bits = 0.01

[potential.review]
require_context = true

[potential.source]
max_staleness_hours = 12
trade_limit = 50
funding_limit = 60
rubik_period = "4H"
rubik_limit = 70
rubik_taker_unit = "1"
disabled_sources = ["messages"]
disabled_symbols = ["BAD-USDT-SWAP"]
""",
        encoding="utf-8",
    )

    legacy = potential.load_config(legacy_path)
    current = potential.load_config(current_path)

    assert legacy.output == Path("data/output/potential/report.md")
    assert legacy.universe == "research"
    assert legacy.bar == "1H"
    assert legacy.source.disabled_sources == ()
    assert current.output == Path("data/output/potential/report.md")
    assert current.refresh_mode == "incremental"
    assert current.fetch_concurrency == 8
    assert current.transition.scan_budget == 80
    assert current.transition.context_scope == "all_scanned"
    assert current.transition.context_limit == 80
    assert current.transition.history_days == 365
    assert current.transition.ngram_length == 4
    assert current.transition.horizon == 8
    assert current.transition.min_information_bits == 0.01
    assert current.review.require_context is True
    assert current.source.max_staleness_hours == 12
    assert current.source.trade_limit == 50
    assert current.source.funding_limit == 60
    assert current.source.rubik_period == "4H"
    assert current.source.rubik_limit == 70
    assert current.source.rubik_taker_unit == "1"
    assert current.source.disabled_sources == ("messages",)
    assert current.source.disabled_symbols == ("BAD-USDT-SWAP",)


def test_universe_context_and_min_bar_selection_respect_scanner_config(monkeypatch) -> None:
    discovery = pl.DataFrame(
        {
            "symbol": ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"],
            "rank_score": [10.0, 9.0, 8.0],
        }
    )
    monkeypatch.setattr(
        potential,
        "discover_candidates",
        lambda *_args, **_kwargs: DiscoveryResult(
            ("BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"),
            discovery,
            pl.DataFrame(),
        ),
    )

    universe = potential.resolve_universe(
        potential.PotentialConfig(transition=TransitionConfig(scan_budget=2))
    )
    all_context = scan.context_symbols(
        potential.PotentialConfig(transition=TransitionConfig(context_scope="all_scanned")),
        universe.symbols,
        {},
    )
    no_patterns = {
        symbol: scan.TransitionInsight(symbol, _state_row("transition", "missing"), ())
        for symbol in universe.symbols
    }

    assert universe.eligible_count == 3
    assert universe.symbols == ("BTC-USDT-SWAP", "ETH-USDT-SWAP")
    assert "OKX swap universe" in universe.selection_note
    assert all_context == universe.symbols
    assert scan.context_symbols(potential.PotentialConfig(), universe.symbols, no_patterns) == ()
    assert potential.target_min_bars(10, "15m") == 960
    assert potential.target_min_bars(10, "4H") == 120


def test_source_events_are_known_at_close_and_exclude_availability_states() -> None:
    bars = pl.DataFrame(
        {
            "symbol": ["BTC-USDT-SWAP", "BTC-USDT-SWAP", "BTC-USDT-SWAP"],
            "timestamp": [1, 2, 3],
            "open": [100.0, 100.0, 98.0],
            "high": [101.0, 101.0, 99.0],
            "low": [99.0, 97.0, 94.0],
            "close": [100.0, 98.0, 95.0],
        }
    )
    source_frames = {
        "open_interest": pl.DataFrame(
            {
                "symbol": ["BTC-USDT-SWAP", "BTC-USDT-SWAP", "BTC-USDT-SWAP"],
                "timestamp": [1, 2, 3],
                "open_interest_usd": [1000.0, 1100.0, 1200.0],
            }
        ),
        "taker_volume": pl.DataFrame(
            {
                "symbol": ["BTC-USDT-SWAP", "BTC-USDT-SWAP"],
                "timestamp": [2, 3],
                "taker_buy_volume": [10.0, 2.0],
                "taker_sell_volume": [2.0, 10.0],
            }
        ),
        "long_short_ratios": pl.DataFrame(
            {
                "symbol": ["BTC-USDT-SWAP", "BTC-USDT-SWAP"],
                "timestamp": [2, 3],
                "long_short_account_ratio": [1.0, 1.2],
            }
        ),
        "messages": pl.DataFrame(
            {"symbol": ["BTC-USDT-SWAP"], "timestamp": [2], "text": ["headline"]}
        ),
    }

    events = potential_outcome.source_events_frame(source_frames, bars, "1H")
    states = set(events.get_column("source_state").to_list())
    assert "short_buildup_with_price_down" in states
    assert "taker_buy_trap" in states
    assert "taker_sell_continuation" in states
    assert "crowded_longs_price_down" in states
    assert not any(str(state).endswith("_observed") for state in states)
    assert "message_observed" not in states


def test_source_outcomes_predictability_and_timeliness_report_missing_futures() -> None:
    events = pl.DataFrame(
        {
            "symbol": ["BTC-USDT-SWAP", "BTC-USDT-SWAP"],
            "source_family": ["trades", "trades"],
            "source_state": ["aggressive_sell_dominance", "aggressive_sell_dominance"],
            "source_direction": ["bearish", "bearish"],
            "provider_timestamp_ms": [1, 2],
            "known_at_ms": [1, 2],
            "aligned_bar": ["1H", "1H"],
            "aligned_bar_close_ms": [1, 2],
            "serialization_status": ["historical_event", "historical_event"],
        },
        schema=potential_outcome.SOURCE_EVENT_SCHEMA,
    )
    bars = pl.DataFrame(
        {
            "symbol": ["BTC-USDT-SWAP", "BTC-USDT-SWAP", "BTC-USDT-SWAP"],
            "timestamp": [1, 2, 3],
            "open": [100.0, 99.0, 97.0],
            "high": [101.0, 100.0, 98.0],
            "low": [98.0, 96.0, 94.0],
            "close": [100.0, 98.0, 95.0],
        }
    )
    snapshot = pl.DataFrame(
        {
            "symbol": ["BTC", "ETH"],
            "source_family": ["books", "funding"],
            "source_state": ["bid_support", "crowded_longs_under_stress"],
            "source_direction": ["bullish", "bearish"],
            "provider_timestamp_ms": [10, 1],
            "known_at_ms": [10, 1],
            "aligned_bar": ["1H", "1H"],
            "aligned_bar_close_ms": [None, 1],
            "serialization_status": ["stored_source_row", "stored_source_row"],
            "outcome_horizon": [1, 8],
            "close_at_event": [None, 100.0],
            "future_close": [None, 95.0],
            "forward_return_pct": [None, -5.0],
            "forward_min_return_pct": [None, -6.0],
            "forward_max_return_pct": [None, 1.0],
            "path_range_pct": [None, 7.0],
            "tail_asymmetry_pct": [None, -5.0],
            "outcome_available": [False, True],
            "outcome_reason": ["future_bar_missing", "available"],
        },
        schema=potential_outcome.SOURCE_OUTCOME_SCHEMA,
    )

    outcomes = potential_outcome.source_outcomes_frame(events, bars)
    predictability = potential_outcome.source_state_predictability_frame(
        outcomes, return_threshold_pct=0.5
    )
    timeliness = potential_outcome.source_timeliness_frame(snapshot)

    first = outcomes.filter(
        (pl.col("outcome_horizon") == 1) & (pl.col("aligned_bar_close_ms") == 1)
    ).row(0, named=True)
    state = predictability.filter(pl.col("outcome_horizon") == 1).row(0, named=True)
    assert first["outcome_available"] is True
    assert first["forward_return_pct"] == -2.0
    assert state["source_state"] == "aggressive_sell_dominance"
    assert state["p_down"] == 1.0
    assert state["dominant_outcome"] == "down"
    assert state["statistical_direction"] == "bearish"
    assert state["predictability_status"] == "insufficient_predictive_sample"
    assert (
        timeliness.filter(pl.col("source_family") == "books").row(0, named=True)[
            "timeliness_status"
        ]
        == "snapshot_or_future_only"
    )
    assert (
        timeliness.filter(pl.col("source_family") == "funding").row(0, named=True)[
            "timeliness_status"
        ]
        == "usable_history"
    )


def test_unified_evidence_uses_neutral_ladder_and_configured_decision_timeframe() -> None:
    symbols = [f"SYM{i:03d}" for i in range(120)]
    observations = [
        _observation(symbol, index, changed=index < 60) for index, symbol in enumerate(symbols)
    ]
    outcomes = [
        _source_outcome(symbol, index, changed=index < 60) for index, symbol in enumerate(symbols)
    ]
    realized = [
        _realized_transition(symbol, index, changed=index < 60)
        for index, symbol in enumerate(symbols)
    ]
    evidence = potential_ladder.potential_evidence_frame(
        pl.DataFrame(observations, schema=potential_state.POTENTIAL_OBSERVATION_SCHEMA),
        pl.DataFrame(outcomes, schema=potential_outcome.SOURCE_OUTCOME_SCHEMA),
        pl.DataFrame(realized, schema=potential_outcome.REALIZED_TRANSITION_SCHEMA),
        return_threshold_pct=0.5,
    )

    configured_timeframe = potential_outcome.potential_outcome_frame(
        pl.DataFrame(
            [_observation("BTC-USDT-SWAP", 999, changed=True) | {"decision_timeframe": "4H"}],
            schema=potential_state.POTENTIAL_OBSERVATION_SCHEMA,
        ),
        pl.DataFrame(schema=potential_outcome.SOURCE_OUTCOME_SCHEMA),
        pl.DataFrame(
            [
                _realized_transition("BTC-USDT-SWAP", 999, changed=True)
                | {"timeframe": "1H", "terminal_direction": "bearish"},
                _realized_transition("BTC-USDT-SWAP", 999, changed=True)
                | {"timeframe": "4H", "terminal_direction": "bullish"},
            ],
            schema=potential_outcome.REALIZED_TRANSITION_SCHEMA,
        ),
        return_threshold_pct=0.5,
    )

    suggestions = set(evidence.get_column("research_suggestion").unique().to_list())
    assert {"market_background", "market_decision_source"} <= set(
        evidence.get_column("evidence_level").to_list()
    )
    assert suggestions <= {
        "rapid_trend_watch",
        "mean_reversion_watch",
        "volatility_expansion_watch",
        "chop_avoid",
        "insufficient_evidence",
    }
    assert not any(str(label).startswith(("bullish", "bearish")) for label in suggestions)
    assert configured_timeframe.height == 1
    assert configured_timeframe.row(0, named=True)["terminal_direction"] == "bullish"


def test_evidence_gate_excludes_market_background_and_requires_stable_information() -> None:
    symbols = [f"SYM{i:03d}" for i in range(40)]
    observations = [
        _observation(symbol, index, changed=(index % 40) < 24)
        for index, symbol in enumerate(symbols * 200)
    ]
    outcomes = [
        _source_outcome(symbol, index, changed=(index % 40) < 24)
        for index, symbol in enumerate(symbols * 200)
    ]
    realized = [
        _realized_transition(symbol, index, changed=(index % 40) < 24)
        for index, symbol in enumerate(symbols * 200)
    ]
    evidence = potential_ladder.potential_evidence_frame(
        pl.DataFrame(observations, schema=potential_state.POTENTIAL_OBSERVATION_SCHEMA),
        pl.DataFrame(outcomes, schema=potential_outcome.SOURCE_OUTCOME_SCHEMA),
        pl.DataFrame(realized, schema=potential_outcome.REALIZED_TRANSITION_SCHEMA),
        return_threshold_pct=0.5,
    )
    selected = evidence.filter(pl.col("selected_evidence_level"))
    assert not (selected.get_column("evidence_level") == "market_background").any(), (
        "market_background must never be a selected evidence level"
    )


def test_kline_history_classifier_and_transition_paths_are_known_at_close() -> None:
    rows = pl.DataFrame(
        {
            "symbol": ["BTC", "BTC", "BTC"],
            "timestamp": [1, 2, 3],
            "source_family": ["kline", "kline", "kline"],
            "scale": ["1H", "1H", "1H"],
            "state_key": ["range", "range", "markdown"],
            "context_event": ["none", "none", "breakdown"],
            "direction_hint": ["neutral", "neutral", "bearish"],
            "quality_weight": [0.5, 0.5, 0.8],
            "missing_flag": [False, False, False],
            "stale_flag": [False, False, False],
        }
    )
    frame = pl.DataFrame(
        {
            "symbol": ["BTC-USDT-SWAP"] * 80,
            "timestamp": list(range(80)),
            "open": [1.0] * 80,
            "high": [2.0] * 80,
            "low": [0.5] * 80,
            "close": [1.5] * 80,
            "volume": [100.0] * 80,
        }
    )

    history = potential_outcome.kline_path_rows(rows, 2)
    classified = potential_state.KlineClassifier("1H").classify(frame)
    missing = potential_state.KlineClassifier("1H").classify(frame.head(1))
    third = history.filter(pl.col("bar_close_ms") == 3).row(0, named=True)

    assert third["transition_path"] == "range -> markdown"
    assert third["transition_kind"] == "state_and_event_transition"
    assert third["state_age_bars"] == 1
    assert third["event_age_bars"] == 1
    assert tuple(classified.columns) == potential_state.STATE_FRAME_COLUMNS
    assert classified.select("source_family").item(0, 0) == "kline"
    assert classified.select("scale").item(0, 0) == "1H"
    assert classified.row(60, named=True)["context_event"] == "none_in_accumulation"
    assert "forward_return" not in classified.columns
    assert missing.select("missing_flag").item() is True
    assert missing.select("direction_hint").item() == "missing"


def test_kline_path_rows_keep_state_runs_separate_by_timeframe() -> None:
    rows = pl.DataFrame(
        {
            "symbol": ["BTC", "BTC", "BTC", "BTC"],
            "timestamp": [1, 1, 2, 2],
            "source_family": ["kline", "kline", "kline", "kline"],
            "scale": ["1H", "4H", "1H", "4H"],
            "state_key": ["range|coil", "markup|trend", "range|coil", "markup|trend"],
            "context_event": ["none", "impulse", "none", "impulse"],
            "direction_hint": ["neutral", "bullish", "neutral", "bullish"],
            "quality_weight": [0.5, 0.8, 0.5, 0.8],
            "missing_flag": [False, False, False, False],
            "stale_flag": [False, False, False, False],
        }
    )

    history = potential_outcome.kline_path_rows(rows, 2).sort("timeframe", "bar_close_ms")

    one_h = history.filter(pl.col("timeframe") == "1H").to_dicts()
    four_h = history.filter(pl.col("timeframe") == "4H").to_dicts()
    assert [row["state_age_bars"] for row in one_h] == [1, 2]
    assert [row["event_age_bars"] for row in one_h] == [1, 2]
    assert [row["state_age_bars"] for row in four_h] == [1, 2]
    assert [row["event_age_bars"] for row in four_h] == [1, 2]
    assert one_h[1]["transition_path"] == "range|coil -> range|coil"
    assert four_h[1]["transition_path"] == "markup|trend -> markup|trend"


@pytest.mark.parametrize(
    ("bundle", "config", "expected_group", "expected_direction", "expected_reason"),
    [
        (
            _decision_bundle(),
            potential.PotentialConfig(),
            "watch",
            "bullish",
            "context_missing",
        ),
        (
            _decision_bundle(
                trades=_state_row("trades", "bullish", state="aggressive_buy_dominance"),
                context=_state_row("context", "neutral", state="context_available", score=0.4),
            ),
            potential.PotentialConfig(),
            "bullish",
            "bullish",
            "",
        ),
        (
            _decision_bundle(
                trades=_state_row("trades", "bearish", state="sell"),
                derivatives=_state_row("derivatives", "bullish", state="buy"),
            ),
            potential.PotentialConfig(),
            "watch",
            "bullish",
            "contradictory_source_evidence",
        ),
        (
            _decision_bundle(transition=_state_row("transition", "bullish", score=0.2)),
            potential.PotentialConfig(transition=TransitionConfig(min_directional_probability=0.9)),
            "watch",
            "bullish",
            "transition_quality_below_threshold",
        ),
        (
            _decision_bundle(
                transition=_state_row(
                    "transition",
                    "missing",
                    state="transition_pattern_missing",
                    score=0.0,
                    reason="transition_pattern_missing",
                    timestamp=None,
                ),
                patterns=(),
            ),
            potential.PotentialConfig(),
            "watch",
            "undecided",
            "transition_path_missing_or_neutral",
        ),
    ],
)
def test_scan_review_decisions_require_transition_quality_and_source_confirmation(
    bundle, config, expected_group, expected_direction, expected_reason
) -> None:
    decision = potential.scan_review_decisions(config, (bundle,))[0]

    assert decision.group == expected_group
    assert decision.direction == expected_direction
    assert decision.block_reason == expected_reason


def test_transition_matching_and_ngram_work_frame_do_not_use_unrelated_patterns() -> None:
    stages = ["accumulation", "markup", "trend_continuation"] * 8
    stages.extend(["markdown", "distribution_or_reversal", "accumulation"])
    frame = pl.DataFrame(
        {
            "timestamp": list(range(len(stages))),
            "open": [float(index + 1) for index in range(len(stages))],
            "high": [float(index + 2) for index in range(len(stages))],
            "low": [float(index) for index in range(len(stages))],
            "close": [float(index + 1) for index in range(len(stages))],
            "market_stage": stages,
            "structure_trend_state": ["uptrend"] * len(stages),
            "liquidity_event_type": ["none"] * len(stages),
        }
    )
    state_frame = pl.DataFrame(
        {
            "symbol": ["BTC-USDT-SWAP"] * len(stages),
            "timestamp": list(range(len(stages))),
            "source_family": ["kline"] * len(stages),
            "scale": ["1H"] * len(stages),
            "state_key": [f"{stage}|uptrend|range_normal|vol_normal" for stage in stages],
            "context_event": ["none_in_trend"] * len(stages),
            "direction_hint": ["bullish"] * len(stages),
            "quality_weight": [0.8] * len(stages),
            "missing_flag": [False] * len(stages),
            "stale_flag": [False] * len(stages),
        },
        schema=STATE_FRAME_SCHEMA,
    )

    analysis = transitions.compute_transition_insights(
        potential.PotentialConfig(
            symbols=("BTC-USDT-SWAP",),
            timeframes=("1H",),
            transition=TransitionConfig(
                horizon=1,
                min_count=4,
                ngram_length=4,
            ),
        ),
        ("BTC-USDT-SWAP",),
        {("BTC-USDT-SWAP", "1H"): frame},
        {("BTC-USDT-SWAP", "1H"): state_frame},
    )

    expected_current_path = (
        "trend_continuation|uptrend|range_normal|vol_normal -> "
        "markdown|uptrend|range_normal|vol_normal -> "
        "distribution_or_reversal|uptrend|range_normal|vol_normal -> "
        "accumulation|uptrend|range_normal|vol_normal"
    )

    assert analysis.insights["BTC-USDT-SWAP"].current.direction == "missing"
    assert analysis.insights["BTC-USDT-SWAP"].patterns == ()
    assert analysis.unsupported
    assert any(path.path == expected_current_path for path in analysis.unsupported)


def test_continuous_features_use_canonical_volume_column() -> None:
    bar_frame = pl.DataFrame(
        {
            "timestamp": list(range(30)),
            "open": [100.0 + index for index in range(30)],
            "high": [101.0 + index for index in range(30)],
            "low": [99.0 + index for index in range(30)],
            "close": [100.5 + index for index in range(30)],
            "volume": [10.0 + index for index in range(30)],
        }
    )
    state_frame = pl.DataFrame({"timestamp": list(range(30))})

    result = potential_state.extract_continuous_features(
        {("BTC-USDT-SWAP", "1H"): bar_frame},
        {("BTC-USDT-SWAP", "1H"): state_frame},
        {},
    )

    assert result.height == 30
    assert "vol_anomaly" in result.columns
    assert result.select(pl.col("vol_anomaly").is_not_null().sum()).item() > 0


def test_source_features_align_as_known_at_close_without_rewriting_source_time() -> None:
    bar_frame = pl.DataFrame(
        {
            "timestamp": [1000, 2000, 3000],
            "open": [1.0, 2.0, 3.0],
            "high": [2.0, 3.0, 4.0],
            "low": [0.5, 1.5, 2.5],
            "close": [1.5, 2.5, 3.5],
            "volume": [10.0, 20.0, 30.0],
        }
    )
    state_frame = pl.DataFrame({"timestamp": [1000, 2000, 3000]})
    source_frames = {
        "funding": pl.DataFrame(
            {
                "symbol": ["BTC-USDT-SWAP", "BTC-USDT-SWAP"],
                "timestamp": [900, 2500],
                "funding_rate": [0.01, 0.03],
            }
        )
    }

    result = potential_state.extract_continuous_features(
        {("BTC-USDT-SWAP", "1H"): bar_frame},
        {("BTC-USDT-SWAP", "1H"): state_frame},
        source_frames,
    ).sort("timestamp")

    rows = {row["timestamp"]: row for row in result.iter_rows(named=True)}
    assert rows[1000]["funding_rate"] == pytest.approx(0.01)
    assert rows[2000]["funding_rate"] == pytest.approx(0.01)
    assert rows[3000]["funding_rate"] == pytest.approx(0.03)
    assert rows[1000]["funding_age_ms"] == 100
    assert rows[2000]["funding_age_ms"] == 1100
    assert rows[3000]["funding_age_ms"] == 500


def test_stale_source_features_are_nulled_after_family_max_age() -> None:
    bar_frame = pl.DataFrame(
        {
            "timestamp": [0, 60 * 60 * 1000, 17 * 60 * 60 * 1000],
            "open": [1.0, 2.0, 3.0],
            "high": [2.0, 3.0, 4.0],
            "low": [0.5, 1.5, 2.5],
            "close": [1.5, 2.5, 3.5],
            "volume": [10.0, 20.0, 30.0],
        }
    )
    state_frame = pl.DataFrame({"timestamp": [0, 60 * 60 * 1000, 17 * 60 * 60 * 1000]})
    source_frames = {
        "funding": pl.DataFrame(
            {
                "symbol": ["BTC-USDT-SWAP"],
                "timestamp": [0],
                "funding_rate": [0.01],
            }
        )
    }

    result = potential_state.extract_continuous_features(
        {("BTC-USDT-SWAP", "1H"): bar_frame},
        {("BTC-USDT-SWAP", "1H"): state_frame},
        source_frames,
    ).sort("timestamp")

    rows = {row["timestamp"]: row for row in result.iter_rows(named=True)}
    assert rows[0]["funding_rate"] == pytest.approx(0.01)
    assert rows[60 * 60 * 1000]["funding_rate"] == pytest.approx(0.01)
    assert rows[17 * 60 * 60 * 1000]["funding_rate"] is None
    assert rows[17 * 60 * 60 * 1000]["funding_age_ms"] == 17 * 60 * 60 * 1000


def test_select_tail_leaves_returns_best_available_when_strict_gate_is_empty() -> None:
    leaf_evidence = pl.DataFrame(
        {
            "leaf_id": [1, 2, 3],
            "tree_direction": ["up", "up", "up"],
            "N_total": [1000, 1000, 1000],
            "N_tail_exceedances": [20, 25, 10],
            "tail_lift": [1.4, 1.2, 1.1],
            "tail_lift_stability": [0.6, 0.8, 0.2],
            "gpd_shape_xi": [0.1, 0.2, 0.3],
            "gpd_scale_sigma": [1.0, 1.1, 1.2],
            "leaf_tail_rate": [0.02, 0.025, 0.01],
            "global_tail_rate": [0.015, 0.015, 0.015],
        }
    )

    selected = select_tail_leaves(leaf_evidence, fallback_top_n=2)

    assert selected.height == 2
    assert selected["selection_mode"].to_list() == ["best_available", "best_available"]
    assert selected["selected_evidence_level"].to_list() == [False, False]
    assert (
        selected["tail_evidence_score"].to_list()[0] >= selected["tail_evidence_score"].to_list()[1]
    )


def test_tailtree_outcome_aggregation_preserves_source_tail_labels() -> None:
    outcomes = pl.DataFrame(
        {
            "symbol": ["BTC-USDT-SWAP", "BTC-USDT-SWAP", "BTC-USDT-SWAP"],
            "decision_bar_close_ms": [1000, 1000, 2000],
            "outcome_bucket": ["flat", "up", "down"],
            "tail_up": [False, True, False],
            "tail_down": [False, False, True],
            "direction_changed": [False, True, False],
            "returned_to_origin": [False, False, True],
        }
    )

    collapsed = _tailtree_outcome_by_decision(outcomes).sort("decision_bar_close_ms")
    rows = {row["decision_bar_close_ms"]: row for row in collapsed.iter_rows(named=True)}

    assert collapsed.height == 2
    assert rows[1000]["outcome_bucket"] == "up"
    assert rows[1000]["tail_up"] is True
    assert rows[1000]["tail_down"] is False
    assert rows[2000]["outcome_bucket"] == "down"
    assert rows[2000]["tail_down"] is True
