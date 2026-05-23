from __future__ import annotations

import polars as pl

from qooi.research import artifacts, frames, metrics, outcomes, patterns, promotion


def _market_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["BTC"] * 5,
            "timeframe": ["1H"] * 5,
            "timestamp": [1, 2, 3, 4, 5],
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.0, 101.0, 102.0, 101.0, 103.0],
            "market_stage_reduced": ["range", "range", "markup", "markup", "range"],
            "structure_trend_state": ["range", "uptrend", "uptrend", "range", "range"],
            "liquidity_event_type": [
                "none",
                "failed_breakout_low",
                "failed_breakout_low",
                "failed_breakout_high",
                "none",
            ],
            "atr_percentile_bucket": ["normal", "high", "high", "normal", "low"],
        }
    )


def test_normalize_research_frame_emits_long_known_at_close_rows():
    research_frame = frames.normalize_research_frame(
        _market_frame(),
        symbol="BTC",
        timeframe="1H",
        state_columns=("market_stage_reduced", "structure_trend_state"),
        event_column="liquidity_event_type",
        context_columns=("atr_percentile_bucket",),
    )

    assert research_frame.height == 10
    assert set(research_frame["state_column"].to_list()) == {
        "market_stage_reduced",
        "structure_trend_state",
    }
    assert "atr_percentile_bucket" in research_frame.columns
    assert "forward_return_pct" not in research_frame.columns


def test_materialize_transition_patterns_is_deterministic_and_no_lookahead():
    research_frame = frames.normalize_research_frame(
        _market_frame(),
        symbol="BTC",
        timeframe="1H",
        state_columns=("market_stage_reduced",),
        event_column="liquidity_event_type",
    )

    transition_patterns = patterns.materialize_transition_patterns(
        research_frame, {"ngram_lengths": (2, 3)}
    )

    assert set(transition_patterns["pattern_family"].to_list()) == {
        "transition",
        "transition_ngram",
    }
    assert transition_patterns.filter(pl.col("ngram_length") == 2).height == 4
    assert transition_patterns.filter(pl.col("ngram_length") == 3).height == 3
    assert "range->markup" in transition_patterns["pattern_value"].to_list()


def test_outcomes_attach_forward_labels_after_pattern_materialization():
    market = _market_frame()
    research_frame = frames.normalize_research_frame(
        market,
        symbol="BTC",
        timeframe="1H",
        state_columns=("market_stage_reduced",),
        event_column="liquidity_event_type",
    )
    static_patterns = patterns.materialize_static_patterns(research_frame)

    outcome_table = outcomes.attach_forward_outcomes(static_patterns, market, (1,))
    long_row = outcome_table.filter(pl.col("event_value") == "failed_breakout_low").row(
        0, named=True
    )
    short_row = outcome_table.filter(pl.col("event_value") == "failed_breakout_high").row(
        0, named=True
    )

    assert long_row["side"] == "long"
    assert short_row["side"] == "short"
    assert "side_return_pct" in outcome_table.columns


def test_metrics_and_promotion_are_separate_pipe_steps():
    market = _market_frame()
    research_frame = frames.normalize_research_frame(
        market,
        symbol="BTC",
        timeframe="1H",
        state_columns=("market_stage_reduced",),
        event_column="liquidity_event_type",
    )
    static_patterns = patterns.materialize_static_patterns(research_frame)
    outcome_table = outcomes.attach_forward_outcomes(static_patterns, market, (1,))

    metric_table = metrics.summarize_returns(
        outcome_table,
        ["pattern_id", "pattern_family", "pattern_source", "symbol", "horizon", "side"],
    )
    assert "passes_candidate_gate" not in metric_table.columns

    scored = promotion.apply_candidate_gate(
        metric_table,
        {"min_rows": 1, "omega_threshold": 0.1, "pwpr_threshold": 0.1},
    )
    assert "passes_candidate_gate" in scored.columns


def test_information_metrics_are_count_table_based():
    frame = pl.DataFrame(
        {
            "prev": ["a", "a", "b", "b"],
            "current": ["a", "a", "b", "b"],
            "event": ["x", "x", "y", "y"],
        }
    )

    assert metrics.entropy(frame, "current") == 1.0
    assert metrics.mutual_information(frame, "prev", "current") == 1.0
    assert metrics.conditional_mutual_information(frame, "prev", "current", "event") == 0.0


def test_artifact_projections_are_views_over_shared_contracts():
    research_frame = frames.normalize_research_frame(
        _market_frame(),
        symbol="BTC",
        timeframe="1H",
        state_columns=("market_stage_reduced",),
        event_column="liquidity_event_type",
    )
    transition_patterns = patterns.materialize_transition_patterns(
        research_frame, {"ngram_lengths": (2,)}
    )

    graph = artifacts.project_transition_graph(transition_patterns)

    assert set(graph["artifact"].to_list()) == {"state-transition-graph"}
    assert {"source_state", "target_state", "transition_probability"} <= set(graph.columns)
