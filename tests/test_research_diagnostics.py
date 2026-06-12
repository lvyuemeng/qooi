from __future__ import annotations

import polars as pl

from qooi.research import behavior_tables, candidates, rule_primitives
from qooi.research import patterns as pattern_tables
from qooi.research.artifacts import ensure_columns


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
    research_frame = pattern_tables.normalize_research_frame(
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
    research_frame = pattern_tables.normalize_research_frame(
        _market_frame(),
        symbol="BTC",
        timeframe="1H",
        state_columns=("market_stage_reduced",),
        event_column="liquidity_event_type",
    )

    transition_patterns = pattern_tables.materialize_transition_patterns(
        research_frame, {"ngram_lengths": (2, 3)}
    )

    assert set(transition_patterns["pattern_family"].to_list()) == {
        "transition",
        "transition_ngram",
    }
    assert transition_patterns.filter(pl.col("ngram_length") == 2).height == 4
    assert transition_patterns.filter(pl.col("ngram_length") == 3).height == 3
    assert "range->markup" in transition_patterns["pattern_value"].to_list()


def test_materialize_transition_patterns_ignores_sparse_null_states():
    research_frame = pattern_tables.normalize_research_frame(
        _market_frame().with_columns(pl.Series("learned_state", [None, "20", None, "13", None])),
        symbol="BTC",
        timeframe="1H",
        state_columns=("learned_state",),
        event_column="liquidity_event_type",
    )

    transition_patterns = pattern_tables.materialize_transition_patterns(research_frame)

    assert transition_patterns.height == 1
    assert transition_patterns.row(0, named=True)["pattern_value"] == "20->13"


def test_outcomes_attach_forward_labels_after_pattern_materialization():
    market = _market_frame()
    research_frame = pattern_tables.normalize_research_frame(
        market,
        symbol="BTC",
        timeframe="1H",
        state_columns=("market_stage_reduced",),
        event_column="liquidity_event_type",
    )
    static_patterns = pattern_tables.materialize_transition_patterns(research_frame)

    outcome_table = pattern_tables.attach_forward_outcomes(static_patterns, market, (1,))
    long_row = outcome_table.filter(pl.col("event_value") == "failed_breakout_low").row(
        0, named=True
    )
    short_row = outcome_table.filter(pl.col("event_value") == "failed_breakout_high").row(
        0, named=True
    )

    assert long_row["side"] == "long"
    assert short_row["side"] == "short"
    assert "side_return_pct" in outcome_table.columns


def test_learned_state_evaluation_outcomes_are_test_only_and_cost_adjusted():
    outcomes = pl.DataFrame(
        {
            "split": ["train", "test", "test"],
            "forward_return_pct": [1.0, 1.0, None],
            "side_return_pct": [1.0, 1.0, None],
        }
    )

    filtered = pattern_tables.filter_evaluation_outcomes(
        outcomes,
        returns_split="test",
        transaction_cost_bps=5.0,
    )

    assert filtered.height == 1
    assert filtered.row(0, named=True)["side_return_pct"] == 0.95
    assert filtered.row(0, named=True)["returns_split"] == "test"


def test_metrics_and_promotion_are_separate_pipe_steps():
    market = _market_frame()
    research_frame = pattern_tables.normalize_research_frame(
        market,
        symbol="BTC",
        timeframe="1H",
        state_columns=("market_stage_reduced",),
        event_column="liquidity_event_type",
    )
    static_patterns = pattern_tables.materialize_transition_patterns(research_frame)
    outcome_table = pattern_tables.attach_forward_outcomes(static_patterns, market, (1,))

    metric_table = pattern_tables.summarize_returns(
        outcome_table,
        ["pattern_id", "pattern_family", "pattern_source", "symbol", "horizon", "side"],
    )
    assert "passes_candidate_gate" not in metric_table.columns

    scored = pattern_tables.apply_candidate_gate(
        metric_table,
        {"min_rows": 1, "omega_threshold": 0.1, "pwpr_threshold": 0.1},
    )
    assert "passes_candidate_gate" in scored.columns


def test_candidate_trades_use_next_bar_entry_and_nonoverlap() -> None:
    market = _market_frame().with_columns(
        pl.lit("none").alias("liquidity_event_type"),
        pl.lit("test").alias("split"),
    )
    research_frame = pattern_tables.normalize_research_frame(
        market.with_columns(pl.Series("learned_state", ["1", "1", "1", "1", "1"])),
        symbol="BTC",
        timeframe="1H",
        state_columns=("learned_state",),
        event_column="liquidity_event_type",
        state_source="vq_rssm",
    )
    patterns = pattern_tables.materialize_state_patterns(research_frame, "vq_rssm")
    scored = pl.DataFrame(
        {
            "pattern_id": [patterns.row(0, named=True)["pattern_id"]],
            "pattern_family": ["state"],
            "pattern_source": ["vq_rssm"],
            "symbol": ["BTC"],
            "horizon": [1],
            "side": [None],
            "rows": [5],
            "positive_rate": [100.0],
            "negative_rate": [0.0],
            "positive_mean": [1.0],
            "negative_mean_abs": [0.0],
            "omega_ratio": [999.0],
            "pwpr": [999.0],
            "sortino_zero": [0.0],
            "mean_side_return_pct": [1.0],
            "invalid_state_present": [False],
            "passes_candidate_gate": [True],
        }
    )

    trades = candidates.build_candidate_nonoverlap_trades(
        patterns,
        market,
        ensure_columns(scored, pattern_tables.SCORED_PATTERN_SCHEMA),
        returns_split="test",
        transaction_cost_bps=5.0,
    )

    first = trades.row(0, named=True)
    assert first["signal_timestamp"] == 1
    assert first["entry_timestamp"] == 2
    assert first["exit_timestamp"] == 3
    assert first["side_return_pct"] == ((102.0 - 101.0) / 101.0 * 100.0) - 0.05
    assert trades.height == 2


def test_information_metrics_are_count_table_based():
    research_frame = pattern_tables.normalize_research_frame(
        _market_frame(),
        symbol="BTC",
        timeframe="1H",
        state_columns=("market_stage_reduced",),
        event_column="liquidity_event_type",
    )

    info = pattern_tables.summarize_transition_information(research_frame)

    assert info.height == 1
    assert "transition_information" in info.columns


def test_information_metrics_ignore_null_state_rows() -> None:
    research_frame = pl.DataFrame(
        {
            "symbol": ["BTC", "BTC", "BTC", "ETH", "ETH", "ETH"],
            "timeframe": ["1H"] * 6,
            "timestamp": [1, 2, 3, 1, 2, 3],
            "state_column": ["behavior_state_id"] * 6,
            "state_value": ["1", None, "2", "9", None, "8"],
            "event_value": ["none"] * 6,
        }
    )

    info = pattern_tables.summarize_transition_information(research_frame)

    assert info.height == 2
    assert sorted(info.get_column("rows").to_list()) == [2, 2]


def test_artifact_projections_are_views_over_shared_contracts():
    research_frame = pattern_tables.normalize_research_frame(
        _market_frame(),
        symbol="BTC",
        timeframe="1H",
        state_columns=("market_stage_reduced",),
        event_column="liquidity_event_type",
    )
    transition_patterns = pattern_tables.materialize_transition_patterns(
        research_frame, {"ngram_lengths": (2,)}
    )

    graph = pattern_tables.project_transition_graph(transition_patterns)

    assert set(graph["artifact"].to_list()) == {"state-transition-graph"}
    assert {"source_state", "target_state", "transition_probability"} <= set(graph.columns)


def test_state_patterns_and_info_are_direct_null_safe_helpers() -> None:
    research_frame = pl.DataFrame(
        {
            "symbol": ["BTC", "BTC", "ETH"],
            "timeframe": ["1H"] * 3,
            "timestamp": [1, 2, 1],
            "state_source": ["vq_rssm"] * 3,
            "state_column": ["behavior_state_id"] * 3,
            "state_value": ["1", None, "2"],
            "event_value": ["none"] * 3,
        }
    )

    patterns = pattern_tables.materialize_state_patterns(research_frame, "vq_rssm")
    info = pattern_tables.summarize_state_info(
        research_frame,
        "state_value",
        ("symbol", "timeframe", "state_column"),
    )

    assert patterns.height == 2
    assert set(patterns.get_column("pattern_source")) == {"vq_rssm"}
    assert sorted(info.get_column("rows")) == [1, 1]


def test_transition_paths_compose_scoring_and_projection_helpers() -> None:
    graph = pl.DataFrame(
        {
            "symbol": ["BTC", "BTC"],
            "timeframe": ["1H", "1H"],
            "state_column": ["behavior_state_id", "behavior_state_id"],
            "source_state": ["1", "1"],
            "target_state": ["2", "3"],
            "rows": [9, 1],
            "source_rows": [10, 10],
            "transition_probability": [0.9, 0.1],
        }
    )

    scored = pattern_tables.with_transition_path_scores(graph)
    paths = pattern_tables.project_transition_paths(graph)

    assert "surprisal_bits" in scored.columns
    assert set(paths.get_column("path_kind")) == {
        "high_probability",
        "low_probability_high_information",
    }


def test_state_diagnostics_compute_future_metrics_without_signal_columns() -> None:
    market = _market_frame().with_columns(
        pl.Series("learned_state", ["1", "1", "2", "2", "1"]),
        pl.lit("test").alias("split"),
    )

    diagnostics = behavior_tables.summarize_state_diagnostics(
        market, "learned_state", (1,), split="test"
    )
    state_one = diagnostics.filter(pl.col("context_value") == "1").row(0, named=True)

    assert state_one["rows"] == 2
    assert state_one["forward_return_mean_pct"] == 0.995049504950495
    assert "signal_timestamp" not in diagnostics.columns


def test_state_transition_chains_are_timestamped_at_last_state() -> None:
    market = _market_frame().with_columns(
        pl.Series("learned_state", ["1", "2", "3", "4", "5"]),
        pl.lit("test").alias("split"),
    )

    chains = behavior_tables.build_state_transition_chains(market, "learned_state", (2, 3))
    row = chains.filter(pl.col("chain_value") == "1->2->3").row(0, named=True)

    assert row["timestamp"] == 3
    assert row["from_state"] == "1"
    assert row["to_state"] == "3"
    assert row["previous_state"] == "2"


def test_chain_information_and_taxonomy_compose_with_rule_signals() -> None:
    market = pl.DataFrame(
        {
            "symbol": ["BTC"] * 8,
            "timeframe": ["1H"] * 8,
            "timestamp": list(range(1, 9)),
            "open": [10.0, 10.5, 11.0, 11.2, 11.5, 11.8, 12.2, 12.8],
            "high": [10.2, 10.7, 11.3, 11.4, 11.8, 12.0, 12.5, 13.0],
            "low": [9.8, 10.3, 10.8, 11.0, 11.2, 11.5, 12.0, 12.5],
            "close": [10.0, 10.6, 11.1, 11.3, 11.6, 11.9, 12.4, 12.9],
            "learned_state": ["1", "2", "1", "2", "1", "2", "1", "2"],
            "split": ["test"] * 8,
        }
    )
    chains = behavior_tables.build_state_transition_chains(market, "learned_state", (2,))
    info = behavior_tables.summarize_state_chain_information(chains, market, (1,))
    taxonomy = behavior_tables.classify_state_taxonomy(
        info.with_columns(
            pl.lit(30).alias("rows"), pl.lit(0.9).alias("normalized_information_gain")
        )
    ).with_columns(pl.lit("trend_smooth").alias("taxonomy_label"))

    signals = rule_primitives.build_rule_primitive_signals(
        market,
        taxonomy,
        "learned_state",
        config=rule_primitives.RulePrimitiveConfig(horizons=(1,), ema_fast=2, ema_slow=3),
    )

    assert not info.is_empty()
    assert not signals.is_empty()
    assert set(signals.get_column("context_kind")) == {"chain"}


def test_rule_primitive_trades_use_next_bar_nonoverlap_and_single_cost() -> None:
    market = _market_frame().with_columns(
        pl.Series("learned_state", ["1", "1", "1", "1", "1"]),
        pl.lit("test").alias("split"),
    )
    signals = pl.DataFrame(
        {
            "context_kind": ["state", "state", "state"],
            "context_value": ["1", "1", "1"],
            "taxonomy_label": ["trend_smooth", "trend_smooth", "trend_smooth"],
            "rule_name": ["ema_trend_follow", "ema_trend_follow", "ema_trend_follow"],
            "symbol": ["BTC", "BTC", "BTC"],
            "timeframe": ["1H", "1H", "1H"],
            "timestamp": [1, 2, 4],
            "state_column": ["learned_state", "learned_state", "learned_state"],
            "horizon": [1, 1, 1],
            "side": ["long", "long", "long"],
            "signal_close": [100.0, 101.0, 101.0],
            "split": ["test", "test", "test"],
        }
    )

    trades = rule_primitives.build_rule_primitive_trades(
        ensure_columns(signals, rule_primitives.RULE_PRIMITIVE_SIGNAL_SCHEMA),
        market,
        transaction_cost_bps=5.0,
    )

    first = trades.row(0, named=True)
    assert first["signal_timestamp"] == 1
    assert first["entry_timestamp"] == 2
    assert first["exit_timestamp"] == 3
    assert first["side_return_pct"] == ((102.0 - 101.0) / 101.0 * 100.0) - 0.05
    assert trades.height == 1


def test_behavior_helpers_return_typed_empty_frames() -> None:
    empty = pl.DataFrame()

    assert behavior_tables.summarize_state_diagnostics(empty, "missing", (1,)).schema == pl.Schema(
        behavior_tables.STATE_DIAGNOSTIC_SCHEMA
    )
    chains = behavior_tables.build_state_transition_chains(empty, "missing", (2,))
    assert chains.schema == pl.Schema(behavior_tables.STATE_TRANSITION_CHAIN_SCHEMA)
    assert rule_primitives.summarize_rule_primitives(empty).schema == pl.Schema(
        rule_primitives.RULE_PRIMITIVE_SUMMARY_SCHEMA
    )
