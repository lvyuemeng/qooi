from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import qooi.scanner.workflow as potential
from qooi.exchange.discovery import DiscoveryResult, empty_discovery_frame
from qooi.scanner import candidates as potential_candidates
from qooi.scanner import classifiers, decisions, source_events, transitions
from qooi.scanner import contracts as scan
from qooi.scanner import evidence as potential_evidence
from qooi.scanner import history as potential_history
from qooi.scanner.classifiers import STATE_FRAME_SCHEMA
from qooi.scanner.workflow import run


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
    }

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
        schema=potential_evidence.POTENTIAL_OBSERVATION_SCHEMA,
    )

    candidates = potential_candidates.candidate_evidence_frame(
        observations, _selected_evidence_for_test()
    )

    assert candidates.height == 1
    row = candidates.row(0, named=True)
    assert row["symbol"] == "BTC-USDT-SWAP"
    assert row["decision_bar_close_ms"] == 3
    assert row["matched_evidence_level"] == "market_decision_source_risk"
    assert row["research_suggestion"] == "rapid_trend_watch"
    assert row["candidate_status"] == "matched_evidence"

def test_candidate_evidence_combines_matched_and_unmatched_latest_observations() -> None:
    observations = pl.DataFrame(
        [
            _observation("BTC-USDT-SWAP", 1, changed=True),
            _observation("ETH-USDT-SWAP", 1, changed=False),
        ],
        schema=potential_evidence.POTENTIAL_OBSERVATION_SCHEMA,
    )

    candidates = potential_candidates.candidate_evidence_frame(
        observations, _selected_evidence_for_test()
    )

    assert candidates.height == 2
    rows = {row["symbol"]: row for row in candidates.iter_rows(named=True)}
    assert rows["BTC-USDT-SWAP"]["candidate_status"] == "matched_evidence"
    assert rows["ETH-USDT-SWAP"]["candidate_status"] == "no_matching_evidence"

def test_candidate_evidence_emits_unmatched_latest_observation_caveat() -> None:
    observations = pl.DataFrame(
        [_observation("BTC-USDT-SWAP", 1, changed=True)],
        schema=potential_evidence.POTENTIAL_OBSERVATION_SCHEMA,
    )
    evidence = pl.DataFrame(schema={"selected_evidence_level": pl.Boolean})

    candidates = potential_candidates.candidate_evidence_frame(observations, evidence)

    assert candidates.height == 1
    row = candidates.row(0, named=True)
    assert row["symbol"] == "BTC-USDT-SWAP"
    assert row["candidate_status"] == "no_selected_evidence"
    assert row["matched_evidence_level"] is None

def test_rank_candidate_evidence_exposes_components_without_trading_signal() -> None:
    observations = pl.DataFrame(
        [_observation("BTC-USDT-SWAP", 1, changed=True)],
        schema=potential_evidence.POTENTIAL_OBSERVATION_SCHEMA,
    )
    candidates = potential_candidates.candidate_evidence_frame(
        observations, _selected_evidence_for_test()
    )

    ranked = potential_candidates.rank_candidate_evidence(candidates)

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
    assert "## Review Rows" in report
    assert "Tiers: 1=top-decile" in report
    assert (diagnostics / "coverage.csv").exists()
    assert (diagnostics / "source-freshness.csv").exists()
    assert (diagnostics / "potential-observation-summary.csv").exists()
    assert (diagnostics / "potential-evidence-summary.csv").exists()
    assert (diagnostics / "potential-evidence-selected.csv").exists()
    assert (diagnostics / "candidate-evidence.csv").exists()
    assert (diagnostics / "candidate-rank.csv").exists()
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
transition_scan_budget = 80
transition_context_scope = "all_scanned"
transition_context_limit = 80
transition_history_days = 365
transition_ngram_length = 4
transition_horizon = 8
transition_min_information_bits = 0.01
require_context_for_review = true
max_source_staleness_hours = 12
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
    assert legacy.disabled_sources == ()
    assert current.output == Path("data/output/potential/report.md")
    assert current.refresh_mode == "incremental"
    assert current.fetch_concurrency == 8
    assert current.transition_scan_budget == 80
    assert current.transition_context_scope == "all_scanned"
    assert current.transition_context_limit == 80
    assert current.transition_history_days == 365
    assert current.transition_ngram_length == 4
    assert current.transition_horizon == 8
    assert current.transition_min_information_bits == 0.01
    assert current.require_context_for_review is True
    assert current.max_source_staleness_hours == 12
    assert current.trade_limit == 50
    assert current.funding_limit == 60
    assert current.rubik_period == "4H"
    assert current.rubik_limit == 70
    assert current.rubik_taker_unit == "1"
    assert current.disabled_sources == ("messages",)
    assert current.disabled_symbols == ("BAD-USDT-SWAP",)

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

    universe = potential.resolve_universe(potential.PotentialConfig(transition_scan_budget=2))
    all_context = scan.context_symbols(
        potential.PotentialConfig(transition_context_scope="all_scanned"), universe.symbols, {}
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

    events = source_events.source_events_frame(source_frames, bars, "1H")
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
        schema=source_events.SOURCE_EVENT_SCHEMA,
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
        schema=source_events.SOURCE_OUTCOME_SCHEMA,
    )

    outcomes = source_events.source_outcomes_frame(events, bars)
    predictability = source_events.source_state_predictability_frame(
        outcomes, return_threshold_pct=0.5
    )
    timeliness = source_events.source_timeliness_frame(snapshot)

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
    assert timeliness.filter(pl.col("source_family") == "books").row(0, named=True)[
        "timeliness_status"
    ] == "snapshot_or_future_only"
    assert timeliness.filter(pl.col("source_family") == "funding").row(0, named=True)[
        "timeliness_status"
    ] == "usable_history"

def test_unified_evidence_uses_neutral_ladder_and_configured_decision_timeframe() -> None:
    symbols = [f"SYM{i:03d}" for i in range(120)]
    observations = [
        _observation(symbol, index, changed=index < 60)
        for index, symbol in enumerate(symbols)
    ]
    outcomes = [
        _source_outcome(symbol, index, changed=index < 60)
        for index, symbol in enumerate(symbols)
    ]
    realized = [
        _realized_transition(symbol, index, changed=index < 60)
        for index, symbol in enumerate(symbols)
    ]
    evidence = potential_evidence.potential_evidence_frame(
        pl.DataFrame(observations, schema=potential_evidence.POTENTIAL_OBSERVATION_SCHEMA),
        pl.DataFrame(outcomes, schema=source_events.SOURCE_OUTCOME_SCHEMA),
        pl.DataFrame(realized, schema=potential_history.REALIZED_TRANSITION_SCHEMA),
        return_threshold_pct=0.5,
    )

    configured_timeframe = potential_evidence.potential_outcome_frame(
        pl.DataFrame(
            [_observation("BTC-USDT-SWAP", 999, changed=True) | {"decision_timeframe": "4H"}],
            schema=potential_evidence.POTENTIAL_OBSERVATION_SCHEMA,
        ),
        pl.DataFrame(schema=source_events.SOURCE_OUTCOME_SCHEMA),
        pl.DataFrame(
            [
                _realized_transition("BTC-USDT-SWAP", 999, changed=True)
                | {"timeframe": "1H", "terminal_direction": "bearish"},
                _realized_transition("BTC-USDT-SWAP", 999, changed=True)
                | {"timeframe": "4H", "terminal_direction": "bullish"},
            ],
            schema=potential_history.REALIZED_TRANSITION_SCHEMA,
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
    evidence = potential_evidence.potential_evidence_frame(
        pl.DataFrame(observations, schema=potential_evidence.POTENTIAL_OBSERVATION_SCHEMA),
        pl.DataFrame(outcomes, schema=source_events.SOURCE_OUTCOME_SCHEMA),
        pl.DataFrame(realized, schema=potential_history.REALIZED_TRANSITION_SCHEMA),
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

    history = potential_history.kline_path_rows(rows, 2)
    classified = classifiers.KlineClassifier("1H").classify(frame)
    missing = classifiers.KlineClassifier("1H").classify(frame.head(1))
    third = history.filter(pl.col("bar_close_ms") == 3).row(0, named=True)

    assert third["transition_path"] == "range -> markdown"
    assert third["transition_kind"] == "state_and_event_transition"
    assert third["state_age_bars"] == 1
    assert third["event_age_bars"] == 1
    assert tuple(classified.columns) == classifiers.STATE_FRAME_COLUMNS
    assert classified.select("source_family").item(0, 0) == "kline"
    assert classified.select("scale").item(0, 0) == "1H"
    assert classified.row(60, named=True)["context_event"] == "none_in_accumulation"
    assert "forward_return" not in classified.columns
    assert missing.select("missing_flag").item() is True
    assert missing.select("direction_hint").item() == "missing"

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
            potential.PotentialConfig(transition_min_directional_probability=0.9),
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
    decision = decisions.scan_review_decisions(config, (bundle,))[0]

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
            transition_horizon=1,
            transition_min_count=4,
            transition_ngram_length=4,
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
