"""Research scanner diagnostics and decision-only reports."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from qooi.exchange.store import HistoryCoverage
from qooi.scanner import (
    PotentialUniverse,
    ReportInputs,
    ScanDecision,
    SourceStateRow,
    SymbolStateBundle,
    TransitionAnalysis,
    TransitionEdge,
    UnsupportedTransitionPath,
)
from qooi.scanner import events as source_eval
from qooi.scanner import features as features_eval
from qooi.scanner import frames as frames_eval
from qooi.scanner import history as history_eval
from qooi.scanner import ladder as ladder_eval
from qooi.scanner import rank as candidate_eval
from qooi.scanner import rank as rank_eval
from qooi.scanner import tailrun as tailrun_eval
from qooi.scanner.classifiers import STATE_FRAME_SCHEMA, validate_state_frame
from qooi.sources.context import SourceAvailability

LadderResult = ladder_eval.LadderResult
TailtreeResult = tailrun_eval.TailtreeResult
TailtreeEvidenceResult = tailrun_eval.TailtreeEvidenceResult


@dataclass(frozen=True)
class DiagnosticFrames:
    coverage: pl.DataFrame
    history_feasibility: pl.DataFrame
    source_freshness: pl.DataFrame
    universe: pl.DataFrame
    fetch_audit: pl.DataFrame
    transition_edges: pl.DataFrame
    transition_patterns: pl.DataFrame
    unsupported_paths: pl.DataFrame
    scan_funnel: pl.DataFrame
    rejection: pl.DataFrame
    watchlist_feasibility: pl.DataFrame
    source_state_health: pl.DataFrame
    potential_observations: pl.DataFrame
    potential_evidence: pl.DataFrame
    candidate_evidence: pl.DataFrame
    candidate_rank: pl.DataFrame
    source_timeliness: pl.DataFrame
    source_state_predictability: pl.DataFrame


@dataclass(frozen=True)
class StateFrames:
    kline: pl.DataFrame
    books: pl.DataFrame
    trades: pl.DataFrame
    derivatives: pl.DataFrame
    context: pl.DataFrame


def write_diagnostics(inputs: ReportInputs) -> None:
    diagnostic_frames = _build_diagnostic_frames(inputs)

    from qooi.scanner.report import report_sections_for

    inputs.report_sections = report_sections_for(inputs.config.evidence.kind)

    state_frames = _build_state_frames(inputs)
    diagnostics = inputs.artifacts.diagnostics_dir
    states = inputs.artifacts.states_dir
    diagnostics.mkdir(parents=True, exist_ok=True)
    states.mkdir(parents=True, exist_ok=True)
    retired_artifacts = {
        "evidence-backtest.csv",
        "evidence-backtest-summary.csv",
        "evidence-baselines.csv",
        "kline-path-history.csv",
        "potential-evidence.csv",
        "potential-observation.csv",
        "realized-transition.csv",
        "source-events.csv",
        "source-outcomes.csv",
        "candidate-evidence.csv",
    }
    for directory in (diagnostics, states):
        for stale in directory.glob("*.parquet"):
            stale.unlink()
    for stale_name in retired_artifacts:
        stale = diagnostics / stale_name
        if stale.exists():
            stale.unlink()

    _write_diagnostic_frames(diagnostic_frames, diagnostics)
    _write_state_frames(state_frames, states)


def _build_diagnostic_frames(inputs: ReportInputs) -> DiagnosticFrames:
    logger = logging.getLogger("qooi.scanner")
    logger.info("features begin")
    history_feasibility = _history_feasibility_frame(inputs.bars.coverage)
    source_freshness = _source_freshness_frame(inputs.context.availability)
    bars = _bars_with_symbol(inputs.bars.frames, inputs.config.bar)
    source_events = source_eval.source_events_frame(
        inputs.context.frames,
        bars,
        inputs.config.bar,
    )
    kline_history = history_eval.kline_path_history_frame(inputs.config, inputs.bars.state_frames)
    source_outcomes = source_eval.source_outcomes_frame(source_events, bars)
    realized_transitions = history_eval.realized_transition_frame(
        kline_history,
        tuple(source_outcomes.get_column("outcome_horizon").unique().to_list())
        if not source_outcomes.is_empty()
        else (inputs.config.transition.horizon,),
    )

    continuous_features = features_eval.extract_continuous_features(
        inputs.bars.frames,
        inputs.bars.state_frames,
        inputs.context.frames,
        decision_timeframe=inputs.config.bar,
    )

    potential_observations = frames_eval.potential_observation_frame(
        kline_history,
        source_events,
        continuous_features,
        decision_timeframe=inputs.config.bar,
        max_source_staleness_hours=inputs.config.source.max_staleness_hours,
    )
    logger.info("observation rows=%d", len(potential_observations))

    logger.info("evidence begin path=%s", inputs.config.evidence.kind)
    pipeline_result = _run_pipeline(
        potential_observations, source_outcomes, realized_transitions, inputs
    )
    logger.info("evidence rows=%d", len(pipeline_result.evidence))
    logger.info(
        "candidates matched=%d ranked=%d",
        pipeline_result.candidates.height,
        pipeline_result.ranked.height,
    )
    # Populate report sections for render_report
    from qooi.scanner.report import report_sections_for

    inputs.report_sections = report_sections_for(inputs.config.evidence.kind)

    source_timeliness = source_eval.source_timeliness_frame(source_outcomes)
    source_state_predictability = source_eval.source_state_predictability_frame(
        source_outcomes,
        return_threshold_pct=inputs.config.transition.return_threshold_pct,
    )
    return DiagnosticFrames(
        coverage=coverage_frame(inputs.bars.coverage),
        history_feasibility=history_feasibility,
        source_freshness=source_freshness,
        universe=_universe_frame(inputs.universe),
        fetch_audit=_fetch_audit_frame(inputs.bars.coverage, inputs.context.availability),
        transition_edges=_transition_edges_frame(inputs.transitions.edges),
        transition_patterns=_transition_patterns_frame(inputs.transitions),
        unsupported_paths=_unsupported_paths_frame(inputs.transitions.unsupported),
        scan_funnel=_scan_funnel_frame(inputs),
        rejection=_rejection_frame(inputs.decisions),
        watchlist_feasibility=_watchlist_feasibility_frame(
            inputs.decisions,
            history_feasibility,
            source_freshness,
        ),
        source_state_health=_source_state_health_frame(inputs.bundles),
        potential_observations=potential_observations,
        potential_evidence=pipeline_result.evidence,
        candidate_evidence=pipeline_result.candidates,
        candidate_rank=pipeline_result.ranked,
        source_timeliness=source_timeliness,
        source_state_predictability=source_state_predictability,
    )


def _build_state_frames(inputs: ReportInputs) -> StateFrames:
    return StateFrames(
        kline=_state_frame_from_rows(
            tuple(bundle.kline for bundle in inputs.bundles), inputs.config.bar
        ),
        books=_state_frame_from_rows(tuple(bundle.books for bundle in inputs.bundles), "snapshot"),
        trades=_state_frame_from_rows(tuple(bundle.trades for bundle in inputs.bundles), "latest"),
        derivatives=_state_frame_from_rows(
            tuple(bundle.derivatives for bundle in inputs.bundles), "latest"
        ),
        context=_state_frame_from_rows(
            tuple(bundle.context for bundle in inputs.bundles), "latest"
        ),
    )


def _write_diagnostic_frames(frames: DiagnosticFrames, diagnostics: Path | str) -> None:
    diagnostics = Path(diagnostics)
    frame_groups = {
        "coverage": frames.coverage,
        "history-feasibility": frames.history_feasibility,
        "source-freshness": frames.source_freshness,
        "universe": frames.universe,
        "fetch-audit": frames.fetch_audit,
        "transition-edges": frames.transition_edges,
        "transition-patterns": frames.transition_patterns,
        "unsupported-current-paths": frames.unsupported_paths,
        "scan-funnel": frames.scan_funnel,
        "rejection-diagnostics": frames.rejection,
        "watchlist-feasibility": frames.watchlist_feasibility,
        "source-state-health": frames.source_state_health,
        "potential-observation-summary": (
            frames.potential_observations.group_by(
                ["source_family", "source_freshness", "market_alignment"],
                maintain_order=True,
            ).len(name="row_count")
            if not frames.potential_observations.is_empty()
            else pl.DataFrame(
                schema={
                    "source_family": pl.String,
                    "source_freshness": pl.String,
                    "market_alignment": pl.String,
                    "row_count": pl.UInt32,
                }
            )
        ),
        "potential-evidence-summary": _potential_evidence_summary(frames.potential_evidence),
        "potential-evidence-selected": _selected_potential_evidence(frames.potential_evidence),
        "candidate-inspection": frames.candidate_evidence,
        "candidate-rank": frames.candidate_rank,
        "source-timeliness": frames.source_timeliness,
        "source-state-predictability": frames.source_state_predictability,
    }
    for name, frame in frame_groups.items():
        frame.write_csv(diagnostics / f"{name}.csv")


def _potential_evidence_summary(evidence: pl.DataFrame) -> pl.DataFrame:
    if evidence.is_empty():
        return pl.DataFrame(
            schema={
                "evidence_level": pl.String,
                "evidence_status": pl.String,
                "transition_status": pl.String,
                "statistical_direction": pl.String,
                "research_suggestion": pl.String,
                "selected_evidence_level": pl.Boolean,
                "row_count": pl.UInt32,
                "median_conditioned_observations": pl.Float64,
                "median_symbol_count": pl.Float64,
                "max_information_gain_bits": pl.Float64,
                "max_transition_information_gain_bits": pl.Float64,
            }
        )
    if "evidence_level" in evidence.columns:
        return evidence.group_by(
            [
                "evidence_level",
                "evidence_status",
                "transition_status",
                "statistical_direction",
                "research_suggestion",
                "selected_evidence_level",
            ],
            maintain_order=True,
        ).agg(
            pl.len().alias("row_count"),
            pl.col("conditioned_observations").median().alias("median_conditioned_observations"),
            pl.col("symbol_count").median().alias("median_symbol_count"),
            pl.col("information_gain_bits").max().alias("max_information_gain_bits"),
            pl.col("transition_information_gain_bits")
            .max()
            .alias("max_transition_information_gain_bits"),
        )
    return evidence.group_by(
        ["tree_direction", "statistical_direction", "research_suggestion"],
        maintain_order=True,
    ).agg(
        pl.len().alias("row_count"),
        pl.col("N_total").median().alias("median_N_total"),
        pl.col("N_tail_exceedances").median().alias("median_N_tail_exceedances"),
        pl.col("tail_lift").max().alias("max_tail_lift"),
        pl.col("information_gain_bits").max().alias("max_information_gain_bits"),
    )


def _selected_potential_evidence(evidence: pl.DataFrame) -> pl.DataFrame:
    if evidence.is_empty():
        return evidence
    if "selected_evidence_level" in evidence.columns:
        return evidence.filter(pl.col("selected_evidence_level"))
    if {"N_tail_exceedances", "tail_lift", "tail_lift_stability"}.issubset(evidence.columns):
        return evidence.filter(
            (pl.col("N_tail_exceedances") >= 30)
            & (pl.col("tail_lift") >= 1.5)
            & (pl.col("tail_lift_stability") >= 0.3)
        )
    return evidence.head(0)


def _write_state_frames(frames: StateFrames, states: Path | str) -> None:
    states = Path(states)
    frame_groups = {
        "kline-state": frames.kline,
        "book-state": frames.books,
        "trade-flow-state": frames.trades,
        "derivatives-state": frames.derivatives,
        "context-state": frames.context,
    }
    for name, frame in frame_groups.items():
        frame.write_csv(states / f"{name}.csv")


def coverage_frame(coverage: tuple[HistoryCoverage, ...]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [item.inst_id for item in coverage],
            "bar": [item.bar for item in coverage],
            "target_rows": [item.target.target_bars for item in coverage],
            "target_days": [item.target.target_days for item in coverage],
            "actual_rows": [item.actual_bars for item in coverage],
            "range_start": [item.actual_start_ms for item in coverage],
            "range_end": [item.actual_end_ms for item in coverage],
            "newest_age_hours": [item.newest_age_hours for item in coverage],
            "coverage_pct": [item.coverage_pct for item in coverage],
            "gap_count": [item.gap_count for item in coverage],
            "duplicate_timestamps": [item.duplicate_timestamps for item in coverage],
            "refreshed": [item.refreshed for item in coverage],
            "notes": [";".join(item.notes) for item in coverage],
        },
        schema={
            "symbol": pl.String,
            "bar": pl.String,
            "target_rows": pl.Int64,
            "target_days": pl.Int64,
            "actual_rows": pl.Int64,
            "range_start": pl.Int64,
            "range_end": pl.Int64,
            "newest_age_hours": pl.Float64,
            "coverage_pct": pl.Float64,
            "gap_count": pl.Int64,
            "duplicate_timestamps": pl.Int64,
            "refreshed": pl.Boolean,
            "notes": pl.String,
        },
    )


def _source_freshness_frame(availability: tuple[SourceAvailability, ...]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "source_family": [row.family for row in availability],
            "symbol": [row.symbol for row in availability],
            "rows": [row.rows for row in availability],
            "latest_timestamp": [row.latest_timestamp for row in availability],
            "status": [row.status for row in availability],
            "warning": [row.warning for row in availability],
        },
        schema={
            "source_family": pl.String,
            "symbol": pl.String,
            "rows": pl.Int64,
            "latest_timestamp": pl.Int64,
            "status": pl.String,
            "warning": pl.String,
        },
    )


def _universe_frame(universe: PotentialUniverse) -> pl.DataFrame:
    if universe.discovery.is_empty():
        return pl.DataFrame(
            {
                "symbol": list(universe.symbols),
                "selected": [True for _symbol in universe.symbols],
                "eligible_count": [universe.eligible_count for _symbol in universe.symbols],
                "selection_note": [universe.selection_note for _symbol in universe.symbols],
            },
            schema={
                "symbol": pl.String,
                "selected": pl.Boolean,
                "eligible_count": pl.Int64,
                "selection_note": pl.String,
            },
        )
    return universe.discovery.with_columns(
        pl.col("symbol").is_in(universe.symbols).alias("selected"),
        pl.lit(universe.eligible_count).alias("eligible_count"),
        pl.lit(universe.selection_note).alias("selection_note"),
    )


def _fetch_audit_frame(
    coverage: tuple[HistoryCoverage, ...], availability: tuple[SourceAvailability, ...]
) -> pl.DataFrame:
    bar_rows = pl.DataFrame(
        {
            "source_family": ["kline" for _item in coverage],
            "symbol": [item.inst_id for item in coverage],
            "scale": [item.bar for item in coverage],
            "status": ["refreshed" if item.refreshed else "cached" for item in coverage],
            "rows": [item.actual_bars for item in coverage],
            "latest_timestamp": [item.actual_end_ms for item in coverage],
            "warning": [";".join(item.notes) for item in coverage],
        }
    )
    source_rows = pl.DataFrame(
        {
            "source_family": [row.family for row in availability],
            "symbol": [row.symbol for row in availability],
            "scale": ["latest" for _row in availability],
            "status": [row.status for row in availability],
            "rows": [row.rows for row in availability],
            "latest_timestamp": [row.latest_timestamp for row in availability],
            "warning": [row.warning for row in availability],
        }
    )
    frames = [frame for frame in (bar_rows, source_rows) if not frame.is_empty()]
    if not frames:
        return pl.DataFrame(
            schema={
                "source_family": pl.String,
                "symbol": pl.String,
                "scale": pl.String,
                "status": pl.String,
                "rows": pl.Int64,
                "latest_timestamp": pl.Int64,
                "warning": pl.String,
            }
        )
    return pl.concat(frames, how="diagonal_relaxed")


def _transition_edges_frame(edges: tuple[TransitionEdge, ...]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "source_family": ["kline" for _edge in edges],
            "scale": [edge.timeframe for edge in edges],
            "prev_state": [edge.prev_state for edge in edges],
            "state": [edge.state for edge in edges],
            "event": [edge.event for edge in edges],
            "count": [edge.count for edge in edges],
            "transition_probability": [edge.transition_probability for edge in edges],
            "transition_information_bits": [edge.transition_information_bits for edge in edges],
            "conditional_transition_information_bits": [
                edge.conditional_transition_information_bits for edge in edges
            ],
        }
    )


def _transition_patterns_frame(transitions: TransitionAnalysis) -> pl.DataFrame:
    patterns = tuple(
        pattern for insight in transitions.insights.values() for pattern in insight.patterns
    )
    return pl.DataFrame(
        {
            "symbol": [pattern.symbol for pattern in patterns],
            "source_family": ["kline" for _pattern in patterns],
            "scale": [pattern.timeframe for pattern in patterns],
            "path": [pattern.path for pattern in patterns],
            "event": [pattern.event for pattern in patterns],
            "count": [pattern.count for pattern in patterns],
            "symbol_count": [pattern.symbol_count for pattern in patterns],
            "effective_count": [pattern.effective_count for pattern in patterns],
            "transition_probability": [pattern.transition_probability for pattern in patterns],
            "recent_transition_probability": [
                pattern.recent_transition_probability for pattern in patterns
            ],
            "long_transition_probability": [
                pattern.long_transition_probability for pattern in patterns
            ],
            "probability_delta": [pattern.probability_delta for pattern in patterns],
            "win_rate": [pattern.win_rate for pattern in patterns],
            "p_up": [pattern.p_up for pattern in patterns],
            "p_down": [pattern.p_down for pattern in patterns],
            "average_forward_return_pct": [
                pattern.average_forward_return_pct for pattern in patterns
            ],
            "median_forward_return_pct": [
                pattern.median_forward_return_pct for pattern in patterns
            ],
            "q10_forward_return_pct": [pattern.q10_forward_return_pct for pattern in patterns],
            "q25_forward_return_pct": [pattern.q25_forward_return_pct for pattern in patterns],
            "q75_forward_return_pct": [pattern.q75_forward_return_pct for pattern in patterns],
            "q90_forward_return_pct": [pattern.q90_forward_return_pct for pattern in patterns],
            "q25_forward_min_return_pct": [
                pattern.q25_forward_min_return_pct for pattern in patterns
            ],
            "q75_forward_max_return_pct": [
                pattern.q75_forward_max_return_pct for pattern in patterns
            ],
            "loss_stop_pct": [pattern.loss_stop_pct for pattern in patterns],
            "profit_stop_pct": [pattern.profit_stop_pct for pattern in patterns],
            "reward_risk": [pattern.reward_risk for pattern in patterns],
            "omega": [pattern.omega for pattern in patterns],
            "pwpr": [pattern.pwpr for pattern in patterns],
            "transition_information_bits": [
                pattern.transition_information_bits for pattern in patterns
            ],
            "conditional_transition_information_bits": [
                pattern.conditional_transition_information_bits for pattern in patterns
            ],
            "direction": [pattern.direction for pattern in patterns],
            "suggestion": [pattern.suggestion for pattern in patterns],
        }
    )


def _unsupported_paths_frame(paths: tuple[UnsupportedTransitionPath, ...]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [path.symbol for path in paths],
            "source_family": ["kline" for _path in paths],
            "scale": [path.timeframe for path in paths],
            "path": [path.path for path in paths],
            "event": [path.event for path in paths],
            "reason": [path.reason for path in paths],
        }
    )


def _scan_funnel_frame(inputs: ReportInputs) -> pl.DataFrame:
    transition_supported = sum(
        1
        for insight in inputs.transitions.insights.values()
        if insight.current.direction in {"bullish", "bearish"}
    )
    rows = (
        ("eligible_symbols", inputs.universe.eligible_count),
        ("transition_scanned_symbols", len(inputs.universe.symbols)),
        ("directional_current_transitions", transition_supported),
        (
            "directional_review_rows",
            sum(1 for decision in inputs.decisions if decision.group in {"bullish", "bearish"}),
        ),
        ("watch_rows", sum(1 for decision in inputs.decisions if decision.group == "watch")),
        ("blocked_rows", sum(1 for decision in inputs.decisions if decision.group == "blocked")),
        ("unsupported_current_paths", len(inputs.transitions.unsupported)),
    )
    return pl.DataFrame(
        {"metric": [row[0] for row in rows], "value": [row[1] for row in rows]},
        schema={"metric": pl.String, "value": pl.Int64},
    )


def _rejection_frame(decisions: tuple[ScanDecision, ...]) -> pl.DataFrame:
    rejected = tuple(decision for decision in decisions if decision.block_reason)
    return pl.DataFrame(
        {
            "symbol": [decision.symbol for decision in rejected],
            "group": [decision.group for decision in rejected],
            "direction": [decision.direction for decision in rejected],
            "reason": [decision.block_reason for decision in rejected],
            "missing_evidence": [";".join(decision.missing_evidence) for decision in rejected],
            "contradictory_evidence": [
                ";".join(decision.contradictory_evidence) for decision in rejected
            ],
        }
    )


def _source_state_health_frame(bundles: tuple[SymbolStateBundle, ...]) -> pl.DataFrame:
    states = tuple(
        state
        for bundle in bundles
        for state in (
            bundle.kline,
            bundle.transition,
            bundle.books,
            bundle.trades,
            bundle.derivatives,
            bundle.context,
        )
    )
    return pl.DataFrame(
        {
            "symbol": [state.symbol for state in states],
            "source_family": [state.family for state in states],
            "timestamp": [state.timestamp for state in states],
            "state": [state.state for state in states],
            "direction": [state.direction for state in states],
            "quality_weight": [state.confidence for state in states],
            "missing_flag": [state.direction in {"missing", "blocked"} for state in states],
            "stale_flag": [state.stale for state in states],
            "missing_reason": [state.missing_reason for state in states],
        }
    )


def _bars_with_symbol(frames: dict[tuple[str, str], pl.DataFrame], bar: str) -> pl.DataFrame:
    symbol_frames = [
        frame.with_columns(pl.lit(symbol).alias("symbol"))
        for (symbol, timeframe), frame in frames.items()
        if timeframe == bar and not frame.is_empty()
    ]
    if not symbol_frames:
        return pl.DataFrame()
    return pl.concat(symbol_frames, how="vertical_relaxed")


def _state_frame_from_rows(rows: tuple[SourceStateRow, ...], scale: str) -> pl.DataFrame:
    return validate_state_frame(
        pl.DataFrame(
            {
                "symbol": [row.symbol for row in rows],
                "timestamp": [row.timestamp for row in rows],
                "source_family": [row.family for row in rows],
                "scale": [scale for _row in rows],
                "state_key": [row.state for row in rows],
                "context_event": [row.evidence for row in rows],
                "direction_hint": [row.direction for row in rows],
                "quality_weight": [row.confidence for row in rows],
                "missing_flag": [row.direction in {"missing", "blocked"} for row in rows],
                "stale_flag": [row.stale for row in rows],
            },
            schema=STATE_FRAME_SCHEMA,
        )
    )


def _history_feasibility_frame(coverage: tuple[HistoryCoverage, ...]) -> pl.DataFrame:
    rows = []
    for item in coverage:
        note_text = ";".join(item.notes)
        rows.append(
            {
                "symbol": item.inst_id,
                "bar": item.bar,
                "target_rows": item.target.target_bars,
                "actual_rows": item.actual_bars,
                "coverage_pct": item.coverage_pct,
                "range_start": item.actual_start_ms,
                "range_end": item.actual_end_ms,
                "newest_age_hours": item.newest_age_hours,
                "gap_count": item.gap_count,
                "duplicate_timestamps": item.duplicate_timestamps,
                "refreshed": item.refreshed,
                "feasibility_status": _history_feasibility_status(item, note_text),
                "feasibility_reason": _history_feasibility_reason(item, note_text),
                "notes": note_text,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "symbol": pl.String,
            "bar": pl.String,
            "target_rows": pl.Int64,
            "actual_rows": pl.Int64,
            "coverage_pct": pl.Float64,
            "range_start": pl.Int64,
            "range_end": pl.Int64,
            "newest_age_hours": pl.Float64,
            "gap_count": pl.Int64,
            "duplicate_timestamps": pl.Int64,
            "refreshed": pl.Boolean,
            "feasibility_status": pl.String,
            "feasibility_reason": pl.String,
            "notes": pl.String,
        },
    )


def _history_feasibility_status(item: HistoryCoverage, note_text: str) -> str:
    if item.actual_bars <= 0:
        return "missing_history"
    if "cache_only=yes" in note_text and item.coverage_pct < 95.0:
        return "cache_incomplete"
    if item.gap_count > 0 or item.duplicate_timestamps > 0:
        return "history_integrity_issue"
    if "page_error" in note_text or "HTTPStatusError" in note_text:
        return "fetch_limited"
    if "starts_after_target_since" in note_text and item.coverage_pct < 95.0:
        return "history_start_limited"
    if item.coverage_pct < 80.0:
        return "low_coverage"
    if item.coverage_pct < 95.0:
        return "partial_coverage"
    return "reviewable_history"


def _history_feasibility_reason(item: HistoryCoverage, note_text: str) -> str:
    if item.actual_bars <= 0:
        return "no bars available for requested history"
    if "cache_only=yes" in note_text and item.coverage_pct < 95.0:
        return "cache-only mode prevented filling the requested history"
    if item.gap_count > 0 or item.duplicate_timestamps > 0:
        return "timeline has gaps or duplicate timestamps"
    if "page_error" in note_text or "HTTPStatusError" in note_text:
        return "fetch stopped with provider or transport error"
    if "starts_after_target_since" in note_text and item.coverage_pct < 95.0:
        return "available exchange/cache history starts after requested horizon"
    if item.coverage_pct < 80.0:
        return "coverage is materially below requested target"
    if item.coverage_pct < 95.0:
        return "coverage is below requested target but may still support current-state review"
    return "requested history is available and timeline is clean"


def _watchlist_feasibility_frame(
    decisions: tuple[ScanDecision, ...],
    history_feasibility: pl.DataFrame,
    source_freshness: pl.DataFrame,
) -> pl.DataFrame:
    rows = pl.DataFrame(
        {
            "symbol": [decision.symbol for decision in decisions],
            "group": [decision.group for decision in decisions],
            "direction": [decision.direction for decision in decisions],
            "confidence": [decision.confidence for decision in decisions],
            "missing_evidence": [";".join(decision.missing_evidence) for decision in decisions],
            "contradictory_evidence": [
                ";".join(decision.contradictory_evidence) for decision in decisions
            ],
            "block_reason": [decision.block_reason for decision in decisions],
        },
        schema={
            "symbol": pl.String,
            "group": pl.String,
            "direction": pl.String,
            "confidence": pl.String,
            "missing_evidence": pl.String,
            "contradictory_evidence": pl.String,
            "block_reason": pl.String,
        },
    )
    history = _symbol_history_feasibility(history_feasibility)
    source = _symbol_source_feasibility(source_freshness)
    rows = rows.join(history, on="symbol", how="left")
    rows = rows.join(source, on="symbol", how="left")
    return rows.with_columns(
        pl.col("history_status").fill_null("history_missing"),
        pl.col("source_status").fill_null("sources_not_loaded"),
        pl.col("history_reason").fill_null("no history coverage row for symbol"),
        pl.col("source_reason").fill_null("no source freshness rows for symbol"),
    ).with_columns(
        pl.when(pl.col("group") == "blocked")
        .then(pl.lit("blocked_by_evidence_gate"))
        .when(pl.col("history_status").is_in(["missing_history", "history_integrity_issue"]))
        .then(pl.lit("blocked_by_history"))
        .when(pl.col("source_status") == "all_sources_missing_or_disabled")
        .then(pl.lit("source_blind_review"))
        .when(pl.col("history_status").is_in(["cache_incomplete", "fetch_limited", "low_coverage"]))
        .then(pl.lit("coverage_limited_review"))
        .otherwise(pl.lit("reviewable"))
        .alias("watchlist_feasibility")
    )


def _symbol_history_feasibility(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(
            schema={"symbol": pl.String, "history_status": pl.String, "history_reason": pl.String}
        )
    status_rank = pl.when(pl.col("feasibility_status") == "missing_history").then(0)
    status_rank = status_rank.when(pl.col("feasibility_status") == "history_integrity_issue").then(
        1
    )
    status_rank = status_rank.when(pl.col("feasibility_status") == "fetch_limited").then(2)
    status_rank = status_rank.when(pl.col("feasibility_status") == "cache_incomplete").then(3)
    status_rank = status_rank.when(pl.col("feasibility_status") == "low_coverage").then(4)
    status_rank = status_rank.when(pl.col("feasibility_status") == "history_start_limited").then(5)
    status_rank = status_rank.when(pl.col("feasibility_status") == "partial_coverage").then(6)
    status_rank = status_rank.otherwise(7).alias("status_rank")
    return (
        frame.with_columns(status_rank)
        .sort(["symbol", "status_rank", "coverage_pct"], descending=[False, False, False])
        .group_by("symbol")
        .agg(
            pl.first("feasibility_status").alias("history_status"),
            pl.first("feasibility_reason").alias("history_reason"),
            pl.min("coverage_pct").alias("min_history_coverage_pct"),
        )
    )


def _symbol_source_feasibility(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(
            schema={"symbol": pl.String, "source_status": pl.String, "source_reason": pl.String}
        )
    return (
        frame.group_by("symbol")
        .agg(
            pl.len().alias("source_family_rows"),
            pl.col("status")
            .is_in(["available", "ok"])
            .cast(pl.Int64)
            .sum()
            .alias("fresh_source_families"),
            pl.col("status")
            .is_in(["missing", "disabled"])
            .cast(pl.Int64)
            .sum()
            .alias("missing_source_families"),
            pl.concat_str("source_family", "status", separator="=")
            .str.concat(";")
            .alias("source_reason"),
        )
        .with_columns(
            pl.when(pl.col("fresh_source_families") > 0)
            .then(pl.lit("source_context_available"))
            .when(pl.col("missing_source_families") >= pl.col("source_family_rows"))
            .then(pl.lit("all_sources_missing_or_disabled"))
            .otherwise(pl.lit("source_context_partial"))
            .alias("source_status")
        )
        .select(
            "symbol",
            "source_status",
            "source_reason",
            "source_family_rows",
            "fresh_source_families",
            "missing_source_families",
        )
    )


def _run_pipeline(
    observations,
    source_outcomes,
    realized_transitions,
    inputs,
) -> LadderResult | TailtreeResult:
    """One dispatch. Returns concrete type — no downstream branching."""
    if inputs.config.evidence.kind == "tailtree":
        return _run_tailtree_pipeline(observations, source_outcomes, realized_transitions, inputs)
    return _run_ladder_pipeline(observations, source_outcomes, realized_transitions, inputs)


def _run_ladder_pipeline(
    observations,
    source_outcomes,
    realized_transitions,
    inputs,
) -> LadderResult:
    """Self-contained ladder pipeline: evidence + candidates."""
    evidence = ladder_eval.potential_evidence_frame(
        observations,
        source_outcomes,
        realized_transitions,
        return_threshold_pct=inputs.config.transition.return_threshold_pct,
    )
    candidates = candidate_eval.candidate_evidence_frame(observations, evidence)
    ranked = rank_eval.rank_candidates(candidates)
    return LadderResult(evidence=evidence, candidates=candidates, ranked=ranked, sections=())


def _run_tailtree_pipeline(
    observations,
    source_outcomes,
    realized_transitions,
    inputs,
) -> TailtreeResult:
    """Self-contained tailtree pipeline: evidence + candidates + trees."""
    result = tailrun_eval.run(observations, source_outcomes, realized_transitions, inputs)
    candidates = candidate_eval.candidate_evidence_frame(
        observations,
        result.evidence,
        tree_up=result.tree_up,
        tree_down=result.tree_down,
    )
    ranked = rank_eval.rank_candidates(candidates)
    return TailtreeResult(
        evidence=result.evidence,
        candidates=candidates,
        ranked=ranked,
        tree_up=result.tree_up,
        tree_down=result.tree_down,
        sections=(),
    )


def _tailtree_model_root(inputs) -> Path:
    return tailrun_eval._tailtree_model_root(inputs)


def _tailtree_feature_schema_hash(
    categorical_features: list[str], continuous_features: list[str]
) -> str:
    return tailrun_eval._tailtree_feature_schema_hash(categorical_features, continuous_features)


def _tailtree_artifact_metadata(
    inputs,
    tree_up: tailrun_eval.TailtreeArtifactTree | None,
    tree_down: tailrun_eval.TailtreeArtifactTree | None,
) -> tailrun_eval.TailtreeArtifactMetadata:
    return tailrun_eval._tailtree_artifact_metadata(inputs, tree_up, tree_down)


def _write_tailtree_artifacts(
    inputs,
    evidence_by_direction: dict[tailrun_eval.TailtreeDirection, pl.DataFrame],
    trees: dict[tailrun_eval.TailtreeDirection, tailrun_eval.TailtreeArtifactTree],
) -> None:
    tailrun_eval._write_tailtree_artifacts(inputs, evidence_by_direction, trees)


def _load_tail_tree_evidence(observations: pl.DataFrame, inputs):
    return tailrun_eval._load_tail_tree_evidence(observations, inputs)


def _build_tail_tree_evidence(
    observations,
    source_outcomes,
    realized_transitions,
    inputs,
):
    return tailrun_eval._build_tail_tree_evidence(
        observations, source_outcomes, realized_transitions, inputs
    )
