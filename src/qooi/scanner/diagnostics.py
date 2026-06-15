"""Research scanner diagnostics and decision-only reports."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import polars as pl

from qooi.exchange.store import HistoryCoverage
from qooi.profiling import ProfileContext
from qooi.scanner import (
    PotentialArtifacts,
    PotentialUniverse,
    ReportInputs,
    ScanDecision,
    SourceStateRow,
    SymbolStateBundle,
    TransitionAnalysis,
    TransitionEdge,
    UnsupportedTransitionPath,
)
from qooi.scanner import feasibility as feasibility_eval
from qooi.scanner import ladder as ladder_eval
from qooi.scanner import outcome as outcome_eval
from qooi.scanner import rank as candidate_eval
from qooi.scanner import rank as rank_eval
from qooi.scanner import state as state_eval
from qooi.scanner import tailrun as tailrun_eval
from qooi.scanner.state import STATE_FRAME_SCHEMA, validate_state_frame
from qooi.sources.context import SourceAvailability

LadderResult = ladder_eval.LadderResult
TailtreeResult = tailrun_eval.TailtreeResult
TailtreeEvidenceResult = tailrun_eval.TailtreeEvidenceResult


@dataclass(frozen=True)
class _TailtreeSettingInputs:
    config: object
    artifacts: PotentialArtifacts


@dataclass(frozen=True)
class DiagnosticFrames:
    coverage: pl.DataFrame
    history_feasibility: pl.DataFrame
    source_freshness: pl.DataFrame
    source_capability: pl.DataFrame
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
    tailtree_selection_efficiency: pl.DataFrame
    candidate_horizon_consistency: pl.DataFrame
    candidate_feasibility: pl.DataFrame
    source_timeliness: pl.DataFrame
    source_state_predictability: pl.DataFrame


@dataclass(frozen=True)
class StateFrames:
    kline: pl.DataFrame
    books: pl.DataFrame
    trades: pl.DataFrame
    derivatives: pl.DataFrame
    context: pl.DataFrame


def write_diagnostics(
    inputs: ReportInputs, profile: ProfileContext | None = None
) -> DiagnosticFrames:
    profile = ProfileContext.disabled_if_none(profile)
    diagnostic_frames = build_diagnostic_frames(inputs, profile)

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

    write_diagnostic_frames(diagnostic_frames, inputs.artifacts)
    _write_state_frames(state_frames, states)
    return diagnostic_frames


def build_diagnostic_frames(
    inputs: ReportInputs,
    profile: ProfileContext | None = None,
) -> DiagnosticFrames:
    return _build_diagnostic_frames(inputs, ProfileContext.disabled_if_none(profile))


def write_diagnostic_frames(
    frames: DiagnosticFrames,
    artifacts: PotentialArtifacts,
) -> None:
    _write_diagnostic_frames(frames, artifacts.diagnostics_dir)


def _build_diagnostic_frames(inputs: ReportInputs, profile: ProfileContext) -> DiagnosticFrames:
    logger = logging.getLogger("qooi.scanner")
    logger.info("features begin")
    with profile.stage("scanner", "diagnostics", "history_feasibility"):
        history_feasibility = feasibility_eval.history_feasibility_frame(inputs.bars.coverage)
    profile.frame("scanner", "diagnostics", "history_feasibility", history_feasibility)
    with profile.stage("scanner", "diagnostics", "source_freshness"):
        source_freshness = _source_freshness_frame(inputs.context.availability)
    profile.frame("scanner", "diagnostics", "source_freshness", source_freshness)
    bars = _bars_with_symbol(inputs.bars.frames, inputs.config.bar)
    profile.frame("scanner", "diagnostics", "decision_bars", bars)
    with profile.stage("scanner", "events", "source_events_frame"):
        source_events = outcome_eval.source_events_frame(
            inputs.context.frames,
            bars,
            inputs.config.bar,
        )
    profile.frame("scanner", "events", "source_events", source_events)
    with profile.stage("scanner", "history", "kline_path_history_frame"):
        kline_history = outcome_eval.kline_path_history_frame(
            inputs.config, inputs.bars.state_frames, inputs.bars.frames
        )
    profile.frame("scanner", "history", "kline_history", kline_history)
    with profile.stage("scanner", "events", "source_outcomes_frame"):
        source_outcomes = outcome_eval.source_outcomes_frame(source_events, bars)
    profile.frame("scanner", "events", "source_outcomes", source_outcomes)
    with profile.stage("scanner", "history", "realized_transition_frame"):
        realized_transitions = outcome_eval.realized_transition_frame(
            kline_history.filter(pl.col("timeframe") == inputs.config.bar),
            inputs.config.evidence.tailtree.outcome_horizon,
        )
    profile.frame("scanner", "history", "realized_transitions", realized_transitions)

    with profile.stage("scanner", "features", "extract_continuous_features"):
        continuous_features = state_eval.continuous_features_frame(
            inputs.bars.frames,
            inputs.bars.state_frames,
            inputs.context.frames,
            decision_timeframe=inputs.config.bar,
        )
    profile.frame("scanner", "features", "continuous_features", continuous_features)

    with profile.stage("scanner", "frames", "potential_observation_frame"):
        potential_observations = state_eval.potential_observation_frame(
            kline_history,
            source_events,
            continuous_features,
            decision_timeframe=inputs.config.bar,
            max_source_staleness_hours=inputs.config.source.max_staleness_hours,
        )
    profile.frame("scanner", "frames", "potential_observations", potential_observations)
    logger.info("observation rows=%d", len(potential_observations))

    logger.info("evidence begin path=%s", inputs.config.evidence.kind)
    with profile.stage("scanner", "diagnostics", "run_pipeline"):
        pipeline_result = _run_pipeline(
            potential_observations, source_events, source_outcomes, realized_transitions, inputs
        )
    profile.frame("scanner", "diagnostics", "potential_evidence", pipeline_result.evidence)
    profile.frame("scanner", "diagnostics", "candidate_evidence", pipeline_result.candidates)
    profile.frame("scanner", "diagnostics", "candidate_rank", pipeline_result.ranked)
    logger.info("evidence rows=%d", len(pipeline_result.evidence))
    logger.info(
        "candidates matched=%d ranked=%d",
        pipeline_result.candidates.height,
        pipeline_result.ranked.height,
    )
    with profile.stage("scanner", "events", "source_timeliness_frame"):
        source_timeliness = outcome_eval.source_timeliness_frame(source_outcomes)
    with profile.stage("scanner", "events", "source_state_predictability_frame"):
        source_state_predictability = outcome_eval.source_state_predictability_frame(
            source_outcomes,
            return_threshold_pct=inputs.config.transition.return_threshold_pct,
        )
    with profile.stage("scanner", "diagnostics", "watchlist_feasibility_frame"):
        watchlist_feasibility = feasibility_eval.watchlist_feasibility_frame(
            inputs.decisions,
            history_feasibility,
            source_freshness,
        )
    profile.frame("scanner", "diagnostics", "watchlist_feasibility", watchlist_feasibility)
    return DiagnosticFrames(
        coverage=coverage_frame(inputs.bars.coverage),
        history_feasibility=history_feasibility,
        source_freshness=source_freshness,
        source_capability=_source_capability_frame(source_freshness),
        universe=_universe_frame(inputs.universe),
        fetch_audit=_fetch_audit_frame(inputs.bars.coverage, inputs.context.availability),
        transition_edges=_transition_edges_frame(inputs.transitions.edges),
        transition_patterns=_transition_patterns_frame(inputs.transitions),
        unsupported_paths=_unsupported_paths_frame(inputs.transitions.unsupported),
        scan_funnel=_scan_funnel_frame(inputs),
        rejection=_rejection_frame(inputs.decisions),
        watchlist_feasibility=watchlist_feasibility,
        source_state_health=_source_state_health_frame(inputs.bundles),
        potential_observations=potential_observations,
        potential_evidence=pipeline_result.evidence,
        candidate_evidence=pipeline_result.candidates,
        candidate_rank=pipeline_result.ranked,
        tailtree_selection_efficiency=pipeline_result.selection_efficiency,
        candidate_horizon_consistency=rank_eval.candidate_horizon_consistency_frame(
            pipeline_result.candidates
        ),
        candidate_feasibility=feasibility_eval.candidate_feasibility_frame(
            pipeline_result.ranked,
            watchlist_feasibility,
        ),
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
        "source-capability": frames.source_capability,
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
        "tailtree-selection-efficiency": frames.tailtree_selection_efficiency,
        "candidate-horizon-consistency": frames.candidate_horizon_consistency,
        "candidate-feasibility": frames.candidate_feasibility,
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
            "latest_age_hours": [row.latest_age_hours for row in availability],
            "freshness_threshold_hours": [row.freshness_threshold_hours for row in availability],
            "provider_cap_rows": [row.provider_cap_rows for row in availability],
            "provider_cap_lookback_days": [row.provider_cap_lookback_days for row in availability],
            "coverage_target_pct": [row.coverage_target_pct for row in availability],
            "coverage_capability_pct": [row.coverage_capability_pct for row in availability],
            "status": [row.status for row in availability],
            "frame_fresh_int": [row.frame_fresh_int for row in availability],
            "frame_stale_int": [row.frame_stale_int for row in availability],
            "frame_missing_int": [row.frame_missing_int for row in availability],
            "provider_bounded_int": [row.provider_bounded_int for row in availability],
            "optional_absent_int": [row.optional_absent_int for row in availability],
            "fetch_failed_frame_fresh_int": [
                row.fetch_failed_frame_fresh_int for row in availability
            ],
            "usable_int": [row.usable_int for row in availability],
            "required_for_review_int": [row.required_for_review_int for row in availability],
            "required_for_evidence_int": [row.required_for_evidence_int for row in availability],
            "rank_penalty_weight": [row.rank_penalty_weight for row in availability],
            "source_penalty_component": [row.source_penalty_component for row in availability],
            "latest_fetch_status": [row.latest_fetch_status for row in availability],
            "latest_fetch_warning": [row.latest_fetch_warning for row in availability],
            "warning": [row.warning for row in availability],
        },
        schema={
            "source_family": pl.String,
            "symbol": pl.String,
            "rows": pl.Int64,
            "latest_timestamp": pl.Int64,
            "latest_age_hours": pl.Float64,
            "freshness_threshold_hours": pl.Float64,
            "provider_cap_rows": pl.Int64,
            "provider_cap_lookback_days": pl.Int64,
            "coverage_target_pct": pl.Float64,
            "coverage_capability_pct": pl.Float64,
            "status": pl.String,
            "frame_fresh_int": pl.Int64,
            "frame_stale_int": pl.Int64,
            "frame_missing_int": pl.Int64,
            "provider_bounded_int": pl.Int64,
            "optional_absent_int": pl.Int64,
            "fetch_failed_frame_fresh_int": pl.Int64,
            "usable_int": pl.Int64,
            "required_for_review_int": pl.Int64,
            "required_for_evidence_int": pl.Int64,
            "rank_penalty_weight": pl.Float64,
            "source_penalty_component": pl.Float64,
            "latest_fetch_status": pl.String,
            "latest_fetch_warning": pl.String,
            "warning": pl.String,
        },
    )


def _source_capability_frame(source_freshness: pl.DataFrame) -> pl.DataFrame:
    columns = [
        "source_family",
        "provider_cap_rows",
        "provider_cap_lookback_days",
        "freshness_threshold_hours",
        "required_for_review_int",
        "required_for_evidence_int",
        "rank_penalty_weight",
    ]
    if source_freshness.is_empty():
        return pl.DataFrame(
            schema={
                "source_family": pl.String,
                "provider_cap_rows": pl.Int64,
                "provider_cap_lookback_days": pl.Int64,
                "freshness_threshold_hours": pl.Float64,
                "required_for_review_int": pl.Int64,
                "required_for_evidence_int": pl.Int64,
                "rank_penalty_weight": pl.Float64,
                "symbol_count": pl.UInt32,
            }
        )
    return source_freshness.group_by(columns, maintain_order=True).len(name="symbol_count")


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




def _run_pipeline(
    observations,
    source_events,
    source_outcomes,
    realized_transitions,
    inputs,
) -> LadderResult | TailtreeResult:
    """One dispatch. Returns concrete type — no downstream branching."""
    if inputs.config.evidence.kind == "tailtree":
        return _run_tailtree_pipeline(
            observations, source_events, source_outcomes, realized_transitions, inputs
        )
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
    candidates = feasibility_eval.join_candidate_source_constraints(
        candidates, inputs.context.availability
    )
    ranked = rank_eval.rank_candidate_evidence(candidates)
    return LadderResult(
        evidence=evidence,
        candidates=candidates,
        ranked=ranked,
        selection_efficiency=pl.DataFrame(
            schema=tailrun_eval.TAILTREE_SELECTION_EFFICIENCY_SCHEMA
        ),
        sections=(),
    )


def _tailtree_setting_configs(config):
    base = config.evidence.tailtree
    settings = [base]
    for setting in base.hpo_settings:
        settings.append(
            base.model_copy(
                update={
                    "objective": setting.objective,
                    "training_profile": setting.training_profile,
                    "model_tag": setting.model_tag,
                    "num_leaves": setting.num_leaves,
                    "min_data_in_leaf": setting.min_data_in_leaf,
                    "learning_rate": setting.learning_rate,
                    "num_iterations": setting.num_iterations,
                    "early_stopping_rounds": setting.early_stopping_rounds,
                    "hpo_settings": (),
                }
            )
        )
    unique = []
    seen = set()
    for tailtree in settings:
        key = (tailtree.objective, tailtree.training_profile, tailtree.model_tag)
        if key in seen:
            continue
        seen.add(key)
        unique.append(tailtree)
    return tuple(unique)


def _config_for_tailtree_setting(config, tailtree):
    return config.model_copy(
        update={
            "evidence": config.evidence.model_copy(update={"tailtree": tailtree}),
        }
    )


def _tailtree_universe_snapshot_id(inputs) -> str:
    return (
        f"{inputs.config.bar}:{inputs.config.days}:"
        f"{inputs.universe.eligible_count}:{len(inputs.universe.symbols)}"
    )


def _run_tailtree_pipeline(
    observations,
    source_events,
    source_outcomes,
    realized_transitions,
    inputs,
) -> TailtreeResult:
    """Self-contained tailtree pipeline: evidence + candidates + trees."""
    selection_frames = []
    summary_frames = []
    primary_result = None
    primary_candidates = pl.DataFrame()
    primary_ranked = pl.DataFrame()
    setting_configs = _tailtree_setting_configs(inputs.config)
    for index, tailtree in enumerate(setting_configs):
        setting_config = _config_for_tailtree_setting(inputs.config, tailtree)
        setting_inputs = cast(
            ReportInputs,
            _TailtreeSettingInputs(setting_config, inputs.artifacts),
        )
        result = tailrun_eval.run(
            observations,
            source_outcomes,
            realized_transitions,
            setting_inputs,
            source_event_row_count=len(source_events),
        )
        candidates = candidate_eval.candidate_evidence_frame(
            observations,
            result.evidence,
            tree_models=result.models,
        )
        candidates = feasibility_eval.join_candidate_source_constraints(
            candidates, inputs.context.availability
        )
        ranked = rank_eval.rank_candidate_evidence(candidates)
        summary_path = inputs.artifacts.diagnostics_dir / "tailtree-run-summary.csv"
        summary = pl.read_csv(summary_path) if summary_path.exists() else pl.DataFrame()
        if not summary.is_empty():
            summary = summary.with_columns(
                pl.lit(tailtree.model_tag).alias("model_tag"),
                pl.lit(tailtree.training_profile).alias("training_profile"),
                pl.lit(tailtree.objective).alias("objective"),
            )
            summary_frames.append(summary)
        selection_frames.append(
            tailrun_eval.tailtree_selection_efficiency_frame(
                ranked,
                run_summary=summary,
                universe_snapshot_id=_tailtree_universe_snapshot_id(inputs),
                model_tag=tailtree.model_tag,
                objective=tailtree.objective,
                training_profile=tailtree.training_profile,
            )
        )
        if index == 0:
            primary_result = result
            primary_candidates = candidates
            primary_ranked = ranked
    selection_efficiency = (
        pl.concat(selection_frames, how="diagonal_relaxed")
        if selection_frames
        else pl.DataFrame(schema=tailrun_eval.TAILTREE_SELECTION_EFFICIENCY_SCHEMA)
    )
    tailtree = inputs.config.evidence.tailtree
    model_root = Path(tailtree.model_dir) / tailtree.model_tag
    if summary_frames:
        combined_summary = pl.concat(summary_frames, how="diagonal_relaxed")
        inputs.artifacts.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        model_root.mkdir(parents=True, exist_ok=True)
        combined_summary.write_csv(inputs.artifacts.diagnostics_dir / "tailtree-run-summary.csv")
        combined_summary.write_csv(model_root / "tailtree-run-summary.csv")
    tailrun_eval.write_tailtree_selection_efficiency(
        selection_efficiency,
        inputs.artifacts.diagnostics_dir,
        model_root,
    )
    if primary_result is None:
        primary_result = tailrun_eval.TailtreeResult(
            evidence=pl.DataFrame(),
            candidates=pl.DataFrame(),
            ranked=pl.DataFrame(),
            selection_efficiency=pl.DataFrame(),
            models={},
            sections=(),
        )
    return TailtreeResult(
        evidence=primary_result.evidence,
        candidates=primary_candidates,
        ranked=primary_ranked,
        selection_efficiency=selection_efficiency,
        models=primary_result.models,
        sections=(),
    )
