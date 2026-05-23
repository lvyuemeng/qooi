"""Dynamic transition discovery bundle construction."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import polars as pl

from qooi.research import artifacts, frames, metrics, outcomes, patterns, promotion
from qooi.research.artifacts import ArtifactBundle


def build_dynamic_transition_bundle(
    prepared_frames: Iterable[pl.DataFrame],
    *,
    frame_specs: Iterable[Mapping[str, object]],
    horizons: tuple[int, ...],
    thresholds: Mapping[str, object],
) -> ArtifactBundle:
    research_frames = []
    market_frames = []
    for frame, spec in zip(prepared_frames, frame_specs, strict=False):
        market_frames.append(frame)
        research_frames.append(
            frames.normalize_research_frame(
                frame,
                symbol=str(spec["symbol"]),
                timeframe=str(spec["timeframe"]),
                state_columns=tuple(spec["state_columns"]),
                event_column=str(spec.get("event_column", "liquidity_event_type")),
                context_columns=tuple(spec.get("context_columns", ())),
            )
        )
    research_frame = frames.concat_research_frames(research_frames)
    transition_patterns = patterns.materialize_transition_patterns(
        research_frame, {"ngram_lengths": thresholds.get("ngram_lengths", (2, 3))}
    )
    none_patterns = patterns.materialize_none_event_context_patterns(
        research_frame, {"context_columns": thresholds.get("none_context_columns", ())}
    )
    all_patterns = patterns.concat_patterns([transition_patterns, none_patterns])
    market = pl.concat(market_frames, how="diagonal_relaxed") if market_frames else pl.DataFrame()
    outcome_table = outcomes.attach_forward_outcomes(all_patterns, market, horizons)
    metric_table = metrics.summarize_returns(
        outcome_table,
        ["pattern_id", "pattern_family", "pattern_source", "symbol", "horizon", "side"],
    )
    scored = promotion.apply_candidate_gate(metric_table, thresholds)
    bundle_tables = {
        "state-transition-graph.csv": artifacts.project_transition_graph(all_patterns),
        "transition-information.csv": metrics.summarize_transition_information(research_frame),
        "transition-ngram-quality.csv": artifacts.project_pattern_quality(
            scored, ("transition", "transition_ngram")
        ),
        "none-event-context-quality.csv": artifacts.project_pattern_quality(
            scored, ("none_event_context",)
        ),
        "scored-patterns.csv": scored,
        "promotion-candidates.csv": artifacts.project_promotion_candidates(scored),
    }
    return ArtifactBundle(
        "dynamic-transition-discovery",
        bundle_tables,
        summary=(f"patterns={all_patterns.height}", f"scored={scored.height}"),
    )
