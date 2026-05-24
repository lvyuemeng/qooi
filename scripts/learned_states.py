"""Run learned behavior-state discovery from config."""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from qooi.exchange.store import CacheStore
from qooi.research.config import ResearchCommandConfig, load_research_command_config
from qooi.research.data import load_cache_for_request, source_inst_ids
from qooi.research.states import StateSequence
from qooi.research.tables import (
    ArtifactBundle,
    apply_candidate_gate,
    attach_forward_outcomes,
    materialize_transition_patterns,
    project_pattern_quality,
    project_transition_graph,
    summarize_returns,
    summarize_transition_information,
    write_bundle,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run learned behavior-state discovery")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    command = load_research_command_config(Path(_parse_args().config))
    config = command.research_evaluation.learned_states
    output_dir = config.checkpoint_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    market_frames = _load_market_frames(command)
    prepared = config.prepare_many(frame for _pair, frame in market_frames)
    checkpoint = None
    states = None
    if "train" in config.actions:
        result = prepared.train_model()
        checkpoint = result.checkpoint
        pl.DataFrame([metric.__dict__ for metric in result.metrics]).write_csv(
            output_dir / "behavior-state-training-metrics.csv"
        )
    if "predict" in config.actions or "evaluate" in config.actions:
        checkpoint = checkpoint or prepared.load_checkpoint()
        states = prepared.predict_states(checkpoint)
        states.frame.write_csv(output_dir / "behavior-state-sequence.csv")
    if "evaluate" in config.actions:
        if states is None:
            raise RuntimeError("evaluate action requires predicted states")
        bundle = _build_evaluation_bundle(market_frames, states, command)
        write_bundle(bundle, output_dir)


def _load_market_frames(command: ResearchCommandConfig):
    config = command.research_evaluation.learned_states
    market_frames = []
    with CacheStore() as store:
        for pair in command.pairs():
            signal_inst_id, _execution_inst_id = source_inst_ids(pair, command.run.data_source)
            request = command.frame_request(pair, bar=config.timeframe)
            frame, _coverage = load_cache_for_request(
                store, signal_inst_id, config.timeframe, request
            )
            market_frames.append(
                (pair, frame.with_columns(pl.lit(pair.asset.symbol).alias("symbol")))
            )
    if not market_frames:
        raise RuntimeError("learned state discovery requires at least one market frame")
    return tuple(market_frames)


def _build_evaluation_bundle(
    market_frames,
    states: StateSequence,
    command: ResearchCommandConfig,
) -> ArtifactBundle:
    config = command.research_evaluation.learned_states
    frames_with_states = []
    research_frames = []
    for pair, market_frame in market_frames:
        frame_with_states = states.attach_to(market_frame)
        frames_with_states.append(frame_with_states)
        research_frames.append(
            states.research_frame(
                frame_with_states,
                symbol=pair.asset.symbol,
                timeframe=config.timeframe,
            )
        )
    research = pl.concat(research_frames, how="diagonal_relaxed")
    market = pl.concat(frames_with_states, how="diagonal_relaxed")
    state_distribution = _state_distribution(states, config.state_column)
    transition_patterns = materialize_transition_patterns(research, {"ngram_lengths": (2, 3)})
    outcome_table = attach_forward_outcomes(
        transition_patterns, market, command.market_state.horizons
    )
    metric_table = summarize_returns(
        outcome_table,
        ["pattern_id", "pattern_family", "pattern_source", "symbol", "horizon", "side"],
    )
    thresholds = command.research_evaluation.pattern_quality.model_dump()
    scored = apply_candidate_gate(metric_table, thresholds)
    tables = {
        "behavior-state-transition-graph.csv": project_transition_graph(
            transition_patterns
        ),
        "behavior-state-transition-information.csv": summarize_transition_information(
            research
        ),
        "behavior-state-forward-quality.csv": project_pattern_quality(
            scored, ("transition", "transition_ngram")
        ),
        "behavior-state-scored-patterns.csv": scored,
        "behavior-state-distribution-by-symbol.csv": state_distribution,
        "behavior-state-cross-asset-stability.csv": _cross_asset_stability(scored),
    }
    return ArtifactBundle(
        "behavior-state-discovery",
        tables,
        summary=(f"patterns={transition_patterns.height}", f"scored={scored.height}"),
    )


def _state_distribution(states: StateSequence, state_column: str) -> pl.DataFrame:
    if states.frame.is_empty():
        return pl.DataFrame()
    counts = states.frame.group_by("symbol", state_column).agg(pl.len().alias("rows"))
    totals = counts.group_by("symbol").agg(pl.col("rows").sum().alias("symbol_rows"))
    return counts.join(totals, on="symbol").with_columns(
        (pl.col("rows") / pl.col("symbol_rows") * 100.0).alias("symbol_pct")
    )


def _cross_asset_stability(scored: pl.DataFrame) -> pl.DataFrame:
    if scored.is_empty() or "symbol" not in scored.columns:
        return pl.DataFrame()
    keys = ["pattern_id", "pattern_family", "pattern_source", "horizon", "side"]
    return scored.group_by(keys).agg(
        pl.col("symbol").n_unique().alias("symbols"),
        pl.len().alias("rows"),
        ((pl.col("mean_side_return_pct") > 0).mean() * 100.0).alias(
            "positive_direction_agreement_pct"
        ),
        pl.col("mean_side_return_pct").mean().alias("mean_side_return_pct"),
        pl.col("passes_candidate_gate").fill_null(False).sum().alias("candidate_symbols"),
    )


if __name__ == "__main__":
    main()
