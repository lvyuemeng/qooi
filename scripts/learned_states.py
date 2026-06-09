"""Run learned behavior-state discovery from config."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import polars as pl

from qooi.dynamic.states import (
    StateSequence,
    mean_hidden,
    project_codebook,
    summarize_hidden,
    summarize_morph,
    summarize_state_stability,
)
from qooi.exchange.store import CacheStore
from qooi.research.artifacts import ArtifactBundle
from qooi.research.behavior_tables import (
    build_state_transition_chains,
    classify_state_taxonomy,
    summarize_state_chain_information,
    summarize_state_diagnostics,
)
from qooi.research.candidates import (
    bootstrap_candidate_trades,
    build_candidate_nonoverlap_trades,
    summarize_candidate_alpha_beta,
    summarize_candidate_direction_asymmetry,
    summarize_candidate_regime_segments,
)
from qooi.research.config import ResearchCommandConfig, load_research_command_config
from qooi.research.data import load_frame_with_raw_rows, provenance_row, source_inst_ids
from qooi.research.patterns import (
    apply_candidate_gate,
    attach_forward_outcomes,
    filter_evaluation_outcomes,
    materialize_state_patterns,
    materialize_transition_patterns,
    normalize_research_frame,
    project_pattern_quality,
    project_transition_graph,
    project_transition_paths,
    summarize_returns,
    summarize_state_info,
    summarize_transition_information,
)
from qooi.research.rule_primitives import (
    build_rule_primitive_baselines,
    build_rule_primitive_signals,
    build_rule_primitive_trades,
    summarize_rule_primitives,
)

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run learned behavior-state discovery")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = _parse_args()
    command = load_research_command_config(Path(args.config))
    output_dir = command.learn.checkpoint_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = command.pairs()
    logger.info(
        "learn_start config=%s output_dir=%s ds=%s universe=%s pairs=%s timeframe=%s",
        args.config,
        output_dir,
        command.run.ds,
        command.run.universe,
        len(pairs),
        command.learn.timeframe,
    )
    load_start = time.perf_counter()
    market_frames, provenance = _load_market_frames(command)
    logger.info(
        "load_done pairs=%s elapsed_s=%.2f",
        len(market_frames),
        time.perf_counter() - load_start,
    )
    provenance_path = output_dir / "behavior-state-data-provenance.csv"
    provenance.write_csv(provenance_path)
    logger.info("provenance_written path=%s rows=%s", provenance_path, provenance.height)
    prepare_start = time.perf_counter()
    prepared = command.learn.prepare_many(frame for _pair, frame in market_frames)
    split_counts = {
        split: sum(1 for item in prepared.windows.splits if item == split)
        for split in ("train", "valid", "test")
    }
    logger.info(
        "prepare_workflow_done windows=%s train_windows=%s valid_windows=%s "
        "test_windows=%s elapsed_s=%.2f",
        len(prepared.windows.features),
        split_counts["train"],
        split_counts["valid"],
        split_counts["test"],
        time.perf_counter() - prepare_start,
    )
    checkpoint = None
    states = None
    for phase in command.learn.run.phases:
        if phase == "train":
            checkpoint = train_phase(command, prepared).checkpoint
        elif phase == "predict":
            checkpoint = checkpoint or prepared.load_checkpoint(command.learn.run_checkpoint_path())
            states = predict_phase(command, prepared, checkpoint)
        elif phase == "evaluate":
            checkpoint = checkpoint or _try_load_checkpoint(
                prepared, command.learn.run_checkpoint_path()
            )
            states = states or _load_state_sequence(
                command.learn.run_states_path(), command.learn.state_column
            )
            evaluate_phase(command, market_frames, prepared, states, checkpoint)


def train_phase(command: ResearchCommandConfig, prepared):
    train_start = time.perf_counter()
    result = prepared.train_model()
    logger.info("train_workflow_done elapsed_s=%.2f", time.perf_counter() - train_start)
    return result


def predict_phase(command: ResearchCommandConfig, prepared, checkpoint) -> StateSequence:
    predict_start = time.perf_counter()
    states = prepared.predict_states(checkpoint)
    logger.info(
        "predict_workflow_done rows=%s elapsed_s=%.2f",
        states.frame.height,
        time.perf_counter() - predict_start,
    )
    states_path = command.learn.run_states_path()
    states_path.parent.mkdir(parents=True, exist_ok=True)
    states.frame.write_csv(states_path)
    logger.info("state_sequence_written path=%s rows=%s", states_path, states.frame.height)
    return states


def evaluate_phase(
    command: ResearchCommandConfig,
    market_frames,
    prepared,
    states: StateSequence,
    checkpoint,
) -> ArtifactBundle:
    bundle_start = time.perf_counter()
    logger.info("evaluation_bundle_start")
    bundle = _build_evaluation_bundle(market_frames, states, command, prepared, checkpoint)
    bundle.write(command.learn.checkpoint_dir)
    logger.info("evaluation_bundle_done elapsed_s=%.2f", time.perf_counter() - bundle_start)
    return bundle


def _try_load_checkpoint(prepared, path: Path):
    return prepared.load_checkpoint(path) if path.exists() else None


def _load_state_sequence(path: Path, state_column: str) -> StateSequence:
    if not path.exists():
        raise FileNotFoundError(f"state sequence not found: {path}")
    return StateSequence(pl.read_csv(path), state_column=state_column)


def _load_market_frames(command: ResearchCommandConfig):
    config = command.learn
    req = command.req.model_copy(update={"days": command.days, "min": command.min_bars})
    market_frames = []
    provenance = []
    pairs = command.pairs()
    logger.info(
        "load_start pairs=%s timeframe=%s ds=%s",
        len(pairs),
        config.timeframe,
        command.run.ds,
    )
    with CacheStore() as store:
        for index, pair in enumerate(pairs, start=1):
            pair_start = time.perf_counter()
            signal_inst_id, _execution_inst_id = source_inst_ids(pair, command.run.ds)
            frame, coverage, raw_rows, note = load_frame_with_raw_rows(
                store, signal_inst_id, config.timeframe, req
            )
            market_frames.append(
                (pair, frame.with_columns(pl.lit(pair.asset.symbol).alias("symbol")))
            )
            provenance.append(
                provenance_row(
                    symbol=pair.asset.symbol,
                    inst_id=signal_inst_id,
                    bar=config.timeframe,
                    req=req,
                    raw_rows=raw_rows,
                    out=frame,
                    coverage=coverage,
                    note=note,
                )
            )
            logger.info(
                "load_pair index=%s/%s symbol=%s inst_id=%s bar=%s ds=%s rows=%s "
                "raw_rows=%s coverage_pct=%.2f refreshed=%s note=%s elapsed_s=%.2f",
                index,
                len(pairs),
                pair.asset.symbol,
                signal_inst_id,
                config.timeframe,
                command.run.ds,
                frame.height,
                raw_rows,
                coverage.coverage_pct,
                coverage.refreshed,
                note,
                time.perf_counter() - pair_start,
            )
    if not market_frames:
        raise RuntimeError("learned state discovery requires at least one market frame")
    return tuple(market_frames), pl.DataFrame(provenance)


def _build_evaluation_bundle(
    market_frames,
    states: StateSequence,
    command: ResearchCommandConfig,
    prepared,
    checkpoint,
) -> ArtifactBundle:
    start_time = time.perf_counter()
    config = command.learn
    frames_with_states = []
    research_frames = []
    for pair, market_frame in market_frames:
        frame_with_states = states.attach_to(market_frame)
        frames_with_states.append(frame_with_states)
        state_events = states.event_frame(market_frame)
        research_frames.append(
            normalize_research_frame(
                state_events,
                symbol=pair.asset.symbol,
                timeframe=config.timeframe,
                state_columns=(config.state_column,),
                event_column="liquidity_event_type",
                context_columns=(),
                state_source="vq_rssm",
            )
        )
    logger.info(
        "evaluation_frames_built rows=%s elapsed_s=%.2f",
        len(research_frames),
        time.perf_counter() - start_time,
    )
    research = pl.concat(research_frames, how="diagonal_relaxed")
    market = pl.concat(frames_with_states, how="diagonal_relaxed")
    state_distribution = _state_distribution(states, config.state_column)
    hidden_summary = pl.DataFrame()
    codebook_reconstructions = pl.DataFrame()
    if checkpoint is not None:
        logger.info("diagnostics_start")
        diagnostics = prepared.predict_diagnostics(checkpoint)
        logger.info("diagnostics_done rows=%s", len(diagnostics.codes))
        hidden_summary = summarize_hidden(states, diagnostics)
        hidden_mean = mean_hidden(diagnostics)
        codebook_reconstructions = project_codebook(
            checkpoint,
            config.feature_columns,
            (("zero_hidden", None), ("mean_hidden", hidden_mean if hidden_mean else None)),
        )
    morphology = summarize_morph(prepared, states)
    temporal_stability = summarize_state_stability(states)
    horizons = config.evaluation.horizons or command.market_state.horizons
    state_patterns = materialize_state_patterns(research, "vq_rssm")
    transition_patterns = materialize_transition_patterns(research, {"ngram_lengths": (2, 3)})
    all_patterns = pl.concat([state_patterns, transition_patterns], how="diagonal_relaxed")
    outcome_table = filter_evaluation_outcomes(
        attach_forward_outcomes(all_patterns, market, horizons),
        returns_split=config.evaluation.returns_split,
        transaction_cost_bps=config.evaluation.transaction_cost_bps,
    )
    metric_table = summarize_returns(
        outcome_table,
        ["pattern_id", "pattern_family", "pattern_source", "symbol", "horizon", "side"],
    )
    thresholds = command.research_evaluation.pattern_quality.model_dump()
    scored = apply_candidate_gate(metric_table, thresholds)
    candidate_trades = build_candidate_nonoverlap_trades(
        all_patterns,
        market,
        scored,
        returns_split=config.evaluation.returns_split,
        transaction_cost_bps=config.evaluation.transaction_cost_bps,
    )
    transition_graph = project_transition_graph(transition_patterns)
    behavior_diagnostics = summarize_state_diagnostics(
        market,
        config.state_column,
        horizons,
        split=config.evaluation.returns_split,
    )
    behavior_chains = build_state_transition_chains(
        market,
        config.state_column,
        (1, 2, 3, 4),
    )
    behavior_chain_information = summarize_state_chain_information(
        behavior_chains,
        market,
        horizons,
    )
    behavior_state_taxonomy = classify_state_taxonomy(behavior_diagnostics)
    behavior_chain_taxonomy = classify_state_taxonomy(behavior_chain_information)
    behavior_taxonomy = pl.concat(
        [behavior_state_taxonomy, behavior_chain_taxonomy], how="diagonal_relaxed"
    )
    behavior_signals = build_rule_primitive_signals(
        market,
        behavior_taxonomy,
        config.state_column,
    )
    behavior_trades = build_rule_primitive_trades(
        behavior_signals,
        market,
        transaction_cost_bps=config.evaluation.transaction_cost_bps,
    )
    tables = {
        "behavior-state-transition-graph.csv": transition_graph,
        "behavior-state-transition-matrix.csv": transition_graph,
        "behavior-state-transition-paths.csv": project_transition_paths(transition_graph),
        "behavior-state-transition-information.csv": summarize_transition_information(research),
        "behavior-state-diagnostics.csv": behavior_diagnostics,
        "behavior-state-transition-chains.csv": behavior_chains,
        "behavior-state-chain-information.csv": behavior_chain_information,
        "behavior-state-chain-taxonomy.csv": behavior_chain_taxonomy,
        "behavior-state-taxonomy.csv": behavior_state_taxonomy,
        "behavior-state-rule-primitive-signals.csv": behavior_signals,
        "behavior-state-rule-primitive-trades.csv": behavior_trades,
        "behavior-state-rule-primitive-summary.csv": summarize_rule_primitives(behavior_trades),
        "behavior-state-rule-primitive-baselines.csv": build_rule_primitive_baselines(
            market,
            horizons,
            transaction_cost_bps=config.evaluation.transaction_cost_bps,
        ),
        "behavior-state-information-metrics.csv": summarize_state_info(
            research,
            "state_value",
            ("symbol", "timeframe", "state_column"),
        ),
        "behavior-state-forward-returns.csv": project_pattern_quality(scored, ("state",)),
        "behavior-state-transition-forward-returns.csv": project_pattern_quality(
            scored, ("transition",)
        ),
        "behavior-state-hidden-summary.csv": hidden_summary,
        "behavior-state-codebook-reconstructions.csv": codebook_reconstructions,
        "behavior-state-morphology.csv": morphology,
        "behavior-state-forward-quality.csv": project_pattern_quality(
            scored, ("transition", "transition_ngram")
        ),
        "behavior-state-scored-patterns.csv": scored,
        "behavior-state-candidate-nonoverlap-trades.csv": candidate_trades,
        "behavior-state-candidate-bootstrap.csv": bootstrap_candidate_trades(candidate_trades),
        "behavior-state-candidate-direction-asymmetry.csv": summarize_candidate_direction_asymmetry(
            candidate_trades
        ),
        "behavior-state-candidate-alpha-beta.csv": summarize_candidate_alpha_beta(candidate_trades),
        "behavior-state-candidate-regime-segments.csv": summarize_candidate_regime_segments(
            candidate_trades, market
        ),
        "behavior-state-distribution-by-symbol.csv": state_distribution,
        "behavior-state-temporal-stability.csv": temporal_stability,
        "behavior-state-cross-asset-stability.csv": _cross_asset_stability(scored),
    }
    logger.info(
        "evaluation_tables_built tables=%s elapsed_s=%.2f",
        len(tables),
        time.perf_counter() - start_time,
    )
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
