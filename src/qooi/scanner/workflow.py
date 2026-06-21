"""Scanner workflow — load → compute → review → report."""

from __future__ import annotations

import asyncio
import tomllib
from pathlib import Path

import polars as pl

from qooi.pipeline import now_ms
from qooi.pipeline.coverage import CoverageRunPolicy
from qooi.pipeline.discovery import rank_discovery, select_symbols
from qooi.pipeline.load import (
    BarLoadRequest,
    MarketLoadPolicy,
    MarketLoadRequest,
    SourceLoadRequest,
    SourceProductLoadRequest,
    load_market,
)
from qooi.profiling import ProfileContext
from qooi.scanner.config import PotentialConfig
from qooi.scanner.output import MarketReadiness, ScannerRunFrames, render_report, review_decisions
from qooi.scanner.tailrun.artifacts import (
    write_tailtree_action_surface,
    write_tailtree_label_distribution,
    write_tailtree_profile_runs,
    write_tailtree_selection_efficiency,
)
from qooi.scanner.tailrun.core import run_tailtree
from qooi.scanner.tailrun.types import (
    TailtreeArtifactTree,
    TailtreeDirection,
    TailtreeInputFrames,
    TailtreeProfileFeedback,
)
from qooi.transport.okx import OkxClient


def run(config_path: Path | str) -> Path:
    return asyncio.run(_run(config_path))


def scanner_market_request(config: PotentialConfig, symbols: tuple[str, ...]) -> MarketLoadRequest:
    if config.bars is None:
        raise ValueError("scanner requires bars config")
    products = []
    for name, product_config in (
        ("books", config.books),
        ("trades", config.trades),
        ("funding", config.funding),
    ):
        if product_config is not None:
            products.append(
                SourceProductLoadRequest(
                    name,
                    product_config.limit,
                    "1H",
                    "2",
                    product_config.max_staleness_hours,
                )
            )
    for name, product_config in (
        ("open_interest", config.open_interest),
        ("taker_volume", config.taker_volume),
        ("long_short_ratios", config.long_short),
    ):
        if product_config is not None:
            products.append(
                SourceProductLoadRequest(
                    name,
                    product_config.limit,
                    product_config.period,
                    product_config.unit,
                    product_config.max_staleness_hours,
                )
            )
    target_days = min(config.bars.days, 180)
    return MarketLoadRequest(
        bars=BarLoadRequest(
            symbols=symbols,
            timeframes=config.bars.timeframes,
            target_days=target_days,
            max_staleness_hours=config.max_staleness_hours,
            latest_staleness_hours=config.bars.latest_staleness_hours,
            refresh_mode=config.bars.refresh_mode,
        ),
        sources=SourceLoadRequest(
            symbols=symbols,
            products=tuple(products),
            target_days=target_days,
            max_staleness_hours=config.max_staleness_hours,
        ),
    )


def scanner_market_policy(config: PotentialConfig) -> MarketLoadPolicy:
    return MarketLoadPolicy(
        coverage=CoverageRunPolicy(
            max_requests=1000,
            max_seconds=900,
            max_requests_per_symbol_product=24,
            concurrency=max(1, config.fetch_concurrency),
            allow_partial=True,
        )
    )


async def _run(config_path: Path | str) -> Path:
    config = load_config(Path(config_path))
    if config.bars is None:
        raise ValueError("scanner requires bars config")

    profile = ProfileContext.from_config(config.profile, config.output.parent / "profile")
    try:
        async with OkxClient() as okx:
            with profile.stage("scanner", "workflow", "resolve_universe"):
                instruments = await okx.instruments()
                tickers = await okx.tickers()
                discovery = rank_discovery(instruments.frame, tickers.frame)
                symbols = select_symbols(discovery, top_n=config.max_symbols)

            request = scanner_market_request(config, symbols)
            policy = scanner_market_policy(config)

            with profile.stage("scanner", "workflow", "load_market"):
                loaded = await load_market(okx, request, policy, instrument_frame=discovery)

        market = MarketReadiness(
            symbols=len(symbols),
            timeframes=len(config.bars.timeframes),
            target_days=request.bars.target_days,
            source_products=len(request.sources.products),
            before=loaded.coverage_before,
            after=loaded.coverage_after,
            stats=loaded.stats,
        )

        bar_frames = loaded.bar_frames
        bar_df = loaded.products["bars"].frame
        source_frames = loaded.source_frames
        source_products = {
            name: result for name, result in loaded.products.items() if name != "bars"
        }

        from qooi.scanner.state import (
            classify_states,
            continuous_features_frame,
            potential_observation_frame,
        )

        with profile.stage("scanner", "workflow", "classify_states"):
            states: dict[tuple[str, str], pl.DataFrame] = {}
            for (symbol, timeframe), frame in bar_frames.items():
                if not frame.is_empty():
                    states[(symbol, timeframe)] = classify_states(frame, scale=timeframe)

        from qooi.scanner.transitions import transitions

        with profile.stage("scanner", "workflow", "transitions"):
            transition_analysis = transitions(config, symbols, bar_frames, states)

        from qooi.scanner.outcome import (
            path_histories,
            realized_transition_frame,
            source_events,
            source_outcomes_frame,
        )

        with profile.stage("scanner", "workflow", "outcome_frames"):
            histories = path_histories(config, states, bar_frames) if states else pl.DataFrame()
            events = (
                source_events(source_frames, bar_df, config.bars.timeframes[0])
                if source_frames and not bar_df.is_empty()
                else pl.DataFrame()
            )
            realized = realized_transition_frame(
                histories, config.evidence.tailtree.outcome_horizon
            )
            source_outcomes = source_outcomes_frame(events, bar_df)

        with profile.stage("scanner", "workflow", "continuous_features"):
            continuous_features = continuous_features_frame(
                bar_frames,
                states,
                source_frames,
                decision_timeframe=config.bars.timeframes[0],
            )

        with profile.stage("scanner", "workflow", "observations"):
            observations = potential_observation_frame(
                histories,
                events,
                continuous_features,
                decision_timeframe=config.bars.timeframes[0],
                max_source_staleness_hours=config.max_staleness_hours,
            )
        profile.frame("scanner", "workflow", "observations", observations)

        tailtree_config = config.evidence.tailtree
        ladder = pl.DataFrame()
        tailtree_evidence = pl.DataFrame()
        models: dict[tuple[int, TailtreeDirection], TailtreeArtifactTree] = {}
        trial_feedback: tuple[TailtreeProfileFeedback, ...] = ()
        selection_efficiency = pl.DataFrame()
        label_distribution = pl.DataFrame()
        action_surface = pl.DataFrame()

        from qooi.scanner.rank import (
            candidate_metric_surface,
            horizon_consistency,
            ladder_candidates,
            rank_candidates,
            rank_ladder_candidates,
            rank_tailtree_candidates,
            tailtree_candidates,
        )

        ladder_ranked = pl.DataFrame()
        tailtree_ranked = pl.DataFrame()
        consistency = pl.DataFrame()

        if config.evidence.kind == "ladder" and not observations.is_empty():
            with profile.stage("scanner", "workflow", "evidence_ladder"):
                from qooi.scanner.ladder import evidence as ladder_evidence

                ladder = ladder_evidence(
                    observations,
                    source_outcomes,
                    realized,
                    return_threshold_pct=tailtree_config.threshold_pct,
                )
                ladder_branch_candidates = ladder_candidates(observations, ladder, latest_only=True)
                ladder_ranked = rank_ladder_candidates(ladder_branch_candidates)

        if config.evidence.kind == "tailtree" and not observations.is_empty():
            with profile.stage("scanner", "workflow", "evidence_tailtree"):
                tailtree_result = run_tailtree(
                    TailtreeInputFrames(
                        observations=observations,
                        source_outcomes=source_outcomes,
                        realized=realized,
                        histories=histories,
                    ),
                    config=config,
                    profile=profile,
                )
                tailtree_evidence = tailtree_result.evidence
                models = tailtree_result.models
                trial_feedback = tailtree_result.profile_runs
                selection_efficiency = tailtree_result.selection_efficiency
                label_distribution = tailtree_result.label_distribution
                action_surface = tailtree_result.action_surface
                if models and not tailtree_evidence.is_empty():
                    tailtree_branch_candidates = tailtree_candidates(
                        observations, tailtree_evidence, models, latest_only=True
                    )
                    tailtree_ranked = rank_tailtree_candidates(tailtree_branch_candidates)
                    consistency = horizon_consistency(tailtree_ranked)

        with profile.stage("scanner", "workflow", "rank_review_report"):
            candidate_surface = candidate_metric_surface(
                ladder=ladder_ranked, tailtree=tailtree_ranked
            )
            ranked = rank_candidates(candidate_surface)
            profile.frame("scanner", "workflow", "ranked", ranked)
            write_tailtree_profile_runs(config.output.parent, trial_feedback)
            write_tailtree_label_distribution(config.output.parent, label_distribution)
            write_tailtree_action_surface(config.output.parent, action_surface)
            write_tailtree_selection_efficiency(
                config.output.parent,
                config.evidence.tailtree.model_dir,
                selection_efficiency,
            )
            prediction_freshness = prediction_freshness_frame(ranked, config)
            decisions = review_decisions(
                ranked,
                prediction_freshness,
                {name: result.health for name, result in source_products.items()},
                config,
            )
            frames = ScannerRunFrames(
                market=market,
                products=loaded.products,
                states=states,
                transitions=transition_analysis,
                histories=histories,
                source_events=events,
                ladder=ladder,
                tailtree=tailtree_evidence,
                ranked=ranked,
                horizon_consistency=consistency,
                action_surface=action_surface,
                prediction_freshness=prediction_freshness,
                decisions=decisions,
            )
            report_md = render_report(frames, config)
            config.output.parent.mkdir(parents=True, exist_ok=True)
            config.output.write_text(report_md, encoding="utf-8")
        return config.output
    finally:
        profile.write()


def prediction_freshness_frame(ranked: pl.DataFrame, config: PotentialConfig) -> pl.DataFrame:
    if ranked.is_empty() or "decision_bar_close_ms" not in ranked.columns:
        return pl.DataFrame(
            schema={
                "symbol": pl.String,
                "decision_bar_close_ms": pl.Int64,
                "prediction_age_hours": pl.Float64,
                "prediction_freshness": pl.String,
            }
        )
    return (
        ranked.select("symbol", "decision_bar_close_ms")
        .unique()
        .with_columns(
            ((pl.lit(now_ms()) - pl.col("decision_bar_close_ms")) / 3_600_000).alias(
                "prediction_age_hours"
            )
        )
        .with_columns(
            pl.when(pl.col("prediction_age_hours") <= config.max_staleness_hours)
            .then(pl.lit("fresh"))
            .otherwise(pl.lit("stale"))
            .alias("prediction_freshness")
        )
    )


def load_config(path: Path) -> PotentialConfig:
    if not path.exists():
        return PotentialConfig()
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return PotentialConfig.model_validate(data.get("potential", data))
