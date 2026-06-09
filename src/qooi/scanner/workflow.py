"""Potential research review report workflow."""

from __future__ import annotations

import asyncio
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, model_validator

from qooi.exchange.context import BookMode
from qooi.exchange.discovery import DiscoveryConfig, discover_candidates, empty_discovery_frame
from qooi.exchange.store import AsyncCacheStore, HistoryCoverage, HistoryRefreshRequest
from qooi.scanner.classifiers import KlineClassifier
from qooi.scanner.contracts import (
    BarFetchResult,
    PotentialArtifacts,
    PotentialUniverse,
    ReportInputs,
    context_symbols,
)
from qooi.scanner.decisions import (
    compute_kline_states,
    compute_source_states,
    scan_review_decisions,
)
from qooi.scanner.diagnostics import write_diagnostics
from qooi.scanner.report import render_report
from qooi.scanner.transitions import compute_transition_insights
from qooi.sources.context import load_source_context

RefreshMode = Literal["incremental", "cache_only", "force"]


class PotentialConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    output: Path = Path("data/output/potential/report.md")
    symbols: tuple[str, ...] = ()
    universe: str = "research"
    bar: str = "1H"
    timeframes: tuple[str, ...] = ("1H", "4H", "1D")
    days: int = 60
    refresh_mode: RefreshMode = "incremental"
    fetch_concurrency: int = 3
    source_refresh_mode: Literal["inherit", "incremental", "cache_only", "force"] = "inherit"
    disabled_sources: tuple[str, ...] = ()
    disabled_symbols: tuple[str, ...] = ()
    book_mode: BookMode = "snapshot"
    book_depth: int = 25
    max_source_staleness_hours: int = 24
    trade_limit: int = 100
    funding_limit: int = 100
    rubik_period: str = "1H"
    rubik_limit: int = 100
    rubik_taker_unit: Literal["0", "1", "2"] = "2"
    transition_horizon: int = 12
    transition_history_days: int = 0
    transition_ngram_length: int = 3
    transition_min_count: int = 20
    transition_return_threshold_pct: float = 0.0
    transition_min_information_bits: float = 0.001
    transition_min_probability: float = 0.05
    transition_min_directional_probability: float = 0.55
    transition_min_reward_risk: float = 1.0
    transition_max_tail_loss_pct: float = 20.0
    transition_recent_window: int = 240
    transition_long_window: int = 1440
    transition_min_probability_delta: float = -0.10
    transition_mae_mfe_horizon: int = 12
    require_context_for_review: bool = True
    transition_context_scope: Literal["candidates", "all_scanned"] = "candidates"
    transition_context_limit: int = 20
    transition_scan_budget: int = 80

    @model_validator(mode="after")
    def normalize_paths_and_timeframes(self) -> PotentialConfig:
        output = (
            self.output
            if self.output.name == "report.md"
            else self.output / "potential" / "report.md"
        )
        timeframes = tuple(dict.fromkeys((*self.timeframes, self.bar)))
        transition_history_days = max(0, self.transition_history_days)
        transition_ngram_length = max(2, self.transition_ngram_length)
        if (
            output == self.output
            and timeframes == self.timeframes
            and transition_history_days == self.transition_history_days
            and transition_ngram_length == self.transition_ngram_length
        ):
            return self
        return self.model_copy(
            update={
                "output": output,
                "timeframes": timeframes,
                "transition_history_days": transition_history_days,
                "transition_ngram_length": transition_ngram_length,
            }
        )


@dataclass(frozen=True)
class DiscoveryWorkflowConfig:
    discovery: DiscoveryConfig


def run(config_path: Path | str) -> Path:
    config = load_config(Path(config_path))
    artifacts = PotentialArtifacts(
        report=config.output,
        diagnostics_dir=config.output.parent / "diagnostics",
        states_dir=config.output.parent / "states",
    )
    universe = resolve_universe(config)
    bars = asyncio.run(load_bars(config, universe.symbols))
    kline_states = compute_kline_states(
        config, universe.symbols, bars.state_frames, bars.frames, bars.coverage
    )
    transitions = compute_transition_insights(
        config, universe.symbols, bars.frames, bars.state_frames
    )
    symbols_with_decision_bars = tuple(
        symbol
        for symbol in universe.symbols
        if not bars.frames.get((symbol, config.bar), pl.DataFrame()).is_empty()
    )
    context = asyncio.run(
        load_source_context(
            config,
            symbols=universe.symbols,
            context_symbols=context_symbols(
                config, symbols_with_decision_bars, transitions.insights
            ),
            discovery=universe.discovery,
        )
    )
    bundles = compute_source_states(
        config, universe.symbols, kline_states, transitions.insights, bars.coverage, context
    )
    decisions = tuple(scan_review_decisions(config, bundles))
    inputs = ReportInputs(
        config, artifacts, universe, bars, context, transitions, bundles, decisions
    )
    write_diagnostics(inputs)
    artifacts.report.parent.mkdir(parents=True, exist_ok=True)
    artifacts.report.write_text(render_report(inputs), encoding="utf-8")
    return artifacts.report


def load_config(path: Path) -> PotentialConfig:
    if not path.exists():
        return PotentialConfig()
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return PotentialConfig.model_validate(data.get("potential", data))


def target_min_bars(days: int, timeframe: str) -> int:
    unit = timeframe[-1].lower()
    value_text = timeframe[:-1]
    value = int(value_text) if value_text.isdigit() else 1
    minutes = (
        value
        if unit == "m"
        else value * 60
        if unit == "h"
        else value * 60 * 24
        if unit == "d"
        else 60
    )
    return max(int(days * 24 * 60 / minutes), 120)


def resolve_universe(config: PotentialConfig) -> PotentialUniverse:
    if config.symbols:
        symbols = tuple(dict.fromkeys(config.symbols))
        return PotentialUniverse(
            symbols=symbols,
            discovery=empty_discovery_frame(),
            selection_note="explicit symbols override OKX universe discovery",
            missing_reason="",
            eligible_count=len(symbols),
        )
    try:
        result = discover_candidates(DiscoveryWorkflowConfig(DiscoveryConfig()))
    except Exception as exc:
        return PotentialUniverse(
            symbols=(),
            discovery=empty_discovery_frame(),
            selection_note="okx universe discovery failed",
            missing_reason=f"{type(exc).__name__}: {exc}",
        )
    if result.discovery.is_empty() or "symbol" not in result.discovery.columns:
        symbols = result.symbols[: config.transition_scan_budget]
    else:
        frame = result.discovery.filter(pl.col("symbol").is_in(result.symbols))
        if "rank_score" in frame.columns:
            frame = frame.sort("rank_score", descending=True)
        symbols = tuple(frame.head(config.transition_scan_budget).get_column("symbol").to_list())
    return PotentialUniverse(
        symbols=symbols,
        discovery=result.discovery,
        selection_note=(
            "symbols selected from OKX swap universe discovery and transition scan budget"
        ),
        missing_reason="" if symbols else "okx discovery returned no eligible symbols",
        eligible_count=len(result.symbols),
    )


async def load_bars(config: PotentialConfig, symbols: tuple[str, ...]) -> BarFetchResult:
    if not symbols:
        return BarFetchResult({}, {}, ())
    semaphore = asyncio.Semaphore(max(1, config.fetch_concurrency))
    frames: dict[tuple[str, str], pl.DataFrame] = {}
    coverage: list[HistoryCoverage] = []

    async with AsyncCacheStore() as store:

        async def load(
            symbol: str, timeframe: str
        ) -> tuple[str, str, pl.DataFrame, HistoryCoverage]:
            request = HistoryRefreshRequest(
                inst_id=symbol,
                bar=timeframe,
                days=max(config.days, config.transition_history_days),
                min_bars=target_min_bars(
                    max(config.days, config.transition_history_days), timeframe
                ),
                refresh=config.refresh_mode in {"incremental", "force"},
                incremental=config.refresh_mode != "force",
                cache_only=config.refresh_mode == "cache_only",
            )
            async with semaphore:
                frame, item = await store.bars(request)
            return symbol, timeframe, frame, item

        requests = ((symbol, timeframe) for symbol in symbols for timeframe in config.timeframes)
        for symbol, timeframe, frame, item in await asyncio.gather(
            *(load(symbol, timeframe) for symbol, timeframe in requests)
        ):
            frames[(symbol, timeframe)] = frame
            coverage.append(item)
    state_frames = _classify_kline_frames(config, symbols, frames)
    return BarFetchResult(frames, state_frames, tuple(coverage))


def _classify_kline_frames(
    config: PotentialConfig,
    symbols: tuple[str, ...],
    frames: dict[tuple[str, str], pl.DataFrame],
) -> dict[tuple[str, str], pl.DataFrame]:
    states: dict[tuple[str, str], pl.DataFrame] = {}
    jobs = [
        (symbol, timeframe, frame)
        for timeframe in config.timeframes
        for symbol in symbols
        if not (frame := frames.get((symbol, timeframe), pl.DataFrame())).is_empty()
    ]
    workers = min(8, max(1, config.fetch_concurrency * 2))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_classify_kline_frame, symbol, timeframe, frame): (symbol, timeframe)
            for symbol, timeframe, frame in jobs
        }
        for future in as_completed(futures):
            states[futures[future]] = future.result()
    return states


def _classify_kline_frame(symbol: str, timeframe: str, frame: pl.DataFrame) -> pl.DataFrame:
    return KlineClassifier(timeframe).classify(
        frame.with_columns(pl.lit(symbol).alias("symbol"))
    )
