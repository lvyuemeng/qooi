"""Candidate discovery and ranking for accumulation scanner runs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx
import polars as pl

from qooi.accumulation.config import AccumulationConfig, DiscoveryConfig
from qooi.accumulation.schema import DISCOVERY_SCHEMA, empty_discovery_frame
from qooi.sources.okx import (
    OKX_BASE_URL,
    fetch_okx_instruments_async,
    fetch_okx_tickers_async,
)


@dataclass(frozen=True)
class DiscoveryResult:
    symbols: tuple[str, ...]
    discovery: pl.DataFrame
    manifest: pl.DataFrame


def discover_candidates(
    config: AccumulationConfig,
    *,
    top_n: int,
    min_volume_usd: float | None = None,
    symbols: tuple[str, ...] = (),
) -> DiscoveryResult:
    return asyncio.run(
        discover_candidates_async(
            config,
            top_n=top_n,
            min_volume_usd=min_volume_usd,
            symbols=symbols,
        )
    )


async def discover_candidates_async(
    config: AccumulationConfig,
    *,
    top_n: int,
    min_volume_usd: float | None = None,
    symbols: tuple[str, ...] = (),
) -> DiscoveryResult:
    async with httpx.AsyncClient(base_url=OKX_BASE_URL, timeout=20.0) as client:
        instruments_result, tickers_result = await asyncio.gather(
            fetch_okx_instruments_async(client), fetch_okx_tickers_async(client)
        )
    discovery = rank_discovery_frame(
        instruments_result.frame,
        tickers_result.frame,
        pl.DataFrame(),
        config.discovery,
        min_volume_usd=min_volume_usd,
        manual_symbols=symbols,
    )
    selected = select_candidate_symbols(discovery, top_n=top_n, manual_symbols=symbols)
    manifest = pl.concat([instruments_result.manifest, tickers_result.manifest], how="vertical")
    return DiscoveryResult(selected, discovery, manifest)


def rank_discovery_frame(
    instruments: pl.DataFrame,
    tickers: pl.DataFrame,
    cache_coverage: pl.DataFrame,
    config: DiscoveryConfig,
    *,
    min_volume_usd: float | None = None,
    manual_symbols: tuple[str, ...] = (),
) -> pl.DataFrame:
    if instruments.is_empty():
        return empty_discovery_frame()
    joined = (
        instruments.join(tickers, on="inst_id", how="left")
        if not tickers.is_empty()
        else instruments
    )
    if "history_coverage_pct" not in joined.columns:
        joined = joined.with_columns(pl.lit(None).cast(pl.Float64).alias("history_coverage_pct"))
    if not cache_coverage.is_empty() and {"symbol", "coverage_pct"}.issubset(
        cache_coverage.columns
    ):
        coverage = cache_coverage.select(
            [pl.col("symbol").alias("inst_id"), pl.col("coverage_pct").alias("_coverage_pct")]
        )
        joined = joined.join(coverage, on="inst_id", how="left").with_columns(
            pl.coalesce([pl.col("_coverage_pct"), pl.col("history_coverage_pct")]).alias(
                "history_coverage_pct"
            )
        )
    min_volume = config.min_volume_usd if min_volume_usd is None else min_volume_usd
    manual = list(manual_symbols)
    live = pl.col("state").is_in(["live", "trading"])
    has_volume = pl.col("quote_volume_24h").fill_null(0.0) >= min_volume
    has_contract = pl.col("ct_val").is_not_null()
    spread_ok = pl.col("spread_bps").is_null() | (pl.col("spread_bps") <= config.max_spread_bps)
    coverage_ok = pl.col("history_coverage_pct").is_null() | (
        pl.col("history_coverage_pct") >= config.min_history_coverage_pct
    )
    manual_expr = pl.col("inst_id").is_in(manual) if manual else pl.lit(False)
    eligible_expr = live & has_volume & has_contract & spread_ok & coverage_ok
    joined = joined.with_columns(
        [
            (eligible_expr | manual_expr).alias("eligible"),
            pl.when(~live)
            .then(pl.lit("not_live"))
            .when(~has_volume & ~manual_expr)
            .then(pl.lit("volume_below_min"))
            .when(~has_contract & ~manual_expr)
            .then(pl.lit("contract_metadata_missing"))
            .when(~spread_ok & ~manual_expr)
            .then(pl.lit("spread_above_max"))
            .when(~coverage_ok & ~manual_expr)
            .then(pl.lit("history_coverage_below_min"))
            .when(manual_expr & ~eligible_expr)
            .then(pl.lit("manual_override"))
            .otherwise(pl.lit(""))
            .alias("exclude_reason"),
            (
                (pl.col("quote_volume_24h").fill_null(0.0).clip(1.0).log10())
                - (pl.col("spread_bps").fill_null(0.0) / config.spread_bps_penalty_scale)
                + (pl.col("history_coverage_pct").fill_null(0.0) / config.coverage_bonus_scale)
                - pl.when(has_contract).then(0.0).otherwise(config.missing_contract_penalty)
            ).alias("rank_score"),
        ]
    )
    for col, dtype in DISCOVERY_SCHEMA.items():
        if col not in joined.columns:
            joined = joined.with_columns(pl.lit(None).cast(dtype).alias(col))
    return joined.select(DISCOVERY_SCHEMA.keys()).sort("rank_score", descending=True)


def select_candidate_symbols(
    discovery: pl.DataFrame, *, top_n: int, manual_symbols: tuple[str, ...] = ()
) -> tuple[str, ...]:
    if discovery.is_empty():
        return manual_symbols
    manual = list(manual_symbols)
    selected = []
    for symbol in manual:
        if symbol not in selected:
            selected.append(symbol)
    for symbol in (
        discovery.filter(pl.col("eligible"))
        .sort("rank_score", descending=True)
        .get_column("symbol")
        .to_list()
    ):
        if symbol not in selected:
            selected.append(symbol)
        if len(selected) >= top_n:
            break
    return tuple(selected[:top_n])
