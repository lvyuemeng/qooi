"""Broad-market discovery collection, ranking, and OKX mapping."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

import httpx
import polars as pl

from qooi.accumulation.config import AccumulationConfig, BroadScanConfig
from qooi.accumulation.schema import (
    BROAD_CANDIDATE_SCHEMA,
    BROAD_MARKET_SCHEMA,
    empty_broad_candidate_frame,
    empty_broad_market_frame,
    empty_broad_news_frame,
    empty_broad_protocol_frame,
    empty_source_manifest_frame,
)
from qooi.sources.coingecko import (
    COINGECKO_BASE_URL,
    fetch_coingecko_markets_async,
    fetch_coingecko_trending_async,
)
from qooi.sources.coinpaprika import COINPAPRIKA_BASE_URL, fetch_coinpaprika_tickers_async
from qooi.sources.cryptopanic import (
    CRYPTOPANIC_BASE_URL,
    fetch_cryptopanic_global_posts_async,
    missing_api_key_result,
)
from qooi.sources.defillama import DEFILLAMA_BASE_URL, fetch_defillama_protocols_async
from qooi.sources.models import SourceResult


@dataclass(frozen=True)
class BroadDiscoveryResult:
    market: pl.DataFrame
    protocols: pl.DataFrame
    news: pl.DataFrame
    candidates: pl.DataFrame
    manifest: pl.DataFrame


def collect_broad_sources(config: AccumulationConfig, *, broad_top_n: int) -> BroadDiscoveryResult:
    return asyncio.run(collect_broad_sources_async(config, broad_top_n=broad_top_n))


def collect_potential_board_universe(
    config: AccumulationConfig, *, top_n: int
) -> BroadDiscoveryResult:
    return asyncio.run(collect_potential_board_universe_async(config, top_n=top_n))


async def collect_potential_board_universe_async(
    config: AccumulationConfig, *, top_n: int
) -> BroadDiscoveryResult:
    """Collect the reduced potential-board universe.

    Active sources are CoinGecko markets/trending plus OKX discovery supplied by
    the caller. CoinGecko trending is retained only as attention annotation; it
    cannot pass the market/liquidity gate without market rows.
    """
    broad = config.broad_scan
    api_key = os.getenv(broad.coingecko.api_key_env, "")
    async with httpx.AsyncClient(base_url=COINGECKO_BASE_URL, timeout=20.0) as client:
        calls = [
            fetch_coingecko_markets_async(
                client,
                page=page,
                per_page=broad.coingecko.per_page,
                api_key=api_key,
                vs_currency=broad.coingecko.vs_currency,
                order=broad.coingecko.order,
                price_change_percentage=broad.coingecko.price_change_percentage,
            )
            for page in range(1, broad.coingecko_pages + 1)
        ]
        if broad.coingecko.include_trending:
            calls.append(fetch_coingecko_trending_async(client, api_key=api_key))
        market_results = await asyncio.gather(*calls)
    market = _concat_or_empty(
        [result.frame for result in market_results], empty_broad_market_frame()
    )
    if not market.is_empty():
        heat = market.filter(pl.col("heat_source").fill_null("") != "")
        regular = market.filter(pl.col("heat_source").fill_null("") == "").head(broad.max_assets)
        market = _concat_or_empty([heat, regular], empty_broad_market_frame())
    manifest = _concat_or_empty(
        [result.manifest for result in market_results], empty_source_manifest_frame()
    )
    candidates = rank_potential_board_universe(market, broad).head(top_n)
    return BroadDiscoveryResult(
        market=market,
        protocols=empty_broad_protocol_frame(),
        news=empty_broad_news_frame(),
        candidates=candidates,
        manifest=manifest,
    )


async def collect_broad_sources_async(
    config: AccumulationConfig, *, broad_top_n: int
) -> BroadDiscoveryResult:
    broad = config.broad_scan
    market_results: list[SourceResult] = []
    protocol_results: list[SourceResult] = []
    news_results: list[SourceResult] = []
    providers = set(broad.providers) | set(broad.optional_providers)
    if "coingecko" in providers:
        api_key = os.getenv(broad.coingecko.api_key_env, "")
        async with httpx.AsyncClient(base_url=COINGECKO_BASE_URL, timeout=20.0) as client:
            coingecko_calls = [
                fetch_coingecko_markets_async(
                    client,
                    page=page,
                    per_page=broad.coingecko.per_page,
                    api_key=api_key,
                    vs_currency=broad.coingecko.vs_currency,
                    order=broad.coingecko.order,
                    price_change_percentage=broad.coingecko.price_change_percentage,
                )
                for page in range(1, broad.coingecko_pages + 1)
            ]
            if broad.coingecko.include_trending:
                coingecko_calls.append(fetch_coingecko_trending_async(client, api_key=api_key))
            market_results.extend(await asyncio.gather(*coingecko_calls))
    if "coinpaprika" in providers:
        async with httpx.AsyncClient(base_url=COINPAPRIKA_BASE_URL, timeout=20.0) as client:
            market_results.append(await fetch_coinpaprika_tickers_async(client))
    if "defillama" in providers:
        async with httpx.AsyncClient(base_url=DEFILLAMA_BASE_URL, timeout=30.0) as client:
            protocol_results.append(await fetch_defillama_protocols_async(client))
    if "cryptopanic" in providers:
        api_key = os.getenv(broad.cryptopanic.api_key_env, "")
        if api_key:
            async with httpx.AsyncClient(base_url=CRYPTOPANIC_BASE_URL, timeout=20.0) as client:
                news_results.append(
                    await fetch_cryptopanic_global_posts_async(
                        client, api_key=api_key, limit=broad.cryptopanic.limit
                    )
                )
        elif not broad.cryptopanic.enabled_without_key:
            news_results.append(missing_api_key_result(api_key_env=broad.cryptopanic.api_key_env))
    market = _concat_or_empty(
        [result.frame for result in market_results], empty_broad_market_frame()
    )
    if not market.is_empty():
        heat = market.filter(pl.col("heat_source").fill_null("") != "")
        regular = market.filter(pl.col("heat_source").fill_null("") == "").head(broad.max_assets)
        market = _concat_or_empty([heat, regular], empty_broad_market_frame())
    protocols = _concat_or_empty(
        [result.frame for result in protocol_results], empty_broad_protocol_frame()
    )
    news = _concat_or_empty([result.frame for result in news_results], empty_broad_news_frame())
    manifest = _concat_or_empty(
        [result.manifest for result in (*market_results, *protocol_results, *news_results)],
        empty_source_manifest_frame(),
    )
    candidates = rank_broad_candidates(market, protocols, news, broad).head(broad_top_n)
    return BroadDiscoveryResult(market, protocols, news, candidates, manifest)


def rank_broad_candidates(
    market: pl.DataFrame, protocols: pl.DataFrame, news: pl.DataFrame, config: BroadScanConfig
) -> pl.DataFrame:
    if market.is_empty():
        return empty_broad_candidate_frame()
    market = _dedupe_market(market)
    protocol_summary = _protocol_summary(protocols)
    news_summary = _news_summary(news)
    frame = market.join(protocol_summary, on="base_ccy", how="left").join(
        news_summary, on="base_ccy", how="left"
    )
    excluded = [base.upper() for base in config.excluded_base_ccy]
    stablecoin = pl.col("base_ccy").is_in(excluded)
    market_cap_low = pl.col("market_cap_usd").fill_null(0.0) < config.min_market_cap_usd
    market_cap_high = pl.col("market_cap_usd").fill_null(float("inf")) > config.max_market_cap_usd
    volume_low = pl.col("volume_24h_usd").fill_null(0.0) < config.min_volume_24h_usd
    active_1h = pl.col("price_change_pct_1h").abs().fill_null(0.0) >= 1.0
    active_24h = pl.col("price_change_pct_24h").abs().fill_null(0.0) >= 3.0
    tvl_growth = pl.col("tvl_change_1d_pct").fill_null(0.0) >= config.tvl_change_1d_min_pct
    has_news = pl.col("news_mentions").fill_null(0) > 0
    trending = pl.col("trending_rank").is_not_null()
    trending_top5 = pl.col("trending_rank") <= 5
    frame = frame.with_columns(
        [
            pl.when(stablecoin)
            .then(pl.lit("stablecoin_excluded"))
            .when(market_cap_low)
            .then(pl.lit("market_cap_below_min"))
            .when(market_cap_high)
            .then(pl.lit("market_cap_above_max"))
            .when(volume_low)
            .then(pl.lit("volume_below_min"))
            .otherwise(pl.lit(""))
            .alias("exclude_reason"),
            (
                pl.col("volume_24h_usd").fill_null(0.0).clip(1.0).log10()
                + pl.col("market_cap_usd").fill_null(0.0).clip(1.0).log10() * 0.25
                + pl.when(active_1h).then(2.0).otherwise(0.0)
                + pl.when(active_24h).then(1.0).otherwise(0.0)
                + pl.when(tvl_growth).then(2.0).otherwise(0.0)
                + pl.when(has_news)
                .then(pl.col("news_mentions").fill_null(0).clip(0, 5))
                .otherwise(0)
                + pl.when(trending).then(config.coingecko.trending_weight).otherwise(0.0)
                + pl.when(trending_top5).then(config.coingecko.trending_weight * 0.5).otherwise(0.0)
            ).alias("broad_score"),
            _reason_expr(
                active_1h, active_24h, tvl_growth, has_news, trending, trending_top5
            ).alias("broad_reasons"),
        ]
    )
    frame = frame.with_columns(
        pl.lit(None).cast(pl.String).alias("okx_symbol"),
        pl.lit(False).alias("okx_mapped"),
    )
    if "rank" in frame.columns:
        frame = frame.drop("rank")
    frame = frame.sort("broad_score", descending=True).with_row_index("rank", offset=1)
    return _coerce_candidates(frame.select(BROAD_CANDIDATE_SCHEMA.keys()))


def rank_potential_board_universe(
    market: pl.DataFrame, config: BroadScanConfig
) -> pl.DataFrame:
    if market.is_empty():
        return empty_broad_candidate_frame()
    frame = _dedupe_market(market)
    excluded = [base.upper() for base in config.excluded_base_ccy]
    has_market = pl.col("market_cap_usd").is_not_null() & pl.col("volume_24h_usd").is_not_null()
    stablecoin = pl.col("base_ccy").is_in(excluded)
    market_cap_low = pl.col("market_cap_usd").fill_null(0.0) < config.min_market_cap_usd
    market_cap_high = pl.col("market_cap_usd").fill_null(float("inf")) > config.max_market_cap_usd
    volume_low = pl.col("volume_24h_usd").fill_null(0.0) < config.min_volume_24h_usd
    active_1h = pl.col("price_change_pct_1h").abs().fill_null(0.0) >= 1.0
    active_24h = pl.col("price_change_pct_24h").abs().fill_null(0.0) >= 3.0
    trending = pl.col("trending_rank").is_not_null()
    trending_top5 = pl.col("trending_rank") <= 5
    frame = frame.with_columns(
        [
            pl.when(~has_market)
            .then(pl.lit("market_data_missing"))
            .when(stablecoin)
            .then(pl.lit("stablecoin_excluded"))
            .when(market_cap_low)
            .then(pl.lit("market_cap_below_min"))
            .when(market_cap_high)
            .then(pl.lit("market_cap_above_max"))
            .when(volume_low)
            .then(pl.lit("volume_below_min"))
            .otherwise(pl.lit(""))
            .alias("exclude_reason"),
            pl.col("volume_24h_usd").fill_null(0.0).clip(1.0).log10().alias(
                "broad_score"
            ),
            _reason_expr(
                active_1h,
                active_24h,
                pl.lit(False),
                pl.lit(False),
                trending,
                trending_top5,
            ).alias("broad_reasons"),
            pl.lit(None).cast(pl.Float64).alias("tvl_usd"),
            pl.lit(None).cast(pl.Float64).alias("tvl_change_1d_pct"),
            pl.lit(None).cast(pl.Int64).alias("news_mentions"),
            pl.lit(None).cast(pl.String).alias("okx_symbol"),
            pl.lit(False).alias("okx_mapped"),
        ]
    )
    if "rank" in frame.columns:
        frame = frame.drop("rank")
    frame = frame.sort(
        ["exclude_reason", "volume_24h_usd", "market_cap_usd", "base_ccy"],
        descending=[False, True, True, False],
        nulls_last=True,
    ).with_row_index("rank", offset=1)
    return _coerce_candidates(frame.select(BROAD_CANDIDATE_SCHEMA.keys()))


def map_broad_to_okx(candidates: pl.DataFrame, okx_discovery: pl.DataFrame) -> pl.DataFrame:
    if candidates.is_empty():
        return empty_broad_candidate_frame()
    required = {"base_ccy", "symbol", "eligible"}
    if okx_discovery.is_empty() or not required.issubset(okx_discovery.columns):
        return _coerce_candidates(
            candidates.with_columns(
                pl.lit(False).alias("okx_mapped"),
                pl.when(pl.col("exclude_reason") == "")
                .then(pl.lit("okx_swap_not_listed"))
                .otherwise(pl.col("exclude_reason"))
                .alias("exclude_reason"),
            )
        )
    if "exclude_reason" not in okx_discovery.columns:
        okx_discovery = okx_discovery.with_columns(pl.lit("").alias("exclude_reason"))
    eligible_okx = (
        okx_discovery.filter(pl.col("eligible"))
        .sort("rank_score", descending=True)
        .unique(subset=["base_ccy"], keep="first")
        .select(["base_ccy", pl.col("symbol").alias("_okx_symbol")])
    )
    listed_okx = (
        okx_discovery.sort("rank_score", descending=True)
        .unique(subset=["base_ccy"], keep="first")
        .select(
            [
                "base_ccy",
                pl.col("symbol").alias("_listed_okx_symbol"),
                pl.col("exclude_reason").fill_null("").alias("_listed_exclude_reason"),
            ]
        )
    )
    mapped = (
        candidates.join(eligible_okx, on="base_ccy", how="left")
        .join(listed_okx, on="base_ccy", how="left")
        .with_columns(
            [
                pl.coalesce(
                    [pl.col("_okx_symbol"), pl.col("_listed_okx_symbol"), pl.lit("")]
                ).alias("okx_symbol"),
                pl.col("_okx_symbol").is_not_null().alias("okx_mapped"),
                pl.when(
                    (pl.col("exclude_reason") == "")
                    & pl.col("_okx_symbol").is_null()
                    & pl.col("_listed_okx_symbol").is_not_null()
                )
                .then(
                    pl.concat_str(
                        [
                            pl.lit("okx_swap_ineligible"),
                            pl.col("_listed_exclude_reason").fill_null("").str.strip_chars(),
                        ],
                        separator="_",
                    ).str.strip_chars("_")
                )
                .when((pl.col("exclude_reason") == "") & pl.col("_okx_symbol").is_null())
                .then(pl.lit("okx_swap_not_listed"))
                .otherwise(pl.col("exclude_reason"))
                .alias("exclude_reason"),
            ]
        )
    )
    return _coerce_candidates(mapped)


def select_deep_symbols(broad_candidates: pl.DataFrame, *, deep_top_n: int) -> tuple[str, ...]:
    if broad_candidates.is_empty():
        return ()
    frame = broad_candidates.filter(pl.col("okx_mapped") & (pl.col("exclude_reason") == ""))
    if frame.is_empty():
        return ()
    return tuple(
        frame.sort("broad_score", descending=True).head(deep_top_n)["okx_symbol"].to_list()
    )


def select_potential_board_symbols(
    broad_candidates: pl.DataFrame, *, deep_top_n: int
) -> tuple[str, ...]:
    if broad_candidates.is_empty():
        return ()
    frame = broad_candidates.filter(pl.col("okx_mapped") & (pl.col("exclude_reason") == ""))
    if frame.is_empty():
        return ()
    return tuple(
        frame.sort(
            ["volume_24h_usd", "market_cap_usd", "okx_symbol"],
            descending=[True, True, False],
            nulls_last=True,
        )
        .head(deep_top_n)["okx_symbol"]
        .to_list()
    )


def _dedupe_market(market: pl.DataFrame) -> pl.DataFrame:
    frame = market.filter(pl.col("base_ccy").is_not_null() & (pl.col("base_ccy") != ""))
    if frame.is_empty():
        return empty_broad_market_frame()
    for col, dtype in BROAD_MARKET_SCHEMA.items():
        if col not in frame.columns:
            frame = frame.with_columns(pl.lit(None).cast(dtype).alias(col))
    return (
        frame.sort(
            ["volume_24h_usd", "market_cap_usd", "trending_rank"],
            descending=[True, True, False],
            nulls_last=True,
        )
        .group_by("base_ccy")
        .agg(
            [
                pl.col("timestamp").max(),
                pl.col("provider")
                .drop_nulls()
                .unique(maintain_order=True)
                .str.join(";")
                .alias("provider"),
                pl.col("coin_id").drop_nulls().first(),
                pl.col("name").drop_nulls().first(),
                pl.col("rank").min(),
                pl.col("price_usd").drop_nulls().first(),
                pl.col("market_cap_usd").max(),
                pl.col("volume_24h_usd").max(),
                pl.col("volume_24h_change_pct").max(),
                pl.col("price_change_pct_1h").max(),
                pl.col("price_change_pct_24h").max(),
                pl.col("last_updated").max(),
                pl.col("trending_rank").min(),
                pl.col("trending_score").max(),
                pl.col("heat_source")
                .drop_nulls()
                .filter(pl.col("heat_source") != "")
                .unique(maintain_order=True)
                .str.join(";")
                .alias("heat_source"),
            ]
        )
    )


def _protocol_summary(protocols: pl.DataFrame) -> pl.DataFrame:
    if protocols.is_empty() or "base_ccy" not in protocols.columns:
        return pl.DataFrame(
            schema={
                "base_ccy": pl.String,
                "tvl_usd": pl.Float64,
                "tvl_change_1d_pct": pl.Float64,
            }
        )
    return protocols.group_by("base_ccy").agg(
        pl.col("tvl_usd").max(), pl.col("tvl_change_1d_pct").max()
    )


def _news_summary(news: pl.DataFrame) -> pl.DataFrame:
    if news.is_empty() or "base_ccy" not in news.columns:
        return pl.DataFrame(schema={"base_ccy": pl.String, "news_mentions": pl.Int64})
    return (
        news.filter(pl.col("base_ccy") != "")
        .group_by("base_ccy")
        .agg(pl.len().alias("news_mentions"))
    )


def _reason_expr(
    active_1h: pl.Expr,
    active_24h: pl.Expr,
    tvl_growth: pl.Expr,
    has_news: pl.Expr,
    trending: pl.Expr,
    trending_top5: pl.Expr,
) -> pl.Expr:
    return pl.concat_str(
        [
            pl.when(active_1h).then(pl.lit("active_1h;")).otherwise(pl.lit("")),
            pl.when(active_24h).then(pl.lit("active_24h;")).otherwise(pl.lit("")),
            pl.when(tvl_growth).then(pl.lit("tvl_growth;")).otherwise(pl.lit("")),
            pl.when(has_news).then(pl.lit("news_mentions;")).otherwise(pl.lit("")),
            pl.when(trending).then(pl.lit("coingecko_trending;")).otherwise(pl.lit("")),
            pl.when(trending_top5).then(pl.lit("trending_top5;")).otherwise(pl.lit("")),
        ]
    ).str.strip_chars(";")


def _coerce_candidates(frame: pl.DataFrame) -> pl.DataFrame:
    for col, dtype in BROAD_CANDIDATE_SCHEMA.items():
        if col not in frame.columns:
            frame = frame.with_columns(pl.lit(None).cast(dtype).alias(col))
        else:
            frame = frame.with_columns(pl.col(col).cast(dtype, strict=False).alias(col))
    return frame.select(BROAD_CANDIDATE_SCHEMA.keys())


def _concat_or_empty(frames: list[pl.DataFrame], empty: pl.DataFrame) -> pl.DataFrame:
    frames = [frame for frame in frames if not frame.is_empty()]
    return pl.concat(frames, how="vertical_relaxed") if frames else empty

