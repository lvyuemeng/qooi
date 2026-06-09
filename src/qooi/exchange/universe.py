"""Broad-market discovery collection, ranking, and OKX mapping."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Protocol

import httpx
import polars as pl

from qooi.exchange.discovery import DiscoveryConfig
from qooi.sources.coingecko import (
    COINGECKO_BASE_URL,
    fetch_coingecko_markets,
    fetch_coingecko_trending,
)
from qooi.sources.coinpaprika import COINPAPRIKA_BASE_URL, fetch_coinpaprika_tickers
from qooi.sources.cryptopanic import (
    CRYPTOPANIC_BASE_URL,
    fetch_cryptopanic_global_posts,
    missing_api_key_result,
)
from qooi.sources.defillama import DEFILLAMA_BASE_URL, fetch_defillama_protocols
from qooi.sources.models import SourceResult
from qooi.sources.schema import SOURCE_MANIFEST_SCHEMA


@dataclass(frozen=True)
class BroadCoinGeckoConfig:
    api_key_env: str = "COINGECKO_DEMO_API_KEY"
    vs_currency: str = "usd"
    order: str = "volume_desc"
    per_page: int = 250
    price_change_percentage: tuple[str, ...] = ("1h", "24h")
    include_trending: bool = True
    trending_weight: float = 4.0


@dataclass(frozen=True)
class BroadCryptoPanicConfig:
    api_key_env: str = "CRYPTOPANIC_API_KEY"
    enabled_without_key: bool = False
    limit: int = 100


@dataclass(frozen=True)
class BroadScanConfig:
    providers: tuple[str, ...] = ("coingecko", "coinpaprika", "defillama")
    optional_providers: tuple[str, ...] = ("cryptopanic",)
    max_assets: int = 500
    coingecko_pages: int = 2
    coinpaprika_max_assets: int = 1000
    min_market_cap_usd: float = 1_000_000.0
    max_market_cap_usd: float = 500_000_000.0
    min_volume_24h_usd: float = 150_000.0
    min_turnover_ratio: float = 0.0
    oversold_enabled: bool = True
    oversold_min_market_cap_usd: float = 5_000_000.0
    oversold_return_24h_max: float = -15.0
    tvl_change_1d_min_pct: float = 20.0
    output_top_n: int = 25
    excluded_base_ccy: tuple[str, ...] = ("USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDE")
    coingecko: BroadCoinGeckoConfig = BroadCoinGeckoConfig()
    cryptopanic: BroadCryptoPanicConfig = BroadCryptoPanicConfig()


class PotentialScanConfig(Protocol):
    trend_excluded_base_ccy: tuple[str, ...]


class BroadWorkflowConfig(Protocol):
    broad_scan: BroadScanConfig
    discovery: DiscoveryConfig
    potential_scan: PotentialScanConfig


BROAD_MARKET_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Int64,
    "provider": pl.String,
    "coin_id": pl.String,
    "base_ccy": pl.String,
    "name": pl.String,
    "rank": pl.Int64,
    "price_usd": pl.Float64,
    "market_cap_usd": pl.Float64,
    "volume_24h_usd": pl.Float64,
    "volume_24h_change_pct": pl.Float64,
    "price_change_pct_1h": pl.Float64,
    "price_change_pct_24h": pl.Float64,
    "last_updated": pl.Int64,
    "trending_rank": pl.Int64,
    "trending_score": pl.Float64,
    "heat_source": pl.String,
}


BROAD_CANDIDATE_SCHEMA: dict[str, pl.DataType] = {
    "rank": pl.Int64,
    "timestamp": pl.Int64,
    "base_ccy": pl.String,
    "coin_id": pl.String,
    "name": pl.String,
    "okx_symbol": pl.String,
    "okx_mapped": pl.Boolean,
    "market_cap_usd": pl.Float64,
    "volume_24h_usd": pl.Float64,
    "price_change_pct_1h": pl.Float64,
    "price_change_pct_24h": pl.Float64,
    "trending_rank": pl.Int64,
    "trending_score": pl.Float64,
    "heat_source": pl.String,
    "tvl_usd": pl.Float64,
    "tvl_change_1d_pct": pl.Float64,
    "news_mentions": pl.Int64,
    "accumulation_source_score": pl.Float64,
    "oversold_source_score": pl.Float64,
    "accumulation_source_fired": pl.Boolean,
    "oversold_source_fired": pl.Boolean,
    "accumulation_source_reasons": pl.String,
    "oversold_source_reasons": pl.String,
    "discovery_sources": pl.String,
    "primary_source": pl.String,
    "source_base_score": pl.Float64,
    "source_reasons": pl.String,
    "broad_reasons": pl.String,
    "exclude_reason": pl.String,
}


def empty_broad_market_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=BROAD_MARKET_SCHEMA)


def empty_broad_protocol_frame() -> pl.DataFrame:
    return pl.DataFrame(schema={"base_ccy": pl.String})


def empty_broad_news_frame() -> pl.DataFrame:
    return pl.DataFrame(schema={"base_ccy": pl.String})


def empty_broad_candidate_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=BROAD_CANDIDATE_SCHEMA)


def empty_source_manifest_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=SOURCE_MANIFEST_SCHEMA)


@dataclass(frozen=True)
class BroadDiscoveryResult:
    market: pl.DataFrame
    protocols: pl.DataFrame
    news: pl.DataFrame
    candidates: pl.DataFrame
    manifest: pl.DataFrame


@dataclass(frozen=True)
class PotentialUniverseRequest:
    config: BroadWorkflowConfig
    broad_top_n: int
    board_pool_n: int


@dataclass(frozen=True)
class PotentialUniverseResult:
    broad_pool_symbols: tuple[str, ...]
    discovery: pl.DataFrame
    board_market: pl.DataFrame
    manifest: pl.DataFrame


def collect_broad_sources(config: BroadWorkflowConfig, *, broad_top_n: int) -> BroadDiscoveryResult:
    return asyncio.run(_collect_broad_sources_task(config, broad_top_n=broad_top_n))


def collect_potential_board_universe(
    config: BroadWorkflowConfig, *, top_n: int
) -> BroadDiscoveryResult:
    return asyncio.run(_collect_potential_board_universe_task(config, top_n=top_n))


def collect_potential_universe(request: PotentialUniverseRequest) -> PotentialUniverseResult:
    """Collect and map the potential-board universe without writing artifacts."""
    from qooi.exchange.discovery import discover_candidates

    board_universe = collect_potential_board_universe(request.config, top_n=request.broad_top_n)
    okx_result = discover_candidates(
        request.config,
        top_n=max(request.broad_top_n, request.board_pool_n, request.config.discovery.top_n),
        min_volume_usd=None,
    )
    board_market = build_okx_first_potential_universe(
        board_universe.market,
        okx_result.discovery,
        request.config.broad_scan,
        request.config.potential_scan.trend_excluded_base_ccy,
    ).head(request.broad_top_n)
    manifest = _concat_nonempty_frames(board_universe.manifest, okx_result.manifest)
    return PotentialUniverseResult(
        broad_pool_symbols=select_board_pool_symbols(board_market, top_n=request.broad_top_n),
        discovery=okx_result.discovery,
        board_market=board_market,
        manifest=manifest,
    )


def build_okx_first_potential_universe(
    market: pl.DataFrame,
    okx_discovery: pl.DataFrame,
    config: BroadScanConfig,
    excluded_base_ccy: tuple[str, ...] = (),
) -> pl.DataFrame:
    if okx_discovery.is_empty() or "base_ccy" not in okx_discovery.columns:
        return empty_broad_candidate_frame()
    market_by_base = _dedupe_market(market) if not market.is_empty() else empty_broad_market_frame()
    if not market_by_base.is_empty():
        market_by_base = market_by_base.select(
            [
                "base_ccy",
                "timestamp",
                "provider",
                "coin_id",
                "name",
                "market_cap_usd",
                "volume_24h_usd",
                "price_change_pct_1h",
                "price_change_pct_24h",
                "trending_rank",
                "trending_score",
                "heat_source",
            ]
        )
    excluded = {base.upper() for base in (*config.excluded_base_ccy, *excluded_base_ccy)}
    okx = okx_discovery.filter(pl.col("eligible")).select(
        [
            pl.col("symbol").alias("okx_symbol"),
            pl.col("base_ccy").cast(pl.String),
            pl.col("quote_volume_24h").cast(pl.Float64).alias("_okx_volume_24h_usd"),
            pl.col("rank_score").cast(pl.Float64).alias("_okx_rank_score"),
        ]
    )
    if okx.is_empty():
        return empty_broad_candidate_frame()
    joined = okx.join(market_by_base, on="base_ccy", how="left")
    has_market = pl.col("market_cap_usd").is_not_null() & pl.col("volume_24h_usd").is_not_null()
    volume = pl.coalesce([pl.col("volume_24h_usd"), pl.col("_okx_volume_24h_usd")])
    turnover = pl.when(pl.col("market_cap_usd") > 0.0).then(
        volume / pl.col("market_cap_usd")
    ).otherwise(None)
    base_excluded = pl.col("base_ccy").str.to_uppercase().is_in(sorted(excluded))
    volume_low = volume.fill_null(0.0) < config.min_volume_24h_usd
    turnover_low = turnover.fill_null(0.0) < config.min_turnover_ratio
    trending = pl.col("trending_rank").is_not_null()
    trending_top5 = pl.col("trending_rank") <= 5
    frame = joined.with_columns(
        [
            pl.coalesce([pl.col("timestamp"), pl.lit(0)]).cast(pl.Int64).alias("timestamp"),
            pl.coalesce([pl.col("coin_id"), pl.col("base_ccy").str.to_lowercase()]).alias(
                "coin_id"
            ),
            pl.coalesce([pl.col("name"), pl.col("base_ccy")]).alias("name"),
            volume.alias("volume_24h_usd"),
            pl.when(base_excluded)
            .then(pl.lit("base_excluded"))
            .when(~has_market)
            .then(pl.lit("market_data_missing"))
            .when(volume_low)
            .then(pl.lit("volume_below_min"))
            .when(turnover_low)
            .then(pl.lit("turnover_below_min"))
            .otherwise(pl.lit(""))
            .alias("exclude_reason"),
            _reason_expr(
                pl.col("price_change_pct_1h").abs().fill_null(0.0) >= 1.0,
                pl.col("price_change_pct_24h").abs().fill_null(0.0) >= 3.0,
                pl.lit(False),
                pl.lit(False),
                trending,
                trending_top5,
            ).alias("_attention_reasons"),
            pl.lit(True).alias("okx_mapped"),
            pl.lit(None).cast(pl.Float64).alias("tvl_usd"),
            pl.lit(None).cast(pl.Float64).alias("tvl_change_1d_pct"),
            pl.lit(None).cast(pl.Int64).alias("news_mentions"),
        ]
    )
    frame = evaluate_source_scores(frame, config)
    if "rank" in frame.columns:
        frame = frame.drop("rank")
    frame = frame.sort(
        ["exclude_reason", "source_base_score", "volume_24h_usd", "okx_symbol"],
        descending=[False, True, True, False],
        nulls_last=True,
    ).with_row_index("rank", offset=1)
    return _coerce_candidates(frame.select(BROAD_CANDIDATE_SCHEMA.keys()))


async def _collect_potential_board_universe_task(
    config: BroadWorkflowConfig, *, top_n: int
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
            fetch_coingecko_markets(
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
            calls.append(fetch_coingecko_trending(client, api_key=api_key))
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


async def _collect_broad_sources_task(
    config: BroadWorkflowConfig, *, broad_top_n: int
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
                fetch_coingecko_markets(
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
                coingecko_calls.append(fetch_coingecko_trending(client, api_key=api_key))
            market_results.extend(await asyncio.gather(*coingecko_calls))
    if "coinpaprika" in providers:
        async with httpx.AsyncClient(base_url=COINPAPRIKA_BASE_URL, timeout=20.0) as client:
            market_results.append(await fetch_coinpaprika_tickers(client))
    if "defillama" in providers:
        async with httpx.AsyncClient(base_url=DEFILLAMA_BASE_URL, timeout=30.0) as client:
            protocol_results.append(await fetch_defillama_protocols(client))
    if "cryptopanic" in providers:
        api_key = os.getenv(broad.cryptopanic.api_key_env, "")
        if api_key:
            async with httpx.AsyncClient(base_url=CRYPTOPANIC_BASE_URL, timeout=20.0) as client:
                news_results.append(
                    await fetch_cryptopanic_global_posts(
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
            ).alias("source_base_score"),
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
    frame = frame.sort("source_base_score", descending=True).with_row_index("rank", offset=1)
    return _coerce_candidates(frame)


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
    turnover = pl.when(pl.col("market_cap_usd") > 0.0).then(
        pl.col("volume_24h_usd") / pl.col("market_cap_usd")
    ).otherwise(None)
    turnover_low = turnover.fill_null(0.0) < config.min_turnover_ratio
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
            .when(turnover_low)
            .then(pl.lit("turnover_below_min"))
            .otherwise(pl.lit(""))
            .alias("exclude_reason"),
            _reason_expr(
                pl.col("price_change_pct_1h").abs().fill_null(0.0) >= 1.0,
                pl.col("price_change_pct_24h").abs().fill_null(0.0) >= 3.0,
                pl.lit(False),
                pl.lit(False),
                trending,
                trending_top5,
            ).alias("_attention_reasons"),
            pl.lit(None).cast(pl.Float64).alias("tvl_usd"),
            pl.lit(None).cast(pl.Float64).alias("tvl_change_1d_pct"),
            pl.lit(None).cast(pl.Int64).alias("news_mentions"),
            pl.lit(None).cast(pl.String).alias("okx_symbol"),
            pl.lit(False).alias("okx_mapped"),
        ]
    )
    frame = evaluate_source_scores(frame, config)
    if "rank" in frame.columns:
        frame = frame.drop("rank")
    frame = frame.sort(
        ["exclude_reason", "source_base_score", "volume_24h_usd", "base_ccy"],
        descending=[False, True, True, False],
        nulls_last=True,
    ).with_row_index("rank", offset=1)
    return _coerce_candidates(frame.select(BROAD_CANDIDATE_SCHEMA.keys()))


def evaluate_source_scores(frame: pl.DataFrame, config: BroadScanConfig) -> pl.DataFrame:
    frame = _clean_source_score_input(frame)
    for evaluator in SOURCE_SCORE_EVALUATORS:
        frame = evaluator(frame, config)
    return _finalize_discovery_sources(frame)


def _clean_source_score_input(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    schema = {
        "market_cap_usd": pl.Float64,
        "volume_24h_usd": pl.Float64,
        "price_change_pct_1h": pl.Float64,
        "price_change_pct_24h": pl.Float64,
        "trending_rank": pl.Int64,
        "trending_score": pl.Float64,
        "_attention_reasons": pl.String,
        "exclude_reason": pl.String,
    }
    for col, dtype in schema.items():
        if col not in frame.columns:
            frame = frame.with_columns(pl.lit(None).cast(dtype).alias(col))
            continue
        frame = frame.with_columns(pl.col(col).cast(dtype, strict=False).alias(col))
    return frame.with_columns(
        [
            pl.col("_attention_reasons").fill_null("").alias("_attention_reasons"),
            pl.col("exclude_reason").fill_null("").alias("exclude_reason"),
        ]
    )


def _evaluate_accumulation_source(frame: pl.DataFrame, config: BroadScanConfig) -> pl.DataFrame:
    turnover = pl.when(pl.col("market_cap_usd") > 0.0).then(
        pl.col("volume_24h_usd") / pl.col("market_cap_usd")
    ).otherwise(None)
    active_1h = pl.col("price_change_pct_1h").abs().fill_null(0.0) >= 1.0
    active_24h = pl.col("price_change_pct_24h").abs().fill_null(0.0) >= 3.0
    turnover_ok = turnover.fill_null(0.0) >= config.min_turnover_ratio
    liquid = pl.col("volume_24h_usd").fill_null(0.0) >= config.min_volume_24h_usd
    score = (
        turnover.fill_null(0.0) * 100.0
        + pl.col("volume_24h_usd").fill_null(0.0).clip(1.0).log10() * 0.25
        + pl.when(active_1h).then(2.0).otherwise(0.0)
        + pl.when(active_24h).then(1.0).otherwise(0.0)
    )
    fired = (pl.col("exclude_reason") == "") & (score > 0.0)
    reasons = pl.concat_str(
        [
            pl.when(turnover_ok).then(pl.lit("turnover_ok;")).otherwise(pl.lit("")),
            pl.when(liquid).then(pl.lit("liquid;")).otherwise(pl.lit("")),
            pl.when(active_1h).then(pl.lit("active_1h;")).otherwise(pl.lit("")),
            pl.when(active_24h).then(pl.lit("active_24h;")).otherwise(pl.lit("")),
        ]
    ).str.strip_chars(";")
    return frame.with_columns(
        [
            pl.when(fired).then(score).otherwise(0.0).alias("accumulation_source_score"),
            fired.alias("accumulation_source_fired"),
            pl.when(fired).then(reasons).otherwise(pl.lit("")).alias(
                "accumulation_source_reasons"
            ),
        ]
    )


def _evaluate_oversold_source(frame: pl.DataFrame, config: BroadScanConfig) -> pl.DataFrame:
    turnover = pl.when(pl.col("market_cap_usd") > 0.0).then(
        pl.col("volume_24h_usd") / pl.col("market_cap_usd")
    ).otherwise(None)
    market_gate = (
        (pl.col("market_cap_usd") >= config.oversold_min_market_cap_usd)
        & (pl.col("market_cap_usd") <= config.max_market_cap_usd)
        & (pl.col("volume_24h_usd") >= config.min_volume_24h_usd)
        & (turnover.fill_null(0.0) >= config.min_turnover_ratio)
        & (pl.col("exclude_reason") != "stablecoin_excluded")
    )
    oversold_24h = pl.col("price_change_pct_24h") <= config.oversold_return_24h_max
    pass_gate = (
        pl.lit(config.oversold_enabled)
        & market_gate.fill_null(False)
        & oversold_24h.fill_null(False)
    )
    selloff = pl.col("price_change_pct_24h").abs().fill_null(0.0).clip(0.0, 40.0) * 1.5
    liquidity = pl.col("volume_24h_usd").fill_null(0.0).clip(1.0).log10().clip(0.0, 8.0)
    turnover_score = (turnover.fill_null(0.0) * 100.0).clip(0.0, 20.0)
    reasons = pl.concat_str(
        [
            pl.when(oversold_24h).then(pl.lit("oversold_24h;")).otherwise(pl.lit("")),
            pl.when(turnover.fill_null(0.0) >= config.min_turnover_ratio)
            .then(pl.lit("turnover_ok;"))
            .otherwise(pl.lit("")),
            pl.when(pl.col("volume_24h_usd") >= config.min_volume_24h_usd)
            .then(pl.lit("liquid;"))
            .otherwise(pl.lit("")),
        ]
    ).str.strip_chars(";")
    return frame.with_columns(
        [
            pl.when(pass_gate).then(selloff + liquidity + turnover_score).otherwise(0.0).alias(
                "oversold_source_score"
            ),
            pass_gate.alias("oversold_source_fired"),
            pl.when(pass_gate).then(reasons).otherwise(pl.lit("")).alias(
                "oversold_source_reasons"
            ),
        ]
    )


def _finalize_discovery_sources(frame: pl.DataFrame) -> pl.DataFrame:
    oversold_wins = pl.col("oversold_source_score") > pl.col("accumulation_source_score")
    sources = pl.concat_str(
        [
            pl.when(pl.col("accumulation_source_fired"))
            .then(pl.lit("accumulation;"))
            .otherwise(pl.lit("")),
            pl.when(pl.col("oversold_source_fired"))
            .then(pl.lit("oversold;"))
            .otherwise(pl.lit("")),
        ]
    ).str.strip_chars(";")
    source_reasons = pl.concat_str(
        [
            pl.when(pl.col("accumulation_source_reasons") != "")
            .then(
                pl.concat_str(
                    [pl.lit("accumulation:"), pl.col("accumulation_source_reasons"), pl.lit(";")]
                )
            )
            .otherwise(pl.lit("")),
            pl.when(pl.col("oversold_source_reasons") != "")
            .then(
                pl.concat_str(
                    [pl.lit("oversold:"), pl.col("oversold_source_reasons"), pl.lit(";")]
                )
            )
            .otherwise(pl.lit("")),
        ]
    ).str.strip_chars(";")
    broad_reasons = pl.concat_str(
        [
            pl.col("_attention_reasons").fill_null(""),
            pl.when(source_reasons != "")
            .then(pl.concat_str([pl.lit(";"), source_reasons]))
            .otherwise(pl.lit("")),
        ]
    ).str.strip_chars(";")
    return frame.with_columns(
        [
            sources.alias("discovery_sources"),
            pl.when(oversold_wins)
            .then(pl.lit("oversold"))
            .when(pl.col("accumulation_source_fired"))
            .then(pl.lit("accumulation"))
            .when(pl.col("oversold_source_fired"))
            .then(pl.lit("oversold"))
            .otherwise(pl.lit(""))
            .alias("primary_source"),
            pl.max_horizontal("accumulation_source_score", "oversold_source_score").alias(
                "source_base_score"
            ),
            source_reasons.alias("source_reasons"),
            broad_reasons.alias("broad_reasons"),
        ]
    )


SOURCE_SCORE_EVALUATORS = (
    _evaluate_accumulation_source,
    _evaluate_oversold_source,
)


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
        frame.sort("source_base_score", descending=True).head(deep_top_n)["okx_symbol"].to_list()
    )


def select_board_pool_symbols(broad_candidates: pl.DataFrame, *, top_n: int) -> tuple[str, ...]:
    if broad_candidates.is_empty():
        return ()
    frame = broad_candidates.filter(pl.col("okx_mapped") & (pl.col("exclude_reason") == ""))
    if frame.is_empty():
        return ()
    return tuple(
        frame.sort(
            ["source_base_score", "volume_24h_usd", "okx_symbol"],
            descending=[True, True, False],
            nulls_last=True,
        )
        .head(top_n)["okx_symbol"]
        .to_list()
    )


def _concat_nonempty_frames(*frames: pl.DataFrame) -> pl.DataFrame:
    nonempty = [frame for frame in frames if not frame.is_empty()]
    return pl.concat(nonempty, how="vertical") if nonempty else empty_source_manifest_frame()


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


