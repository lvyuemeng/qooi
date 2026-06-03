from __future__ import annotations

import polars as pl

from qooi.accumulation.config import BroadScanConfig
from qooi.exchange.universe import (
    map_broad_to_okx,
    rank_broad_candidates,
    rank_potential_board_universe,
    select_deep_symbols,
)
from qooi.sources.coingecko import normalize_coingecko_markets, normalize_coingecko_trending
from qooi.sources.defillama import normalize_defillama_protocols


def test_coingecko_markets_normalizer_maps_broad_market_rows() -> None:
    frame = normalize_coingecko_markets(
        [
            {
                "id": "dogecoin",
                "symbol": "doge",
                "name": "Dogecoin",
                "market_cap_rank": 9,
                "current_price": 0.22,
                "market_cap": 30_000_000_000,
                "total_volume": 4_000_000_000,
                "price_change_percentage_1h_in_currency": 1.2,
                "price_change_percentage_24h_in_currency": 5.5,
                "last_updated": "2026-06-02T00:00:00.000Z",
            }
        ]
    )

    assert frame.to_dicts()[0] == {
        "timestamp": frame["timestamp"][0],
        "provider": "coingecko",
        "coin_id": "dogecoin",
        "base_ccy": "DOGE",
        "name": "Dogecoin",
        "rank": 9,
        "price_usd": 0.22,
        "market_cap_usd": 30_000_000_000.0,
        "volume_24h_usd": 4_000_000_000.0,
        "volume_24h_change_pct": None,
        "price_change_pct_1h": 1.2,
        "price_change_pct_24h": 5.5,
        "last_updated": 1780358400000,
        "trending_rank": None,
        "trending_score": None,
        "heat_source": "",
    }


def test_coingecko_trending_normalizer_maps_heat_rows() -> None:
    frame = normalize_coingecko_trending(
        {
            "coins": [
                {
                    "item": {
                        "id": "dogecoin",
                        "symbol": "doge",
                        "name": "Dogecoin",
                        "market_cap_rank": 9,
                        "score": 12.5,
                        "data": {
                            "market_cap": "$30,000,000,000",
                            "total_volume": {"usd": 4_000_000_000},
                            "price_change_percentage_24h": {"usd": 5.5},
                        },
                    }
                }
            ]
        }
    )

    row = frame.to_dicts()[0]
    assert row["provider"] == "coingecko_trending"
    assert row["base_ccy"] == "DOGE"
    assert row["market_cap_usd"] == 30_000_000_000.0
    assert row["volume_24h_usd"] == 4_000_000_000.0
    assert row["price_change_pct_24h"] == 5.5
    assert row["trending_rank"] == 1
    assert row["trending_score"] == 12.5
    assert row["heat_source"] == "coingecko_trending"


def test_defillama_protocols_normalizer_maps_protocol_rows() -> None:
    frame = normalize_defillama_protocols(
        [
            {
                "slug": "aave",
                "symbol": "AAVE",
                "name": "Aave",
                "category": "Lending",
                "chains": ["Ethereum", "Base"],
                "tvl": 1_000_000_000,
                "change_1d": 21.0,
                "change_7d": 8.0,
            }
        ]
    )

    assert frame.select("provider", "protocol", "base_ccy", "chains").to_dicts() == [
        {
            "provider": "defillama",
            "protocol": "aave",
            "base_ccy": "AAVE",
            "chains": "Ethereum,Base",
        }
    ]


def test_broad_candidate_ranking_gates_and_boosts_without_alert_score() -> None:
    market = pl.DataFrame(
        [
            {
                "timestamp": 1,
                "provider": "coingecko",
                "coin_id": "tether",
                "base_ccy": "USDT",
                "name": "Tether",
                "rank": 1,
                "price_usd": 1.0,
                "market_cap_usd": 100_000_000_000.0,
                "volume_24h_usd": 80_000_000_000.0,
                "volume_24h_change_pct": None,
                "price_change_pct_1h": 0.0,
                "price_change_pct_24h": 0.0,
                "last_updated": 1,
            },
            {
                "timestamp": 1,
                "provider": "coingecko",
                "coin_id": "aave",
                "base_ccy": "AAVE",
                "name": "Aave",
                "rank": 50,
                "price_usd": 100.0,
                "market_cap_usd": 150_000_000.0,
                "volume_24h_usd": 50_000_000.0,
                "volume_24h_change_pct": None,
                "price_change_pct_1h": 1.5,
                "price_change_pct_24h": 4.0,
                "last_updated": 1,
            },
        ]
    )
    protocols = pl.DataFrame(
        [
            {
                "timestamp": 1,
                "provider": "defillama",
                "protocol": "aave",
                "base_ccy": "AAVE",
                "name": "Aave",
                "category": "Lending",
                "chains": "Ethereum",
                "tvl_usd": 1_000_000_000.0,
                "tvl_change_1d_pct": 25.0,
                "tvl_change_7d_pct": 10.0,
            }
        ]
    )
    news = pl.DataFrame(
        [
            {
                "timestamp": 1,
                "provider": "cryptopanic",
                "source_id": "1",
                "title": "Aave news",
                "url": "https://example.test",
                "base_ccy": "AAVE",
                "sentiment": "news",
            }
        ]
    )

    ranked = rank_broad_candidates(market, protocols, news, BroadScanConfig())
    rows = {row["base_ccy"]: row for row in ranked.to_dicts()}

    assert rows["USDT"]["exclude_reason"] == "stablecoin_excluded"
    assert rows["AAVE"]["exclude_reason"] == ""
    assert rows["AAVE"]["news_mentions"] == 1
    assert "tvl_growth" in rows["AAVE"]["broad_reasons"]
    assert "score_total" not in ranked.columns


def test_broad_candidate_ranking_preserves_trending_evidence_after_dedupe() -> None:
    market = pl.DataFrame(
        [
            {
                "timestamp": 1,
                "provider": "coingecko",
                "coin_id": "dogecoin",
                "base_ccy": "DOGE",
                "name": "Dogecoin",
                "rank": 9,
                "price_usd": 0.22,
                "market_cap_usd": 300_000_000.0,
                "volume_24h_usd": 4_000_000_000.0,
                "volume_24h_change_pct": None,
                "price_change_pct_1h": 0.1,
                "price_change_pct_24h": 0.5,
                "last_updated": 1,
                "trending_rank": None,
                "trending_score": None,
                "heat_source": "",
            },
            {
                "timestamp": 2,
                "provider": "coingecko_trending",
                "coin_id": "dogecoin",
                "base_ccy": "DOGE",
                "name": "Dogecoin",
                "rank": 9,
                "price_usd": None,
                "market_cap_usd": None,
                "volume_24h_usd": None,
                "volume_24h_change_pct": None,
                "price_change_pct_1h": None,
                "price_change_pct_24h": None,
                "last_updated": None,
                "trending_rank": 3,
                "trending_score": 10.0,
                "heat_source": "coingecko_trending",
            },
        ]
    )

    ranked = rank_broad_candidates(market, pl.DataFrame(), pl.DataFrame(), BroadScanConfig())
    row = ranked.to_dicts()[0]

    assert row["trending_rank"] == 3
    assert row["trending_score"] == 10.0
    assert row["heat_source"] == "coingecko_trending"
    assert "coingecko_trending" in row["broad_reasons"]
    assert "trending_top5" in row["broad_reasons"]
    assert row["exclude_reason"] == ""


def test_broad_candidate_ranking_excludes_trending_large_caps() -> None:
    market = pl.DataFrame(
        [
            {
                "timestamp": 1,
                "provider": "coingecko_trending",
                "coin_id": "ethereum",
                "base_ccy": "ETH",
                "name": "Ethereum",
                "rank": 2,
                "price_usd": None,
                "market_cap_usd": 250_000_000_000.0,
                "volume_24h_usd": 10_000_000_000.0,
                "volume_24h_change_pct": None,
                "price_change_pct_1h": 2.0,
                "price_change_pct_24h": 4.0,
                "last_updated": None,
                "trending_rank": 1,
                "trending_score": 20.0,
                "heat_source": "coingecko_trending",
            }
        ]
    )

    row = rank_broad_candidates(
        market, pl.DataFrame(), pl.DataFrame(), BroadScanConfig()
    ).to_dicts()[0]

    assert row["exclude_reason"] == "market_cap_above_max"
    assert "coingecko_trending" in row["broad_reasons"]


def test_potential_board_universe_keeps_trending_as_annotation_only() -> None:
    market = pl.DataFrame(
        [
            {
                "timestamp": 1,
                "provider": "coingecko_trending",
                "coin_id": "hypecoin",
                "base_ccy": "HYPE",
                "name": "Hypecoin",
                "rank": 100,
                "price_usd": None,
                "market_cap_usd": None,
                "volume_24h_usd": None,
                "volume_24h_change_pct": None,
                "price_change_pct_1h": None,
                "price_change_pct_24h": None,
                "last_updated": None,
                "trending_rank": 1,
                "trending_score": 20.0,
                "heat_source": "coingecko_trending",
            },
            {
                "timestamp": 2,
                "provider": "coingecko",
                "coin_id": "basecoin",
                "base_ccy": "BASE",
                "name": "Basecoin",
                "rank": 120,
                "price_usd": 1.0,
                "market_cap_usd": 200_000_000.0,
                "volume_24h_usd": 20_000_000.0,
                "volume_24h_change_pct": None,
                "price_change_pct_1h": 0.5,
                "price_change_pct_24h": 2.0,
                "last_updated": 2,
                "trending_rank": None,
                "trending_score": None,
                "heat_source": "",
            },
        ]
    )

    rows = {
        row["base_ccy"]: row
        for row in rank_potential_board_universe(market, BroadScanConfig()).to_dicts()
    }

    assert rows["HYPE"]["exclude_reason"] == "market_data_missing"
    assert "coingecko_trending" in rows["HYPE"]["broad_reasons"]
    assert rows["BASE"]["exclude_reason"] == ""


def test_broad_to_okx_mapping_selects_only_tradable_swaps() -> None:
    candidates = pl.DataFrame(
        [
            {
                "rank": 1,
                "timestamp": 1,
                "base_ccy": "AAVE",
                "coin_id": "aave",
                "name": "Aave",
                "okx_symbol": None,
                "okx_mapped": False,
                "market_cap_usd": 1_000_000.0,
                "volume_24h_usd": 1_000_000.0,
                "price_change_pct_1h": 1.0,
                "price_change_pct_24h": 2.0,
                "tvl_usd": None,
                "tvl_change_1d_pct": None,
                "news_mentions": 0,
                "broad_score": 10.0,
                "broad_reasons": "active_1h",
                "exclude_reason": "",
            },
            {
                "rank": 2,
                "timestamp": 1,
                "base_ccy": "NOTLISTED",
                "coin_id": "notlisted",
                "name": "Not Listed",
                "okx_symbol": None,
                "okx_mapped": False,
                "market_cap_usd": 1_000_000.0,
                "volume_24h_usd": 1_000_000.0,
                "price_change_pct_1h": 1.0,
                "price_change_pct_24h": 2.0,
                "tvl_usd": None,
                "tvl_change_1d_pct": None,
                "news_mentions": 0,
                "broad_score": 9.0,
                "broad_reasons": "active_1h",
                "exclude_reason": "",
            },
        ]
    )
    okx = pl.DataFrame(
        [
            {
                "base_ccy": "AAVE",
                "symbol": "AAVE-USDT-SWAP",
                "eligible": True,
                "rank_score": 10.0,
            }
        ]
    )

    mapped = map_broad_to_okx(candidates, okx)
    rows = {row["base_ccy"]: row for row in mapped.to_dicts()}

    assert rows["AAVE"]["okx_symbol"] == "AAVE-USDT-SWAP"
    assert rows["AAVE"]["okx_mapped"] is True
    assert rows["NOTLISTED"]["exclude_reason"] == "okx_swap_not_listed"
    assert select_deep_symbols(mapped, deep_top_n=10) == ("AAVE-USDT-SWAP",)


def test_broad_to_okx_mapping_reports_listed_but_ineligible_swap() -> None:
    candidates = pl.DataFrame(
        [
            {
                "rank": 1,
                "timestamp": 1,
                "base_ccy": "BTC",
                "coin_id": "bitcoin",
                "name": "Bitcoin",
                "okx_symbol": None,
                "okx_mapped": False,
                "market_cap_usd": 1_000_000.0,
                "volume_24h_usd": 1_000_000.0,
                "price_change_pct_1h": 1.0,
                "price_change_pct_24h": 2.0,
                "tvl_usd": None,
                "tvl_change_1d_pct": None,
                "news_mentions": 0,
                "broad_score": 10.0,
                "broad_reasons": "active_1h",
                "exclude_reason": "",
            }
        ]
    )
    okx = pl.DataFrame(
        [
            {
                "base_ccy": "BTC",
                "symbol": "BTC-USDT-SWAP",
                "eligible": False,
                "exclude_reason": "volume_below_min",
                "rank_score": 10.0,
            }
        ]
    )

    mapped = map_broad_to_okx(candidates, okx)
    row = mapped.to_dicts()[0]

    assert row["okx_symbol"] == "BTC-USDT-SWAP"
    assert row["okx_mapped"] is False
    assert row["exclude_reason"] == "okx_swap_ineligible_volume_below_min"
    assert select_deep_symbols(mapped, deep_top_n=10) == ()

