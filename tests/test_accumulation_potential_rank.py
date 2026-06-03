from __future__ import annotations

import polars as pl

from qooi.accumulation.config import PotentialScanConfig
from qooi.strategies.potential import rank_potential_candidates, render_potential_report


def test_potential_rank_prioritizes_base_and_expansion_without_optional_evidence() -> None:
    features = pl.DataFrame(
        [
            _feature("BASE-USDT-SWAP", 50, volume_spike_ratio_1h_20h=1.0),
            _feature(
                "FIRST-USDT-SWAP",
                50,
                volume_spike_ratio_1h_20h=4.0,
                first_volume_expansion=True,
                return_1h=0.02,
            ),
            _feature(
                "LATE-USDT-SWAP",
                50,
                price_to_90d_low=1.8,
                bb_width_percentile_90d=0.8,
                range_position_90d_pct=0.9,
                price_vs_vwap_24h_pct=-0.02,
            ),
        ]
    )
    scores = pl.DataFrame(
        {
            "symbol": ["BASE-USDT-SWAP", "FIRST-USDT-SWAP", "LATE-USDT-SWAP"],
            "timestamp": [50, 50, 50],
            "score_total": [-25, 0, 10],
            "alert_level": ["none", "none", "none"],
            "missing_evidence": [
                "onchain_missing;whale_missing;messages_missing",
                "onchain_missing;messages_missing",
                "",
            ],
        }
    )

    ranked = rank_potential_candidates(
        features,
        strict_scores=scores,
        broad_candidates=_broad_context(
            ["BASE-USDT-SWAP", "FIRST-USDT-SWAP", "LATE-USDT-SWAP"]
        ),
        config=PotentialScanConfig(min_history_hours=10),
    )
    rows = ranked.to_dicts()

    assert rows[0]["symbol"] in {"BASE-USDT-SWAP", "FIRST-USDT-SWAP"}
    assert ranked.filter(pl.col("symbol") == "BASE-USDT-SWAP")["action_state"][0] in {
        "review_now",
        "watch_base",
    }
    assert (
        "onchain_missing"
        in ranked.filter(pl.col("symbol") == "BASE-USDT-SWAP")["missing_evidence"][0]
    )
    assert ranked.filter(pl.col("symbol") == "LATE-USDT-SWAP").is_empty()
    assert ranked.filter(pl.col("symbol") == "BASE-USDT-SWAP")["strict_score_total"][0] == -25


def test_potential_report_uses_altcoin_trend_readout() -> None:
    ranked = rank_potential_candidates(
        pl.DataFrame([_feature("FIRST-USDT-SWAP", 50, volume_spike_ratio_1h_20h=4.0)]),
        broad_candidates=_broad_context(["FIRST-USDT-SWAP"]),
        config=PotentialScanConfig(min_history_hours=10),
    )

    report = render_potential_report(ranked)

    assert "# Altcoin Potential Trend Report" in report
    assert "A Review: Prepared Base + Funds Building/Confirmed" in report
    assert "tier=" in report
    assert "funds=" in report
    assert "gate=" in report
    assert "need=" in report
    assert "old filter" not in report.lower()


def test_potential_rank_excludes_large_cap_even_when_near_low() -> None:
    ranked = rank_potential_candidates(
        pl.DataFrame([_feature("ETH-USDT-SWAP", 50)]),
        broad_candidates=_broad_context(["ETH-USDT-SWAP"], market_cap_usd=250_000_000_000.0),
        config=PotentialScanConfig(min_history_hours=10),
    )

    assert ranked.is_empty()


def test_potential_rank_requires_near_low_or_compression() -> None:
    features = pl.DataFrame(
        [
            _feature(
                "EXTENDED-USDT-SWAP",
                50,
                price_to_90d_low=1.6,
                bb_width_percentile_90d=0.7,
            ),
            _feature("COMPRESSED-USDT-SWAP", 50, price_to_90d_low=1.6, bb_width_percentile_90d=0.2),
        ]
    )

    ranked = rank_potential_candidates(
        features,
        broad_candidates=_broad_context(["EXTENDED-USDT-SWAP", "COMPRESSED-USDT-SWAP"]),
        config=PotentialScanConfig(min_history_hours=10),
    )

    assert ranked["symbol"].to_list() == ["COMPRESSED-USDT-SWAP"]


def test_potential_rank_funds_missing_stays_missing_confirmation() -> None:
    ranked = rank_potential_candidates(
        pl.DataFrame(
            [
                _feature(
                    "MISS-USDT-SWAP",
                    50,
                    taker_buy_ratio=None,
                    open_interest_usd_change_24h=None,
                    depth_imbalance_25_mean=None,
                    large_trade_buy_ratio=None,
                )
            ]
        ),
        broad_candidates=_broad_context(["MISS-USDT-SWAP"]),
        config=PotentialScanConfig(min_history_hours=10),
    )
    row = ranked.to_dicts()[0]

    assert row["funds_state"] == "funds_missing"
    assert row["board_priority"] == "B_watch"
    assert "need_onchain_flow" in row["next_confirmation_needed"]
    assert "need_book_support" in row["next_confirmation_needed"]


def test_potential_rank_funds_building_promotes_review_tier() -> None:
    ranked = rank_potential_candidates(
        pl.DataFrame(
            [
                _feature(
                    "BUILD-USDT-SWAP",
                    50,
                    flow_zscore=-4.0,
                    net_exchange_flow=-100.0,
                    depth_imbalance_25_mean=0.35,
                    large_trade_buy_ratio=None,
                    mention_growth=None,
                )
            ]
        ),
        broad_candidates=_broad_context(["BUILD-USDT-SWAP"]),
        config=PotentialScanConfig(min_history_hours=10),
    )
    row = ranked.to_dicts()[0]

    assert row["funds_state"] in {"funds_building", "funds_confirmed"}
    assert row["board_priority"] == "A_review"
    assert "onchain_outflow" in row["funds_positive_reasons"]
    assert "book_support" in row["funds_positive_reasons"]


def test_potential_rank_funds_confirmed_requires_breadth() -> None:
    ranked = rank_potential_candidates(
        pl.DataFrame(
            [
                _feature(
                    "FLOW-USDT-SWAP",
                    50,
                    flow_zscore=-4.0,
                    net_exchange_flow=-100.0,
                    taker_buy_ratio=None,
                    open_interest_usd_change_24h=None,
                    depth_imbalance_25_mean=None,
                    large_trade_buy_ratio=None,
                    mention_growth=None,
                )
            ]
        ),
        broad_candidates=_broad_context(["FLOW-USDT-SWAP"]),
        config=PotentialScanConfig(min_history_hours=10),
    )

    assert ranked["funds_state"][0] == "funds_building"


def test_potential_rank_okx_native_funds_can_build_review_tier() -> None:
    ranked = rank_potential_candidates(
        pl.DataFrame(
            [
                _feature(
                    "OKX-USDT-SWAP",
                    50,
                    taker_buy_ratio=0.62,
                    open_interest_usd_change_24h=0.03,
                    depth_imbalance_25_mean=0.08,
                    large_trade_buy_ratio=None,
                )
            ]
        ),
        broad_candidates=_broad_context(["OKX-USDT-SWAP"]),
        config=PotentialScanConfig(min_history_hours=10),
    )
    row = ranked.to_dicts()[0]

    assert row["funds_state"] == "funds_building"
    assert row["board_priority"] == "A_review"
    assert "taker_buying" in row["funds_positive_reasons"]
    assert "oi_expanding" in row["funds_positive_reasons"]


def test_potential_rank_cooldown_downtrend_stays_context() -> None:
    ranked = rank_potential_candidates(
        pl.DataFrame(
            [
                _feature(
                    "COOL-USDT-SWAP",
                    50,
                    base_duration_hours=12,
                    new_low_count_30d=1,
                    ma_30d_slope_14d=-0.08,
                    return_60d=-0.45,
                    return_90d=-0.55,
                    price_vs_vwap_24h_pct=-0.02,
                    price_vs_ma_7d_pct=-0.04,
                    reclaim_state="below_ma",
                    taker_buy_ratio=0.65,
                    open_interest_usd_change_24h=0.03,
                )
            ]
        ),
        broad_candidates=_broad_context(["COOL-USDT-SWAP"]),
        config=PotentialScanConfig(min_history_hours=10),
    )
    row = ranked.to_dicts()[0]

    assert row["stage"] == "cooldown_downtrend"
    assert row["board_priority"] == "C_context"


def test_potential_rank_falling_knife_stays_avoid() -> None:
    ranked = rank_potential_candidates(
        pl.DataFrame(
            [
                _feature(
                    "KNIFE-USDT-SWAP",
                    50,
                    base_duration_hours=0,
                    new_low_count_30d=8,
                    ma_30d_slope_14d=-0.08,
                    price_vs_vwap_24h_pct=-0.02,
                    reclaim_state="below_ma",
                    taker_buy_ratio=0.65,
                    open_interest_usd_change_24h=0.03,
                )
            ]
        ),
        broad_candidates=_broad_context(["KNIFE-USDT-SWAP"]),
        config=PotentialScanConfig(min_history_hours=10),
    )
    row = ranked.to_dicts()[0]

    assert row["stage"] == "falling_knife"
    assert row["board_priority"] == "D_avoid"


def test_potential_rank_funds_rejected_for_inflow_or_message_overheat() -> None:
    ranked = rank_potential_candidates(
        pl.DataFrame(
            [
                _feature(
                    "HOT-USDT-SWAP",
                    50,
                    flow_zscore=4.0,
                    net_exchange_flow=100.0,
                    mention_growth=4.0,
                    emotion_news_ratio=0.8,
                    fundamental_news_ratio=0.1,
                )
            ]
        ),
        broad_candidates=_broad_context(["HOT-USDT-SWAP"]),
        config=PotentialScanConfig(min_history_hours=10),
    )
    row = ranked.to_dicts()[0]

    assert row["funds_state"] == "funds_rejected"
    assert row["board_priority"] == "D_avoid"
    assert "exchange_inflow" in row["funds_negative_reasons"]
    assert "message_overheat" in row["funds_negative_reasons"]


def test_potential_rank_setup_pass_reason_variants() -> None:
    ranked = rank_potential_candidates(
        pl.DataFrame(
            [
                _feature("NEAR-USDT-SWAP", 50, price_to_90d_low=1.1, bb_width_percentile_90d=0.5),
                _feature("COMP-USDT-SWAP", 50, price_to_90d_low=1.5, bb_width_percentile_90d=0.1),
                _feature("BOTH-USDT-SWAP", 50, price_to_90d_low=1.1, bb_width_percentile_90d=0.1),
            ]
        ),
        broad_candidates=_broad_context(["NEAR-USDT-SWAP", "COMP-USDT-SWAP", "BOTH-USDT-SWAP"]),
        config=PotentialScanConfig(min_history_hours=10),
    )
    reasons = {row["symbol"]: row["setup_pass_reason"] for row in ranked.to_dicts()}

    assert reasons["NEAR-USDT-SWAP"] == "near_low"
    assert reasons["COMP-USDT-SWAP"] == "compressed"
    assert reasons["BOTH-USDT-SWAP"] == "near_low_and_compressed"


def _feature(symbol: str, timestamp: int, **overrides):
    row = {
        "symbol": symbol,
        "timestamp": timestamp,
        "close": 10.0,
        "quote_volume_24h": 700_000.0,
        "history_hours": 100,
        "price_to_90d_low": 1.05,
        "price_to_30d_high": 1.5,
        "range_position_90d_pct": 0.2,
        "range_width_90d_pct": 0.2,
        "bb_width_20d_pct": 0.03,
        "bb_width_percentile_90d": 0.1,
        "volume_contraction_10d_90d": 0.4,
        "volume_spike_ratio_1h_20h": 1.0,
        "prior_spike_count_5d": 0,
        "first_volume_expansion": False,
        "return_1h": 0.0,
        "return_24h": 0.0,
        "return_72h": 0.0,
        "return_30d": -0.05,
        "return_60d": -0.10,
        "return_90d": -0.15,
        "drawdown_30d_pct": -0.1,
        "new_low_15d": False,
        "new_low_count_30d": 0,
        "higher_low_count_30d": 2,
        "base_duration_hours": 240,
        "price_vs_ma_7d_pct": 0.01,
        "price_vs_ma_30d_pct": -0.01,
        "ma_7d_slope_7d": 0.01,
        "ma_30d_slope_14d": -0.01,
        "downtrend_deceleration": True,
        "reclaim_state": "ma7_reclaim",
        "structure_block_reason": "",
        "vwap_24h": 10.0,
        "price_vs_vwap_24h_pct": 0.02,
        "vwap_slope_24h": 0.01,
        "taker_buy_ratio": 0.66,
        "taker_volume_imbalance": 0.2,
        "open_interest_usd_change_24h": 0.02,
        "net_exchange_flow": None,
        "flow_zscore": None,
        "whale_accumulation_ratio": None,
        "depth_imbalance_25_mean": 0.3,
        "large_trade_buy_ratio": 0.7,
        "mention_growth": None,
        "fundamental_news_ratio": None,
        "emotion_news_ratio": None,
        "source_coverage_score": 0.9,
        "data_quality_warning": "",
    }
    row.update(overrides)
    return row


def _broad_context(
    symbols: list[str],
    *,
    market_cap_usd: float = 100_000_000.0,
    volume_24h_usd: float = 1_000_000.0,
) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "rank": index,
                "okx_symbol": symbol,
                "market_cap_usd": market_cap_usd,
                "volume_24h_usd": volume_24h_usd,
                "broad_score": 10.0,
                "broad_reasons": "coingecko_trending",
            }
            for index, symbol in enumerate(symbols, start=1)
        ]
    )

