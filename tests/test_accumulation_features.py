from __future__ import annotations

import polars as pl
import pytest

from qooi.accumulation.features import (
    HOUR_MS,
    compute_depth_features,
    compute_flow_features,
    compute_funds_flow_features,
    compute_funds_message_features,
    compute_funds_orderbook_features,
    compute_funds_volume_features,
    compute_long_short_ratio_features,
    compute_open_interest_features,
    compute_realtime_confirmation_features,
    compute_realtime_orderbook_features,
    compute_realtime_trade_features,
    compute_structure_features,
    compute_taker_volume_features,
    join_hourly_accumulation_features,
)


def test_flow_zscore_sign_semantics_and_negative_streak() -> None:
    frame = pl.DataFrame(
        {
            "timestamp": [idx * HOUR_MS for idx in range(6)],
            "inflow": [10.0, 10.0, 10.0, 10.0, 0.0, 0.0],
            "outflow": [0.0, 0.0, 0.0, 0.0, 100.0, 100.0],
        }
    )

    out = compute_flow_features(frame, window_hours=4)

    assert out["net_exchange_flow"][-1] < 0.0
    assert out["flow_zscore"][-1] < 0.0
    assert out["flow_zscore_negative_streak_hours"][-1] >= 2


def test_funds_flow_features_preserve_inflow_minus_outflow_semantics() -> None:
    flow = pl.DataFrame(
        {
            "timestamp": [idx * HOUR_MS for idx in range(4)],
            "inflow": [10.0, 5.0, 0.0, 0.0],
            "outflow": [0.0, 20.0, 25.0, 30.0],
        }
    )
    prices = pl.DataFrame(
        {
            "timestamp": [idx * HOUR_MS for idx in range(4)],
            "close": [100.0, 100.5, 100.0, 99.5],
        }
    )

    out = compute_funds_flow_features(flow, prices, window_hours=4, total_supply=1_000.0)

    assert out["net_exchange_flow_30d"][-1] == -60.0
    assert out["exchange_outflow_dominant"][-1] is True
    assert out["exchange_inflow_dominant"][-1] is False
    assert out["net_exchange_flow_to_supply_30d"][-1] == -0.06


def test_funds_orderbook_features_mark_bid_support_only_on_down_flat_window() -> None:
    books = pl.DataFrame(
        {
            "timestamp": [idx * 60_000 for idx in range(4)],
            "bid_depth_bps_10": [40.0, 50.0, 70.0, 90.0],
            "ask_depth_bps_10": [80.0, 80.0, 80.0, 80.0],
            "bid_depth_bps_25": [80.0, 90.0, 110.0, 130.0],
            "ask_depth_bps_25": [100.0, 100.0, 100.0, 100.0],
            "bid_depth_bps_50": [150.0, 160.0, 170.0, 180.0],
            "ask_depth_bps_50": [150.0, 150.0, 150.0, 150.0],
        }
    )
    prices = pl.DataFrame(
        {
            "timestamp": [idx * 60_000 for idx in range(4)],
            "close": [100.0, 99.9, 99.8, 99.7],
        }
    )

    out = compute_funds_orderbook_features(books, prices, window_minutes=4)

    assert out.sort("timestamp").tail(1)["bid_support_rising_on_down_move"][0] is True


def test_funds_volume_features_aggregate_hourly_bars_to_daily_ratio() -> None:
    bars = pl.DataFrame(
        [
            {"timestamp": idx * HOUR_MS, "close": close, "vol": vol}
            for idx, (close, vol) in enumerate(
                [(10.0, 10.0)] * 23
                + [(11.0, 20.0)]
                + [(11.0, 5.0)] * 23
                + [(10.0, 5.0)]
            )
        ]
    )
    trades = pl.DataFrame(
        {
            "timestamp": [0, HOUR_MS],
            "side": ["buy", "sell"],
            "notional_usd": [30_000.0, 10_000.0],
        }
    )

    out = compute_funds_volume_features(bars, trades, large_trade_usd=10_000.0)

    assert out.sort("timestamp").tail(1)["up_day_volume_to_down_day_volume_30d"][0] > 1.0
    assert out.sort("timestamp").tail(1)["large_trade_buy_sell_ratio_24h"][0] == 3.0


def test_funds_message_features_missing_messages_are_missing_not_overheated() -> None:
    empty = compute_funds_message_features(pl.DataFrame())

    assert empty.is_empty()

    out = compute_funds_message_features(
        pl.DataFrame({"timestamp": [0], "mention_growth": [None]})
    )
    assert out["message_quiet_30d"][0] is False
    assert out["emotion_overheat_30d"][0] is False


def test_depth_features_uses_top_25_fallback_for_top_10() -> None:
    books = pl.DataFrame(
        {
            "timestamp": [0, 1_000, 2_000],
            "ob_bid_vol_25": [130.0, 140.0, 150.0],
            "ob_ask_vol_25": [70.0, 60.0, 50.0],
            "ob_imbalance_25": [0.3, 0.4, 0.5],
        }
    )

    out = compute_depth_features(books)

    assert out.height == 1
    assert out["depth_imbalance_10_mean"][0] == out["depth_imbalance_25_mean"][0]
    assert out["depth_imbalance_10_slope"][0] == 0.2


def test_structure_features_emit_range_economics() -> None:
    prices = pl.DataFrame(
        {
            "timestamp": [idx * HOUR_MS for idx in range(4)],
            "close": [10.0, 20.0, 15.0, 12.0],
        }
    )

    out = compute_structure_features(prices)
    latest = out.sort("timestamp").tail(1)

    assert latest["range_low_px"][0] == 10.0
    assert latest["range_high_px"][0] == 20.0
    assert latest["upside_to_range_high_pct"][0] == (20.0 / 12.0) - 1.0
    assert latest["downside_to_range_low_pct"][0] == (12.0 / 10.0) - 1.0
    assert latest["range_reward_risk"][0] == ((20.0 / 12.0) - 1.0) / ((12.0 / 10.0) - 1.0)
    assert latest["structure_invalidation_px"][0] == 10.0
    assert latest["structure_target_px"][0] == 20.0


def test_join_uses_backward_asof_and_warns_on_missing_sources() -> None:
    prices = pl.DataFrame(
        {
            "timestamp": [0, HOUR_MS, 2 * HOUR_MS],
            "close": [100.0, 101.0, 102.0],
        }
    )
    flow = pl.DataFrame({"timestamp": [HOUR_MS + 1], "inflow": [0.0], "outflow": [100.0]})

    out = join_hourly_accumulation_features(
        symbol="BTC-USDT-SWAP",
        inst_id="BTC-USDT-SWAP",
        price_frame=prices,
        flow_frame=flow,
    )

    assert out.filter(pl.col("timestamp") == HOUR_MS)["net_exchange_flow"][0] is None
    assert out.filter(pl.col("timestamp") == 2 * HOUR_MS)["net_exchange_flow"][0] == -100.0
    assert "book_missing" in out["data_quality_warning"][0]


def test_stale_book_evidence_is_null_and_warned() -> None:
    prices = pl.DataFrame(
        {
            "timestamp": [0, HOUR_MS, 3 * HOUR_MS],
            "close": [100.0, 99.0, 98.0],
        }
    )
    books = pl.DataFrame(
        {
            "timestamp": [0],
            "ob_bid_vol_25": [130.0],
            "ob_ask_vol_25": [70.0],
            "ob_imbalance_25": [0.3],
        }
    )

    out = join_hourly_accumulation_features(
        symbol="BTC-USDT-SWAP",
        inst_id="BTC-USDT-SWAP",
        price_frame=prices,
        book_frame=books,
        max_source_staleness_hours=1,
    )
    stale = out.filter(pl.col("timestamp") == 3 * HOUR_MS)

    assert stale["depth_imbalance_25_mean"][0] is None
    assert "books_stale" in stale["data_quality_warning"][0]


def test_taker_volume_features_compute_ratio_and_imbalance() -> None:
    taker = pl.DataFrame(
        {
            "timestamp": [HOUR_MS + 1, HOUR_MS + 2],
            "taker_buy_volume": [60.0, 40.0],
            "taker_sell_volume": [40.0, 60.0],
        }
    )

    out = compute_taker_volume_features(taker)

    assert out["timestamp"][0] == HOUR_MS
    assert out["taker_volume_total"][0] == 200.0
    assert out["taker_buy_ratio"][0] == 0.5
    assert out["taker_volume_imbalance"][0] == 0.0


def test_taker_volume_zero_total_emits_null_ratios() -> None:
    taker = pl.DataFrame(
        {
            "timestamp": [HOUR_MS],
            "taker_buy_volume": [0.0],
            "taker_sell_volume": [0.0],
        }
    )

    out = compute_taker_volume_features(taker)

    assert out["taker_buy_ratio"][0] is None
    assert out["taker_volume_imbalance"][0] is None


def test_long_short_ratio_features_pick_latest_row_per_hour() -> None:
    ratios = pl.DataFrame(
        {
            "timestamp": [HOUR_MS + 1, HOUR_MS + 2],
            "long_short_account_ratio": [1.1, 1.2],
            "top_trader_long_short_account_ratio": [1.3, 1.4],
        }
    )

    out = compute_long_short_ratio_features(ratios)

    assert out.height == 1
    assert out["long_short_account_ratio"][0] == 1.2
    assert out["top_trader_long_short_account_ratio"][0] == 1.4


def test_open_interest_24h_change_requires_timestamp_lag() -> None:
    oi = pl.DataFrame(
        {
            "timestamp": [0, 23 * HOUR_MS, 24 * HOUR_MS],
            "open_interest": [100.0, 120.0, 150.0],
            "open_interest_usd": [1000.0, 1100.0, 1300.0],
        }
    )

    out = compute_open_interest_features(oi)

    latest = out.filter(pl.col("timestamp") == 24 * HOUR_MS)
    irregular = out.filter(pl.col("timestamp") == 23 * HOUR_MS)

    assert latest["open_interest_change_24h"][0] == 0.5
    assert latest["open_interest_usd_change_24h"][0] == pytest.approx(0.3)
    assert irregular["open_interest_change_24h"][0] is None


def test_join_includes_rubik_context_and_stale_nulls() -> None:
    prices = pl.DataFrame(
        {
            "timestamp": [0, HOUR_MS, 4 * HOUR_MS],
            "close": [100.0, 101.0, 102.0],
        }
    )
    taker = pl.DataFrame(
        {
            "timestamp": [HOUR_MS],
            "taker_buy_volume": [70.0],
            "taker_sell_volume": [30.0],
        }
    )
    ratios = pl.DataFrame(
        {
            "timestamp": [HOUR_MS],
            "long_short_account_ratio": [1.2],
            "top_trader_long_short_position_ratio": [2.0],
        }
    )

    out = join_hourly_accumulation_features(
        symbol="BTC-USDT-SWAP",
        inst_id="BTC-USDT-SWAP",
        price_frame=prices,
        taker_volume_frame=taker,
        long_short_ratio_frame=ratios,
        max_source_staleness_hours=1,
    )
    fresh = out.filter(pl.col("timestamp") == HOUR_MS)
    stale = out.filter(pl.col("timestamp") == 4 * HOUR_MS)

    assert fresh["taker_buy_ratio"][0] == 0.7
    assert fresh["top_trader_long_short_position_ratio"][0] == 2.0
    assert stale["taker_buy_ratio"][0] is None
    assert stale["long_short_account_ratio"][0] is None
    assert "taker_volume_stale" in stale["data_quality_warning"][0]


def test_realtime_trade_features_require_persistent_buy_vwap_acceptance() -> None:
    trades = pl.DataFrame(
        {
            "timestamp": [idx * 60_000 for idx in range(15)],
            "price": [100.0 + idx * 0.2 for idx in range(15)],
            "notional_usd": [10_000.0] * 15,
            "side": ["buy"] * 12 + ["sell"] * 3,
        }
    )

    out = compute_realtime_trade_features(
        trades, window_minutes=15, subwindow_minutes=5, large_trade_usd=50_000.0
    )
    latest = out.sort("timestamp").tail(1)

    assert latest["aggressive_buy_ratio"][0] == 0.8
    assert latest["price_vs_vwap_pct"][0] > 0.0
    assert latest["positive_trade_windows"][0] >= 2


def test_realtime_orderbook_features_measure_depth_distribution_over_windows() -> None:
    books = pl.DataFrame(
        {
            "timestamp": [idx * 60_000 for idx in range(15)],
            "bid_depth_bps_10": [80.0] * 15,
            "ask_depth_bps_10": [40.0] * 15,
            "bid_depth_bps_25": [120.0 + idx for idx in range(15)],
            "ask_depth_bps_25": [60.0] * 15,
            "bid_depth_bps_50": [180.0] * 15,
            "ask_depth_bps_50": [120.0] * 15,
            "spread_bps": [8.0] * 15,
        }
    )

    out = compute_realtime_orderbook_features(books, window_minutes=15, subwindow_minutes=5)
    latest = out.sort("timestamp").tail(1)

    assert latest["depth_imbalance_15m_mean"][0] > 0.0
    assert latest["spread_bps_5m_median"][0] == 8.0
    assert latest["bid_depth_rebuild_15m"][0] > 1.0
    assert latest["bid_support_ratio_10_25"][0] > 0.0
    assert latest["positive_book_windows"][0] >= 3


def test_realtime_confirmation_does_not_confirm_from_single_book_snapshot() -> None:
    trades = pl.DataFrame(
        {
            "timestamp": [idx * 60_000 for idx in range(15)],
            "price": [100.0 + idx * 0.1 for idx in range(15)],
            "notional_usd": [10_000.0] * 15,
            "side": ["buy"] * 15,
        }
    )
    books = pl.DataFrame(
        {
            "timestamp": [14 * 60_000],
            "bid_depth_bps_25": [1_000.0],
            "ask_depth_bps_25": [100.0],
            "bid_depth_bps_50": [1_200.0],
            "ask_depth_bps_50": [200.0],
            "spread_bps": [5.0],
        }
    )

    trade_features = compute_realtime_trade_features(trades, window_minutes=15)
    book_features = compute_realtime_orderbook_features(books, window_minutes=15)
    confirmation = compute_realtime_confirmation_features(
        trade_features, book_features, structure_stage="base_ready"
    )
    latest = confirmation.sort("timestamp").tail(1)

    assert latest["realtime_confirmation_state"][0] != "trend_confirming"


def test_realtime_confirmation_overlays_structure_context() -> None:
    trades = pl.DataFrame(
        {
            "timestamp": [idx * 60_000 for idx in range(15)],
            "price": [100.0 + idx * 0.2 for idx in range(15)],
            "notional_usd": [100_000.0] * 15,
            "side": ["buy"] * 15,
        }
    )
    books = pl.DataFrame(
        {
            "timestamp": [idx * 60_000 for idx in range(15)],
            "bid_depth_bps_25": [200.0 + idx * 5.0 for idx in range(15)],
            "ask_depth_bps_25": [80.0] * 15,
            "bid_depth_bps_50": [260.0 + idx * 5.0 for idx in range(15)],
            "ask_depth_bps_50": [140.0] * 15,
            "spread_bps": [5.0] * 15,
        }
    )

    trade_features = compute_realtime_trade_features(
        trades, window_minutes=15, large_trade_usd=50_000.0
    )
    book_features = compute_realtime_orderbook_features(books, window_minutes=15)
    confirmation = compute_realtime_confirmation_features(
        trade_features, book_features, structure_stage="base_ready"
    )
    latest = confirmation.sort("timestamp").tail(1)

    assert latest["realtime_confirmation_state"][0] == "trend_confirming"
    assert latest["structure_overlay"][0] == "structure_confirmed"

