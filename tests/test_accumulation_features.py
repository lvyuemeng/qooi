from __future__ import annotations

import polars as pl

from qooi.accumulation.features import (
    HOUR_MS,
    compute_depth_features,
    compute_flow_features,
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
