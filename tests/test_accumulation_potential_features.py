from __future__ import annotations

import polars as pl

from qooi.accumulation.features import HOUR_MS
from qooi.strategies.potential import (
    _add_base_duration,
    compute_potential_features_batch,
)


def test_potential_features_compute_structure_volume_vwap_and_first_spike() -> None:
    symbol = "PENGU-USDT-SWAP"
    rows = []
    for idx in range(800):
        rows.append(
            {
                "symbol": symbol,
                "timestamp": idx * HOUR_MS,
                "close": 10.0 + idx * 0.001,
                "vol": 100.0,
            }
        )
    rows[-1]["close"] = 10.5
    rows[-1]["vol"] = 500.0
    bars = pl.DataFrame(rows)

    out = compute_potential_features_batch(
        bars, min_history_hours=720, full_history_hours=720, volume_spike_ratio=3.0
    )
    latest = out.sort("timestamp").tail(1)

    assert latest["history_hours"][0] == 800
    assert latest["price_to_90d_low"][0] > 1.0
    assert 0.0 <= latest["range_position_90d_pct"][0] <= 1.0
    assert latest["bb_width_percentile_90d"][0] is not None
    assert latest["volume_contraction_10d_90d"][0] > 0.0
    assert latest["volume_spike_ratio_1h_20h"][0] >= 3.0
    assert latest["prior_spike_count_5d"][0] == 0
    assert latest["first_volume_expansion"][0] is True
    assert latest["vwap_24h"][0] is not None
    assert latest["price_vs_vwap_24h_pct"][0] is not None


def test_potential_features_prior_spike_blocks_first_expansion() -> None:
    symbol = "TRIA-USDT-SWAP"
    rows = [
        {"symbol": symbol, "timestamp": idx * HOUR_MS, "close": 10.0, "vol": 100.0}
        for idx in range(800)
    ]
    rows[-25]["vol"] = 500.0
    rows[-1]["vol"] = 500.0

    out = compute_potential_features_batch(
        pl.DataFrame(rows), min_history_hours=720, full_history_hours=720, volume_spike_ratio=3.0
    )
    latest = out.sort("timestamp").tail(1)

    assert latest["prior_spike_count_5d"][0] >= 1
    assert latest["first_volume_expansion"][0] is False


def test_potential_features_warn_on_short_history() -> None:
    bars = pl.DataFrame(
        {
            "symbol": ["EDGE-USDT-SWAP"] * 3,
            "timestamp": [0, HOUR_MS, 2 * HOUR_MS],
            "close": [10.0, 10.1, 10.2],
            "vol": [100.0, 100.0, 100.0],
        }
    )

    out = compute_potential_features_batch(bars, min_history_hours=720)

    assert "insufficient_history" in out.sort("timestamp").tail(1)["data_quality_warning"][0]


def test_potential_bb_percentile_is_causal_for_historical_rows() -> None:
    symbol = "SUI-USDT-SWAP"
    prefix_rows = [
        {
            "symbol": symbol,
            "timestamp": idx * HOUR_MS,
            "close": 10.0 + (idx % 20) * 0.01,
            "vol": 100.0,
        }
        for idx in range(800)
    ]
    future_rows = [
        {
            "symbol": symbol,
            "timestamp": idx * HOUR_MS,
            "close": 20.0 + (idx % 5) * 5.0,
            "vol": 100.0,
        }
        for idx in range(800, 900)
    ]

    prefix = compute_potential_features_batch(
        pl.DataFrame(prefix_rows), min_history_hours=20, full_history_hours=720
    )
    full = compute_potential_features_batch(
        pl.DataFrame([*prefix_rows, *future_rows]), min_history_hours=20, full_history_hours=720
    )

    prefix_value = prefix.filter(pl.col("timestamp") == 799 * HOUR_MS)[
        "bb_width_percentile_90d"
    ][0]
    full_value = full.filter(pl.col("timestamp") == 799 * HOUR_MS)[
        "bb_width_percentile_90d"
    ][0]

    assert full_value == prefix_value


def test_potential_features_detect_active_downtrend_cooldown() -> None:
    symbol = "COOL-USDT-SWAP"
    rows = [
        {
            "symbol": symbol,
            "timestamp": idx * HOUR_MS,
            "close": 20.0 - idx * 0.01,
            "vol": 100.0,
        }
        for idx in range(900)
    ]

    out = compute_potential_features_batch(
        pl.DataFrame(rows), min_history_hours=720, full_history_hours=720
    )
    latest = out.sort("timestamp").tail(1)

    assert latest["new_low_count_30d"][0] > 2
    assert latest["ma_30d_slope_14d"][0] < 0.0
    assert "active_lower_lows" in latest["structure_block_reason"][0]


def test_potential_features_detect_stable_base_duration_and_reclaim() -> None:
    symbol = "BASE-USDT-SWAP"
    rows = []
    for idx in range(500):
        rows.append(
            {
                "symbol": symbol,
                "timestamp": idx * HOUR_MS,
                "close": 20.0 - idx * 0.02,
                "vol": 100.0,
            }
        )
    for idx in range(500, 1300):
        rows.append(
            {
                "symbol": symbol,
                "timestamp": idx * HOUR_MS,
                "close": 10.0 + (idx - 500) * 0.001,
                "vol": 100.0,
            }
        )

    out = compute_potential_features_batch(
        pl.DataFrame(rows), min_history_hours=720, full_history_hours=720
    )
    latest = out.sort("timestamp").tail(1)

    assert latest["base_duration_hours"][0] >= 168
    assert latest["new_low_count_30d"][0] <= 2
    assert latest["reclaim_state"][0] in {"ma7_reclaim", "ma30_reclaim", "reclaim_hold"}


def test_vectorized_base_duration_preserves_reset_semantics() -> None:
    frame = pl.DataFrame(
        [
            {
                "symbol": "A-USDT-SWAP",
                "timestamp": 1,
                "price_to_90d_low": 1.1,
                "range_position_90d_pct": 0.2,
                "bb_width_percentile_90d": 0.2,
                "_new_low_30d": False,
            },
            {
                "symbol": "A-USDT-SWAP",
                "timestamp": 2,
                "price_to_90d_low": 1.1,
                "range_position_90d_pct": 0.2,
                "bb_width_percentile_90d": 0.2,
                "_new_low_30d": False,
            },
            {
                "symbol": "A-USDT-SWAP",
                "timestamp": 3,
                "price_to_90d_low": 1.1,
                "range_position_90d_pct": 0.2,
                "bb_width_percentile_90d": 0.2,
                "_new_low_30d": True,
            },
            {
                "symbol": "A-USDT-SWAP",
                "timestamp": 4,
                "price_to_90d_low": 1.1,
                "range_position_90d_pct": 0.2,
                "bb_width_percentile_90d": 0.2,
                "_new_low_30d": False,
            },
            {
                "symbol": "A-USDT-SWAP",
                "timestamp": 5,
                "price_to_90d_low": 2.0,
                "range_position_90d_pct": 0.8,
                "bb_width_percentile_90d": 0.8,
                "_new_low_30d": False,
            },
            {
                "symbol": "B-USDT-SWAP",
                "timestamp": 1,
                "price_to_90d_low": 1.1,
                "range_position_90d_pct": 0.2,
                "bb_width_percentile_90d": 0.2,
                "_new_low_30d": False,
            },
        ]
    )

    out = _add_base_duration(frame).sort(["symbol", "timestamp"])

    assert out.filter(pl.col("symbol") == "A-USDT-SWAP")["base_duration_hours"].to_list() == [
        1,
        2,
        0,
        1,
        0,
    ]
    assert out.filter(pl.col("symbol") == "B-USDT-SWAP")["base_duration_hours"].to_list() == [1]


def test_potential_structure_maturity_features_are_causal() -> None:
    symbol = "CAUSAL-USDT-SWAP"
    prefix_rows = [
        {"symbol": symbol, "timestamp": idx * HOUR_MS, "close": 10.0, "vol": 100.0}
        for idx in range(800)
    ]
    future_rows = [
        {"symbol": symbol, "timestamp": idx * HOUR_MS, "close": 20.0, "vol": 100.0}
        for idx in range(800, 900)
    ]

    prefix = compute_potential_features_batch(
        pl.DataFrame(prefix_rows), min_history_hours=20, full_history_hours=720
    )
    full = compute_potential_features_batch(
        pl.DataFrame([*prefix_rows, *future_rows]), min_history_hours=20, full_history_hours=720
    )

    prefix_row = prefix.filter(pl.col("timestamp") == 799 * HOUR_MS)
    full_row = full.filter(pl.col("timestamp") == 799 * HOUR_MS)

    assert full_row["base_duration_hours"][0] == prefix_row["base_duration_hours"][0]
    assert full_row["ma_30d_slope_14d"][0] == prefix_row["ma_30d_slope_14d"][0]

