import polars as pl

from qooi.scanner.state import source_time_series_features_frame
from qooi.scanner.tailrun.artifacts import write_tailtree_source_timeseries_features


def test_source_time_series_features_build_funding_state_path_without_zscore() -> None:
    source_frames = {
        "funding": pl.DataFrame(
            {
                "symbol": ["BTC", "BTC", "BTC", "BTC"],
                "timestamp": [1, 2, 3, 4],
                "funding_rate": [0.0001, 0.0002, -0.0001, -0.0002],
            }
        )
    }
    bars = pl.DataFrame(
        {
            "symbol": ["BTC", "BTC", "BTC", "BTC"],
            "timestamp": [1, 2, 3, 4],
            "close": [100.0, 99.0, 101.0, 100.0],
        }
    )
    decision_keys = pl.DataFrame({"symbol": ["BTC", "BTC"], "timestamp": [3, 4]})

    features = source_time_series_features_frame(source_frames, bars, decision_keys)

    assert "funding_rate_z_30d" not in features.columns
    assert "funding_level_state" in features.columns
    assert "funding_level_transition" in features.columns
    assert "funding_path_24h" in features.columns
    row = features.filter(pl.col("timestamp") == 4).row(0, named=True)
    assert row["funding_level_state"] == "funding_negative"
    assert row["funding_level_transition"] == "funding_persistence"
    assert row["funding_direction_run_length"] == 2
    assert row["funding_path_24h"] == (
        "funding_positive -> funding_positive -> funding_negative -> funding_negative"
    )


def test_source_time_series_features_build_lsr_flow_and_paths() -> None:
    source_frames = {
        "long_short_ratios": pl.DataFrame(
            {
                "symbol": ["ETH", "ETH", "ETH"],
                "timestamp": [1, 2, 3],
                "long_short_account_ratio": [1.2, 1.4, 0.8],
            }
        ),
        "open_interest": pl.DataFrame(
            {
                "symbol": ["ETH", "ETH", "ETH"],
                "timestamp": [1, 2, 3],
                "open_interest": [100.0, 120.0, 110.0],
            }
        ),
        "taker_volume": pl.DataFrame(
            {
                "symbol": ["ETH", "ETH", "ETH"],
                "timestamp": [1, 2, 3],
                "taker_buy_volume": [10.0, 20.0, 5.0],
                "taker_sell_volume": [10.0, 10.0, 20.0],
            }
        ),
    }
    bars = pl.DataFrame(
        {
            "symbol": ["ETH", "ETH", "ETH"],
            "timestamp": [1, 2, 3],
            "close": [100.0, 102.0, 101.0],
        }
    )
    decision_keys = pl.DataFrame({"symbol": ["ETH"], "timestamp": [3]})

    features = source_time_series_features_frame(source_frames, bars, decision_keys)

    assert not any("z_" in column or column.endswith("_z") for column in features.columns)
    row = features.row(0, named=True)
    assert row["lsr_level_state"] == "lsr_short_crowding"
    assert row["lsr_level_transition"] == "lsr_flip"
    assert row["oi_flow_state"] == "oi_unwind"
    assert row["oi_flow_transition"] == "oi_flip"
    assert row["taker_pressure_state"] == "taker_sell_pressure"
    assert row["taker_pressure_transition"] == "taker_flip"


def test_write_tailtree_source_timeseries_features_writes_csv(tmp_path) -> None:
    frame = pl.DataFrame(
        {"symbol": ["BTC"], "timestamp": [1], "funding_level_state": ["funding_positive"]}
    )

    write_tailtree_source_timeseries_features(tmp_path, frame)

    assert (tmp_path / "tailtree-source-timeseries-features.csv").exists()
