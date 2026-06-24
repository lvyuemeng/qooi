import polars as pl

from qooi.scanner.state import _kline_continuous_features


def test_kline_continuous_features_include_pre_entry_cleanliness_columns() -> None:
    timestamps = list(range(1, 61))
    close = [100.0 + ((-1) ** i) * (i % 5) + i * 0.2 for i in timestamps]
    bars = {
        ("BTC-USDT-SWAP", "1H"): pl.DataFrame(
            {
                "timestamp": timestamps,
                "open": [value - 0.3 for value in close],
                "high": [value + 1.0 for value in close],
                "low": [value - 1.0 for value in close],
                "close": close,
                "volume": [1000.0 + i for i in timestamps],
            }
        )
    }
    states = {
        ("BTC-USDT-SWAP", "1H"): pl.DataFrame(
            {
                "timestamp": timestamps,
                "atr_percentile_100": [50.0] * len(timestamps),
                "range_width_atr": [2.0] * len(timestamps),
            }
        )
    }

    features = _kline_continuous_features(bars, states, "1H")

    expected = {
        "return_sign_flip_rate_6h",
        "return_sign_flip_rate_24h",
        "body_to_range_mean_24h",
        "range_expansion_24h_to_7d",
        "close_position_24h",
        "prior_runup_6h",
        "prior_drawdown_6h",
        "return_efficiency_24h",
    }
    assert expected.issubset(set(features.columns))

    mature = features.filter(pl.col("timestamp") == 60).row(0, named=True)
    for column in expected:
        assert mature[column] is not None


def test_kline_empty_schema_includes_pre_entry_cleanliness_columns() -> None:
    features = _kline_continuous_features({}, {}, "1H")

    assert {
        "return_sign_flip_rate_6h",
        "return_sign_flip_rate_24h",
        "body_to_range_mean_24h",
        "range_expansion_24h_to_7d",
        "close_position_24h",
        "prior_runup_6h",
        "prior_drawdown_6h",
        "return_efficiency_24h",
    }.issubset(set(features.columns))
