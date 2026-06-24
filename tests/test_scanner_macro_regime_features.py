import polars as pl

from qooi.scanner.state import _market_context_features


def test_market_context_features_join_cross_symbol_regime_by_timestamp() -> None:
    features = pl.DataFrame(
        {
            "symbol": ["A", "B", "C", "A", "B", "C"],
            "timestamp": [1, 1, 1, 2, 2, 2],
            "bar_return_1h_pct": [1.0, -1.0, 3.0, 2.0, 4.0, -2.0],
            "bar_return_4h_pct": [2.0, 0.0, 4.0, 3.0, 5.0, -1.0],
            "bar_return_24h_pct": [3.0, -1.0, 5.0, 4.0, 6.0, -2.0],
        }
    )

    out = _market_context_features(features).sort(["timestamp", "symbol"])

    expected = {
        "market_return_1h_median",
        "market_return_4h_median",
        "market_return_24h_median",
        "market_abs_return_24h_median",
        "market_dispersion_24h",
        "market_positive_return_24h_share",
        "symbol_vs_market_return_24h",
        "symbol_vs_market_return_4h",
        "symbol_abs_return_vs_market_24h",
    }
    assert expected.issubset(set(out.columns))

    first = out.filter((pl.col("timestamp") == 1) & (pl.col("symbol") == "A")).row(
        0, named=True
    )
    assert first["market_return_24h_median"] == 3.0
    assert first["market_positive_return_24h_share"] == 2 / 3
    assert first["symbol_vs_market_return_24h"] == 0.0


def test_market_context_features_empty_schema() -> None:
    out = _market_context_features(pl.DataFrame())

    assert {
        "market_return_24h_median",
        "market_dispersion_24h",
        "symbol_vs_market_return_24h",
    }.issubset(set(out.columns))
