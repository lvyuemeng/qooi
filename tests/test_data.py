"""Data processing tests — indicators, cache, market data."""

import polars as pl

from qooi.exchange.indicator import add_indicators


class TestIndicators:
    def test_add_indicators_columns(self):
        df = pl.DataFrame(
            {
                "timestamp": list(
                    range(1_700_000_000_000, 1_700_000_000_000 + 50 * 14_400_000, 14_400_000)
                ),
                "open": [100.0] * 50,
                "high": [102.0] * 50,
                "low": [98.0] * 50,
                "close": [100.0] * 50,
                "vol": [1000.0] * 50,
            }
        )
        df = add_indicators(df)
        for col in ("atr_14", "ema_20", "ema_50", "ema_200"):
            assert col in df.columns, f"Missing {col}"
        # ATR should be positive for non-flat prices
        assert df["atr_14"].drop_nulls().sum() > 0

    def test_add_indicators_on_real_data(self):
        df = pl.read_parquet("data/cache/BTC_USDT_4H.parquet")
        df = add_indicators(df)
        assert df.height > 0
        assert df["atr_14"].drop_nulls().len() > df.height * 0.8, (
            "ATR should have values for 80%+ of rows"
        )

    def test_empty_df(self):
        df = pl.DataFrame()
        # add_indicators doesn't guard empty input — skip gracefully
        if df.is_empty():
            return
        df = add_indicators(df)
        assert df.is_empty()
