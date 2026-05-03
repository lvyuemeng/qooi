"""Technical indicators for OHLCV data — built on Polars.

All functions accept/return Polars DataFrames with at minimum columns:
    timestamp, close  (and open/high/low/vol where relevant)
"""

from __future__ import annotations

import polars as pl


def sma(df: pl.DataFrame, period: int = 20, col: str = "close") -> pl.Series:
    """Simple Moving Average."""
    return df[col].rolling_mean(period)


def ema(df: pl.DataFrame, period: int = 20, col: str = "close") -> pl.Series:
    """Exponential Moving Average (span = period)."""
    return df[col].ewm_mean(span=period, min_periods=period)


def rsi(df: pl.DataFrame, period: int = 14, col: str = "close") -> pl.Series:
    """Relative Strength Index."""
    delta = df[col].diff()
    gain = delta.to_list()
    loss = delta.to_list()

    gain_series = pl.Series([0.0 if v is None or v < 0 else float(v) for v in gain])
    loss_series = pl.Series([0.0 if v is None or v > 0 else abs(float(v)) for v in loss])

    avg_gain = gain_series.rolling_mean(period)
    avg_loss = loss_series.rolling_mean(period)

    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100.0 - (100.0 / (1.0 + rs))


def atr(df: pl.DataFrame, period: int = 14) -> pl.Series:
    """Average True Range."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()

    # Take max of the three per row
    true_range = pl.Series(
        [
            max(a or 0.0, b or 0.0, c or 0.0)
            for a, b, c in zip(tr1.to_list(), tr2.to_list(), tr3.to_list())
        ]
    )
    return true_range.rolling_mean(period)


def bollinger_bands(
    df: pl.DataFrame, period: int = 20, std_dev: float = 2.0, col: str = "close"
) -> tuple[pl.Series, pl.Series, pl.Series]:
    """Bollinger Bands — returns (middle, upper, lower)."""
    middle = sma(df, period, col)
    std = df[col].rolling_std(period)
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return middle, upper, lower


def volatility(df: pl.DataFrame, period: int = 20, col: str = "close") -> pl.Series:
    """Historical volatility — standard deviation of log returns."""
    log_ret = df[col].log().diff()
    return log_ret.rolling_std(period)


def add_indicators(df: pl.DataFrame) -> pl.DataFrame:
    """Convenience: add the most common indicators in one call.

    Adds columns: sma_20, sma_50, ema_12, ema_26, rsi_14, atr_14, bb_upper, bb_lower
    """
    return df.with_columns(
        [
            sma(df, 20).alias("sma_20"),
            sma(df, 50).alias("sma_50"),
            ema(df, 12).alias("ema_12"),
            ema(df, 26).alias("ema_26"),
            rsi(df, 14).alias("rsi_14"),
            atr(df, 14).alias("atr_14"),
            volatility(df, 20).alias("volatility_20"),
            sma(df, 20).alias("bb_middle"),
        ]
    ).with_columns(
        [
            (pl.col("bb_middle") + 2.0 * pl.col("close").rolling_std(20)).alias("bb_upper"),
            (pl.col("bb_middle") - 2.0 * pl.col("close").rolling_std(20)).alias("bb_lower"),
        ]
    )
