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


def vumanchu_swing(
    df: pl.DataFrame,
    period: int = 20,
    multiplier: float = 3.5,
) -> tuple[pl.Series, pl.Series]:
    """VuManChu Swing Free — range filter with long/short conditions.

    1. Compute range size = EMA(abs(close - close[1]), period) * multiplier
    2. Range filter = smoothed series within the range channel
    3. longCondition = close breaks above range filter by range size
    4. shortCondition = close breaks below range filter by range size

    Returns (long_condition, short_condition) as +1/0/-1 style signals.
    """
    close = df["close"]
    range_raw = (close - close.shift(1)).abs().ewm_mean(span=period, min_periods=period)
    range_size = range_raw * multiplier
    # Smooth the range filter — ema of price with range as noise buffer
    rf = close.ewm_mean(span=period, min_periods=period)
    rf_upper = rf + range_size
    rf_lower = rf - range_size

    long_cond = (close > rf_upper).cast(pl.Int32)
    short_cond = (close < rf_lower).cast(pl.Int32)

    # Persist signal until opposite triggers
    long_signal = long_cond.shift(1).fill_null(0)
    short_signal = short_cond.shift(1).fill_null(0)
    return long_signal, short_signal


def add_indicators(df: pl.DataFrame) -> pl.DataFrame:
    """Convenience: add the most common indicators in one call."""
    long_sig, short_sig = vumanchu_swing(df)
    return df.with_columns(
        [
            sma(df, 20).alias("sma_20"),
            sma(df, 50).alias("sma_50"),
            ema(df, 12).alias("ema_12"),
            ema(df, 26).alias("ema_26"),
            ema(df, 50).alias("ema_50"),
            ema(df, 200).alias("ema_200"),
            rsi(df, 14).alias("rsi_14"),
            atr(df, 14).alias("atr_14"),
            volatility(df, 20).alias("volatility_20"),
            sma(df, 20).alias("bb_middle"),
            long_sig.alias("vm_long"),
            short_sig.alias("vm_short"),
        ]
    ).with_columns(
        [
            (pl.col("bb_middle") + 2.0 * pl.col("close").rolling_std(20)).alias("bb_upper"),
            (pl.col("bb_middle") - 2.0 * pl.col("close").rolling_std(20)).alias("bb_lower"),
        ]
    )
