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


def adx(df: pl.DataFrame, period: int = 14) -> pl.Series:
    """Average Directional Index — trend strength (0-100)."""
    high = df["high"].to_list()
    low = df["low"].to_list()
    close = df["close"].to_list()

    plus_dm = [0.0] * len(df)
    minus_dm = [0.0] * len(df)
    tr = [0.0] * len(df)

    for i in range(1, len(df)):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0.0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0.0
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    plus_dm_smooth = pl.Series(plus_dm).rolling_mean(period)
    minus_dm_smooth = pl.Series(minus_dm).rolling_mean(period)
    tr_smooth = pl.Series(tr).rolling_mean(period)

    plus_di = 100.0 * plus_dm_smooth / tr_smooth.replace(0, 1e-10)
    minus_di = 100.0 * minus_dm_smooth / tr_smooth.replace(0, 1e-10)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)
    adx_vals = dx.rolling_mean(period)
    return adx_vals


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
    channel_deviation_mult: float = 0.0,
) -> tuple[pl.Series, pl.Series]:
    """VuManChu Swing Free — range filter with long/short conditions.

    1. Compute range size = EMA(abs(close - close[1]), period) * multiplier
    2. Range filter = smoothed series within the range channel
    3. longCondition = close breaks above upper band (favorable)
    4. shortCondition = close breaks below lower band (favorable)
    5. Signal persists until opposite trigger or channel deviation exit

    When ``channel_deviation_mult > 0``, the position flattens if price
    deviates more than ``channel_deviation_mult * range_size`` from the
    range filter midpoint. This prevents the VM from holding a position
    through a crash where the channel widens with volatility.

    Returns (long_signal, short_signal) as 1/0 booleans.
    """
    close = df["close"]
    range_raw = (close - close.shift(1)).abs().ewm_mean(span=period, min_periods=period)
    range_size = range_raw * multiplier
    rf = close.ewm_mean(span=period, min_periods=period)
    rf_upper = rf + range_size
    rf_lower = rf - range_size

    long_entry = close > rf_upper
    short_entry = close < rf_lower

    if channel_deviation_mult > 0:
        exit_threshold = range_size * channel_deviation_mult
        too_far = (close > rf + exit_threshold) | (close < rf - exit_threshold)
    else:
        too_far = None

    state = pl.Series([0.0] * len(df))
    prev = 0.0
    for i in range(len(df)):
        if too_far is not None and too_far[i] and prev != 0:
            prev = 0.0
        elif long_entry[i]:
            prev = 1.0
        elif short_entry[i]:
            prev = -1.0
        state[i] = prev

    long_signal = (state == 1.0).cast(pl.Int32)
    short_signal = (state == -1.0).cast(pl.Int32)
    return long_signal, short_signal


def add_indicators(df: pl.DataFrame, vm_channel_deviation_mult: float = 1.5) -> pl.DataFrame:
    """Convenience: add the most common indicators in one call.

    Args:
        vm_channel_deviation_mult: Exit VM position when price deviates
            beyond this multiple of the channel width from midpoint.
            Set to 0 to disable (original behavior). Default 1.5 avoids
            the "channel expansion trap" during crashes.
    """
    long_sig, short_sig = vumanchu_swing(df, channel_deviation_mult=vm_channel_deviation_mult)
    adx_vals = adx(df, 14)
    return df.with_columns(
        [
            sma(df, 20).alias("sma_20"),
            sma(df, 50).alias("sma_50"),
            ema(df, 12).alias("ema_12"),
            ema(df, 26).alias("ema_26"),
            ema(df, 20).alias("ema_20"),
            ema(df, 50).alias("ema_50"),
            ema(df, 200).alias("ema_200"),
            rsi(df, 14).alias("rsi_14"),
            atr(df, 14).alias("atr_14"),
            volatility(df, 20).alias("volatility_20"),
            sma(df, 20).alias("bb_middle"),
            long_sig.alias("vm_long"),
            short_sig.alias("vm_short"),
            adx_vals.alias("adx_14"),
        ]
    ).with_columns(
        [
            (pl.col("bb_middle") + 2.0 * pl.col("close").rolling_std(20)).alias("bb_upper"),
            (pl.col("bb_middle") - 2.0 * pl.col("close").rolling_std(20)).alias("bb_lower"),
        ]
    )
