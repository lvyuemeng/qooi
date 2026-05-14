"""Composable strategy feature builders."""

from __future__ import annotations

from collections.abc import Callable

import polars as pl

FeatureFn = Callable[[pl.DataFrame], pl.DataFrame]


def add_momentum_return(period: int, *, output: str = "momentum_return") -> FeatureFn:
    def _add(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns((pl.col("close") / pl.col("close").shift(period) - 1).alias(output))

    return _add


def add_volume_average(period: int = 20, *, output: str = "vol_avg") -> FeatureFn:
    def _add(df: pl.DataFrame) -> pl.DataFrame:
        vol_col = "vol" if "vol" in df.columns else "volume"
        return df.with_columns(pl.col(vol_col).rolling_mean(period).alias(output))

    return _add


def add_trend_maturity(
    *,
    ema_mid: int = 50,
    ema_slow: int = 200,
    output: str = "trend_bars",
) -> FeatureFn:
    def _add(df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return df.with_columns(pl.lit(0).alias(output))
        e_mid = df[f"ema_{ema_mid}"].to_list()
        e_slow = df[f"ema_{ema_slow}"].to_list()
        bars: list[int] = []
        trend_bars = 0
        for mid, slow in zip(e_mid, e_slow, strict=False):
            uptrend = mid is not None and slow is not None and mid > 0 and slow > 0 and mid > slow
            downtrend = mid is not None and slow is not None and mid > 0 and slow > 0 and mid < slow
            if uptrend:
                trend_bars = trend_bars + 1 if trend_bars >= 0 else 1
            elif downtrend:
                trend_bars = trend_bars - 1 if trend_bars <= 0 else -1
            else:
                trend_bars = 0
            bars.append(trend_bars)
        return df.with_columns(pl.Series(output, bars, dtype=pl.Int64))

    return _add


def add_utc_hour(*, timestamp_col: str = "timestamp", output: str = "hour_utc") -> FeatureFn:
    def _add(df: pl.DataFrame) -> pl.DataFrame:
        if "datetime" in df.columns:
            return df.with_columns(pl.col("datetime").dt.hour().fill_null(12).alias(output))
        hour = ((pl.col(timestamp_col) // 3_600_000) % 24).fill_null(12)
        return df.with_columns(hour.alias(output))

    return _add


def add_price_structure(period_short: int = 5, period_long: int = 20) -> FeatureFn:
    def _add(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col("low").shift(1).rolling_min(period_short).alias("low_short"),
            pl.col("low").shift(1).rolling_min(period_long).alias("low_long"),
            pl.col("high").shift(1).rolling_max(period_short).alias("high_short"),
            pl.col("high").shift(1).rolling_max(period_long).alias("high_long"),
        )

    return _add
