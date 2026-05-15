"""Composable strategy feature builders."""

from __future__ import annotations

from collections.abc import Callable
from math import log, sqrt
from statistics import median

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


def add_z_score(
    period: int = 20, *, col: str = "close", output: str = "close_z_score"
) -> FeatureFn:
    def _add(df: pl.DataFrame) -> pl.DataFrame:
        mean = pl.col(col).rolling_mean(period)
        std = pl.col(col).rolling_std(period)
        safe_std = pl.when(std.abs() > 1e-10).then(std).otherwise(1e-10)
        return df.with_columns(((pl.col(col) - mean) / safe_std).alias(output))

    return _add


def add_ewma_z_score(
    span: int = 48,
    *,
    col: str = "close",
    output: str = "ewma_z_score",
) -> FeatureFn:
    def _add(df: pl.DataFrame) -> pl.DataFrame:
        mean = pl.col(col).ewm_mean(span=span, min_samples=span)
        variance = ((pl.col(col) - mean) ** 2).ewm_mean(span=span, min_samples=span)
        std = variance.sqrt()
        safe_std = pl.when(std.abs() > 1e-10).then(std).otherwise(1e-10)
        return df.with_columns(((pl.col(col) - mean) / safe_std).alias(output))

    return _add


def add_robust_z_score(
    period: int = 96,
    *,
    col: str = "close",
    output: str = "robust_z_score",
) -> FeatureFn:
    def _add(df: pl.DataFrame) -> pl.DataFrame:
        values = [float(v) if v is not None else None for v in df[col].to_list()]
        z_values: list[float | None] = []
        for idx, value in enumerate(values):
            window = [v for v in values[max(0, idx - period + 1) : idx + 1] if v is not None]
            if value is None or len(window) < period:
                z_values.append(None)
                continue
            med = median(window)
            mad = median([abs(v - med) for v in window])
            safe_mad = mad if abs(mad) > 1e-10 else 1e-10
            z_values.append((value - med) / (1.4826 * safe_mad))
        return df.with_columns(pl.Series(output, z_values, dtype=pl.Float64))

    return _add


def add_volatility_scaled_return(
    ret_period: int = 1,
    vol_span: int = 48,
    *,
    output: str = "vol_scaled_return",
) -> FeatureFn:
    def _add(df: pl.DataFrame) -> pl.DataFrame:
        ret = (pl.col("close") / pl.col("close").shift(ret_period)).log()
        vol = (ret**2).ewm_mean(span=vol_span, min_samples=vol_span).sqrt()
        safe_vol = pl.when(vol.abs() > 1e-10).then(vol).otherwise(1e-10)
        return df.with_columns((ret / safe_vol).alias(output))

    return _add


def add_dynamic_z_blend(
    *,
    z_col: str = "close_z_score",
    ewma_col: str = "ewma_z_score",
    robust_col: str = "robust_z_score",
    output: str = "dynamic_z_score",
) -> FeatureFn:
    def _add(df: pl.DataFrame) -> pl.DataFrame:
        available = [col for col in (z_col, ewma_col, robust_col) if col in df.columns]
        if not available:
            return df.with_columns(pl.lit(None, dtype=pl.Float64).alias(output))
        return df.with_columns(
            pl.mean_horizontal(*(pl.col(col) for col in available)).alias(output)
        )

    return _add


def add_volatility_regime(
    short_span: int = 24,
    long_span: int = 168,
    *,
    output: str = "volatility_regime",
) -> FeatureFn:
    def _add(df: pl.DataFrame) -> pl.DataFrame:
        ret = (pl.col("close") / pl.col("close").shift(1)).log()
        short = (ret**2).ewm_mean(span=short_span, min_samples=short_span).sqrt()
        long = (ret**2).ewm_mean(span=long_span, min_samples=long_span).sqrt()
        safe_long = pl.when(long.abs() > 1e-10).then(long).otherwise(1e-10)
        ratio = short / safe_long
        regime = pl.when(ratio < 0.75).then(-1).when(ratio > 1.5).then(1).otherwise(0)
        return df.with_columns(
            short.alias("realized_vol_short"),
            long.alias("realized_vol_long"),
            ratio.alias("volatility_ratio"),
            regime.alias(output),
        )

    return _add


def add_garch_like_volatility(
    omega: float = 0.0,
    alpha: float = 0.08,
    beta: float = 0.90,
    *,
    output: str = "conditional_volatility",
) -> FeatureFn:
    def _add(df: pl.DataFrame) -> pl.DataFrame:
        closes = [float(v) if v is not None else None for v in df["close"].to_list()]
        returns = [
            None
            if prev is None or curr is None or prev <= 0 or curr <= 0
            else log(curr / prev)
            for prev, curr in zip([None, *closes[:-1]], closes, strict=False)
        ]
        variance = 0.0
        vols: list[float | None] = []
        z_returns: list[float | None] = []
        for ret in returns:
            if ret is None:
                vols.append(None)
                z_returns.append(None)
                continue
            variance = omega + alpha * (ret**2) + beta * variance
            vol = sqrt(max(variance, 1e-20))
            vols.append(vol)
            z_returns.append(ret / vol if vol > 1e-10 else None)
        return df.with_columns(
            pl.Series(output, vols, dtype=pl.Float64),
            pl.Series("garch_z_return", z_returns, dtype=pl.Float64),
        )

    return _add


def add_macd_histogram(
    *,
    fast_ema: int = 12,
    slow_ema: int = 26,
    signal_period: int = 9,
    output: str = "macd_hist",
) -> FeatureFn:
    def _add(df: pl.DataFrame) -> pl.DataFrame:
        macd = pl.col(f"ema_{fast_ema}") - pl.col(f"ema_{slow_ema}")
        signal = macd.ewm_mean(span=signal_period, min_samples=signal_period)
        return df.with_columns((macd - signal).alias(output))

    return _add
