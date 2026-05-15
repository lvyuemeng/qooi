"""Composable strategy condition builders."""

from __future__ import annotations

import polars as pl


def uptrend(ema_mid: int = 50, ema_slow: int = 200) -> pl.Expr:
    return (pl.col(f"ema_{ema_mid}") > 0) & (pl.col(f"ema_{ema_slow}") > 0) & (
        pl.col(f"ema_{ema_mid}") > pl.col(f"ema_{ema_slow}")
    )


def downtrend(ema_mid: int = 50, ema_slow: int = 200) -> pl.Expr:
    return (pl.col(f"ema_{ema_mid}") > 0) & (pl.col(f"ema_{ema_slow}") > 0) & (
        pl.col(f"ema_{ema_mid}") < pl.col(f"ema_{ema_slow}")
    )


def trend_mature(min_bars: int) -> pl.Expr:
    return pl.col("trend_bars").abs() >= min_bars


def adx_above(threshold: float = 20.0) -> pl.Expr:
    return pl.col("adx_14") > threshold


def session_between(start_hour: int = 8, end_hour: int = 22) -> pl.Expr:
    return pl.col("hour_utc").is_between(start_hour, end_hour)


def volume_spike(multiplier: float = 1.5) -> pl.Expr:
    return pl.col("vol") > multiplier * pl.col("vol_avg")


def above_ema(period: int = 20) -> pl.Expr:
    return pl.col("close") > pl.col(f"ema_{period}")


def below_ema(period: int = 20) -> pl.Expr:
    return pl.col("close") < pl.col(f"ema_{period}")


def higher_low_structure() -> pl.Expr:
    return pl.col("low_short") > pl.col("low_long")


def lower_high_structure() -> pl.Expr:
    return pl.col("high_short") < pl.col("high_long")


def momentum_gt(threshold: float) -> pl.Expr:
    return pl.col("momentum_return") > threshold


def momentum_lt(threshold: float) -> pl.Expr:
    return pl.col("momentum_return") < threshold


def rsi_cross_from_oversold(
    *, rsi_period: int = 14, oversold: float = 30.0, bounce: float = 25.0
) -> pl.Expr:
    rsi = pl.col(f"rsi_{rsi_period}")
    return (rsi > bounce) & (rsi.shift(1) <= oversold)


def rsi_bounce_held(*, rsi_period: int = 14, confirmation: float = 20.0) -> pl.Expr:
    rsi = pl.col(f"rsi_{rsi_period}")
    return (rsi > confirmation) & (rsi.shift(1) > confirmation)


def rsi_above(*, rsi_period: int = 14, threshold: float = 50.0) -> pl.Expr:
    return pl.col(f"rsi_{rsi_period}") > threshold


def rsi_below(*, rsi_period: int = 14, threshold: float = 50.0) -> pl.Expr:
    return pl.col(f"rsi_{rsi_period}") < threshold


def zscore_below(threshold: float, *, col: str = "close_z_score") -> pl.Expr:
    return pl.col(col) <= threshold


def zscore_above(threshold: float, *, col: str = "close_z_score") -> pl.Expr:
    return pl.col(col) >= threshold


def zscore_reverted_long(exit_level: float = 0.0, *, col: str = "close_z_score") -> pl.Expr:
    return pl.col(col) >= exit_level


def zscore_reverted_short(exit_level: float = 0.0, *, col: str = "close_z_score") -> pl.Expr:
    return pl.col(col) <= -exit_level


def dynamic_z_below(threshold: float, *, col: str = "dynamic_z_score") -> pl.Expr:
    return pl.col(col) <= threshold


def dynamic_z_above(threshold: float, *, col: str = "dynamic_z_score") -> pl.Expr:
    return pl.col(col) >= threshold


def dynamic_z_reverted_long(exit_level: float = 0.0, *, col: str = "dynamic_z_score") -> pl.Expr:
    return pl.col(col) >= -abs(exit_level)


def dynamic_z_reverted_short(exit_level: float = 0.0, *, col: str = "dynamic_z_score") -> pl.Expr:
    return pl.col(col) <= abs(exit_level)


def volatility_ratio_below(threshold: float, *, col: str = "volatility_ratio") -> pl.Expr:
    return pl.col(col) <= threshold


def macd_hist_above(threshold: float = 0.0, *, col: str = "macd_hist") -> pl.Expr:
    return pl.col(col) > threshold


def macd_hist_below(threshold: float = 0.0, *, col: str = "macd_hist") -> pl.Expr:
    return pl.col(col) < threshold
