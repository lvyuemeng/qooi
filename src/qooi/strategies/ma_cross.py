"""Stateless signal expressions — pure Polars, no loop or mutable state.

Each is a ``pl.Expr`` that can be passed directly to ``Backtest(signal_expr=…)``.
"""

from __future__ import annotations

import polars as pl


def sma_cross_signal(fast: int = 10, slow: int = 30) -> pl.Expr:
    fast_sma = pl.col("close").rolling_mean(fast)
    slow_sma = pl.col("close").rolling_mean(slow)
    return (
        pl.when(fast_sma > slow_sma).then(1.0).when(fast_sma < slow_sma).then(-1.0).otherwise(0.0)
    ).alias("signal")


def bollinger_signal(period: int = 20, std_dev: float = 2.0) -> pl.Expr:
    mid = pl.col("close").rolling_mean(period)
    std = pl.col("close").rolling_std(period)
    return (
        pl.when(pl.col("close") < mid - std_dev * std)
        .then(1.0)
        .when(pl.col("close") > mid + std_dev * std)
        .then(-1.0)
        .otherwise(0.0)
    ).alias("signal")


def ema_vumanchu_signal(ema_short: int = 20, ema_long: int = 50) -> pl.Expr:
    trend_up = pl.col(f"ema_{ema_short}") > pl.col(f"ema_{ema_long}")
    trend_dn = pl.col(f"ema_{ema_short}") < pl.col(f"ema_{ema_long}")
    return (
        pl.when(trend_up & (pl.col("vm_long") == 1))
        .then(1.0)
        .when(trend_dn & (pl.col("vm_short") == 1))
        .then(-1.0)
        .otherwise(0.0)
    ).alias("signal")
