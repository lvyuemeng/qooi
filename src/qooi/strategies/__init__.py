"""SMA crossover strategy — example strategy for the backtester.

Produces a signal Series:
    +1 (long)  when fast SMA crosses above slow SMA
    -1 (short) when fast SMA crosses below slow SMA
     0 (flat)  otherwise

Usage::

    from qooi.strategies.ma_cross import sma_cross_signal
    bt = Backtest(df, signal_fn=sma_cross_signal(fast=10, slow=30))
    result = bt.run()
"""

from __future__ import annotations

import polars as pl


def sma_cross_signal(fast: int = 10, slow: int = 30) -> pl.Expr:
    """Return a Polars expression: +1 (long), -1 (short), 0 (flat)."""
    sma_fast = pl.col("close").rolling_mean(fast).alias("sma_fast")
    sma_slow = pl.col("close").rolling_mean(slow).alias("sma_slow")
    return (
        pl.when(sma_fast > sma_slow).then(1.0).when(sma_fast < sma_slow).then(-1.0).otherwise(0.0)
    ).alias("signal")


def ema_cross_signal(fast: int = 12, slow: int = 26) -> pl.Expr:
    """EMA crossover: +1 (long), -1 (short), 0 (flat)."""
    ema_fast = pl.col("close").ewm_mean(span=fast, min_periods=fast)
    ema_slow = pl.col("close").ewm_mean(span=slow, min_periods=slow)
    return (
        pl.when(ema_fast > ema_slow).then(1.0).when(ema_fast < ema_slow).then(-1.0).otherwise(0.0)
    ).alias("signal")


def bollinger_signal(period: int = 20, std_dev: float = 2.0) -> pl.Expr:
    """Bollinger Band mean-reversion signal.

    +1 (long)  when close < lower band (oversold → bounce)
    -1 (short) when close > upper band (overbought → revert)
     0 (flat)  otherwise

    Requires columns ``bb_middle``, ``bb_upper``, ``bb_lower``
    (computed by :func:`add_indicators`).
    """
    middle = pl.col("close").rolling_mean(period)
    std = pl.col("close").rolling_std(period)
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return (
        pl.when(pl.col("close") < lower)
        .then(1.0)
        .when(pl.col("close") > upper)
        .then(-1.0)
        .otherwise(0.0)
    ).alias("signal")
