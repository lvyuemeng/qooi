"""Pair trading signal generation — rolling hedge ratio and spread z-score."""

from __future__ import annotations

import math

import polars as pl


def build_pair_frame(
    left: pl.DataFrame,
    right: pl.DataFrame,
    left_name: str = "left",
    right_name: str = "right",
) -> pl.DataFrame:
    """Align two OHLCV frames on timestamp and rename close columns."""
    right_frame = right.select(["timestamp", pl.col("close").alias(f"close_{right_name}")])
    left_frame = left.select(["timestamp", pl.col("close").alias(f"close_{left_name}")])
    return left_frame.join(right_frame, on="timestamp", how="inner").sort("timestamp")


def pair_spread_signal(
    df: pl.DataFrame,
    left_col: str = "close_left",
    right_col: str = "close_right",
    beta_window: int = 240,
    z_window: int = 120,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 3.5,
    max_hold_bars: int = 48,
) -> pl.DataFrame:
    """Create spread-trading signal with rolling hedge ratio and z-score.

    Output columns:
    - `signal`: +1 long spread, -1 short spread, 0 flat
    - `hedge_ratio`: rolling beta of left vs right log prices
    - `spread`
    - `zscore`
    """
    if df.is_empty():
        return df.with_columns(pl.lit(0.0).alias("signal"))

    left = df[left_col].fill_nan(0).fill_null(0).to_list()
    right = df[right_col].fill_nan(0).fill_null(0).to_list()

    log_left = [math.log(x) if x > 0 else 0.0 for x in left]
    log_right = [math.log(x) if x > 0 else 0.0 for x in right]

    hedge_ratio = [0.0] * len(df)
    spread = [0.0] * len(df)
    zscore = [0.0] * len(df)
    signal = [0.0] * len(df)

    active = 0
    bars_in_trade = 0

    for i in range(max(beta_window, z_window), len(df)):
        xs = log_left[i - beta_window : i]
        ys = log_right[i - beta_window : i]
        ym = sum(ys) / len(ys)
        xm = sum(xs) / len(xs)
        var_y = sum((y - ym) ** 2 for y in ys)
        cov_xy = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
        beta = cov_xy / var_y if var_y > 1e-12 else 1.0
        hedge_ratio[i] = beta

        spread[i] = log_left[i] - beta * log_right[i]
        sw = spread[i - z_window : i]
        mean = sum(sw) / len(sw)
        var = sum((x - mean) ** 2 for x in sw) / max(1, len(sw) - 1)
        std = math.sqrt(var) if var > 0 else 0.0
        z = (spread[i] - mean) / std if std > 1e-12 else 0.0
        zscore[i] = z

        if active == 0:
            if z <= -entry_z:
                active = 1
                bars_in_trade = 0
            elif z >= entry_z:
                active = -1
                bars_in_trade = 0
        else:
            bars_in_trade += 1
            if abs(z) <= exit_z or abs(z) >= stop_z or bars_in_trade >= max_hold_bars:
                active = 0
                bars_in_trade = 0

        signal[i] = float(active)

    return df.with_columns(
        [
            pl.Series(hedge_ratio).alias("hedge_ratio"),
            pl.Series(spread).alias("spread"),
            pl.Series(zscore).alias("zscore"),
            pl.Series(signal).alias("signal"),
        ]
    )
