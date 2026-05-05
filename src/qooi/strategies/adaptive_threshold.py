"""Adaptive entry threshold — rolling Sharpe-weighted signal gating.

Interpretable, deterministic, self-correcting: when long signals have
been losing recently, make future long entries stricter.
"""

from __future__ import annotations

import polars as pl


def _ema(arr: list[float], period: int) -> list[float]:
    result = [0.0] * len(arr)
    alpha = 2.0 / (period + 1)
    result[0] = arr[0] if arr else 0.0
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result


def add_adaptive_threshold(
    df: pl.DataFrame,
    signal_col: str = "signal",
    *,
    lookback: int = 50,
    base_threshold: float = 0.40,
    max_threshold: float = 0.70,
    min_threshold: float = 0.25,
    max_hold_bars: int = 6,
) -> pl.DataFrame:
    """Adaptive entry gating using rolling directional Sharpe.

    For every bar, we track:
      - rolling P&L per long entry
      - rolling P&L per short entry

    When long P&L is negative → raise long entry threshold (harder to enter).
    When short P&L is negative → raise short entry threshold.

    This is completely interpretable — the ``adaptive_threshold_long``
    and ``adaptive_threshold_short`` columns can be plotted directly.
    """
    if df.is_empty() or signal_col not in df.columns:
        return df.with_columns(
            pl.lit(base_threshold).alias("adaptive_threshold_long"),
            pl.lit(base_threshold).alias("adaptive_threshold_short"),
        )

    close = df["close"].to_list()
    sig = df[signal_col].to_list()
    n = len(df)

    long_pnl_ema = [0.0] * n
    short_pnl_ema = [0.0] * n
    threshold_long = [base_threshold] * n
    threshold_short = [base_threshold] * n

    active = 0.0
    entry_price = 0.0

    for i in range(1, n):
        prev_sig = sig[i - 1]
        pnl = 0.0

        if active != 0.0 and prev_sig != active:
            # Close trade
            if entry_price > 0:
                pnl = active * (close[i - 1] / entry_price - 1)
            active = prev_sig
            if active:
                entry_price = close[i - 1]
        elif active == 0.0 and prev_sig != 0:
            active = prev_sig
            entry_price = close[i - 1]

        # Update rolling P&L
        if i > lookback:
            long_pnl_ema[i] = long_pnl_ema[i - 1]
            short_pnl_ema[i] = short_pnl_ema[i - 1]

        if pnl != 0:
            if active > 0:
                long_pnl_ema[i] = pnl
            else:
                short_pnl_ema[i] = pnl
        else:
            alpha = 2.0 / (lookback + 1)
            long_pnl_ema[i] = (
                alpha * (long_pnl_ema[i - min(max_hold_bars, i)] or 0)
                + (1 - alpha) * long_pnl_ema[i - 1]
            )
            short_pnl_ema[i] = (
                alpha * (short_pnl_ema[i - min(max_hold_bars, i)] or 0)
                + (1 - alpha) * short_pnl_ema[i - 1]
            )

        # Map P&L to threshold
        threshold_long[i] = _ema_to_threshold(
            long_pnl_ema[i], base_threshold, min_threshold, max_threshold
        )
        threshold_short[i] = _ema_to_threshold(
            short_pnl_ema[i], base_threshold, min_threshold, max_threshold
        )

    return df.with_columns(
        [
            pl.Series(threshold_long).alias("adaptive_threshold_long"),
            pl.Series(threshold_short).alias("adaptive_threshold_short"),
        ]
    )


def _ema_to_threshold(pnl_ema: float, base: float, min_val: float, max_val: float) -> float:
    """Map rolling P&L to an entry threshold.

    Positive P&L EMA → lower threshold (easier to enter).
    Negative P&L EMA → higher threshold (harder to enter).
    """
    if pnl_ema > 0.02:
        return min_val  # performing well — be aggressive
    if pnl_ema > 0.005:
        return base - (pnl_ema - 0.005) / 0.015 * (base - min_val)
    if pnl_ema > -0.005:
        return base  # neutral
    # Losing — tighten
    return _clip(base + (abs(pnl_ema) - 0.005) / 0.02 * (max_val - base), base, max_val)


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def apply_adaptive_gate(
    df: pl.DataFrame,
    signal_col: str = "signal",
) -> pl.DataFrame:
    """Filter signal column using adaptive thresholds.

    Requires columns: ``adaptive_threshold_long``, ``adaptive_threshold_short``
    (produced by ``add_adaptive_threshold``).

    For each bar where signal > 0: check if |signal| > threshold_long.
    For each bar where signal < 0: check if |signal| > threshold_short.
    If below threshold → set signal to 0.
    """
    if signal_col not in df.columns:
        return df

    sig = df[signal_col].to_list()
    tl = (
        df["adaptive_threshold_long"].to_list()
        if "adaptive_threshold_long" in df.columns
        else [0.4] * len(df)
    )
    ts = (
        df["adaptive_threshold_short"].to_list()
        if "adaptive_threshold_short" in df.columns
        else [0.4] * len(df)
    )

    new_sig = [0.0] * len(df)
    for i in range(len(df)):
        s = sig[i]
        if s > 0 and abs(s) < tl[i]:
            new_sig[i] = 0.0
        elif s < 0 and abs(s) < ts[i]:
            new_sig[i] = 0.0
        else:
            new_sig[i] = s

    return df.with_columns(pl.Series(new_sig).alias(signal_col))
