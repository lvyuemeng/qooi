"""RSI mean-reversion 1H strategy — oversold bounces in confirmed uptrends.

Principle: In a trending market (ADX > 20, EMA50 > EMA200), sharp RSI
oversold readings often reverse quickly. This captures counter-trend entries
that align with the macro direction.

Entry (long only):
  - EMA50 > EMA200, ADX(14) > 20, UTC hour 08-22
  - RSI(14) drops below 30, then crosses back above 25 (bounce confirmed)
  - Confirmation: next bar's RSI does NOT drop back below 20 (bar → bar+1)
  - Price structure: higher low (5-bar low > 20-bar low)

Exit:
  - Hard stop at entry_px - 1.5× ATR
  - Target at entry_px + 1.5× ATR
  - RSI exit at RSI > 50
  - Time stop after 12 bars
  - Trend flip (EMA50 < EMA200)

Circuit breaker: 2 consecutive losers → suspend until 20-bar high break.

Signal column: 1=long, 0=flat.
"""

from __future__ import annotations

import polars as pl


def rsi_reversion_signal(
    df: pl.DataFrame,
    *,
    rsi_period: int = 14,
    rsi_oversold: float = 30.0,
    rsi_bounce: float = 25.0,
    rsi_exit: float = 50.0,
    rsi_confirmation: float = 20.0,
    ema_mid: int = 50,
    ema_slow: int = 200,
    stop_mult: float = 1.5,
    target_mult: float = 1.5,
    max_bars_held: int = 12,
    circuit_breaker_losses: int = 2,
    circuit_breaker_lookback: int = 20,
) -> pl.DataFrame:
    """Stateful RSI mean-reversion signal with tiered exits.

    Returns ``df`` with a ``signal`` column (1 = long, 0 = flat).
    """
    if df.is_empty():
        return df.with_columns(pl.lit(0.0).alias("signal"))

    core_needed = {
        f"ema_{ema_mid}",
        f"ema_{ema_slow}",
        f"rsi_{rsi_period}",
        "close",
        "low",
        "high",
        "atr_14",
        "adx_14",
        "timestamp",
    }
    missing = core_needed - set(df.columns)
    if missing:
        return df.with_columns(pl.lit(0.0).alias("signal"))

    close = df["close"].to_list()
    low = df["low"].to_list()
    high = df["high"].to_list()
    atr = df["atr_14"].to_list()
    e50 = df[f"ema_{ema_mid}"].to_list()
    e200 = df[f"ema_{ema_slow}"].to_list()
    rsi = df[f"rsi_{rsi_period}"].to_list()
    adx = df["adx_14"].to_list()
    ts_col = df["timestamp"].to_list()
    has_datetime = "datetime" in df.columns
    dt_col = df["datetime"].to_list() if has_datetime else None

    n = len(df)
    signal = [0.0] * n
    pos = 0
    start = max(ema_slow, rsi_period, 20, 14) + 1

    entry_px = 0.0
    atr_at_entry = 0.0
    bars_in_pos = 0
    loss_streak = 0
    suspended = False
    suspension_px = 0.0

    for i in range(start, n):
        uptrend = e50[i] > 0 and e200[i] > 0 and e50[i] > e200[i]
        adx_ok = adx[i] is not None and adx[i] > 20

        if suspended and close[i] > suspension_px:
            suspended = False

        if close[i] <= 0:
            signal[i] = float(pos)
            continue

        if dt_col is not None and i < len(dt_col):
            hv = dt_col[i]
            hour_utc = hv.hour if hv is not None else 12
        else:
            hour_utc = (ts_col[i] // 3600000) % 24 if ts_col[i] else 12
        in_session = 8 <= int(hour_utc) <= 22

        low_5 = min(low[max(0, i - 5) : i]) if i >= 5 else low[i]
        low_20 = min(low[max(0, i - 20) : i]) if i >= 20 else low[i]
        structure_ok = low_5 > low_20

        if pos == 0:
            rsi_cross = rsi[i] > rsi_bounce and rsi[i - 1] <= rsi_oversold
            rsi_bounce_held = i >= 2 and rsi[i] > rsi_confirmation and rsi[i - 1] > rsi_confirmation

            long_ok = (
                uptrend
                and adx_ok
                and in_session
                and rsi_cross
                and rsi_bounce_held
                and structure_ok
                and not suspended
            )
            if long_ok:
                pos = 1
                entry_px = close[i]
                atr_at_entry = atr[i] if atr[i] > 0 else 50.0
                bars_in_pos = 0
        else:
            bars_in_pos += 1
            stop_px = entry_px - stop_mult * atr_at_entry
            target_px = entry_px + target_mult * atr_at_entry

            exit_reason = ""

            if close[i] <= stop_px:
                exit_reason = "stop"
            elif close[i] >= target_px:
                exit_reason = "target"
            elif rsi[i] > rsi_exit:
                exit_reason = "rsi_exit"
            elif bars_in_pos >= max_bars_held:
                exit_reason = "time"
            elif not uptrend:
                exit_reason = "trend_flip"

            if exit_reason:
                pnl_pct = close[i] / entry_px - 1
                if pnl_pct < 0:
                    loss_streak += 1
                else:
                    loss_streak = 0

                if loss_streak >= circuit_breaker_losses:
                    suspended = True
                    suspension_px = max(high[max(0, i - circuit_breaker_lookback) : i + 1])
                    loss_streak = 0

                pos = 0
                bars_in_pos = 0

        signal[i] = float(pos)

    return df.with_columns(pl.Series(signal).alias("signal"))


def rsi_reversion_signal_expr(
    rsi_period: int = 14,
    rsi_oversold: float = 30.0,
    rsi_bounce: float = 25.0,
    rsi_exit: float = 50.0,
    rsi_confirmation: float = 20.0,
    ema_mid: int = 50,
    ema_slow: int = 200,
    stop_mult: float = 1.5,
    target_mult: float = 1.5,
    max_bars_held: int = 12,
    circuit_breaker_losses: int = 2,
    circuit_breaker_lookback: int = 20,
) -> pl.Expr:
    """Polars Expr wrapper for rsi_reversion_signal."""

    needed = [
        f"ema_{ema_mid}",
        f"ema_{ema_slow}",
        f"rsi_{rsi_period}",
        "close",
        "low",
        "high",
        "atr_14",
        "adx_14",
        "timestamp",
    ]

    def _compute(s: pl.Series) -> pl.Series:
        df = s.struct.unnest()
        result = rsi_reversion_signal(
            df,
            rsi_period=rsi_period,
            rsi_oversold=rsi_oversold,
            rsi_bounce=rsi_bounce,
            rsi_exit=rsi_exit,
            rsi_confirmation=rsi_confirmation,
            ema_mid=ema_mid,
            ema_slow=ema_slow,
            stop_mult=stop_mult,
            target_mult=target_mult,
            max_bars_held=max_bars_held,
            circuit_breaker_losses=circuit_breaker_losses,
            circuit_breaker_lookback=circuit_breaker_lookback,
        )
        return result["signal"]

    return pl.struct(needed).map_batches(_compute, return_dtype=pl.Float64).alias("signal")
