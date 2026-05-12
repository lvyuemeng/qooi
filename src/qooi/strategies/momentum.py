"""Time-series momentum strategy — pure directional persistence.

Principle: Buy when past N-bar return is positive; sell when negative.
This exploits trend persistence without requiring complex pullback timing.
ADX gate ensures we only trade in trending environments.

Entry:
  - Long: 6-bar return > 0, close > EMA20, ADX(14) > 20
  - Short: 6-bar return < 0, close < EMA20, ADX(14) > 20
  - Price structure: higher low for longs / lower high for shorts
  - Trend maturity: EMA50/200 direction must persist for ≥10 bars

Exit (tiered, built into state machine):
  - Hard stop at entry_px ± stop_mult * atr (1.5×)
  - Target at entry_px ± target_mult * atr (1.3×), activates trailing
  - Trailing stop at highest/lowest close ± trail_mult * atr (2.0×)
  - Time stop after max_bars_held bars without target
  - Trend reversal (EMA50/200 flip)

Circuit breaker: 2 consecutive losers → suspend until 20-bar high/low break.

Signal column: 1=long, -1=short, 0=flat.
"""

from __future__ import annotations

import polars as pl


def momentum_signal(
    df: pl.DataFrame,
    *,
    mom_bars: int = 6,
    ema_fast: int = 20,
    ema_mid: int = 50,
    ema_slow: int = 200,
    stop_mult: float = 1.5,
    target_mult: float = 1.3,
    trail_mult: float = 2.0,
    max_bars_held: int = 10,
    circuit_breaker_losses: int = 2,
    circuit_breaker_lookback: int = 20,
) -> pl.DataFrame:
    """Stateful momentum signal with tiered exits.

    Returns ``df`` with a ``signal`` column (1 = long, -1 = short, 0 = flat).
    """
    if df.is_empty():
        return df.with_columns(pl.lit(0.0).alias("signal"))

    needed = {
        f"ema_{ema_fast}",
        f"ema_{ema_mid}",
        f"ema_{ema_slow}",
        "close",
        "low",
        "high",
        "atr_14",
        "adx_14",
    }
    missing = needed - set(df.columns)
    if missing:
        return df.with_columns(pl.lit(0.0).alias("signal"))

    close = df["close"].to_list()
    low = df["low"].to_list()
    high = df["high"].to_list()
    atr = df["atr_14"].to_list()
    e20 = df[f"ema_{ema_fast}"].to_list()
    e50 = df[f"ema_{ema_mid}"].to_list()
    e200 = df[f"ema_{ema_slow}"].to_list()
    adx = df["adx_14"].to_list()

    n = len(df)
    signal = [0.0] * n
    pos = 0
    start = max(ema_slow, mom_bars, 20, 14) + 1

    entry_px = 0.0
    atr_at_entry = 0.0
    trail_high = 0.0
    trail_low = 0.0
    target_hit = False
    bars_in_pos = 0
    loss_streak = 0
    suspended_until_long = False
    suspended_until_short = False
    suspension_px = 0.0
    trend_bars = 0

    for i in range(start, n):
        uptrend = e50[i] > 0 and e200[i] > 0 and e50[i] > e200[i]
        downtrend = e50[i] > 0 and e200[i] > 0 and e50[i] < e200[i]

        if uptrend:
            trend_bars = trend_bars + 1 if trend_bars >= 0 else 1
        elif downtrend:
            trend_bars = trend_bars - 1 if trend_bars <= 0 else -1
        else:
            trend_bars = 0
        trend_mature = abs(trend_bars) >= 10

        if suspended_until_long and close[i] > suspension_px:
            suspended_until_long = False
        if suspended_until_short and close[i] < suspension_px:
            suspended_until_short = False

        if close[i] <= 0 or e20[i] <= 0:
            signal[i] = float(pos)
            continue

        adx_ok = adx[i] is not None and adx[i] > 20
        mom_ret = (
            (close[i] / close[i - mom_bars] - 1)
            if i >= mom_bars and close[i - mom_bars] > 0
            else 0.0
        )

        above_ema = close[i] > e20[i]
        below_ema = close[i] < e20[i]

        low_5 = min(low[max(0, i - 5) : i]) if i >= 5 else low[i]
        low_20 = min(low[max(0, i - 20) : i]) if i >= 20 else low[i]
        high_5 = max(high[max(0, i - 5) : i]) if i >= 5 else high[i]
        high_20 = max(high[max(0, i - 20) : i]) if i >= 20 else high[i]

        if pos == 0:
            long_ok = (
                uptrend
                and trend_mature
                and adx_ok
                and mom_ret > 0
                and above_ema
                and low_5 > low_20
                and not suspended_until_long
            )
            if long_ok:
                pos = 1
                entry_px = close[i]
                atr_at_entry = atr[i] if atr[i] > 0 else 50.0
                trail_high = high[i]
                trail_low = low[i]
                target_hit = False
                bars_in_pos = 0

            short_ok = (
                downtrend
                and trend_mature
                and adx_ok
                and mom_ret < 0
                and below_ema
                and high_5 < high_20
                and not suspended_until_short
            )
            if not pos and short_ok:
                pos = -1
                entry_px = close[i]
                atr_at_entry = atr[i] if atr[i] > 0 else 50.0
                trail_high = high[i]
                trail_low = low[i]
                target_hit = False
                bars_in_pos = 0
        else:
            bars_in_pos += 1
            d = 1 if pos > 0 else -1

            if high[i] > trail_high:
                trail_high = high[i]
            if low[i] < trail_low:
                trail_low = low[i]

            stop_px = entry_px - d * stop_mult * atr_at_entry
            target_px = entry_px + d * target_mult * atr_at_entry

            exit_reason = ""

            if not target_hit:
                if d * (stop_px - close[i]) >= 0:
                    exit_reason = "stop"
            else:
                trail_stop = (
                    trail_high - trail_mult * atr_at_entry
                    if pos > 0
                    else trail_low + trail_mult * atr_at_entry
                )
                if pos > 0 and close[i] <= trail_stop:
                    exit_reason = "trailing_stop"
                elif pos < 0 and close[i] >= trail_stop:
                    exit_reason = "trailing_stop"

            if not exit_reason and not target_hit:
                if d * (close[i] - target_px) >= 0:
                    target_hit = True

            if not exit_reason and not target_hit and bars_in_pos >= max_bars_held:
                exit_reason = "time"

            if not exit_reason:
                if pos == 1 and not uptrend:
                    exit_reason = "trend_flip"
                elif pos == -1 and not downtrend:
                    exit_reason = "trend_flip"
                elif e50[i] <= 0 or e200[i] <= 0:
                    exit_reason = "ema_invalid"

            if exit_reason:
                pnl_pct = d * (close[i] / entry_px - 1)
                if pnl_pct < 0:
                    loss_streak += 1
                else:
                    loss_streak = 0

                if loss_streak >= circuit_breaker_losses:
                    if pos > 0:
                        suspended_until_long = True
                        suspension_px = max(high[max(0, i - circuit_breaker_lookback) : i + 1])
                    else:
                        suspended_until_short = True
                        suspension_px = min(low[max(0, i - circuit_breaker_lookback) : i + 1])
                    loss_streak = 0

                pos = 0
                target_hit = False
                bars_in_pos = 0

        signal[i] = float(pos)

    return df.with_columns(pl.Series(signal).alias("signal"))


def momentum_signal_expr(
    mom_bars: int = 6,
    ema_fast: int = 20,
    ema_mid: int = 50,
    ema_slow: int = 200,
    stop_mult: float = 1.5,
    target_mult: float = 1.3,
    trail_mult: float = 2.0,
    max_bars_held: int = 10,
    circuit_breaker_losses: int = 2,
    circuit_breaker_lookback: int = 20,
) -> pl.Expr:
    """Polars Expr wrapper for momentum_signal."""

    needed = [
        f"ema_{ema_fast}",
        f"ema_{ema_mid}",
        f"ema_{ema_slow}",
        "close",
        "low",
        "high",
        "atr_14",
        "adx_14",
    ]

    def _compute(s: pl.Series) -> pl.Series:
        df = s.struct.unnest()
        result = momentum_signal(
            df,
            mom_bars=mom_bars,
            ema_fast=ema_fast,
            ema_mid=ema_mid,
            ema_slow=ema_slow,
            stop_mult=stop_mult,
            target_mult=target_mult,
            trail_mult=trail_mult,
            max_bars_held=max_bars_held,
            circuit_breaker_losses=circuit_breaker_losses,
            circuit_breaker_lookback=circuit_breaker_lookback,
        )
        return result["signal"]

    return pl.struct(needed).map_batches(_compute, return_dtype=pl.Float64).alias("signal")
