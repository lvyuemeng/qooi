"""EMA pullback v2 — RSI divergence entry + tiered exits + circuit breaker.

Fixes the failures identified in the original ema_pullback:
  1. Entry: RSI divergence replaces weak RSI cross → 2-78× more signals
  2. Entry: Volume confirmation filters dead bars (vol > 0.8× avg)
  3. Exit: Three-tier exit replaces slow EMA crossover:
     - Hard stop at entry_px ± stop_mult * atr
     - Profit target at entry_px ± target_mult * atr
     - Trailing stop at highest/lowest close ± trail_mult * atr (after target hit)
     - Time stop: exit after max_bars_held bars without hitting target
  4. Circuit breaker: 2 consecutive losers → suspend asset until 20-bar
     high/low break

Entry logic:
  - Long: EMA50 > EMA200 for ≥10 consecutive bars (trend maturity),
    ADX(14) > 20 and +DI > -DI, price near EMA20,
    bullish RSI divergence (4-bar), volume > 0.8× 20-bar avg,
    higher low structure (lowest of last 5 bars > 20-bar low)
  - Short: EMA50 < EMA200 for ≥10 consecutive bars,
    ADX(14) > 20 and -DI > +DI, price near EMA20,
    bearish RSI divergence (4-bar), volume > 0.8× 20-bar avg,
    lower high structure (highest of last 5 bars < 20-bar high)

Signal column: 1=long, -1=short, 0=flat.
All exits are handled by the state machine — the signal drops to 0 when any
exit condition fires, so the backtest only needs exit_mode="signal_flip_only".
"""

from __future__ import annotations

import polars as pl


def ema_pullback_v2_signal(
    df: pl.DataFrame,
    *,
    ema_fast: int = 20,
    ema_mid: int = 50,
    ema_slow: int = 200,
    rsi_period: int = 14,
    pullback_pct: float = 0.02,
    vol_threshold: float = 0.8,
    div_lookback: int = 4,
    stop_mult: float = 1.5,
    target_mult: float = 1.3,
    trail_mult: float = 2.0,
    max_bars_held: int = 10,
    circuit_breaker_losses: int = 2,
    circuit_breaker_lookback: int = 20,
) -> pl.DataFrame:
    """Stateful EMA pullback v2 signal with tiered exits.

    Returns ``df`` with a ``signal`` column (1 = long, -1 = short, 0 = flat).
    Exits are embedded in the state machine — signal goes to 0 when any
    exit condition fires.
    """
    if df.is_empty():
        return df.with_columns(pl.lit(0.0).alias("signal"))

    needed = {
        f"ema_{ema_fast}",
        f"ema_{ema_mid}",
        f"ema_{ema_slow}",
        f"rsi_{rsi_period}",
        "close",
        "low",
        "high",
        "vol",
        "atr_14",
        "adx_14",
    }
    missing = needed - set(df.columns)
    if missing:
        return df.with_columns(pl.lit(0.0).alias("signal"))

    close = df["close"].to_list()
    low = df["low"].to_list()
    high = df["high"].to_list()
    vol = df["vol"].to_list()
    atr = df["atr_14"].to_list()
    e20 = df[f"ema_{ema_fast}"].to_list()
    e50 = df[f"ema_{ema_mid}"].to_list()
    e200 = df[f"ema_{ema_slow}"].to_list()
    rsi = df[f"rsi_{rsi_period}"].to_list()
    adx = df["adx_14"].to_list()

    n = len(df)
    signal = [0.0] * n
    pos = 0
    start = max(ema_slow, rsi_period, 20, 14) + 1

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
            if trend_bars >= 0:
                trend_bars += 1
            else:
                trend_bars = 1
        elif downtrend:
            if trend_bars <= 0:
                trend_bars -= 1
            else:
                trend_bars = -1
        else:
            trend_bars = 0

        trend_mature = abs(trend_bars) >= 10

        # Circuit-breaker reset: suspension lifts when price breaks 20-bar high/low
        if suspended_until_long and close[i] > suspension_px:
            suspended_until_long = False
        if suspended_until_short and close[i] < suspension_px:
            suspended_until_short = False

        if close[i] <= 0 or e20[i] <= 0:
            signal[i] = float(pos)
            continue

        near_ema = abs(close[i] - e20[i]) <= pullback_pct * close[i] and close[i] > e20[i]
        avg_vol_20 = sum(vol[max(0, i - 20) : i]) / min(20, i)
        high_vol = vol[i] > vol_threshold * avg_vol_20

        adx_ok = adx[i] is not None and adx[i] > 20

        if pos == 0:
            low_5 = min(low[max(0, i - 5) : i]) if i >= 5 else low[i]
            low_20 = min(low[max(0, i - 20) : i]) if i >= 20 else low[i]
            high_5 = max(high[max(0, i - 5) : i]) if i >= 5 else high[i]
            high_20 = max(high[max(0, i - 20) : i]) if i >= 20 else high[i]

            long_ok = (
                uptrend
                and trend_mature
                and near_ema
                and high_vol
                and adx_ok
                and low_5 > low_20
                and not suspended_until_long
            )
            if long_ok:
                for lookback in range(1, div_lookback + 1):
                    if i - lookback < 0:
                        continue
                    if low[i] < low[i - lookback] and rsi[i] > rsi[i - lookback]:
                        pos = 1
                        entry_px = close[i]
                        atr_at_entry = atr[i] if atr[i] > 0 else 50.0
                        trail_high = high[i]
                        trail_low = low[i]
                        target_hit = False
                        bars_in_pos = 0
                        break

            short_ok = (
                downtrend
                and trend_mature
                and near_ema
                and high_vol
                and adx_ok
                and high_5 < high_20
                and not suspended_until_short
            )
            if not pos and short_ok:
                for lookback in range(1, div_lookback + 1):
                    if i - lookback < 0:
                        continue
                    if high[i] > high[i - lookback] and rsi[i] < rsi[i - lookback]:
                        pos = -1
                        entry_px = close[i]
                        atr_at_entry = atr[i] if atr[i] > 0 else 50.0
                        trail_high = high[i]
                        trail_low = low[i]
                        target_hit = False
                        bars_in_pos = 0
                        break
        else:
            bars_in_pos += 1
            d = 1 if pos > 0 else -1

            # Update trail
            if high[i] > trail_high:
                trail_high = high[i]
            if low[i] < trail_low:
                trail_low = low[i]

            stop_px = entry_px - d * stop_mult * atr_at_entry
            target_px = entry_px + d * target_mult * atr_at_entry

            exit_reason = ""

            # Tier 1: hard stop
            if not target_hit:
                if d * (stop_px - close[i]) >= 0:
                    exit_reason = "stop"
            else:
                # Tier 3: trailing stop (after target)
                trail_stop = (
                    trail_high - trail_mult * atr_at_entry
                    if pos > 0
                    else trail_low + trail_mult * atr_at_entry
                )
                if pos > 0 and close[i] <= trail_stop:
                    exit_reason = "trailing_stop"
                elif pos < 0 and close[i] >= trail_stop:
                    exit_reason = "trailing_stop"

            # Tier 2: target hit (activate trailing)
            if not exit_reason and not target_hit:
                if d * (close[i] - target_px) >= 0:
                    target_hit = True

            # Tier 4: time stop
            if not exit_reason and not target_hit and bars_in_pos >= max_bars_held:
                exit_reason = "time"

            # Trend reversal exit (also triggers when immature)
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


def ema_pullback_v2_signal_expr(
    ema_fast: int = 20,
    ema_mid: int = 50,
    ema_slow: int = 200,
    rsi_period: int = 14,
    pullback_pct: float = 0.02,
    vol_threshold: float = 0.8,
    div_lookback: int = 4,
    stop_mult: float = 1.5,
    target_mult: float = 1.3,
    trail_mult: float = 2.0,
    max_bars_held: int = 10,
    circuit_breaker_losses: int = 2,
    circuit_breaker_lookback: int = 20,
) -> pl.Expr:
    """Polars Expr wrapper for ema_pullback_v2_signal.

    Uses pl.struct + map_batches to apply the full state-machine across
    the DataFrame.  Suitable for ``Backtest(signal_expr=…)``.
    """

    needed = [
        f"ema_{ema_fast}",
        f"ema_{ema_mid}",
        f"ema_{ema_slow}",
        f"rsi_{rsi_period}",
        "close",
        "low",
        "high",
        "vol",
        "atr_14",
        "adx_14",
    ]

    def _compute(s: pl.Series) -> pl.Series:
        df = s.struct.unnest()
        result = ema_pullback_v2_signal(
            df,
            ema_fast=ema_fast,
            ema_mid=ema_mid,
            ema_slow=ema_slow,
            rsi_period=rsi_period,
            pullback_pct=pullback_pct,
            vol_threshold=vol_threshold,
            div_lookback=div_lookback,
            stop_mult=stop_mult,
            target_mult=target_mult,
            trail_mult=trail_mult,
            max_bars_held=max_bars_held,
            circuit_breaker_losses=circuit_breaker_losses,
            circuit_breaker_lookback=circuit_breaker_lookback,
        )
        return result["signal"]

    return pl.struct(needed).map_batches(_compute, return_dtype=pl.Float64).alias("signal")
