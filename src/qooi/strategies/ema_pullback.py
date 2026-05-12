"""EMA trend + pullback + RSI crossing signal.

Logic:
  1. Trend: EMA50 > EMA200 → uptrend; EMA50 < EMA200 → downtrend
  2. Pullback: price pulls back near EMA20 within the trend direction
  3. RSI: exit oversold (cross above 30) in uptrend pullback; exit
     overbought (cross below 70) in downtrend pullback

Returns a column ``signal`` with values -1, 0, or 1 (discrete entry signals).
Exit signals are triggered by trend reversal (EMA50/200 flip).
"""

from __future__ import annotations

import polars as pl


def ema_pullback_signal_expr(
    ema_fast: int = 20,
    ema_mid: int = 50,
    ema_slow: int = 200,
    rsi_period: int = 14,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
    pullback_pct: float = 0.02,  # max distance from EMA20 as fraction of close
) -> pl.Expr:
    """Pure Polars Expr: EMA50/200 trend filter + EMA20 pullback + RSI cross.

    Entries:
      - Long: EMA50 > EMA200, price within pullback_pct of EMA20, RSI crosses above oversold
      - Short: EMA50 < EMA200, price within pullback_pct of EMA20, RSI crosses below overbought

    Flips:
      - Exit/Reverse on EMA50 crossing EMA200 (trend reversal)

    State-machine behaviour:
      Once a long/short entry fires, the signal stays at 1/-1 until:
        1. EMA50 crosses EMA200 (trend reversal) → reset to 0
        2. EMAs become invalid (≤0) → reset to 0
      Trend-reversal exits clear the signal immediately — the next entry
      will only fire on a fresh RSI cross.

    Returns a pl.Expr suitable for ``Backtest(signal_expr=…)``.
    This is NOT a pure expression — it delegates to ema_pullback_signal for
    correct state-machine behaviour, avoiding the forward_fill bug.
    """

    needed = [
        f"ema_{ema_fast}",
        f"ema_{ema_mid}",
        f"ema_{ema_slow}",
        f"rsi_{rsi_period}",
        "close",
    ]

    def _compute_signal(s: pl.Series) -> pl.Series:
        df = s.struct.unnest()
        result = ema_pullback_signal(
            df,
            ema_fast=ema_fast,
            ema_mid=ema_mid,
            ema_slow=ema_slow,
            rsi_period=rsi_period,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
            pullback_pct=pullback_pct,
        )
        return result["signal"]

    return pl.struct(needed).map_batches(_compute_signal, return_dtype=pl.Float64).alias("signal")


def ema_pullback_signal(
    df: pl.DataFrame,
    ema_fast: int = 20,
    ema_mid: int = 50,
    ema_slow: int = 200,
    rsi_period: int = 14,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
    pullback_pct: float = 0.02,
) -> pl.DataFrame:
    """Stateful EMA pullback signal on a DataFrame.

    Returns ``df`` with a ``signal`` column (1 = long, -1 = short, 0 = flat).
    Entry logic and trend-reversal exit are identical to ``ema_pullback_signal_expr``,
    but this variant uses a Python state-machine loop for simpler debugging.

    The state machine:
      - Enters when all conditions align and no position is active.
      - Exits only when EMA50/200 flip (trend reversal), NOT on RSI extremes.
    """
    if df.is_empty():
        return df.with_columns(pl.lit(0.0).alias("signal"))

    needed = {
        f"ema_{ema_fast}",
        f"ema_{ema_mid}",
        f"ema_{ema_slow}",
        f"rsi_{rsi_period}",
        "close",
    }
    missing = needed - set(df.columns)
    if missing:
        return df.with_columns(pl.lit(0.0).alias("signal"))

    close = df["close"].to_list()
    e20 = df[f"ema_{ema_fast}"].to_list()
    e50 = df[f"ema_{ema_mid}"].to_list()
    e200 = df[f"ema_{ema_slow}"].to_list()
    rsi = df[f"rsi_{rsi_period}"].to_list()

    n = len(df)
    signal = [0.0] * n
    pos = 0  # 1=long, -1=short, 0=flat

    for i in range(max(ema_slow, rsi_period) + 1, n):
        uptrend = e50[i] > 0 and e200[i] > 0 and e50[i] > e200[i]
        downtrend = e50[i] > 0 and e200[i] > 0 and e50[i] < e200[i]
        near = close[i] > 0 and abs(close[i] - e20[i]) <= pullback_pct * close[i]

        if pos == 0:
            if uptrend and near:
                rsi_cross = rsi[i] > rsi_oversold and rsi[i - 1] <= rsi_oversold
                if rsi_cross:
                    pos = 1
            elif downtrend and near:
                rsi_cross = rsi[i] < rsi_overbought and rsi[i - 1] >= rsi_overbought
                if rsi_cross:
                    pos = -1
        else:
            # Check trend reversal
            if pos == 1 and e50[i] > 0 and e200[i] > 0 and e50[i] < e200[i]:
                pos = 0
            elif pos == -1 and e50[i] > 0 and e200[i] > 0 and e50[i] > e200[i]:
                pos = 0
            # Edge case: EMAs go invalid
            elif e50[i] <= 0 or e200[i] <= 0:
                pos = 0

        signal[i] = float(pos)

    return df.with_columns(pl.Series(signal).alias("signal"))
