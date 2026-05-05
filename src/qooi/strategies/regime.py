"""Interpretable regime classifier — uses multi-timeframe EMA to score market state.

All computations are deterministic and inspectable:
  - regime_score  = (close - EMA_day*4) / ATR  → trend bias
  - momentum_score = multi-horizon price change / ATR → speed
  - confidence     = volume normalization         → conviction

The output is a time-series DataFrame that can be joined with any signal
column for downstream strategy filters.
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


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def add_regime_features(
    df: pl.DataFrame,
    *,
    atr_col: str = "atr_14",
    ema_slow_period: int = 200,
    momentum_bars: tuple[int, int, int] = (6, 24, 96),
) -> pl.DataFrame:
    """Add regime features to a DataFrame.

    Returns the input frame with extra columns:

      regime_score    — trend bias in [-1, 1]
      regime_strength — |regime_score|
      mom_fast / mom_mid / mom_slow — momentum scores in [-1, 1]
      volume_conf     — volume conviction in [0.25, 1.0]
    """
    close = df["close"].to_list()
    vol = df["vol"].fill_nan(0).fill_null(0).to_list()

    atr = (
        df[atr_col].fill_nan(0).fill_null(0).to_list() if atr_col in df.columns else [1.0] * len(df)
    )

    ema_slow = _ema(close, ema_slow_period)

    vol_ma = list(pl.Series(vol).rolling_mean(20)) if len(df) >= 20 else vol[:]

    regime_score = [0.0] * len(df)
    regime_strength = [0.0] * len(df)
    mom_fast_s = [0.0] * len(df)
    mom_mid_s = [0.0] * len(df)
    mom_slow_s = [0.0] * len(df)
    volume_conf = [0.25] * len(df)

    mf, mm, ms = momentum_bars
    min_bars = max(ema_slow_period, ms, 20)
    for i in range(min_bars, len(df)):
        a = atr[i] if atr[i] > 0 else 1.0
        trend = (close[i] - ema_slow[i]) / (a * 3.0)
        regime_score[i] = _clip(trend, -1.0, 1.0)
        regime_strength[i] = abs(regime_score[i])
        mom_fast_s[i] = _clip((close[i] - close[i - mf]) / (a * 2.0), -1.0, 1.0)
        mom_mid_s[i] = _clip((close[i] - close[i - mm]) / (a * 3.0), -1.0, 1.0)
        mom_slow_s[i] = _clip((close[i] - close[i - ms]) / (a * 4.0), -1.0, 1.0)
        vm = vol_ma[i] if i < len(vol_ma) and vol_ma[i] > 0 else 1.0
        volume_conf[i] = _clip(0.5 + (vol[i] / vm - 1.0) * 0.3, 0.25, 1.0)

    return df.with_columns(
        [
            pl.Series(regime_score).alias("regime_score"),
            pl.Series(regime_strength).alias("regime_strength"),
            pl.Series(mom_fast_s).alias("regime_mom_fast"),
            pl.Series(mom_mid_s).alias("regime_mom_mid"),
            pl.Series(mom_slow_s).alias("regime_mom_slow"),
            pl.Series(volume_conf).alias("regime_vol_conf"),
        ]
    )
