"""Flow pipeline — OFI confirmation + regime features + adaptive threshold.

All computations use Polars window expressions where possible; only the
stateful position tracking and the adaptive-threshold gating loop remain
as lightweight Python helpers.
"""

from __future__ import annotations

import polars as pl

# =============================================================================
# 1. OFI flow — native Polars
# =============================================================================


def add_ofi_flow_columns(
    df: pl.DataFrame,
    *,
    flow_window: int = 12,
    atr_col: str = "atr_14",
) -> pl.DataFrame:
    if df.is_empty():
        return df

    close = pl.col("close")
    open_p = pl.col("open")
    vol = pl.col("vol").fill_nan(0).fill_null(0)
    signed = pl.when(close > open_p).then(vol).when(close < open_p).then(-vol).otherwise(0.0)

    net_flow = signed.rolling_sum(flow_window)
    atr = pl.col(atr_col).fill_nan(0).fill_null(0) if atr_col in df.columns else pl.lit(1.0)
    flow_score = (net_flow / (atr * close.clip(1e-9))).clip(-1.0, 1.0)

    return df.with_columns(
        signed.alias("ofi_signed_vol"),
        net_flow.alias("ofi_net_flow"),
        flow_score.fill_null(0).alias("ofi_flow_score"),
    )


def apply_micro_confirmation(
    df: pl.DataFrame,
    signal_col: str = "signal",
    flow_col: str = "ofi_flow_score",
) -> pl.DataFrame:
    if signal_col not in df.columns or flow_col not in df.columns:
        return df

    s = pl.col(signal_col)
    f = pl.col(flow_col)
    direction = s.sign()
    multiplier = pl.when(f.abs() < 0.05).then(0.6).when(direction * f > 0).then(1.0).otherwise(0.4)
    return df.with_columns((s * multiplier).alias(signal_col))


# =============================================================================
# 2. Regime features — native Polars
# =============================================================================


def add_regime_features(
    df: pl.DataFrame,
    *,
    atr_col: str = "atr_14",
    ema_slow_period: int = 200,
    momentum_bars: tuple[int, int, int] = (6, 24, 96),
) -> pl.DataFrame:
    if df.is_empty():
        return df

    close = pl.col("close")
    atr = pl.col(atr_col).fill_nan(0).fill_null(0) if atr_col in df.columns else pl.lit(1.0)
    ema_slow = close.ewm_mean(span=ema_slow_period, min_periods=ema_slow_period)
    vol_ma = pl.col("vol").fill_nan(0).fill_null(0).rolling_mean(20)

    regime_score = ((close - ema_slow) / (atr * 3.0)).clip(-1.0, 1.0)
    regime_strength = regime_score.abs()

    mf, mm, ms = momentum_bars
    mom_fast = ((close - close.shift(mf)) / (atr * 2.0)).clip(-1.0, 1.0)
    mom_mid = ((close - close.shift(mm)) / (atr * 3.0)).clip(-1.0, 1.0)
    mom_slow = ((close - close.shift(ms)) / (atr * 4.0)).clip(-1.0, 1.0)

    vol_conf = (0.5 + (pl.col("vol").fill_nan(0) / vol_ma - 1.0) * 0.3).clip(0.25, 1.0)

    return df.with_columns(
        regime_score.fill_null(0).alias("regime_score"),
        regime_strength.fill_null(0).alias("regime_strength"),
        mom_fast.fill_null(0).alias("regime_mom_fast"),
        mom_mid.fill_null(0).alias("regime_mom_mid"),
        mom_slow.fill_null(0).alias("regime_mom_slow"),
        vol_conf.fill_null(0.25).alias("regime_vol_conf"),
    )


# =============================================================================
# 3. Adaptive threshold — lightweight stateful loop
# =============================================================================


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _thresh(pnl_ema: float, base: float, lo: float, hi: float) -> float:
    if pnl_ema > 0.02:
        return lo
    if pnl_ema > 0.005:
        return base - (pnl_ema - 0.005) / 0.015 * (base - lo)
    if pnl_ema > -0.005:
        return base
    return _clip(base + (abs(pnl_ema) - 0.005) / 0.02 * (hi - base), base, hi)


def _ema_update(prev: float, cur: float, period: int) -> float:
    alpha = 2.0 / (period + 1)
    return alpha * cur + (1 - alpha) * prev


def add_adaptive_threshold(
    df: pl.DataFrame,
    signal_col: str = "signal",
    *,
    lookback: int = 50,
    base_threshold: float = 0.40,
    max_threshold: float = 0.70,
    min_threshold: float = 0.25,
) -> pl.DataFrame:
    """Add ``adaptive_threshold_long`` and ``adaptive_threshold_short`` columns.

    Uses a minimal Python loop because the threshold update depends on
    the trade state (entry/exit) which is inherently sequential.
    """
    if df.is_empty() or signal_col not in df.columns:
        return df.with_columns(
            pl.lit(base_threshold).alias("adaptive_threshold_long"),
            pl.lit(base_threshold).alias("adaptive_threshold_short"),
        )

    close = df["close"].to_list()
    sig = df[signal_col].to_list()
    n = len(df)

    tl = [base_threshold] * n
    ts = [base_threshold] * n
    lpe = 0.0
    spe = 0.0
    active = 0.0
    entry = 0.0

    for i in range(1, n):
        ps = sig[i - 1]
        pnl = 0.0
        if active and ps != active:
            if entry > 0:
                pnl = active * (close[i - 1] / entry - 1)
            active = ps
            if active:
                entry = close[i - 1]
        elif active == 0.0 and ps:
            active = ps
            entry = close[i - 1]

        if active > 0:
            lpe = _ema_update(lpe, pnl or 0.0, lookback)
        elif active < 0:
            spe = _ema_update(spe, pnl or 0.0, lookback)

        tl[i] = _thresh(lpe, base_threshold, min_threshold, max_threshold)
        ts[i] = _thresh(spe, base_threshold, min_threshold, max_threshold)

    return df.with_columns(
        pl.Series(tl).alias("adaptive_threshold_long"),
        pl.Series(ts).alias("adaptive_threshold_short"),
    )


def apply_adaptive_gate(
    df: pl.DataFrame,
    signal_col: str = "signal",
) -> pl.DataFrame:
    if signal_col not in df.columns:
        return df
    sig = pl.col(signal_col)
    tl = (
        pl.col("adaptive_threshold_long")
        if "adaptive_threshold_long" in df.columns
        else pl.lit(0.4)
    )
    ts = (
        pl.col("adaptive_threshold_short")
        if "adaptive_threshold_short" in df.columns
        else pl.lit(0.4)
    )
    return df.with_columns(
        pl.when((sig > 0) & (sig < tl))
        .then(0.0)
        .when((sig < 0) & (sig.abs() < ts))
        .then(0.0)
        .otherwise(sig)
        .alias(signal_col)
    )
