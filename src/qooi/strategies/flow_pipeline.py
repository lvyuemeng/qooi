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
    # Scale-invariant normalization: net flow as fraction of total volume.
    # Replaces the old price-biased (net_flow / (atr × close)) which gave
    # BTC scores 1,450× smaller than ETH — making cross-asset comparison impossible.
    vol_total = signed.abs().rolling_sum(flow_window).clip(1e-9)
    flow_score = (net_flow / vol_total).fill_null(0).clip(-1.0, 1.0)

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
# 3. Regime gate — zero-out signal in strong trends
# =============================================================================


def apply_regime_gate(
    df: pl.DataFrame,
    signal_col: str = "signal",
    regime_col: str = "regime_score",
    max_regime: float = 0.7,
) -> pl.DataFrame:
    """Zero out the signal when regime strength exceeds ``max_regime``.

    In strong trending markets the ensemble signal predicts the wrong
    direction (mean-reversion bias).  This gate skips entries entirely
    when the trend is too strong.
    """
    if signal_col not in df.columns or regime_col not in df.columns:
        return df
    return df.with_columns(
        pl.when(pl.col(regime_col).abs() > max_regime)
        .then(0.0)
        .otherwise(pl.col(signal_col))
        .alias(signal_col)
    )
