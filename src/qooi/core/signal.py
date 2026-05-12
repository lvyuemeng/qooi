"""Shared signal pipeline — identical for live trading and backtesting.

Live:     signal = compute(symbol, timeframe, threshold)
Backtest: df = compute_all(df, threshold) → adds "signal" column

Both paths call the same subroutines in the same order.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from qooi.exchange.indicator import add_indicators
from qooi.strategies.flow_pipeline import (
    add_ofi_flow_columns,
    add_regime_features,
    apply_regime_gate,
)


@dataclass
class SignalResult:
    symbol: str
    timeframe: str
    timestamp: int
    signal: float
    flow: float = 0.0
    threshold: float = 0.0
    atr: float = 0.0
    regime_strength: float = 0.0
    mom_fast: float = 0.0
    vol_conf: float = 0.5


def compute_dataframe(df: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """Full pipeline on cached data — for backtesting."""
    df = add_indicators(df)
    df = add_regime_features(df)
    df = add_ofi_flow_columns(df)
    df = apply_regime_gate(df, signal_col="ofi_flow_score")
    ofi = pl.col("ofi_flow_score")
    sig = pl.when(ofi.abs() >= threshold).then(ofi).otherwise(0.0)
    return df.with_columns(sig.alias("signal"))


def compute_single(
    symbol: str, timeframe: str, threshold: float, *, limit: int = 500
) -> SignalResult | None:
    """Compute signal for live trading — one bar, from OKX candles."""
    from qooi.exchange.market import MarketData

    md = MarketData("okx")
    df = md.candles(symbol, timeframe=timeframe, limit=limit, cache=True)
    if df.is_empty():
        return None
    df = add_indicators(df)
    df = add_regime_features(df)
    df = add_ofi_flow_columns(df)
    df = apply_regime_gate(df, signal_col="ofi_flow_score")

    ofi = float(df["ofi_flow_score"][-1])
    sig = round(ofi, 4) if abs(ofi) >= threshold else 0.0
    return SignalResult(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=int(df["timestamp"][-1]),
        signal=sig,
        flow=round(ofi, 4),
        threshold=threshold,
        atr=round(float(df["atr_14"][-1] or 0.0), 2),
        regime_strength=round(float(df["regime_strength"][-1] or 0.0), 3),
        mom_fast=round(float(df["regime_mom_fast"][-1] or 0.0), 3),
        vol_conf=round(float(df["regime_vol_conf"][-1] or 0.5), 3),
    )
