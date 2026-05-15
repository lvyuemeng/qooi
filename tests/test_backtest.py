"""Legacy OFI flow parity tests."""

import polars as pl

from qooi.strategies.indicators import (
    add_indicators,
    add_ofi_flow_columns,
    add_regime_features,
    apply_regime_gate,
    compute_flow_pipeline_frame,
)


def _load(name: str) -> pl.DataFrame:
    df = pl.read_parquet(f"data/cache/{name}_4H.parquet")
    if "vol" not in df.columns and "volume" in df.columns:
        df = df.rename({"volume": "vol"})
    return df


def test_flow_pipeline_matches_manual_composition():
    df = _load("BTC_USDT")
    manual = add_indicators(df.clone())
    manual = add_regime_features(manual)
    manual = add_ofi_flow_columns(manual)
    manual = apply_regime_gate(manual, signal_col="ofi_flow_score")
    ofi = pl.col("ofi_flow_score")
    manual = manual.with_columns(
        pl.when(ofi.abs() >= 0.25).then(ofi).otherwise(0.0).alias("signal")
    )
    computed = compute_flow_pipeline_frame(df.clone(), 0.25)
    assert computed["signal"].to_list() == manual["signal"].to_list()
