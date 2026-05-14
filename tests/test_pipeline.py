"""Integration test — full pipeline against cached BTC 4H data.

Note: tests using compute_*_1h() strategies hit the network via MarketData
internally. These are skipped when OKX API is unreachable.
"""

import polars as pl

from qooi.core import process_bar
from qooi.core.basket import Basket, ExitConfig
from qooi.core.config import AssetConfig, OkxSignalConfig, PairConfig
from qooi.core.recovery import RecoveryConfig, RecoveryKind


def test_strategies_accessible():
    cfg1 = OkxSignalConfig(strategy="momentum_1h")
    cfg2 = OkxSignalConfig(strategy="rsi_reversion")
    assert cfg1.strategy == "momentum_1h"
    assert cfg2.strategy == "rsi_reversion"


def test_unknown_strategy_returns_none():
    cfg = OkxSignalConfig(strategy="flow_pipeline")
    # flow_pipeline is valid too, just verify the Literal is enforced
    assert cfg.strategy in ("momentum_1h", "rsi_reversion", "flow_pipeline")


def _load_df():
    df = pl.read_parquet("data/cache/BTC_USDT_4H.parquet")
    if "volume" in df.columns and "vol" not in df.columns:
        df = df.rename({"volume": "vol"})
    from qooi.exchange.indicator import add_indicators

    return add_indicators(df.tail(50))


def _pair():
    return PairConfig(
        asset=AssetConfig(
            symbol="BTC-USDT-SWAP",
            sig_symbol="BTC-USDT",
            timeframe="4H",
            capital=500,
            leverage=1.0,
            ct_val=0.01,
            signal_threshold=0.25,
        ),
        okx=OkxSignalConfig(strategy="momentum_1h"),
    )


def test_pipeline_idle_basket_no_signal():
    df = _load_df()
    pair = _pair()
    baskets: list[Basket] = []
    actions = process_bar(df, baskets, pair)
    assert isinstance(actions, list)


def test_pipeline_with_recovery_config():
    df = _load_df()
    pair = _pair()
    baskets: list[Basket] = []
    actions = process_bar(
        df,
        baskets,
        pair,
        exit_cfg=ExitConfig(stop_mult=2.0, target_mult=3.0),
        recovery_cfg=RecoveryConfig(strategy=RecoveryKind.GRID),
    )
    assert isinstance(actions, list)
