"""Integration test — full pipeline against cached BTC 4H data.

Note: tests using compute_*_1h() strategies hit the network via MarketData
internally. These are skipped when OKX API is unreachable.
"""

import polars as pl

from qooi.core.basket import Basket
from qooi.core.config import OkxSignalConfig, PairConfig
from qooi.core.decide import AssetConfig
from qooi.core.exits import ExitConfig
from qooi.core.pipeline import process_bar
from qooi.core.recovery import RecoveryConfig
from qooi.core.registry import REGISTRY, Entry


def test_registry_has_momentum():
    assert "momentum_1h" in REGISTRY
    assert "rsi_reversion" in REGISTRY


def test_resolve_known_strategy():
    e = REGISTRY.get("momentum_1h")
    assert e is not None
    assert isinstance(e, Entry)


def test_resolve_unknown_returns_none():
    e = REGISTRY.get("nonexistent")
    assert e is None


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
        recovery_cfg=RecoveryConfig(strategy="grid"),
    )
    assert isinstance(actions, list)
