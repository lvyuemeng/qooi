"""AssetConfig and SignalResult smoke tests."""

from qooi.core.decide import AssetConfig
from qooi.core.indicators import SignalResult


def test_asset_config_defaults():
    cfg = AssetConfig(symbol="ETH-USDT-SWAP")
    assert cfg.leverage == 2.0
    assert cfg.capital == 500.0


def test_signal_result_construction():
    sr = SignalResult(symbol="X", timeframe="4h", timestamp=1234567890, signal=1.0)
    assert sr.signal == 1.0
    assert sr.flow == 0.0
