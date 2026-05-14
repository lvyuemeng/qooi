"""Composable strategy tests."""

import polars as pl
import pytest

from qooi.core.config import OkxSignalConfig
from qooi.strategies import compute_signal_frame, momentum_burst_spec, rsi_bounce_reversion_spec


def _df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": list(range(80)),
            "open": [100.0] * 80,
            "high": [101.0] * 80,
            "low": [99.0] * 80,
            "close": [100.0 + i * 0.01 for i in range(80)],
            "vol": [100.0] * 80,
            "atr_14": [1.0] * 80,
            "ema_50": [101.0] * 80,
            "ema_200": [100.0] * 80,
            "adx_14": [25.0] * 80,
            "rsi_14": [50.0] * 80,
        }
    )


def test_composed_strategy_adds_signal_column():
    df = compute_signal_frame(_df(), momentum_burst_spec())
    assert "signal" in df.columns


def test_composed_strategy_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        compute_signal_frame(_df(), "unknown")


@pytest.mark.parametrize("old_name", ["momentum_1h", "rsi_reversion"])
def test_composed_strategy_rejects_old_names(old_name):
    with pytest.raises(ValueError):
        compute_signal_frame(_df(), old_name)


def test_rsi_bounce_reversion_adds_long_only_signal_column():
    df = compute_signal_frame(_df(), rsi_bounce_reversion_spec())
    assert "signal" in df.columns
    assert set(df["signal"].drop_nulls().unique().to_list()) <= {0.0, 1.0}


def test_compute_signal_frame_precomputes_indicators_from_raw_ohlcv():
    raw = _df().drop("atr_14", "ema_50", "ema_200", "adx_14", "rsi_14")
    df = compute_signal_frame(raw, "momentum_burst")
    assert "signal" in df.columns
    assert "atr_14" in df.columns
    assert "ema_200" in df.columns


def test_okx_signal_config_is_data_only():
    assert not hasattr(OkxSignalConfig(), "compute")
    assert not hasattr(OkxSignalConfig(), "strategy")
