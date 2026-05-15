"""Composable strategy tests."""

import polars as pl
import pytest

from qooi.core.config import OkxSignalConfig
from qooi.strategies import (
    adaptive_zscore_mean_reversion_spec,
    compute_signal_frame,
    momentum_burst_spec,
    robust_zscore_mean_reversion_spec,
    rsi_bounce_reversion_spec,
    rsi_macd_trend_spec,
    zscore_mean_reversion_spec,
)
from qooi.strategies.features import (
    add_dynamic_z_blend,
    add_ewma_z_score,
    add_garch_like_volatility,
    add_robust_z_score,
    add_volatility_regime,
    add_z_score,
)
from qooi.strategies.specs import HoldPolicy, SignalRule, StrategySpec, apply_strategy_spec


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


def test_composed_strategy_adds_required_signal_module_columns():
    df = compute_signal_frame(_df(), momentum_burst_spec())
    assert {
        "raw_entry_signal",
        "entry_signal",
        "signal_strength",
        "signal_id",
        "position_signal",
        "exit_signal",
        "signal",
    } <= set(df.columns)


def test_composed_strategy_rejects_unknown_strategy():
    with pytest.raises(TypeError):
        compute_signal_frame(_df(), "unknown")


@pytest.mark.parametrize("old_name", ["momentum_1h", "rsi_reversion"])
def test_composed_strategy_rejects_old_names(old_name):
    with pytest.raises(TypeError):
        compute_signal_frame(_df(), old_name)


def test_rsi_bounce_reversion_adds_long_only_signal_column():
    df = compute_signal_frame(_df(), rsi_bounce_reversion_spec())
    assert "signal" in df.columns
    assert set(df["signal"].drop_nulls().unique().to_list()) <= {0.0, 1.0}


@pytest.mark.parametrize(
    "spec",
    [
        zscore_mean_reversion_spec(),
        adaptive_zscore_mean_reversion_spec(),
        robust_zscore_mean_reversion_spec(),
        rsi_macd_trend_spec(),
    ],
)
def test_signal_module_specs_compute_explicit_outputs(spec):
    df = compute_signal_frame(_df(), spec)

    assert "signal_strength" in df.columns
    assert "signal_id" in df.columns
    assert "exit_signal" in df.columns


def test_dynamic_stat_features_add_explicit_columns_without_division_errors():
    df = pl.DataFrame(
        {
            "timestamp": list(range(220)),
            "open": [100.0] * 220,
            "high": [101.0] * 220,
            "low": [99.0] * 220,
            "close": [100.0] * 220,
            "vol": [100.0] * 220,
        }
    )
    for feature in (
        add_z_score(20),
        add_ewma_z_score(24),
        add_robust_z_score(30),
        add_volatility_regime(short_span=12, long_span=36),
        add_garch_like_volatility(),
        add_dynamic_z_blend(),
    ):
        df = feature(df)

    assert {
        "close_z_score",
        "ewma_z_score",
        "robust_z_score",
        "realized_vol_short",
        "realized_vol_long",
        "volatility_ratio",
        "volatility_regime",
        "conditional_volatility",
        "garch_z_return",
        "dynamic_z_score",
    } <= set(df.columns)
    assert df.select(pl.col("dynamic_z_score").drop_nulls().is_nan().sum()).item() == 0


def test_robust_z_is_less_distorted_after_single_outlier():
    closes = [100.0] * 80 + [200.0, 100.0] + [100.0] * 20
    df = pl.DataFrame({"close": closes})
    df = add_z_score(20)(df)
    df = add_robust_z_score(20)(df)

    standard_after_outlier = abs(float(df["close_z_score"][81]))
    robust_after_outlier = abs(float(df["robust_z_score"][81]))

    assert robust_after_outlier <= standard_after_outlier


def test_compute_signal_frame_precomputes_indicators_from_raw_ohlcv():
    raw = _df().drop("atr_14", "ema_50", "ema_200", "adx_14", "rsi_14")
    df = compute_signal_frame(raw, momentum_burst_spec())
    assert "signal" in df.columns
    assert "atr_14" in df.columns
    assert "ema_200" in df.columns


def test_directional_hold_policy_exits_only_matching_side():
    df = pl.DataFrame(
        {
            "timestamp": [1, 2, 3, 4],
            "entry": [1.0, 0.0, 0.0, 0.0],
            "long_exit": [False, False, True, False],
            "short_exit": [True, True, False, False],
        }
    )
    spec = StrategySpec(
        name="test_directional_exit",
        required_columns=("timestamp", "entry", "long_exit", "short_exit"),
        features=(),
        entries=(SignalRule("long", 1, pl.col("entry") > 0),),
        hold=HoldPolicy(
            exit_long_when=pl.col("long_exit"),
            exit_short_when=pl.col("short_exit"),
        ),
    )

    out = apply_strategy_spec(df, spec)

    assert out["position_signal"].to_list() == [1.0, 1.0, 0.0, 0.0]
    assert out["exit_signal"].to_list() == [False, False, True, False]


def test_okx_signal_config_is_data_only():
    assert not hasattr(OkxSignalConfig(), "compute")
    assert not hasattr(OkxSignalConfig(), "strategy")
