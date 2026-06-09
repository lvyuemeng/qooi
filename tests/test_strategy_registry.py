"""Strategy boundary tests.

The strategy area is intentionally unstable. These tests keep the public signal
contract and a few known-at-close structural invariants without freezing every
historical strategy variant as ideal design.
"""

from __future__ import annotations

from dataclasses import replace

import polars as pl
import pytest

from qooi.strategies import (
    compute_signal_frame,
    ema_trend_baseline_spec,
    momentum_burst_spec,
    rsi_bounce_reversion_spec,
    rsi_macd_trend_spec,
    structure_event_reversal_v1_spec,
    structure_event_trend_aligned_mtf_confirm_v1_spec,
    structure_event_trend_aligned_v1_spec,
)
from qooi.strategies.catalog import BENCHMARK_GROUPS, strategy_selection
from qooi.strategies.indicators import add_garch_like_volatility, add_volatility_regime
from qooi.strategies.specs import HoldPolicy, SignalRule, StrategySpec, apply_strategy_spec
from qooi.strategies.structure import (
    add_liquidity_sweep_features,
    add_none_context_diagnostics,
    add_price_structure_stage_features,
)

SIGNAL_COLUMNS = {
    "raw_entry_signal",
    "entry_signal",
    "signal_strength",
    "signal_id",
    "position_signal",
    "exit_signal",
    "signal",
}


def _ohlcv_frame(
    rows: int = 80,
    *,
    close_start: float = 100.0,
    close_step: float = 0.01,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": list(range(rows)),
            "open": [100.0] * rows,
            "high": [101.0] * rows,
            "low": [99.0] * rows,
            "close": [close_start + i * close_step for i in range(rows)],
            "vol": [100.0] * rows,
            "atr_14": [1.0] * rows,
            "ema_50": [101.0] * rows,
            "ema_200": [100.0] * rows,
            "adx_14": [25.0] * rows,
            "rsi_14": [50.0] * rows,
        }
    )


def _liquidity_frame(**overrides) -> pl.DataFrame:
    data = {
        "open": [100.0, 100.0, 100.0, 98.0, 102.0, 107.0, 96.0],
        "high": [101.0, 102.0, 103.0, 100.0, 106.0, 112.0, 98.0],
        "low": [99.0, 98.0, 97.0, 95.0, 101.0, 101.0, 94.0],
        "close": [100.0, 100.0, 100.0, 98.0, 102.0, 102.0, 95.0],
        "vol": [100.0, 100.0, 100.0, 200.0, 180.0, 180.0, 190.0],
        "atr_14": [1.0] * 7,
    }
    data.update(overrides)
    return pl.DataFrame(data)


def _structural_event_frame(**overrides) -> pl.DataFrame:
    data = {
        "timestamp": [1],
        "open": [100.0],
        "high": [101.0],
        "low": [99.0],
        "close": [100.0],
        "vol": [100.0],
        "atr_14": [1.0],
        "liquidity_event_type": ["failed_breakout_low"],
        "failed_breakout_low": [True],
        "failed_breakout_high": [False],
        "prior_liquidity_low": [98.0],
        "prior_liquidity_high": [102.0],
        "event_quality_score": [2.0],
        "volume_impulse": [True],
        "structure_trend_state": ["uptrend"],
        "market_stage": ["accumulation"],
        "market_stage_reason": ["compressed_near_low"],
        "breakout_acceptance_low": [False],
        "breakout_acceptance_high": [False],
        "m15_confirm_long": [True],
        "m15_confirm_short": [False],
        "m15_confirm_available": [True],
    }
    data.update(overrides)
    rows = max(len(value) for value in data.values())
    data = {
        key: value * rows if len(value) == 1 and rows > 1 else value
        for key, value in data.items()
    }
    return pl.DataFrame(data)


def _featureless_structural_copy(spec):
    return replace(
        spec,
        features=(),
        required_columns=(
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "vol",
            "atr_14",
            "liquidity_event_type",
            "failed_breakout_low",
            "failed_breakout_high",
            "prior_liquidity_low",
            "prior_liquidity_high",
            "event_quality_score",
            "volume_impulse",
            "structure_trend_state",
            "market_stage",
            "market_stage_reason",
            "breakout_acceptance_low",
            "breakout_acceptance_high",
            "m15_confirm_long",
            "m15_confirm_short",
            "m15_confirm_available",
        ),
    )


def _featureless_reversal_spec(**kwargs):
    return _featureless_structural_copy(structure_event_reversal_v1_spec(**kwargs))


def _featureless_trend_spec(**kwargs):
    return _featureless_structural_copy(structure_event_trend_aligned_v1_spec(**kwargs))


def _featureless_no_range_spec(**kwargs):
    return _featureless_structural_copy(
        structure_event_trend_aligned_v1_spec(
            exclude_market_stages=("range",),
            exclude_market_stage_reasons=("compressed_mid_range",),
            name="structure_event_trend_aligned_no_range_v1",
            **kwargs,
        )
    )


def _featureless_no_range_longs_spec(**kwargs):
    return _featureless_structural_copy(
        structure_event_trend_aligned_v1_spec(
            exclude_long_market_stages=("range",),
            exclude_long_market_stage_reasons=("compressed_mid_range",),
            name="structure_event_trend_aligned_no_range_longs_v1",
            **kwargs,
        )
    )


@pytest.mark.parametrize(
    "spec",
    [
        ema_trend_baseline_spec(),
        momentum_burst_spec(),
        rsi_bounce_reversion_spec(),
        rsi_macd_trend_spec(),
        structure_event_reversal_v1_spec(),
        structure_event_trend_aligned_v1_spec(),
        structure_event_trend_aligned_mtf_confirm_v1_spec(),
    ],
)
def test_public_strategy_specs_emit_required_signal_columns(spec):
    df = compute_signal_frame(_ohlcv_frame(), spec)

    assert SIGNAL_COLUMNS <= set(df.columns)


def test_compute_signal_frame_rejects_non_spec_and_can_prepare_raw_ohlcv():
    raw = _ohlcv_frame().drop("atr_14", "ema_50", "ema_200", "adx_14", "rsi_14")

    with pytest.raises(TypeError):
        compute_signal_frame(raw, "unknown")

    df = compute_signal_frame(raw, momentum_burst_spec())
    assert {"signal", "atr_14", "ema_200"} <= set(df.columns)


def test_generic_stat_features_add_explicit_columns_without_division_errors():
    df = _ohlcv_frame(220, close_step=0.0).drop(
        "atr_14", "ema_50", "ema_200", "adx_14", "rsi_14"
    )
    for feature in (
        add_volatility_regime(short_span=12, long_span=36),
        add_garch_like_volatility(),
    ):
        df = feature(df)

    assert {
        "realized_vol_short",
        "realized_vol_long",
        "volatility_ratio",
        "volatility_regime",
        "conditional_volatility",
        "garch_z_return",
    } <= set(df.columns)


def test_liquidity_sweep_features_emit_warmup_reclaim_failure_and_acceptance_labels():
    out = add_liquidity_sweep_features(lookback=3, volume_period=3, volume_mult=1.2)(
        _liquidity_frame()
    )
    acceptance = add_liquidity_sweep_features(lookback=3, volume_period=3, volume_mult=1.2)(
        pl.DataFrame(
            {
                "open": [100.0, 100.0, 100.0, 102.0, 107.0, 96.0],
                "high": [101.0, 102.0, 103.0, 106.0, 112.0, 98.0],
                "low": [99.0, 98.0, 97.0, 101.0, 101.0, 94.0],
                "close": [100.0, 100.0, 100.0, 105.0, 102.0, 95.0],
                "vol": [100.0, 100.0, 100.0, 200.0, 180.0, 190.0],
                "atr_14": [1.0] * 6,
            }
        )
    )

    assert out["swept_low"][:3].to_list() == [False, False, False]
    assert out["swept_high"][:3].to_list() == [False, False, False]
    assert out["bullish_liquidity_sweep"][3]
    assert out["bearish_liquidity_sweep"][4]
    assert out["failed_breakout_high"][5]
    assert out["failed_breakout_low"][6]
    assert out["liquidity_event_type"][3] == "bullish_reclaim"
    assert out["liquidity_event_type"][4] == "bearish_reclaim"
    assert out["liquidity_event_type"][5] == "bearish_reclaim"
    assert out["liquidity_event_type"][6] == "failed_breakout_low"
    assert acceptance["breakout_acceptance_high"][3]
    assert acceptance["breakout_acceptance_low"][5]
    assert float(out["event_quality_score"][4]) > 0.0


def test_structure_stage_features_are_known_at_close_and_diagnostic() -> None:
    df = pl.DataFrame(
        {
            "open": [10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 14.5],
            "high": [10.2, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0, 14.0, 15.5],
            "low": [8.0, 9.0, 8.5, 9.5, 9.0, 10.0, 9.5, 10.5, 10.0, 11.0],
            "close": [10.0, 11.5, 10.7, 12.5, 11.7, 13.5, 12.7, 14.5, 13.7, 15.0],
            "atr_14": [1.0] * 10,
        }
    )
    feature = add_price_structure_stage_features(
        swing_lookback=1,
        range_lookback=3,
        trend_window=4,
        range_width_atr_max=10.0,
    )

    full = feature(df)
    prefix = feature(df.slice(0, 6))

    assert full["market_stage"][:3].to_list() == ["warmup", "warmup", "warmup"]
    assert full["stage_unknown_reason"][0] == "warmup"
    assert full["structure_trend_state"][5] == "uptrend"
    assert full["range_compression"][4]
    for column in (
        "structure_trend_state",
        "market_stage",
        "structure_reason",
        "market_stage_reason",
        "stage_unknown_reason",
    ):
        assert prefix[column][5] == full[column][5]


def test_none_context_diagnostics_emit_stable_buckets_without_lookahead():
    df = pl.DataFrame(
        {
            "open": [100.0] * 105,
            "high": [101.0] * 105,
            "low": [99.0] * 105,
            "close": [100.0] * 105,
            "atr_14": [1.0] * 104 + [3.0],
        }
    )
    df = add_liquidity_sweep_features(lookback=3, volume_period=3)(df)
    out = add_none_context_diagnostics(atr_period=100)(df)

    assert out["atr_percentile_bucket"][98] == "unknown"
    assert out["atr_percentile_bucket"][104] == "extreme"
    assert out["key_level_proximity_bucket"][104] == "near_prior_high_no_breach"


@pytest.mark.parametrize(
    ("frame", "expected_entry", "expected_id"),
    [
        (_structural_event_frame(), 1.0, "long_failed_breakout_low"),
        (
            _structural_event_frame(
                liquidity_event_type=["failed_breakout_high"],
                failed_breakout_low=[False],
                failed_breakout_high=[True],
            ),
            -1.0,
            "short_failed_breakout_high",
        ),
        (_structural_event_frame(volume_impulse=[False]), 0.0, None),
        (_structural_event_frame(event_quality_score=[1.49]), 0.0, None),
    ],
)
def test_structure_reversal_entries_respect_direction_volume_and_quality_gates(
    frame, expected_entry, expected_id
):
    out = apply_strategy_spec(frame, _featureless_reversal_spec())

    assert out["entry_signal"].to_list() == [expected_entry]
    if expected_id is not None:
        assert out["signal_id"].to_list() == [expected_id]


def test_structure_reversal_rejects_reclaim_entries_and_exits_on_accepted_breakout():
    reclaim = _structural_event_frame(
        liquidity_event_type=["bullish_reclaim"],
        failed_breakout_low=[False],
        event_quality_score=[3.0],
    )
    exit_frame = _structural_event_frame(
        timestamp=[1, 2],
        liquidity_event_type=["failed_breakout_low", "breakout_acceptance_low"],
        failed_breakout_low=[True, False],
        event_quality_score=[2.0, 0.0],
        volume_impulse=[True, False],
        breakout_acceptance_low=[False, True],
    )

    reclaim_entry = apply_strategy_spec(reclaim, _featureless_reversal_spec())[
        "entry_signal"
    ].to_list()
    assert reclaim_entry == [0.0]
    with pytest.raises(ValueError, match="include_reclaim_sweeps"):
        structure_event_reversal_v1_spec(include_reclaim_sweeps=True)
    out = apply_strategy_spec(exit_frame, _featureless_reversal_spec())
    assert out["position_signal"].to_list() == [1.0, 0.0]
    assert out["exit_signal"].to_list() == [False, True]


@pytest.mark.parametrize(
    ("frame", "spec", "expected"),
    [
        (
            _structural_event_frame(structure_trend_state=["uptrend"]),
            _featureless_trend_spec(),
            1.0,
        ),
        (
            _structural_event_frame(structure_trend_state=["downtrend"]),
            _featureless_trend_spec(),
            0.0,
        ),
        (
            _structural_event_frame(
                liquidity_event_type=["failed_breakout_high"],
                failed_breakout_low=[False],
                failed_breakout_high=[True],
                structure_trend_state=["downtrend"],
            ),
            _featureless_trend_spec(),
            -1.0,
        ),
        (
            _structural_event_frame(
                market_stage=["range"],
                market_stage_reason=["compressed_mid_range"],
            ),
            _featureless_no_range_spec(),
            0.0,
        ),
        (
            _structural_event_frame(
                liquidity_event_type=["failed_breakout_high"],
                failed_breakout_low=[False],
                failed_breakout_high=[True],
                structure_trend_state=["downtrend"],
                market_stage=["range"],
                market_stage_reason=["compressed_mid_range"],
            ),
            _featureless_no_range_longs_spec(),
            -1.0,
        ),
    ],
)
def test_structure_trend_aligned_entries_keep_direction_and_range_filter_contract(
    frame, spec, expected
):
    assert apply_strategy_spec(frame, spec)["entry_signal"].to_list() == [expected]


def test_structure_trend_aligned_mtf_confirm_requires_matching_confirmation():
    spec = _featureless_structural_copy(structure_event_trend_aligned_mtf_confirm_v1_spec())

    confirmed = apply_strategy_spec(_structural_event_frame(m15_confirm_long=[True]), spec)
    blocked = apply_strategy_spec(_structural_event_frame(m15_confirm_long=[False]), spec)

    assert confirmed["entry_signal"].to_list() == [1.0]
    assert blocked["entry_signal"].to_list() == [0.0]


def test_strategy_catalog_keeps_unpromoted_range_variants_in_development_group():
    for name in (
        "structure_event_trend_aligned_no_range_v1",
        "structure_event_trend_aligned_no_range_longs_v1",
    ):
        spec = strategy_selection((name,)).strategies[0]

        assert spec.name == name
        assert name in BENCHMARK_GROUPS["structure-development"]
        assert name not in BENCHMARK_GROUPS["candidate"]


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
