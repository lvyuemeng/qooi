"""Composable strategy tests."""

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
from qooi.strategies.features import (
    add_garch_like_volatility,
    add_liquidity_sweep_features,
    add_none_context_diagnostics,
    add_price_structure_stage_features,
    add_volatility_regime,
)
from qooi.strategies.specs import HoldPolicy, SignalRule, StrategySpec, apply_strategy_spec


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
        "open": [100.0, 100.0, 100.0, 98.0, 102.0],
        "high": [101.0, 102.0, 103.0, 100.0, 106.0],
        "low": [99.0, 98.0, 97.0, 95.0, 101.0],
        "close": [100.0, 100.0, 100.0, 98.0, 102.0],
        "vol": [100.0, 100.0, 100.0, 200.0, 180.0],
        "atr_14": [1.0] * 5,
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


def _featureless_structural_spec(**kwargs):
    spec = structure_event_reversal_v1_spec(**kwargs)
    return _featureless_structural_copy(spec)


def _featureless_trend_aligned_spec(**kwargs):
    spec = structure_event_trend_aligned_v1_spec(**kwargs)
    return _featureless_structural_copy(spec)


def _featureless_mtf_trend_aligned_spec(**kwargs):
    spec = structure_event_trend_aligned_mtf_confirm_v1_spec(**kwargs)
    return _featureless_structural_copy(spec)


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
            "breakout_acceptance_low",
            "breakout_acceptance_high",
            "m15_confirm_long",
            "m15_confirm_short",
            "m15_confirm_available",
        ),
    )


def test_composed_strategy_adds_required_signal_module_columns():
    df = compute_signal_frame(_ohlcv_frame(), momentum_burst_spec())
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
        compute_signal_frame(_ohlcv_frame(), "unknown")


def test_rsi_bounce_reversion_adds_long_only_signal_column():
    df = compute_signal_frame(_ohlcv_frame(), rsi_bounce_reversion_spec())
    assert "signal" in df.columns
    assert set(df["signal"].drop_nulls().unique().to_list()) <= {0.0, 1.0}


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
def test_signal_module_specs_compute_explicit_outputs(spec):
    df = compute_signal_frame(_ohlcv_frame(), spec)

    assert "signal_strength" in df.columns
    assert "signal_id" in df.columns
    assert "exit_signal" in df.columns


def test_generic_stat_features_add_explicit_columns_without_division_errors():
    df = _ohlcv_frame(220, close_step=0.0).drop("atr_14", "ema_50", "ema_200", "adx_14", "rsi_14")
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


def test_liquidity_sweep_features_detect_bullish_and_bearish_reclaims():
    df = _liquidity_frame()

    out = add_liquidity_sweep_features(lookback=3, volume_period=3, volume_mult=1.2)(df)

    assert out["prior_liquidity_low"][3] == 97.0
    assert out["swept_low"][3]
    assert out["reclaimed_low"][3]
    assert out["bullish_liquidity_sweep"][3]
    assert out["volume_impulse"][3]
    assert out["sweep_distance_atr"][3] == 2.0
    assert out["prior_liquidity_high"][4] == 103.0
    assert out["swept_high"][4]
    assert out["reclaimed_high"][4]
    assert out["bearish_liquidity_sweep"][4]


def test_liquidity_sweep_features_detect_failed_sweeps_and_warmup_is_safe():
    df = pl.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 96.0, 104.0],
            "high": [101.0, 102.0, 103.0, 98.0, 106.0],
            "low": [99.0, 98.0, 97.0, 95.0, 102.0],
            "close": [100.0, 100.0, 100.0, 96.0, 105.0],
            "volume": [100.0] * 5,
            "atr_14": [1.0] * 5,
        }
    )

    out = add_liquidity_sweep_features(lookback=3, volume_period=3)(df)

    assert out["swept_low"][:3].to_list() == [False, False, False]
    assert out["swept_high"][:3].to_list() == [False, False, False]
    assert out["failed_bullish_sweep"][3]
    assert not out["bullish_liquidity_sweep"][3]
    assert out["failed_bearish_sweep"][4]
    assert not out["bearish_liquidity_sweep"][4]


def test_liquidity_sweep_features_classify_acceptance_and_reclaim_quality():
    df = pl.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 102.0, 107.0, 96.0],
            "high": [101.0, 102.0, 103.0, 106.0, 112.0, 98.0],
            "low": [99.0, 98.0, 97.0, 101.0, 101.0, 94.0],
            "close": [100.0, 100.0, 100.0, 105.0, 102.0, 95.0],
            "vol": [100.0, 100.0, 100.0, 200.0, 180.0, 190.0],
            "atr_14": [1.0] * 6,
        }
    )

    out = add_liquidity_sweep_features(lookback=3, volume_period=3, volume_mult=1.2)(df)

    assert out["breakout_acceptance_high"][3]
    assert out["liquidity_event_type"][3] == "breakout_acceptance_high"
    assert out["failed_breakout_high"][4]
    assert out["bearish_rejection_bar"][4]
    assert out["liquidity_event_type"][4] == "bearish_reclaim"
    assert out["breakout_acceptance_low"][5]
    assert out["liquidity_event_type"][5] == "breakout_acceptance_low"
    assert float(out["event_quality_score"][4]) > 0.0


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
    assert out["near_prior_high_no_breach"][104]
    assert out["near_prior_low_no_breach"][104]
    assert out["key_level_proximity_bucket"][104] == "near_prior_high_no_breach"


def test_price_structure_stage_features_detect_uptrend_and_range_location():
    df = pl.DataFrame(
        {
            "open": [10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 14.5],
            "high": [10.2, 12.0, 11.0, 13.0, 12.0, 14.0, 13.0, 15.0, 14.0, 15.5],
            "low": [8.0, 9.0, 8.5, 9.5, 9.0, 10.0, 9.5, 10.5, 10.0, 11.0],
            "close": [10.0, 11.5, 10.7, 12.5, 11.7, 13.5, 12.7, 14.5, 13.7, 15.0],
            "atr_14": [1.0] * 10,
        }
    )

    out = add_price_structure_stage_features(
        swing_lookback=1,
        range_lookback=3,
        trend_window=4,
        range_width_atr_max=10.0,
    )(df)

    assert out["swing_high_confirmed"][:2].to_list() == [False, False]
    assert out["structure_higher_high"][4]
    assert out["structure_higher_low"][5]
    assert out["structure_trend_state"][5] == "uptrend"
    assert out["range_high"][4] == 13.0
    assert out["range_low"][4] == 8.5
    assert out["range_compression"][4]
    assert out["market_stage"][4] in {"range", "warmup", "accumulation"}
    assert out["market_stage_reason"][0] == "warmup_range_not_ready"
    assert out["stage_unknown_reason"][0] == "warmup"


def test_price_structure_stage_features_detect_downtrend_and_warmup_is_safe():
    df = pl.DataFrame(
        {
            "open": [15.0, 14.0, 14.5, 13.0, 13.5, 12.0, 12.5, 11.0, 11.5, 10.0],
            "high": [15.5, 15.0, 15.2, 14.0, 14.2, 13.0, 13.2, 12.0, 12.2, 11.0],
            "low": [14.0, 13.0, 13.5, 12.0, 12.5, 11.0, 11.5, 10.0, 10.5, 9.0],
            "close": [15.0, 13.5, 14.7, 12.5, 13.7, 11.5, 12.7, 10.5, 11.7, 9.5],
            "atr_14": [1.0] * 10,
        }
    )

    out = add_price_structure_stage_features(
        swing_lookback=1,
        range_lookback=3,
        trend_window=4,
        range_width_atr_max=10.0,
    )(df)

    assert out["market_stage"][:3].to_list() == ["warmup", "warmup", "warmup"]
    assert out["structure_lower_high"][5]
    assert out["structure_lower_low"][4]
    assert out["structure_trend_state"][5] == "downtrend"
    assert out["range_compression"][4]


def test_price_structure_stage_features_explain_non_warmup_unknown_stage():
    df = pl.DataFrame(
        {
            "open": [10.0, 10.2, 9.8, 10.1, 10.0, 10.3],
            "high": [12.0, 12.2, 12.1, 12.3, 12.0, 12.4],
            "low": [8.0, 7.8, 7.9, 7.7, 8.0, 7.6],
            "close": [10.0, 10.1, 10.0, 10.2, 10.1, 10.2],
            "atr_14": [0.25] * 6,
        }
    )

    out = add_price_structure_stage_features(
        swing_lookback=1,
        range_lookback=3,
        trend_window=3,
        range_width_atr_max=2.0,
    )(df)

    assert out["market_stage"][4] == "wide_range"
    assert out["stage_unknown_reason"][4] == "wide_range"


def test_price_structure_stage_features_do_not_change_when_future_bars_are_appended():
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
    target_idx = 5

    full = feature(df)
    prefix = feature(df.slice(0, target_idx + 1))

    for column in (
        "structure_trend_state",
        "market_stage",
        "structure_reason",
        "market_stage_reason",
        "stage_unknown_reason",
    ):
        assert prefix[column][target_idx] == full[column][target_idx]


def test_compute_signal_frame_precomputes_indicators_from_raw_ohlcv():
    raw = _ohlcv_frame().drop("atr_14", "ema_50", "ema_200", "adx_14", "rsi_14")
    df = compute_signal_frame(raw, momentum_burst_spec())
    assert "signal" in df.columns
    assert "atr_14" in df.columns
    assert "ema_200" in df.columns


def test_structure_event_reversal_feature_stack_adds_structural_columns():
    raw = _ohlcv_frame().drop("atr_14", "ema_50", "ema_200", "adx_14", "rsi_14")

    df = compute_signal_frame(raw, structure_event_reversal_v1_spec())

    assert {
        "liquidity_event_type",
        "event_quality_score",
        "atr_percentile_bucket",
        "structure_trend_state",
        "market_stage",
        "signal",
    } <= set(df.columns)


def test_structure_event_reversal_enters_long_failed_breakout_low():
    out = apply_strategy_spec(_structural_event_frame(), _featureless_structural_spec())

    assert out["entry_signal"].to_list() == [1.0]
    assert out["signal_id"].to_list() == ["long_failed_breakout_low"]


def test_structure_event_reversal_enters_short_failed_breakout_high():
    frame = _structural_event_frame(
        liquidity_event_type=["failed_breakout_high"],
        failed_breakout_low=[False],
        failed_breakout_high=[True],
    )

    out = apply_strategy_spec(frame, _featureless_structural_spec())


    assert out["entry_signal"].to_list() == [-1.0]
    assert out["signal_id"].to_list() == ["short_failed_breakout_high"]


def test_structure_event_reversal_volume_and_quality_gates_block_entries():
    spec = _featureless_structural_spec()

    assert apply_strategy_spec(
        _structural_event_frame(volume_impulse=[False]), spec
    )["entry_signal"].to_list() == [0.0]
    assert apply_strategy_spec(
        _structural_event_frame(event_quality_score=[1.49]), spec
    )["entry_signal"].to_list() == [0.0]


def test_structure_event_reversal_can_disable_volume_gate_explicitly():
    out = apply_strategy_spec(
        _structural_event_frame(volume_impulse=[False]),
        _featureless_structural_spec(require_volume_impulse=False),
    )

    assert out["entry_signal"].to_list() == [1.0]


def test_structure_event_reversal_reclaim_disabled_by_default_and_true_rejected():
    reclaim = _structural_event_frame(
        liquidity_event_type=["bullish_reclaim"],
        failed_breakout_low=[False],
        event_quality_score=[3.0],
    )

    out = apply_strategy_spec(reclaim, _featureless_structural_spec())

    assert out["entry_signal"].to_list() == [0.0]
    with pytest.raises(ValueError, match="include_reclaim_sweeps"):
        structure_event_reversal_v1_spec(include_reclaim_sweeps=True)


def test_structure_event_reversal_directional_exit_on_accepted_breakout():
    frame = _structural_event_frame(
        timestamp=[1, 2],
        open=[100.0, 100.0],
        high=[101.0, 101.0],
        low=[99.0, 99.0],
        close=[100.0, 100.0],
        vol=[100.0, 100.0],
        atr_14=[1.0, 1.0],
        liquidity_event_type=["failed_breakout_low", "breakout_acceptance_low"],
        failed_breakout_low=[True, False],
        failed_breakout_high=[False, False],
        prior_liquidity_low=[98.0, 98.0],
        prior_liquidity_high=[102.0, 102.0],
        event_quality_score=[2.0, 0.0],
        volume_impulse=[True, False],
        structure_trend_state=["uptrend", "uptrend"],
        breakout_acceptance_low=[False, True],
        breakout_acceptance_high=[False, False],
    )

    out = apply_strategy_spec(frame, _featureless_structural_spec())

    assert out["position_signal"].to_list() == [1.0, 0.0]
    assert out["exit_signal"].to_list() == [False, True]


def test_structure_event_trend_aligned_computes_explicit_signal_columns():
    out = compute_signal_frame(_structural_event_frame(), _featureless_trend_aligned_spec())

    assert {
        "raw_entry_signal",
        "entry_signal",
        "signal_strength",
        "signal_id",
        "position_signal",
        "exit_signal",
        "signal",
    } <= set(out.columns)


@pytest.mark.parametrize(
    ("trend_state", "expected"),
    [
        ("uptrend", 1.0),
        ("downtrend", 0.0),
        ("range", 0.0),
        ("unknown", 0.0),
    ],
)
def test_structure_event_trend_aligned_long_requires_uptrend(trend_state, expected):
    out = apply_strategy_spec(
        _structural_event_frame(structure_trend_state=[trend_state]),
        _featureless_trend_aligned_spec(),
    )

    assert out["entry_signal"].to_list() == [expected]


@pytest.mark.parametrize(
    ("trend_state", "expected"),
    [
        ("downtrend", -1.0),
        ("uptrend", 0.0),
        ("range", 0.0),
        ("unknown", 0.0),
    ],
)
def test_structure_event_trend_aligned_short_requires_downtrend(trend_state, expected):
    frame = _structural_event_frame(
        liquidity_event_type=["failed_breakout_high"],
        failed_breakout_low=[False],
        failed_breakout_high=[True],
        structure_trend_state=[trend_state],
    )

    out = apply_strategy_spec(frame, _featureless_trend_aligned_spec())

    assert out["entry_signal"].to_list() == [expected]


def test_structure_event_trend_aligned_volume_and_quality_gates_block_entries():
    spec = _featureless_trend_aligned_spec()

    assert apply_strategy_spec(
        _structural_event_frame(volume_impulse=[False]), spec
    )["entry_signal"].to_list() == [0.0]
    assert apply_strategy_spec(
        _structural_event_frame(event_quality_score=[1.49]), spec
    )["entry_signal"].to_list() == [0.0]


def test_structure_event_trend_aligned_can_disable_volume_gate_explicitly():
    out = apply_strategy_spec(
        _structural_event_frame(volume_impulse=[False]),
        _featureless_trend_aligned_spec(require_volume_impulse=False),
    )

    assert out["entry_signal"].to_list() == [1.0]


def test_structure_event_trend_aligned_mtf_confirm_requires_matching_confirmation():
    spec = _featureless_mtf_trend_aligned_spec()

    confirmed = apply_strategy_spec(_structural_event_frame(m15_confirm_long=[True]), spec)
    blocked = apply_strategy_spec(_structural_event_frame(m15_confirm_long=[False]), spec)

    assert confirmed["entry_signal"].to_list() == [1.0]
    assert blocked["entry_signal"].to_list() == [0.0]


def test_structure_event_trend_aligned_directional_exit_on_accepted_breakout():
    frame = _structural_event_frame(
        timestamp=[1, 2],
        open=[100.0, 100.0],
        high=[101.0, 101.0],
        low=[99.0, 99.0],
        close=[100.0, 100.0],
        vol=[100.0, 100.0],
        atr_14=[1.0, 1.0],
        liquidity_event_type=["failed_breakout_low", "breakout_acceptance_low"],
        failed_breakout_low=[True, False],
        failed_breakout_high=[False, False],
        prior_liquidity_low=[98.0, 98.0],
        prior_liquidity_high=[102.0, 102.0],
        event_quality_score=[2.0, 0.0],
        volume_impulse=[True, False],
        structure_trend_state=["uptrend", "uptrend"],
        breakout_acceptance_low=[False, True],
        breakout_acceptance_high=[False, False],
    )

    out = apply_strategy_spec(frame, _featureless_trend_aligned_spec())

    assert out["position_signal"].to_list() == [1.0, 0.0]
    assert out["exit_signal"].to_list() == [False, True]


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
