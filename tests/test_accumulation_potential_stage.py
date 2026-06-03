from __future__ import annotations

from qooi.strategies.potential import PotentialStageThresholds, classify_potential_stage


def _base_row(**overrides):
    row = {
        "history_hours": 800,
        "quote_volume_24h": 600_000.0,
        "source_coverage_score": 0.9,
        "price_to_90d_low": 1.1,
        "range_position_90d_pct": 0.2,
        "bb_width_percentile_90d": 0.2,
        "volume_contraction_10d_90d": 0.5,
        "new_low_15d": False,
        "drawdown_30d_pct": -0.1,
        "volume_spike_ratio_1h_20h": 1.0,
        "prior_spike_count_5d": 0,
        "return_1h": 0.0,
        "return_24h": 0.0,
        "price_vs_vwap_24h_pct": 0.0,
        "base_duration_hours": 240,
        "new_low_count_30d": 0,
        "higher_low_count_30d": 2,
        "price_vs_ma_7d_pct": 0.01,
        "price_vs_ma_30d_pct": -0.01,
        "ma_7d_slope_7d": 0.01,
        "ma_30d_slope_14d": -0.01,
        "return_60d": -0.15,
        "return_90d": -0.20,
        "reclaim_state": "ma7_reclaim",
        "range_width_90d_pct": 0.2,
        "price_to_30d_high": 1.3,
        "taker_buy_ratio": 0.55,
        "depth_imbalance_25_mean": 0.0,
    }
    row.update(overrides)
    return row


def test_classifies_stealth_and_base_ready() -> None:
    assert classify_potential_stage(_base_row(quote_volume_24h=100_000.0))[0] == "stealth_base"
    assert classify_potential_stage(_base_row())[0] == "base_ready"


def test_classifies_first_expansion_and_controlled_lift() -> None:
    first = _base_row(volume_spike_ratio_1h_20h=4.0, return_1h=0.02)
    lift = _base_row(
        price_to_90d_low=1.8,
        range_position_90d_pct=0.5,
        bb_width_percentile_90d=0.8,
        volume_contraction_10d_90d=1.0,
        return_24h=0.04,
        price_vs_vwap_24h_pct=0.03,
    )

    assert classify_potential_stage(first)[0] == "first_expansion"
    assert classify_potential_stage(lift)[0] == "controlled_lift"


def test_classifies_pump_late_and_insufficient_history() -> None:
    pump = _base_row(prior_spike_count_5d=3, return_24h=0.18)
    late = _base_row(
        range_position_90d_pct=0.9,
        price_to_30d_high=1.02,
        price_vs_vwap_24h_pct=-0.02,
    )
    short = _base_row(history_hours=12)

    assert classify_potential_stage(pump)[0] == "pump_chop"
    assert classify_potential_stage(late)[0] == "late_distribution_risk"
    assert (
        classify_potential_stage(short, PotentialStageThresholds(min_history_hours=720))[0]
        == "insufficient_history"
    )


def test_missing_structure_values_do_not_count_as_base_evidence() -> None:
    row = _base_row(
        price_to_90d_low=None,
        range_position_90d_pct=None,
        bb_width_percentile_90d=None,
        volume_contraction_10d_90d=None,
    )

    stage, _confidence = classify_potential_stage(row)

    assert stage != "stealth_base"
    assert stage != "base_ready"


def test_near_low_active_lower_lows_are_falling_knife() -> None:
    row = _base_row(
        base_duration_hours=12,
        new_low_count_30d=8,
        ma_30d_slope_14d=-0.08,
        price_vs_vwap_24h_pct=-0.03,
        reclaim_state="below_ma",
    )

    assert classify_potential_stage(row)[0] == "falling_knife"


def test_long_downtrend_without_reclaim_is_cooldown_downtrend() -> None:
    row = _base_row(
        base_duration_hours=24,
        new_low_count_30d=1,
        ma_30d_slope_14d=-0.06,
        return_60d=-0.40,
        return_90d=-0.55,
        price_vs_vwap_24h_pct=-0.02,
        price_vs_ma_7d_pct=-0.03,
        reclaim_state="below_ma",
    )

    assert classify_potential_stage(row)[0] == "cooldown_downtrend"


def test_stable_base_with_reclaim_remains_base_ready() -> None:
    row = _base_row(
        base_duration_hours=240,
        new_low_count_30d=0,
        ma_30d_slope_14d=-0.005,
        price_vs_ma_7d_pct=0.02,
        reclaim_state="ma7_reclaim",
    )

    assert classify_potential_stage(row)[0] == "base_ready"

