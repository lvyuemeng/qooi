from __future__ import annotations

import polars as pl

from qooi.research.diagnostics import (
    add_forward_outcomes,
    add_market_state_reductions,
    classifier_health,
    joint_forward_quality,
    trade_record_control,
)


def test_classifier_health_returns_exportable_rows():
    result = classifier_health(
        pl.DataFrame(
            {
                "market_stage": ["range"],
                "structure_trend_state": ["range"],
                "structure_reason": ["compressed"],
                "stage_unknown_reason": ["none"],
            }
        ),
        label="BTC 1H",
    )

    assert set(result.frame["artifact"].to_list()) == {"classifier-health"}
    assert "required_classifier_columns" in result.text


def test_add_forward_outcomes_uses_future_as_label_only():
    frame = add_forward_outcomes(
        pl.DataFrame({"timestamp": [1, 2, 3], "close": [100.0, 101.0, 99.0]}),
        symbol="BTC",
        horizons=(1,),
    )

    assert frame["fwd_1_return_pct"].to_list()[0] == 1.0
    assert frame["fwd_1_direction"].to_list()[:2] == ["up", "down"]


def test_market_state_reduction_preserves_raw_labels_and_projects_semantics():
    frame = add_market_state_reductions(
        pl.DataFrame({"market_stage": ["wide_range", "trend_continuation", "range"]})
    )

    assert frame["market_stage"].to_list() == ["wide_range", "trend_continuation", "range"]
    assert frame["market_stage_reduced"].to_list() == [
        "wide_range",
        "trend_continuation",
        "range",
    ]


def test_joint_quality_side_normalizes_short_returns():
    frame = pl.DataFrame(
        {
            "symbol": ["BTC"] * 4,
            "timestamp": [1, 2, 3, 4],
            "d1_market_stage_reduced": ["range"] * 4,
            "liquidity_event_type": ["failed_breakout_high"] * 4,
            "fwd_1_return_pct": [-1.0, -2.0, 1.0, -3.0],
        }
    )

    result = joint_forward_quality(
        frame,
        horizons=(1,),
        min_rows=1,
        transition_min_rows=1,
        omega_threshold=1.5,
        pwpr_threshold=2.0,
        prior_strength=10,
        invalid_values=("warmup", "unknown", "data_error"),
    ).frame

    row = result.filter(pl.col("artifact") == "joint-forward-quality").row(0, named=True)
    assert row["side"] == "short"
    assert row["mean_side_return_pct"] > 0.0


def test_joint_quality_emits_required_artifacts():
    frame = pl.DataFrame(
        {
            "symbol": ["BTC"] * 6,
            "timestamp": list(range(6)),
            "market_stage_reduced": ["range", "markup", "markup", "range", "markup", "range"],
            "h4_market_stage_reduced": ["range"] * 6,
            "d1_market_stage_reduced": ["range"] * 6,
            "d1_structure_trend_state": ["uptrend"] * 6,
            "liquidity_event_type": ["failed_breakout_low"] * 6,
            "fwd_1_return_pct": [1.0, 2.0, 1.0, -0.5, 2.0, 1.0],
        }
    )

    artifacts = set(
        joint_forward_quality(
            frame,
            horizons=(1,),
            min_rows=1,
            transition_min_rows=1,
            omega_threshold=1.5,
            pwpr_threshold=2.0,
            prior_strength=10,
            invalid_values=("warmup", "unknown", "data_error"),
        )
        .frame["artifact"]
        .to_list()
    )

    assert "configuration-intrinsic-quality" in artifacts
    assert "joint-forward-quality" in artifacts
    assert "transition-event-quality" in artifacts
    assert "joint-reduction-comparison" in artifacts
    assert "inner-connection-reduction-quality" in artifacts


def test_joint_quality_shrinkage_changes_rank_fields():
    frame = pl.DataFrame(
        {
            "symbol": ["BTC"] * 12,
            "timestamp": list(range(12)),
            "d1_market_stage_reduced": ["rare"] * 2 + ["common"] * 10,
            "liquidity_event_type": ["failed_breakout_low"] * 12,
            "fwd_1_return_pct": [10.0, 10.0] + [1.0] * 10,
        }
    )

    rows = joint_forward_quality(
        frame,
        horizons=(1,),
        min_rows=1,
        transition_min_rows=1,
        omega_threshold=1.5,
        pwpr_threshold=2.0,
        prior_strength=10,
        invalid_values=("warmup", "unknown", "data_error"),
    ).frame.filter(pl.col("artifact") == "joint-forward-quality")
    rare = rows.filter(pl.col("joint_group") == "rare").row(0, named=True)

    assert rare["shrunk_mean_side_return_pct"] < rare["bucket_mean_side_return_pct"]
    assert "rank_delta" in rows.columns


def test_trade_record_control_is_optional_and_strategy_conditioned():
    result = trade_record_control(
        pl.DataFrame(
            {
                "side": ["buy", "buy", "sell", "sell"],
                "entry_market_stage": ["range"] * 4,
                "entry_d1_structure_trend_state": ["uptrend", "uptrend", "downtrend", "downtrend"],
                "net_pnl_usd": [1.0, 1.0, -1.0, -1.0],
            }
        ),
        min_base_trades=1,
        min_cell_trades=1,
        practical_delta_threshold=0.1,
    )

    assert not result.frame.is_empty()
    assert set(result.frame["artifact"].to_list()) == {"trade-record-modulation"}
