from __future__ import annotations

import polars as pl

import qooi.strategies.features as features
from qooi.strategies.features import (
    RangeWidthThresholdConfig,
    StructureClassifierConfig,
    add_price_structure_stage_features,
)


def _frame(rows: int = 180) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [idx * 3_600_000 for idx in range(rows)],
            "open": [100.0 + (idx % 3) for idx in range(rows)],
            "high": [102.0 + (idx % 5) for idx in range(rows)],
            "low": [98.0 - (idx % 4) for idx in range(rows)],
            "close": [100.5 + (idx % 2) for idx in range(rows)],
            "atr_14": [1.0 + (idx % 7) * 0.05 for idx in range(rows)],
        }
    )


def test_structure_classifier_config_public_constructors():
    default = StructureClassifierConfig.default()
    fixed = StructureClassifierConfig.fixed(range_width_atr_max=7.5)
    rolling = StructureClassifierConfig.rolling_quantile(quantile=0.7, window=90, min_samples=30)

    assert default.range_width_threshold.mode == "rolling_quantile"
    assert fixed.range_width_threshold == RangeWidthThresholdConfig(mode="fixed", fixed_atr_max=7.5)
    assert rolling.range_width_threshold.quantile == 0.7
    assert rolling.range_width_threshold.window == 90
    assert rolling.range_width_threshold.min_samples == 30


def test_structure_classifier_facades_are_removed():
    assert not hasattr(features, "StructureClassifier")
    assert not hasattr(features, "classify_price_structure_frame")


def test_add_price_structure_stage_features_is_classifier_api():
    config = StructureClassifierConfig.fixed(range_width_atr_max=8.0)
    frame = _frame()

    out = add_price_structure_stage_features(config=config)(frame)

    columns = [
        "structure_trend_state",
        "market_stage",
        "structure_reason",
        "market_stage_reason",
        "stage_unknown_reason",
        "range_width_atr_threshold",
        "range_width_threshold_source",
    ]
    assert set(columns) <= set(out.columns)


def test_add_price_structure_stage_features_accepts_config_and_legacy_kwargs():
    frame = _frame(80)

    via_config = add_price_structure_stage_features(
        config=StructureClassifierConfig.fixed(range_width_atr_max=0.01)
    )(frame)
    via_legacy = add_price_structure_stage_features(range_width_atr_max=0.01)(frame)

    assert via_config.select("market_stage", "stage_unknown_reason").equals(
        via_legacy.select("market_stage", "stage_unknown_reason")
    )


def test_structure_classifier_uses_dynamic_threshold_without_lookahead():
    base = _frame(160)
    future = pl.DataFrame(
        {
            "timestamp": [(160 + idx) * 3_600_000 for idx in range(40)],
            "open": [100.0] * 40,
            "high": [200.0] * 40,
            "low": [1.0] * 40,
            "close": [100.0] * 40,
            "atr_14": [1.0] * 40,
        }
    )
    config = StructureClassifierConfig.rolling_quantile(window=60, min_samples=20)

    before = add_price_structure_stage_features(config=config)(base)
    after = add_price_structure_stage_features(config=config)(pl.concat([base, future]))

    audit_columns = [
        "range_width_atr_threshold",
        "range_width_threshold_source",
        "market_stage",
        "stage_unknown_reason",
    ]
    assert before.select(audit_columns).equals(after.head(base.height).select(audit_columns))


def test_structure_classifier_exposes_threshold_audit_columns():
    config = StructureClassifierConfig.rolling_quantile(window=30, min_samples=10)
    out = add_price_structure_stage_features(config=config)(_frame(90))

    assert {
        "range_width_atr_threshold",
        "range_width_threshold_mode",
        "range_width_threshold_ready",
        "range_width_threshold_source",
    } <= set(out.columns)
    assert set(out["range_width_threshold_mode"].unique().to_list()) == {"rolling_quantile"}
    assert "rolling_quantile" in set(out["range_width_threshold_source"].unique().to_list())
