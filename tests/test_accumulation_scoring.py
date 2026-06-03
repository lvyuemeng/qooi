from __future__ import annotations

import polars as pl
import pytest

from qooi.accumulation.config import AccumulationConfig, ScoringConfig
from qooi.accumulation.scoring import score_accumulation_features


def _feature(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "timestamp": 1,
        "symbol": "BTC-USDT-SWAP",
        "close": 100.0,
        "ma200": 90.0,
        "return_24h": -0.03,
        "flow_zscore": -3.5,
        "flow_zscore_negative_streak_hours": 2,
        "whale_accumulation_ratio": 0.8,
        "depth_imbalance_10_mean": 0.4,
        "resilience_score": 0.7,
        "data_quality_warning": "onchain_partial",
    }
    row.update(overrides)
    return row


def test_score_exact_contributions_and_explanation() -> None:
    out = score_accumulation_features(pl.DataFrame([_feature()]), ScoringConfig())

    assert out["onchain_score"][0] == 40
    assert out["orderbook_score"][0] == 20
    assert out["trade_score"][0] == 15
    assert out["score_total"][0] == 75
    assert out["alert_level"][0] == "red"
    assert "exchange_outflow_zscore_streak" in out["explanation"][0]


def test_negative_filters_suppress_positive_alert() -> None:
    out = score_accumulation_features(
        pl.DataFrame(
            [
                _feature(
                    flow_zscore=4.0,
                    flow_zscore_negative_streak_hours=0,
                    close=80.0,
                    ma200=100.0,
                    depth_imbalance_10_mean=-0.2,
                    whale_accumulation_ratio=0.8,
                    resilience_score=0.7,
                )
            ]
        ),
        ScoringConfig(),
    )

    assert out["negative_score"][0] == -55
    assert out["score_total"][0] < 20
    assert out["alert_level"][0] == "none"
    assert out["suggestion_type"][0] == "reject_or_deprioritize"


def test_missing_message_fields_are_neutral() -> None:
    out = score_accumulation_features(pl.DataFrame([_feature()]), ScoringConfig())

    assert out["message_score"][0] == 0
    assert out["message_not_overheated"][0] is False
    assert out["message_overheated"][0] is False


def test_trade_metadata_missing_blocks_high_confidence() -> None:
    out = score_accumulation_features(
        pl.DataFrame(
            [
                _feature(
                    source_coverage_score=1.0,
                    data_quality_warning="trade_notional_metadata_missing",
                )
            ]
        ),
        ScoringConfig(),
    )

    assert out["confidence_level"][0] != "high"
    assert "trade_notional_metadata_missing" in out["missing_evidence"][0]


def test_zero_score_structure_only_row_is_rejected() -> None:
    out = score_accumulation_features(
        pl.DataFrame(
            [
                _feature(
                    flow_zscore=0.0,
                    flow_zscore_negative_streak_hours=0,
                    whale_accumulation_ratio=0.0,
                    depth_imbalance_10_mean=0.0,
                    resilience_score=0.0,
                    return_24h=0.0,
                    source_coverage_score=1.0,
                    range_position_pct=0.5,
                    data_quality_warning="",
                )
            ]
        ),
        ScoringConfig(),
    )

    assert out["score_total"][0] == 0
    assert out["explanation"][0] == "no_rules_fired"
    assert out["suggestion_type"][0] == "reject_or_deprioritize"


def test_disabled_source_tokens_are_not_missing_evidence() -> None:
    out = score_accumulation_features(
        pl.DataFrame([_feature(data_quality_warning="messages_disabled;onchain_disabled")]),
        ScoringConfig(),
    )

    assert out["missing_evidence"][0] == ""


def test_default_config_parses_and_invalid_thresholds_raise() -> None:
    cfg = AccumulationConfig.model_validate({})
    assert cfg.run.out
    with pytest.raises(ValueError):
        ScoringConfig(flow_outflow_z=float("inf"))

