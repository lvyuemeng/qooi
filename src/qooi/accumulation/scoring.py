"""Rule-based accumulation-like scoring."""

from __future__ import annotations

from typing import Any

import polars as pl

from qooi.accumulation.config import ScoringConfig
from qooi.accumulation.schema import SCORE_SCHEMA, empty_score_frame


def _value(row: dict[str, Any], name: str, default: Any = None) -> Any:
    value = row.get(name, default)
    return default if value is None else value


def _alert(score: int, config: ScoringConfig) -> str:
    if score > config.red_threshold:
        return "red"
    if score > config.orange_threshold:
        return "orange"
    if score >= config.yellow_threshold:
        return "yellow"
    return "none"


def score_accumulation_features(
    frame: pl.DataFrame, config: ScoringConfig | None = None
) -> pl.DataFrame:
    if frame.is_empty():
        return empty_score_frame()
    cfg = config or ScoringConfig()
    rows = []
    for row in frame.to_dicts():
        flow_outflow = (
            _value(row, "flow_zscore", 0.0) < cfg.flow_outflow_z
            and _value(row, "flow_zscore_negative_streak_hours", 0) >= 2
        )
        whale = _value(row, "whale_accumulation_ratio", 0.0) > cfg.whale_accumulation_threshold
        depth = (
            _value(row, "depth_imbalance_10_mean", _value(row, "depth_imbalance_25_mean", 0.0))
            > cfg.depth_imbalance_threshold
            and _value(row, "return_24h", 0.0) < cfg.down_day_return_threshold
        )
        resilience = _value(row, "resilience_score", 0.0) > cfg.resilience_threshold
        has_message = (
            row.get("mention_growth") is not None and row.get("fundamental_news_ratio") is not None
        )
        message_not_hot = bool(
            has_message
            and _value(row, "mention_growth", 0.0) < cfg.mention_growth_hot
            and _value(row, "fundamental_news_ratio", 0.0) >= _value(row, "emotion_news_ratio", 0.0)
        )
        inflow = _value(row, "flow_zscore", 0.0) > cfg.flow_inflow_z
        message_hot = bool(
            has_message
            and _value(row, "mention_growth", 0.0) >= cfg.mention_growth_hot
            and _value(row, "emotion_news_ratio", 0.0) > _value(row, "fundamental_news_ratio", 0.0)
        )
        weak_depth = (
            _value(row, "depth_imbalance_10_mean", _value(row, "depth_imbalance_25_mean", 0.0))
            < 0.0
        )
        below_ma = (
            row.get("close") is not None
            and row.get("ma200") is not None
            and float(row["close"]) < float(row["ma200"])
        )
        below_ma_weak_depth = bool(below_ma and weak_depth)
        onchain_score = (25 if flow_outflow else 0) + (15 if whale else 0)
        orderbook_score = 20 if depth else 0
        trade_score = 15 if resilience else 0
        message_score = 10 if message_not_hot else 0
        negative_score = (
            (-30 if inflow else 0)
            + (-20 if message_hot else 0)
            + (-25 if below_ma_weak_depth else 0)
        )
        total = onchain_score + orderbook_score + trade_score + message_score + negative_score
        explanations = []
        positives = []
        negatives = []
        for fired, text in (
            (flow_outflow, "exchange_outflow_zscore_streak +25"),
            (whale, "whale_accumulation_high +15"),
            (depth, "depth_support_on_down_day +20"),
            (resilience, "resilience_high +15"),
            (message_not_hot, "message_not_overheated +10"),
            (inflow, "exchange_inflow_spike -30"),
            (message_hot, "message_overheated -20"),
            (below_ma_weak_depth, "below_ma200_weak_depth -25"),
        ):
            if fired:
                explanations.append(text)
                if " -" in text:
                    negatives.append(text)
                else:
                    positives.append(text)
        coverage_score = float(_value(row, "source_coverage_score", 0.0) or 0.0)
        data_warning = str(row.get("data_quality_warning") or "")
        missing = _missing_evidence(data_warning)
        confidence = _confidence_level(row, missing, coverage_score)
        structure_state = _structure_state(row)
        flow_state = _flow_state(row, missing)
        attention_state = _attention_state(row, missing)
        activation_state = _activation_state(row)
        risk_state = _risk_state(row, missing, confidence)
        preparation_state = _preparation_state(structure_state, risk_state, coverage_score)
        suggestion_type = _suggestion_type(
            structure_state=structure_state,
            preparation_state=preparation_state,
            flow_state=flow_state,
            attention_state=attention_state,
            activation_state=activation_state,
            risk_state=risk_state,
            confidence_level=confidence,
        )
        rows.append(
            {
                "timestamp": int(row["timestamp"]),
                "symbol": str(row["symbol"]),
                "score_total": total,
                "alert_level": _alert(total, cfg),
                "onchain_score": onchain_score,
                "orderbook_score": orderbook_score,
                "trade_score": trade_score,
                "message_score": message_score,
                "negative_score": negative_score,
                "flow_outflow_3sigma_2h": flow_outflow,
                "whale_accumulation_high": whale,
                "depth_support_on_down_day": depth,
                "resilience_high": resilience,
                "message_not_overheated": message_not_hot,
                "exchange_inflow_spike": inflow,
                "message_overheated": message_hot,
                "below_ma200_weak_depth": below_ma_weak_depth,
                "explanation": ";".join(explanations) or "no_rules_fired",
                "positive_components": ";".join(positives),
                "negative_filters": ";".join(negatives),
                "source_coverage_score": coverage_score,
                "confidence_level": confidence,
                "structure_state": structure_state,
                "preparation_state": preparation_state,
                "flow_state": flow_state,
                "attention_state": attention_state,
                "activation_state": activation_state,
                "risk_state": risk_state,
                "suggestion_type": suggestion_type,
                "missing_evidence": missing,
                "data_quality_warning": data_warning,
            }
        )
    return pl.DataFrame(rows, schema=SCORE_SCHEMA)


def _missing_evidence(warning: str) -> str:
    if not warning:
        return ""
    parts = []
    for token in warning.split(";"):
        if token.endswith("_missing") or token.endswith("metadata_missing"):
            parts.append(token)
    return ";".join(dict.fromkeys(parts))


def _confidence_level(row: dict[str, Any], missing: str, coverage_score: float) -> str:
    if row.get("close") is None or "price_missing" in missing:
        return "blocked"
    families = 0
    if (
        row.get("depth_imbalance_10_mean") is not None
        or row.get("depth_imbalance_25_mean") is not None
    ):
        families += 1
    trade_available = (
        "trade_notional_metadata_missing" not in missing
        and (
            row.get("resilience_score") is not None
            or row.get("large_trade_buy_ratio") is not None
        )
    )
    if trade_available:
        families += 1
    if row.get("flow_zscore") is not None or row.get("whale_accumulation_ratio") is not None:
        families += 1
    if row.get("funding_rate") is not None or row.get("open_interest_change_24h") is not None:
        families += 1
    if row.get("mention_growth") is not None:
        families += 1
    if (
        coverage_score >= 0.85
        and families >= 3
        and (trade_available or row.get("flow_zscore") is not None)
    ):
        return "high"
    if coverage_score >= 0.55 and families >= 2:
        return "medium"
    return "low"


def _structure_state(row: dict[str, Any]) -> str:
    if row.get("close") is None:
        return "unknown"
    if _value(row, "quote_volume_24h", 0.0) < 1_000_000.0:
        return "illiquid"
    stage = str(row.get("structure_stage") or "")
    range_pos = row.get("range_position_pct")
    compression = row.get("volatility_compression_pctile")
    if stage == "breakout":
        return "breakout"
    if range_pos is not None and float(range_pos) <= 0.25:
        return "range_low"
    if compression is not None and float(compression) <= 0.35:
        return "compressed"
    if range_pos is not None and float(range_pos) >= 0.80:
        return "extended"
    return "range_mid"


def _flow_state(row: dict[str, Any], missing: str) -> str:
    if "onchain_missing" in missing and row.get("funding_rate") is None:
        return "missing"
    flow_z = row.get("flow_zscore")
    depth = row.get("depth_imbalance_10_mean") or row.get("depth_imbalance_25_mean")
    resilience = row.get("resilience_score")
    if flow_z is not None and float(flow_z) > 3.0:
        return "distribution_like"
    positives = [
        flow_z is not None and float(flow_z) < -3.0,
        depth is not None and float(depth) > 0.30,
        resilience is not None and float(resilience) > 0.60,
    ]
    if sum(bool(value) for value in positives) >= 2:
        return "accumulation_like"
    if any(value is not None for value in (flow_z, depth, resilience, row.get("funding_rate"))):
        return "neutral"
    return "unknown"


def _attention_state(row: dict[str, Any], missing: str) -> str:
    mention_growth = row.get("mention_growth")
    emotion = _value(row, "emotion_news_ratio", 0.0)
    fundamental = _value(row, "fundamental_news_ratio", 0.0)
    polymarket_count = _value(row, "polymarket_related_market_count", 0) or 0
    if "polymarket_unmatched" in missing:
        return "unknown"
    if mention_growth is not None and float(mention_growth) >= 3.0 and emotion > fundamental:
        return "hype_hot"
    if mention_growth is not None and float(mention_growth) >= 1.5:
        return "narrative_rising"
    if int(polymarket_count) > 0:
        return "narrative_rising"
    if mention_growth is not None and fundamental >= emotion:
        return "fundamental_led"
    return "unknown"


def _activation_state(row: dict[str, Any]) -> str:
    if row.get("close") is None:
        return "inactive"
    ret_24h = _value(row, "return_24h", 0.0) or 0.0
    stage = str(row.get("structure_stage") or "")
    if stage == "breakout" and ret_24h > 0.10:
        return "overextended"
    if stage == "breakout":
        return "active"
    if ret_24h > 0.03:
        return "early"
    return "inactive"


def _risk_state(row: dict[str, Any], missing: str, confidence_level: str) -> str:
    warning = str(row.get("data_quality_warning") or "")
    if confidence_level == "blocked" or "price_missing" in missing:
        return "data_insufficient"
    if _value(row, "quote_volume_24h", 0.0) < 1_000_000.0:
        return "liquidity_risk"
    if _value(row, "polymarket_event_driven_context", False) or "event_driven_context" in warning:
        return "event_sensitive"
    if "stale" in warning or "missing" in missing:
        return "source_conflicted"
    return "normal"


def _preparation_state(structure_state: str, risk_state: str, coverage_score: float) -> str:
    if risk_state in {"data_insufficient", "liquidity_risk"}:
        return "blocked"
    if structure_state in {"compressed", "range_low"} and coverage_score >= 0.35:
        return "prepare_watch"
    if structure_state in {"range_mid", "breakout"} and coverage_score >= 0.55:
        return "watchlist"
    return "not_ready"


def _suggestion_type(
    *,
    structure_state: str,
    preparation_state: str,
    flow_state: str,
    attention_state: str,
    activation_state: str,
    risk_state: str,
    confidence_level: str,
) -> str:
    if preparation_state == "blocked" or confidence_level == "blocked":
        return "reject_or_deprioritize"
    if risk_state == "event_sensitive" and preparation_state in {"prepare_watch", "watchlist"}:
        return "event_sensitive_watch"
    if activation_state in {"active", "overextended"}:
        return "trend_active_review"
    if flow_state == "accumulation_like" and structure_state not in {
        "illiquid",
        "extended",
        "unknown",
    }:
        return "flow_confirmed_watch"
    if attention_state in {"narrative_rising", "hype_hot", "fundamental_led"}:
        return "monitor_context"
    if preparation_state in {"prepare_watch", "watchlist"}:
        return "prepare_watch"
    return "reject_or_deprioritize"
