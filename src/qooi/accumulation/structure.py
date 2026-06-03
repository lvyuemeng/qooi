"""Pure structure evaluation helpers for accumulation readouts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StructureSettings:
    low_position_threshold: float = 0.35
    balanced_position_threshold: float = 0.50
    near_resistance_position_threshold: float = 0.75
    min_target_upside_pct: float = 0.05
    min_balanced_reward_risk: float = 1.0
    min_favorable_reward_risk: float = 1.5
    min_invalidation_buffer_pct: float = 0.03
    max_invalidation_buffer_pct: float = 0.08
    invalidation_buffer_range_fraction: float = 0.05


@dataclass(frozen=True)
class StructureEvaluation:
    stage: str
    quality: str
    verdict: str
    plain_reason: str
    current_px: float | None
    support_px: float | None
    target_px: float | None
    position_pct: float | None
    upside_to_target_pct: float | None
    risk_to_support_pct: float | None
    buffered_invalidation_px: float | None
    risk_to_invalidation_pct: float | None
    reward_risk: float | None
    blockers: tuple[str, ...]


def evaluate_structure_row(
    row: dict[str, object], settings: StructureSettings = StructureSettings()
) -> StructureEvaluation:
    current = None if row["close"] is None else float(row["close"])
    support = None if row["range_low_px"] is None else float(row["range_low_px"])
    target = None if row["range_high_px"] is None else float(row["range_high_px"])
    position = None if row["range_position_pct"] is None else float(row["range_position_pct"])
    upside = (
        None
        if row["upside_to_range_high_pct"] is None
        else float(row["upside_to_range_high_pct"])
    )
    risk_to_support = (
        None
        if row["downside_to_range_low_pct"] is None
        else float(row["downside_to_range_low_pct"])
    )
    raw_rr = None if row["range_reward_risk"] is None else float(row["range_reward_risk"])
    buffered_invalidation, risk_to_invalidation = _buffered_invalidation(
        current, support, target, settings
    )
    reward_risk = _reward_risk(upside, risk_to_invalidation, raw_rr)
    quality = _range_quality(position, upside, reward_risk, settings)
    stage = setup_stage(row, quality)
    blockers = late_move_blockers(row, quality)
    verdict = structure_verdict(row, quality, stage, blockers)
    return StructureEvaluation(
        stage=stage,
        quality=quality,
        verdict=verdict,
        plain_reason=_plain_reason(quality, verdict, upside, risk_to_invalidation, reward_risk),
        current_px=current,
        support_px=support,
        target_px=target,
        position_pct=position,
        upside_to_target_pct=upside,
        risk_to_support_pct=risk_to_support,
        buffered_invalidation_px=buffered_invalidation,
        risk_to_invalidation_pct=risk_to_invalidation,
        reward_risk=reward_risk,
        blockers=blockers,
    )


def setup_stage(row: dict[str, object], quality: str | None = None) -> str:
    if not row:
        return "unknown"
    structure = str(row["structure_state"] or "")
    activation = str(row["activation_state"] or "")
    quality = quality or evaluate_structure_row(row).quality
    if structure == "compressed":
        return "compressed_setup"
    if structure == "range_low":
        return "low_range_setup"
    if activation == "early" and quality in {"favorable_range_setup", "balanced_range_setup"}:
        return "early_setup"
    if structure == "breakout":
        return "breakout_watch"
    if structure == "extended" or activation == "overextended":
        return "extended_or_late"
    if structure == "range_mid":
        return "trend_mid"
    return "unknown"


def order_preparation_state(row: dict[str, object]) -> str:
    if not row:
        return "missing"
    positive = str(row["top_positive_components"] or row["positive_components"] or "")
    if "depth_support_on_down_day" in positive:
        return "order_supported"
    if "resilience_high" in positive:
        return "resilience_supported"
    if "whale_accumulation_high" in positive or "exchange_outflow_zscore_streak" in positive:
        return "flow_supported"
    if float(row["depth_imbalance_25_mean"] or 0.0) >= 0.25:
        return "order_supported"
    if float(row["resilience_score"] or 0.0) >= 0.55:
        return "resilience_supported"
    if float(row["large_trade_buy_ratio"] or 0.0) >= 0.60:
        return "order_supported"
    if any(row[name] is not None for name in _ORDER_EVIDENCE_FIELDS):
        return "mixed_or_neutral"
    return "missing"


def late_move_blockers(row: dict[str, object], quality: str | None = None) -> tuple[str, ...]:
    blockers = []
    structure = str(row["structure_state"] or "")
    activation = str(row["activation_state"] or "")
    positive = str(row["top_positive_components"] or row["positive_components"] or "")
    quality = quality or evaluate_structure_row(row).quality
    if structure == "extended":
        blockers.append("extended_structure")
    if activation == "overextended":
        blockers.append("overextended_activation")
    if structure == "breakout" and not positive.strip():
        blockers.append("breakout_without_accumulation_evidence")
    if quality in {"near_resistance", "poor_range_reward"}:
        blockers.append(quality)
    return tuple(dict.fromkeys(blockers))


def structure_verdict(
    row: dict[str, object], quality: str, stage: str, blockers: tuple[str, ...]
) -> str:
    order = order_preparation_state(row)
    if not row:
        return "data_blocked"
    if any(token in blockers for token in ("extended_structure", "overextended_activation")):
        return "avoid_late"
    if quality in {"near_resistance", "poor_range_reward"}:
        return "wait_pullback"
    if quality in {"favorable_range_setup", "balanced_range_setup"} and stage in {
        "compressed_setup",
        "low_range_setup",
        "early_setup",
        "unknown",
    }:
        return "review_now"
    if order in {"order_supported", "resilience_supported", "flow_supported"}:
        return "watch_orderbook"
    if stage in {"compressed_setup", "low_range_setup", "early_setup"}:
        return "needs_confirmation"
    if quality == "range_unknown":
        return "data_blocked"
    return "wait_pullback"


def _range_quality(
    position: float | None,
    upside: float | None,
    reward_risk: float | None,
    settings: StructureSettings,
) -> str:
    if position is None or upside is None or reward_risk is None:
        return "range_unknown"
    if position >= settings.near_resistance_position_threshold or upside < 0.03:
        return "near_resistance"
    if upside >= 0.0 and reward_risk < settings.min_balanced_reward_risk:
        return "poor_range_reward"
    if (
        position <= settings.low_position_threshold
        and upside >= settings.min_target_upside_pct
        and reward_risk >= settings.min_favorable_reward_risk
    ):
        return "favorable_range_setup"
    if (
        position <= settings.balanced_position_threshold
        and reward_risk >= settings.min_balanced_reward_risk
    ):
        return "balanced_range_setup"
    return "range_mid"


def _buffered_invalidation(
    current: float | None,
    support: float | None,
    target: float | None,
    settings: StructureSettings,
) -> tuple[float | None, float | None]:
    if current is None or support is None or target is None or current <= 0 or support <= 0:
        return None, None
    range_width_pct = max((target - support) / current, 0.0)
    buffer_pct = max(
        settings.min_invalidation_buffer_pct,
        min(
            settings.max_invalidation_buffer_pct,
            range_width_pct * settings.invalidation_buffer_range_fraction,
        ),
    )
    buffered = support * (1.0 - buffer_pct)
    if buffered <= 0:
        return None, None
    return buffered, (current / buffered) - 1.0


def _reward_risk(
    upside: float | None, risk_to_invalidation: float | None, raw_rr: float | None
) -> float | None:
    if upside is None:
        return None
    if risk_to_invalidation is not None and risk_to_invalidation > 0:
        return upside / risk_to_invalidation
    return raw_rr


def _plain_reason(
    quality: str,
    verdict: str,
    upside: float | None,
    risk: float | None,
    reward_risk: float | None,
) -> str:
    if quality in {"favorable_range_setup", "balanced_range_setup"}:
        return "near support with favorable upside/risk"
    if quality == "near_resistance":
        return "near resistance; wait for pullback toward support"
    if quality == "poor_range_reward":
        return "poor reward/risk; wait for pullback toward support"
    if verdict == "watch_orderbook":
        return "order-book support present, but structure needs confirmation"
    if quality == "range_unknown":
        return "range structure unavailable from current deep data"
    if upside is not None and risk is not None and reward_risk is not None:
        return "mid-range structure with limited current edge"
    return "structure needs confirmation"


_ORDER_EVIDENCE_FIELDS = (
    "depth_imbalance_25_mean",
    "resilience_score",
    "large_trade_buy_ratio",
    "open_interest_change_24h",
)
