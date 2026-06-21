"""Scanner output — review and report."""

from __future__ import annotations

from dataclasses import dataclass
from math import log1p
from typing import Literal

import polars as pl

from qooi.pipeline.coverage import CoveragePlan
from qooi.pipeline.load import LoadStats
from qooi.pipeline.types import FrameHealth, ProductResult
from qooi.scanner import TransitionAnalysis
from qooi.scanner.config import PotentialConfig

BULLET = "-"
SEP = "=" * 60


@dataclass(frozen=True)
class ReviewDecision:
    symbol: str
    action: Literal["promote", "watch", "skip"]
    reason: str
    direction: str = ""
    horizon: int | None = None
    score: float = 0.0
    tail_lift: float = 0.0
    support: float = 0.0
    utility: float = 0.0
    freshness: str = ""
    age_hours: float | None = None
    source_freshness: str = ""


@dataclass(frozen=True)
class MarketReadiness:
    symbols: int
    timeframes: int
    target_days: int
    source_products: int
    before: dict[str, CoveragePlan]
    after: dict[str, CoveragePlan]
    stats: LoadStats


@dataclass(frozen=True)
class ScannerRunFrames:
    market: MarketReadiness
    products: dict[str, ProductResult]
    states: dict[tuple[str, str], pl.DataFrame]
    transitions: TransitionAnalysis | None
    histories: pl.DataFrame
    source_events: pl.DataFrame
    ladder: pl.DataFrame
    tailtree: pl.DataFrame
    ranked: pl.DataFrame
    horizon_consistency: pl.DataFrame
    action_surface: pl.DataFrame
    prediction_freshness: pl.DataFrame
    decisions: list[ReviewDecision]


def check_bar_health(health: FrameHealth) -> str:
    if health.actual_rows <= 0:
        return "no bars"
    if health.coverage_pct < 80.0:
        return f"low coverage ({health.coverage_pct:.0f}%)"
    if health.gaps > 10:
        return f"many gaps ({health.gaps})"
    return "ok"


def check_source_health(products: dict[str, FrameHealth]) -> str:
    issues = []
    for product, h in products.items():
        if h.status == "missing":
            issues.append(f"{product}: missing")
        elif h.actual_rows <= 0:
            issues.append(f"{product}: no data")
    return "; ".join(issues) if issues else "ok"


def review_decisions(
    ranked: pl.DataFrame,
    freshness: pl.DataFrame,
    source_health: dict[str, FrameHealth],
    config: PotentialConfig,
) -> list[ReviewDecision]:
    """Promote/watch/skip; only latest prediction rows are freshness-gated."""
    freshness_by_symbol = _freshness_by_symbol(freshness)
    if not freshness.is_empty():
        stale = freshness.filter(pl.col("prediction_freshness") != "fresh")
        if not stale.is_empty():
            return [
                ReviewDecision(
                    str(row.get("symbol", "?")),
                    "skip",
                    (
                        f"{row.get('prediction_freshness')} prediction "
                        f"({float(row.get('prediction_age_hours') or 0.0):.1f}h old)"
                    ),
                    freshness=str(row.get("prediction_freshness") or ""),
                    age_hours=float(row.get("prediction_age_hours") or 0.0),
                )
                for row in stale.to_dicts()
            ]
    source_bad = [p for p, h in source_health.items() if h.status == "missing"]
    if source_bad:
        return [ReviewDecision("*", "skip", f"missing sources: {', '.join(source_bad)}")]

    if ranked.is_empty():
        return [ReviewDecision("*", "skip", "no candidates")]

    tailtree_selection = (
        config.evidence.tailtree.selection if config.evidence.kind == "tailtree" else None
    )
    raw_rows = ranked.to_dicts()
    conflict_symbols = _material_conflict_symbols(raw_rows, tailtree_selection)
    best_row_by_symbol: dict[str, dict[str, object]] = {}
    best_quality_by_symbol: dict[str, float] = {}
    directions_by_symbol: dict[str, set[str]] = {}
    for row in raw_rows:
        symbol = str(row.get("symbol", "?"))
        direction = str(row.get("direction") or "")
        if direction:
            directions_by_symbol.setdefault(symbol, set()).add(direction)
        quality = _side_quality_score(row)
        if symbol not in best_quality_by_symbol or quality > best_quality_by_symbol[symbol]:
            best_quality_by_symbol[symbol] = quality
            best_row_by_symbol[symbol] = row

    rows = list(best_row_by_symbol.values())
    rows.sort(key=lambda row: float(row.get("rank_score") or 0.0), reverse=True)

    promote_limit = len(rows)
    if tailtree_selection is not None and tailtree_selection.top_k:
        ordered_top_k = sorted(int(value) for value in tailtree_selection.top_k if int(value) > 0)
        promote_limit = ordered_top_k[1] if len(ordered_top_k) > 1 else ordered_top_k[0]
    promoted = 0
    results: list[ReviewDecision] = []
    for row in rows:
        symbol = str(row.get("symbol", "?"))
        direction = str(row.get("direction") or "")
        score = _float_value(row.get("rank_score"))
        support = _float_value(row.get("support_count"))
        tail_lift = _float_value(row.get("tail_lift"))
        utility = _float_value(row.get("utility_proxy"))
        source_freshness = str(row.get("source_freshness") or "")
        missing_sources = _int_value(row.get("required_missing_source_count"))
        stale_sources = _int_value(row.get("required_stale_source_count"))
        horizon = _int_or_none(row.get("outcome_horizon"))
        fresh_status, age_hours = freshness_by_symbol.get(symbol, ("", None))
        comparable_surface = "branch" in row and "support_count" in row
        matched = support > 0.0 if comparable_surface else score > 0.0

        decision = _decision_row(
            symbol=symbol,
            action="watch",
            reason="",
            direction=direction,
            horizon=horizon,
            score=score,
            tail_lift=tail_lift,
            support=support,
            utility=utility,
            freshness=fresh_status,
            age_hours=age_hours,
            source_freshness=source_freshness,
        )

        if support <= 0.0 or not matched:
            results.append(_with_decision(decision, "skip", "no matching evidence"))
            continue
        if missing_sources > 0:
            results.append(
                _with_decision(decision, "skip", f"missing required sources={missing_sources}")
            )
            continue
        if stale_sources > 0 or source_freshness == "stale":
            results.append(
                _with_decision(decision, "watch", f"stale required sources={stale_sources}")
            )
            continue
        if comparable_surface and tailtree_selection is not None:
            min_support = float(tailtree_selection.min_selected_observation_count)
            min_tail_lift = float(tailtree_selection.min_valid_tail_lift)
            if support < min_support:
                results.append(
                    _with_decision(decision, "skip", f"support {support:.0f} < {min_support:.0f}")
                )
                continue
            if tail_lift < min_tail_lift:
                results.append(
                    _with_decision(
                        decision, "watch", f"tail_lift {tail_lift:.3f} < {min_tail_lift:.3f}"
                    )
                )
                continue
        dropped_directions = sorted(directions_by_symbol.get(symbol, set()) - {direction})
        if symbol in conflict_symbols:
            directions = " vs ".join(sorted(directions_by_symbol.get(symbol, set())))
            results.append(
                _with_decision(
                    decision,
                    "watch",
                    f"direction conflict: {directions}; abstain from promotion",
                )
            )
            continue
        if promoted >= promote_limit:
            results.append(
                _with_decision(decision, "watch", f"outside promote top_{promote_limit}")
            )
            continue
        action: Literal["promote", "watch", "skip"] = "promote" if score > 0.0 else "watch"
        if action == "promote":
            promoted += 1
        reason = _action_reason(decision)
        if dropped_directions:
            reason = f"{reason}; resolved opposite direction: {', '.join(dropped_directions)}"
        results.append(_with_decision(decision, action, reason))
    return results if results else [ReviewDecision("*", "skip", "no candidates")]


def _material_conflict_symbols(rows: list[dict[str, object]], selection: object | None) -> set[str]:
    min_support = 0.0
    min_tail_lift = 1.0
    if selection is not None:
        min_support = float(getattr(selection, "min_selected_observation_count", 0.0))
        min_tail_lift = float(getattr(selection, "min_valid_tail_lift", 1.0))
    material: dict[str, set[str]] = {}
    for row in rows:
        symbol = str(row.get("symbol", "?"))
        direction = str(row.get("direction") or "")
        if not direction:
            continue
        if _float_value(row.get("rank_score")) <= 0.0:
            continue
        if _float_value(row.get("support_count")) < min_support:
            continue
        if _float_value(row.get("tail_lift")) < min_tail_lift:
            continue
        material.setdefault(symbol, set()).add(direction)
    return {symbol for symbol, directions in material.items() if len(directions) > 1}


def _side_quality_score(row: dict[str, object]) -> float:
    """Promotion-side tie breaker for same-symbol opposite-direction rows."""
    tail_lift = _float_value(row.get("tail_lift"))
    support = _float_value(row.get("support_count"))
    utility = _float_value(row.get("utility_proxy"))
    return tail_lift + log1p(max(support, 0.0)) + utility


def _float_value(value: object) -> float:
    if isinstance(value, int | float | str):
        return float(value)
    return 0.0


def _int_value(value: object) -> int:
    if isinstance(value, int | float | str):
        return int(value)
    return 0


def _status_counts(plan: CoveragePlan | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    if plan is None:
        return counts
    for state in plan.states:
        counts[state.status] = counts.get(state.status, 0) + 1
    return counts


def _status_counts(plan: CoveragePlan | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    if plan is None:
        return counts
    for state in plan.states:
        counts[state.status] = counts.get(state.status, 0) + 1
    return counts


def _freshness_by_symbol(frame: pl.DataFrame) -> dict[str, tuple[str, float | None]]:
    if frame.is_empty():
        return {}
    result: dict[str, tuple[str, float | None]] = {}
    for row in frame.to_dicts():
        result[str(row.get("symbol", "?"))] = (
            str(row.get("prediction_freshness") or ""),
            _float_or_none(row.get("prediction_age_hours")),
        )
    return result


def _decision_row(
    *,
    symbol: str,
    action: Literal["promote", "watch", "skip"],
    reason: str,
    direction: str,
    horizon: int | None,
    score: float,
    tail_lift: float,
    support: float,
    utility: float,
    freshness: str,
    age_hours: float | None,
    source_freshness: str,
) -> ReviewDecision:
    return ReviewDecision(
        symbol=symbol,
        action=action,
        reason=reason,
        direction=direction,
        horizon=horizon,
        score=score,
        tail_lift=tail_lift,
        support=support,
        utility=utility,
        freshness=freshness,
        age_hours=age_hours,
        source_freshness=source_freshness,
    )


def _with_decision(
    decision: ReviewDecision, action: Literal["promote", "watch", "skip"], reason: str
) -> ReviewDecision:
    return ReviewDecision(
        symbol=decision.symbol,
        action=action,
        reason=reason,
        direction=decision.direction,
        horizon=decision.horizon,
        score=decision.score,
        tail_lift=decision.tail_lift,
        support=decision.support,
        utility=decision.utility,
        freshness=decision.freshness,
        age_hours=decision.age_hours,
        source_freshness=decision.source_freshness,
    )


def _action_reason(decision: ReviewDecision) -> str:
    side = _side_phrase(decision.direction)
    return f"{side}: score={decision.score:.3f} tail_lift={decision.tail_lift:.3f}"


def _int_or_none(value: object) -> int | None:
    if not isinstance(value, int | float | str):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: object) -> float | None:
    if not isinstance(value, int | float | str):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _direction_label(direction: str) -> str:
    normalized = direction.lower()
    if normalized in {"up", "bullish", "long"}:
        return f"↑ {direction or 'up'}"
    if normalized in {"down", "bearish", "short"}:
        return f"↓ {direction or 'down'}"
    return direction or "?"


def _side_phrase(direction: str) -> str:
    normalized = direction.lower()
    if normalized in {"up", "bullish", "long"}:
        return "long-bias research watch"
    if normalized in {"down", "bearish", "short"}:
        return "short-bias research watch"
    return "directional research watch"


def _horizon_label(horizon: int | None) -> str:
    return f"h{horizon}" if horizon is not None else "h?"


def _age_label(age_hours: float | None) -> str:
    return f"{age_hours:.1f}h" if age_hours is not None else "?h"


def _coverage_summary(plans: dict[str, CoveragePlan]) -> str:
    totals: dict[str, int] = {}
    for plan in plans.values():
        for status, count in _status_counts(plan).items():
            totals[status] = totals.get(status, 0) + count
    return ", ".join(f"{status}={count}" for status, count in sorted(totals.items())) or "none"


def _source_gate_lines(products: dict[str, ProductResult]) -> list[str]:
    lines = []
    for name, result in sorted((p, r) for p, r in products.items() if p != "bars"):
        h = result.health
        lines.append(
            f"{BULLET} {name}: {h.status} | rows={h.actual_rows:_} | "
            f"age={h.age_hours:.1f}h | coverage={h.coverage_pct:.1f}%"
        )
    return lines or [f"{BULLET} no enabled source products"]


def _prediction_freshness_summary(frame: pl.DataFrame) -> str:
    if frame.is_empty():
        return "no prediction candidates"
    ages = [float(row.get("prediction_age_hours") or 0.0) for row in frame.to_dicts()]
    stale = frame.filter(pl.col("prediction_freshness") != "fresh").height
    return (
        f"candidates={frame.height} | stale={stale} | age_range={min(ages):.1f}h..{max(ages):.1f}h"
    )


def _candidate_line(decision: ReviewDecision) -> str:
    emoji = {"promote": "✅", "watch": "👀", "skip": "❌"}[decision.action]
    source = decision.source_freshness or "source?"
    fresh = decision.freshness or "freshness?"
    return (
        f"{BULLET} {emoji} {decision.symbol} {_direction_label(decision.direction)} "
        f"{_horizon_label(decision.horizon)} | score={decision.score:.2f} "
        f"lift={decision.tail_lift:.2f} support={decision.support:,.0f} "
        f"utility={decision.utility:.2f} | "
        f"{fresh} {_age_label(decision.age_hours)} | sources={source} | {decision.reason}"
    )


def _direction_symbols(decisions: list[ReviewDecision], direction: str) -> str:
    names = [
        d.symbol for d in decisions if d.action == "promote" and d.direction.lower() == direction
    ]
    return ", ".join(names) if names else "none"


def _watch_reasons(decisions: list[ReviewDecision]) -> list[str]:
    counts: dict[str, list[str]] = {}
    for decision in decisions:
        if decision.action != "watch":
            continue
        key = decision.reason.split(":", 1)[0]
        counts.setdefault(key, []).append(decision.symbol)
    return [
        f"{BULLET} {reason}: {len(symbols)} symbols ({', '.join(symbols[:6])})"
        for reason, symbols in counts.items()
    ]


def _tailtree_action_surface_lines(frame: pl.DataFrame) -> list[str]:
    if frame.is_empty():
        return [f"{BULLET} no tailtree action surface"]
    action_rows = frame.group_by("actionability").agg(pl.len().alias("rows")).sort("actionability")
    side_rows = (
        frame.group_by("action_side", "actionability")
        .agg(pl.len().alias("rows"))
        .sort("action_side", "actionability")
    )
    blocker_rows = (
        frame.filter(pl.col("blocker_reason") != "")
        .group_by("action_side", "blocker_reason")
        .agg(pl.len().alias("rows"))
        .sort("action_side", "blocker_reason")
    )
    state_rows = (
        frame.group_by("action_side", "best_path_state")
        .agg(pl.len().alias("rows"))
        .sort("action_side", "best_path_state")
    )
    total = max(frame.height, 1)
    lines = [f"{BULLET} action surface rows={frame.height:_}"]
    for row in action_rows.to_dicts():
        rows = int(row["rows"])
        lines.append(
            f"{BULLET} actionability {row['actionability']}: rows={rows:_} rate={rows / total:.3f}"
        )
    lines.append("Side/action split:")
    for row in side_rows.to_dicts():
        lines.append(
            f"{BULLET} {row['action_side']} {row['actionability']}: rows={int(row['rows']):_}"
        )
    if not blocker_rows.is_empty():
        lines.append("Blockers:")
        for row in blocker_rows.to_dicts():
            lines.append(
                f"{BULLET} {row['action_side']} {row['blocker_reason']}: rows={int(row['rows']):_}"
            )
    lines.append("Path-state profile:")
    for row in state_rows.to_dicts():
        lines.append(
            f"{BULLET} {row['action_side']} {row['best_path_state']}: rows={int(row['rows']):_}"
        )
    return lines


def _tailtree_promotion_gate_lines(frame: pl.DataFrame) -> list[str]:
    if frame.is_empty():
        return [f"{BULLET} promotion gates: no action surface"]
    rows: list[str] = []
    for side in ("up", "down"):
        side_frame = frame.filter(pl.col("action_side") == side)
        trade_count = side_frame.filter(pl.col("actionability") == "trade_candidate").height
        mean_side_margin = (
            float(side_frame.select(pl.col("calibrated_side_margin").mean()).item())
            if not side_frame.is_empty() and "calibrated_side_margin" in side_frame.columns
            else 0.0
        )
        false_count = (
            int(side_frame.select(pl.col("false_direction_int").fill_null(0).sum()).item())
            if not side_frame.is_empty() and "false_direction_int" in side_frame.columns
            else 0
        )
        false_rate = false_count / side_frame.height if not side_frame.is_empty() else 0.0
        promoted = trade_count > 0 and mean_side_margin > 0.0 and false_rate < 0.20
        status = "candidate annotation" if promoted else "market-state only"
        if side == "down" and not promoted:
            status = "market-state only; do not promote short"
        rows.append(
            f"{BULLET} {side}: {status} | trade_candidates={trade_count:_} "
            f"mean_calibrated_side_margin={mean_side_margin:.3f} "
            f"false_direction_rate={false_rate:.3f}"
        )
    return rows


def render_report(frames: ScannerRunFrames, config: PotentialConfig) -> str:
    lines = ["# Scanner Report", f"Generated from: {config.output}\n"]

    market = frames.market
    bars = frames.products.get("bars")
    bar_health = bars.health if bars is not None else None

    lines.append(SEP)
    lines.append("## Run & Data Gate")
    lines.append(SEP)
    lines.append(
        f"Symbols: {market.symbols} | Timeframes: {market.timeframes} | "
        f"Target days: {market.target_days} | Source products: {market.source_products}"
    )
    lines.append(
        f"Executed pages: bars={market.stats.bar_pages:_} | "
        f"sources={market.stats.source_pages} | "
        f"provider_bounded={market.stats.provider_bounded}"
    )
    lines.append(f"Coverage after load: {_coverage_summary(market.after)}")
    if bar_health is not None:
        lines.append(
            f"Training bars: rows={bar_health.actual_rows:_} | "
            f"coverage={bar_health.coverage_pct:.1f}% | "
            f"age={bar_health.age_hours:.1f}h | gaps={bar_health.gaps} | "
            f"duplicates={bar_health.duplicates} | status={check_bar_health(bar_health)}"
        )
    lines.append("")

    lines.append(SEP)
    lines.append("## Source/Freshness Gate")
    lines.append(SEP)
    lines.append(
        f"Prediction freshness: {_prediction_freshness_summary(frames.prediction_freshness)}"
    )
    lines.extend(_source_gate_lines(frames.products))
    lines.append("")

    if frames.transitions is not None and frames.transitions.insights:
        lines.append(SEP)
        lines.append("## Kline Context")
        lines.append(SEP)
        for insight in list(frames.transitions.insights.values())[:8]:
            lines.append(
                f"{BULLET} {insight.symbol}: current kline consensus={insight.consensus} | "
                f"patterns={len(insight.patterns)}"
            )
        lines.append("")

    lines.append(SEP)
    lines.append("## Candidate Board")
    lines.append(SEP)
    if not frames.decisions:
        lines.append(f"{BULLET} no decisions")
    else:
        promote = [d for d in frames.decisions if d.action == "promote"]
        watch = [d for d in frames.decisions if d.action == "watch"]
        skip = [d for d in frames.decisions if d.action == "skip"]
        lines.append(f"Promote: {len(promote)} | Watch: {len(watch)} | Skip: {len(skip)}")
        for decision in frames.decisions[:25]:
            lines.append(_candidate_line(decision))
    lines.append("")

    lines.append(SEP)
    lines.append("## Decision Plan")
    lines.append(SEP)
    promote = [d for d in frames.decisions if d.action == "promote"]
    watch = [d for d in frames.decisions if d.action == "watch"]
    skip = [d for d in frames.decisions if d.action == "skip"]
    lines.append(f"Long-bias research watches: {_direction_symbols(promote, 'up')}")
    lines.append(f"Short-bias research watches: {_direction_symbols(promote, 'down')}")
    if promote:
        lines.append(
            "Primary action: manually inspect promoted side(s), liquidity, fees, "
            "funding, spread, and chart context before any strategy test."
        )
    else:
        lines.append(
            "Primary action: no promoted candidate; inspect watch blockers before expanding risk."
        )
    reason_lines = _watch_reasons(frames.decisions)
    if reason_lines:
        lines.append("Watch blockers:")
        lines.extend(reason_lines[:8])
    if skip:
        lines.append(f"Skipped rows: {len(skip)}; inspect missing evidence/freshness before reuse.")
    lines.append(
        "Risk note: scanner output is research evidence only; it is not live-trading authorization."
    )
    lines.append("")

    lines.append(SEP)
    lines.append("## Tailtree Action Surface")
    lines.append(SEP)
    lines.extend(_tailtree_action_surface_lines(frames.action_surface))
    lines.append("Promotion gates:")
    lines.extend(_tailtree_promotion_gate_lines(frames.action_surface))
    lines.append("")

    lines.append(SEP)
    lines.append("## Model Evidence Appendix")
    lines.append(SEP)
    tailtree = config.evidence.tailtree
    profile_text = ", ".join(
        f"{profile.profile_id}:{profile.objective}/{profile.training.kind}"
        for profile in tailtree.profiles
    )
    lines.append(f"Tailtree lifecycle: {tailtree.lifecycle} | Profiles: {profile_text or 'none'}")
    if frames.tailtree.is_empty():
        lines.append(f"{BULLET} no tailtree evidence")
    else:
        for row in frames.tailtree.head(8).to_dicts():
            lines.append(
                f"{BULLET} {_direction_label(str(row.get('tree_direction') or ''))} "
                f"bucket={row.get('leaf_id', row.get('score_bucket'))}: "
                f"N={row.get('N_total')} tails={row.get('N_tail_exceedances')} "
                f"tail_lift={float(row.get('tail_lift') or 0.0):.3f} "
                f"utility={float(row.get('tail_utility_mean') or 0.0):.3f}"
            )
    if frames.horizon_consistency.is_empty():
        lines.append(f"{BULLET} horizon consistency: no rows")
    else:
        max_horizons = max(
            int(row.get("horizon_count") or 0) for row in frames.horizon_consistency.to_dicts()
        )
        if max_horizons <= 1:
            configured = ", ".join(f"h{h}" for h in tailtree.outcome_horizon)
            lines.append(
                f"{BULLET} horizon consistency: single configured horizon ({configured}); "
                "no cross-horizon confirmation available"
            )
        else:
            lines.append("Horizon consistency:")
            for row in frames.horizon_consistency.head(6).to_dicts():
                lines.append(
                    f"{BULLET} {row.get('symbol')} "
                    f"{_direction_label(str(row.get('tree_direction') or ''))}: "
                    f"horizons={row.get('horizon_count')} strong={row.get('strong_horizon_count')} "
                    f"conflict_penalty={float(row.get('conflict_penalty_score') or 0.0):.3f}"
                )

    return "\n".join(lines)
