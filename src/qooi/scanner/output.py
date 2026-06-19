"""Scanner output — review and report."""

from __future__ import annotations

from dataclasses import dataclass
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
    if not freshness.is_empty():
        stale = freshness.filter(pl.col("prediction_freshness") != "fresh")
        if not stale.is_empty():
            return [
                ReviewDecision(
                    str(row.get("symbol", "?")),
                    "skip",
                    f"{row.get('prediction_freshness')} prediction "
                    f"({row.get('prediction_age_hours', 0):.1f}h old)",
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
    rows = ranked.to_dicts()
    best_direction_by_symbol: dict[str, str] = {}
    best_score_by_symbol: dict[str, float] = {}
    directions_by_symbol: dict[str, set[str]] = {}
    for row in rows:
        symbol = str(row.get("symbol", "?"))
        direction = str(row.get("direction") or "")
        score = float(row.get("rank_score") or 0.0)
        if direction:
            directions_by_symbol.setdefault(symbol, set()).add(direction)
        if symbol not in best_score_by_symbol or score > best_score_by_symbol[symbol]:
            best_score_by_symbol[symbol] = score
            best_direction_by_symbol[symbol] = direction

    promote_limit = len(rows)
    if tailtree_selection is not None and tailtree_selection.top_k:
        ordered_top_k = sorted(int(value) for value in tailtree_selection.top_k if int(value) > 0)
        promote_limit = ordered_top_k[1] if len(ordered_top_k) > 1 else ordered_top_k[0]
    promoted = 0
    results: list[ReviewDecision] = []
    for row in rows:
        symbol = str(row.get("symbol", "?"))
        direction = str(row.get("direction") or "")
        score = float(row.get("rank_score") or 0.0)
        support = float(row.get("support_count") or 0.0)
        tail_lift = float(row.get("tail_lift") or 0.0)
        source_freshness = str(row.get("source_freshness") or "")
        missing_sources = int(row.get("required_missing_source_count") or 0)
        stale_sources = int(row.get("required_stale_source_count") or 0)
        comparable_surface = "branch" in row and "support_count" in row
        matched = support > 0.0 if comparable_surface else score > 0.0

        if support <= 0.0 or not matched:
            results.append(ReviewDecision(symbol, "skip", "no matching evidence"))
            continue
        if missing_sources > 0:
            results.append(
                ReviewDecision(symbol, "skip", f"missing required sources={missing_sources}")
            )
            continue
        if stale_sources > 0 or source_freshness == "stale":
            results.append(
                ReviewDecision(symbol, "watch", f"stale required sources={stale_sources}")
            )
            continue
        if comparable_surface and tailtree_selection is not None:
            min_support = float(tailtree_selection.min_selected_observation_count)
            min_tail_lift = float(tailtree_selection.min_valid_tail_lift)
            if support < min_support:
                results.append(
                    ReviewDecision(symbol, "skip", f"support {support:.0f} < {min_support:.0f}")
                )
                continue
            if tail_lift < min_tail_lift:
                results.append(
                    ReviewDecision(
                        symbol, "watch", f"tail_lift {tail_lift:.3f} < {min_tail_lift:.3f}"
                    )
                )
                continue
        if len(
            directions_by_symbol.get(symbol, set())
        ) > 1 and direction != best_direction_by_symbol.get(symbol):
            results.append(
                ReviewDecision(symbol, "watch", f"conflicting weaker {direction} direction")
            )
            continue
        if promoted >= promote_limit:
            results.append(ReviewDecision(symbol, "watch", f"outside promote top_{promote_limit}"))
            continue
        action: Literal["promote", "watch", "skip"] = "promote" if score > 0.0 else "watch"
        if action == "promote":
            promoted += 1
        results.append(
            ReviewDecision(symbol, action, f"score={score:.3f} tail_lift={tail_lift:.3f}")
        )
    return results if results else [ReviewDecision("*", "skip", "no candidates")]


def _status_counts(plan: CoveragePlan | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    if plan is None:
        return counts
    for state in plan.states:
        counts[state.status] = counts.get(state.status, 0) + 1
    return counts


def _readiness_line(name: str, plan: CoveragePlan | None) -> str:
    counts = _status_counts(plan)
    statuses = " ".join(f"{status}={count}" for status, count in sorted(counts.items()))
    jobs = len(plan.jobs) if plan is not None else 0
    pages = plan.allocated_pages() if plan is not None else 0
    remaining = plan.estimated_pages if plan is not None else 0
    return (
        f"{BULLET} **{name}**: {statuses or 'missing'} | "
        f"jobs={jobs} pages={pages} remaining={remaining}"
    )


def render_report(frames: ScannerRunFrames, config: PotentialConfig) -> str:
    lines = ["# Scanner Report", f"Generated from: {config.output}\n"]

    market = frames.market
    lines.append(SEP)
    lines.append("## Market Data Readiness")
    lines.append(SEP)
    lines.append(
        f"Symbols: {market.symbols} | Timeframes: {market.timeframes} | "
        f"Target days: {market.target_days} | Source products: {market.source_products}"
    )
    lines.append(
        f"Executed pages: bars={market.stats.bar_pages:_} | sources={market.stats.source_pages} | "
        f"provider_bounded={market.stats.provider_bounded}"
    )
    lines.append("Before:")
    for name, plan in sorted(market.before.items()):
        lines.append(_readiness_line(name, plan))
    lines.append("After:")
    for name, plan in sorted(market.after.items()):
        lines.append(_readiness_line(name, plan))
    lines.append("")

    if bars := frames.products.get("bars"):
        h = bars.health
        lines.append(SEP)
        lines.append("## Training Data Completeness")
        lines.append(SEP)
        lines.append(
            f"Rows: {h.actual_rows:_} | Coverage: {h.coverage_pct:.1f}% | "
            f"Age: {h.age_hours:.1f}h | Gaps: {h.gaps} | Duplicates: {h.duplicates} | "
            f"Status: {check_bar_health(h)}"
        )
        lines.append("")

    lines.append(SEP)
    lines.append("## Source Completeness")
    lines.append(SEP)
    for name, result in sorted((p, r) for p, r in frames.products.items() if p != "bars"):
        h = result.health
        lines.append(
            f"{BULLET} **{name}**: {h.status} | {h.actual_rows:_} rows | "
            f"age={h.age_hours:.1f}h | coverage={h.coverage_pct:.1f}%"
        )
    if len(frames.products) == 1:
        lines.append(f"{BULLET} no enabled source products")
    lines.append("")

    lines.append(SEP)
    lines.append("## Prediction Freshness")
    lines.append(SEP)
    if frames.prediction_freshness.is_empty():
        lines.append(f"{BULLET} no prediction candidates")
    else:
        for row in frames.prediction_freshness.head(20).to_dicts():
            lines.append(
                f"{BULLET} {row.get('symbol')}: {row.get('prediction_freshness')} | "
                f"age={float(row.get('prediction_age_hours') or 0.0):.1f}h"
            )
    lines.append("")

    if frames.transitions is not None and frames.transitions.insights:
        lines.append(SEP)
        lines.append("## Kline + Transition Summary")
        lines.append(SEP)
        for insight in list(frames.transitions.insights.values())[:10]:
            lines.append(
                f"{BULLET} {insight.symbol}: consensus={insight.consensus} "
                f"patterns={len(insight.patterns)}"
            )
        lines.append("")

    lines.append(SEP)
    lines.append("## Ladder Evidence")
    lines.append(SEP)
    if frames.ladder.is_empty():
        lines.append(f"{BULLET} no ladder evidence")
    else:
        selected = (
            frames.ladder.filter(pl.col("selected_evidence_level"))
            if "selected_evidence_level" in frames.ladder.columns
            else frames.ladder
        )
        for row in selected.head(10).to_dicts():
            lines.append(
                f"{BULLET} {row.get('evidence_level')}: "
                f"support={row.get('conditioned_observations')} "
                f"gain={float(row.get('information_gain_bits') or 0.0):.4f} "
                f"lift_up={float(row.get('lift_up') or 0.0):.3f} "
                f"lift_down={float(row.get('lift_down') or 0.0):.3f}"
            )
    lines.append("")

    lines.append(SEP)
    lines.append("## Tailtree Evidence")
    lines.append(SEP)
    tailtree = config.evidence.tailtree
    profile_text = ", ".join(
        f"{profile.profile_id}:{profile.objective}/{profile.training.kind}"
        for profile in tailtree.profiles
    )
    lines.append(f"Lifecycle: {tailtree.lifecycle} | Profiles: {profile_text or 'none'}")
    if frames.tailtree.is_empty():
        lines.append(f"{BULLET} no tailtree evidence")
    else:
        for row in frames.tailtree.head(10).to_dicts():
            lines.append(
                f"{BULLET} {row.get('tree_direction')} "
                f"leaf={row.get('leaf_id', row.get('score_bucket'))}: "
                f"N={row.get('N_total')} tails={row.get('N_tail_exceedances')} "
                f"tail_lift={float(row.get('tail_lift') or 0.0):.3f} "
                f"utility={float(row.get('tail_utility_mean') or 0.0):.3f}"
            )
    lines.append("")

    lines.append(SEP)
    lines.append("## Comparable Candidate Surface")
    lines.append(SEP)
    if frames.ranked.is_empty():
        lines.append(f"{BULLET} no ranked candidates")
    else:
        if "branch" in frames.ranked.columns:
            branch_counts = frames.ranked.group_by("branch").len().sort("branch")
            lines.append(
                "Branch counts: "
                + ", ".join(f"{row['branch']}={row['len']}" for row in branch_counts.to_dicts())
            )
        if "candidate_status" in frames.ranked.columns:
            status_counts = (
                frames.ranked.group_by("candidate_status").len().sort("candidate_status")
            )
            lines.append(
                "Candidate status counts: "
                + ", ".join(
                    f"{row['candidate_status']}={row['len']}" for row in status_counts.to_dicts()
                )
            )
        for row in frames.ranked.head(10).to_dicts():
            lines.append(
                f"{BULLET} {row.get('branch', '?')} {row.get('symbol')}: "
                f"score={float(row.get('rank_score') or 0.0):.3f} "
                f"tail_lift={float(row.get('tail_lift') or 0.0):.3f} "
                f"support={float(row.get('support_count') or 0.0):.0f}"
            )
    lines.append("")

    lines.append(SEP)
    lines.append("## Horizon Consistency")
    lines.append(SEP)
    if frames.horizon_consistency.is_empty():
        lines.append(f"{BULLET} no horizon consistency rows")
    else:
        for row in frames.horizon_consistency.head(10).to_dicts():
            lines.append(
                f"{BULLET} {row.get('symbol')} {row.get('tree_direction')}: "
                f"horizons={row.get('horizon_count')} "
                f"score={float(row.get('consistency_rank_score') or 0.0):.3f}"
            )
    lines.append("")

    lines.append(SEP)
    lines.append("## Decisions")
    lines.append(SEP)
    promote = [d for d in frames.decisions if d.action == "promote"]
    watch = [d for d in frames.decisions if d.action == "watch"]
    skip = [d for d in frames.decisions if d.action == "skip"]
    lines.append(f"Promote: {len(promote)} | Watch: {len(watch)} | Skip: {len(skip)}")
    for decision in frames.decisions:
        emoji = {"promote": "✅", "watch": "👀", "skip": "❌"}[decision.action]
        lines.append(
            f"{BULLET} {emoji} {decision.symbol}: **{decision.action}** — {decision.reason}"
        )

    return "\n".join(lines)
