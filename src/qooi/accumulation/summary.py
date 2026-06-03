"""Candidate summary and rationale rendering for accumulation scanner output."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from qooi.accumulation.schema import (
    CANDIDATE_DETAIL_SCHEMA,
    CANDIDATE_SUMMARY_SCHEMA,
    NEXT_FETCH_ACTION_SCHEMA,
    empty_candidate_detail_frame,
    empty_candidate_summary_frame,
    empty_next_fetch_action_frame,
)
from qooi.accumulation.structure import (
    StructureEvaluation,
    evaluate_structure_row,
    order_preparation_state,
    setup_stage,
)
from qooi.sources.coverage import missing_evidence_for_symbol

ALERT_RANK = {"red": 3, "orange": 2, "yellow": 1, "none": 0}
CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1, "blocked": 0}
_SELECTION_ROW_DEFAULTS: dict[str, object] = {
    "rank": 0,
    "timestamp": 0,
    "symbol": "",
    "score_total": 0,
    "alert_level": "none",
    "source_coverage_score": 0.0,
    "confidence_level": "blocked",
    "structure_state": "unknown",
    "preparation_state": "not_ready",
    "flow_state": "unknown",
    "attention_state": "unknown",
    "activation_state": "inactive",
    "risk_state": "data_insufficient",
    "suggestion_type": "reject_or_deprioritize",
    "top_positive_components": "",
    "top_negative_filters": "",
    "positive_components": "",
    "negative_filters": "",
    "data_quality_warning": "",
    "missing_evidence": "",
    "next_fetch_action": "",
    "return_24h": None,
    "close": None,
    "range_position_pct": None,
    "range_low_px": None,
    "range_high_px": None,
    "upside_to_range_high_pct": None,
    "downside_to_range_low_pct": None,
    "range_reward_risk": None,
    "structure_invalidation_px": None,
    "structure_target_px": None,
    "volatility_compression_pctile": None,
    "depth_imbalance_25_mean": None,
    "large_trade_buy_ratio": None,
    "resilience_score": None,
    "funding_rate": None,
    "open_interest_change_24h": None,
    "open_interest_usd_change_24h": None,
    "taker_buy_ratio": None,
    "taker_volume_imbalance": None,
    "long_short_account_ratio": None,
    "top_trader_long_short_account_ratio": None,
    "top_trader_long_short_position_ratio": None,
}


@dataclass(frozen=True)
class CandidateReadoutSettings:
    top_n: int = 25
    include_rejected: bool = True


@dataclass(frozen=True)
class NextFetchPolicy:
    yellow_threshold: int = 20
    watchlist_suggestions: tuple[str, ...] = (
        "prepare_watch",
        "trend_active_review",
        "flow_confirmed_watch",
        "monitor_context",
    )
    actionable_sources: tuple[str, ...] = (
        "discovery",
        "trades",
        "funding",
        "open_interest",
    )


@dataclass(frozen=True)
class SelectionReadout:
    symbol: str
    bucket: str
    verdict: str
    score: float
    broad_rank: int
    broad_score: float
    deep_score: int
    alert_level: str
    preparation_state: str
    activation_state: str
    structure: StructureEvaluation
    reasons: tuple[str, ...]
    blockers: tuple[str, ...]
    missing: tuple[str, ...]


class SelectionMarkdownFormatter:
    def section(self, title: str, rows: Sequence[SelectionReadout]) -> list[str]:
        lines = [f"## {title}", ""]
        if title == "Best Structure Setups" and not rows:
            lines.extend(["No high-value structure setups in this scan.", ""])
            return lines
        if not rows:
            lines.extend(["No rows in this section.", ""])
            return lines
        for row in rows:
            lines.append(self.row(row))
        lines.append("")
        return lines

    def row(self, readout: SelectionReadout) -> str:
        return (
            f"- {readout.symbol}: verdict={readout.verdict}, bucket={readout.bucket}, "
            f"selection_score={readout.score:.2f}, broad_rank={readout.broad_rank}, "
            f"broad_score={readout.broad_score:.2f}, deep_score={readout.deep_score}, "
            f"alert={readout.alert_level}, {self.structure(readout.structure)}, "
            f"why={readout.structure.plain_reason}, reasons={self.tokens(readout.reasons)}, "
            f"blockers={self.tokens(readout.blockers)}, missing={self.tokens(readout.missing)}"
        )

    def structure(self, evaluation: StructureEvaluation) -> str:
        return (
            f"px={_fmt_float(evaluation.current_px)}, "
            f"support={_fmt_float(evaluation.support_px)}, "
            f"target={_fmt_float(evaluation.target_px)}, "
            f"pos={_fmt_pct(evaluation.position_pct)}, "
            f"upside={_fmt_pct(evaluation.upside_to_target_pct)}, "
            f"risk={_fmt_pct(evaluation.risk_to_invalidation_pct)}, "
            f"R:R={_fmt_float(evaluation.reward_risk)}, "
            f"range_quality={evaluation.quality}"
        )

    def tokens(self, values: Sequence[str], *, empty: str = "none") -> str:
        return (
            ";".join(dict.fromkeys(token for token in values if token and token != "none"))
            or empty
        )


def build_candidate_summary(
    scores: pl.DataFrame,
    coverage: pl.DataFrame,
    *,
    top_n: int = 10,
    policy: NextFetchPolicy = NextFetchPolicy(),
) -> pl.DataFrame:
    if scores.is_empty():
        return empty_candidate_summary_frame()
    frame = scores.sort(["symbol", "timestamp"]).group_by("symbol").tail(1)
    rows = []
    for row in frame.to_dicts():
        symbol = str(row["symbol"])
        missing = str(row["missing_evidence"] or missing_evidence_for_symbol(coverage, symbol))
        action = _next_fetch_action(row, missing, coverage, policy)
        positives = str(row["positive_components"] or _positive_from_explanation(row))
        negatives = str(row["negative_filters"] or "")
        rows.append(
            {
                "rank": 0,
                "timestamp": int(row["timestamp"]),
                "symbol": symbol,
                "alert_level": str(row["alert_level"]),
                "score_total": int(row["score_total"] or 0),
                "source_coverage_score": float(row["source_coverage_score"] or 0.0),
                "confidence_level": str(row["confidence_level"]),
                "structure_state": str(row["structure_state"]),
                "preparation_state": str(row["preparation_state"]),
                "flow_state": str(row["flow_state"]),
                "attention_state": str(row["attention_state"]),
                "activation_state": str(row["activation_state"]),
                "risk_state": str(row["risk_state"]),
                "suggestion_type": str(row["suggestion_type"]),
                "top_positive_components": positives,
                "top_negative_filters": negatives,
                "data_quality_warning": str(row["data_quality_warning"] or ""),
                "missing_evidence": missing,
                "next_fetch_action": action,
                "rationale": _rationale(row, positives, negatives, missing, action),
            }
        )
    out = _rank_readout(pl.DataFrame(rows))
    out = (
        out.head(top_n)
        .drop(["rank", "_alert_rank", "_confidence_rank"], strict=False)
        .with_row_index("rank", offset=1)
    )
    for col, dtype in CANDIDATE_SUMMARY_SCHEMA.items():
        if col not in out.columns:
            out = out.with_columns(pl.lit(None).cast(dtype).alias(col))
    return out.select(CANDIDATE_SUMMARY_SCHEMA.keys())


def build_candidate_detail(
    scores: pl.DataFrame,
    features: pl.DataFrame,
    coverage: pl.DataFrame,
    *,
    settings: CandidateReadoutSettings = CandidateReadoutSettings(),
    policy: NextFetchPolicy = NextFetchPolicy(),
) -> pl.DataFrame:
    if scores.is_empty():
        return empty_candidate_detail_frame()
    latest_scores = scores.sort(["symbol", "timestamp"]).group_by("symbol").tail(1)
    latest_features = (
        features.sort(["symbol", "timestamp"]).group_by("symbol").tail(1)
        if not features.is_empty()
        else pl.DataFrame()
    )
    if not latest_features.is_empty():
        feature_cols = [
            "symbol",
            "close",
            "return_24h",
            "range_position_pct",
            "range_low_px",
            "range_high_px",
            "upside_to_range_high_pct",
            "downside_to_range_low_pct",
            "range_reward_risk",
            "structure_invalidation_px",
            "structure_target_px",
            "volatility_compression_pctile",
            "depth_imbalance_25_mean",
            "large_trade_buy_ratio",
            "resilience_score",
            "funding_rate",
            "open_interest_change_24h",
            "open_interest_usd_change_24h",
            "taker_buy_ratio",
            "taker_volume_imbalance",
            "long_short_account_ratio",
            "top_trader_long_short_account_ratio",
            "top_trader_long_short_position_ratio",
        ]
        latest_features = latest_features.select(feature_cols)
        frame = latest_scores.join(latest_features, on="symbol", how="left")
    else:
        frame = latest_scores
    rows = []
    for row in frame.to_dicts():
        symbol = str(row["symbol"])
        missing = str(row["missing_evidence"] or missing_evidence_for_symbol(coverage, symbol))
        rows.append(
            {
                "rank": 0,
                "timestamp": int(row["timestamp"]),
                "symbol": symbol,
                "score_total": int(row["score_total"] or 0),
                "alert_level": str(row["alert_level"]),
                "confidence_level": str(row["confidence_level"]),
                "suggestion_type": str(row["suggestion_type"]),
                "structure_state": str(row["structure_state"]),
                "preparation_state": str(row["preparation_state"]),
                "flow_state": str(row["flow_state"]),
                "attention_state": str(row["attention_state"]),
                "activation_state": str(row["activation_state"]),
                "risk_state": str(row["risk_state"]),
                "return_24h": row["return_24h"],
                "close": row["close"],
                "range_position_pct": row["range_position_pct"],
                "range_low_px": row["range_low_px"],
                "range_high_px": row["range_high_px"],
                "upside_to_range_high_pct": row["upside_to_range_high_pct"],
                "downside_to_range_low_pct": row["downside_to_range_low_pct"],
                "range_reward_risk": row["range_reward_risk"],
                "structure_invalidation_px": row["structure_invalidation_px"],
                "structure_target_px": row["structure_target_px"],
                "volatility_compression_pctile": row["volatility_compression_pctile"],
                "depth_imbalance_25_mean": row["depth_imbalance_25_mean"],
                "large_trade_buy_ratio": row["large_trade_buy_ratio"],
                "resilience_score": row["resilience_score"],
                "funding_rate": row["funding_rate"],
                "open_interest_change_24h": row["open_interest_change_24h"],
                "open_interest_usd_change_24h": row["open_interest_usd_change_24h"],
                "taker_buy_ratio": row["taker_buy_ratio"],
                "taker_volume_imbalance": row["taker_volume_imbalance"],
                "long_short_account_ratio": row["long_short_account_ratio"],
                "top_trader_long_short_account_ratio": row["top_trader_long_short_account_ratio"],
                "top_trader_long_short_position_ratio": row["top_trader_long_short_position_ratio"],
                "positive_components": str(row["positive_components"] or ""),
                "negative_filters": str(row["negative_filters"] or ""),
                "missing_evidence": missing,
                "next_fetch_action": _next_fetch_action(row, missing, coverage, policy),
                "data_quality_warning": str(row["data_quality_warning"] or ""),
            }
        )
    out = _rank_readout(pl.DataFrame(rows))
    if not settings.include_rejected and "suggestion_type" in out.columns:
        out = out.filter(pl.col("suggestion_type") != "reject_or_deprioritize")
    out = (
        out.head(settings.top_n)
        .drop(["rank", "_alert_rank", "_confidence_rank"], strict=False)
        .with_row_index("rank", offset=1)
    )
    for col, dtype in CANDIDATE_DETAIL_SCHEMA.items():
        if col not in out.columns:
            out = out.with_columns(pl.lit(None).cast(dtype).alias(col))
    return out.select(CANDIDATE_DETAIL_SCHEMA.keys())


def build_next_fetch_actions(
    scores: pl.DataFrame,
    coverage: pl.DataFrame,
    *,
    policy: NextFetchPolicy = NextFetchPolicy(),
) -> pl.DataFrame:
    if scores.is_empty():
        return empty_next_fetch_action_frame()
    rows = []
    latest = scores.sort(["symbol", "timestamp"]).group_by("symbol").tail(1)
    for row in latest.to_dicts():
        symbol = str(row["symbol"])
        missing = str(row["missing_evidence"] or missing_evidence_for_symbol(coverage, symbol))
        rows.extend(
            next_fetch_actions_for_row(
                row, _actionable_missing(missing, coverage, symbol, policy), policy=policy
            )
        )
    if not rows:
        return empty_next_fetch_action_frame()
    return pl.DataFrame(rows, schema=NEXT_FETCH_ACTION_SCHEMA).sort(["priority", "symbol"])


def render_candidate_rationale(summary: pl.DataFrame) -> str:
    if summary.is_empty():
        return "# Candidate Rationale\n\nNo candidates were scored.\n"
    lines = ["# Candidate Rationale", ""]
    for row in summary.to_dicts():
        lines.append(
            f"{row['rank']}. {row['symbol']}: {row['alert_level']} / "
            f"{row['confidence_level']} confidence, score={row['score_total']}, "
            f"suggestion={row['suggestion_type']}"
        )
        lines.append(f"Rationale: {row['rationale']}")
        if row["next_fetch_action"]:
            lines.append(f"Next fetch: {row['next_fetch_action']}")
        lines.append("")
    return "\n".join(lines)


def render_scan_feedback(
    summary: pl.DataFrame,
    detail: pl.DataFrame,
    next_fetch: pl.DataFrame,
    coverage: pl.DataFrame,
    broad_candidates: pl.DataFrame | None = None,
) -> str:
    """Render a deterministic run-level Markdown summary for scanner artifacts."""
    summary_rows = summary.to_dicts() if not summary.is_empty() else []
    next_fetch_rows = next_fetch.to_dicts() if not next_fetch.is_empty() else []
    detail_by_symbol = _rows_by_symbol(detail)
    summary_by_symbol = _rows_by_symbol(summary)
    broad_frame = broad_candidates if broad_candidates is not None else pl.DataFrame()
    selection_rows = _selection_rows(
        broad_frame,
        summary_by_symbol,
        detail_by_symbol,
        coverage,
    )
    excluded_count, unmapped_count = _broad_candidate_counts(broad_frame)
    alert_counts = {
        level: sum(1 for row in summary_rows if str(row["alert_level"] or "none") == level)
        for level in ("red", "orange", "yellow")
    }
    candidate_rows = [
        row
        for row in summary_rows
        if str(row["suggestion_type"] or "") != "reject_or_deprioritize"
        and _has_positive_readout(row)
    ]
    weak_rows = [row for row in summary_rows if row not in candidate_rows]
    watchlist_count = sum(
        1
        for row in candidate_rows
        if str(row["suggestion_type"] or "") in {"prepare_watch", "trend_active_review"}
    )
    selection_watchlist_count = sum(
        1
        for row in selection_rows
        if row.preparation_state in {"prepare_watch", "watchlist"}
    )
    rejected_count = sum(
        1
        for row in summary_rows
        if str(row["suggestion_type"] or "") == "reject_or_deprioritize"
    )
    lines = [
        "# Accumulation Scan Feedback",
        "",
        "Research-only: scanner artifacts do not authorize live trading.",
        "",
        "## Run Readout",
        "",
        f"- Candidates reviewed: {len(summary_rows)}",
        "- Alerts: "
        f"red={alert_counts['red']}, orange={alert_counts['orange']}, "
        f"yellow={alert_counts['yellow']}",
        f"- Strict positive candidates: {len(candidate_rows)}",
        f"- Strict watchlist/preparation candidates: {watchlist_count}",
        f"- Strict rejected/deprioritized: {rejected_count}",
        f"- Selection candidates: {len(selection_rows)}",
        f"- Selection watchlist/preparation candidates: {selection_watchlist_count}",
        f"- Broad excluded/unmapped: excluded={excluded_count}, unmapped={unmapped_count}",
        f"- Follow-up actions: {len(next_fetch_rows)}",
        "",
        "## Top Candidates",
        "",
    ]
    if not summary_rows:
        lines.append("No candidates were scored.")
        lines.append("")
    elif not candidate_rows:
        lines.append("No positive-evidence candidates in this run.")
        lines.append("")
    for row in candidate_rows[:5]:
        symbol = str(row["symbol"] or "")
        detail_row = detail_by_symbol.get(symbol, {})
        positives = _first_text(
            row["top_positive_components"], detail_row.get("positive_components"), "none"
        )
        negatives = _first_text(
            row["top_negative_filters"], detail_row.get("negative_filters"), "none"
        )
        warning = _first_text(
            row["data_quality_warning"], detail_row.get("data_quality_warning")
        )
        if warning:
            negatives = f"{negatives}; {warning}" if negatives != "none" else warning
        missing = _first_text(
            row["missing_evidence"], detail_row.get("missing_evidence"), "none"
        )
        missing = _render_missing_evidence(missing, coverage, symbol)
        action = _first_text(
            row["next_fetch_action"], detail_row.get("next_fetch_action"), "none"
        )
        lines.extend(
            [
                f"{row['rank']}. {symbol}: {row['alert_level']}/"
                f"{row['confidence_level']}, "
                f"score={row['score_total']}, "
                f"suggestion={row['suggestion_type']}",
                f"   Why: {_clean_reason_text(positives)}",
                f"   Risks: {negatives}",
                f"   Missing Evidence: {missing}",
                f"   Next: {action}",
                "",
            ]
        )
    if broad_candidates is not None and not broad_candidates.is_empty():
        lines.extend(["## Selection Readout", ""])
        if not selection_rows:
            lines.append(
                "Broad scan found rows, but no mapped, non-excluded OKX candidates were available."
            )
            lines.append("")
        else:
            lines.append(
                "Review ranking only: broad_score is not merged into accumulation score_total."
            )
            lines.append("")
            formatter = SelectionMarkdownFormatter()
            grouped_rows = _group_selection_readouts(selection_rows[: _selection_limit(summary)])
            for title, rows in grouped_rows:
                lines.extend(formatter.section(title, rows))
    rule_lines = _rule_activity_lines(summary_rows, detail_by_symbol)
    if rule_lines:
        lines.extend(["## Rule Activity", ""])
        lines.extend(rule_lines)
        lines.append("")
    if weak_rows:
        lines.extend(["## Rejected Or Neutral Rows", ""])
        for row in weak_rows[:5]:
            symbol = str(row["symbol"] or "")
            detail_row = detail_by_symbol.get(symbol, {})
            reason = _first_text(
                row["top_positive_components"],
                detail_row.get("positive_components"),
                row["top_negative_filters"],
                detail_row.get("negative_filters"),
                "no_rules_fired",
            )
            missing = _first_text(
                row["missing_evidence"], detail_row.get("missing_evidence"), "none"
            )
            lines.append(
                f"- {symbol}: score={row['score_total']}, "
                f"suggestion={row['suggestion_type']}, "
                f"why={_clean_reason_text(reason)}, "
                f"missing={_render_missing_evidence(missing, coverage, symbol)}"
            )
        lines.append("")
    lines.extend(["## Source Availability", ""])
    availability_lines = _source_availability_lines(coverage)
    lines.extend(
        availability_lines if availability_lines else ["- No skipped or disabled sources recorded."]
    )
    lines.extend(["", "## Coverage Warnings", ""])
    coverage_lines = _coverage_warning_lines(coverage)
    lines.extend(
        coverage_lines if coverage_lines else ["- No actionable coverage warnings recorded."]
    )
    lines.extend(["", "## Next Fetch Queue", ""])
    if next_fetch_rows:
        for row in next_fetch_rows[:10]:
            secret = " requires secret" if row["requires_secret"] else ""
            lines.append(
                f"- priority {row['priority']}: {row['symbol']} -> "
                f"{row['phase']} {row['source']} "
                f"({row['reason']}{secret})"
            )
    else:
        lines.append("- No follow-up fetch actions queued.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- High score with low confidence means partial evidence, not confirmation.",
            "- Few alerts means few confirmed accumulation signals, not few potential assets.",
            "- Selection candidates are a research review surface and are not trading signals.",
            "- Source coverage score measures required market sources; "
            "optional source gaps are separate blockers.",
            "- Missing data is explicit evidence quality information, not neutral signal.",
            "",
        ]
    )
    return "\n".join(lines)


def _next_fetch_action(
    row: dict, missing: str, coverage: pl.DataFrame, policy: NextFetchPolicy
) -> str:
    symbol = str(row["symbol"])
    actions = next_fetch_actions_for_row(
        row, _actionable_missing(missing, coverage, symbol, policy), policy=policy
    )
    if not actions:
        return ""
    action = actions[0]
    if action["source"] == "discovery":
        return "discover contract metadata"
    if action["source"] == "trades":
        return "collect-market trades"
    if action["source"] in {"funding", "open_interest"}:
        return "collect-market context"
    return str(action["phase"])


def next_fetch_actions_for_row(
    row: dict,
    missing: str,
    *,
    policy: NextFetchPolicy = NextFetchPolicy(),
) -> list[dict[str, object]]:
    symbol = str(row["symbol"])
    score = int(row["score_total"] or 0)
    suggestion = str(row["suggestion_type"] or "")
    watchlist = suggestion in policy.watchlist_suggestions
    rejected = suggestion == "reject_or_deprioritize"
    actions = []
    actionable = set(policy.actionable_sources)
    if "discovery" in actionable and (
        "contract_metadata_missing" in missing or "trade_notional_metadata_missing" in missing
    ):
        actions.append(
            _action(
                symbol,
                1,
                "discovery",
                "discover",
                "contract metadata missing",
                "medium",
                False,
            )
        )
    if (
        not rejected
        and "trades" in actionable
        and "trades_missing" in missing
        and (score >= policy.yellow_threshold or watchlist)
    ):
        actions.append(
            _action(
                symbol,
                1,
                "trades",
                "collect-market",
                "trade resilience missing",
                "medium",
                False,
            )
        )
    if (
        not rejected
        and "onchain" in actionable
        and "onchain" in missing
        and (score >= policy.yellow_threshold or watchlist)
    ):
        actions.append(
            _action(symbol, 2, "onchain", "collect-onchain", "exchange flow missing", "high", True)
        )
    message_missing = any(
        token in missing for token in ("messages_missing", "message_classifications_missing")
    )
    if (
        not rejected
        and "messages" in actionable
        and message_missing
        and (score >= policy.yellow_threshold or watchlist)
    ):
        actions.append(
            _action(
                symbol, 3, "messages", "collect-context", "message context missing", "medium", False
            )
        )
    for source in ("funding", "open_interest"):
        if source in actionable and f"{source}_missing" in missing:
            actions.append(
                _action(
                    symbol,
                    3,
                    source,
                    "collect-market",
                    f"{source} context missing",
                    "low",
                    False,
                )
            )
    return actions


def _rows_by_symbol(frame: pl.DataFrame) -> dict[str, dict[str, object]]:
    if frame.is_empty() or "symbol" not in frame.columns:
        return {}
    return {str(row.get("symbol") or ""): row for row in frame.to_dicts()}


def _selection_limit(summary: pl.DataFrame) -> int:
    return max(25, summary.height if not summary.is_empty() else 0)


def _selection_rows(
    broad_candidates: pl.DataFrame,
    summary_by_symbol: dict[str, dict[str, object]],
    detail_by_symbol: dict[str, dict[str, object]],
    coverage: pl.DataFrame,
) -> list[SelectionReadout]:
    if broad_candidates.is_empty() or "okx_symbol" not in broad_candidates.columns:
        return []
    frame = broad_candidates
    if "rank" in frame.columns:
        frame = frame.sort("rank")
    rows = []
    for broad in frame.to_dicts():
        symbol = str(broad.get("okx_symbol") or "").strip()
        if not symbol or not _bool_value(broad.get("okx_mapped")):
            continue
        exclude_reason = str(broad.get("exclude_reason") or "").strip()
        if exclude_reason:
            continue
        summary_row = summary_by_symbol.get(symbol, {})
        detail_row = detail_by_symbol.get(symbol, {})
        merged = {**_SELECTION_ROW_DEFAULTS, **detail_row, **summary_row}
        missing = _first_text(
            merged.get("missing_evidence"),
            missing_evidence_for_symbol(coverage, symbol),
            "none",
        )
        structure = evaluate_structure_row(merged)
        setup = setup_stage(merged, structure.quality)
        order = order_preparation_state(merged)
        missing_tokens = _token_tuple(missing)
        blockers = _join_token_tuples(
            _selection_blockers(missing, merged, has_deep=bool(merged)), structure.blockers
        )
        range_quality = structure.quality
        bucket = _selection_bucket(broad, merged, blockers, setup, order, range_quality)
        reasons = _selection_reasons(broad, merged, setup, order, range_quality)
        score = _selection_score(broad, merged, blockers, bucket, setup, order, range_quality)
        rows.append(
            SelectionReadout(
                symbol=symbol,
                bucket=bucket,
                verdict=_selection_verdict(bucket, structure.verdict, blockers),
                score=score,
                broad_rank=int(broad.get("rank") or 0),
                broad_score=_float_value(broad.get("broad_score")),
                deep_score=int(merged.get("score_total") or 0),
                alert_level=str(merged.get("alert_level") or "none"),
                preparation_state=str(merged.get("preparation_state") or "unknown"),
                activation_state=str(merged.get("activation_state") or "unknown"),
                structure=structure,
                reasons=_token_tuple(reasons),
                blockers=blockers,
                missing=missing_tokens,
            )
        )
    bucket_rank = {
        "strict_alert": 4,
        "accumulation_setup": 3,
        "orderbook_watch": 2,
        "early_trend_review": 1,
        "late_trend_review": 0,
        "data_blocked": -1,
    }
    return sorted(
        rows,
        key=lambda row: (
            bucket_rank.get(row.bucket, 0),
            row.score,
            -row.broad_rank,
        ),
        reverse=True,
    )


def _group_selection_readouts(
    rows: Sequence[SelectionReadout],
) -> list[tuple[str, list[SelectionReadout]]]:
    bad_blockers = {
        "near_resistance",
        "poor_range_reward",
        "extended_structure",
        "overextended_activation",
        "breakout_without_accumulation_evidence",
    }
    best = [
        row
        for row in rows
        if row.bucket == "strict_alert"
        or (
            row.bucket == "accumulation_setup"
            and row.structure.quality in {"favorable_range_setup", "balanced_range_setup"}
            and not bad_blockers.intersection(row.blockers)
        )
        or (
            row.bucket == "orderbook_watch"
            and not {"near_resistance", "poor_range_reward"}.intersection(row.blockers)
        )
    ]
    best_symbols = {row.symbol for row in best}
    data_blocked = [
        row for row in rows if row.bucket == "data_blocked" and row.symbol not in best_symbols
    ]
    late = [
        row
        for row in rows
        if row.symbol not in best_symbols
        and row.bucket != "data_blocked"
        and (row.bucket == "late_trend_review" or bad_blockers.intersection(row.blockers))
    ]
    late_symbols = {row.symbol for row in late}
    watch = [
        row
        for row in rows
        if row.symbol not in best_symbols
        and row.symbol not in late_symbols
        and row.bucket != "data_blocked"
    ]
    return [
        ("Best Structure Setups", best),
        ("Watchlist: Needs Pullback Or Confirmation", watch),
        ("Late Or Poor Structure", late),
        ("Data Blocked Broad Rows", data_blocked),
    ]


def _selection_bucket(
    broad: dict[str, object],
    merged: dict[str, object],
    blockers: tuple[str, ...],
    setup: str,
    order: str,
    range_quality: str,
) -> str:
    if str(merged.get("alert_level") or "none") != "none":
        return "strict_alert"
    if not merged or "deep_not_collected" in blockers or range_quality == "range_unknown":
        return "data_blocked"
    bad_range = range_quality in {"near_resistance", "poor_range_reward"}
    if not bad_range and (
        setup in {"compressed_setup", "low_range_setup", "early_setup"}
        or range_quality in {
            "favorable_range_setup",
            "balanced_range_setup",
        }
    ):
        return "accumulation_setup"
    if order in {"order_supported", "resilience_supported", "flow_supported"}:
        return "orderbook_watch"
    broad_reasons = str(broad.get("broad_reasons") or "")
    broad_active = "active_1h" in broad_reasons or "active_24h" in broad_reasons
    if broad_active and str(merged.get("activation_state") or "") == "early" and not bad_range:
        return "early_trend_review"
    if broad_active:
        return "late_trend_review"
    if blockers:
        return "data_blocked"
    return "late_trend_review"


def _selection_score(
    broad: dict[str, object],
    merged: dict[str, object],
    blockers: tuple[str, ...],
    bucket: str,
    setup: str,
    order: str,
    range_quality: str,
) -> float:
    score = _float_value(broad.get("broad_score"))
    if str(merged.get("alert_level") or "none") != "none":
        score += 4.0
    if bucket == "accumulation_setup":
        score += 3.0
    if bucket == "orderbook_watch":
        score += 2.0
    if bucket == "early_trend_review":
        score += 1.0
    if bucket == "late_trend_review":
        score -= 1.5
    if setup == "compressed_setup":
        score += 1.0
    if setup == "low_range_setup":
        score += 1.0
    if str(merged.get("activation_state") or "") == "early" and "extended" not in blockers:
        score += 0.5
    if order in {"order_supported", "resilience_supported", "flow_supported"}:
        score += 0.75
    if range_quality == "favorable_range_setup":
        score += 1.5
    if range_quality == "balanced_range_setup":
        score += 0.75
    if "near_resistance" in blockers:
        score -= 2.0
    if "poor_range_reward" in blockers:
        score -= 1.5
    if any(token in blockers for token in ("extended_structure", "overextended_activation")):
        score -= 2.5
    if _float_value(merged.get("source_coverage_score")) >= 0.85:
        score += 1.0
    if str(merged.get("top_negative_filters") or merged.get("negative_filters") or "").strip():
        score -= 2.0
    if blockers:
        score -= 1.0
    if "deep_not_collected" in blockers:
        score -= 1.0
    return score


def _selection_reasons(
    broad: dict[str, object],
    merged: dict[str, object],
    setup: str,
    order: str,
    range_quality: str,
) -> str:
    parts = []
    broad_reasons = str(broad.get("broad_reasons") or "").strip()
    if broad_reasons:
        parts.append(f"broad={broad_reasons}")
    if setup != "unknown":
        parts.append(f"setup={setup}")
    if range_quality != "range_unknown":
        parts.append(f"range={range_quality}")
    if order != "missing":
        parts.append(f"order={order}")
    parts.extend(_derivatives_reasons(merged))
    if str(merged.get("alert_level") or "none") != "none":
        parts.append("strict_accumulation_alert")
    for key, label in (
        ("preparation_state", "prep"),
        ("activation_state", "activation"),
        ("structure_state", "structure"),
    ):
        value = str(merged.get(key) or "").strip()
        if value and value not in {"unknown", "not_ready", "inactive"}:
            parts.append(f"{label}={value}")
    return ";".join(dict.fromkeys(parts))


def _derivatives_reasons(merged: dict[str, object]) -> list[str]:
    reasons = []
    oi_change = _optional_float(merged.get("open_interest_usd_change_24h"))
    if oi_change is None:
        oi_change = _optional_float(merged.get("open_interest_change_24h"))
    if oi_change is not None:
        if oi_change > 0.0:
            reasons.append("derivatives=oi_expanding")
        elif oi_change < 0.0:
            reasons.append("derivatives=oi_contracting")
    taker_buy_ratio = _optional_float(merged.get("taker_buy_ratio"))
    if taker_buy_ratio is not None:
        if taker_buy_ratio >= 0.60:
            reasons.append("taker=taker_buy_dominant")
        elif taker_buy_ratio <= 0.40:
            reasons.append("taker=taker_sell_dominant")
    top_position = _optional_float(merged.get("top_trader_long_short_position_ratio"))
    top_account = _optional_float(merged.get("top_trader_long_short_account_ratio"))
    top_ratio = top_position if top_position is not None else top_account
    if top_ratio is not None:
        if top_ratio >= 2.5:
            reasons.append("top_trader=crowded_long")
        elif top_ratio > 1.0:
            reasons.append("top_trader=long_bias")
        elif top_ratio < 0.75:
            reasons.append("top_trader=short_bias")
    return reasons


def _selection_blockers(missing: str, merged: dict[str, object], *, has_deep: bool) -> str:
    blockers = []
    if not has_deep:
        blockers.append("deep_not_collected")
    for token in [part for part in str(missing or "").split(";") if part]:
        if token in {
            "onchain_missing",
            "whale_missing",
            "messages_missing",
            "message_classifications_missing",
        }:
            blockers.append(token)
    warnings = str(merged.get("data_quality_warning") or "")
    for token in [part for part in warnings.split(";") if part]:
        if token in {
            "onchain_missing",
            "whale_missing",
            "messages_missing",
            "message_classifications_missing",
        }:
            blockers.append(token)
    return ";".join(dict.fromkeys(blockers)) or "none"


def _broad_candidate_counts(broad_candidates: pl.DataFrame) -> tuple[int, int]:
    if broad_candidates.is_empty():
        return 0, 0
    excluded = 0
    unmapped = 0
    for row in broad_candidates.to_dicts():
        if not _bool_value(row.get("okx_mapped")):
            unmapped += 1
        if str(row.get("exclude_reason") or "").strip():
            excluded += 1
    return excluded, unmapped


def _rule_activity_lines(
    summary_rows: list[dict[str, object]], detail_by_symbol: dict[str, dict[str, object]]
) -> list[str]:
    if not summary_rows:
        return []
    symbols = [str(row.get("symbol") or "") for row in summary_rows]
    positive_scores = sum(1 for row in summary_rows if int(row.get("score_total") or 0) > 0)
    lines = [f"- Latest reviewed symbols with positive score: {positive_scores} / {len(symbols)}"]
    component_fields = {
        "depth_support_on_down_day": "depth_support_on_down_day",
        "resilience_high": "resilience_high",
    }
    missing_fields = {
        "flow_outflow_3sigma_2h": ("onchain_missing",),
        "whale_accumulation_high": ("whale_missing", "onchain_missing"),
        "message_not_overheated": ("messages_missing", "message_classifications_missing"),
    }
    for label, token in component_fields.items():
        fired = sum(1 for row in summary_rows if token in _component_text(row, detail_by_symbol))
        lines.append(f"- {label} fired: {fired}")
    for label, tokens in missing_fields.items():
        missing = sum(
            1
            for row in summary_rows
            if any(token in _missing_text(row, detail_by_symbol) for token in tokens)
        )
        lines.append(f"- {label} unavailable/missing: {missing}")
    below_ma = sum(
        1
        for row in summary_rows
        if "below_ma200_weak_depth" in _negative_text(row, detail_by_symbol)
    )
    lines.append(f"- below_ma200_weak_depth fired: {below_ma}")
    return lines


def _component_text(
    row: dict[str, object], detail_by_symbol: dict[str, dict[str, object]]
) -> str:
    detail = detail_by_symbol.get(str(row.get("symbol") or ""), {})
    return ";".join(
        [
            str(row.get("top_positive_components") or ""),
            str(detail.get("positive_components") or ""),
        ]
    )


def _negative_text(
    row: dict[str, object], detail_by_symbol: dict[str, dict[str, object]]
) -> str:
    detail = detail_by_symbol.get(str(row.get("symbol") or ""), {})
    return ";".join(
        [
            str(row.get("top_negative_filters") or ""),
            str(detail.get("negative_filters") or ""),
        ]
    )


def _missing_text(
    row: dict[str, object], detail_by_symbol: dict[str, dict[str, object]]
) -> str:
    detail = detail_by_symbol.get(str(row.get("symbol") or ""), {})
    return ";".join(
        [
            str(row.get("missing_evidence") or ""),
            str(detail.get("missing_evidence") or ""),
            str(row.get("data_quality_warning") or ""),
            str(detail.get("data_quality_warning") or ""),
        ]
    )


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() == "true"


def _float_value(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _token_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, tuple):
        return tuple(
            str(token).strip() for token in value if str(token).strip() and token != "none"
        )
    return tuple(
        dict.fromkeys(
            token.strip()
            for token in str(value or "").split(";")
            if token.strip() and token.strip() != "none"
        )
    )


def _join_token_tuples(*values: object) -> tuple[str, ...]:
    tokens = []
    for value in values:
        tokens.extend(_token_tuple(value))
    return tuple(dict.fromkeys(tokens))


def _selection_verdict(
    bucket: str, structure_verdict: str, blockers: tuple[str, ...]
) -> str:
    if bucket == "data_blocked":
        return "data_blocked"
    if bucket == "strict_alert" and structure_verdict == "data_blocked":
        return "review_now"
    if bucket == "orderbook_watch" and structure_verdict not in {"wait_pullback", "avoid_late"}:
        return "watch_orderbook"
    if any(token in blockers for token in ("extended_structure", "overextended_activation")):
        return "avoid_late"
    return structure_verdict


def _fmt_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 10:
        return f"{value:.2f}"
    if abs(value) >= 1:
        return f"{value:.4f}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _first_text(*values: object) -> str:
    default = ""
    if values:
        default = str(values[-1]) if values[-1] is not None else ""
        values = values[:-1]
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return default


def _has_positive_readout(row: dict[str, object]) -> bool:
    if int(row.get("score_total", 0) or 0) > 0:
        return True
    return bool(str(row.get("top_positive_components") or "").strip())


def _clean_reason_text(text: str) -> str:
    cleaned = text.strip()
    return "no positive scanner rules fired" if cleaned == "no_rules_fired" else cleaned


def _latest_manifest_row(coverage: pl.DataFrame, symbol: str, source: str) -> dict[str, object]:
    if coverage.is_empty() or not {"symbol", "source"}.issubset(coverage.columns):
        return {}
    rows = coverage.filter((pl.col("symbol") == symbol) & (pl.col("source") == source))
    if rows.is_empty() and source == "open_interest":
        rows = coverage.filter(
            (pl.col("symbol") == symbol) & (pl.col("source") == "open_interest_history")
        )
    if rows.is_empty():
        return {}
    if "timestamp" in rows.columns:
        rows = rows.sort("timestamp")
    return rows.tail(1).to_dicts()[0]


def _source_reason(coverage: pl.DataFrame, symbol: str, source: str) -> tuple[str, bool]:
    row = _latest_manifest_row(coverage, symbol, source)
    if not row:
        return f"{source.replace('_', ' ')} collection has not run", True
    status = str(row.get("status") or "missing")
    warning = str(row.get("warning") or "")
    if status in {"ok", "partial"}:
        return "available", False
    if source == "books" and status == "skipped":
        return "book collection skipped by run mode", False
    if source == "messages":
        labels = {
            "messages_disabled": "messages disabled in config",
            "local_messages_path_missing": "local message CSV path is not configured",
            "local_messages_file_missing": "configured local message CSV file does not exist",
            "local_messages_missing": "local message CSV has no rows for this symbol",
            "message_classifications_missing": "message classifications missing",
        }
        return (
            labels.get(warning, warning.replace("_", " ") or "message evidence missing"),
            status != "skipped",
        )
    if source == "onchain":
        labels = {
            "onchain_disabled": "on-chain collection disabled in config",
            "onchain_missing": "on-chain exchange-flow evidence missing",
            "whale_missing": "whale ownership evidence missing",
            "onchain_token_mapping_missing": "on-chain token mapping is not configured",
            "onchain_provider_not_implemented": "on-chain provider collector is not implemented",
        }
        if warning in {"onchain_provider_not_implemented", "onchain_token_mapping_missing"}:
            return labels[warning], False
        return (
            labels.get(warning, warning.replace("_", " ") or "on-chain evidence missing"),
            status != "skipped",
        )
    if source == "polymarket_markets":
        labels = {
            "polymarket_disabled": "Polymarket context disabled by config",
            "polymarket_query_disabled": "Polymarket query disabled by config",
            "polymarket_query_missing": (
                "Polymarket query could not be derived from symbol metadata"
            ),
            "polymarket_unmatched": "Polymarket returned no related markets for generated queries",
            "polymarket_alias_missing": "legacy Polymarket alias gate prevented fetch",
        }
        return (
            labels.get(warning, warning.replace("_", " ") or "Polymarket context missing"),
            status not in {"skipped", "failed"},
        )
    return (
        warning.replace("_", " ") or f"{source.replace('_', ' ')} evidence missing",
        status != "skipped",
    )


def _render_missing_evidence(missing: str, coverage: pl.DataFrame, symbol: str) -> str:
    if not missing or missing == "none":
        return "none"
    rendered = []
    source_by_token = {
        "messages_missing": "messages",
        "message_classifications_missing": "messages",
        "onchain_missing": "onchain",
        "whale_missing": "onchain",
        "book_missing": "books",
    }
    for token in [part for part in missing.split(";") if part]:
        source = source_by_token.get(token)
        if source:
            reason, _actionable = _source_reason(coverage, symbol, source)
            rendered.append(reason)
        else:
            rendered.append(token.replace("_", " "))
    return "; ".join(dict.fromkeys(rendered)) or "none"


def _actionable_missing(
    missing: str, coverage: pl.DataFrame, symbol: str, policy: NextFetchPolicy
) -> str:
    actionable_sources = set(policy.actionable_sources)
    kept = []
    source_by_token = {
        "trades_missing": "trades",
        "funding_missing": "funding",
        "open_interest_missing": "open_interest",
        "onchain_missing": "onchain",
        "whale_missing": "onchain",
        "messages_missing": "messages",
        "message_classifications_missing": "messages",
        "contract_metadata_missing": "discovery",
        "trade_notional_metadata_missing": "discovery",
    }
    for token in [part for part in missing.split(";") if part]:
        source = source_by_token.get(token)
        if source is None:
            kept.append(token)
            continue
        if source not in actionable_sources:
            continue
        reason, actionable = _source_reason(coverage, symbol, source)
        if actionable or reason == "available":
            kept.append(token)
    return ";".join(dict.fromkeys(kept))


def _source_availability_lines(coverage: pl.DataFrame) -> list[str]:
    if coverage.is_empty():
        return []
    counts: dict[tuple[str, str, str], int] = {}
    for row in _latest_manifest_rows(coverage):
        status = str(row.get("status") or "").strip()
        warning = str(row.get("warning") or "").strip()
        source = str(row.get("source") or "unknown").strip() or "unknown"
        if status != "skipped" and not warning.endswith("_disabled"):
            continue
        symbol = str(row.get("symbol") or "")
        reason, _actionable = _source_reason(coverage, symbol, source)
        key = (source, status or "skipped", reason)
        counts[key] = counts.get(key, 0) + 1
    return [
        f"- {source} {status}: {reason} ({count} symbols)"
        for (source, status, reason), count in sorted(counts.items())
    ]


def _coverage_warning_lines(coverage: pl.DataFrame) -> list[str]:
    if coverage.is_empty():
        return []
    counts: dict[tuple[str, str, str], int] = {}
    for row in _latest_manifest_rows(coverage):
        warning = str(row.get("warning") or "").strip()
        status = str(row.get("status") or "unknown").strip() or "unknown"
        if status == "ok":
            continue
        if status == "skipped" or warning.endswith("_disabled"):
            continue
        source = str(row.get("source") or "unknown").strip() or "unknown"
        key = (source, status, warning or "no_warning")
        counts[key] = counts.get(key, 0) + 1
    return [
        f"- {source}/{status}/{warning}: {count}"
        for (source, status, warning), count in sorted(counts.items())
    ]


def _latest_manifest_rows(coverage: pl.DataFrame) -> list[dict[str, object]]:
    if coverage.is_empty() or not {"symbol", "source"}.issubset(coverage.columns):
        return coverage.to_dicts() if not coverage.is_empty() else []
    frame = coverage
    if "timestamp" in frame.columns:
        frame = frame.sort(["symbol", "source", "timestamp"])
    return frame.group_by(["symbol", "source"]).tail(1).to_dicts()


def _rank_readout(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    return frame.with_columns(
        [
            pl.col("alert_level").replace_strict(ALERT_RANK, default=0).alias("_alert_rank"),
            pl.col("confidence_level")
            .replace_strict(CONFIDENCE_RANK, default=0)
            .alias("_confidence_rank"),
        ]
    ).sort(["_alert_rank", "score_total", "_confidence_rank", "timestamp"], descending=True)


def _positive_from_explanation(row: dict) -> str:
    parts = [
        part for part in str(row.get("explanation") or "").split(";") if part and " -" not in part
    ]
    return ";".join(parts[:3])


def _rationale(row: dict, positives: str, negatives: str, missing: str, action: str) -> str:
    bits = [
        f"suggestion: {row.get('suggestion_type', 'reject_or_deprioritize')}",
        "axes: "
        f"structure={row.get('structure_state', 'unknown')}, "
        f"prep={row.get('preparation_state', 'not_ready')}, "
        f"flow={row.get('flow_state', 'unknown')}, "
        f"attention={row.get('attention_state', 'unknown')}, "
        f"activation={row.get('activation_state', 'inactive')}, "
        f"risk={row.get('risk_state', 'data_insufficient')}",
        f"score components: {positives or 'none'}",
    ]
    if negatives:
        bits.append(f"negative filters: {negatives}")
    bits.append(f"coverage: {row.get('source_coverage_score', 0.0)}")
    if missing:
        bits.append(f"missing evidence: {missing}")
    if action:
        bits.append(f"next: {action}")
    return "; ".join(bits)


def _action(
    symbol: str,
    priority: int,
    source: str,
    phase: str,
    reason: str,
    expected_confidence_delta: str,
    requires_secret: bool,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "priority": priority,
        "source": source,
        "phase": phase,
        "reason": reason,
        "expected_confidence_delta": expected_confidence_delta,
        "requires_secret": requires_secret,
    }
