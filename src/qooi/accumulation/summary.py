"""Candidate summary and rationale rendering for accumulation scanner output."""

from __future__ import annotations

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
from qooi.sources.coverage import missing_evidence_for_symbol

ALERT_RANK = {"red": 3, "orange": 2, "yellow": 1, "none": 0}
CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1, "blocked": 0}


@dataclass(frozen=True)
class CandidateReadoutSettings:
    top_n: int = 25
    include_rejected: bool = True


@dataclass(frozen=True)
class NextFetchPolicy:
    yellow_threshold: int = 20
    watchlist_suggestions: tuple[str, ...] = ("prepare_watch", "trend_active_review")
    secret_sources: tuple[str, ...] = ("onchain",)


def build_candidate_summary(
    scores: pl.DataFrame, coverage: pl.DataFrame, *, top_n: int = 10
) -> pl.DataFrame:
    if scores.is_empty():
        return empty_candidate_summary_frame()
    frame = scores
    if "timestamp" in frame.columns:
        frame = frame.sort(["symbol", "timestamp"]).group_by("symbol").tail(1)
    rows = []
    for row in frame.to_dicts():
        symbol = str(row.get("symbol", ""))
        missing = str(row.get("missing_evidence") or missing_evidence_for_symbol(coverage, symbol))
        action = _next_fetch_action(row, missing)
        positives = str(row.get("positive_components") or _positive_from_explanation(row))
        negatives = str(row.get("negative_filters") or "")
        rows.append(
            {
                "rank": 0,
                "timestamp": int(row.get("timestamp", 0) or 0),
                "symbol": symbol,
                "alert_level": str(row.get("alert_level", "none")),
                "score_total": int(row.get("score_total", 0) or 0),
                "source_coverage_score": float(row.get("source_coverage_score", 0.0) or 0.0),
                "confidence_level": str(row.get("confidence_level", "blocked")),
                "structure_state": str(row.get("structure_state", "unknown")),
                "preparation_state": str(row.get("preparation_state", "not_ready")),
                "flow_state": str(row.get("flow_state", "unknown")),
                "attention_state": str(row.get("attention_state", "unknown")),
                "activation_state": str(row.get("activation_state", "inactive")),
                "risk_state": str(row.get("risk_state", "data_insufficient")),
                "suggestion_type": str(row.get("suggestion_type", "reject_or_deprioritize")),
                "top_positive_components": positives,
                "top_negative_filters": negatives,
                "data_quality_warning": str(row.get("data_quality_warning") or ""),
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
) -> pl.DataFrame:
    if scores.is_empty():
        return empty_candidate_detail_frame()
    latest_scores = scores.sort(["symbol", "timestamp"]).group_by("symbol").tail(1)
    latest_features = (
        features.sort(["symbol", "timestamp"]).group_by("symbol").tail(1)
        if not features.is_empty() and "symbol" in features.columns
        else pl.DataFrame()
    )
    if not latest_features.is_empty():
        feature_cols = [
            "symbol",
            "return_24h",
            "range_position_pct",
            "volatility_compression_pctile",
            "depth_imbalance_25_mean",
            "large_trade_buy_ratio",
            "resilience_score",
            "funding_rate",
            "open_interest_change_24h",
        ]
        latest_features = latest_features.select(
            [col for col in feature_cols if col in latest_features.columns]
        )
        frame = latest_scores.join(latest_features, on="symbol", how="left")
    else:
        frame = latest_scores
    rows = []
    for row in frame.to_dicts():
        symbol = str(row.get("symbol", ""))
        missing = str(row.get("missing_evidence") or missing_evidence_for_symbol(coverage, symbol))
        rows.append(
            {
                "rank": 0,
                "timestamp": int(row.get("timestamp", 0) or 0),
                "symbol": symbol,
                "score_total": int(row.get("score_total", 0) or 0),
                "alert_level": str(row.get("alert_level", "none")),
                "confidence_level": str(row.get("confidence_level", "blocked")),
                "suggestion_type": str(row.get("suggestion_type", "reject_or_deprioritize")),
                "structure_state": str(row.get("structure_state", "unknown")),
                "preparation_state": str(row.get("preparation_state", "not_ready")),
                "flow_state": str(row.get("flow_state", "unknown")),
                "attention_state": str(row.get("attention_state", "unknown")),
                "activation_state": str(row.get("activation_state", "inactive")),
                "risk_state": str(row.get("risk_state", "data_insufficient")),
                "return_24h": row.get("return_24h"),
                "range_position_pct": row.get("range_position_pct"),
                "volatility_compression_pctile": row.get("volatility_compression_pctile"),
                "depth_imbalance_25_mean": row.get("depth_imbalance_25_mean"),
                "large_trade_buy_ratio": row.get("large_trade_buy_ratio"),
                "resilience_score": row.get("resilience_score"),
                "funding_rate": row.get("funding_rate"),
                "open_interest_change_24h": row.get("open_interest_change_24h"),
                "positive_components": str(row.get("positive_components") or ""),
                "negative_filters": str(row.get("negative_filters") or ""),
                "missing_evidence": missing,
                "next_fetch_action": _next_fetch_action(row, missing),
                "data_quality_warning": str(row.get("data_quality_warning") or ""),
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
        symbol = str(row.get("symbol", ""))
        missing = str(row.get("missing_evidence") or missing_evidence_for_symbol(coverage, symbol))
        rows.extend(next_fetch_actions_for_row(row, missing, policy=policy))
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
            f"suggestion={row.get('suggestion_type', 'reject_or_deprioritize')}"
        )
        lines.append(f"Rationale: {row['rationale']}")
        if row.get("next_fetch_action"):
            lines.append(f"Next fetch: {row['next_fetch_action']}")
        lines.append("")
    return "\n".join(lines)


def _next_fetch_action(row: dict, missing: str) -> str:
    actions = next_fetch_actions_for_row(row, missing)
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
    symbol = str(row.get("symbol", ""))
    score = int(row.get("score_total", 0) or 0)
    suggestion = str(row.get("suggestion_type") or "")
    watchlist = suggestion in policy.watchlist_suggestions
    rejected = suggestion == "reject_or_deprioritize"
    actions = []
    if "contract_metadata_missing" in missing or "trade_notional_metadata_missing" in missing:
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
    if not rejected and "onchain" in missing and (score >= policy.yellow_threshold or watchlist):
        actions.append(
            _action(symbol, 2, "onchain", "collect-onchain", "exchange flow missing", "high", True)
        )
    message_missing = any(
        token in missing for token in ("messages_missing", "message_classifications_missing")
    )
    if not rejected and message_missing and (score >= policy.yellow_threshold or watchlist):
        actions.append(
            _action(
                symbol, 3, "messages", "collect-context", "message context missing", "medium", False
            )
        )
    for source in ("funding", "open_interest"):
        if f"{source}_missing" in missing:
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
