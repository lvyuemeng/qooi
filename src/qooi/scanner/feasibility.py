"""Scanner feasibility projection products."""

from __future__ import annotations

import polars as pl

from qooi.exchange.store import HistoryCoverage
from qooi.scanner import ScanDecision
from qooi.sources.context import SourceAvailability

CANDIDATE_FEASIBILITY_SCHEMA = {
    "symbol": pl.String,
    "outcome_horizon": pl.Int64,
    "watchlist_feasibility": pl.String,
    "rank_score": pl.Float64,
    "rank_tier": pl.String,
    "source_penalty_score": pl.Float64,
    "required_missing_source_count": pl.Int64,
    "required_stale_source_count": pl.Int64,
    "provider_bounded_source_count": pl.Int64,
    "optional_absent_source_count": pl.Int64,
    "min_history_coverage_pct": pl.Float64,
    "min_source_capability_coverage_pct": pl.Float64,
    "tree_direction": pl.String,
    "matched_evidence_level": pl.String,
    "tail_lift": pl.Float64,
    "gpd_shape_xi": pl.Float64,
    "N_tail_exceedances": pl.Int64,
    "source_status": pl.String,
    "history_status": pl.String,
    "candidate_reason": pl.String,
}

HISTORY_FEASIBILITY_SCHEMA = {
    "symbol": pl.String,
    "bar": pl.String,
    "target_rows": pl.Int64,
    "actual_rows": pl.Int64,
    "coverage_pct": pl.Float64,
    "range_start": pl.Int64,
    "range_end": pl.Int64,
    "newest_age_hours": pl.Float64,
    "gap_count": pl.Int64,
    "duplicate_timestamps": pl.Int64,
    "refreshed": pl.Boolean,
    "feasibility_status": pl.String,
    "feasibility_reason": pl.String,
    "notes": pl.String,
}

_CANDIDATE_FEASIBILITY_COLUMNS = tuple(CANDIDATE_FEASIBILITY_SCHEMA)

_CANDIDATE_SELECTION_COLUMNS = (
    "symbol",
    "outcome_horizon",
    "rank_score",
    "rank_tier",
    "source_penalty_score",
    "required_missing_source_count",
    "required_stale_source_count",
    "provider_bounded_source_count",
    "optional_absent_source_count",
    "tree_direction",
    "matched_evidence_level",
    "tail_lift",
    "gpd_shape_xi",
    "N_tail_exceedances",
    "rank_reason",
)

_WATCHLIST_FEASIBILITY_COLUMNS = (
    "symbol",
    "watchlist_feasibility",
    "min_history_coverage_pct",
    "min_source_capability_coverage_pct",
    "source_status",
    "history_status",
)

_SOURCE_CONSTRAINT_SCHEMA = {
    "symbol": pl.String,
    "required_missing_source_count": pl.Int64,
    "required_stale_source_count": pl.Int64,
    "provider_bounded_source_count": pl.Int64,
    "optional_absent_source_count": pl.Int64,
    "source_penalty_score": pl.Float64,
}

_EMPTY_HISTORY_STATUS_SCHEMA = {
    "symbol": pl.String,
    "history_status": pl.String,
    "history_reason": pl.String,
}

_EMPTY_SOURCE_STATUS_SCHEMA = {
    "symbol": pl.String,
    "source_status": pl.String,
    "source_reason": pl.String,
}


def history_feasibility_frame(coverage: tuple[HistoryCoverage, ...]) -> pl.DataFrame:
    rows = []
    for item in coverage:
        note_text = ";".join(item.notes)
        rows.append(
            {
                "symbol": item.inst_id,
                "bar": item.bar,
                "target_rows": item.target.target_bars,
                "actual_rows": item.actual_bars,
                "coverage_pct": item.coverage_pct,
                "range_start": item.actual_start_ms,
                "range_end": item.actual_end_ms,
                "newest_age_hours": item.newest_age_hours,
                "gap_count": item.gap_count,
                "duplicate_timestamps": item.duplicate_timestamps,
                "refreshed": item.refreshed,
                "feasibility_status": _history_feasibility_status(item, note_text),
                "feasibility_reason": _history_feasibility_reason(item, note_text),
                "notes": note_text,
            }
        )
    return pl.DataFrame(rows, schema=HISTORY_FEASIBILITY_SCHEMA)


def watchlist_feasibility_frame(
    decisions: tuple[ScanDecision, ...],
    history_feasibility: pl.DataFrame,
    source_freshness: pl.DataFrame,
) -> pl.DataFrame:
    rows = pl.DataFrame(
        {
            "symbol": [decision.symbol for decision in decisions],
            "group": [decision.group for decision in decisions],
            "direction": [decision.direction for decision in decisions],
            "confidence": [decision.confidence for decision in decisions],
            "missing_evidence": [";".join(decision.missing_evidence) for decision in decisions],
            "contradictory_evidence": [
                ";".join(decision.contradictory_evidence) for decision in decisions
            ],
            "block_reason": [decision.block_reason for decision in decisions],
        },
        schema={
            "symbol": pl.String,
            "group": pl.String,
            "direction": pl.String,
            "confidence": pl.String,
            "missing_evidence": pl.String,
            "contradictory_evidence": pl.String,
            "block_reason": pl.String,
        },
    )
    history = _symbol_history_feasibility(history_feasibility)
    source = _symbol_source_feasibility(source_freshness)
    rows = rows.join(history, on="symbol", how="left")
    rows = rows.join(source, on="symbol", how="left")
    return rows.with_columns(
        pl.col("history_status").fill_null("history_missing"),
        pl.col("source_status").fill_null("sources_not_loaded"),
        pl.col("history_reason").fill_null("no history coverage row for symbol"),
        pl.col("source_reason").fill_null("no source freshness rows for symbol"),
    ).with_columns(
        pl.when(pl.col("group") == "blocked")
        .then(pl.lit("blocked_by_evidence_gate"))
        .when(pl.col("history_status").is_in(["missing_history", "history_integrity_issue"]))
        .then(pl.lit("blocked_by_history"))
        .when(
            pl.col("source_status").is_in(
                ["required_sources_missing", "required_sources_stale"]
            )
        )
        .then(pl.lit("source_limited_review"))
        .when(pl.col("history_status").is_in(["cache_incomplete", "fetch_limited", "low_coverage"]))
        .then(pl.lit("coverage_limited_review"))
        .otherwise(pl.lit("reviewable"))
        .alias("watchlist_feasibility")
    )


def candidate_feasibility_frame(
    candidate_rank: pl.DataFrame, watchlist_feasibility: pl.DataFrame
) -> pl.DataFrame:
    if candidate_rank.is_empty():
        return pl.DataFrame(schema=CANDIDATE_FEASIBILITY_SCHEMA)
    best = (
        candidate_rank.sort(
            [
                "rank_score",
                "source_penalty_score",
                "required_missing_source_count",
                "required_stale_source_count",
                "tail_lift",
            ],
            descending=[True, False, False, False, True],
        )
        .unique(subset=["symbol"], keep="first", maintain_order=True)
        .with_columns(
            _rank_tier_expr().alias("rank_tier"),
        )
    )
    selected = best.select(_CANDIDATE_SELECTION_COLUMNS)
    if watchlist_feasibility.is_empty():
        return selected.with_columns(
            pl.lit("unclassified").alias("watchlist_feasibility"),
            pl.lit(None, dtype=pl.Float64).alias("min_history_coverage_pct"),
            pl.lit(None, dtype=pl.Float64).alias("min_source_capability_coverage_pct"),
            pl.lit("missing").alias("source_status"),
            pl.lit("missing").alias("history_status"),
            pl.col("rank_reason").alias("candidate_reason"),
        ).select(_CANDIDATE_FEASIBILITY_COLUMNS)

    return (
        selected.join(
            watchlist_feasibility.select(_WATCHLIST_FEASIBILITY_COLUMNS),
            on="symbol",
            how="left",
        )
        .with_columns(
            pl.col("watchlist_feasibility").fill_null("unclassified"),
            pl.col("source_status").fill_null("missing"),
            pl.col("history_status").fill_null("missing"),
        )
        .with_columns(_candidate_reason_expr().alias("candidate_reason"))
        .select(_CANDIDATE_FEASIBILITY_COLUMNS)
    )


def join_candidate_source_constraints(
    candidates: pl.DataFrame, availability: tuple[SourceAvailability, ...]
) -> pl.DataFrame:
    if candidates.is_empty() or not availability:
        return candidates
    source_context = pl.DataFrame(
        {
            "symbol": [row.symbol for row in availability],
            "required_missing_source_count": [row.frame_missing_int for row in availability],
            "required_stale_source_count": [row.frame_stale_int for row in availability],
            "provider_bounded_source_count": [row.provider_bounded_int for row in availability],
            "optional_absent_source_count": [row.optional_absent_int for row in availability],
            "source_penalty_score": [row.source_penalty_component for row in availability],
        },
        schema=_SOURCE_CONSTRAINT_SCHEMA,
    )
    if source_context.is_empty():
        return candidates
    by_symbol = source_context.group_by("symbol").agg(
        pl.sum("required_missing_source_count"),
        pl.sum("required_stale_source_count"),
        pl.sum("provider_bounded_source_count"),
        pl.sum("optional_absent_source_count"),
        pl.sum("source_penalty_score"),
    )
    existing = [
        column
        for column in by_symbol.columns
        if column in candidates.columns and column != "symbol"
    ]
    base = candidates.drop(existing) if existing else candidates
    return base.join(by_symbol, on="symbol", how="left")


def _history_feasibility_status(item: HistoryCoverage, note_text: str) -> str:
    if item.actual_bars <= 0:
        return "missing_history"
    if "cache_only=yes" in note_text and item.coverage_pct < 95.0:
        return "cache_incomplete"
    if item.gap_count > 0 or item.duplicate_timestamps > 0:
        return "history_integrity_issue"
    if "page_error" in note_text or "HTTPStatusError" in note_text:
        return "fetch_limited"
    if "starts_after_target_since" in note_text and item.coverage_pct < 95.0:
        return "history_start_limited"
    if item.coverage_pct < 80.0:
        return "low_coverage"
    if item.coverage_pct < 95.0:
        return "partial_coverage"
    return "reviewable_history"


def _history_feasibility_reason(item: HistoryCoverage, note_text: str) -> str:
    if item.actual_bars <= 0:
        return "no bars available for requested history"
    if "cache_only=yes" in note_text and item.coverage_pct < 95.0:
        return "cache-only mode prevented filling the requested history"
    if item.gap_count > 0 or item.duplicate_timestamps > 0:
        return "timeline has gaps or duplicate timestamps"
    if "page_error" in note_text or "HTTPStatusError" in note_text:
        return "fetch stopped with provider or transport error"
    if "starts_after_target_since" in note_text and item.coverage_pct < 95.0:
        return "available exchange/cache history starts after requested horizon"
    if item.coverage_pct < 80.0:
        return "coverage is materially below requested target"
    if item.coverage_pct < 95.0:
        return "coverage is below requested target but may still support current-state review"
    return "requested history is available and timeline is clean"


def _symbol_history_feasibility(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(schema=_EMPTY_HISTORY_STATUS_SCHEMA)
    status_rank = pl.when(pl.col("feasibility_status") == "missing_history").then(0)
    status_rank = status_rank.when(
        pl.col("feasibility_status") == "history_integrity_issue"
    ).then(1)
    status_rank = status_rank.when(pl.col("feasibility_status") == "fetch_limited").then(2)
    status_rank = status_rank.when(pl.col("feasibility_status") == "cache_incomplete").then(3)
    status_rank = status_rank.when(pl.col("feasibility_status") == "low_coverage").then(4)
    status_rank = status_rank.when(
        pl.col("feasibility_status") == "history_start_limited"
    ).then(5)
    status_rank = status_rank.when(pl.col("feasibility_status") == "partial_coverage").then(6)
    status_rank = status_rank.otherwise(7).alias("status_rank")
    return (
        frame.with_columns(status_rank)
        .sort(["symbol", "status_rank", "coverage_pct"], descending=[False, False, False])
        .group_by("symbol")
        .agg(
            pl.first("feasibility_status").alias("history_status"),
            pl.first("feasibility_reason").alias("history_reason"),
            pl.min("coverage_pct").alias("min_history_coverage_pct"),
        )
    )


def _symbol_source_feasibility(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(schema=_EMPTY_SOURCE_STATUS_SCHEMA)
    return (
        frame.group_by("symbol")
        .agg(
            pl.len().alias("source_family_rows"),
            pl.col("usable_int").sum().alias("fresh_source_families"),
            pl.col("frame_missing_int").sum().alias("missing_source_families"),
            pl.col("frame_missing_int").sum().alias("required_missing_source_count"),
            pl.col("frame_stale_int").sum().alias("required_stale_source_count"),
            pl.col("provider_bounded_int").sum().alias("provider_bounded_source_count"),
            pl.col("optional_absent_int").sum().alias("optional_absent_source_count"),
            pl.when(pl.col("required_for_review_int") == 1)
            .then(pl.col("coverage_capability_pct"))
            .otherwise(None)
            .min()
            .alias("min_source_capability_coverage_pct"),
            pl.sum("source_penalty_component").alias("source_penalty_score"),
            pl.concat_str("source_family", "status", separator="=")
            .str.join(";")
            .alias("source_reason"),
        )
        .with_columns(
            pl.when(pl.col("required_missing_source_count") > 0)
            .then(pl.lit("required_sources_missing"))
            .when(pl.col("required_stale_source_count") > 0)
            .then(pl.lit("required_sources_stale"))
            .when(pl.col("fresh_source_families") > 0)
            .then(pl.lit("source_context_available"))
            .otherwise(pl.lit("source_context_partial"))
            .alias("source_status")
        )
        .select(
            "symbol",
            "source_status",
            "source_reason",
            "source_family_rows",
            "fresh_source_families",
            "missing_source_families",
            "required_missing_source_count",
            "required_stale_source_count",
            "provider_bounded_source_count",
            "optional_absent_source_count",
            "min_source_capability_coverage_pct",
            "source_penalty_score",
        )
    )


def _candidate_reason_expr() -> pl.Expr:
    return (
        pl.when(pl.col("required_missing_source_count") > 0)
        .then(
            pl.lit("missing_required_sources=")
            + pl.col("required_missing_source_count").cast(pl.String)
        )
        .when(pl.col("required_stale_source_count") > 0)
        .then(
            pl.lit("stale_required_sources=")
            + pl.col("required_stale_source_count").cast(pl.String)
        )
        .when(pl.col("watchlist_feasibility") == "coverage_limited_review")
        .then(pl.lit("history=") + pl.col("history_status"))
        .when(pl.col("watchlist_feasibility") == "blocked_by_evidence_gate")
        .then(pl.lit("evidence_gate_blocked"))
        .when(pl.col("watchlist_feasibility") == "reviewable")
        .then(pl.lit("reviewable"))
        .otherwise(pl.col("rank_reason"))
    )


def _rank_tier_expr() -> pl.Expr:
    return (
        pl.when(
            (pl.col("tail_lift").fill_null(0.0) >= 2.0)
            & (pl.col("N_tail_exceedances").fill_null(0) >= 50)
            & (pl.col("gpd_shape_xi").fill_null(0.0) > 0.15)
        )
        .then(pl.lit("1"))
        .when(
            (pl.col("tail_lift").fill_null(0.0) >= 1.5)
            & (pl.col("N_tail_exceedances").fill_null(0) >= 30)
        )
        .then(pl.lit("2"))
        .when(pl.col("tail_lift").fill_null(0.0) >= 1.0)
        .then(pl.lit("3"))
        .otherwise(pl.lit("—"))
    )
