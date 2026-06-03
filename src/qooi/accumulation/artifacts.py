"""CSV artifact catalog for accumulation scanner outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

from qooi.accumulation.schema import (
    BACKTEST_EVENT_SCHEMA,
    BROAD_CANDIDATE_SCHEMA,
    BROAD_MARKET_SCHEMA,
    BROAD_NEWS_SCHEMA,
    BROAD_PROTOCOL_SCHEMA,
    CANDIDATE_DETAIL_SCHEMA,
    CANDIDATE_SUMMARY_SCHEMA,
    DISCOVERY_SCHEMA,
    FEATURE_SCHEMA,
    MESSAGE_CLASSIFICATION_SCHEMA,
    NEXT_FETCH_ACTION_SCHEMA,
    SCORE_SCHEMA,
    SOURCE_BARS_SCHEMA,
    SOURCE_BOOKS_SCHEMA,
    SOURCE_FUNDING_SCHEMA,
    SOURCE_LONG_SHORT_RATIO_SCHEMA,
    SOURCE_MANIFEST_SCHEMA,
    SOURCE_MESSAGES_SCHEMA,
    SOURCE_ONCHAIN_FLOWS_SCHEMA,
    SOURCE_OPEN_INTEREST_SCHEMA,
    SOURCE_POLYMARKET_EVENTS_SCHEMA,
    SOURCE_POLYMARKET_MARKETS_SCHEMA,
    SOURCE_TAKER_VOLUME_SCHEMA,
    SOURCE_TRADES_SCHEMA,
)
from qooi.strategies.potential import (
    POTENTIAL_BOARD_SCHEMA,
    POTENTIAL_SOURCE_SCHEMA,
)

ArtifactName = Literal[
    "candidate_discovery",
    "source_manifest",
    "data_coverage",
    "source_bars",
    "source_books",
    "source_trades",
    "source_funding",
    "source_open_interest",
    "source_taker_volume",
    "source_long_short_ratios",
    "source_onchain_flows",
    "source_messages",
    "source_polymarket_events",
    "source_polymarket_markets",
    "message_classifications",
    "features",
    "scores",
    "alerts",
    "candidate_detail",
    "candidate_summary",
    "next_fetch_actions",
    "backtest_events",
    "backtest_summary",
    "broad_market_snapshot",
    "broad_protocol_snapshot",
    "broad_news_snapshot",
    "broad_candidates",
    "potential_board",
    "potential_sources",
]


@dataclass(frozen=True)
class ArtifactSpec:
    name: ArtifactName
    relative_path: str
    schema: dict[str, pl.DataType]
    required: bool = True


ARTIFACT_SPECS: dict[ArtifactName, ArtifactSpec] = {
    "candidate_discovery": ArtifactSpec(
        "candidate_discovery", "candidate-discovery.csv", DISCOVERY_SCHEMA
    ),
    "source_manifest": ArtifactSpec(
        "source_manifest", "source-manifest.csv", SOURCE_MANIFEST_SCHEMA
    ),
    "data_coverage": ArtifactSpec(
        "data_coverage", "accumulation-data-coverage.csv", SOURCE_MANIFEST_SCHEMA
    ),
    "source_bars": ArtifactSpec("source_bars", "sources/bars.csv", SOURCE_BARS_SCHEMA),
    "source_books": ArtifactSpec("source_books", "sources/books.csv", SOURCE_BOOKS_SCHEMA),
    "source_trades": ArtifactSpec("source_trades", "sources/trades.csv", SOURCE_TRADES_SCHEMA),
    "source_funding": ArtifactSpec("source_funding", "sources/funding.csv", SOURCE_FUNDING_SCHEMA),
    "source_open_interest": ArtifactSpec(
        "source_open_interest", "sources/open-interest.csv", SOURCE_OPEN_INTEREST_SCHEMA
    ),
    "source_taker_volume": ArtifactSpec(
        "source_taker_volume", "sources/taker-volume-contract.csv", SOURCE_TAKER_VOLUME_SCHEMA
    ),
    "source_long_short_ratios": ArtifactSpec(
        "source_long_short_ratios", "sources/long-short-ratios.csv", SOURCE_LONG_SHORT_RATIO_SCHEMA
    ),
    "source_onchain_flows": ArtifactSpec(
        "source_onchain_flows", "sources/onchain-flows.csv", SOURCE_ONCHAIN_FLOWS_SCHEMA
    ),
    "source_messages": ArtifactSpec(
        "source_messages", "sources/messages-normalized.csv", SOURCE_MESSAGES_SCHEMA
    ),
    "source_polymarket_events": ArtifactSpec(
        "source_polymarket_events",
        "sources/polymarket-events.csv",
        SOURCE_POLYMARKET_EVENTS_SCHEMA,
    ),
    "source_polymarket_markets": ArtifactSpec(
        "source_polymarket_markets",
        "sources/polymarket-markets.csv",
        SOURCE_POLYMARKET_MARKETS_SCHEMA,
    ),
    "message_classifications": ArtifactSpec(
        "message_classifications",
        "sources/message-classifications.csv",
        MESSAGE_CLASSIFICATION_SCHEMA,
    ),
    "features": ArtifactSpec("features", "accumulation-features.csv", FEATURE_SCHEMA),
    "scores": ArtifactSpec("scores", "accumulation-scores.csv", SCORE_SCHEMA),
    "alerts": ArtifactSpec("alerts", "accumulation-alerts.csv", SCORE_SCHEMA),
    "candidate_detail": ArtifactSpec(
        "candidate_detail", "candidate-detail.csv", CANDIDATE_DETAIL_SCHEMA
    ),
    "candidate_summary": ArtifactSpec(
        "candidate_summary", "candidate-summary.csv", CANDIDATE_SUMMARY_SCHEMA
    ),
    "next_fetch_actions": ArtifactSpec(
        "next_fetch_actions", "next-fetch-actions.csv", NEXT_FETCH_ACTION_SCHEMA
    ),
    "backtest_events": ArtifactSpec(
        "backtest_events", "accumulation-backtest-events.csv", BACKTEST_EVENT_SCHEMA
    ),
    "backtest_summary": ArtifactSpec("backtest_summary", "accumulation-backtest-summary.csv", {}),
    "broad_market_snapshot": ArtifactSpec(
        "broad_market_snapshot", "broad/market-snapshot.csv", BROAD_MARKET_SCHEMA
    ),
    "broad_protocol_snapshot": ArtifactSpec(
        "broad_protocol_snapshot", "broad/protocols.csv", BROAD_PROTOCOL_SCHEMA
    ),
    "broad_news_snapshot": ArtifactSpec("broad_news_snapshot", "broad/news.csv", BROAD_NEWS_SCHEMA),
    "broad_candidates": ArtifactSpec(
        "broad_candidates", "broad/candidates.csv", BROAD_CANDIDATE_SCHEMA
    ),
    "potential_board": ArtifactSpec(
        "potential_board", "potential/board.csv", POTENTIAL_BOARD_SCHEMA
    ),
    "potential_sources": ArtifactSpec(
        "potential_sources", "potential/sources.csv", POTENTIAL_SOURCE_SCHEMA
    ),
}


def artifact_spec(name: ArtifactName) -> ArtifactSpec:
    return ARTIFACT_SPECS[name]
