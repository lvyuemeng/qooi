"""Reusable source schemas shared by scanner source manifests."""

from __future__ import annotations

import polars as pl

SOURCE_MANIFEST_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Int64,
    "symbol": pl.String,
    "source": pl.String,
    "phase": pl.String,
    "status": pl.String,
    "backend": pl.String,
    "endpoint": pl.String,
    "rows": pl.Int64,
    "range_start": pl.Int64,
    "range_end": pl.Int64,
    "coverage_pct": pl.Float64,
    "warning": pl.String,
    "stop_reason": pl.String,
}

SOURCE_BARS_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.String,
    "timestamp": pl.Int64,
    "datetime": pl.String,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "vol": pl.Float64,
}

SOURCE_BOOKS_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.String,
    "timestamp": pl.Int64,
    "ob_bid_price": pl.Float64,
    "ob_ask_price": pl.Float64,
    "ob_bid_vol_5": pl.Float64,
    "ob_ask_vol_5": pl.Float64,
    "ob_bid_vol_10": pl.Float64,
    "ob_ask_vol_10": pl.Float64,
    "ob_bid_vol_25": pl.Float64,
    "ob_ask_vol_25": pl.Float64,
    "ob_bid_vol": pl.Float64,
    "ob_ask_vol": pl.Float64,
    "ob_imbalance_5": pl.Float64,
    "ob_imbalance_10": pl.Float64,
    "ob_imbalance_25": pl.Float64,
}

SOURCE_TRADES_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.String,
    "timestamp": pl.Int64,
    "trade_id": pl.String,
    "price": pl.Float64,
    "size": pl.Float64,
    "side": pl.String,
    "notional_usd": pl.Float64,
}

SOURCE_FUNDING_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.String,
    "timestamp": pl.Int64,
    "funding_time": pl.Int64,
    "funding_rate": pl.Float64,
}

SOURCE_OPEN_INTEREST_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.String,
    "timestamp": pl.Int64,
    "open_interest": pl.Float64,
    "open_interest_ccy": pl.Float64,
    "open_interest_usd": pl.Float64,
}

SOURCE_TAKER_VOLUME_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.String,
    "timestamp": pl.Int64,
    "taker_sell_volume": pl.Float64,
    "taker_buy_volume": pl.Float64,
    "taker_volume_unit": pl.String,
}

SOURCE_LONG_SHORT_RATIO_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.String,
    "timestamp": pl.Int64,
    "long_short_account_ratio": pl.Float64,
    "top_trader_long_short_account_ratio": pl.Float64,
    "top_trader_long_short_position_ratio": pl.Float64,
}

SOURCE_ONCHAIN_FLOWS_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.String,
    "timestamp": pl.Int64,
    "inflow": pl.Float64,
    "outflow": pl.Float64,
    "net_exchange_flow": pl.Float64,
}

SOURCE_MESSAGES_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.String,
    "timestamp": pl.Int64,
    "source": pl.String,
    "source_id": pl.String,
    "author_id_hash": pl.String,
    "text_hash": pl.String,
    "lang": pl.String,
    "text": pl.String,
    "url": pl.String,
    "engagement_count": pl.Int64,
    "reply_count": pl.Int64,
    "repost_count": pl.Int64,
}

SOURCE_POLYMARKET_MARKETS_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.String,
    "timestamp": pl.Int64,
    "query": pl.String,
    "provider": pl.String,
    "market_id": pl.String,
    "event_id": pl.String,
    "slug": pl.String,
    "question": pl.String,
    "description": pl.String,
    "category": pl.String,
    "active": pl.Boolean,
    "closed": pl.Boolean,
    "start_time": pl.Int64,
    "end_time": pl.Int64,
    "volume_24h": pl.Float64,
    "volume_1w": pl.Float64,
    "volume_1mo": pl.Float64,
    "volume_total": pl.Float64,
    "liquidity": pl.Float64,
    "open_interest": pl.Float64,
    "yes_price": pl.Float64,
    "no_price": pl.Float64,
    "last_trade_price": pl.Float64,
    "best_bid": pl.Float64,
    "best_ask": pl.Float64,
    "spread": pl.Float64,
    "price_change_1h": pl.Float64,
    "price_change_1d": pl.Float64,
    "matched_alias": pl.String,
    "match_method": pl.String,
    "url": pl.String,
    "data_quality_warning": pl.String,
}

SOURCE_POLYMARKET_EVENTS_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.String,
    "timestamp": pl.Int64,
    "query": pl.String,
    "provider": pl.String,
    "event_id": pl.String,
    "slug": pl.String,
    "title": pl.String,
    "description": pl.String,
    "category": pl.String,
    "active": pl.Boolean,
    "closed": pl.Boolean,
    "start_time": pl.Int64,
    "end_time": pl.Int64,
    "volume_24h": pl.Float64,
    "volume_1w": pl.Float64,
    "volume_1mo": pl.Float64,
    "volume_total": pl.Float64,
    "liquidity": pl.Float64,
    "open_interest": pl.Float64,
    "market_count": pl.Int64,
    "comment_count": pl.Int64,
    "matched_alias": pl.String,
    "match_method": pl.String,
    "url": pl.String,
    "data_quality_warning": pl.String,
}

MESSAGE_CLASSIFICATION_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.String,
    "timestamp": pl.Int64,
    "message_id": pl.String,
    "message_type": pl.String,
    "message_type_confidence": pl.Float64,
    "stage_hint": pl.String,
    "model_name": pl.String,
    "model_version": pl.String,
    "data_quality_warning": pl.String,
}

