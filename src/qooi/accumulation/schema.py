"""Canonical schemas for accumulation scanner artifacts."""

from __future__ import annotations

import polars as pl

from qooi.sources.schema import SOURCE_MANIFEST_SCHEMA

FEATURE_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Int64,
    "symbol": pl.String,
    "inst_id": pl.String,
    "chain": pl.String,
    "token_address": pl.String,
    "close": pl.Float64,
    "return_1h": pl.Float64,
    "return_24h": pl.Float64,
    "ma200": pl.Float64,
    "net_exchange_flow": pl.Float64,
    "flow_zscore": pl.Float64,
    "flow_zscore_negative_streak_hours": pl.Int64,
    "whale_accumulation_ratio": pl.Float64,
    "depth_imbalance_10_mean": pl.Float64,
    "depth_imbalance_25_mean": pl.Float64,
    "depth_imbalance_10_slope": pl.Float64,
    "large_sell_events": pl.Int64,
    "resilient_sell_events": pl.Int64,
    "resilience_score": pl.Float64,
    "large_trade_buy_ratio": pl.Float64,
    "mention_growth": pl.Float64,
    "fundamental_news_ratio": pl.Float64,
    "emotion_news_ratio": pl.Float64,
    "quote_volume_24h": pl.Float64,
    "spread_bps_mean": pl.Float64,
    "depth_rebuild_score": pl.Float64,
    "funding_rate": pl.Float64,
    "funding_zscore": pl.Float64,
    "open_interest_usd": pl.Float64,
    "open_interest_change_24h": pl.Float64,
    "open_interest_usd_change_24h": pl.Float64,
    "taker_buy_volume": pl.Float64,
    "taker_sell_volume": pl.Float64,
    "taker_buy_ratio": pl.Float64,
    "taker_volume_total": pl.Float64,
    "taker_volume_imbalance": pl.Float64,
    "long_short_account_ratio": pl.Float64,
    "top_trader_long_short_account_ratio": pl.Float64,
    "top_trader_long_short_position_ratio": pl.Float64,
    "volatility_compression_pctile": pl.Float64,
    "range_position_pct": pl.Float64,
    "range_low_px": pl.Float64,
    "range_high_px": pl.Float64,
    "range_width_pct": pl.Float64,
    "upside_to_range_high_pct": pl.Float64,
    "downside_to_range_low_pct": pl.Float64,
    "range_reward_risk": pl.Float64,
    "structure_invalidation_px": pl.Float64,
    "structure_target_px": pl.Float64,
    "structure_stage": pl.String,
    "source_coverage_score": pl.Float64,
    "polymarket_related_market_count": pl.Int64,
    "polymarket_volume_24h_total": pl.Float64,
    "polymarket_event_driven_context": pl.Boolean,
    "data_quality_warning": pl.String,
}

SCORE_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Int64,
    "symbol": pl.String,
    "score_total": pl.Int64,
    "alert_level": pl.String,
    "onchain_score": pl.Int64,
    "orderbook_score": pl.Int64,
    "trade_score": pl.Int64,
    "message_score": pl.Int64,
    "negative_score": pl.Int64,
    "flow_outflow_3sigma_2h": pl.Boolean,
    "whale_accumulation_high": pl.Boolean,
    "depth_support_on_down_day": pl.Boolean,
    "resilience_high": pl.Boolean,
    "message_not_overheated": pl.Boolean,
    "exchange_inflow_spike": pl.Boolean,
    "message_overheated": pl.Boolean,
    "below_ma200_weak_depth": pl.Boolean,
    "explanation": pl.String,
    "positive_components": pl.String,
    "negative_filters": pl.String,
    "source_coverage_score": pl.Float64,
    "confidence_level": pl.String,
    "structure_state": pl.String,
    "preparation_state": pl.String,
    "flow_state": pl.String,
    "attention_state": pl.String,
    "activation_state": pl.String,
    "risk_state": pl.String,
    "suggestion_type": pl.String,
    "missing_evidence": pl.String,
    "data_quality_warning": pl.String,
}

DISCOVERY_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.String,
    "inst_id": pl.String,
    "inst_type": pl.String,
    "state": pl.String,
    "base_ccy": pl.String,
    "quote_ccy": pl.String,
    "settle_ccy": pl.String,
    "ct_val": pl.Float64,
    "ct_val_ccy": pl.String,
    "list_time": pl.Int64,
    "quote_volume_24h": pl.Float64,
    "last": pl.Float64,
    "bid_px": pl.Float64,
    "ask_px": pl.Float64,
    "spread_bps": pl.Float64,
    "history_coverage_pct": pl.Float64,
    "eligible": pl.Boolean,
    "exclude_reason": pl.String,
    "rank_score": pl.Float64,
}

BROAD_MARKET_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Int64,
    "provider": pl.String,
    "coin_id": pl.String,
    "base_ccy": pl.String,
    "name": pl.String,
    "rank": pl.Int64,
    "price_usd": pl.Float64,
    "market_cap_usd": pl.Float64,
    "volume_24h_usd": pl.Float64,
    "volume_24h_change_pct": pl.Float64,
    "price_change_pct_1h": pl.Float64,
    "price_change_pct_24h": pl.Float64,
    "last_updated": pl.Int64,
    "trending_rank": pl.Int64,
    "trending_score": pl.Float64,
    "heat_source": pl.String,
}

BROAD_PROTOCOL_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Int64,
    "provider": pl.String,
    "protocol": pl.String,
    "base_ccy": pl.String,
    "name": pl.String,
    "category": pl.String,
    "chains": pl.String,
    "tvl_usd": pl.Float64,
    "tvl_change_1d_pct": pl.Float64,
    "tvl_change_7d_pct": pl.Float64,
}

BROAD_NEWS_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Int64,
    "provider": pl.String,
    "source_id": pl.String,
    "title": pl.String,
    "url": pl.String,
    "base_ccy": pl.String,
    "sentiment": pl.String,
}

BROAD_CANDIDATE_SCHEMA: dict[str, pl.DataType] = {
    "rank": pl.Int64,
    "timestamp": pl.Int64,
    "base_ccy": pl.String,
    "coin_id": pl.String,
    "name": pl.String,
    "okx_symbol": pl.String,
    "okx_mapped": pl.Boolean,
    "market_cap_usd": pl.Float64,
    "volume_24h_usd": pl.Float64,
    "price_change_pct_1h": pl.Float64,
    "price_change_pct_24h": pl.Float64,
    "trending_rank": pl.Int64,
    "trending_score": pl.Float64,
    "heat_source": pl.String,
    "tvl_usd": pl.Float64,
    "tvl_change_1d_pct": pl.Float64,
    "news_mentions": pl.Int64,
    "broad_score": pl.Float64,
    "broad_reasons": pl.String,
    "exclude_reason": pl.String,
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

CANDIDATE_SUMMARY_SCHEMA: dict[str, pl.DataType] = {
    "rank": pl.Int64,
    "timestamp": pl.Int64,
    "symbol": pl.String,
    "alert_level": pl.String,
    "score_total": pl.Int64,
    "source_coverage_score": pl.Float64,
    "confidence_level": pl.String,
    "structure_state": pl.String,
    "preparation_state": pl.String,
    "flow_state": pl.String,
    "attention_state": pl.String,
    "activation_state": pl.String,
    "risk_state": pl.String,
    "suggestion_type": pl.String,
    "top_positive_components": pl.String,
    "top_negative_filters": pl.String,
    "data_quality_warning": pl.String,
    "missing_evidence": pl.String,
    "next_fetch_action": pl.String,
    "rationale": pl.String,
}

CANDIDATE_DETAIL_SCHEMA: dict[str, pl.DataType] = {
    "rank": pl.Int64,
    "timestamp": pl.Int64,
    "symbol": pl.String,
    "score_total": pl.Int64,
    "alert_level": pl.String,
    "confidence_level": pl.String,
    "suggestion_type": pl.String,
    "structure_state": pl.String,
    "preparation_state": pl.String,
    "flow_state": pl.String,
    "attention_state": pl.String,
    "activation_state": pl.String,
    "risk_state": pl.String,
    "return_24h": pl.Float64,
    "close": pl.Float64,
    "range_position_pct": pl.Float64,
    "range_low_px": pl.Float64,
    "range_high_px": pl.Float64,
    "upside_to_range_high_pct": pl.Float64,
    "downside_to_range_low_pct": pl.Float64,
    "range_reward_risk": pl.Float64,
    "structure_invalidation_px": pl.Float64,
    "structure_target_px": pl.Float64,
    "volatility_compression_pctile": pl.Float64,
    "depth_imbalance_25_mean": pl.Float64,
    "large_trade_buy_ratio": pl.Float64,
    "resilience_score": pl.Float64,
    "funding_rate": pl.Float64,
    "open_interest_change_24h": pl.Float64,
    "open_interest_usd_change_24h": pl.Float64,
    "taker_buy_ratio": pl.Float64,
    "taker_volume_imbalance": pl.Float64,
    "long_short_account_ratio": pl.Float64,
    "top_trader_long_short_account_ratio": pl.Float64,
    "top_trader_long_short_position_ratio": pl.Float64,
    "positive_components": pl.String,
    "negative_filters": pl.String,
    "missing_evidence": pl.String,
    "next_fetch_action": pl.String,
    "data_quality_warning": pl.String,
}

NEXT_FETCH_ACTION_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.String,
    "priority": pl.Int64,
    "source": pl.String,
    "phase": pl.String,
    "reason": pl.String,
    "expected_confidence_delta": pl.String,
    "requires_secret": pl.Boolean,
}

BACKTEST_EVENT_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Int64,
    "symbol": pl.String,
    "alert_level": pl.String,
    "score_total": pl.Int64,
    "entry_close": pl.Float64,
    "return_3h": pl.Float64,
    "return_7h": pl.Float64,
    "return_24h": pl.Float64,
    "return_3d": pl.Float64,
    "return_7d": pl.Float64,
    "max_drawdown_24h": pl.Float64,
    "max_drawdown_7d": pl.Float64,
    "hit_take_profit_5pct_7d": pl.Boolean,
    "hit_stop_loss_5pct_7d": pl.Boolean,
}


def empty_feature_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=FEATURE_SCHEMA)


def empty_score_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=SCORE_SCHEMA)


def empty_backtest_event_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=BACKTEST_EVENT_SCHEMA)


def empty_discovery_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=DISCOVERY_SCHEMA)


def empty_broad_market_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=BROAD_MARKET_SCHEMA)


def empty_broad_protocol_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=BROAD_PROTOCOL_SCHEMA)


def empty_broad_news_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=BROAD_NEWS_SCHEMA)


def empty_broad_candidate_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=BROAD_CANDIDATE_SCHEMA)


def empty_source_manifest_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=SOURCE_MANIFEST_SCHEMA)


def empty_candidate_summary_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=CANDIDATE_SUMMARY_SCHEMA)


def empty_candidate_detail_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=CANDIDATE_DETAIL_SCHEMA)


def empty_next_fetch_action_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=NEXT_FETCH_ACTION_SCHEMA)
