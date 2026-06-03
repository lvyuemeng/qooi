"""Pure feature engineering for accumulation-like scans."""

from __future__ import annotations

from functools import reduce

import polars as pl

from qooi.accumulation.schema import FEATURE_SCHEMA, empty_feature_frame

HOUR_MS = 3_600_000
MINUTE_MS = 60_000


def _hour_col(name: str = "timestamp") -> pl.Expr:
    return (pl.col(name) // HOUR_MS * HOUR_MS).alias("timestamp")


def compute_price_features(price_frame: pl.DataFrame, *, ma_hours: int = 200) -> pl.DataFrame:
    if price_frame.is_empty():
        return pl.DataFrame(
            schema={
                "timestamp": pl.Int64,
                "close": pl.Float64,
                "return_1h": pl.Float64,
                "return_24h": pl.Float64,
                "ma200": pl.Float64,
            }
        )
    return (
        price_frame.select([pl.col("timestamp").cast(pl.Int64), pl.col("close").cast(pl.Float64)])
        .sort("timestamp")
        .with_columns(
            [
                (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("return_1h"),
                (pl.col("close") / pl.col("close").shift(24) - 1.0).alias("return_24h"),
                pl.col("close").rolling_mean(window_size=ma_hours, min_samples=1).alias("ma200"),
            ]
        )
    )


def compute_price_features_batch(price_frame: pl.DataFrame, *, ma_hours: int = 200) -> pl.DataFrame:
    if price_frame.is_empty():
        return pl.DataFrame(
            schema={
                "timestamp": pl.Int64,
                "symbol": pl.String,
                "close": pl.Float64,
                "return_1h": pl.Float64,
                "return_24h": pl.Float64,
                "ma200": pl.Float64,
            }
        )
    if "symbol" not in price_frame.columns:
        raise ValueError("batch price features require a symbol column")
    return (
        price_frame.select(
            [
                pl.col("symbol").cast(pl.String),
                pl.col("timestamp").cast(pl.Int64),
                pl.col("close").cast(pl.Float64),
            ]
        )
        .sort(["symbol", "timestamp"])
        .with_columns(
            [
                (pl.col("close") / pl.col("close").shift(1).over("symbol") - 1.0).alias(
                    "return_1h"
                ),
                (pl.col("close") / pl.col("close").shift(24).over("symbol") - 1.0).alias(
                    "return_24h"
                ),
                pl.col("close")
                .rolling_mean(window_size=ma_hours, min_samples=1)
                .over("symbol")
                .alias("ma200"),
            ]
        )
    )


def compute_structure_features(price_frame: pl.DataFrame) -> pl.DataFrame:
    schema = {
        "timestamp": pl.Int64,
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
    }
    if price_frame.is_empty():
        return pl.DataFrame(schema=schema)
    frame = price_frame.select(
        [pl.col("timestamp").cast(pl.Int64), pl.col("close").cast(pl.Float64)]
    ).sort("timestamp")
    return (
        frame.with_columns(
            [
                pl.col("close")
                .pct_change()
                .rolling_std(window_size=24, min_samples=2)
                .alias("_vol_24h"),
                pl.col("close").rolling_min(window_size=168, min_samples=2).alias("_range_low"),
                pl.col("close").rolling_max(window_size=168, min_samples=2).alias("_range_high"),
            ]
        )
        .with_columns(
            [
                pl.when((pl.col("_range_high") - pl.col("_range_low")) > 0.0)
                .then(pl.col("_range_low"))
                .otherwise(None)
                .alias("range_low_px"),
                pl.when((pl.col("_range_high") - pl.col("_range_low")) > 0.0)
                .then(pl.col("_range_high"))
                .otherwise(None)
                .alias("range_high_px"),
                pl.col("_vol_24h").rank().truediv(pl.len()).alias("volatility_compression_pctile"),
                pl.when((pl.col("_range_high") - pl.col("_range_low")) > 0.0)
                .then(
                    (pl.col("close") - pl.col("_range_low"))
                    / (pl.col("_range_high") - pl.col("_range_low"))
                )
                .otherwise(None)
                .alias("range_position_pct"),
                pl.when(pl.col("close") >= pl.col("_range_high"))
                .then(pl.lit("breakout"))
                .when(pl.col("close") <= pl.col("_range_low"))
                .then(pl.lit("range_low"))
                .otherwise(pl.lit("range"))
                .alias("structure_stage"),
            ]
        )
        .with_columns(
            [
                pl.when((pl.col("range_high_px") - pl.col("range_low_px")) > 0.0)
                .then((pl.col("range_high_px") - pl.col("range_low_px")) / pl.col("close"))
                .otherwise(None)
                .alias("range_width_pct"),
                pl.when(pl.col("range_high_px") > 0.0)
                .then((pl.col("range_high_px") / pl.col("close")) - 1.0)
                .otherwise(None)
                .alias("upside_to_range_high_pct"),
                pl.when(pl.col("range_low_px") > 0.0)
                .then((pl.col("close") / pl.col("range_low_px")) - 1.0)
                .otherwise(None)
                .alias("downside_to_range_low_pct"),
                pl.col("range_low_px").alias("structure_invalidation_px"),
                pl.col("range_high_px").alias("structure_target_px"),
            ]
        )
        .with_columns(
            pl.when(
                (pl.col("upside_to_range_high_pct") >= 0.0)
                & (pl.col("downside_to_range_low_pct") > 0.0)
            )
            .then(pl.col("upside_to_range_high_pct") / pl.col("downside_to_range_low_pct"))
            .otherwise(None)
            .alias("range_reward_risk")
        )
        .select(schema.keys())
    )


def compute_funding_features(funding_frame: pl.DataFrame) -> pl.DataFrame:
    schema = {"timestamp": pl.Int64, "funding_rate": pl.Float64, "funding_zscore": pl.Float64}
    if funding_frame.is_empty() or "funding_rate" not in funding_frame.columns:
        return pl.DataFrame(schema=schema)
    return (
        funding_frame.select(
            [pl.col("timestamp").cast(pl.Int64), pl.col("funding_rate").cast(pl.Float64)]
        )
        .sort("timestamp")
        .with_columns(
            [
                pl.col("funding_rate").rolling_mean(window_size=21, min_samples=2).alias("_mean"),
                pl.col("funding_rate").rolling_std(window_size=21, min_samples=2).alias("_std"),
            ]
        )
        .with_columns(
            pl.when(pl.col("_std") > 0.0)
            .then((pl.col("funding_rate") - pl.col("_mean")) / pl.col("_std"))
            .otherwise(None)
            .alias("funding_zscore")
        )
        .select(schema.keys())
    )


def compute_open_interest_features(open_interest_frame: pl.DataFrame) -> pl.DataFrame:
    schema = {
        "timestamp": pl.Int64,
        "open_interest_usd": pl.Float64,
        "open_interest_change_24h": pl.Float64,
        "open_interest_usd_change_24h": pl.Float64,
    }
    if open_interest_frame.is_empty() or "open_interest" not in open_interest_frame.columns:
        return pl.DataFrame(schema=schema)
    cols = [pl.col("timestamp").cast(pl.Int64), pl.col("open_interest").cast(pl.Float64)]
    if "open_interest_usd" in open_interest_frame.columns:
        cols.append(pl.col("open_interest_usd").cast(pl.Float64))
    else:
        cols.append(pl.lit(None).cast(pl.Float64).alias("open_interest_usd"))
    frame = (
        open_interest_frame.select(
            cols
        )
        .sort("timestamp")
        .group_by("timestamp")
        .last()
    )
    lag = frame.select(
        [
            (pl.col("timestamp") + 24 * HOUR_MS).alias("timestamp"),
            pl.col("open_interest").alias("_open_interest_24h"),
            pl.col("open_interest_usd").alias("_open_interest_usd_24h"),
        ]
    )
    return (
        frame.join(lag, on="timestamp", how="left")
        .with_columns(
            [
                pl.when(pl.col("_open_interest_24h") > 0.0)
                .then(pl.col("open_interest") / pl.col("_open_interest_24h") - 1.0)
                .otherwise(None)
                .alias("open_interest_change_24h"),
                pl.when(pl.col("_open_interest_usd_24h") > 0.0)
                .then(pl.col("open_interest_usd") / pl.col("_open_interest_usd_24h") - 1.0)
                .otherwise(None)
                .alias("open_interest_usd_change_24h"),
            ]
        )
        .select(schema.keys())
        .sort("timestamp")
    )


def compute_taker_volume_features(taker_frame: pl.DataFrame) -> pl.DataFrame:
    schema = {
        "timestamp": pl.Int64,
        "taker_buy_volume": pl.Float64,
        "taker_sell_volume": pl.Float64,
        "taker_buy_ratio": pl.Float64,
        "taker_volume_total": pl.Float64,
        "taker_volume_imbalance": pl.Float64,
    }
    if taker_frame.is_empty() or not {
        "taker_buy_volume",
        "taker_sell_volume",
    }.issubset(taker_frame.columns):
        return pl.DataFrame(schema=schema)
    return (
        taker_frame.with_columns(_hour_col())
        .group_by("timestamp")
        .agg(
            [
                pl.col("taker_buy_volume").cast(pl.Float64).sum().alias("taker_buy_volume"),
                pl.col("taker_sell_volume").cast(pl.Float64).sum().alias("taker_sell_volume"),
            ]
        )
        .with_columns(
            (pl.col("taker_buy_volume") + pl.col("taker_sell_volume")).alias(
                "taker_volume_total"
            )
        )
        .with_columns(
            [
                pl.when(pl.col("taker_volume_total") > 0.0)
                .then(pl.col("taker_buy_volume") / pl.col("taker_volume_total"))
                .otherwise(None)
                .alias("taker_buy_ratio"),
                pl.when(pl.col("taker_volume_total") > 0.0)
                .then(
                    (pl.col("taker_buy_volume") - pl.col("taker_sell_volume"))
                    / pl.col("taker_volume_total")
                )
                .otherwise(None)
                .alias("taker_volume_imbalance"),
            ]
        )
        .select(schema.keys())
        .sort("timestamp")
    )


REALTIME_TRADE_FEATURE_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Int64,
    "window_seconds": pl.Int64,
    "last_price": pl.Float64,
    "aggressive_buy_ratio": pl.Float64,
    "aggressive_imbalance": pl.Float64,
    "large_trade_buy_ratio": pl.Float64,
    "large_trade_spike_z": pl.Float64,
    "vwap": pl.Float64,
    "price_vs_vwap_pct": pl.Float64,
    "vwap_slope": pl.Float64,
    "positive_trade_windows": pl.Int64,
    "negative_trade_windows": pl.Int64,
}


REALTIME_BOOK_FEATURE_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Int64,
    "window_seconds": pl.Int64,
    "depth_imbalance_5m_mean": pl.Float64,
    "depth_imbalance_5m_p25": pl.Float64,
    "depth_imbalance_15m_mean": pl.Float64,
    "spread_bps_5m_median": pl.Float64,
    "bid_depth_rebuild_15m": pl.Float64,
    "ask_pressure_15m_mean": pl.Float64,
    "book_stability_score_15m": pl.Float64,
    "bid_support_ratio_10_25": pl.Float64,
    "ask_wall_ratio_25_50": pl.Float64,
    "positive_book_windows": pl.Int64,
    "negative_book_windows": pl.Int64,
}


REALTIME_CONFIRMATION_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Int64,
    "window_seconds": pl.Int64,
    "realtime_confirmation_score": pl.Float64,
    "realtime_confirmation_state": pl.String,
    "structure_overlay": pl.String,
    "positive_reasons": pl.String,
    "risk_reasons": pl.String,
    "data_quality_warning": pl.String,
}


FUNDS_FLOW_FEATURE_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Int64,
    "net_exchange_flow_30d": pl.Float64,
    "net_exchange_flow_to_supply_30d": pl.Float64,
    "exchange_outflow_days_30d": pl.Int64,
    "exchange_inflow_days_30d": pl.Int64,
    "exchange_outflow_trend_30d": pl.Float64,
    "exchange_outflow_dominant": pl.Boolean,
    "exchange_inflow_dominant": pl.Boolean,
    "quiet_outflow_absorption": pl.Boolean,
}

FUNDS_ORDERBOOK_FEATURE_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Int64,
    "bid_ask_depth_ratio_10_1h_mean": pl.Float64,
    "bid_ask_depth_ratio_10_down_slope": pl.Float64,
    "bid_support_rising_on_down_move": pl.Boolean,
    "depth_rebuild_after_sell_1h": pl.Float64,
}

FUNDS_VOLUME_FEATURE_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Int64,
    "large_trade_buy_sell_ratio_24h": pl.Float64,
    "up_day_volume_to_down_day_volume_30d": pl.Float64,
    "quiet_up_volume_bias_30d": pl.Boolean,
    "last_hour_volume_share_30d": pl.Float64,
    "last_hour_accumulation_days_30d": pl.Int64,
}

FUNDS_MESSAGE_FEATURE_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Int64,
    "message_quiet_30d": pl.Boolean,
    "emotion_overheat_30d": pl.Boolean,
    "fundamental_context_present": pl.Boolean,
}


def compute_funds_flow_features(
    hourly_flow_frame: pl.DataFrame,
    price_frame: pl.DataFrame,
    *,
    total_supply: float | None = None,
    window_hours: int = 720,
) -> pl.DataFrame:
    if hourly_flow_frame.is_empty() or "timestamp" not in hourly_flow_frame.columns:
        return pl.DataFrame(schema=FUNDS_FLOW_FEATURE_SCHEMA)
    frame = hourly_flow_frame.with_columns(pl.col("timestamp").cast(pl.Int64, strict=False))
    if "net_exchange_flow" not in frame.columns:
        if not {"inflow", "outflow"}.issubset(frame.columns):
            return pl.DataFrame(schema=FUNDS_FLOW_FEATURE_SCHEMA)
        frame = frame.with_columns(
            (pl.col("inflow").fill_null(0.0) - pl.col("outflow").fill_null(0.0)).alias(
                "net_exchange_flow"
            )
        )
    frame = frame.select(
        [pl.col("timestamp"), pl.col("net_exchange_flow").cast(pl.Float64, strict=False)]
    ).sort("timestamp")
    returns = (
        price_frame.select(
            [
                pl.col("timestamp").cast(pl.Int64, strict=False),
                pl.col("close").cast(pl.Float64, strict=False),
            ]
        )
        .sort("timestamp")
        .with_columns((pl.col("close") / pl.col("close").shift(24) - 1.0).alias("return_24h"))
        if not price_frame.is_empty() and {"timestamp", "close"}.issubset(price_frame.columns)
        else pl.DataFrame(schema={"timestamp": pl.Int64, "return_24h": pl.Float64})
    )
    out = (
        frame.with_columns(
            [
                pl.col("net_exchange_flow")
                .rolling_sum(window_size=window_hours, min_samples=1)
                .alias("net_exchange_flow_30d"),
                (pl.col("net_exchange_flow") < 0.0)
                .cast(pl.Int64)
                .rolling_sum(window_size=window_hours, min_samples=1)
                .alias("exchange_outflow_days_30d"),
                (pl.col("net_exchange_flow") > 0.0)
                .cast(pl.Int64)
                .rolling_sum(window_size=window_hours, min_samples=1)
                .alias("exchange_inflow_days_30d"),
                (
                    pl.col("net_exchange_flow")
                    - pl.col("net_exchange_flow").shift(min(window_hours, 24))
                ).alias("exchange_outflow_trend_30d"),
            ]
        )
        .join_asof(returns, on="timestamp", strategy="backward")
        .with_columns(
            [
                pl.when(total_supply is not None and total_supply > 0.0)
                .then(pl.col("net_exchange_flow_30d") / float(total_supply))
                .otherwise(None)
                .alias("net_exchange_flow_to_supply_30d"),
                (pl.col("net_exchange_flow_30d") < 0.0).alias("exchange_outflow_dominant"),
                (pl.col("net_exchange_flow_30d") > 0.0).alias("exchange_inflow_dominant"),
                (
                    (pl.col("net_exchange_flow_30d") < 0.0)
                    & pl.col("return_24h").abs().fill_null(0.0).lt(0.10)
                ).alias("quiet_outflow_absorption"),
            ]
        )
    )
    return _coerce_schema(out, FUNDS_FLOW_FEATURE_SCHEMA)


def compute_funds_orderbook_features(
    book_frame: pl.DataFrame,
    price_frame: pl.DataFrame,
    *,
    window_minutes: int = 60,
) -> pl.DataFrame:
    if book_frame.is_empty() or "timestamp" not in book_frame.columns:
        return pl.DataFrame(schema=FUNDS_ORDERBOOK_FEATURE_SCHEMA)
    frame = _normalize_realtime_books(book_frame)
    if frame.is_empty():
        return pl.DataFrame(schema=FUNDS_ORDERBOOK_FEATURE_SCHEMA)
    if not price_frame.is_empty() and {"timestamp", "close"}.issubset(price_frame.columns):
        prices = price_frame.select(
            [pl.col("timestamp").cast(pl.Int64), pl.col("close").cast(pl.Float64)]
        ).sort("timestamp")
    else:
        prices = pl.DataFrame(schema={"timestamp": pl.Int64, "close": pl.Float64})
    rows = []
    for timestamp in _minute_endpoints(frame):
        window = _window(frame, timestamp, window_minutes * MINUTE_MS)
        if window.is_empty():
            continue
        ratio = _depth_ratio(window, "bid_depth_bps_10", "ask_depth_bps_10")
        price_window = _window(prices, timestamp, window_minutes * MINUTE_MS)
        price_down_flat = price_window.height < 2 or float(price_window["close"][-1]) <= float(
            price_window["close"][0]
        ) * 1.002
        rows.append(
            {
                "timestamp": timestamp,
                "bid_ask_depth_ratio_10_1h_mean": ratio,
                "bid_ask_depth_ratio_10_down_slope": _ratio_slope(
                    window, "bid_depth_bps_10", "ask_depth_bps_10"
                ),
                "bid_support_rising_on_down_move": bool(
                    price_down_flat
                    and (_ratio_slope(window, "bid_depth_bps_10", "ask_depth_bps_10") or 0.0) > 0.0
                ),
                "depth_rebuild_after_sell_1h": _depth_rebuild(window),
            }
        )
    return _coerce_schema(pl.DataFrame(rows), FUNDS_ORDERBOOK_FEATURE_SCHEMA)


def compute_funds_volume_features(
    bars: pl.DataFrame,
    trade_frame: pl.DataFrame,
    *,
    large_trade_usd: float = 10_000.0,
    window_hours: int = 720,
) -> pl.DataFrame:
    if bars.is_empty() or not {"timestamp", "close", "vol"}.issubset(bars.columns):
        return pl.DataFrame(schema=FUNDS_VOLUME_FEATURE_SCHEMA)
    bar_frame = bars.select(
        [
            (pl.col("timestamp").cast(pl.Int64) // (24 * HOUR_MS) * 24 * HOUR_MS).alias("day"),
            pl.col("timestamp").cast(pl.Int64),
            pl.col("close").cast(pl.Float64),
            pl.col("vol").cast(pl.Float64),
        ]
    ).sort("timestamp")
    daily = (
        bar_frame.group_by("day")
        .agg(
            [
                pl.col("timestamp").last().alias("timestamp"),
                pl.col("close").first().alias("open_close"),
                pl.col("close").last().alias("close"),
                pl.col("vol").sum().alias("volume"),
                pl.col("vol").last().alias("last_hour_volume"),
            ]
        )
        .sort("timestamp")
        .with_columns((pl.col("close") >= pl.col("open_close")).alias("_up_day"))
    )
    trade_ratio = _large_trade_buy_sell_ratio(trade_frame, large_trade_usd=large_trade_usd)
    out = daily.with_columns(
        [
            pl.when((~pl.col("_up_day")).any())
            .then(
                pl.col("volume").filter(pl.col("_up_day")).sum()
                / pl.col("volume").filter(~pl.col("_up_day")).sum()
            )
            .otherwise(None)
            .alias("up_day_volume_to_down_day_volume_30d"),
            pl.when(pl.col("volume") > 0.0)
            .then(pl.col("last_hour_volume") / pl.col("volume"))
            .otherwise(None)
            .alias("last_hour_volume_share_30d"),
        ]
    ).with_columns(
        [
            (pl.col("up_day_volume_to_down_day_volume_30d") >= 1.3).alias(
                "quiet_up_volume_bias_30d"
            ),
            (pl.col("last_hour_volume_share_30d") >= 0.10)
            .cast(pl.Int64)
            .rolling_sum(window_size=max(1, window_hours // 24), min_samples=1)
            .alias("last_hour_accumulation_days_30d"),
            pl.lit(trade_ratio).cast(pl.Float64).alias("large_trade_buy_sell_ratio_24h"),
        ]
    )
    return _coerce_schema(out, FUNDS_VOLUME_FEATURE_SCHEMA)


def compute_funds_message_features(
    message_features: pl.DataFrame,
    *,
    quiet_growth_max: float = 1.2,
    overheat_growth_min: float = 3.0,
) -> pl.DataFrame:
    if message_features.is_empty() or "timestamp" not in message_features.columns:
        return pl.DataFrame(schema=FUNDS_MESSAGE_FEATURE_SCHEMA)
    frame = message_features.select(
        [
            pl.col("timestamp").cast(pl.Int64),
            pl.col("mention_growth").cast(pl.Float64, strict=False)
            if "mention_growth" in message_features.columns
            else pl.lit(None).cast(pl.Float64).alias("mention_growth"),
            pl.col("fundamental_news_ratio").cast(pl.Float64, strict=False)
            if "fundamental_news_ratio" in message_features.columns
            else pl.lit(None).cast(pl.Float64).alias("fundamental_news_ratio"),
            pl.col("emotion_news_ratio").cast(pl.Float64, strict=False)
            if "emotion_news_ratio" in message_features.columns
            else pl.lit(None).cast(pl.Float64).alias("emotion_news_ratio"),
        ]
    )
    out = frame.with_columns(
        [
            (pl.col("mention_growth") <= quiet_growth_max)
            .fill_null(False)
            .alias("message_quiet_30d"),
            (
                (pl.col("mention_growth") >= overheat_growth_min)
                & (pl.col("emotion_news_ratio") > pl.col("fundamental_news_ratio"))
            )
            .fill_null(False)
            .alias("emotion_overheat_30d"),
            (pl.col("fundamental_news_ratio") > 0.0)
            .fill_null(False)
            .alias("fundamental_context_present"),
        ]
    )
    return _coerce_schema(out, FUNDS_MESSAGE_FEATURE_SCHEMA)


def compute_realtime_trade_features(
    trade_frame: pl.DataFrame,
    *,
    window_minutes: int = 15,
    subwindow_minutes: int = 5,
    large_trade_usd: float = 50_000.0,
) -> pl.DataFrame:
    """Compute rolling trade/VWAP confirmation features from normalized trades."""
    if trade_frame.is_empty() or not {"timestamp", "price", "side"}.issubset(
        trade_frame.columns
    ):
        return pl.DataFrame(schema=REALTIME_TRADE_FEATURE_SCHEMA)
    frame = _normalize_realtime_trades(trade_frame)
    if frame.is_empty():
        return pl.DataFrame(schema=REALTIME_TRADE_FEATURE_SCHEMA)
    rows = []
    for timestamp in _minute_endpoints(frame):
        window = _window(frame, timestamp, window_minutes * MINUTE_MS)
        previous = _window(
            frame,
            timestamp - window_minutes * MINUTE_MS,
            window_minutes * MINUTE_MS,
        )
        if window.is_empty():
            continue
        buy_notional = _sum_side(window, "buy")
        sell_notional = _sum_side(window, "sell")
        total_notional = buy_notional + sell_notional
        large = window.filter(pl.col("notional_usd") >= large_trade_usd)
        large_total = _sum_notional(large)
        large_buy_ratio = (
            _sum_side(large, "buy") / large_total if large_total > 0.0 else None
        )
        large_spike_z = _large_trade_spike_z(window, previous, large_trade_usd)
        vwap = _vwap(window)
        last_price = float(window.sort("timestamp").tail(1)["price"][0])
        subwindows = _trade_subwindow_states(window, subwindow_minutes=subwindow_minutes)
        rows.append(
            {
                "timestamp": timestamp,
                "window_seconds": window_minutes * 60,
                "last_price": last_price,
                "aggressive_buy_ratio": buy_notional / total_notional
                if total_notional > 0.0
                else None,
                "aggressive_imbalance": (buy_notional - sell_notional) / total_notional
                if total_notional > 0.0
                else None,
                "large_trade_buy_ratio": large_buy_ratio,
                "large_trade_spike_z": large_spike_z,
                "vwap": vwap,
                "price_vs_vwap_pct": last_price / vwap - 1.0 if vwap and vwap > 0.0 else None,
                "vwap_slope": _vwap_slope(window, previous),
                "positive_trade_windows": subwindows[0],
                "negative_trade_windows": subwindows[1],
            }
        )
    return _coerce_schema(pl.DataFrame(rows), REALTIME_TRADE_FEATURE_SCHEMA)


def compute_realtime_orderbook_features(
    book_frame: pl.DataFrame,
    *,
    window_minutes: int = 15,
    subwindow_minutes: int = 5,
) -> pl.DataFrame:
    """Compute rolling order-book distribution features without trusting single snapshots."""
    if book_frame.is_empty() or "timestamp" not in book_frame.columns:
        return pl.DataFrame(schema=REALTIME_BOOK_FEATURE_SCHEMA)
    frame = _normalize_realtime_books(book_frame)
    if frame.is_empty():
        return pl.DataFrame(schema=REALTIME_BOOK_FEATURE_SCHEMA)
    rows = []
    for timestamp in _minute_endpoints(frame):
        window_15m = _window(frame, timestamp, window_minutes * MINUTE_MS)
        window_5m = _window(frame, timestamp, subwindow_minutes * MINUTE_MS)
        if window_15m.is_empty() or window_5m.is_empty():
            continue
        positive_windows, negative_windows = _book_subwindow_states(
            window_15m, subwindow_minutes=subwindow_minutes
        )
        rows.append(
            {
                "timestamp": timestamp,
                "window_seconds": window_minutes * 60,
                "depth_imbalance_5m_mean": _mean(window_5m, "depth_imbalance_25"),
                "depth_imbalance_5m_p25": _quantile(window_5m, "depth_imbalance_25", 0.25),
                "depth_imbalance_15m_mean": _mean(window_15m, "depth_imbalance_25"),
                "spread_bps_5m_median": _quantile(window_5m, "spread_bps", 0.5),
                "bid_depth_rebuild_15m": _depth_rebuild(window_15m),
                "ask_pressure_15m_mean": _mean(window_15m, "ask_pressure_25"),
                "book_stability_score_15m": _book_stability(window_15m),
                "bid_support_ratio_10_25": _mean(window_15m, "bid_support_ratio_10_25"),
                "ask_wall_ratio_25_50": _mean(window_15m, "ask_wall_ratio_25_50"),
                "positive_book_windows": positive_windows,
                "negative_book_windows": negative_windows,
            }
        )
    return _coerce_schema(pl.DataFrame(rows), REALTIME_BOOK_FEATURE_SCHEMA)


def compute_realtime_confirmation_features(
    trade_features: pl.DataFrame,
    book_features: pl.DataFrame,
    *,
    structure_stage: str = "",
) -> pl.DataFrame:
    """Overlay realtime acceptance evidence on an existing structure stage."""
    if trade_features.is_empty() and book_features.is_empty():
        return pl.DataFrame(schema=REALTIME_CONFIRMATION_SCHEMA)
    frame = _join_realtime_feature_frames(trade_features, book_features)
    rows = [_confirmation_row(row, structure_stage=structure_stage) for row in frame.to_dicts()]
    return _coerce_schema(pl.DataFrame(rows), REALTIME_CONFIRMATION_SCHEMA)


def _normalize_realtime_trades(frame: pl.DataFrame) -> pl.DataFrame:
    cols = [
        pl.col("timestamp").cast(pl.Int64, strict=False),
        pl.col("price").cast(pl.Float64, strict=False),
        pl.col("side").cast(pl.String).str.to_lowercase(),
    ]
    if "notional_usd" in frame.columns:
        cols.append(pl.col("notional_usd").cast(pl.Float64, strict=False))
    elif "size" in frame.columns:
        cols.append(
            (pl.col("price") * pl.col("size"))
            .cast(pl.Float64, strict=False)
            .alias("notional_usd")
        )
    else:
        cols.append(pl.lit(None).cast(pl.Float64).alias("notional_usd"))
    return (
        frame.select(cols)
        .filter(
            pl.col("timestamp").is_not_null()
            & pl.col("price").is_not_null()
            & pl.col("notional_usd").is_not_null()
            & pl.col("side").is_in(["buy", "sell"])
        )
        .sort("timestamp")
    )


def _normalize_realtime_books(frame: pl.DataFrame) -> pl.DataFrame:
    base = frame.with_columns(pl.col("timestamp").cast(pl.Int64, strict=False)).filter(
        pl.col("timestamp").is_not_null()
    )
    base = _ensure_depth_column(base, "bid_depth_bps_10", ("ob_bid_vol_5",))
    base = _ensure_depth_column(base, "ask_depth_bps_10", ("ob_ask_vol_5",))
    base = _ensure_depth_column(base, "bid_depth_bps_25", ("ob_bid_vol_25", "ob_bid_vol"))
    base = _ensure_depth_column(base, "ask_depth_bps_25", ("ob_ask_vol_25", "ob_ask_vol"))
    base = _ensure_depth_column(base, "bid_depth_bps_50", ("bid_depth_bps_25",))
    base = _ensure_depth_column(base, "ask_depth_bps_50", ("ask_depth_bps_25",))
    if "spread_bps" not in base.columns:
        if {"ob_bid_price", "ob_ask_price"}.issubset(base.columns):
            mid = (pl.col("ob_bid_price") + pl.col("ob_ask_price")) / 2.0
            base = base.with_columns(
                pl.when(mid > 0.0)
                .then((pl.col("ob_ask_price") - pl.col("ob_bid_price")) / mid * 10_000.0)
                .otherwise(None)
                .alias("spread_bps")
            )
        else:
            base = base.with_columns(pl.lit(None).cast(pl.Float64).alias("spread_bps"))
    else:
        base = base.with_columns(pl.col("spread_bps").cast(pl.Float64, strict=False))
    bid25 = pl.col("bid_depth_bps_25")
    ask25 = pl.col("ask_depth_bps_25")
    bid50_outer = (pl.col("bid_depth_bps_50") - pl.col("bid_depth_bps_25")).clip(0.0)
    ask50_outer = (pl.col("ask_depth_bps_50") - pl.col("ask_depth_bps_25")).clip(0.0)
    return base.with_columns(
        [
            pl.when((bid25 + ask25) > 0.0)
            .then((bid25 - ask25) / (bid25 + ask25))
            .otherwise(None)
            .alias("depth_imbalance_25"),
            pl.when((bid25 + ask25) > 0.0)
            .then(ask25 / (bid25 + ask25))
            .otherwise(None)
            .alias("ask_pressure_25"),
            pl.when(pl.col("bid_depth_bps_50") > 0.0)
            .then(
                (pl.col("bid_depth_bps_25") - pl.col("bid_depth_bps_10")).clip(0.0)
                / pl.col("bid_depth_bps_50")
            )
            .otherwise(None)
            .alias("bid_support_ratio_10_25"),
            pl.when(pl.col("ask_depth_bps_50") > 0.0)
            .then(ask50_outer / pl.col("ask_depth_bps_50"))
            .otherwise(None)
            .alias("ask_wall_ratio_25_50"),
            bid50_outer.alias("_bid_depth_outer_25_50"),
        ]
    ).sort("timestamp")


def _ensure_depth_column(
    frame: pl.DataFrame, column: str, fallbacks: tuple[str, ...]
) -> pl.DataFrame:
    if column in frame.columns:
        return frame.with_columns(pl.col(column).cast(pl.Float64, strict=False))
    for fallback in fallbacks:
        if fallback in frame.columns:
            return frame.with_columns(pl.col(fallback).cast(pl.Float64, strict=False).alias(column))
    return frame.with_columns(pl.lit(None).cast(pl.Float64).alias(column))


def _minute_endpoints(frame: pl.DataFrame) -> list[int]:
    return (
        frame.with_columns((pl.col("timestamp") // MINUTE_MS * MINUTE_MS).alias("_minute"))
        .get_column("_minute")
        .unique()
        .sort()
        .to_list()
    )


def _window(frame: pl.DataFrame, end_timestamp: int, window_ms: int) -> pl.DataFrame:
    return frame.filter(
        (pl.col("timestamp") > end_timestamp - window_ms) & (pl.col("timestamp") <= end_timestamp)
    )


def _sum_side(frame: pl.DataFrame, side: str) -> float:
    return _sum_notional(frame.filter(pl.col("side") == side))


def _sum_notional(frame: pl.DataFrame) -> float:
    if frame.is_empty() or "notional_usd" not in frame.columns:
        return 0.0
    value = frame.get_column("notional_usd").sum()
    return float(value or 0.0)


def _vwap(frame: pl.DataFrame) -> float | None:
    total = _sum_notional(frame)
    if total <= 0.0:
        return None
    value = (frame.get_column("price") * frame.get_column("notional_usd")).sum() / total
    return float(value)


def _vwap_slope(current: pl.DataFrame, previous: pl.DataFrame) -> float | None:
    current_vwap = _vwap(current)
    previous_vwap = _vwap(previous)
    if current_vwap is None or previous_vwap is None or previous_vwap <= 0.0:
        return None
    return current_vwap / previous_vwap - 1.0


def _large_trade_spike_z(
    current: pl.DataFrame, previous: pl.DataFrame, large_trade_usd: float
) -> float | None:
    current_large = _sum_notional(current.filter(pl.col("notional_usd") >= large_trade_usd))
    previous_large = previous.filter(pl.col("notional_usd") >= large_trade_usd)
    if previous_large.height < 2:
        return None
    values = previous_large.get_column("notional_usd")
    mean = values.mean()
    std = values.std()
    if mean is None or std is None or std <= 0.0:
        return None
    return (current_large - float(mean)) / float(std)


def _trade_subwindow_states(
    frame: pl.DataFrame, *, subwindow_minutes: int
) -> tuple[int, int]:
    positive = 0
    negative = 0
    for subwindow in _subwindows(frame, subwindow_minutes=subwindow_minutes):
        total = _sum_notional(subwindow)
        if total <= 0.0:
            continue
        buy_ratio = _sum_side(subwindow, "buy") / total
        vwap = _vwap(subwindow)
        last_price = float(subwindow.sort("timestamp").tail(1)["price"][0])
        if buy_ratio >= 0.60 and vwap is not None and last_price >= vwap:
            positive += 1
        if buy_ratio < 0.45 or (vwap is not None and last_price < vwap):
            negative += 1
    return positive, negative


def _book_subwindow_states(
    frame: pl.DataFrame, *, subwindow_minutes: int
) -> tuple[int, int]:
    positive = 0
    negative = 0
    for subwindow in _subwindows(frame, subwindow_minutes=subwindow_minutes):
        imbalance = _mean(subwindow, "depth_imbalance_25")
        spread = _quantile(subwindow, "spread_bps", 0.5)
        stability = _book_stability(subwindow)
        if imbalance is not None and imbalance >= 0.0 and stability >= 0.5:
            positive += 1
        if (imbalance is not None and imbalance <= -0.20) or (spread is not None and spread > 75.0):
            negative += 1
    return positive, negative


def _subwindows(frame: pl.DataFrame, *, subwindow_minutes: int) -> list[pl.DataFrame]:
    if frame.is_empty():
        return []
    min_ts = int(frame["timestamp"].min())
    max_ts = int(frame["timestamp"].max())
    size = subwindow_minutes * MINUTE_MS
    windows = []
    start = min_ts
    while start <= max_ts:
        end = start + size
        subwindow = frame.filter((pl.col("timestamp") >= start) & (pl.col("timestamp") < end))
        if not subwindow.is_empty():
            windows.append(subwindow)
        start = end
    return windows


def _mean(frame: pl.DataFrame, column: str) -> float | None:
    if frame.is_empty() or column not in frame.columns:
        return None
    value = frame.get_column(column).drop_nulls().mean()
    return None if value is None else float(value)


def _quantile(frame: pl.DataFrame, column: str, quantile: float) -> float | None:
    if frame.is_empty() or column not in frame.columns:
        return None
    value = frame.get_column(column).drop_nulls().quantile(quantile)
    return None if value is None else float(value)


def _depth_rebuild(frame: pl.DataFrame) -> float | None:
    if frame.height < 2 or "bid_depth_bps_25" not in frame.columns:
        return None
    ordered = frame.sort("timestamp")
    first = ordered.head(max(1, ordered.height // 3)).get_column("bid_depth_bps_25").mean()
    last = ordered.tail(max(1, ordered.height // 3)).get_column("bid_depth_bps_25").mean()
    if first is None or first <= 0.0 or last is None:
        return None
    return float(last / first)


def _depth_ratio(frame: pl.DataFrame, bid: str, ask: str) -> float | None:
    if frame.is_empty() or bid not in frame.columns or ask not in frame.columns:
        return None
    bid_mean = frame.get_column(bid).drop_nulls().mean()
    ask_mean = frame.get_column(ask).drop_nulls().mean()
    if bid_mean is None or ask_mean is None or ask_mean <= 0.0:
        return None
    return float(bid_mean) / float(ask_mean)


def _ratio_slope(frame: pl.DataFrame, bid: str, ask: str) -> float | None:
    if frame.height < 2 or bid not in frame.columns or ask not in frame.columns:
        return None
    ordered = frame.sort("timestamp")
    first_bid = ordered.get_column(bid)[0]
    first_ask = ordered.get_column(ask)[0]
    last_bid = ordered.get_column(bid)[-1]
    last_ask = ordered.get_column(ask)[-1]
    if first_bid is None or first_ask is None or last_bid is None or last_ask is None:
        return None
    if float(first_ask) <= 0.0 or float(last_ask) <= 0.0:
        return None
    return float(last_bid) / float(last_ask) - float(first_bid) / float(first_ask)


def _large_trade_buy_sell_ratio(
    trade_frame: pl.DataFrame, *, large_trade_usd: float
) -> float | None:
    if trade_frame.is_empty() or not {"side", "notional_usd"}.issubset(trade_frame.columns):
        return None
    large = trade_frame.with_columns(
        [
            pl.col("side").cast(pl.String).str.to_lowercase(),
            pl.col("notional_usd").cast(pl.Float64, strict=False),
        ]
    ).filter(pl.col("notional_usd") >= large_trade_usd)
    buy = _sum_side(large, "buy")
    sell = _sum_side(large, "sell")
    if sell <= 0.0:
        return 99.0 if buy > 0.0 else None
    return buy / sell


def _book_stability(frame: pl.DataFrame) -> float:
    if frame.is_empty():
        return 0.0
    spread = _quantile(frame, "spread_bps", 0.5)
    depth = _mean(frame, "bid_depth_bps_25")
    ask_depth = _mean(frame, "ask_depth_bps_25")
    if depth is None or ask_depth is None or depth + ask_depth <= 0.0:
        return 0.0
    spread_score = 1.0 if spread is None else max(0.0, min(1.0, 1.0 - spread / 100.0))
    return spread_score


def _join_realtime_feature_frames(
    trade_features: pl.DataFrame, book_features: pl.DataFrame
) -> pl.DataFrame:
    if trade_features.is_empty():
        return book_features.sort("timestamp")
    if book_features.is_empty():
        return trade_features.sort("timestamp")
    return trade_features.sort("timestamp").join_asof(
        book_features.sort("timestamp"), on="timestamp", strategy="backward"
    )


def _confirmation_row(row: dict[str, object], *, structure_stage: str) -> dict[str, object]:
    score = 0.0
    positives = []
    risks = []
    warnings = []
    if _row_float(row, "aggressive_buy_ratio") >= 0.65:
        score += 25.0
        positives.append("buyer_dominance")
    if _row_float(row, "large_trade_buy_ratio") >= 0.60:
        score += 15.0
        positives.append("large_buy_confirm")
    if _row_float(row, "price_vs_vwap_pct") >= 0.0 and _row_float(row, "vwap_slope") > 0.0:
        score += 15.0
        positives.append("vwap_acceptance")
    if _row_float(row, "depth_imbalance_15m_mean") >= 0.20:
        score += 10.0
        positives.append("depth_support")
    if _row_float(row, "bid_depth_rebuild_15m") >= 0.80:
        score += 10.0
        positives.append("bid_rebuild")
    persistent_windows = int(row.get("positive_trade_windows") or 0) + int(
        row.get("positive_book_windows") or 0
    )
    if persistent_windows >= 3:
        score += 10.0
        positives.append("persistent_acceptance")
    if _row_float(row, "aggressive_buy_ratio") < 0.45:
        score -= 15.0
        risks.append("sell_pressure")
    if _row_float(row, "depth_imbalance_15m_mean") <= -0.20:
        score -= 10.0
        risks.append("weak_depth")
    if _row_float(row, "price_vs_vwap_pct") < 0.0:
        score -= 15.0
        risks.append("below_vwap")
    if _row_float(row, "book_stability_score_15m") <= 0.0:
        score -= 20.0
        warnings.append("book_data_unstable")
    state = _realtime_state(score, risks, warnings)
    overlay = _structure_overlay(structure_stage, state)
    return {
        "timestamp": row.get("timestamp"),
        "window_seconds": row.get("window_seconds"),
        "realtime_confirmation_score": score,
        "realtime_confirmation_state": state,
        "structure_overlay": overlay,
        "positive_reasons": ";".join(positives),
        "risk_reasons": ";".join(risks),
        "data_quality_warning": ";".join(warnings),
    }


def _realtime_state(score: float, risks: list[str], warnings: list[str]) -> str:
    if warnings:
        return "data_unstable"
    if "below_vwap" in risks or "weak_depth" in risks:
        return "distribution_risk" if "sell_pressure" in risks else "chop_risk"
    if score >= 60.0:
        return "trend_confirming"
    if score >= 35.0:
        return "trend_building"
    if score >= 15.0:
        return "base_absorbing"
    return "chop_risk" if risks else "trend_building"


def _structure_overlay(structure_stage: str, realtime_state: str) -> str:
    favorable = {"stealth_base", "base_ready", "first_expansion", "controlled_lift"}
    late = {"pump_chop", "late_distribution_risk"}
    if realtime_state == "data_unstable":
        return "data_unstable"
    if structure_stage in favorable and realtime_state in {"trend_confirming", "trend_building"}:
        return "structure_confirmed"
    if structure_stage in favorable and realtime_state in {"chop_risk", "distribution_risk"}:
        return "structure_rejected"
    if structure_stage in late and realtime_state in {"chop_risk", "distribution_risk"}:
        return "structure_late_confirmed"
    return "structure_waiting"


def _row_float(row: dict[str, object], key: str) -> float:
    value = row.get(key)
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _coerce_schema(frame: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(schema=schema)
    for col, dtype in schema.items():
        if col not in frame.columns:
            frame = frame.with_columns(pl.lit(None).cast(dtype).alias(col))
        else:
            frame = frame.with_columns(pl.col(col).cast(dtype, strict=False).alias(col))
    return frame.select(schema.keys()).sort("timestamp")


def compute_long_short_ratio_features(ratio_frame: pl.DataFrame) -> pl.DataFrame:
    schema = {
        "timestamp": pl.Int64,
        "long_short_account_ratio": pl.Float64,
        "top_trader_long_short_account_ratio": pl.Float64,
        "top_trader_long_short_position_ratio": pl.Float64,
    }
    if ratio_frame.is_empty() or "timestamp" not in ratio_frame.columns:
        return pl.DataFrame(schema=schema)
    cols = [pl.col("timestamp").cast(pl.Int64)]
    for col in schema:
        if col == "timestamp":
            continue
        if col in ratio_frame.columns:
            cols.append(pl.col(col).cast(pl.Float64))
        else:
            cols.append(pl.lit(None).cast(pl.Float64).alias(col))
    return (
        ratio_frame.select(cols)
        .sort("timestamp")
        .with_columns(_hour_col())
        .group_by("timestamp")
        .last()
        .select(schema.keys())
        .sort("timestamp")
    )


def compute_flow_features(hourly_flow_frame: pl.DataFrame, window_hours: int = 168) -> pl.DataFrame:
    schema = {
        "timestamp": pl.Int64,
        "net_exchange_flow": pl.Float64,
        "flow_zscore": pl.Float64,
        "flow_zscore_negative_streak_hours": pl.Int64,
    }
    if hourly_flow_frame.is_empty():
        return pl.DataFrame(schema=schema)
    frame = hourly_flow_frame
    if "net_exchange_flow" not in frame.columns:
        frame = frame.with_columns(
            (pl.col("inflow").fill_null(0.0) - pl.col("outflow").fill_null(0.0)).alias(
                "net_exchange_flow"
            )
        )
    frame = frame.select(
        [pl.col("timestamp").cast(pl.Int64), pl.col("net_exchange_flow").cast(pl.Float64)]
    ).sort("timestamp")
    z = frame.with_columns(
        [
            pl.col("net_exchange_flow")
            .rolling_mean(window_size=window_hours, min_samples=2)
            .alias("_mean"),
            pl.col("net_exchange_flow")
            .rolling_std(window_size=window_hours, min_samples=2)
            .alias("_std"),
        ]
    ).with_columns(
        pl.when(pl.col("_std").is_not_null() & (pl.col("_std") > 0.0))
        .then((pl.col("net_exchange_flow") - pl.col("_mean")) / pl.col("_std"))
        .otherwise(0.0)
        .alias("flow_zscore")
    )
    streak = []
    current = 0
    for value in z.get_column("flow_zscore").to_list():
        current = current + 1 if value is not None and value < 0 else 0
        streak.append(current)
    return z.with_columns(pl.Series("flow_zscore_negative_streak_hours", streak)).select(
        schema.keys()
    )


def compute_whale_features(balance_snapshots: pl.DataFrame) -> pl.DataFrame:
    if balance_snapshots.is_empty():
        return pl.DataFrame(schema={"timestamp": pl.Int64, "whale_accumulation_ratio": pl.Float64})
    if "whale_accumulation_ratio" in balance_snapshots.columns:
        return balance_snapshots.select(
            [
                pl.col("timestamp").cast(pl.Int64),
                pl.col("whale_accumulation_ratio").cast(pl.Float64),
            ]
        ).sort("timestamp")
    required = {"whale_balance", "total_supply"}
    if not required.issubset(balance_snapshots.columns):
        raise ValueError(
            "balance snapshots require whale_accumulation_ratio or whale_balance/total_supply"
        )
    return balance_snapshots.select(
        [
            pl.col("timestamp").cast(pl.Int64),
            (pl.col("whale_balance") / pl.col("total_supply"))
            .cast(pl.Float64)
            .alias("whale_accumulation_ratio"),
        ]
    ).sort("timestamp")


def _imbalance_from_depth(frame: pl.DataFrame, bid: str, ask: str, alias: str) -> pl.Expr:
    return (
        pl.when((pl.col(bid) + pl.col(ask)) > 0.0)
        .then((pl.col(bid) - pl.col(ask)) / (pl.col(bid) + pl.col(ask)))
        .otherwise(None)
        .alias(alias)
    )


def compute_depth_features(book_frame: pl.DataFrame) -> pl.DataFrame:
    schema = {
        "timestamp": pl.Int64,
        "depth_imbalance_10_mean": pl.Float64,
        "depth_imbalance_25_mean": pl.Float64,
        "depth_imbalance_10_slope": pl.Float64,
    }
    if book_frame.is_empty():
        return pl.DataFrame(schema=schema)
    frame = book_frame.with_columns(_hour_col())
    if "ob_imbalance_10" not in frame.columns:
        if {"ob_bid_vol_10", "ob_ask_vol_10"}.issubset(frame.columns):
            frame = frame.with_columns(
                _imbalance_from_depth(frame, "ob_bid_vol_10", "ob_ask_vol_10", "ob_imbalance_10")
            )
        elif "ob_imbalance_25" in frame.columns:
            frame = frame.with_columns(pl.col("ob_imbalance_25").alias("ob_imbalance_10"))
        else:
            frame = frame.with_columns(
                _imbalance_from_depth(frame, "ob_bid_vol_25", "ob_ask_vol_25", "ob_imbalance_10")
            )
    if "ob_imbalance_25" not in frame.columns:
        frame = frame.with_columns(
            _imbalance_from_depth(frame, "ob_bid_vol_25", "ob_ask_vol_25", "ob_imbalance_25")
        )
    return (
        frame.group_by("timestamp")
        .agg(
            [
                pl.col("ob_imbalance_10").mean().alias("depth_imbalance_10_mean"),
                pl.col("ob_imbalance_25").mean().alias("depth_imbalance_25_mean"),
                (pl.col("ob_imbalance_10").last() - pl.col("ob_imbalance_10").first()).alias(
                    "depth_imbalance_10_slope"
                ),
            ]
        )
        .sort("timestamp")
    )


def compute_resilience_features(
    trade_frame: pl.DataFrame,
    price_frame: pl.DataFrame,
    *,
    large_trade_usd: float = 50_000.0,
    resilience_minutes: int = 5,
) -> pl.DataFrame:
    schema = {
        "timestamp": pl.Int64,
        "large_sell_events": pl.Int64,
        "resilient_sell_events": pl.Int64,
        "resilience_score": pl.Float64,
        "large_trade_buy_ratio": pl.Float64,
    }
    if trade_frame.is_empty():
        return pl.DataFrame(schema=schema)
    if "notional_usd" not in trade_frame.columns:
        return pl.DataFrame(schema=schema)
    notional = trade_frame.get_column("notional_usd").cast(pl.Float64, strict=False)
    if notional.null_count() == trade_frame.height:
        return pl.DataFrame(schema=schema)
    trades = trade_frame.with_columns(
        [
            pl.col("timestamp").cast(pl.Int64, strict=False),
            pl.col("price").cast(pl.Float64, strict=False),
            pl.col("notional_usd").cast(pl.Float64, strict=False),
            pl.col("side").cast(pl.String),
        ]
    ).sort("timestamp").with_columns(_hour_col())
    large = trades.filter(pl.col("notional_usd") >= large_trade_usd)
    if large.is_empty():
        return trades.group_by("timestamp").agg(
            [
                pl.lit(0).alias("large_sell_events"),
                pl.lit(0).alias("resilient_sell_events"),
                pl.lit(0.0).alias("resilience_score"),
                pl.lit(None).cast(pl.Float64).alias("large_trade_buy_ratio"),
            ]
        )
    prices = (
        price_frame.select(["timestamp", "close"]).sort("timestamp")
        if not price_frame.is_empty()
        else pl.DataFrame()
    )
    resilient_by_trade: dict[str, bool] = {}
    if not prices.is_empty():
        for row in large.filter(pl.col("side") == "sell").to_dicts():
            ts = int(row["timestamp"])
            px = float(row["price"])
            future = prices.filter(
                (pl.col("timestamp") > ts)
                & (pl.col("timestamp") <= ts + resilience_minutes * 60_000)
            )
            resilient_by_trade[str(row.get("trade_id", ts))] = bool(
                future.height and future["close"].max() >= px
            )
    sell = large.filter(pl.col("side") == "sell")
    sell_flags = [
        resilient_by_trade.get(str(row.get("trade_id", row["timestamp"])), False)
        for row in sell.to_dicts()
    ]
    sell = (
        sell.with_columns(pl.Series("_resilient", sell_flags))
        if sell.height
        else sell.with_columns(pl.lit(False).alias("_resilient"))
    )
    sell_hour = sell.group_by("timestamp").agg(
        [
            pl.len().alias("large_sell_events"),
            pl.col("_resilient").sum().alias("resilient_sell_events"),
        ]
    )
    buy_ratio = large.group_by("timestamp").agg(
        (
            pl.when(pl.col("side") == "buy").then(pl.col("notional_usd")).otherwise(0.0).sum()
            / pl.col("notional_usd").sum()
        ).alias("large_trade_buy_ratio")
    )
    return (
        sell_hour.join(buy_ratio, on="timestamp", how="outer_coalesce")
        .with_columns(
            pl.when(pl.col("large_sell_events") > 0)
            .then(pl.col("resilient_sell_events") / pl.col("large_sell_events"))
            .otherwise(0.0)
            .alias("resilience_score")
        )
        .select(schema.keys())
        .sort("timestamp")
    )


def compute_message_features(
    messages: pl.DataFrame,
    classifications: pl.DataFrame,
    *,
    window_hours: int = 168,
) -> pl.DataFrame:
    schema = {
        "timestamp": pl.Int64,
        "mention_growth": pl.Float64,
        "fundamental_news_ratio": pl.Float64,
        "emotion_news_ratio": pl.Float64,
    }
    if messages.is_empty() or "timestamp" not in messages.columns:
        return pl.DataFrame(schema=schema)
    msg = messages.with_columns(_hour_col()).select(
        [
            pl.col("timestamp").cast(pl.Int64),
            pl.col("source_id").cast(pl.String),
        ]
    )
    if not classifications.is_empty() and {"message_id", "message_type"}.issubset(
        classifications.columns
    ):
        classes = classifications.select(
            [pl.col("message_id").cast(pl.String), pl.col("message_type").cast(pl.String)]
        )
        msg = msg.join(classes, left_on="source_id", right_on="message_id", how="left")
    else:
        msg = msg.with_columns(pl.lit("unknown_or_noise").alias("message_type"))
    grouped = msg.group_by("timestamp").agg(
        [
            pl.len().alias("_mentions"),
            (pl.col("message_type") == "fundamental").sum().alias("_fundamental"),
            (pl.col("message_type") == "community_emotion").sum().alias("_emotion"),
        ]
    )
    return (
        grouped.sort("timestamp")
        .with_columns(
            [
                pl.col("_mentions")
                .rolling_mean(window_size=window_hours, min_samples=2)
                .alias("_baseline"),
                pl.when(pl.col("_mentions") > 0)
                .then(pl.col("_fundamental") / pl.col("_mentions"))
                .otherwise(None)
                .alias("fundamental_news_ratio"),
                pl.when(pl.col("_mentions") > 0)
                .then(pl.col("_emotion") / pl.col("_mentions"))
                .otherwise(None)
                .alias("emotion_news_ratio"),
            ]
        )
        .with_columns(
            pl.when(pl.col("_baseline") > 0.0)
            .then(pl.col("_mentions") / pl.col("_baseline"))
            .otherwise(None)
            .alias("mention_growth")
        )
        .select(schema.keys())
    )


def join_hourly_accumulation_features(
    *,
    symbol: str,
    inst_id: str,
    price_frame: pl.DataFrame,
    chain: str = "",
    token_address: str = "",
    flow_frame: pl.DataFrame | None = None,
    whale_frame: pl.DataFrame | None = None,
    book_frame: pl.DataFrame | None = None,
    trade_frame: pl.DataFrame | None = None,
    discovery_row: dict[str, object] | None = None,
    funding_frame: pl.DataFrame | None = None,
    open_interest_frame: pl.DataFrame | None = None,
    taker_volume_frame: pl.DataFrame | None = None,
    long_short_ratio_frame: pl.DataFrame | None = None,
    message_frame: pl.DataFrame | None = None,
    classification_frame: pl.DataFrame | None = None,
    source_coverage_score: float | None = None,
    flow_zscore_window_hours: int = 168,
    large_trade_usd: float = 50_000.0,
    resilience_minutes: int = 5,
    ma_hours: int = 200,
    max_source_staleness_hours: int = 48,
) -> pl.DataFrame:
    if price_frame.is_empty():
        return empty_feature_frame()
    base = compute_price_features(price_frame, ma_hours=ma_hours).with_columns(
        [
            pl.lit(symbol).alias("symbol"),
            pl.lit(inst_id).alias("inst_id"),
            pl.lit(chain).alias("chain"),
            pl.lit(token_address).alias("token_address"),
        ]
    )
    frames = [base]
    warnings = []
    frames.append(compute_structure_features(price_frame))
    if flow_frame is not None and not flow_frame.is_empty():
        frames.append(compute_flow_features(flow_frame, window_hours=flow_zscore_window_hours))
    else:
        warnings.append("onchain_missing")
    if whale_frame is not None and not whale_frame.is_empty():
        frames.append(compute_whale_features(whale_frame))
    else:
        warnings.append("whale_missing")
    if book_frame is not None and not book_frame.is_empty():
        frames.append(_with_source_timestamp(compute_depth_features(book_frame), "books"))
    else:
        warnings.append("book_missing")
    if trade_frame is not None and not trade_frame.is_empty():
        if (
            "notional_usd" not in trade_frame.columns
            or trade_frame.get_column("notional_usd").null_count() == trade_frame.height
        ):
            warnings.append("trade_notional_metadata_missing")
        frames.append(
            _with_source_timestamp(
                compute_resilience_features(
                    trade_frame,
                    price_frame,
                    large_trade_usd=large_trade_usd,
                    resilience_minutes=resilience_minutes,
                ),
                "trades",
            )
        )
    else:
        warnings.append("trades_missing")
    if funding_frame is not None and not funding_frame.is_empty():
        frames.append(_with_source_timestamp(compute_funding_features(funding_frame), "funding"))
    else:
        warnings.append("funding_missing")
    if open_interest_frame is not None and not open_interest_frame.is_empty():
        frames.append(
            _with_source_timestamp(
                compute_open_interest_features(open_interest_frame), "open_interest"
            )
        )
    else:
        warnings.append("open_interest_missing")
    if taker_volume_frame is not None and not taker_volume_frame.is_empty():
        frames.append(
            _with_source_timestamp(
                compute_taker_volume_features(taker_volume_frame), "taker_volume"
            )
        )
    else:
        warnings.append("taker_volume_missing")
    if long_short_ratio_frame is not None and not long_short_ratio_frame.is_empty():
        frames.append(
            _with_source_timestamp(
                compute_long_short_ratio_features(long_short_ratio_frame), "long_short_ratios"
            )
        )
    else:
        warnings.append("long_short_ratios_missing")
    if message_frame is not None and not message_frame.is_empty():
        if classification_frame is None or classification_frame.is_empty():
            warnings.append("message_classifications_missing")
        classifications = (
            classification_frame if classification_frame is not None else pl.DataFrame()
        )
        frames.append(compute_message_features(message_frame, classifications))
    else:
        warnings.append("messages_missing")
    joined = reduce(
        lambda left, right: left.join_asof(
            right.sort("timestamp"), on="timestamp", strategy="backward"
        ),
        frames,
    )
    for column in ("mention_growth", "fundamental_news_ratio", "emotion_news_ratio"):
        if column not in joined.columns:
            joined = joined.with_columns(pl.lit(None).cast(pl.Float64).alias(column))
    joined = joined.with_columns(
        [
            pl.lit(_discovery_float(discovery_row, "quote_volume_24h"))
            .cast(pl.Float64)
            .alias("quote_volume_24h"),
            pl.lit(_discovery_float(discovery_row, "spread_bps"))
            .cast(pl.Float64)
            .alias("spread_bps_mean"),
            pl.lit(None).cast(pl.Float64).alias("depth_rebuild_score"),
            pl.lit(source_coverage_score).cast(pl.Float64).alias("source_coverage_score"),
            pl.lit(";".join(warnings)).alias("data_quality_warning"),
        ]
    )
    joined = _mark_stale(
        joined,
        "books",
        ("depth_imbalance_10_mean", "depth_imbalance_25_mean", "depth_imbalance_10_slope"),
        max_source_staleness_hours=max_source_staleness_hours,
    )
    joined = _mark_stale(
        joined,
        "trades",
        ("large_sell_events", "resilient_sell_events", "resilience_score", "large_trade_buy_ratio"),
        max_source_staleness_hours=max_source_staleness_hours,
    )
    joined = _mark_stale(
        joined,
        "funding",
        ("funding_rate", "funding_zscore"),
        max_source_staleness_hours=max_source_staleness_hours,
    )
    joined = _mark_stale(
        joined,
        "open_interest",
        ("open_interest_usd", "open_interest_change_24h", "open_interest_usd_change_24h"),
        max_source_staleness_hours=max_source_staleness_hours,
    )
    joined = _mark_stale(
        joined,
        "taker_volume",
        (
            "taker_buy_volume",
            "taker_sell_volume",
            "taker_buy_ratio",
            "taker_volume_total",
            "taker_volume_imbalance",
        ),
        max_source_staleness_hours=max_source_staleness_hours,
    )
    joined = _mark_stale(
        joined,
        "long_short_ratios",
        (
            "long_short_account_ratio",
            "top_trader_long_short_account_ratio",
            "top_trader_long_short_position_ratio",
        ),
        max_source_staleness_hours=max_source_staleness_hours,
    )
    for col, dtype in FEATURE_SCHEMA.items():
        if col not in joined.columns:
            joined = joined.with_columns(pl.lit(None).cast(dtype).alias(col))
    return joined.select(pl.col(col).cast(dtype) for col, dtype in FEATURE_SCHEMA.items())


def join_hourly_accumulation_features_batch(
    *,
    price_frame: pl.DataFrame,
    symbols: tuple[str, ...],
    discovery: pl.DataFrame | None = None,
    books_by_symbol: dict[str, pl.DataFrame] | None = None,
    trades_by_symbol: dict[str, pl.DataFrame] | None = None,
    funding_by_symbol: dict[str, pl.DataFrame] | None = None,
    open_interest_by_symbol: dict[str, pl.DataFrame] | None = None,
    taker_volume_by_symbol: dict[str, pl.DataFrame] | None = None,
    long_short_ratios_by_symbol: dict[str, pl.DataFrame] | None = None,
    messages_by_symbol: dict[str, pl.DataFrame] | None = None,
    classifications_by_symbol: dict[str, pl.DataFrame] | None = None,
    coverage_scores: dict[str, float] | None = None,
    flow_zscore_window_hours: int = 168,
    large_trade_usd: float = 50_000.0,
    resilience_minutes: int = 5,
    ma_hours: int = 200,
    max_source_staleness_hours: int = 48,
) -> pl.DataFrame:
    if price_frame.is_empty():
        return empty_feature_frame()
    frames = []
    for symbol in symbols:
        symbol_prices = (
            price_frame.filter(pl.col("symbol") == symbol)
            if "symbol" in price_frame.columns
            else price_frame
        )
        if symbol_prices.is_empty():
            continue
        discovery_row = _discovery_row(discovery, symbol)
        frames.append(
            join_hourly_accumulation_features(
                symbol=symbol,
                inst_id=symbol,
                price_frame=symbol_prices,
                book_frame=(books_by_symbol or {}).get(symbol),
                trade_frame=(trades_by_symbol or {}).get(symbol),
                discovery_row=discovery_row,
                funding_frame=(funding_by_symbol or {}).get(symbol),
                open_interest_frame=(open_interest_by_symbol or {}).get(symbol),
                taker_volume_frame=(taker_volume_by_symbol or {}).get(symbol),
                long_short_ratio_frame=(long_short_ratios_by_symbol or {}).get(symbol),
                message_frame=(messages_by_symbol or {}).get(symbol),
                classification_frame=(classifications_by_symbol or {}).get(symbol),
                source_coverage_score=(coverage_scores or {}).get(symbol),
                flow_zscore_window_hours=flow_zscore_window_hours,
                large_trade_usd=large_trade_usd,
                resilience_minutes=resilience_minutes,
                ma_hours=ma_hours,
                max_source_staleness_hours=max_source_staleness_hours,
            )
        )
    return pl.concat(frames, how="vertical") if frames else empty_feature_frame()


def _discovery_row(discovery: pl.DataFrame | None, symbol: str) -> dict[str, object] | None:
    if discovery is None or discovery.is_empty() or "symbol" not in discovery.columns:
        return None
    rows = discovery.filter(pl.col("symbol") == symbol).head(1)
    return rows.to_dicts()[0] if not rows.is_empty() else None


def _discovery_float(row: dict[str, object] | None, key: str) -> float | None:
    if not row:
        return None
    value = row.get(key)
    return None if value is None else float(value)


def _with_source_timestamp(frame: pl.DataFrame, source: str) -> pl.DataFrame:
    if frame.is_empty() or "timestamp" not in frame.columns:
        return frame
    return frame.with_columns(pl.col("timestamp").alias(f"_{source}_timestamp"))


def _mark_stale(
    frame: pl.DataFrame,
    source: str,
    columns: tuple[str, ...],
    *,
    max_source_staleness_hours: int,
) -> pl.DataFrame:
    source_ts = f"_{source}_timestamp"
    if frame.is_empty() or source_ts not in frame.columns:
        return frame
    stale = (pl.col(source_ts).is_not_null()) & (
        (pl.col("timestamp") - pl.col(source_ts))
        > max_source_staleness_hours * HOUR_MS
    )
    updates = [
        pl.when(stale).then(None).otherwise(pl.col(col)).alias(col)
        for col in columns
        if col in frame.columns
    ]
    updates.append(
        pl.when(stale)
        .then(
            pl.concat_str(
                [pl.col("data_quality_warning").fill_null(""), pl.lit(f"{source}_stale")],
                separator=";",
            ).str.strip_chars(";")
        )
        .otherwise(pl.col("data_quality_warning"))
        .alias("data_quality_warning")
    )
    return frame.with_columns(updates)
