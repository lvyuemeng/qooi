"""Pure feature engineering for accumulation-like scans."""

from __future__ import annotations

from functools import reduce

import polars as pl

from qooi.accumulation.schema import FEATURE_SCHEMA, empty_feature_frame

HOUR_MS = 3_600_000


def _hour_col(name: str = "timestamp") -> pl.Expr:
    return (pl.col(name) // HOUR_MS * HOUR_MS).alias("timestamp")


def _warning(base: pl.DataFrame, warning: str) -> pl.DataFrame:
    if base.is_empty():
        return base
    if "data_quality_warning" not in base.columns:
        return base.with_columns(pl.lit(warning).alias("data_quality_warning"))
    return base.with_columns(
        pl.concat_str(
            [pl.col("data_quality_warning").fill_null(""), pl.lit(warning)], separator=";"
        )
        .str.strip_chars(";")
        .alias("data_quality_warning")
    )


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
    schema = {"timestamp": pl.Int64, "open_interest_change_24h": pl.Float64}
    if open_interest_frame.is_empty() or "open_interest" not in open_interest_frame.columns:
        return pl.DataFrame(schema=schema)
    return (
        open_interest_frame.select(
            [pl.col("timestamp").cast(pl.Int64), pl.col("open_interest").cast(pl.Float64)]
        )
        .sort("timestamp")
        .with_columns(
            (pl.col("open_interest") / pl.col("open_interest").shift(24) - 1.0).alias(
                "open_interest_change_24h"
            )
        )
        .select(schema.keys())
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
    message_frame: pl.DataFrame | None = None,
    classification_frame: pl.DataFrame | None = None,
    source_coverage_score: float | None = None,
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
        frames.append(compute_flow_features(flow_frame))
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
            _with_source_timestamp(compute_resilience_features(trade_frame, price_frame), "trades")
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
        ("open_interest_change_24h",),
        max_source_staleness_hours=max_source_staleness_hours,
    )
    for col, dtype in FEATURE_SCHEMA.items():
        if col not in joined.columns:
            joined = joined.with_columns(pl.lit(None).cast(dtype).alias(col))
    return joined.select(FEATURE_SCHEMA.keys())


def join_hourly_accumulation_features_batch(
    *,
    price_frame: pl.DataFrame,
    symbols: tuple[str, ...],
    discovery: pl.DataFrame | None = None,
    books_by_symbol: dict[str, pl.DataFrame] | None = None,
    trades_by_symbol: dict[str, pl.DataFrame] | None = None,
    funding_by_symbol: dict[str, pl.DataFrame] | None = None,
    open_interest_by_symbol: dict[str, pl.DataFrame] | None = None,
    messages_by_symbol: dict[str, pl.DataFrame] | None = None,
    classifications_by_symbol: dict[str, pl.DataFrame] | None = None,
    coverage_scores: dict[str, float] | None = None,
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
                message_frame=(messages_by_symbol or {}).get(symbol),
                classification_frame=(classifications_by_symbol or {}).get(symbol),
                source_coverage_score=(coverage_scores or {}).get(symbol),
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
