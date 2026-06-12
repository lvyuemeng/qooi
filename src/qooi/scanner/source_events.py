"""Potential scanner source event and outcome evaluation."""

from __future__ import annotations

import polars as pl

from qooi.scanner import entropy_expr, outcome_bucket_expr, pct_change_expr

SOURCE_EVENT_SCHEMA = {
    "symbol": pl.String,
    "source_family": pl.String,
    "source_state": pl.String,
    "source_direction": pl.String,
    "provider_timestamp_ms": pl.Int64,
    "known_at_ms": pl.Int64,
    "aligned_bar": pl.String,
    "aligned_bar_close_ms": pl.Int64,
    "serialization_status": pl.String,
}

SOURCE_OUTCOME_SCHEMA = {
    **SOURCE_EVENT_SCHEMA,
    "outcome_horizon": pl.Int64,
    "close_at_event": pl.Float64,
    "future_close": pl.Float64,
    "forward_return_pct": pl.Float64,
    "forward_min_return_pct": pl.Float64,
    "forward_max_return_pct": pl.Float64,
    "path_range_pct": pl.Float64,
    "tail_asymmetry_pct": pl.Float64,
    "outcome_available": pl.Boolean,
    "outcome_reason": pl.String,
}

SOURCE_OUTCOME_HORIZONS = {
    "books": (1,),
    "trades": (1, 4),
    "taker_volume": (4, 12),
    "open_interest": (4, 12, 24),
    "funding": (8, 24, 48),
    "long_short_ratios": (4, 12, 24),
}


def source_events_frame(
    source_frames: dict[str, pl.DataFrame], bars: pl.DataFrame, bar: str
) -> pl.DataFrame:
    prices = _source_price_context_frame(bars)
    frames = [
        event_frame
        for event_frame in (
            _book_events_frame(source_frames.get("books", pl.DataFrame()), prices, bar),
            _trade_events_frame(source_frames.get("trades", pl.DataFrame()), prices, bar),
            _funding_events_frame(source_frames.get("funding", pl.DataFrame()), prices, bar),
            _open_interest_events_frame(
                source_frames.get("open_interest", pl.DataFrame()), prices, bar
            ),
            _taker_volume_events_frame(
                source_frames.get("taker_volume", pl.DataFrame()), prices, bar
            ),
            _long_short_ratio_events_frame(
                source_frames.get("long_short_ratios", pl.DataFrame()), prices, bar
            ),
        )
        if not event_frame.is_empty()
    ]
    if not frames:
        return pl.DataFrame(schema=SOURCE_EVENT_SCHEMA)
    return pl.concat(frames, how="vertical_relaxed").select(*SOURCE_EVENT_SCHEMA.keys())


def _source_price_context_frame(bars: pl.DataFrame) -> pl.DataFrame:
    if bars.is_empty() or not {"symbol", "timestamp", "close"}.issubset(bars.columns):
        return pl.DataFrame(
            schema={
                "symbol": pl.String,
                "timestamp": pl.Int64,
                "price_return_pct": pl.Float64,
            }
        )
    if "previous_close" in bars.columns:
        return bars.sort("symbol", "timestamp").select(
            "symbol",
            "timestamp",
            pct_change_expr("close", "previous_close").alias("price_return_pct"),
        )
    return (
        bars.sort("symbol", "timestamp")
        .with_columns(pl.col("close").shift(1).over("symbol").alias("previous_close"))
        .select(
            "symbol",
            "timestamp",
            pct_change_expr("close", "previous_close").alias("price_return_pct"),
        )
    )


def _with_price_context(frame: pl.DataFrame, prices: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty() or prices.is_empty() or not {"symbol", "timestamp"}.issubset(frame.columns):
        return frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("price_return_pct"))
    return frame.sort("symbol", "timestamp").join_asof(
        prices.sort("symbol", "timestamp"),
        on="timestamp",
        by="symbol",
        strategy="backward",
        check_sortedness=False,
    )


def _book_events_frame(frame: pl.DataFrame, prices: pl.DataFrame, bar: str) -> pl.DataFrame:
    if frame.is_empty() or not {"symbol", "timestamp"}.issubset(frame.columns):
        return pl.DataFrame(schema=SOURCE_EVENT_SCHEMA)
    frame = _with_price_context(frame, prices)
    imbalance = _first_existing_column(
        frame, ("ob_imbalance_25", "ob_imbalance_10", "ob_imbalance_5")
    )
    if imbalance is None:
        return pl.DataFrame(schema=SOURCE_EVENT_SCHEMA)
    state = (
        pl.when(pl.col(imbalance) >= 0.2)
        .then(pl.lit("bid_support"))
        .when(pl.col(imbalance) <= -0.2)
        .then(pl.lit("ask_pressure"))
        .otherwise(pl.lit("balanced_book"))
    )
    direction = (
        pl.when(pl.col(imbalance) >= 0.2)
        .then(pl.lit("bullish"))
        .when(pl.col(imbalance) <= -0.2)
        .then(pl.lit("bearish"))
        .otherwise(pl.lit("neutral"))
    )
    return _source_event_base(
        frame.with_columns(state.alias("source_state"), direction.alias("source_direction")),
        "books",
        bar,
    )


def _trade_events_frame(frame: pl.DataFrame, prices: pl.DataFrame, bar: str) -> pl.DataFrame:
    if frame.is_empty() or not {"symbol", "timestamp", "side"}.issubset(frame.columns):
        return pl.DataFrame(schema=SOURCE_EVENT_SCHEMA)
    value_col = _first_existing_column(frame, ("notional_usd", "notional", "size"))
    value = pl.col(value_col).cast(pl.Float64, strict=False) if value_col else pl.lit(1.0)
    by_timestamp = frame.group_by("symbol", "timestamp").agg(
        pl.when(pl.col("side") == "buy").then(value).otherwise(0.0).sum().alias("buy_value"),
        pl.when(pl.col("side") == "sell").then(value).otherwise(0.0).sum().alias("sell_value"),
    )
    by_timestamp = _with_price_context(by_timestamp, prices)
    ratio = pl.col("buy_value") / pl.when(pl.col("sell_value") > 0.0).then(
        pl.col("sell_value")
    ).otherwise(None)
    price_up = pl.col("price_return_pct") > 0.0
    price_down = pl.col("price_return_pct") < 0.0
    buy_dominant = ratio >= 1.25
    sell_dominant = ratio <= 0.8
    return _source_event_base(
        by_timestamp.with_columns(
            pl.when(buy_dominant & price_up)
            .then(pl.lit("buy_aggression_continuation"))
            .when(buy_dominant & price_down)
            .then(pl.lit("trapped_buying_pressure"))
            .when(sell_dominant & price_down)
            .then(pl.lit("sell_aggression_continuation"))
            .when(sell_dominant & price_up)
            .then(pl.lit("sell_absorption_pressure"))
            .otherwise(None)
            .alias("source_state"),
            pl.when(buy_dominant & price_up)
            .then(pl.lit("bullish"))
            .when(buy_dominant & price_down)
            .then(pl.lit("bearish"))
            .when(sell_dominant & price_down)
            .then(pl.lit("bearish"))
            .when(sell_dominant & price_up)
            .then(pl.lit("bullish"))
            .otherwise(None)
            .alias("source_direction"),
        ).filter(pl.col("source_state").is_not_null()),
        "trades",
        bar,
    )


def _funding_events_frame(frame: pl.DataFrame, prices: pl.DataFrame, bar: str) -> pl.DataFrame:
    if frame.is_empty() or not {"symbol", "timestamp"}.issubset(frame.columns):
        return pl.DataFrame(schema=SOURCE_EVENT_SCHEMA)
    if "funding_rate" not in frame.columns:
        return pl.DataFrame(schema=SOURCE_EVENT_SCHEMA)
    frame = _with_price_context(frame, prices)
    funding = pl.col("funding_rate").cast(pl.Float64, strict=False)
    price_up = pl.col("price_return_pct") > 0.0
    price_down = pl.col("price_return_pct") < 0.0
    return _source_event_base(
        frame.with_columns(
            pl.when((funding > 0.0) & price_down)
            .then(pl.lit("crowded_longs_under_stress"))
            .when((funding < 0.0) & price_up)
            .then(pl.lit("crowded_shorts_under_stress"))
            .otherwise(None)
            .alias("source_state"),
            pl.when((funding > 0.0) & price_down)
            .then(pl.lit("bearish"))
            .when((funding < 0.0) & price_up)
            .then(pl.lit("bullish"))
            .otherwise(None)
            .alias("source_direction"),
        ).filter(pl.col("source_state").is_not_null()),
        "funding",
        bar,
    )


def _open_interest_events_frame(
    frame: pl.DataFrame, prices: pl.DataFrame, bar: str
) -> pl.DataFrame:
    if frame.is_empty() or not {"symbol", "timestamp"}.issubset(frame.columns):
        return pl.DataFrame(schema=SOURCE_EVENT_SCHEMA)
    value_col = _first_existing_column(frame, ("open_interest_usd", "open_interest"))
    if value_col is None:
        return pl.DataFrame(schema=SOURCE_EVENT_SCHEMA)
    frame = _with_price_context(frame, prices)
    oi_delta = pl.col(value_col).cast(pl.Float64, strict=False).diff().over("symbol")
    price_up = pl.col("price_return_pct") > 0.0
    price_down = pl.col("price_return_pct") < 0.0
    oi_up = oi_delta > 0.0
    oi_down = oi_delta < 0.0
    return _source_event_base(
        frame.with_columns(
            pl.when(oi_up & price_down)
            .then(pl.lit("short_buildup_with_price_down"))
            .when(oi_up & price_up)
            .then(pl.lit("long_buildup_with_price_up"))
            .when(oi_down & price_down)
            .then(pl.lit("long_liquidation_flush"))
            .when(oi_down & price_up)
            .then(pl.lit("short_covering_rally"))
            .otherwise(None)
            .alias("source_state"),
            pl.when(oi_up & price_down)
            .then(pl.lit("bearish"))
            .when(oi_up & price_up)
            .then(pl.lit("bullish"))
            .when(oi_down & price_down)
            .then(pl.lit("bearish"))
            .when(oi_down & price_up)
            .then(pl.lit("bullish"))
            .otherwise(None)
            .alias("source_direction"),
        ).filter(pl.col("source_state").is_not_null()),
        "open_interest",
        bar,
    )


def _taker_volume_events_frame(frame: pl.DataFrame, prices: pl.DataFrame, bar: str) -> pl.DataFrame:
    required = {"symbol", "timestamp", "taker_buy_volume", "taker_sell_volume"}
    if frame.is_empty() or not required.issubset(frame.columns):
        return pl.DataFrame(schema=SOURCE_EVENT_SCHEMA)
    frame = _with_price_context(frame, prices)
    buy = pl.col("taker_buy_volume").cast(pl.Float64, strict=False)
    sell = pl.col("taker_sell_volume").cast(pl.Float64, strict=False)
    ratio = buy / pl.when(sell > 0.0).then(sell).otherwise(None)
    price_up = pl.col("price_return_pct") > 0.0
    price_down = pl.col("price_return_pct") < 0.0
    buy_dominant = ratio >= 1.25
    sell_dominant = ratio <= 0.8
    return _source_event_base(
        frame.with_columns(
            pl.when(buy_dominant & price_up)
            .then(pl.lit("taker_buy_continuation"))
            .when(buy_dominant & price_down)
            .then(pl.lit("taker_buy_trap"))
            .when(sell_dominant & price_down)
            .then(pl.lit("taker_sell_continuation"))
            .when(sell_dominant & price_up)
            .then(pl.lit("taker_sell_absorption"))
            .otherwise(None)
            .alias("source_state"),
            pl.when(buy_dominant & price_up)
            .then(pl.lit("bullish"))
            .when(buy_dominant & price_down)
            .then(pl.lit("bearish"))
            .when(sell_dominant & price_down)
            .then(pl.lit("bearish"))
            .when(sell_dominant & price_up)
            .then(pl.lit("bullish"))
            .otherwise(None)
            .alias("source_direction"),
        ).filter(pl.col("source_state").is_not_null()),
        "taker_volume",
        bar,
    )


def _long_short_ratio_events_frame(
    frame: pl.DataFrame, prices: pl.DataFrame, bar: str
) -> pl.DataFrame:
    if frame.is_empty() or not {"symbol", "timestamp"}.issubset(frame.columns):
        return pl.DataFrame(schema=SOURCE_EVENT_SCHEMA)
    value_col = _first_existing_column(
        frame,
        (
            "long_short_account_ratio",
            "top_trader_long_short_account_ratio",
            "top_trader_long_short_position_ratio",
        ),
    )
    if value_col is None:
        return pl.DataFrame(schema=SOURCE_EVENT_SCHEMA)
    frame = _with_price_context(frame, prices)
    ratio_delta = pl.col(value_col).cast(pl.Float64, strict=False).diff().over("symbol")
    price_up = pl.col("price_return_pct") > 0.0
    price_down = pl.col("price_return_pct") < 0.0
    crowding_longs = ratio_delta > 0.0
    crowding_shorts = ratio_delta < 0.0
    return _source_event_base(
        frame.with_columns(
            pl.when(crowding_longs & price_down)
            .then(pl.lit("crowded_longs_price_down"))
            .when(crowding_shorts & price_up)
            .then(pl.lit("crowded_shorts_price_up"))
            .otherwise(None)
            .alias("source_state"),
            pl.when(crowding_longs & price_down)
            .then(pl.lit("bearish"))
            .when(crowding_shorts & price_up)
            .then(pl.lit("bullish"))
            .otherwise(None)
            .alias("source_direction"),
        ).filter(pl.col("source_state").is_not_null()),
        "long_short_ratios",
        bar,
    )


def _source_event_base(frame: pl.DataFrame, family: str, bar: str) -> pl.DataFrame:
    return frame.select("symbol", "timestamp", "source_state", "source_direction").with_columns(
        pl.lit(family).alias("source_family"),
        pl.col("timestamp").cast(pl.Int64, strict=False).alias("provider_timestamp_ms"),
        pl.col("timestamp").cast(pl.Int64, strict=False).alias("known_at_ms"),
        pl.lit(bar).alias("aligned_bar"),
        pl.col("timestamp").cast(pl.Int64, strict=False).alias("aligned_bar_close_ms"),
        pl.lit("stored_source_row").alias("serialization_status"),
    )


def _first_existing_column(frame: pl.DataFrame, columns: tuple[str, ...]) -> str | None:
    for col in columns:
        if col in frame.columns:
            return col
    return None


def source_outcomes_frame(source_events: pl.DataFrame, bars: pl.DataFrame) -> pl.DataFrame:
    if source_events.is_empty():
        return pl.DataFrame(schema=SOURCE_OUTCOME_SCHEMA)
    frames = []
    for family in source_events.get_column("source_family").unique().to_list():
        family_events = source_events.filter(pl.col("source_family") == str(family))
        for horizon in SOURCE_OUTCOME_HORIZONS.get(str(family), (12,)):
            frames.append(_source_outcomes_for_horizon(family_events, bars, horizon=horizon))
    frames = [frame for frame in frames if not frame.is_empty()]
    if not frames:
        return pl.DataFrame(schema=SOURCE_OUTCOME_SCHEMA)
    return pl.concat(frames, how="vertical_relaxed")


def _source_outcomes_for_horizon(
    source_events: pl.DataFrame, bars: pl.DataFrame, *, horizon: int
) -> pl.DataFrame:
    if bars.is_empty() or not {"symbol", "timestamp", "close", "high", "low"} <= set(bars.columns):
        return source_events.with_columns(
            pl.lit(horizon).alias("outcome_horizon"),
            pl.lit(None, dtype=pl.Float64).alias("close_at_event"),
            pl.lit(None, dtype=pl.Float64).alias("future_close"),
            pl.lit(None, dtype=pl.Float64).alias("forward_return_pct"),
            pl.lit(None, dtype=pl.Float64).alias("forward_min_return_pct"),
            pl.lit(None, dtype=pl.Float64).alias("forward_max_return_pct"),
            pl.lit(None, dtype=pl.Float64).alias("path_range_pct"),
            pl.lit(None, dtype=pl.Float64).alias("tail_asymmetry_pct"),
            pl.lit(False).alias("outcome_available"),
            pl.lit("bar_history_missing").alias("outcome_reason"),
        ).select(*SOURCE_OUTCOME_SCHEMA.keys())
    window_high = pl.concat_list(
        [pl.col("high").shift(-offset).over("symbol") for offset in range(1, horizon + 1)]
    )
    window_low = pl.concat_list(
        [pl.col("low").shift(-offset).over("symbol") for offset in range(1, horizon + 1)]
    )
    bar_outcomes = (
        bars.sort("symbol", "timestamp")
        .with_columns(
            pl.col("close").shift(-horizon).over("symbol").alias("future_close"),
            window_high.list.max().alias("future_high"),
            window_low.list.min().alias("future_low"),
        )
        .select(
            "symbol",
            pl.col("timestamp").alias("bar_close_ms"),
            pl.col("close").alias("close_at_event"),
            "future_close",
            pct_change_expr("future_close", "close").alias("forward_return_pct"),
            pct_change_expr("future_low", "close").alias("forward_min_return_pct"),
            pct_change_expr("future_high", "close").alias("forward_max_return_pct"),
        )
        .with_columns(
            (pl.col("forward_max_return_pct") - pl.col("forward_min_return_pct")).alias(
                "path_range_pct"
            ),
            (pl.col("forward_max_return_pct") + pl.col("forward_min_return_pct")).alias(
                "tail_asymmetry_pct"
            ),
        )
    )
    aligned = (
        source_events.drop("aligned_bar_close_ms")
        .sort("symbol", "known_at_ms")
        .join_asof(
            bar_outcomes,
            left_on="known_at_ms",
            right_on="bar_close_ms",
            by="symbol",
            strategy="forward",
            check_sortedness=False,
        )
        .with_columns(
            pl.lit(horizon).alias("outcome_horizon"),
            pl.col("bar_close_ms").alias("aligned_bar_close_ms"),
            pl.col("forward_return_pct").is_not_null().alias("outcome_available"),
            pl.when(pl.col("forward_return_pct").is_not_null())
            .then(pl.lit("available"))
            .when(pl.col("known_at_ms").is_null())
            .then(pl.lit("event_not_time_serialized"))
            .otherwise(pl.lit("future_bar_missing"))
            .alias("outcome_reason"),
        )
    )
    return aligned.select(*SOURCE_OUTCOME_SCHEMA.keys())


def source_timeliness_frame(outcomes: pl.DataFrame) -> pl.DataFrame:
    schema = {
        "source_family": pl.String,
        "outcome_horizon": pl.Int64,
        "rows": pl.UInt32,
        "available_rows": pl.UInt32,
        "future_missing_rows": pl.UInt32,
        "availability_rate": pl.Float64,
        "min_known_at_ms": pl.Int64,
        "max_known_at_ms": pl.Int64,
        "min_aligned_bar_close_ms": pl.Int64,
        "max_aligned_bar_close_ms": pl.Int64,
        "timeliness_status": pl.String,
    }
    if outcomes.is_empty():
        return pl.DataFrame(schema=schema)
    return (
        outcomes.group_by("source_family", "outcome_horizon")
        .agg(
            pl.len().alias("rows"),
            pl.col("outcome_available").sum().alias("available_rows"),
            (pl.col("outcome_reason") == "future_bar_missing").sum().alias("future_missing_rows"),
            pl.col("outcome_available").mean().alias("availability_rate"),
            pl.col("known_at_ms").min().alias("min_known_at_ms"),
            pl.col("known_at_ms").max().alias("max_known_at_ms"),
            pl.col("aligned_bar_close_ms").min().alias("min_aligned_bar_close_ms"),
            pl.col("aligned_bar_close_ms").max().alias("max_aligned_bar_close_ms"),
        )
        .with_columns(
            pl.when(pl.col("availability_rate") >= 0.80)
            .then(pl.lit("usable_history"))
            .when(pl.col("availability_rate") > 0.0)
            .then(pl.lit("partial_history"))
            .otherwise(pl.lit("snapshot_or_future_only"))
            .alias("timeliness_status")
        )
        .select(*schema.keys())
        .sort(["source_family", "outcome_horizon"])
    )


def source_state_predictability_frame(
    outcomes: pl.DataFrame, *, return_threshold_pct: float
) -> pl.DataFrame:
    schema = {
        "source_family": pl.String,
        "source_state": pl.String,
        "outcome_horizon": pl.Int64,
        "observations": pl.UInt32,
        "symbol_count": pl.UInt32,
        "p_up": pl.Float64,
        "p_down": pl.Float64,
        "p_flat": pl.Float64,
        "baseline_p_up": pl.Float64,
        "baseline_p_down": pl.Float64,
        "baseline_p_flat": pl.Float64,
        "lift_up": pl.Float64,
        "lift_down": pl.Float64,
        "lift_flat": pl.Float64,
        "avg_forward_return_pct": pl.Float64,
        "median_forward_return_pct": pl.Float64,
        "q25_forward_return_pct": pl.Float64,
        "q75_forward_return_pct": pl.Float64,
        "avg_forward_min_return_pct": pl.Float64,
        "avg_forward_max_return_pct": pl.Float64,
        "outcome_entropy_bits": pl.Float64,
        "baseline_entropy_bits": pl.Float64,
        "information_gain_bits": pl.Float64,
        "dominant_outcome": pl.String,
        "statistical_direction": pl.String,
        "predictability_status": pl.String,
    }
    if outcomes.is_empty():
        return pl.DataFrame(schema=schema)
    scored = outcomes.filter(
        pl.col("outcome_available") & pl.col("source_state").is_not_null()
    ).with_columns(outcome_bucket_expr(return_threshold_pct).alias("outcome_bucket"))
    if scored.is_empty():
        return pl.DataFrame(schema=schema)
    baseline = (
        scored.group_by("source_family", "outcome_horizon")
        .agg(
            (pl.col("outcome_bucket") == "up").mean().alias("baseline_p_up"),
            (pl.col("outcome_bucket") == "down").mean().alias("baseline_p_down"),
            (pl.col("outcome_bucket") == "flat").mean().alias("baseline_p_flat"),
        )
        .with_columns(
            entropy_expr("baseline_p_up", "baseline_p_down", "baseline_p_flat").alias(
                "baseline_entropy_bits"
            )
        )
    )
    by_state = (
        scored.group_by("source_family", "source_state", "outcome_horizon")
        .agg(
            pl.len().alias("observations"),
            pl.col("symbol").n_unique().alias("symbol_count"),
            (pl.col("outcome_bucket") == "up").mean().alias("p_up"),
            (pl.col("outcome_bucket") == "down").mean().alias("p_down"),
            (pl.col("outcome_bucket") == "flat").mean().alias("p_flat"),
            pl.col("forward_return_pct").mean().alias("avg_forward_return_pct"),
            pl.col("forward_return_pct").median().alias("median_forward_return_pct"),
            pl.col("forward_return_pct").quantile(0.25).alias("q25_forward_return_pct"),
            pl.col("forward_return_pct").quantile(0.75).alias("q75_forward_return_pct"),
            pl.col("forward_min_return_pct").mean().alias("avg_forward_min_return_pct"),
            pl.col("forward_max_return_pct").mean().alias("avg_forward_max_return_pct"),
        )
        .with_columns(entropy_expr("p_up", "p_down", "p_flat").alias("outcome_entropy_bits"))
    )
    return (
        by_state.join(baseline, on=("source_family", "outcome_horizon"), how="left")
        .with_columns(
            (pl.col("p_up") - pl.col("baseline_p_up")).alias("lift_up"),
            (pl.col("p_down") - pl.col("baseline_p_down")).alias("lift_down"),
            (pl.col("p_flat") - pl.col("baseline_p_flat")).alias("lift_flat"),
            (pl.col("baseline_entropy_bits") - pl.col("outcome_entropy_bits")).alias(
                "information_gain_bits"
            ),
            pl.when((pl.col("p_up") >= pl.col("p_down")) & (pl.col("p_up") >= pl.col("p_flat")))
            .then(pl.lit("up"))
            .when((pl.col("p_down") >= pl.col("p_up")) & (pl.col("p_down") >= pl.col("p_flat")))
            .then(pl.lit("down"))
            .otherwise(pl.lit("flat"))
            .alias("dominant_outcome"),
        )
        .with_columns(
            pl.when(pl.col("dominant_outcome") == "up")
            .then(pl.lit("bullish"))
            .when(pl.col("dominant_outcome") == "down")
            .then(pl.lit("bearish"))
            .otherwise(pl.lit("neutral"))
            .alias("statistical_direction"),
            pl.when(
                (pl.col("observations") >= 100)
                & (pl.col("symbol_count") >= 20)
                & (pl.col("information_gain_bits") > 0.0)
            )
            .then(pl.lit("usable_predictive_sample"))
            .otherwise(pl.lit("insufficient_predictive_sample"))
            .alias("predictability_status"),
        )
        .select(*schema.keys())
        .sort(["source_family", "source_state", "outcome_horizon"])
    )
