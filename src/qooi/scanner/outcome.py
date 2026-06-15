"""Future/path/source outcome products for the scanner."""

from __future__ import annotations

import polars as pl

from qooi.scanner import entropy_expr, outcome_bucket_expr, pct_change_expr

KLINE_PATH_HISTORY_SCHEMA = {
    "symbol": pl.String,
    "timeframe": pl.String,
    "bar_close_ms": pl.Int64,
    "close": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "direction_hint": pl.String,
    "regime_state": pl.String,
    "structure_state": pl.String,
    "core_context": pl.String,
    "state_age_bars": pl.UInt32,
    "state_age_bucket": pl.String,
    "event_state": pl.String,
    "event_age_bars": pl.UInt32,
    "event_age_bucket": pl.String,
    "range_state": pl.String,
    "vol_state": pl.String,
    "state_changed": pl.Boolean,
    "event_changed": pl.Boolean,
    "fresh_event": pl.Boolean,
    "transition_kind": pl.String,
    "extreme_range": pl.String,
    "extreme_vol": pl.String,
    "compression_state": pl.String,
    "expansion_state": pl.String,
    "transition_path": pl.String,
    "full_context": pl.String,
}

def kline_path_history_frame(
    config,
    state_frames: dict[tuple[str, str], pl.DataFrame],
    bar_frames: dict[tuple[str, str], pl.DataFrame],
) -> pl.DataFrame:
    histories = []
    for timeframe in config.timeframes:
        for (symbol, frame_timeframe), frame in state_frames.items():
            if frame_timeframe != timeframe or frame.is_empty():
                continue
            rows = _state_rows_with_ohlc(
                frame, bar_frames.get((symbol, frame_timeframe), pl.DataFrame())
            )
            if rows.is_empty() or rows.get_column("missing_flag").all():
                continue
            histories.append(kline_path_rows(rows, config.transition.ngram_length))
    if not histories:
        return pl.DataFrame(schema=KLINE_PATH_HISTORY_SCHEMA)
    return pl.concat(histories, how="vertical_relaxed").select(*KLINE_PATH_HISTORY_SCHEMA.keys())


def _state_rows_with_ohlc(state_rows: pl.DataFrame, bars: pl.DataFrame) -> pl.DataFrame:
    """Attach raw OHLC values to known-at-close classifier state rows."""
    if bars.is_empty() or not {"timestamp", "close", "high", "low"} <= set(bars.columns):
        return state_rows.with_columns(
            _optional_float_col(state_rows, "close"),
            _optional_float_col(state_rows, "high"),
            _optional_float_col(state_rows, "low"),
        )
    return state_rows.join(
        bars.select(
            "timestamp",
            pl.col("close").cast(pl.Float64, strict=False),
            pl.col("high").cast(pl.Float64, strict=False),
            pl.col("low").cast(pl.Float64, strict=False),
        ),
        on="timestamp",
        how="left",
        coalesce=True,
    )


def kline_path_rows(rows: pl.DataFrame, ngram_length: int) -> pl.DataFrame:
    ngram_length = max(2, ngram_length)
    group_keys = ("symbol", "scale") if rows.get_column("scale").n_unique() > 1 else ("symbol",)
    state_changed = (
        pl.col("state_key") != pl.col("state_key").shift(1).over(*group_keys)
    ).fill_null(True)
    event_changed = (
        pl.col("context_event") != pl.col("context_event").shift(1).over(*group_keys)
    ).fill_null(True)
    return (
        rows.sort("symbol", "timestamp")
        .with_columns(
            pl.col("state_key").str.split("|").list.get(0, null_on_oob=True).alias("regime_state"),
            pl.col("state_key")
            .str.split("|")
            .list.get(1, null_on_oob=True)
            .alias("structure_state"),
            pl.col("state_key").str.split("|").list.get(2, null_on_oob=True).alias("range_state"),
            pl.col("state_key").str.split("|").list.get(3, null_on_oob=True).alias("vol_state"),
            pl.concat_str("state_key", "context_event", separator="|").alias("full_context"),
            pl.concat_str(
                [
                    pl.col("state_key").shift(offset).over(*group_keys)
                    for offset in range(ngram_length - 1, 0, -1)
                ]
                + [pl.col("state_key")],
                separator=" -> ",
            ).alias("transition_path"),
            state_changed.alias("state_changed"),
            event_changed.alias("event_changed"),
            state_changed.cast(pl.Int64).cum_sum().over(*group_keys).alias("state_run"),
            event_changed.cast(pl.Int64).cum_sum().over(*group_keys).alias("event_run"),
        )
        .with_columns(
            pl.col("direction_hint").fill_null("unknown_direction"),
            pl.col("regime_state").fill_null("unknown_regime"),
            pl.col("structure_state").fill_null("unknown_structure"),
            pl.col("range_state").fill_null("range_unknown"),
            pl.col("vol_state").fill_null("vol_unknown"),
            pl.col("context_event").fill_null("none_event").alias("event_state"),
        )
        .with_columns(
            pl.concat_str("regime_state", "structure_state", separator="|").alias("core_context"),
            (~pl.col("event_state").str.starts_with("none")).alias("fresh_event"),
            pl.when(pl.col("state_changed") & pl.col("event_changed"))
            .then(pl.lit("state_and_event_transition"))
            .when(pl.col("state_changed"))
            .then(pl.lit("state_transition"))
            .when(pl.col("event_changed") & ~pl.col("event_state").str.starts_with("none"))
            .then(pl.lit("event_trigger"))
            .otherwise(pl.lit("same_context"))
            .alias("transition_kind"),
            pl.when(pl.col("range_state") == "range_tight")
            .then(pl.lit("tight_extreme"))
            .when(pl.col("range_state") == "range_wide")
            .then(pl.lit("wide_extreme"))
            .otherwise(pl.lit("normal_range"))
            .alias("extreme_range"),
            pl.when(pl.col("vol_state") == "vol_low")
            .then(pl.lit("low_vol_extreme"))
            .when(pl.col("vol_state") == "vol_high")
            .then(pl.lit("high_vol_extreme"))
            .when(pl.col("vol_state") == "vol_unknown")
            .then(pl.lit("vol_unknown"))
            .otherwise(pl.lit("normal_vol"))
            .alias("extreme_vol"),
            pl.when(pl.col("range_state") == "range_tight")
            .then(pl.lit("compressed"))
            .otherwise(pl.lit("not_compressed"))
            .alias("compression_state"),
            pl.when(pl.col("range_state") == "range_wide")
            .then(pl.lit("expanded"))
            .otherwise(pl.lit("not_expanded"))
            .alias("expansion_state"),
        )
        .with_columns(
            pl.cum_count("state_key")
            .over("symbol", "scale", "state_run")
            .cast(pl.UInt32)
            .alias("state_age_bars"),
            pl.cum_count("context_event")
            .over("symbol", "scale", "event_run")
            .cast(pl.UInt32)
            .alias("event_age_bars"),
        )
        .with_columns(
            _age_bucket_expr("state_age_bars").alias("state_age_bucket"),
            _age_bucket_expr("event_age_bars").alias("event_age_bucket"),
        )
        .select(
            pl.col("symbol").cast(pl.String),
            pl.col("scale").alias("timeframe"),
            pl.col("timestamp").alias("bar_close_ms"),
            _optional_float_col(rows, "close"),
            _optional_float_col(rows, "high"),
            _optional_float_col(rows, "low"),
            "direction_hint",
            "regime_state",
            "structure_state",
            "core_context",
            "state_age_bars",
            "state_age_bucket",
            "event_state",
            "event_age_bars",
            "event_age_bucket",
            "range_state",
            "vol_state",
            "state_changed",
            "event_changed",
            "fresh_event",
            "transition_kind",
            "extreme_range",
            "extreme_vol",
            "compression_state",
            "expansion_state",
            "transition_path",
            "full_context",
        )
    )


def _optional_float_col(frame: pl.DataFrame, column: str) -> pl.Expr:
    if column in frame.columns:
        return pl.col(column).cast(pl.Float64, strict=False)
    return pl.lit(None, dtype=pl.Float64).alias(column)


def _age_bucket_expr(column: str) -> pl.Expr:
    age = pl.col(column).cast(pl.UInt32, strict=False)
    return (
        pl.when(age <= 1)
        .then(pl.lit("new"))
        .when(age <= 3)
        .then(pl.lit("young"))
        .when(age <= 12)
        .then(pl.lit("mature"))
        .otherwise(pl.lit("stale"))
    )


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



REALIZED_TRANSITION_SCHEMA = {
    "symbol": pl.String,
    "timeframe": pl.String,
    "bar_close_ms": pl.Int64,
    "outcome_horizon": pl.Int64,
    "terminal_direction": pl.String,
    "terminal_regime_state": pl.String,
    "terminal_structure_state": pl.String,
    "terminal_core_context": pl.String,
    "terminal_transition_kind": pl.String,
    "direction_changed": pl.Boolean,
    "regime_changed": pl.Boolean,
    "structure_changed": pl.Boolean,
    "core_context_changed": pl.Boolean,
    "event_fired": pl.Boolean,
    "returned_to_origin": pl.Boolean,
    "time_to_direction_change_bars": pl.Int64,
    "time_to_core_change_bars": pl.Int64,
    "transition_count": pl.Int64,
    "forward_return_pct": pl.Float64,
    "forward_min_return_pct": pl.Float64,
    "forward_max_return_pct": pl.Float64,
    "path_range_pct": pl.Float64,
    "tail_asymmetry_pct": pl.Float64,
    "time_to_max_bar": pl.Int64,
    "time_to_min_bar": pl.Int64,
    "close_retention_ratio": pl.Float64,
    "post_max_drawdown_pct": pl.Float64,
    "post_min_rebound_pct": pl.Float64,
    "path_efficiency": pl.Float64,
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


def _pct_change_value(future: object, current: object) -> float | None:
    if future is None or current in (None, 0):
        return None
    return (float(future) - float(current)) / float(current) * 100.0


def _transition_count(values: list[object | None], origin: object | None) -> int:
    count = 0
    previous = origin
    for value in values:
        if value is None:
            continue
        if value != previous:
            count += 1
        previous = value
    return count


def _first_change_offset(values: list[object | None], origin: object | None) -> int | None:
    for offset, value in enumerate(values, start=1):
        if value is not None and value != origin:
            return offset
    return None


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def realized_transition_frame(
    kline_history: pl.DataFrame, horizons: tuple[int, ...]
) -> pl.DataFrame:
    if kline_history.is_empty() or not horizons:
        return pl.DataFrame(schema=REALIZED_TRANSITION_SCHEMA)
    required = {"symbol", "timeframe", "bar_close_ms", "close", "high", "low"}
    if not required <= set(kline_history.columns):
        return pl.DataFrame(schema=REALIZED_TRANSITION_SCHEMA)

    rows: list[dict[str, object]] = []
    sorted_history = kline_history.sort("symbol", "timeframe", "bar_close_ms")
    for key, group in sorted_history.group_by("symbol", "timeframe", maintain_order=True):
        symbol, timeframe = key if isinstance(key, tuple) else (key, None)
        records = group.to_dicts()
        for index, row in enumerate(records):
            close = row.get("close")
            for horizon in sorted({int(h) for h in horizons if int(h) > 0}):
                future_rows = records[index + 1 : index + horizon + 1]
                terminal = records[index + horizon] if index + horizon < len(records) else None
                future_direction = [r.get("direction_hint") for r in future_rows]
                future_core = [r.get("core_context") for r in future_rows]
                future_event = [r.get("event_state") for r in future_rows]
                terminal_close = terminal.get("close") if terminal else None
                forward_return = _pct_change_value(terminal_close, close)
                highs = [float(r["high"]) for r in future_rows if r.get("high") is not None]
                lows = [float(r["low"]) for r in future_rows if r.get("low") is not None]
                future_high = max(highs) if highs else None
                future_low = min(lows) if lows else None
                forward_max = _pct_change_value(future_high, close)
                forward_min = _pct_change_value(future_low, close)
                path_range = (
                    forward_max - forward_min
                    if forward_max is not None and forward_min is not None
                    else None
                )
                time_to_max = (
                    next(
                        (
                            offset
                            for offset, future in enumerate(future_rows, start=1)
                            if future.get("high") is not None
                            and float(future["high"]) == future_high
                        ),
                        None,
                    )
                    if future_high is not None
                    else None
                )
                time_to_min = (
                    next(
                        (
                            offset
                            for offset, future in enumerate(future_rows, start=1)
                            if future.get("low") is not None and float(future["low"]) == future_low
                        ),
                        None,
                    )
                    if future_low is not None
                    else None
                )
                terminal_direction = terminal.get("direction_hint") if terminal else None
                terminal_core = terminal.get("core_context") if terminal else None
                terminal_regime = terminal.get("regime_state") if terminal else None
                terminal_structure = terminal.get("structure_state") if terminal else None
                terminal_transition = terminal.get("transition_kind") if terminal else None
                origin_core = row.get("core_context")
                direction_change_time = _first_change_offset(
                    future_direction, row.get("direction_hint")
                )
                core_change_time = _first_change_offset(future_core, origin_core)
                close_retention = (
                    _safe_div(forward_return, forward_max)
                    if forward_return is not None and forward_return >= 0
                    else _safe_div(forward_return, forward_min)
                )
                post_max_drawdown = (
                    forward_max - forward_return
                    if forward_max is not None and forward_return is not None
                    else None
                )
                post_min_rebound = (
                    forward_return - forward_min
                    if forward_min is not None and forward_return is not None
                    else None
                )
                out = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "bar_close_ms": row.get("bar_close_ms"),
                    "outcome_horizon": horizon,
                    "terminal_direction": terminal_direction,
                    "terminal_regime_state": terminal_regime,
                    "terminal_structure_state": terminal_structure,
                    "terminal_core_context": terminal_core,
                    "terminal_transition_kind": terminal_transition,
                    "direction_changed": (
                        terminal_direction != row.get("direction_hint") if terminal else None
                    ),
                    "regime_changed": (
                        terminal_regime != row.get("regime_state") if terminal else None
                    ),
                    "structure_changed": (
                        terminal_structure != row.get("structure_state") if terminal else None
                    ),
                    "core_context_changed": terminal_core != origin_core if terminal else None,
                    "event_fired": any(
                        event is not None and not str(event).startswith("none")
                        for event in future_event
                    ),
                    "returned_to_origin": bool(
                        terminal_core == origin_core and core_change_time is not None
                    )
                    if terminal
                    else None,
                    "time_to_direction_change_bars": direction_change_time,
                    "time_to_core_change_bars": core_change_time,
                    "transition_count": _transition_count(future_core, origin_core),
                    "forward_return_pct": forward_return,
                    "forward_min_return_pct": forward_min,
                    "forward_max_return_pct": forward_max,
                    "path_range_pct": path_range,
                    "tail_asymmetry_pct": (
                        forward_max + forward_min
                        if forward_max is not None and forward_min is not None
                        else None
                    ),
                    "time_to_max_bar": time_to_max,
                    "time_to_min_bar": time_to_min,
                    "close_retention_ratio": close_retention,
                    "post_max_drawdown_pct": post_max_drawdown,
                    "post_min_rebound_pct": post_min_rebound,
                    "path_efficiency": (
                        abs(forward_return) / path_range
                        if forward_return is not None and path_range not in (None, 0)
                        else None
                    ),
                }
                rows.append(out)
    if not rows:
        return pl.DataFrame(schema=REALIZED_TRANSITION_SCHEMA)
    return pl.DataFrame(rows, schema=REALIZED_TRANSITION_SCHEMA).select(
        *REALIZED_TRANSITION_SCHEMA.keys()
    )


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
    return pl.concat(frames, how="vertical_relaxed").select(*SOURCE_OUTCOME_SCHEMA.keys())


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


def potential_outcome_frame(
    observations: pl.DataFrame,
    source_outcomes: pl.DataFrame,
    realized_transitions: pl.DataFrame,
    *,
    return_threshold_pct: float,
) -> pl.DataFrame:
    market = (
        observations.unique(
            subset=["symbol", "decision_timeframe", "decision_bar_close_ms"], keep="first"
        )
        .join(
            realized_transitions,
            left_on=("symbol", "decision_timeframe", "decision_bar_close_ms"),
            right_on=("symbol", "timeframe", "bar_close_ms"),
            how="inner",
        )
        .filter(pl.col("terminal_core_context").is_not_null())
        .with_columns(
            pl.lit(None, dtype=pl.String).alias("source_family"),
            pl.lit(None, dtype=pl.String).alias("source_state"),
            pl.lit(None, dtype=pl.String).alias("source_direction"),
            pl.lit(None, dtype=pl.Int64).alias("source_known_at_ms"),
            pl.lit(None, dtype=pl.Int64).alias("source_age_ms"),
            pl.lit("missing").alias("source_freshness"),
            pl.lit("source_missing").alias("source_market_alignment"),
            _terminal_direction_bucket_expr().alias("outcome_bucket"),
            (pl.col("forward_max_return_pct") >= return_threshold_pct).alias("tail_up"),
            (pl.col("forward_min_return_pct") <= -return_threshold_pct).alias("tail_down"),
        )
    )
    frames = [market] if not market.is_empty() else []
    if not source_outcomes.is_empty():
        scored = source_outcomes.filter(
            pl.col("outcome_available") & pl.col("source_state").is_not_null()
        ).with_columns(
            outcome_bucket_expr(return_threshold_pct).alias("outcome_bucket"),
            (pl.col("forward_max_return_pct") >= return_threshold_pct).alias("tail_up"),
            (pl.col("forward_min_return_pct") <= -return_threshold_pct).alias("tail_down"),
        )
        source_joined = observations.join(
            scored.select(
                "symbol",
                "source_family",
                "source_state",
                "known_at_ms",
                "outcome_horizon",
                "forward_return_pct",
                "forward_min_return_pct",
                "forward_max_return_pct",
                "path_range_pct",
                "tail_asymmetry_pct",
                "outcome_bucket",
                "tail_up",
                "tail_down",
            ),
            left_on=("symbol", "source_family", "source_state", "source_known_at_ms"),
            right_on=("symbol", "source_family", "source_state", "known_at_ms"),
            how="inner",
        )
        if not source_joined.is_empty():
            frames.append(
                source_joined.join(
                    realized_transitions,
                    left_on=(
                        "symbol",
                        "decision_timeframe",
                        "decision_bar_close_ms",
                        "outcome_horizon",
                    ),
                    right_on=("symbol", "timeframe", "bar_close_ms", "outcome_horizon"),
                    how="inner",
                ).filter(pl.col("terminal_core_context").is_not_null())
            )
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def _terminal_direction_bucket_expr() -> pl.Expr:
    return (
        pl.when(pl.col("terminal_direction").str.contains("bull|up"))
        .then(pl.lit("up"))
        .when(pl.col("terminal_direction").str.contains("bear|down"))
        .then(pl.lit("down"))
        .otherwise(pl.lit("flat"))
    )


__all__ = [
    "KLINE_PATH_HISTORY_SCHEMA",
    "REALIZED_TRANSITION_SCHEMA",
    "SOURCE_EVENT_SCHEMA",
    "SOURCE_OUTCOME_SCHEMA",
    "SOURCE_OUTCOME_HORIZONS",
    "kline_path_history_frame",
    "kline_path_rows",
    "potential_outcome_frame",
    "realized_transition_frame",
    "source_events_frame",
    "source_outcomes_frame",
    "source_state_predictability_frame",
    "source_timeliness_frame",
]

