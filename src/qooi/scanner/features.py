"""Continuous known-at-close feature extraction.

All features are Polars-native, no future data. Produces per-(symbol, timestamp)
rows consumed by evidence.potential_observation_frame.
"""

from __future__ import annotations

import polars as pl


def extract_continuous_features(
    bars: dict[tuple[str, str], pl.DataFrame],
    state_frames: dict[tuple[str, str], pl.DataFrame],
    source_frames: dict[str, pl.DataFrame],
    *,
    decision_timeframe: str = "1H",
) -> pl.DataFrame:
    """Extract continuous features from OHLCV and source frames.

    Returns one frame with columns: symbol, timestamp,
    atr_percentile, range_width_atr, imbalance_value, buy_sell_ratio,
    oi_delta, funding_rate, taker_buy_sell_ratio, return_1bar, return_4bar,
    return_24bar, vol_anomaly, spread_bps, close_to_range_high_ratio.
    Source event/snapshot rows are aligned as known-at-close values by symbol
    with backward as-of joins; raw source timestamps are not overwritten.
    """
    kline_features = _kline_continuous_features(bars, state_frames, decision_timeframe)
    decision_keys = (
        kline_features.select("symbol", "timestamp")
        if not kline_features.is_empty()
        else None
    )
    source_features = _source_continuous_features(source_frames, decision_keys=decision_keys)

    if kline_features.is_empty():
        return source_features
    if source_features.is_empty():
        return kline_features

    return kline_features.join(
        source_features, on=["symbol", "timestamp"], how="left"
    ).select(kline_features.columns + [c for c in source_features.columns if c not in ("symbol", "timestamp")])


def _kline_continuous_features(
    bars: dict[tuple[str, str], pl.DataFrame],
    state_frames: dict[tuple[str, str], pl.DataFrame],
    decision_timeframe: str,
) -> pl.DataFrame:
    """Extract continuous features from kline data: ATR, returns, vol anomaly."""
    frames: list[pl.DataFrame] = []
    for (symbol, tf), frame in bars.items():
        if tf != decision_timeframe:
            continue
        if frame.is_empty():
            continue
        state = state_frames.get((symbol, tf))
        if state is None or state.is_empty():
            continue

        required = {"timestamp", "open", "high", "low", "close", "volume"}
        if not required.issubset(frame.columns):
            continue

        work = frame.select(
            "timestamp", "open", "high", "low", "close", "volume"
        ).sort("timestamp")

        # ATR percentile from classifier state frame (already computed)
        atr_pct = None
        if "atr_percentile_100" in state.columns:
            atr_pct = state.select("timestamp", pl.col("atr_percentile_100").alias("atr_percentile")).sort("timestamp")

        # Range width from classifier state frame
        range_width = None
        if "range_width_atr" in state.columns:
            range_width = state.select("timestamp", pl.col("range_width_atr")).sort("timestamp")

        # Returns
        work = work.with_columns(
            ((pl.col("close") - pl.col("close").shift(1)) / pl.col("close").shift(1) * 100.0).alias("return_1bar"),
            ((pl.col("close") - pl.col("close").shift(4)) / pl.col("close").shift(4) * 100.0).alias("return_4bar"),
            ((pl.col("close") - pl.col("close").shift(24)) / pl.col("close").shift(24) * 100.0).alias("return_24bar"),
        )

        # Volume anomaly
        work = work.with_columns(
            (pl.col("volume") / pl.col("volume").rolling_mean(20, min_samples=5)).alias("vol_anomaly"),
        )

        # Close-to-range ratio (from OHLCV directly: 48-bar range)
        work = work.with_columns(
            pl.col("high").shift(1).rolling_max(48).alias("range_high_48"),
            pl.col("low").shift(1).rolling_min(48).alias("range_low_48"),
        ).with_columns(
            (
                (pl.col("close") - pl.col("range_low_48"))
                / (pl.col("range_high_48") - pl.col("range_low_48"))
            ).alias("close_to_range_high_ratio"),
        )

        cols = ["symbol", "timestamp", "return_1bar", "return_4bar", "return_24bar", "vol_anomaly", "close_to_range_high_ratio"]
        out = work.select(pl.lit(symbol).alias("symbol"), *cols[1:])

        if atr_pct is not None:
            out = out.join(atr_pct, on="timestamp", how="left")
        else:
            out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("atr_percentile"))

        if range_width is not None:
            out = out.join(range_width, on="timestamp", how="left")
        else:
            out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("range_width_atr"))

        frames.append(out)

    if not frames:
        return pl.DataFrame(schema={
            "symbol": pl.String, "timestamp": pl.Int64,
            "return_1bar": pl.Float64, "return_4bar": pl.Float64, "return_24bar": pl.Float64,
            "vol_anomaly": pl.Float64, "close_to_range_high_ratio": pl.Float64,
            "atr_percentile": pl.Float64, "range_width_atr": pl.Float64,
        })
    return pl.concat(frames, how="vertical_relaxed")


def _source_continuous_features(
    source_frames: dict[str, pl.DataFrame],
    *,
    decision_keys: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Extract source features and optionally align them to decision-bar closes."""
    all_frames: list[pl.DataFrame] = []

    for family, frame in source_frames.items():
        if frame.is_empty() or "symbol" not in frame.columns or "timestamp" not in frame.columns:
            continue

        work = frame.select("symbol", "timestamp").unique()

        if family == "books":
            work = _book_continuous(work, frame)
        elif family == "trades":
            work = _trade_continuous(work, frame)
        elif family == "funding":
            work = _funding_continuous(work, frame)
        elif family == "open_interest":
            work = _oi_continuous(work, frame)
        elif family == "taker_volume":
            work = _taker_continuous(work, frame)
        elif family == "long_short_ratios":
            work = _lsr_continuous(work, frame)
        else:
            continue

        if decision_keys is not None:
            work = _align_source_family_to_decision_keys(family, work, decision_keys)
            if work.is_empty():
                continue

        all_frames.append(work)

    if not all_frames:
        return pl.DataFrame()

    result = pl.concat(
        [frame.select("symbol", "timestamp") for frame in all_frames],
        how="vertical_relaxed",
    ).unique()
    for frame in all_frames:
        value_cols = [c for c in frame.columns if c not in ("symbol", "timestamp")]
        if value_cols:
            result = result.join(
                frame.select(["symbol", "timestamp"] + value_cols),
                on=["symbol", "timestamp"],
                how="left",
                coalesce=True,
            )

    return result


def _align_source_family_to_decision_keys(
    family: str,
    frame: pl.DataFrame,
    decision_keys: pl.DataFrame,
) -> pl.DataFrame:
    value_cols = [col for col in frame.columns if col not in ("symbol", "timestamp")]
    if not value_cols or decision_keys.is_empty():
        return pl.DataFrame()

    prefix = _source_family_prefix(family)
    events = (
        frame.select("symbol", pl.col("timestamp").alias("source_timestamp_ms"), *value_cols)
        .drop_nulls(subset=value_cols)
        .sort(["symbol", "source_timestamp_ms"])
    )
    if events.is_empty():
        return pl.DataFrame()

    aligned = (
        decision_keys.select("symbol", "timestamp")
        .unique()
        .sort(["symbol", "timestamp"])
        .join_asof(
            events,
            left_on="timestamp",
            right_on="source_timestamp_ms",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
    )
    if aligned.is_empty():
        return aligned

    return aligned.with_columns(
        (pl.col("timestamp") - pl.col("source_timestamp_ms")).alias(f"{prefix}_age_ms")
    ).drop("source_timestamp_ms")


def _source_family_prefix(family: str) -> str:
    return {
        "books": "book",
        "trades": "trade",
        "funding": "funding",
        "open_interest": "oi",
        "taker_volume": "taker",
        "long_short_ratios": "lsr",
    }.get(family, family)


def _first_float_col(frame: pl.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in frame.columns:
            return col
    return None


def _book_continuous(work: pl.DataFrame, frame: pl.DataFrame) -> pl.DataFrame:
    imbalance_col = _first_float_col(frame, ("ob_imbalance_25", "ob_imbalance_10", "ob_imbalance_5"))
    bid_col = _first_float_col(frame, ("ob_bid_price",))
    ask_col = _first_float_col(frame, ("ob_ask_price",))

    needed = [c for c in (imbalance_col, bid_col, ask_col) if c]
    if not needed:
        return work

    work = work.join(frame.select(["symbol", "timestamp"] + needed), on=["symbol", "timestamp"], how="left")
    exprs = []

    if imbalance_col:
        exprs.append(pl.col(imbalance_col).cast(pl.Float64).alias("imbalance_value"))
    if bid_col and ask_col:
        mid = (pl.col(bid_col).cast(pl.Float64) + pl.col(ask_col).cast(pl.Float64)) / 2.0
        exprs.append(((pl.col(ask_col) - pl.col(bid_col)) / mid * 10000.0).alias("spread_bps"))
        if "close" in frame.columns:
            work = work.join(frame.select("symbol", "timestamp", pl.col("close").cast(pl.Float64)), on=["symbol", "timestamp"], how="left")
            exprs.append(((pl.col("close") - mid) / mid * 100.0).alias("close_to_mid_pct"))

    work = work.with_columns(exprs)
    keep = {"symbol", "timestamp", "imbalance_value", "spread_bps", "close_to_mid_pct"}
    return work.select([c for c in work.columns if c in keep])


def _trade_continuous(work: pl.DataFrame, frame: pl.DataFrame) -> pl.DataFrame:
    # Aggregate trade frame: buy vs sell notional
    if "side" not in frame.columns:
        return work

    value_col = _first_float_col(frame, ("notional_usd", "notional", "size"))
    if not value_col:
        return work

    agg = (
        frame.group_by(["symbol", "timestamp"])
        .agg(
            pl.col(value_col).filter(pl.col("side") == "buy").sum().alias("buy_notional"),
            pl.col(value_col).filter(pl.col("side") == "sell").sum().alias("sell_notional"),
        )
        .with_columns(
            (pl.col("buy_notional") / pl.when(pl.col("sell_notional") > 0).then(pl.col("sell_notional")).otherwise(None)).alias("buy_sell_ratio")
        )
    )
    return work.join(agg.select("symbol", "timestamp", "buy_sell_ratio"), on=["symbol", "timestamp"], how="left")


def _funding_continuous(work: pl.DataFrame, frame: pl.DataFrame) -> pl.DataFrame:
    if "funding_rate" in frame.columns:
        return work.join(
            frame.select("symbol", "timestamp", pl.col("funding_rate").cast(pl.Float64).alias("funding_rate")),
            on=["symbol", "timestamp"], how="left"
        )
    return work


def _oi_continuous(work: pl.DataFrame, frame: pl.DataFrame) -> pl.DataFrame:
    value_col = _first_float_col(frame, ("open_interest_usd", "open_interest"))
    if not value_col:
        return work

    oi = frame.select("symbol", "timestamp", pl.col(value_col).cast(pl.Float64).alias("oi_value")).sort(["symbol", "timestamp"])
    oi = oi.with_columns(
        (pl.col("oi_value") - pl.col("oi_value").shift(1).over("symbol")).alias("oi_delta")
    )
    return work.join(oi.select("symbol", "timestamp", "oi_delta"), on=["symbol", "timestamp"], how="left")


def _taker_continuous(work: pl.DataFrame, frame: pl.DataFrame) -> pl.DataFrame:
    if "taker_buy_volume" not in frame.columns or "taker_sell_volume" not in frame.columns:
        return work

    taker = frame.select(
        "symbol",
        "timestamp",
        pl.col("taker_buy_volume").cast(pl.Float64),
        pl.col("taker_sell_volume").cast(pl.Float64),
    ).with_columns(
        (pl.col("taker_buy_volume") / pl.when(pl.col("taker_sell_volume") > 0).then(pl.col("taker_sell_volume")).otherwise(None)).alias("taker_buy_sell_ratio")
    )
    return work.join(taker.select("symbol", "timestamp", "taker_buy_sell_ratio"), on=["symbol", "timestamp"], how="left")


def _lsr_continuous(work: pl.DataFrame, frame: pl.DataFrame) -> pl.DataFrame:
    ratio_col = _first_float_col(frame, (
        "top_trader_long_short_position_ratio",
        "top_trader_long_short_account_ratio",
        "long_short_account_ratio",
    ))
    if ratio_col:
        return work.join(
            frame.select("symbol", "timestamp", pl.col(ratio_col).cast(pl.Float64).alias("long_short_ratio")),
            on=["symbol", "timestamp"], how="left"
        )
    return work
