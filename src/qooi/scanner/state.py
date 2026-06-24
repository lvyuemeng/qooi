"""Known-at-close scanner state, classifiers, features, and observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

from qooi.core.evaluate import format_table


class StructureState:
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    RANGE = "range"
    UNKNOWN = "unknown"


class MarketStage:
    WARMUP = "warmup"
    DATA_ERROR = "data_error"
    MARKUP = "markup"
    MARKDOWN = "markdown"
    ACCUMULATION = "accumulation"
    DISTRIBUTION_OR_REVERSAL = "distribution_or_reversal"
    RANGE = StructureState.RANGE
    TREND_CONTINUATION = "trend_continuation"
    WIDE_RANGE = "wide_range"
    TRANSITION = "transition"
    UNKNOWN = StructureState.UNKNOWN


class ClassifierColumn:
    STRUCTURE_TREND_STATE = "structure_trend_state"
    MARKET_STAGE = "market_stage"
    RANGE_WIDTH_ATR = "range_width_atr"


StateDirection = Literal["bullish", "bearish", "neutral", "blocked", "missing"]
KLINE_REQUIRED_COLUMNS = ("symbol", "timestamp", "open", "high", "low", "close", "volume")
KLINE_MIN_ROWS = 20

STATE_FRAME_COLUMNS = (
    "symbol",
    "timestamp",
    "source_family",
    "scale",
    "state_key",
    "context_event",
    "direction_hint",
    "quality_weight",
    "missing_flag",
    "stale_flag",
)
STATE_FRAME_SCHEMA = {
    "symbol": pl.String,
    "timestamp": pl.Int64,
    "source_family": pl.String,
    "scale": pl.String,
    "state_key": pl.String,
    "context_event": pl.String,
    "direction_hint": pl.String,
    "quality_weight": pl.Float64,
    "missing_flag": pl.Boolean,
    "stale_flag": pl.Boolean,
}

CLASSIFIER_HEALTH_SCHEMA = {
    "artifact": pl.Utf8,
    "label": pl.Utf8,
    "health_check": pl.Utf8,
    "status": pl.Utf8,
    "value": pl.Float64,
    "threshold": pl.Float64,
    "reason": pl.Utf8,
}


@dataclass(frozen=True)
class ClassifierHealthResult:
    frame: pl.DataFrame
    text: str


@dataclass(frozen=True)
class KlineClassifier:
    scale: str

    def classify(self, frame: pl.DataFrame) -> pl.DataFrame:
        if "volume" not in frame.columns and "vol" in frame.columns:
            frame = frame.with_columns(pl.col("vol").alias("volume"))
        missing = [column for column in KLINE_REQUIRED_COLUMNS if column not in frame.columns]
        if missing or frame.height < KLINE_MIN_ROWS:
            symbol = _state_frame_symbol(frame)
            state = _missing_state_frame(
                symbol=symbol,
                family="kline",
                scale=self.scale,
                reason=";".join(f"{column}_missing" for column in missing) or "kline_rows_missing",
            )
            return validate_state_frame(state)
        state = classify_states(frame, self.scale)
        return validate_state_frame(state)


def classifier_health(frame: pl.DataFrame, *, label: str = "") -> ClassifierHealthResult:
    required = ("structure_trend_state", "market_stage", "structure_reason", "stage_unknown_reason")
    present = [column for column in required if column in frame.columns]
    rows = [
        {
            "artifact": "classifier-health",
            "label": label,
            "health_check": "required_classifier_columns",
            "status": "pass" if len(present) == len(required) else "fail",
            "value": len(present) / max(len(required), 1) * 100.0,
            "threshold": 100.0,
            "reason": f"present={len(present)}/{len(required)} rows={frame.height}",
        }
    ]
    for column in ("market_stage", "structure_trend_state", "liquidity_event_type"):
        if column in frame.columns and frame.height:
            unique = int(frame.select(pl.col(column).n_unique()).item() or 0)
            threshold = max(20, frame.height // 5)
            rows.append(
                {
                    "artifact": "classifier-health",
                    "label": label,
                    "health_check": f"{column}_cardinality",
                    "status": "warn" if unique > threshold else "pass",
                    "value": float(unique),
                    "threshold": float(threshold),
                    "reason": f"unique={unique}",
                }
            )
    out = pl.DataFrame(rows, schema=CLASSIFIER_HEALTH_SCHEMA)
    text = f"{label}\n" if label else ""
    text += format_table(
        ["Health check", "Status", "Reason"],
        [[str(row["health_check"]), str(row["status"]), str(row["reason"])] for row in rows],
    )
    return ClassifierHealthResult(out, text)


def classify_states(frame: pl.DataFrame, scale: str) -> pl.DataFrame:
    close_prev = pl.col("close").shift(1)
    true_range = pl.max_horizontal(
        (pl.col("high") - pl.col("low")).abs(),
        (pl.col("high") - close_prev).abs(),
        (pl.col("low") - close_prev).abs(),
    )
    atr_14 = true_range.rolling_mean(14)
    atr_rank = atr_14.rolling_rank(window_size=100)
    atr_count = atr_14.is_not_null().cast(pl.Int64).rolling_sum(window_size=100)
    frame = frame.with_columns(
        atr_14.alias("atr_14"),
        pl.when(atr_count > 1)
        .then((atr_rank - 1) / (atr_count - 1) * 100.0)
        .otherwise(None)
        .alias("atr_percentile_100"),
        pl.col("high").shift(2).rolling_max(5).alias("prior_high_window"),
        pl.col("low").shift(2).rolling_min(5).alias("prior_low_window"),
        pl.col("high").shift(1).rolling_max(48).alias("range_high"),
        pl.col("low").shift(1).rolling_min(48).alias("range_low"),
        pl.col("high").shift(1).rolling_max(20).alias("prior_liquidity_high"),
        pl.col("low").shift(1).rolling_min(20).alias("prior_liquidity_low"),
    )
    safe_atr = pl.when(pl.col("atr_14").abs() > 1e-10).then(pl.col("atr_14")).otherwise(None)
    swing_high = pl.col("prior_high_window").is_not_null() & (
        pl.col("high").shift(1) >= pl.col("prior_high_window")
    )
    swing_low = pl.col("prior_low_window").is_not_null() & (
        pl.col("low").shift(1) <= pl.col("prior_low_window")
    )
    frame = frame.with_columns(
        pl.when(swing_high).then(pl.col("high").shift(1)).otherwise(None).alias("swing_high_value"),
        pl.when(swing_low).then(pl.col("low").shift(1)).otherwise(None).alias("swing_low_value"),
        ((pl.col("range_high") - pl.col("range_low")) / safe_atr).alias(
            ClassifierColumn.RANGE_WIDTH_ATR
        ),
    ).with_columns(
        pl.col("swing_high_value").forward_fill().alias("last_swing_high"),
        pl.col("swing_low_value").forward_fill().alias("last_swing_low"),
    )
    higher_high = (
        swing_high
        & pl.col("last_swing_high").shift(1).is_not_null()
        & (pl.col("high").shift(1) > pl.col("last_swing_high").shift(1))
    ).fill_null(False)
    lower_high = (
        swing_high
        & pl.col("last_swing_high").shift(1).is_not_null()
        & (pl.col("high").shift(1) < pl.col("last_swing_high").shift(1))
    ).fill_null(False)
    higher_low = (
        swing_low
        & pl.col("last_swing_low").shift(1).is_not_null()
        & (pl.col("low").shift(1) > pl.col("last_swing_low").shift(1))
    ).fill_null(False)
    lower_low = (
        swing_low
        & pl.col("last_swing_low").shift(1).is_not_null()
        & (pl.col("low").shift(1) < pl.col("last_swing_low").shift(1))
    ).fill_null(False)
    range_ready = pl.col("range_high").is_not_null() & pl.col("range_low").is_not_null()
    hh_count = higher_high.cast(pl.Int64).rolling_sum(24)
    hl_count = higher_low.cast(pl.Int64).rolling_sum(24)
    lh_count = lower_high.cast(pl.Int64).rolling_sum(24)
    ll_count = lower_low.cast(pl.Int64).rolling_sum(24)
    uptrend = (hh_count > 0) & (hl_count > 0) & (hh_count + hl_count >= lh_count + ll_count)
    downtrend = (lh_count > 0) & (ll_count > 0) & (lh_count + ll_count >= hh_count + hl_count)
    trend_conflict = (uptrend & downtrend).fill_null(False)
    range_compression = (range_ready & (pl.col(ClassifierColumn.RANGE_WIDTH_ATR) <= 8.0)).fill_null(
        False
    )
    near_range_high = (
        range_ready & ((pl.col("range_high") - pl.col("close")).abs() <= safe_atr)
    ).fill_null(False)
    near_range_low = (
        range_ready & ((pl.col("close") - pl.col("range_low")).abs() <= safe_atr)
    ).fill_null(False)
    markup = ((pl.col("close") > pl.col("range_high")) & uptrend & ~trend_conflict).fill_null(False)
    markdown = ((pl.col("close") < pl.col("range_low")) & downtrend & ~trend_conflict).fill_null(
        False
    )
    transition = (range_ready & trend_conflict).fill_null(False)
    trend_continuation = ((uptrend | downtrend) & ~markup & ~markdown & ~trend_conflict).fill_null(
        False
    )
    wide_range = (range_ready & ~range_compression & ~(uptrend | downtrend)).fill_null(False)
    swept_high = (
        pl.col("prior_liquidity_high").is_not_null()
        & (pl.col("high") > pl.col("prior_liquidity_high"))
    ).fill_null(False)
    swept_low = (
        pl.col("prior_liquidity_low").is_not_null()
        & (pl.col("low") < pl.col("prior_liquidity_low"))
    ).fill_null(False)
    breakout_acceptance_high = (
        pl.col("prior_liquidity_high").is_not_null()
        & (pl.col("close") > pl.col("prior_liquidity_high"))
    ).fill_null(False)
    breakout_acceptance_low = (
        pl.col("prior_liquidity_low").is_not_null()
        & (pl.col("close") < pl.col("prior_liquidity_low"))
    ).fill_null(False)
    market_stage = (
        pl.when(~range_ready)
        .then(pl.lit(MarketStage.WARMUP))
        .when(markup)
        .then(pl.lit(MarketStage.MARKUP))
        .when(markdown)
        .then(pl.lit(MarketStage.MARKDOWN))
        .when(transition)
        .then(pl.lit(MarketStage.TRANSITION))
        .when(range_compression & near_range_low)
        .then(pl.lit(MarketStage.ACCUMULATION))
        .when(range_compression & near_range_high)
        .then(pl.lit(MarketStage.DISTRIBUTION_OR_REVERSAL))
        .when(range_compression)
        .then(pl.lit(MarketStage.RANGE))
        .when(trend_continuation)
        .then(pl.lit(MarketStage.TREND_CONTINUATION))
        .when(wide_range)
        .then(pl.lit(MarketStage.WIDE_RANGE))
        .otherwise(pl.lit(MarketStage.UNKNOWN))
    )
    structure = (
        pl.when(~range_ready)
        .then(pl.lit(StructureState.UNKNOWN))
        .when(uptrend)
        .then(pl.lit(StructureState.UPTREND))
        .when(downtrend)
        .then(pl.lit(StructureState.DOWNTREND))
        .when(range_compression)
        .then(pl.lit(StructureState.RANGE))
        .otherwise(pl.lit(StructureState.UNKNOWN))
    )
    liquidity_event = (
        pl.when(breakout_acceptance_high)
        .then(pl.lit("breakout_acceptance_high"))
        .when(breakout_acceptance_low)
        .then(pl.lit("breakout_acceptance_low"))
        .when(swept_high & ~breakout_acceptance_high)
        .then(pl.lit("failed_breakout_high"))
        .when(swept_low & ~breakout_acceptance_low)
        .then(pl.lit("failed_breakout_low"))
        .otherwise(pl.lit("none"))
    )
    frame = frame.with_columns(
        market_stage.alias(ClassifierColumn.MARKET_STAGE),
        structure.alias(ClassifierColumn.STRUCTURE_TREND_STATE),
        liquidity_event.alias("liquidity_event_type"),
    )
    direction = (
        pl.when(
            pl.col(ClassifierColumn.MARKET_STAGE).is_in(
                [MarketStage.ACCUMULATION, MarketStage.MARKUP]
            )
            | (pl.col(ClassifierColumn.STRUCTURE_TREND_STATE) == StructureState.UPTREND)
        )
        .then(pl.lit("bullish"))
        .when(
            pl.col(ClassifierColumn.MARKET_STAGE).is_in(
                [MarketStage.DISTRIBUTION_OR_REVERSAL, MarketStage.MARKDOWN]
            )
            | (pl.col(ClassifierColumn.STRUCTURE_TREND_STATE) == StructureState.DOWNTREND)
        )
        .then(pl.lit("bearish"))
        .when(
            pl.col(ClassifierColumn.MARKET_STAGE).is_in(
                [MarketStage.DATA_ERROR, MarketStage.WARMUP]
            )
        )
        .then(pl.lit("blocked"))
        .otherwise(pl.lit("neutral"))
    )
    return frame.with_columns(
        pl.concat_str(
            [
                pl.col(ClassifierColumn.MARKET_STAGE).cast(pl.String),
                pl.col(ClassifierColumn.STRUCTURE_TREND_STATE).cast(pl.String),
                _bucket_range_width_expr(),
                _bucket_volatility_expr(),
            ],
            separator="|",
        ).alias("state_key"),
        pl.when(pl.col("liquidity_event_type") != "none")
        .then(pl.col("liquidity_event_type"))
        .when(pl.col(ClassifierColumn.MARKET_STAGE).cast(pl.String).str.contains("accumulation"))
        .then(pl.lit("none_in_accumulation"))
        .when(pl.col(ClassifierColumn.MARKET_STAGE).cast(pl.String).str.contains("distribution"))
        .then(pl.lit("none_in_distribution"))
        .when(pl.col(ClassifierColumn.STRUCTURE_TREND_STATE).cast(pl.String).str.contains("trend"))
        .then(pl.lit("none_in_trend"))
        .otherwise(pl.lit("none_in_compression"))
        .alias("context_event"),
        direction.alias("direction_hint"),
    ).select(
        pl.col("symbol").cast(pl.String),
        pl.col("timestamp").cast(pl.Int64, strict=False),
        pl.lit("kline").alias("source_family"),
        pl.lit(scale).alias("scale"),
        pl.col("state_key"),
        pl.col("context_event"),
        pl.col("direction_hint"),
        pl.when(pl.col("direction_hint").is_in(["bullish", "bearish"]))
        .then(0.8)
        .otherwise(0.45)
        .alias("quality_weight"),
        pl.lit(False).alias("missing_flag"),
        pl.lit(False).alias("stale_flag"),
    )


def _missing_state_frame(*, symbol: str, family: str, scale: str, reason: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [symbol],
            "timestamp": [None],
            "source_family": [family],
            "scale": [scale],
            "state_key": [reason],
            "context_event": [reason],
            "direction_hint": ["missing"],
            "quality_weight": [0.0],
            "missing_flag": [True],
            "stale_flag": [False],
        },
        schema=STATE_FRAME_SCHEMA,
    )


def validate_state_frame(frame: pl.DataFrame) -> pl.DataFrame:
    missing = [column for column in STATE_FRAME_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"source state frame missing columns: {', '.join(missing)}")
    return frame.with_columns(
        *(
            pl.col(column).cast(dtype, strict=False).alias(column)
            for column, dtype in STATE_FRAME_SCHEMA.items()
        )
    ).select(STATE_FRAME_COLUMNS)


def _state_frame_symbol(frame: pl.DataFrame) -> str:
    if frame.is_empty() or "symbol" not in frame.columns:
        return "*"
    value = frame.select("symbol").drop_nulls().head(1)
    return "*" if value.is_empty() else str(value.item())


def _bucket_range_width_expr() -> pl.Expr:
    value = pl.col(ClassifierColumn.RANGE_WIDTH_ATR).cast(pl.Float64, strict=False)
    return (
        pl.when(value.is_null())
        .then(pl.lit("range_unknown"))
        .when(value < 1.0)
        .then(pl.lit("range_tight"))
        .when(value < 3.0)
        .then(pl.lit("range_normal"))
        .otherwise(pl.lit("range_wide"))
    )


def _bucket_volatility_expr() -> pl.Expr:
    value = pl.col("atr_percentile_100").cast(pl.Float64, strict=False)
    return (
        pl.when(value.is_null())
        .then(pl.lit("vol_unknown"))
        .when(value < 25.0)
        .then(pl.lit("vol_low"))
        .when(value < 75.0)
        .then(pl.lit("vol_normal"))
        .otherwise(pl.lit("vol_high"))
    )


HOUR_MS = 60 * 60 * 1000
SOURCE_FEATURE_MAX_AGE_MS = {
    "books": HOUR_MS,
    "trades": HOUR_MS,
    "funding": 16 * HOUR_MS,
    "open_interest": 4 * HOUR_MS,
    "taker_volume": 4 * HOUR_MS,
    "long_short_ratios": 4 * HOUR_MS,
}


def extract_continuous_features(
    bars: dict[tuple[str, str], pl.DataFrame],
    state_frames: dict[tuple[str, str], pl.DataFrame],
    source_frames: dict[str, pl.DataFrame],
    *,
    decision_timeframe: str = "1H",
) -> pl.DataFrame:
    """Extract continuous features from OHLCV and source frames.

    Returns one frame with columns: symbol, timestamp,
    atr_percentile, range_width_atr, source pressure/freshness values,
    and normalized bar/source features such as bar_return_24h_pct,
    bar_return_24h_per_vol_7d, funding_rate_bps, and oi_change_pct.
    Source event/snapshot rows are aligned as known-at-close values by symbol
    with backward as-of joins; raw source timestamps are not overwritten.
    """
    kline_features = _kline_continuous_features(bars, state_frames, decision_timeframe)
    decision_keys = (
        kline_features.select("symbol", "timestamp") if not kline_features.is_empty() else None
    )
    source_features = _source_continuous_features(source_frames, decision_keys=decision_keys)

    if kline_features.is_empty():
        return source_features
    if source_features.is_empty():
        return kline_features

    return kline_features.join(source_features, on=["symbol", "timestamp"], how="left").select(
        kline_features.columns
        + [c for c in source_features.columns if c not in ("symbol", "timestamp")]
    )


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

        work = frame.select("timestamp", "open", "high", "low", "close", "volume").sort("timestamp")

        # ATR percentile from classifier state frame (already computed)
        atr_pct = None
        if "atr_percentile_100" in state.columns:
            atr_pct = state.select(
                "timestamp", pl.col("atr_percentile_100").alias("atr_percentile")
            ).sort("timestamp")

        # Range width from classifier state frame
        range_width = None
        if "range_width_atr" in state.columns:
            range_width = state.select("timestamp", pl.col("range_width_atr")).sort("timestamp")

        # Returns. Keep raw returns, then add volatility-scaled aliases for
        # cross-symbol comparability.
        work = (
            work.with_columns(
                (
                    (pl.col("close") - pl.col("close").shift(1)) / pl.col("close").shift(1) * 100.0
                ).alias("bar_return_1h_pct"),
                (
                    (pl.col("close") - pl.col("close").shift(4)) / pl.col("close").shift(4) * 100.0
                ).alias("bar_return_4h_pct"),
                (
                    (pl.col("close") - pl.col("close").shift(24))
                    / pl.col("close").shift(24)
                    * 100.0
                ).alias("bar_return_24h_pct"),
            )
            .with_columns(
                pl.col("bar_return_1h_pct")
                .rolling_std(168, min_samples=24)
                .alias("bar_return_vol_7d")
            )
            .with_columns(
                (
                    pl.col("bar_return_4h_pct")
                    / pl.when(pl.col("bar_return_vol_7d") > 0.0)
                    .then(pl.col("bar_return_vol_7d"))
                    .otherwise(None)
                ).alias("bar_return_4h_per_vol_7d"),
                (
                    pl.col("bar_return_24h_pct")
                    / pl.when(pl.col("bar_return_vol_7d") > 0.0)
                    .then(pl.col("bar_return_vol_7d"))
                    .otherwise(None)
                ).alias("bar_return_24h_per_vol_7d"),
            )
        )

        # Volume anomaly
        work = work.with_columns(
            (pl.col("volume") / pl.col("volume").rolling_mean(20, min_samples=5)).alias(
                "bar_volume_1h_to_ma_20h"
            ),
        )

        # Pre-entry cleanliness features. These use only current/prior OHLCV rows
        # available at the decision close and target chop, exhaustion, and path
        # efficiency rather than generic tail magnitude.
        work = (
            work.with_columns(
                (pl.col("high") - pl.col("low")).alias("bar_range_1h"),
                (pl.col("close") - pl.col("open")).abs().alias("bar_body_1h"),
                pl.col("bar_return_1h_pct").sign().alias("bar_return_sign"),
            )
            .with_columns(
                (
                    pl.col("bar_return_sign") * pl.col("bar_return_sign").shift(1) < 0
                )
                .cast(pl.Float64)
                .alias("return_sign_flip"),
                (
                    pl.col("bar_body_1h")
                    / pl.when(pl.col("bar_range_1h") > 0.0)
                    .then(pl.col("bar_range_1h"))
                    .otherwise(None)
                ).alias("body_to_range_1h"),
            )
            .with_columns(
                pl.col("return_sign_flip")
                .rolling_mean(6, min_samples=3)
                .alias("return_sign_flip_rate_6h"),
                pl.col("return_sign_flip")
                .rolling_mean(24, min_samples=12)
                .alias("return_sign_flip_rate_24h"),
                pl.col("body_to_range_1h")
                .rolling_mean(24, min_samples=12)
                .alias("body_to_range_mean_24h"),
                (
                    pl.col("bar_range_1h").rolling_mean(24, min_samples=12)
                    / pl.when(pl.col("bar_range_1h").rolling_mean(168, min_samples=24) > 0.0)
                    .then(pl.col("bar_range_1h").rolling_mean(168, min_samples=24))
                    .otherwise(None)
                ).alias("range_expansion_24h_to_7d"),
                (
                    (pl.col("close") - pl.col("low").shift(1).rolling_min(24))
                    / (
                        pl.col("high").shift(1).rolling_max(24)
                        - pl.col("low").shift(1).rolling_min(24)
                    )
                ).alias("close_position_24h"),
                (
                    (pl.col("close").shift(1) - pl.col("low").shift(1).rolling_min(6))
                    / pl.col("close").shift(1)
                    * 100.0
                ).alias("prior_runup_6h"),
                (
                    (pl.col("high").shift(1).rolling_max(6) - pl.col("close").shift(1))
                    / pl.col("close").shift(1)
                    * 100.0
                ).alias("prior_drawdown_6h"),
                (
                    (pl.col("close") - pl.col("close").shift(24)).abs()
                    / pl.when(
                        pl.col("bar_return_1h_pct")
                        .abs()
                        .rolling_sum(24, min_samples=12)
                        > 0.0
                    )
                    .then(pl.col("bar_return_1h_pct").abs().rolling_sum(24, min_samples=12))
                    .otherwise(None)
                ).alias("return_efficiency_24h"),
            )
        )

        # Close-to-range ratio (from OHLCV directly: 48-bar range)
        work = work.with_columns(
            pl.col("high").shift(1).rolling_max(48).alias("range_high_48"),
            pl.col("low").shift(1).rolling_min(48).alias("range_low_48"),
        ).with_columns(
            (
                (pl.col("close") - pl.col("range_low_48"))
                / (pl.col("range_high_48") - pl.col("range_low_48"))
            ).alias("bar_close_position_48h"),
        )

        cols = [
            "symbol",
            "timestamp",
            "bar_return_1h_pct",
            "bar_return_4h_pct",
            "bar_return_24h_pct",
            "bar_return_4h_per_vol_7d",
            "bar_return_24h_per_vol_7d",
            "bar_volume_1h_to_ma_20h",
            "return_sign_flip_rate_6h",
            "return_sign_flip_rate_24h",
            "body_to_range_mean_24h",
            "range_expansion_24h_to_7d",
            "close_position_24h",
            "prior_runup_6h",
            "prior_drawdown_6h",
            "return_efficiency_24h",
            "bar_close_position_48h",
        ]
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
        return pl.DataFrame(
            schema={
                "symbol": pl.String,
                "timestamp": pl.Int64,
                "bar_return_1h_pct": pl.Float64,
                "bar_return_4h_pct": pl.Float64,
                "bar_return_24h_pct": pl.Float64,
                "bar_return_4h_per_vol_7d": pl.Float64,
                "bar_return_24h_per_vol_7d": pl.Float64,
                "bar_volume_1h_to_ma_20h": pl.Float64,
                "return_sign_flip_rate_6h": pl.Float64,
                "return_sign_flip_rate_24h": pl.Float64,
                "body_to_range_mean_24h": pl.Float64,
                "range_expansion_24h_to_7d": pl.Float64,
                "close_position_24h": pl.Float64,
                "prior_runup_6h": pl.Float64,
                "prior_drawdown_6h": pl.Float64,
                "return_efficiency_24h": pl.Float64,
                "bar_close_position_48h": pl.Float64,
                "atr_percentile": pl.Float64,
                "range_width_atr": pl.Float64,
            }
        )
    return _market_context_features(pl.concat(frames, how="vertical_relaxed"))


def _market_context_features(kline_features: pl.DataFrame) -> pl.DataFrame:
    """Attach known-at-close cross-symbol market context to kline features."""
    schema = {
        "symbol": pl.String,
        "timestamp": pl.Int64,
        "market_return_1h_median": pl.Float64,
        "market_return_4h_median": pl.Float64,
        "market_return_24h_median": pl.Float64,
        "market_abs_return_24h_median": pl.Float64,
        "market_dispersion_24h": pl.Float64,
        "market_positive_return_24h_share": pl.Float64,
        "symbol_vs_market_return_24h": pl.Float64,
        "symbol_vs_market_return_4h": pl.Float64,
        "symbol_abs_return_vs_market_24h": pl.Float64,
    }
    if kline_features.is_empty():
        return pl.DataFrame(schema=schema)
    required = {"symbol", "timestamp", "bar_return_4h_pct", "bar_return_24h_pct"}
    if not required.issubset(kline_features.columns):
        return kline_features
    market = kline_features.group_by("timestamp").agg(
        pl.col("bar_return_1h_pct").median().alias("market_return_1h_median"),
        pl.col("bar_return_4h_pct").median().alias("market_return_4h_median"),
        pl.col("bar_return_24h_pct").median().alias("market_return_24h_median"),
        pl.col("bar_return_24h_pct").abs().median().alias("market_abs_return_24h_median"),
        pl.col("bar_return_24h_pct").std().alias("market_dispersion_24h"),
        (pl.col("bar_return_24h_pct") > 0.0)
        .cast(pl.Float64)
        .mean()
        .alias("market_positive_return_24h_share"),
    )
    return kline_features.join(market, on="timestamp", how="left").with_columns(
        (pl.col("bar_return_24h_pct") - pl.col("market_return_24h_median")).alias(
            "symbol_vs_market_return_24h"
        ),
        (pl.col("bar_return_4h_pct") - pl.col("market_return_4h_median")).alias(
            "symbol_vs_market_return_4h"
        ),
        (pl.col("bar_return_24h_pct").abs() - pl.col("market_abs_return_24h_median")).alias(
            "symbol_abs_return_vs_market_24h"
        ),
    )


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


def source_time_series_features_frame(
    source_frames: dict[str, pl.DataFrame],
    bars: pl.DataFrame,
    decision_keys: pl.DataFrame,
) -> pl.DataFrame:
    """Build kline-like source state/path features aligned to decision bars."""
    if decision_keys.is_empty() or not {"symbol", "timestamp"}.issubset(decision_keys.columns):
        return pl.DataFrame()
    frames = [
        _funding_time_series_features(source_frames.get("funding", pl.DataFrame()), bars),
        _lsr_time_series_features(source_frames.get("long_short_ratios", pl.DataFrame()), bars),
        _oi_time_series_features(source_frames.get("open_interest", pl.DataFrame()), bars),
        _taker_time_series_features(source_frames.get("taker_volume", pl.DataFrame())),
    ]
    aligned = decision_keys.select("symbol", "timestamp").unique().sort(["symbol", "timestamp"])
    for frame in frames:
        if frame.is_empty():
            continue
        value_cols = [column for column in frame.columns if column not in ("symbol", "timestamp")]
        if not value_cols:
            continue
        aligned = aligned.join_asof(
            frame.sort(["symbol", "timestamp"]),
            on="timestamp",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
    return aligned


def _price_return_context(bars: pl.DataFrame) -> pl.DataFrame:
    if bars.is_empty() or not {"symbol", "timestamp", "close"}.issubset(bars.columns):
        return pl.DataFrame(
            schema={"symbol": pl.String, "timestamp": pl.Int64, "price_return_pct": pl.Float64}
        )
    return (
        bars.sort("symbol", "timestamp")
        .with_columns(pl.col("close").shift(1).over("symbol").alias("previous_close"))
        .select(
            "symbol",
            "timestamp",
            (
                (pl.col("close") - pl.col("previous_close"))
                / pl.col("previous_close")
                * 100.0
            ).alias("price_return_pct"),
        )
    )


def _state_path_expr(column: str, window: int = 24) -> pl.Expr:
    return pl.concat_str(
        [pl.col(column).shift(offset).over("symbol") for offset in range(window - 1, 0, -1)]
        + [pl.col(column)],
        separator=" -> ",
        ignore_nulls=True,
    )


def _source_state_run_columns(frame: pl.DataFrame, state_col: str, prefix: str) -> pl.DataFrame:
    changed = (pl.col(state_col) != pl.col(state_col).shift(1).over("symbol")).fill_null(True)
    run_col = f"{prefix}_run"
    age_col = f"{prefix}_age_bars"
    transition_col = f"{prefix}_transition"
    transition_root = prefix.split("_")[0]
    return (
        frame.sort("symbol", "timestamp")
        .with_columns(changed.alias("_state_changed"))
        .with_columns(
            pl.col("_state_changed").cast(pl.Int64).cum_sum().over("symbol").alias(run_col)
        )
        .with_columns(
            pl.cum_count(state_col).over("symbol", run_col).cast(pl.UInt32).alias(age_col)
        )
        .with_columns(
            pl.when(pl.col(state_col).str.ends_with("missing"))
            .then(pl.lit(f"{transition_root}_missing"))
            .when(pl.col(age_col) == 1)
            .then(pl.lit(f"{transition_root}_flip"))
            .when(pl.col(age_col) > 1)
            .then(pl.lit(f"{transition_root}_persistence"))
            .otherwise(pl.lit(f"{transition_root}_initial"))
            .alias(transition_col)
        )
        .drop("_state_changed", run_col)
    )


def _funding_time_series_features(frame: pl.DataFrame, bars: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty() or not {"symbol", "timestamp", "funding_rate"}.issubset(frame.columns):
        return pl.DataFrame()
    work = (
        frame.select("symbol", "timestamp", pl.col("funding_rate").cast(pl.Float64, strict=False))
        .sort("symbol", "timestamp")
        .join(_price_return_context(bars), on=["symbol", "timestamp"], how="left")
        .with_columns(
            pl.when(pl.col("funding_rate") > 0.0)
            .then(pl.lit("funding_positive"))
            .when(pl.col("funding_rate") < 0.0)
            .then(pl.lit("funding_negative"))
            .when(pl.col("funding_rate").is_null())
            .then(pl.lit("funding_missing"))
            .otherwise(pl.lit("funding_neutral"))
            .alias("funding_level_state")
        )
    )
    work = _source_state_run_columns(work, "funding_level_state", "funding_level")
    return work.with_columns(
        pl.col("funding_level_age_bars").alias("funding_direction_run_length"),
        _state_path_expr("funding_level_state").alias("funding_path_24h"),
        pl.when((pl.col("funding_rate") > 0.0) & (pl.col("price_return_pct") < 0.0))
        .then(pl.lit("positive_funding_price_down"))
        .when((pl.col("funding_rate") < 0.0) & (pl.col("price_return_pct") > 0.0))
        .then(pl.lit("negative_funding_price_up"))
        .when(pl.col("price_return_pct").is_null())
        .then(pl.lit("funding_price_flat_or_missing"))
        .otherwise(pl.lit("funding_price_aligned"))
        .alias("funding_price_divergence_24h"),
    ).select(
        "symbol",
        "timestamp",
        "funding_level_state",
        "funding_level_age_bars",
        "funding_level_transition",
        "funding_direction_run_length",
        "funding_path_24h",
        "funding_price_divergence_24h",
    )


def _lsr_time_series_features(frame: pl.DataFrame, bars: pl.DataFrame) -> pl.DataFrame:
    ratio_col = _first_float_col(
        frame,
        (
            "top_trader_long_short_position_ratio",
            "top_trader_long_short_account_ratio",
            "long_short_account_ratio",
        ),
    )
    if frame.is_empty() or ratio_col is None or not {"symbol", "timestamp"}.issubset(frame.columns):
        return pl.DataFrame()
    work = (
        frame.select(
            "symbol",
            "timestamp",
            pl.col(ratio_col).cast(pl.Float64, strict=False).log().alias("lsr_log_ratio"),
        )
        .sort("symbol", "timestamp")
        .join(_price_return_context(bars), on=["symbol", "timestamp"], how="left")
        .with_columns(
            (pl.col("lsr_log_ratio") - pl.col("lsr_log_ratio").shift(24).over("symbol")).alias(
                "lsr_log_ratio_change_24h"
            )
        )
        .with_columns(
            pl.when(pl.col("lsr_log_ratio") > 0.0)
            .then(pl.lit("lsr_long_crowding"))
            .when(pl.col("lsr_log_ratio") < 0.0)
            .then(pl.lit("lsr_short_crowding"))
            .when(pl.col("lsr_log_ratio").is_null())
            .then(pl.lit("lsr_missing"))
            .otherwise(pl.lit("lsr_neutral"))
            .alias("lsr_level_state")
        )
    )
    work = _source_state_run_columns(work, "lsr_level_state", "lsr_level")
    return work.with_columns(
        pl.col("lsr_level_age_bars").alias("lsr_direction_run_length"),
        _state_path_expr("lsr_level_state").alias("lsr_path_24h"),
        pl.when((pl.col("lsr_log_ratio") > 0.0) & (pl.col("price_return_pct") < 0.0))
        .then(pl.lit("long_crowding_price_down"))
        .when((pl.col("lsr_log_ratio") < 0.0) & (pl.col("price_return_pct") > 0.0))
        .then(pl.lit("short_crowding_price_up"))
        .when(pl.col("price_return_pct").is_null())
        .then(pl.lit("lsr_price_flat_or_missing"))
        .otherwise(pl.lit("lsr_price_aligned"))
        .alias("lsr_price_divergence_24h"),
    ).select(
        "symbol",
        "timestamp",
        "lsr_level_state",
        "lsr_level_age_bars",
        "lsr_level_transition",
        "lsr_direction_run_length",
        "lsr_path_24h",
        "lsr_log_ratio_change_24h",
        "lsr_price_divergence_24h",
    )


def _oi_time_series_features(frame: pl.DataFrame, bars: pl.DataFrame) -> pl.DataFrame:
    value_col = _first_float_col(frame, ("open_interest_usd", "open_interest"))
    if frame.is_empty() or value_col is None or not {"symbol", "timestamp"}.issubset(frame.columns):
        return pl.DataFrame()
    work = (
        frame.select(
            "symbol",
            "timestamp",
            pl.col(value_col).cast(pl.Float64, strict=False).alias("oi_value"),
        )
        .sort("symbol", "timestamp")
        .join(_price_return_context(bars), on=["symbol", "timestamp"], how="left")
        .with_columns(
            (
                (pl.col("oi_value") - pl.col("oi_value").shift(24).over("symbol"))
                / pl.col("oi_value").shift(24).over("symbol")
                * 100.0
            ).alias("oi_change_pct_24h"),
            (
                (pl.col("oi_value") - pl.col("oi_value").shift(1).over("symbol"))
                / pl.col("oi_value").shift(1).over("symbol")
                * 100.0
            ).alias("_oi_step_change_pct"),
        )
        .with_columns(
            pl.when(pl.col("_oi_step_change_pct") > 0.0)
            .then(pl.lit("oi_build"))
            .when(pl.col("_oi_step_change_pct") < 0.0)
            .then(pl.lit("oi_unwind"))
            .when(pl.col("_oi_step_change_pct").is_null())
            .then(pl.lit("oi_missing"))
            .otherwise(pl.lit("oi_flat"))
            .alias("oi_flow_state")
        )
    )
    work = _source_state_run_columns(work, "oi_flow_state", "oi_flow")
    return work.with_columns(
        pl.col("oi_flow_age_bars").alias("oi_flow_run_length"),
        _state_path_expr("oi_flow_state").alias("oi_price_flow_path_24h"),
    ).select(
        "symbol",
        "timestamp",
        "oi_flow_state",
        "oi_flow_age_bars",
        "oi_flow_transition",
        "oi_flow_run_length",
        "oi_change_pct_24h",
        "oi_price_flow_path_24h",
    )


def _taker_time_series_features(frame: pl.DataFrame) -> pl.DataFrame:
    required = {"symbol", "timestamp", "taker_buy_volume", "taker_sell_volume"}
    if frame.is_empty() or not required.issubset(frame.columns):
        return pl.DataFrame()
    buy = pl.col("taker_buy_volume").cast(pl.Float64, strict=False)
    sell = pl.col("taker_sell_volume").cast(pl.Float64, strict=False)
    pressure = (buy - sell) / pl.when((buy + sell) > 0.0).then(buy + sell).otherwise(None)
    work = (
        frame.select("symbol", "timestamp", pressure.alias("taker_buy_pressure"))
        .sort("symbol", "timestamp")
        .with_columns(
            pl.col("taker_buy_pressure")
            .rolling_mean(24, min_samples=1)
            .over("symbol")
            .alias("taker_buy_pressure_24h_mean")
        )
        .with_columns(
            pl.when(pl.col("taker_buy_pressure") > 0.2)
            .then(pl.lit("taker_buy_pressure"))
            .when(pl.col("taker_buy_pressure") < -0.2)
            .then(pl.lit("taker_sell_pressure"))
            .when(pl.col("taker_buy_pressure").is_null())
            .then(pl.lit("taker_missing"))
            .otherwise(pl.lit("taker_balanced"))
            .alias("taker_pressure_state")
        )
    )
    work = _source_state_run_columns(work, "taker_pressure_state", "taker_pressure")
    return work.with_columns(
        pl.col("taker_pressure_age_bars").alias("taker_pressure_run_length"),
        _state_path_expr("taker_pressure_state").alias("taker_pressure_path_24h"),
    ).select(
        "symbol",
        "timestamp",
        "taker_pressure_state",
        "taker_pressure_age_bars",
        "taker_pressure_transition",
        "taker_pressure_run_length",
        "taker_buy_pressure_24h_mean",
        "taker_pressure_path_24h",
    )


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

    age_col = f"{prefix}_age_ms"
    aligned = aligned.with_columns(
        (pl.col("timestamp") - pl.col("source_timestamp_ms")).alias(age_col)
    )
    max_age_ms = SOURCE_FEATURE_MAX_AGE_MS.get(family)
    if max_age_ms is not None:
        aligned = aligned.with_columns(
            *[
                pl.when(pl.col(age_col) <= max_age_ms).then(pl.col(col)).otherwise(None).alias(col)
                for col in value_cols
            ]
        )
    return aligned.drop("source_timestamp_ms")


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
    imbalance_col = _first_float_col(
        frame, ("ob_imbalance_25", "ob_imbalance_10", "ob_imbalance_5")
    )
    bid_col = _first_float_col(frame, ("ob_bid_price",))
    ask_col = _first_float_col(frame, ("ob_ask_price",))

    needed = [c for c in (imbalance_col, bid_col, ask_col) if c]
    if not needed:
        return work

    work = work.join(
        frame.select(["symbol", "timestamp"] + needed), on=["symbol", "timestamp"], how="left"
    )
    exprs = []

    if imbalance_col:
        exprs.append(pl.col(imbalance_col).cast(pl.Float64).alias("imbalance_value"))
    if bid_col and ask_col:
        mid = (pl.col(bid_col).cast(pl.Float64) + pl.col(ask_col).cast(pl.Float64)) / 2.0
        exprs.append(((pl.col(ask_col) - pl.col(bid_col)) / mid * 10000.0).alias("spread_bps"))
        if "close" in frame.columns:
            work = work.join(
                frame.select("symbol", "timestamp", pl.col("close").cast(pl.Float64)),
                on=["symbol", "timestamp"],
                how="left",
            )
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
            (
                pl.col("buy_notional")
                / pl.when(pl.col("sell_notional") > 0).then(pl.col("sell_notional")).otherwise(None)
            ).alias("buy_sell_ratio")
        )
    )
    return work.join(
        agg.select("symbol", "timestamp", "buy_sell_ratio"), on=["symbol", "timestamp"], how="left"
    )


def _funding_continuous(work: pl.DataFrame, frame: pl.DataFrame) -> pl.DataFrame:
    if "funding_rate" in frame.columns:
        return work.join(
            frame.select(
                "symbol",
                "timestamp",
                pl.col("funding_rate").cast(pl.Float64).alias("funding_rate_raw"),
                (pl.col("funding_rate").cast(pl.Float64) * 10000.0).alias("funding_rate_bps"),
            ),
            on=["symbol", "timestamp"],
            how="left",
        )
    return work


def _oi_continuous(work: pl.DataFrame, frame: pl.DataFrame) -> pl.DataFrame:
    value_col = _first_float_col(frame, ("open_interest_usd", "open_interest"))
    if not value_col:
        return work

    oi = frame.select(
        "symbol", "timestamp", pl.col(value_col).cast(pl.Float64).alias("oi_value")
    ).sort(["symbol", "timestamp"])
    previous_oi = pl.col("oi_value").shift(1).over("symbol")
    oi = oi.with_columns(
        (pl.col("oi_value") - previous_oi).alias("oi_change_raw"),
        (
            (pl.col("oi_value") - previous_oi)
            / pl.when(previous_oi > 0.0).then(previous_oi).otherwise(None)
            * 100.0
        ).alias("oi_change_pct"),
    )
    return work.join(
        oi.select("symbol", "timestamp", "oi_change_raw", "oi_change_pct"),
        on=["symbol", "timestamp"],
        how="left",
    )


def _taker_continuous(work: pl.DataFrame, frame: pl.DataFrame) -> pl.DataFrame:
    if "taker_buy_volume" not in frame.columns or "taker_sell_volume" not in frame.columns:
        return work

    taker = frame.select(
        "symbol",
        "timestamp",
        pl.col("taker_buy_volume").cast(pl.Float64),
        pl.col("taker_sell_volume").cast(pl.Float64),
    ).with_columns(
        (
            pl.col("taker_buy_volume")
            / pl.when(pl.col("taker_sell_volume") > 0)
            .then(pl.col("taker_sell_volume"))
            .otherwise(None)
        ).alias("taker_buy_sell_ratio_raw"),
        (
            (pl.col("taker_buy_volume") - pl.col("taker_sell_volume"))
            / pl.when((pl.col("taker_buy_volume") + pl.col("taker_sell_volume")) > 0.0)
            .then(pl.col("taker_buy_volume") + pl.col("taker_sell_volume"))
            .otherwise(None)
        ).alias("taker_buy_pressure"),
    )
    return work.join(
        taker.select("symbol", "timestamp", "taker_buy_sell_ratio_raw", "taker_buy_pressure"),
        on=["symbol", "timestamp"],
        how="left",
    )


def _lsr_continuous(work: pl.DataFrame, frame: pl.DataFrame) -> pl.DataFrame:
    ratio_col = _first_float_col(
        frame,
        (
            "top_trader_long_short_position_ratio",
            "top_trader_long_short_account_ratio",
            "long_short_account_ratio",
        ),
    )
    if ratio_col:
        return work.join(
            frame.select(
                "symbol",
                "timestamp",
                pl.col(ratio_col).cast(pl.Float64).alias("lsr_ratio_raw"),
                pl.col(ratio_col).cast(pl.Float64).log().alias("lsr_log_ratio"),
            ),
            on=["symbol", "timestamp"],
            how="left",
        )
    return work


def continuous_features_frame(
    bars: dict[tuple[str, str], pl.DataFrame],
    state_frames: dict[tuple[str, str], pl.DataFrame],
    source_frames: dict[str, pl.DataFrame],
    *,
    decision_timeframe: str = "1H",
) -> pl.DataFrame:
    return extract_continuous_features(
        bars,
        state_frames,
        source_frames,
        decision_timeframe=decision_timeframe,
    )


POTENTIAL_OBSERVATION_SCHEMA = {
    "symbol": pl.String,
    "decision_timeframe": pl.String,
    "decision_bar_close_ms": pl.Int64,
    "background_regime": pl.String,
    "background_structure": pl.String,
    "background_range": pl.String,
    "background_vol": pl.String,
    "swing_regime": pl.String,
    "swing_core": pl.String,
    "swing_range": pl.String,
    "swing_transition": pl.String,
    "decision_direction": pl.String,
    "decision_regime": pl.String,
    "decision_core": pl.String,
    "decision_range": pl.String,
    "decision_vol": pl.String,
    "decision_event": pl.String,
    "decision_event_age_bucket": pl.String,
    "decision_transition": pl.String,
    "source_family": pl.String,
    "source_state": pl.String,
    "source_direction": pl.String,
    "source_known_at_ms": pl.Int64,
    "source_age_ms": pl.Int64,
    "source_freshness": pl.String,
    "market_alignment": pl.String,
    "source_market_alignment": pl.String,
    "risk_context": pl.String,
}


def potential_observation_frame(
    kline_history: pl.DataFrame,
    source_events: pl.DataFrame,
    continuous_features: pl.DataFrame | None = None,
    *,
    decision_timeframe: str,
    max_source_staleness_hours: int,
) -> pl.DataFrame:
    if (
        kline_history.is_empty()
        or decision_timeframe not in kline_history.get_column("timeframe").to_list()
    ):
        return pl.DataFrame(schema=POTENTIAL_OBSERVATION_SCHEMA)
    decision = _potential_state_columns(kline_history, decision_timeframe, "decision").rename(
        {"bar_close_ms": "decision_bar_close_ms"}
    )
    if decision.is_empty():
        return pl.DataFrame(schema=POTENTIAL_OBSERVATION_SCHEMA)
    observations = decision.with_columns(pl.lit(decision_timeframe).alias("decision_timeframe"))
    for timeframe, prefix in (("4H", "swing"), ("1D", "background")):
        state = _potential_state_columns(kline_history, timeframe, prefix)
        if state.is_empty():
            observations = observations.with_columns(
                *[
                    pl.lit(None, dtype=pl.String).alias(column)
                    for column in _potential_state_output_columns(prefix)
                ]
            )
            continue
        observations = observations.sort("symbol", "decision_bar_close_ms").join_asof(
            state.sort("symbol", "bar_close_ms"),
            left_on="decision_bar_close_ms",
            right_on="bar_close_ms",
            by="symbol",
            strategy="backward",
            check_sortedness=False,
        )
    observations = observations.with_columns(
        pl.when(pl.col("background_regime") == pl.col("swing_regime"))
        .then(pl.lit("background_swing_aligned"))
        .when(pl.col("background_regime").is_null() | pl.col("swing_regime").is_null())
        .then(pl.lit("market_context_missing"))
        .otherwise(pl.lit("background_swing_conflict"))
        .alias("market_alignment"),
        pl.concat_str("decision_range", "decision_vol", separator="|").alias("risk_context"),
    )
    if source_events.is_empty():
        return _join_continuous_features(
            _potential_observation_without_source(observations), continuous_features
        )
    source_rows = source_events.filter(pl.col("source_state").is_not_null()).select(
        "symbol",
        "source_family",
        "source_state",
        "source_direction",
        pl.col("known_at_ms").alias("source_known_at_ms"),
    )
    if source_rows.is_empty():
        return _join_continuous_features(
            _potential_observation_without_source(observations), continuous_features
        )
    frames = []
    max_age_ms = max_source_staleness_hours * 60 * 60 * 1000
    for family in source_rows.get_column("source_family").drop_nulls().unique().to_list():
        family_source = source_rows.filter(pl.col("source_family") == family)
        frames.append(
            observations.sort("symbol", "decision_bar_close_ms")
            .join_asof(
                family_source.sort("symbol", "source_known_at_ms"),
                left_on="decision_bar_close_ms",
                right_on="source_known_at_ms",
                by="symbol",
                strategy="backward",
                check_sortedness=False,
            )
            .with_columns(
                (pl.col("decision_bar_close_ms") - pl.col("source_known_at_ms")).alias(
                    "source_age_ms"
                ),
                pl.when(pl.col("source_known_at_ms").is_null())
                .then(pl.lit("missing"))
                .when((pl.col("decision_bar_close_ms") - pl.col("source_known_at_ms")) > max_age_ms)
                .then(pl.lit("stale"))
                .otherwise(pl.lit("fresh"))
                .alias("source_freshness"),
                pl.when(pl.col("source_direction").is_null())
                .then(pl.lit("source_missing"))
                .when(pl.col("source_direction") == pl.col("decision_direction"))
                .then(pl.lit("source_agrees_with_decision"))
                .when(pl.col("source_direction") == "neutral")
                .then(pl.lit("source_neutral"))
                .otherwise(pl.lit("source_conflicts_with_decision"))
                .alias("source_market_alignment"),
            )
        )
    if not frames:
        result = _potential_observation_without_source(observations)
    else:
        result = pl.concat(frames, how="vertical_relaxed").select(
            *POTENTIAL_OBSERVATION_SCHEMA.keys()
        )

    return _join_continuous_features(result, continuous_features)


def _join_continuous_features(
    observations: pl.DataFrame, continuous_features: pl.DataFrame | None
) -> pl.DataFrame:
    if continuous_features is None or continuous_features.is_empty():
        return observations
    if not {
        "symbol",
        "decision_bar_close_ms",
    }.issubset(observations.columns) or not {"symbol", "timestamp"}.issubset(
        continuous_features.columns
    ):
        return observations
    cf_cols = [
        c
        for c in continuous_features.columns
        if c not in ("symbol", "timestamp") and c not in observations.columns
    ]
    if not cf_cols:
        return observations
    return observations.join(
        continuous_features.select(["symbol", "timestamp"] + cf_cols).unique(
            subset=["symbol", "timestamp"], keep="last"
        ),
        left_on=["symbol", "decision_bar_close_ms"],
        right_on=["symbol", "timestamp"],
        how="left",
    )


def _potential_observation_without_source(observations: pl.DataFrame) -> pl.DataFrame:
    return observations.with_columns(
        pl.lit(None, dtype=pl.String).alias("source_family"),
        pl.lit(None, dtype=pl.String).alias("source_state"),
        pl.lit(None, dtype=pl.String).alias("source_direction"),
        pl.lit(None, dtype=pl.Int64).alias("source_known_at_ms"),
        pl.lit(None, dtype=pl.Int64).alias("source_age_ms"),
        pl.lit("missing").alias("source_freshness"),
        pl.lit("source_missing").alias("source_market_alignment"),
    ).select(*POTENTIAL_OBSERVATION_SCHEMA.keys())


def _potential_state_columns(
    kline_history: pl.DataFrame, timeframe: str, prefix: str
) -> pl.DataFrame:
    frame = kline_history.filter(pl.col("timeframe") == timeframe)
    if frame.is_empty():
        return pl.DataFrame()
    if prefix == "background":
        return frame.select(
            "symbol",
            "bar_close_ms",
            pl.col("regime_state").alias("background_regime"),
            pl.col("structure_state").alias("background_structure"),
            pl.col("range_state").alias("background_range"),
            pl.col("vol_state").alias("background_vol"),
        )
    if prefix == "swing":
        return frame.with_columns(
            pl.concat_str("core_context", "transition_kind", separator="|").alias(
                "swing_transition"
            )
        ).select(
            "symbol",
            "bar_close_ms",
            pl.col("regime_state").alias("swing_regime"),
            pl.col("core_context").alias("swing_core"),
            pl.col("range_state").alias("swing_range"),
            "swing_transition",
        )
    return frame.with_columns(
        pl.concat_str("core_context", "transition_kind", separator="|").alias("decision_transition")
    ).select(
        "symbol",
        "bar_close_ms",
        pl.col("direction_hint").alias("decision_direction"),
        pl.col("regime_state").alias("decision_regime"),
        pl.col("core_context").alias("decision_core"),
        pl.col("range_state").alias("decision_range"),
        pl.col("vol_state").alias("decision_vol"),
        pl.col("event_state").alias("decision_event"),
        pl.col("event_age_bucket").alias("decision_event_age_bucket"),
        "decision_transition",
    )


def _potential_state_output_columns(prefix: str) -> tuple[str, ...]:
    if prefix == "background":
        return ("background_regime", "background_structure", "background_range", "background_vol")
    if prefix == "swing":
        return ("swing_regime", "swing_core", "swing_range", "swing_transition")
    return (
        "decision_direction",
        "decision_regime",
        "decision_core",
        "decision_range",
        "decision_vol",
        "decision_event",
        "decision_event_age_bucket",
        "decision_transition",
    )


__all__ = [
    "CLASSIFIER_HEALTH_SCHEMA",
    "ClassifierColumn",
    "ClassifierHealthResult",
    "KLINE_REQUIRED_COLUMNS",
    "KlineClassifier",
    "MarketStage",
    "STATE_FRAME_COLUMNS",
    "STATE_FRAME_SCHEMA",
    "SOURCE_FEATURE_MAX_AGE_MS",
    "StateDirection",
    "StructureState",
    "classifier_health",
    "extract_continuous_features",
    "source_time_series_features_frame",
    "validate_state_frame",
    "POTENTIAL_OBSERVATION_SCHEMA",
    "potential_observation_frame",
]
