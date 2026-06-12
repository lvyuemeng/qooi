"""Known-at-close source classifiers for research scanner workflows."""

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
        state = _classify_kline_state_frame(frame, self.scale)
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
        [[row["health_check"], row["status"], row["reason"]] for row in rows],
    )
    return ClassifierHealthResult(out, text)


def _classify_kline_state_frame(frame: pl.DataFrame, scale: str) -> pl.DataFrame:
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
