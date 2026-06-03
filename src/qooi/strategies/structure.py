"""Composable structural strategy feature builders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import polars as pl

from qooi.strategies.indicators import IndicatorFn
from qooi.strategies.semantics import (
    ClassifierColumn,
    LiquidityEvent,
    MarketStage,
    MarketStageReason,
    StageUnknownReason,
    StructureReason,
    StructureState,
)

FeatureFn = IndicatorFn


def add_momentum_return(period: int, *, output: str = "momentum_return") -> FeatureFn:
    def _add(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns((pl.col("close") / pl.col("close").shift(period) - 1).alias(output))

    return _add


def add_volume_average(period: int = 20, *, output: str = "vol_avg") -> FeatureFn:
    def _add(df: pl.DataFrame) -> pl.DataFrame:
        vol_col = "vol" if "vol" in df.columns else "volume"
        return df.with_columns(pl.col(vol_col).rolling_mean(period).alias(output))

    return _add


def add_trend_maturity(
    *,
    ema_mid: int = 50,
    ema_slow: int = 200,
    output: str = "trend_bars",
) -> FeatureFn:
    def _add(df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return df.with_columns(pl.lit(0).alias(output))
        e_mid = df[f"ema_{ema_mid}"].to_list()
        e_slow = df[f"ema_{ema_slow}"].to_list()
        bars: list[int] = []
        trend_bars = 0
        for mid, slow in zip(e_mid, e_slow, strict=False):
            uptrend = mid is not None and slow is not None and mid > 0 and slow > 0 and mid > slow
            downtrend = mid is not None and slow is not None and mid > 0 and slow > 0 and mid < slow
            if uptrend:
                trend_bars = trend_bars + 1 if trend_bars >= 0 else 1
            elif downtrend:
                trend_bars = trend_bars - 1 if trend_bars <= 0 else -1
            else:
                trend_bars = 0
            bars.append(trend_bars)
        return df.with_columns(pl.Series(output, bars, dtype=pl.Int64))

    return _add


def add_utc_hour(*, timestamp_col: str = "timestamp", output: str = "hour_utc") -> FeatureFn:
    def _add(df: pl.DataFrame) -> pl.DataFrame:
        if "datetime" in df.columns:
            return df.with_columns(pl.col("datetime").dt.hour().fill_null(12).alias(output))
        hour = ((pl.col(timestamp_col) // 3_600_000) % 24).fill_null(12)
        return df.with_columns(hour.alias(output))

    return _add


def add_price_structure(period_short: int = 5, period_long: int = 20) -> FeatureFn:
    def _add(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            pl.col("low").shift(1).rolling_min(period_short).alias("low_short"),
            pl.col("low").shift(1).rolling_min(period_long).alias("low_long"),
            pl.col("high").shift(1).rolling_max(period_short).alias("high_short"),
            pl.col("high").shift(1).rolling_max(period_long).alias("high_long"),
        )

    return _add


def add_liquidity_sweep_features(
    *,
    lookback: int = 20,
    volume_period: int = 20,
    volume_mult: float = 1.2,
    sweep_buffer_atr: float = 0.0,
    wick_min: float = 0.35,
) -> FeatureFn:
    """Add shifted liquidity sweep/reclaim diagnostics without lookahead."""

    def _add(df: pl.DataFrame) -> pl.DataFrame:
        prior_high = pl.col("high").shift(1).rolling_max(lookback)
        prior_low = pl.col("low").shift(1).rolling_min(lookback)
        atr = pl.col("atr_14") if "atr_14" in df.columns else pl.lit(None, dtype=pl.Float64)
        safe_atr = pl.when(atr.abs() > 1e-10).then(atr).otherwise(None)
        buffer = pl.when(atr.is_not_null()).then(atr * sweep_buffer_atr).otherwise(0.0)
        high_breach = (pl.col("high") - prior_high).clip(0.0, None)
        low_breach = (prior_low - pl.col("low")).clip(0.0, None)
        bar_range = (pl.col("high") - pl.col("low")).abs()
        safe_range = pl.when(bar_range > 1e-10).then(bar_range).otherwise(1e-10)
        body = (pl.col("close") - pl.col("open")).abs()
        safe_body = pl.when(body > 1e-10).then(body).otherwise(1e-10)
        lower_wick_size = pl.min_horizontal("open", "close") - pl.col("low")
        upper_wick_size = pl.col("high") - pl.max_horizontal("open", "close")
        lower_wick_body_ratio = lower_wick_size / safe_body
        upper_wick_body_ratio = upper_wick_size / safe_body
        wick_body_ratio = pl.max_horizontal(lower_wick_body_ratio, upper_wick_body_ratio)
        close_location = (pl.col("close") - pl.col("low")) / safe_range
        swept_high = (prior_high.is_not_null() & (pl.col("high") > prior_high + buffer)).fill_null(
            False
        )
        swept_low = (prior_low.is_not_null() & (pl.col("low") < prior_low - buffer)).fill_null(
            False
        )
        reclaimed_high = (swept_high & (pl.col("close") < prior_high)).fill_null(False)
        reclaimed_low = (swept_low & (pl.col("close") > prior_low)).fill_null(False)
        breakout_acceptance_high = (
            prior_high.is_not_null() & (pl.col("close") > prior_high + buffer)
        ).fill_null(False)
        breakout_acceptance_low = (
            prior_low.is_not_null() & (pl.col("close") < prior_low - buffer)
        ).fill_null(False)

        if "lower_wick_ratio" in df.columns and "upper_wick_ratio" in df.columns:
            lower_wick = pl.col("lower_wick_ratio") >= wick_min
            upper_wick = pl.col("upper_wick_ratio") >= wick_min
        else:
            lower_wick = (lower_wick_size / safe_range) >= wick_min
            upper_wick = (upper_wick_size / safe_range) >= wick_min

        bullish_rejection_bar = (reclaimed_low & lower_wick.fill_null(False)).fill_null(False)
        bearish_rejection_bar = (reclaimed_high & upper_wick.fill_null(False)).fill_null(False)
        failed_breakout_high = (swept_high & ~breakout_acceptance_high).fill_null(False)
        failed_breakout_low = (swept_low & ~breakout_acceptance_low).fill_null(False)
        event_quality_score = (
            pl.when(swept_high | swept_low)
            .then(
                pl.max_horizontal(high_breach, low_breach).truediv(safe_atr).clip(0.0, 1.0)
                + wick_body_ratio.clip(0.0, 1.0)
                + pl.max_horizontal(close_location, 1.0 - close_location).clip(0.0, 1.0)
            )
            .otherwise(0.0)
        )
        liquidity_event_type = (
            pl.when(bullish_rejection_bar)
            .then(pl.lit(LiquidityEvent.BULLISH_RECLAIM))
            .when(bearish_rejection_bar)
            .then(pl.lit(LiquidityEvent.BEARISH_RECLAIM))
            .when(breakout_acceptance_high)
            .then(pl.lit(LiquidityEvent.BREAKOUT_ACCEPTANCE_HIGH))
            .when(breakout_acceptance_low)
            .then(pl.lit(LiquidityEvent.BREAKOUT_ACCEPTANCE_LOW))
            .when(failed_breakout_high)
            .then(pl.lit(LiquidityEvent.FAILED_BREAKOUT_HIGH))
            .when(failed_breakout_low)
            .then(pl.lit(LiquidityEvent.FAILED_BREAKOUT_LOW))
            .otherwise(pl.lit(LiquidityEvent.NONE))
        )

        volume_exprs: list[pl.Expr]
        vol_col = "vol" if "vol" in df.columns else "volume" if "volume" in df.columns else ""
        if vol_col:
            vol_avg = pl.col(vol_col).shift(1).rolling_mean(volume_period)
            volume_exprs = [
                vol_avg.alias("liquidity_sweep_volume_avg"),
                (
                    vol_avg.is_not_null()
                    & (pl.col(vol_col) > vol_avg * volume_mult)
                )
                .fill_null(False)
                .alias("volume_impulse"),
            ]
        else:
            volume_exprs = [
                pl.lit(None, dtype=pl.Float64).alias("liquidity_sweep_volume_avg"),
                pl.lit(False).alias("volume_impulse"),
            ]

        if vol_col:
            volume_boost = pl.when(
                vol_avg.is_not_null() & (pl.col(vol_col) > vol_avg * volume_mult)
            ).then(1.0).otherwise(0.0)
        else:
            volume_boost = pl.lit(0.0)

        return df.with_columns(
            prior_high.alias("prior_liquidity_high"),
            prior_low.alias("prior_liquidity_low"),
            swept_high.alias("swept_high"),
            swept_low.alias("swept_low"),
            reclaimed_high.alias("reclaimed_high"),
            reclaimed_low.alias("reclaimed_low"),
            (reclaimed_low & lower_wick.fill_null(False)).alias("bullish_liquidity_sweep"),
            (reclaimed_high & upper_wick.fill_null(False)).alias("bearish_liquidity_sweep"),
            (swept_low & ~reclaimed_low).alias("failed_bullish_sweep"),
            (swept_high & ~reclaimed_high).alias("failed_bearish_sweep"),
            wick_body_ratio.alias("wick_body_ratio"),
            lower_wick_body_ratio.alias("lower_wick_body_ratio"),
            upper_wick_body_ratio.alias("upper_wick_body_ratio"),
            bullish_rejection_bar.alias("bullish_rejection_bar"),
            bearish_rejection_bar.alias("bearish_rejection_bar"),
            breakout_acceptance_high.alias("breakout_acceptance_high"),
            breakout_acceptance_low.alias("breakout_acceptance_low"),
            failed_breakout_high.alias("failed_breakout_high"),
            failed_breakout_low.alias("failed_breakout_low"),
            (event_quality_score + volume_boost).alias("event_quality_score"),
            liquidity_event_type.alias("liquidity_event_type"),
            (pl.max_horizontal(high_breach, low_breach) / safe_atr).alias("sweep_distance_atr"),
            *volume_exprs,
        )

    return _add


@dataclass(frozen=True)
class RangeWidthThresholdConfig:
    mode: Literal["fixed", "rolling_quantile"] = "rolling_quantile"
    fixed_atr_max: float = 8.0
    quantile: float = 0.65
    window: int = 480
    min_samples: int = 120
    fallback: Literal["fixed", "data_error"] = "fixed"


@dataclass(frozen=True)
class StructureClassifierConfig:
    swing_lookback: int = 5
    range_lookback: int = 48
    trend_window: int = 24
    level_proximity_atr: float = 1.0
    range_width_threshold: RangeWidthThresholdConfig = field(
        default_factory=RangeWidthThresholdConfig
    )

    @classmethod
    def default(cls) -> StructureClassifierConfig:
        return cls()

    @classmethod
    def fixed(cls, range_width_atr_max: float = 8.0) -> StructureClassifierConfig:
        return cls(
            range_width_threshold=RangeWidthThresholdConfig(
                mode="fixed",
                fixed_atr_max=range_width_atr_max,
            )
        )

    @classmethod
    def rolling_quantile(
        cls,
        *,
        quantile: float = 0.65,
        window: int = 480,
        min_samples: int = 120,
        fallback: Literal["fixed", "data_error"] = "fixed",
        fixed_atr_max: float = 8.0,
    ) -> StructureClassifierConfig:
        return cls(
            range_width_threshold=RangeWidthThresholdConfig(
                mode="rolling_quantile",
                fixed_atr_max=fixed_atr_max,
                quantile=quantile,
                window=window,
                min_samples=min_samples,
                fallback=fallback,
            )
        )


def add_price_structure_stage_features(
    *,
    config: StructureClassifierConfig | None = None,
    swing_lookback: int = 5,
    range_lookback: int = 48,
    trend_window: int = 24,
    range_width_atr_max: float | None = None,
    level_proximity_atr: float = 1.0,
) -> FeatureFn:
    """Add no-lookahead structure, range, and coarse lifecycle diagnostics."""

    if config is not None:
        normalized = config
    elif range_width_atr_max is None:
        normalized = StructureClassifierConfig(
            swing_lookback=swing_lookback,
            range_lookback=range_lookback,
            trend_window=trend_window,
            level_proximity_atr=level_proximity_atr,
        )
    else:
        normalized = StructureClassifierConfig(
            swing_lookback=swing_lookback,
            range_lookback=range_lookback,
            trend_window=trend_window,
            level_proximity_atr=level_proximity_atr,
            range_width_threshold=RangeWidthThresholdConfig(
                mode="fixed",
                fixed_atr_max=range_width_atr_max,
            ),
        )

    def _add(df: pl.DataFrame) -> pl.DataFrame:
        threshold_config = normalized.range_width_threshold
        atr = pl.col("atr_14") if "atr_14" in df.columns else pl.lit(None, dtype=pl.Float64)
        safe_atr = pl.when(atr.abs() > 1e-10).then(atr).otherwise(None)
        prior_high_window = pl.col("high").shift(2).rolling_max(normalized.swing_lookback)
        prior_low_window = pl.col("low").shift(2).rolling_min(normalized.swing_lookback)
        swing_high = prior_high_window.is_not_null() & (
            pl.col("high").shift(1) >= prior_high_window
        )
        swing_low = prior_low_window.is_not_null() & (
            pl.col("low").shift(1) <= prior_low_window
        )
        swing_high_value = pl.when(swing_high).then(pl.col("high").shift(1)).otherwise(None)
        swing_low_value = pl.when(swing_low).then(pl.col("low").shift(1)).otherwise(None)
        last_swing_high = swing_high_value.forward_fill()
        last_swing_low = swing_low_value.forward_fill()
        prev_swing_high = last_swing_high.shift(1)
        prev_swing_low = last_swing_low.shift(1)
        higher_high = (
            swing_high & prev_swing_high.is_not_null() & (pl.col("high").shift(1) > prev_swing_high)
        ).fill_null(False)
        lower_high = (
            swing_high & prev_swing_high.is_not_null() & (pl.col("high").shift(1) < prev_swing_high)
        ).fill_null(False)
        higher_low = (
            swing_low & prev_swing_low.is_not_null() & (pl.col("low").shift(1) > prev_swing_low)
        ).fill_null(False)
        lower_low = (
            swing_low & prev_swing_low.is_not_null() & (pl.col("low").shift(1) < prev_swing_low)
        ).fill_null(False)
        hh_count = higher_high.cast(pl.Int64).rolling_sum(normalized.trend_window)
        hl_count = higher_low.cast(pl.Int64).rolling_sum(normalized.trend_window)
        lh_count = lower_high.cast(pl.Int64).rolling_sum(normalized.trend_window)
        ll_count = lower_low.cast(pl.Int64).rolling_sum(normalized.trend_window)
        range_high = pl.col("high").shift(1).rolling_max(normalized.range_lookback)
        range_low = pl.col("low").shift(1).rolling_min(normalized.range_lookback)
        range_mid = (range_high + range_low) / 2.0
        range_width_atr = (range_high - range_low) / safe_atr
        range_ready = range_high.is_not_null() & range_low.is_not_null()
        data_error = range_ready & (
            pl.col("high").is_null()
            | pl.col("low").is_null()
            | pl.col("close").is_null()
            | safe_atr.is_null()
        )
        if threshold_config.mode == "fixed":
            threshold = pl.lit(threshold_config.fixed_atr_max, dtype=pl.Float64)
            threshold_ready = pl.lit(True)
            threshold_source = pl.lit("fixed")
        else:
            rolling_threshold = range_width_atr.shift(1).rolling_quantile(
                threshold_config.quantile,
                window_size=threshold_config.window,
                min_samples=threshold_config.min_samples,
            )
            rolling_ready = rolling_threshold.is_not_null()
            threshold = (
                pl.when(rolling_ready)
                .then(rolling_threshold)
                .when(threshold_config.fallback == "fixed")
                .then(pl.lit(threshold_config.fixed_atr_max, dtype=pl.Float64))
                .otherwise(None)
            )
            threshold_ready = rolling_ready | pl.lit(threshold_config.fallback == "fixed")
            threshold_source = (
                pl.when(rolling_ready)
                .then(pl.lit("rolling_quantile"))
                .when(threshold_config.fallback == "fixed")
                .then(pl.lit("fixed_fallback"))
                .otherwise(pl.lit("data_error"))
            )
        range_compression = (range_ready & (range_width_atr <= threshold)).fill_null(False)
        near_range_high = (
            range_ready
            & ((range_high - pl.col("close")).abs() <= safe_atr * normalized.level_proximity_atr)
        ).fill_null(False)
        near_range_low = (
            range_ready
            & ((pl.col("close") - range_low).abs() <= safe_atr * normalized.level_proximity_atr)
        ).fill_null(False)
        uptrend = (hh_count > 0) & (hl_count > 0) & (hh_count + hl_count >= lh_count + ll_count)
        downtrend = (lh_count > 0) & (ll_count > 0) & (lh_count + ll_count >= hh_count + hl_count)
        trend_conflict = (uptrend & downtrend).fill_null(False)
        markup = ((pl.col("close") > range_high) & uptrend & ~trend_conflict).fill_null(False)
        markdown = ((pl.col("close") < range_low) & downtrend & ~trend_conflict).fill_null(False)
        trend_continuation = (
            (uptrend | downtrend) & ~markup & ~markdown & ~trend_conflict
        ).fill_null(False)
        wide_range = (range_ready & ~range_compression & ~(uptrend | downtrend)).fill_null(False)
        transition = (range_ready & trend_conflict).fill_null(False)
        structure_trend = (
            pl.when(~range_ready)
            .then(pl.lit(StructureState.UNKNOWN))
            .when(data_error | trend_conflict)
            .then(pl.lit(StructureState.UNKNOWN))
            .when(uptrend)
            .then(pl.lit(StructureState.UPTREND))
            .when(downtrend)
            .then(pl.lit(StructureState.DOWNTREND))
            .when(range_compression)
            .then(pl.lit(StructureState.RANGE))
            .otherwise(pl.lit(StructureState.UNKNOWN))
        )
        structure_reason = (
            pl.when(~range_ready)
            .then(pl.lit(StructureReason.WARMUP))
            .when(data_error)
            .then(pl.lit(StructureReason.DATA_ERROR))
            .when(trend_conflict)
            .then(pl.lit(StructureReason.AMBIGUOUS_TRANSITION))
            .when(uptrend)
            .then(pl.lit(StructureReason.HIGHER_HIGH_HIGHER_LOW))
            .when(downtrend)
            .then(pl.lit(StructureReason.LOWER_HIGH_LOWER_LOW))
            .when(range_compression)
            .then(pl.lit(StructureReason.COMPRESSED_RANGE))
            .otherwise(pl.lit(StructureReason.AMBIGUOUS_STRUCTURE))
        )
        market_stage = (
            pl.when(~range_ready)
            .then(pl.lit(MarketStage.WARMUP))
            .when(data_error)
            .then(pl.lit(MarketStage.DATA_ERROR))
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
        market_stage_reason = (
            pl.when(~range_ready)
            .then(pl.lit(MarketStageReason.WARMUP))
            .when(data_error)
            .then(pl.lit(MarketStageReason.DATA_ERROR))
            .when(markup)
            .then(pl.lit(MarketStageReason.MARKUP_BREAKOUT))
            .when(markdown)
            .then(pl.lit(MarketStageReason.MARKDOWN_BREAKOUT))
            .when(transition)
            .then(pl.lit(MarketStageReason.AMBIGUOUS_TRANSITION))
            .when(range_compression & near_range_low)
            .then(pl.lit(MarketStageReason.COMPRESSED_NEAR_LOW))
            .when(range_compression & near_range_high)
            .then(pl.lit(MarketStageReason.COMPRESSED_NEAR_HIGH))
            .when(range_compression)
            .then(pl.lit(MarketStageReason.COMPRESSED_MID_RANGE))
            .when(trend_continuation)
            .then(pl.lit(MarketStageReason.TREND_WITHOUT_RANGE_BREAK))
            .when(wide_range)
            .then(pl.lit(MarketStageReason.WIDE_RANGE_NO_STAGE))
            .otherwise(pl.lit(MarketStageReason.UNKNOWN_UNHANDLED))
        )
        stage_unknown_reason = (
            pl.when(~range_ready)
            .then(pl.lit(StageUnknownReason.WARMUP))
            .when(data_error)
            .then(pl.lit(StageUnknownReason.DATA_ERROR))
            .when(trend_continuation)
            .then(pl.lit(StageUnknownReason.NONE))
            .when(wide_range)
            .then(pl.lit(StageUnknownReason.WIDE_RANGE))
            .when(transition)
            .then(pl.lit(StageUnknownReason.TRANSITION))
            .otherwise(pl.lit(StageUnknownReason.NONE))
        )

        return df.with_columns(
            swing_high.fill_null(False).alias("swing_high_confirmed"),
            swing_low.fill_null(False).alias("swing_low_confirmed"),
            last_swing_high.alias("last_swing_high"),
            last_swing_low.alias("last_swing_low"),
            higher_high.alias("structure_higher_high"),
            higher_low.alias("structure_higher_low"),
            lower_high.alias("structure_lower_high"),
            lower_low.alias("structure_lower_low"),
            structure_trend.alias(ClassifierColumn.STRUCTURE_TREND_STATE),
            range_high.alias("range_high"),
            range_low.alias("range_low"),
            range_mid.alias("range_mid"),
            range_width_atr.alias(ClassifierColumn.RANGE_WIDTH_ATR),
            threshold.alias(ClassifierColumn.RANGE_WIDTH_ATR_THRESHOLD),
            pl.lit(threshold_config.mode).alias(ClassifierColumn.RANGE_WIDTH_THRESHOLD_MODE),
            threshold_ready.fill_null(False).alias(ClassifierColumn.RANGE_WIDTH_THRESHOLD_READY),
            threshold_source.alias(ClassifierColumn.RANGE_WIDTH_THRESHOLD_SOURCE),
            range_compression.alias("range_compression"),
            near_range_high.alias("near_range_high"),
            near_range_low.alias("near_range_low"),
            market_stage.alias(ClassifierColumn.MARKET_STAGE),
            structure_reason.alias(ClassifierColumn.STRUCTURE_REASON),
            market_stage_reason.alias(ClassifierColumn.MARKET_STAGE_REASON),
            stage_unknown_reason.alias(ClassifierColumn.STAGE_UNKNOWN_REASON),
        )

    return _add


def add_volatility_scaled_return(
    ret_period: int = 1,
    vol_span: int = 48,
    *,
    output: str = "vol_scaled_return",
) -> FeatureFn:
    def _add(df: pl.DataFrame) -> pl.DataFrame:
        ret = (pl.col("close") / pl.col("close").shift(ret_period)).log()
        vol = (ret**2).ewm_mean(span=vol_span, min_samples=vol_span).sqrt()
        safe_vol = pl.when(vol.abs() > 1e-10).then(vol).otherwise(1e-10)
        return df.with_columns((ret / safe_vol).alias(output))

    return _add


def add_none_context_diagnostics(
    *,
    atr_period: int = 100,
    proximity_atr: float = 1.0,
) -> FeatureFn:
    """Add diagnostic-only context for residual liquidity-event ``none`` trades."""

    def _add(df: pl.DataFrame) -> pl.DataFrame:
        atr_values = [float(v) if v is not None else None for v in df["atr_14"].to_list()]
        percentiles: list[float | None] = []
        for idx, value in enumerate(atr_values):
            window = [
                v
                for v in atr_values[max(0, idx - atr_period + 1) : idx + 1]
                if v is not None
            ]
            if value is None or len(window) < atr_period:
                percentiles.append(None)
                continue
            rank = sum(1 for v in window if v <= value)
            percentiles.append(rank / len(window) * 100.0)

        atr_pct = pl.col("atr_percentile_100").cast(pl.Float64)
        atr_bucket = (
            pl.when(atr_pct.is_null())
            .then(pl.lit("unknown"))
            .when(atr_pct < 25.0)
            .then(pl.lit("low"))
            .when(atr_pct < 75.0)
            .then(pl.lit("normal"))
            .when(atr_pct < 90.0)
            .then(pl.lit("high"))
            .otherwise(pl.lit("extreme"))
        )
        atr = pl.col("atr_14") if "atr_14" in df.columns else pl.lit(None, dtype=pl.Float64)
        safe_atr = pl.when(atr.abs() > 1e-10).then(atr).otherwise(None)
        prior_high = (
            pl.col("prior_liquidity_high")
            if "prior_liquidity_high" in df.columns
            else pl.lit(None, dtype=pl.Float64)
        )
        prior_low = (
            pl.col("prior_liquidity_low")
            if "prior_liquidity_low" in df.columns
            else pl.lit(None, dtype=pl.Float64)
        )
        distance_high = (prior_high - pl.col("close")).abs() / safe_atr
        distance_low = (pl.col("close") - prior_low).abs() / safe_atr
        near_high = (
            prior_high.is_not_null()
            & safe_atr.is_not_null()
            & (distance_high <= proximity_atr)
            & (pl.col("high") <= prior_high)
        ).fill_null(False)
        near_low = (
            prior_low.is_not_null()
            & safe_atr.is_not_null()
            & (distance_low <= proximity_atr)
            & (pl.col("low") >= prior_low)
        ).fill_null(False)
        proximity_bucket = (
            pl.when(near_high)
            .then(pl.lit("near_prior_high_no_breach"))
            .when(near_low)
            .then(pl.lit("near_prior_low_no_breach"))
            .when(prior_high.is_not_null() & prior_low.is_not_null() & safe_atr.is_not_null())
            .then(pl.lit("mid_range_far_from_key_level"))
            .otherwise(pl.lit("breached_or_unknown"))
        )
        return df.with_columns(
            pl.Series("atr_percentile_100", percentiles, dtype=pl.Float64),
        ).with_columns(
            atr_bucket.alias("atr_percentile_bucket"),
            distance_high.alias("distance_to_prior_high_atr"),
            distance_low.alias("distance_to_prior_low_atr"),
            near_high.alias("near_prior_high_no_breach"),
            near_low.alias("near_prior_low_no_breach"),
            proximity_bucket.alias("key_level_proximity_bucket"),
        )

    return _add
