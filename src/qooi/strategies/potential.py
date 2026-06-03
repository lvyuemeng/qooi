'Potential-coin board construction, feature scoring, and report rendering.'

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import polars as pl

from qooi.accumulation.config import PotentialScanConfig
from qooi.accumulation.features import HOUR_MS

POTENTIAL_FEATURE_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Int64,
    "symbol": pl.String,
    "close": pl.Float64,
    "quote_volume_24h": pl.Float64,
    "history_hours": pl.Int64,
    "price_to_90d_low": pl.Float64,
    "price_to_30d_high": pl.Float64,
    "range_position_90d_pct": pl.Float64,
    "range_width_90d_pct": pl.Float64,
    "bb_width_20d_pct": pl.Float64,
    "bb_width_percentile_90d": pl.Float64,
    "volume_contraction_10d_90d": pl.Float64,
    "volume_spike_ratio_1h_20h": pl.Float64,
    "prior_spike_count_5d": pl.Int64,
    "first_volume_expansion": pl.Boolean,
    "return_1h": pl.Float64,
    "return_24h": pl.Float64,
    "return_72h": pl.Float64,
    "return_30d": pl.Float64,
    "return_60d": pl.Float64,
    "return_90d": pl.Float64,
    "drawdown_30d_pct": pl.Float64,
    "new_low_15d": pl.Boolean,
    "new_low_count_30d": pl.Int64,
    "higher_low_count_30d": pl.Int64,
    "base_duration_hours": pl.Int64,
    "ma_7d": pl.Float64,
    "ma_30d": pl.Float64,
    "price_vs_ma_7d_pct": pl.Float64,
    "price_vs_ma_30d_pct": pl.Float64,
    "ma_7d_slope_7d": pl.Float64,
    "ma_30d_slope_14d": pl.Float64,
    "downtrend_deceleration": pl.Boolean,
    "reclaim_state": pl.String,
    "structure_block_reason": pl.String,
    "vwap_24h": pl.Float64,
    "price_vs_vwap_24h_pct": pl.Float64,
    "vwap_slope_24h": pl.Float64,
    "taker_buy_ratio": pl.Float64,
    "taker_volume_imbalance": pl.Float64,
    "open_interest_usd_change_24h": pl.Float64,
    "net_exchange_flow": pl.Float64,
    "flow_zscore": pl.Float64,
    "whale_accumulation_ratio": pl.Float64,
    "depth_imbalance_25_mean": pl.Float64,
    "large_trade_buy_ratio": pl.Float64,
    "mention_growth": pl.Float64,
    "fundamental_news_ratio": pl.Float64,
    "emotion_news_ratio": pl.Float64,
    "source_coverage_score": pl.Float64,
    "data_quality_warning": pl.String,
}

POTENTIAL_CANDIDATE_SCHEMA: dict[str, pl.DataType] = {
    "rank": pl.Int64,
    "timestamp": pl.Int64,
    "symbol": pl.String,
    "base_ccy": pl.String,
    "market_cap_usd": pl.Float64,
    "volume_24h_usd": pl.Float64,
    "potential_score": pl.Float64,
    "stage": pl.String,
    "stage_confidence": pl.String,
    "setup_quality": pl.String,
    "confirmation_state": pl.String,
    "risk_state": pl.String,
    "action_state": pl.String,
    "board_priority": pl.String,
    "setup_pass_reason": pl.String,
    "funds_state": pl.String,
    "funds_score_total": pl.Float64,
    "funds_positive_reasons": pl.String,
    "funds_negative_reasons": pl.String,
    "next_confirmation_needed": pl.String,
    "evidence_families_present": pl.String,
    "price_to_90d_low": pl.Float64,
    "bb_width_percentile_90d": pl.Float64,
    "range_position_90d_pct": pl.Float64,
    "base_duration_hours": pl.Int64,
    "new_low_count_30d": pl.Int64,
    "higher_low_count_30d": pl.Int64,
    "price_vs_ma_7d_pct": pl.Float64,
    "price_vs_ma_30d_pct": pl.Float64,
    "ma_7d_slope_7d": pl.Float64,
    "ma_30d_slope_14d": pl.Float64,
    "reclaim_state": pl.String,
    "structure_block_reason": pl.String,
    "volume_spike_ratio_1h_20h": pl.Float64,
    "first_volume_expansion": pl.Boolean,
    "price_vs_vwap_24h_pct": pl.Float64,
    "taker_buy_ratio": pl.Float64,
    "open_interest_usd_change_24h": pl.Float64,
    "net_exchange_flow": pl.Float64,
    "flow_zscore": pl.Float64,
    "whale_accumulation_ratio": pl.Float64,
    "depth_imbalance_25_mean": pl.Float64,
    "large_trade_buy_ratio": pl.Float64,
    "mention_growth": pl.Float64,
    "strict_score_total": pl.Int64,
    "strict_alert_level": pl.String,
    "broad_rank": pl.Int64,
    "broad_score": pl.Float64,
    "broad_reasons": pl.String,
    "positive_reasons": pl.String,
    "risk_reasons": pl.String,
    "missing_evidence": pl.String,
    "data_quality_warning": pl.String,
}

POTENTIAL_BOARD_SCHEMA: dict[str, pl.DataType] = {
    "rank": pl.Int64,
    "timestamp": pl.Int64,
    "symbol": pl.String,
    "base_ccy": pl.String,
    "market_data_provider": pl.String,
    "attention_source": pl.String,
    "okx_mapped": pl.Boolean,
    "market_cap_usd": pl.Float64,
    "volume_24h_usd": pl.Float64,
    "price_change_pct_1h": pl.Float64,
    "price_change_pct_24h": pl.Float64,
    "gate_state": pl.String,
    "gate_reason": pl.String,
    "structure_state": pl.String,
    "structure_score": pl.Float64,
    "base_duration_hours": pl.Int64,
    "new_low_count_30d": pl.Int64,
    "reclaim_state": pl.String,
    "ma_7d_slope_7d": pl.Float64,
    "ma_30d_slope_14d": pl.Float64,
    "structure_blockers": pl.String,
    "funds_state": pl.String,
    "funds_score": pl.Float64,
    "funds_positive": pl.String,
    "funds_negative": pl.String,
    "evidence_families": pl.String,
    "board_bucket": pl.String,
    "decision_reason": pl.String,
    "next_confirmation": pl.String,
    "data_warning": pl.String,
}

POTENTIAL_SOURCE_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": pl.Int64,
    "source": pl.String,
    "provider": pl.String,
    "endpoint": pl.String,
    "status": pl.String,
    "rows": pl.Int64,
    "warning": pl.String,
    "elapsed_ms": pl.Int64,
}


def empty_potential_feature_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=POTENTIAL_FEATURE_SCHEMA)


def empty_potential_candidate_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=POTENTIAL_CANDIDATE_SCHEMA)


def empty_potential_board_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=POTENTIAL_BOARD_SCHEMA)


def empty_potential_source_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=POTENTIAL_SOURCE_SCHEMA)


def compute_potential_features_batch(
    bars: pl.DataFrame,
    strict_features: pl.DataFrame | None = None,
    *,
    min_history_hours: int = 720,
    full_history_hours: int = 2160,
    volume_spike_ratio: float = 3.0,
    first_spike_lookback_hours: int = 120,
) -> pl.DataFrame:
    if bars.is_empty():
        return empty_potential_feature_frame()
    if "symbol" not in bars.columns:
        raise ValueError("potential features require a symbol column")
    frame = _base_bar_features(
        bars,
        full_history_hours=full_history_hours,
        volume_spike_ratio=volume_spike_ratio,
        first_spike_lookback_hours=first_spike_lookback_hours,
    )
    frame = _add_base_duration(frame)
    frame = _join_strict_overlay(
        frame, strict_features if strict_features is not None else pl.DataFrame()
    )
    frame = frame.with_columns(
        pl.when(pl.col("history_hours") < min_history_hours)
        .then(pl.lit("insufficient_history"))
        .otherwise(pl.col("data_quality_warning").fill_null(""))
        .alias("data_quality_warning")
    )
    return _coerce_potential_features(frame)


def _base_bar_features(
    bars: pl.DataFrame,
    *,
    full_history_hours: int,
    volume_spike_ratio: float,
    first_spike_lookback_hours: int,
) -> pl.DataFrame:
    frame = (
        bars.select(
            [
                pl.col("symbol").cast(pl.String),
                (pl.col("timestamp").cast(pl.Int64) // HOUR_MS * HOUR_MS).alias("timestamp"),
                pl.col("close").cast(pl.Float64),
                pl.col("vol").cast(pl.Float64).fill_null(0.0).alias("vol"),
            ]
        )
        .sort(["symbol", "timestamp"])
        .group_by(["symbol", "timestamp"])
        .last()
        .sort(["symbol", "timestamp"])
    )
    return (
        frame.with_columns(
            [
                pl.cum_count("timestamp").over("symbol").cast(pl.Int64).alias("history_hours"),
                (pl.col("close") * pl.col("vol")).alias("_quote_volume"),
                pl.col("close")
                .rolling_min(full_history_hours, min_samples=2)
                .over("symbol")
                .alias("_low_90d"),
                pl.col("close")
                .rolling_max(full_history_hours, min_samples=2)
                .over("symbol")
                .alias("_high_90d"),
                pl.col("close").rolling_max(720, min_samples=2).over("symbol").alias("_high_30d"),
                pl.col("close").rolling_min(360, min_samples=2).over("symbol").alias("_low_15d"),
                pl.col("close").rolling_max(720, min_samples=2).over("symbol").alias("_peak_30d"),
                pl.col("close").rolling_mean(480, min_samples=20).over("symbol").alias("_bb_mid"),
                pl.col("close").rolling_std(480, min_samples=20).over("symbol").alias("_bb_std"),
                pl.col("vol")
                .rolling_mean(240, min_samples=24)
                .over("symbol")
                .alias("_vol_mean_10d"),
                pl.col("vol")
                .rolling_mean(full_history_hours, min_samples=24)
                .over("symbol")
                .alias("_vol_mean_90d"),
                pl.col("vol").rolling_mean(20, min_samples=2).over("symbol").alias("_vol_mean_20h"),
                (pl.col("close") / pl.col("close").shift(1).over("symbol") - 1.0).alias(
                    "return_1h"
                ),
                (pl.col("close") / pl.col("close").shift(24).over("symbol") - 1.0).alias(
                    "return_24h"
                ),
                (pl.col("close") / pl.col("close").shift(72).over("symbol") - 1.0).alias(
                    "return_72h"
                ),
                (pl.col("close") / pl.col("close").shift(720).over("symbol") - 1.0).alias(
                    "return_30d"
                ),
                (pl.col("close") / pl.col("close").shift(1440).over("symbol") - 1.0).alias(
                    "return_60d"
                ),
                (
                    pl.col("close") / pl.col("close").shift(full_history_hours).over("symbol")
                    - 1.0
                ).alias("return_90d"),
                pl.col("close").rolling_mean(168, min_samples=24).over("symbol").alias("ma_7d"),
                pl.col("close")
                .rolling_mean(720, min_samples=120)
                .over("symbol")
                .alias("ma_30d"),
                pl.col("close").rolling_min(24, min_samples=2).over("symbol").alias("_low_24h"),
                pl.col("close")
                .rolling_min(720, min_samples=2)
                .over("symbol")
                .alias("_low_30d_roll"),
            ]
        )
        .with_columns(
            pl.col("_low_30d_roll").shift(1).over("symbol").alias("_prior_low_30d")
        )
        .with_columns(
            [
                pl.col("_quote_volume")
                .rolling_sum(24, min_samples=1)
                .over("symbol")
                .alias("quote_volume_24h"),
                pl.when(pl.col("_low_90d") > 0.0)
                .then(pl.col("close") / pl.col("_low_90d"))
                .otherwise(None)
                .alias("price_to_90d_low"),
                pl.when(pl.col("close") > 0.0)
                .then(pl.col("_high_30d") / pl.col("close"))
                .otherwise(None)
                .alias("price_to_30d_high"),
                pl.when((pl.col("_high_90d") - pl.col("_low_90d")) > 0.0)
                .then(
                    (pl.col("close") - pl.col("_low_90d"))
                    / (pl.col("_high_90d") - pl.col("_low_90d"))
                )
                .otherwise(None)
                .alias("range_position_90d_pct"),
                pl.when(pl.col("close") > 0.0)
                .then((pl.col("_high_90d") - pl.col("_low_90d")) / pl.col("close"))
                .otherwise(None)
                .alias("range_width_90d_pct"),
                pl.when(pl.col("_bb_mid") > 0.0)
                .then((pl.col("_bb_std") * 4.0) / pl.col("_bb_mid"))
                .otherwise(None)
                .alias("bb_width_20d_pct"),
                pl.when(pl.col("_vol_mean_90d") > 0.0)
                .then(pl.col("_vol_mean_10d") / pl.col("_vol_mean_90d"))
                .otherwise(None)
                .alias("volume_contraction_10d_90d"),
                pl.when(pl.col("_vol_mean_20h") > 0.0)
                .then(pl.col("vol") / pl.col("_vol_mean_20h"))
                .otherwise(None)
                .alias("volume_spike_ratio_1h_20h"),
                pl.when(pl.col("_peak_30d") > 0.0)
                .then(pl.col("close") / pl.col("_peak_30d") - 1.0)
                .otherwise(None)
                .alias("drawdown_30d_pct"),
                (pl.col("close") <= pl.col("_low_15d")).fill_null(False).alias("new_low_15d"),
                (pl.col("close") < pl.col("_prior_low_30d")).fill_null(False).alias(
                    "_new_low_30d"
                ),
                pl.col("_quote_volume")
                .rolling_sum(24, min_samples=1)
                .over("symbol")
                .alias("_vwap_num"),
                pl.col("vol").rolling_sum(24, min_samples=1).over("symbol").alias("_vwap_den"),
            ]
        )
        .with_columns(
            [
                (pl.col("volume_spike_ratio_1h_20h") >= volume_spike_ratio)
                .cast(pl.Int64)
                .alias("_is_spike"),
                pl.when(pl.col("_vwap_den") > 0.0)
                .then(pl.col("_vwap_num") / pl.col("_vwap_den"))
                .otherwise(None)
                .alias("vwap_24h"),
            ]
        )
        .with_columns(
            _rolling_percentile_expr("bb_width_20d_pct", full_history_hours).over("symbol")
        )
        .with_columns(
            [
                pl.col("_is_spike")
                .shift(1)
                .rolling_sum(first_spike_lookback_hours, min_samples=1)
                .over("symbol")
                .fill_null(0)
                .cast(pl.Int64)
                .alias("prior_spike_count_5d"),
                pl.when(pl.col("vwap_24h") > 0.0)
                .then(pl.col("close") / pl.col("vwap_24h") - 1.0)
                .otherwise(None)
                .alias("price_vs_vwap_24h_pct"),
                (pl.col("vwap_24h") / pl.col("vwap_24h").shift(24).over("symbol") - 1.0).alias(
                    "vwap_slope_24h"
                ),
                pl.when(pl.col("ma_7d") > 0.0)
                .then(pl.col("close") / pl.col("ma_7d") - 1.0)
                .otherwise(None)
                .alias("price_vs_ma_7d_pct"),
                pl.when(pl.col("ma_30d") > 0.0)
                .then(pl.col("close") / pl.col("ma_30d") - 1.0)
                .otherwise(None)
                .alias("price_vs_ma_30d_pct"),
                (pl.col("ma_7d") / pl.col("ma_7d").shift(168).over("symbol") - 1.0).alias(
                    "ma_7d_slope_7d"
                ),
                (
                    pl.col("ma_30d") / pl.col("ma_30d").shift(336).over("symbol") - 1.0
                ).alias("ma_30d_slope_14d"),
                pl.col("_new_low_30d")
                .cast(pl.Int64)
                .rolling_sum(720, min_samples=1)
                .over("symbol")
                .cast(pl.Int64)
                .alias("new_low_count_30d"),
                (pl.col("_low_24h") > pl.col("_low_24h").shift(24).over("symbol"))
                .cast(pl.Int64)
                .rolling_sum(720, min_samples=1)
                .over("symbol")
                .fill_null(0)
                .cast(pl.Int64)
                .alias("higher_low_count_30d"),
            ]
        )
        .with_columns(
            (
                (pl.col("volume_spike_ratio_1h_20h") >= volume_spike_ratio)
                & (pl.col("prior_spike_count_5d") == 0)
            ).alias("first_volume_expansion")
        )
        .with_columns(pl.lit("").alias("data_quality_warning"))
        .with_columns(
            [
                (
                    (pl.col("ma_30d_slope_14d") >= -0.03)
                    & (
                        pl.col("return_30d").fill_null(-1.0)
                        >= pl.col("return_60d").fill_null(-1.0)
                    )
                )
                .fill_null(False)
                .alias("downtrend_deceleration"),
                _reclaim_state_expr(),
            ]
        )
        .with_columns(_structure_block_reason_expr())
    )


def _reclaim_state_expr() -> pl.Expr:
    return (
        pl.when(
            (pl.col("price_vs_ma_30d_pct") >= 0.0)
            & (pl.col("ma_7d_slope_7d").fill_null(0.0) >= 0.0)
        )
        .then(pl.lit("reclaim_hold"))
        .when(pl.col("price_vs_ma_30d_pct") >= 0.0)
        .then(pl.lit("ma30_reclaim"))
        .when(pl.col("price_vs_ma_7d_pct") >= 0.0)
        .then(pl.lit("ma7_reclaim"))
        .when(pl.col("price_vs_vwap_24h_pct") >= 0.0)
        .then(pl.lit("vwap_reclaim"))
        .otherwise(pl.lit("below_ma"))
        .alias("reclaim_state")
    )


def _structure_block_reason_expr() -> pl.Expr:
    active_lows = pl.when(pl.col("new_low_count_30d") > 2).then(
        pl.lit("active_lower_lows;")
    ).otherwise(pl.lit(""))
    ma_down = pl.when(pl.col("ma_30d_slope_14d") < -0.03).then(
        pl.lit("ma30_downtrend;")
    ).otherwise(pl.lit(""))
    no_reclaim = pl.when(pl.col("reclaim_state") == "below_ma").then(
        pl.lit("no_reclaim;")
    ).otherwise(pl.lit(""))
    falling = pl.when(
        (pl.col("new_low_count_30d") > 2)
        & (pl.col("ma_30d_slope_14d").fill_null(0.0) < -0.03)
    ).then(pl.lit("falling_knife;")).otherwise(pl.lit(""))
    return (active_lows + ma_down + no_reclaim + falling).str.strip_chars(";").alias(
        "structure_block_reason"
    )


def _add_base_duration(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame.with_columns(pl.lit(None).cast(pl.Int64).alias("base_duration_hours"))
    base_zone = (
        (pl.col("price_to_90d_low").fill_null(99.0) <= 1.30)
        | (pl.col("range_position_90d_pct").fill_null(1.0) <= 0.35)
        | (pl.col("bb_width_percentile_90d").fill_null(1.0) <= 0.30)
    )
    reset = pl.col("_new_low_30d").fill_null(False) | ~base_zone
    return (
        frame.sort(["symbol", "timestamp"])
        .with_columns(reset.alias("_base_reset"))
        .with_columns(
            pl.col("_base_reset").cast(pl.Int64).cum_sum().over("symbol").alias("_base_reset_group")
        )
        .with_columns(
            pl.when(pl.col("_base_reset"))
            .then(pl.lit(0))
            .otherwise(
                (~pl.col("_base_reset"))
                .cast(pl.Int64)
                .cum_sum()
                .over(["symbol", "_base_reset_group"])
            )
            .cast(pl.Int64)
            .alias("base_duration_hours")
        )
        .drop(["_base_reset", "_base_reset_group"])
    )


def _rolling_percentile_expr(col: str, window_size: int) -> pl.Expr:
    current = pl.col(col)
    low = current.rolling_min(window_size=window_size, min_samples=20)
    high = current.rolling_max(window_size=window_size, min_samples=20)
    spread = high - low
    return (
        pl.when(current.is_null() | (spread <= 0.0))
        .then(None)
        .otherwise((current - low) / spread)
        .clip(0.0, 1.0)
        .alias("bb_width_percentile_90d")
    )


def _join_strict_overlay(frame: pl.DataFrame, strict_features: pl.DataFrame) -> pl.DataFrame:
    overlay_cols = [
        "timestamp",
        "symbol",
        "taker_buy_ratio",
        "taker_volume_imbalance",
        "open_interest_usd_change_24h",
        "net_exchange_flow",
        "flow_zscore",
        "whale_accumulation_ratio",
        "depth_imbalance_25_mean",
        "large_trade_buy_ratio",
        "mention_growth",
        "fundamental_news_ratio",
        "emotion_news_ratio",
        "source_coverage_score",
    ]
    if strict_features.is_empty() or not {"timestamp", "symbol"}.issubset(strict_features.columns):
        return frame.with_columns(
            [
                pl.lit(None).cast(pl.Float64).alias("taker_buy_ratio"),
                pl.lit(None).cast(pl.Float64).alias("taker_volume_imbalance"),
                pl.lit(None).cast(pl.Float64).alias("open_interest_usd_change_24h"),
                pl.lit(None).cast(pl.Float64).alias("net_exchange_flow"),
                pl.lit(None).cast(pl.Float64).alias("flow_zscore"),
                pl.lit(None).cast(pl.Float64).alias("whale_accumulation_ratio"),
                pl.lit(None).cast(pl.Float64).alias("depth_imbalance_25_mean"),
                pl.lit(None).cast(pl.Float64).alias("large_trade_buy_ratio"),
                pl.lit(None).cast(pl.Float64).alias("mention_growth"),
                pl.lit(None).cast(pl.Float64).alias("fundamental_news_ratio"),
                pl.lit(None).cast(pl.Float64).alias("emotion_news_ratio"),
                pl.lit(None).cast(pl.Float64).alias("source_coverage_score"),
            ]
        )
    overlay = strict_features.select(
        [
            pl.col(col).cast(POTENTIAL_FEATURE_SCHEMA[col], strict=False).alias(col)
            for col in overlay_cols
            if col in strict_features.columns
        ]
    )
    return frame.join(overlay, on=["timestamp", "symbol"], how="left")


def _coerce_potential_features(frame: pl.DataFrame) -> pl.DataFrame:
    for col, dtype in POTENTIAL_FEATURE_SCHEMA.items():
        if col not in frame.columns:
            frame = frame.with_columns(pl.lit(None).cast(dtype).alias(col))
        else:
            frame = frame.with_columns(pl.col(col).cast(dtype, strict=False).alias(col))
    return frame.select(POTENTIAL_FEATURE_SCHEMA.keys()).sort(["symbol", "timestamp"])


@dataclass(frozen=True)
class PotentialStageThresholds:
    near_low_ratio: float = 1.30
    compression_pctile_max: float = 0.30
    volume_contraction_max: float = 0.60
    volume_spike_ratio: float = 3.0
    range_low_max_pct: float = 0.35
    late_range_min_pct: float = 0.80
    taker_buy_confirm_min: float = 0.65
    min_history_hours: int = 720
    min_volume_24h_usd: float = 500_000.0
    min_base_duration_hours: int = 168
    max_new_low_count_30d: int = 2
    max_ma30_down_slope: float = -0.03
    ma7_reclaim_min_pct: float = 0.0
    ma30_reclaim_min_pct: float = 0.0
    strong_downtrend_60d_return: float = -0.35
    strong_downtrend_90d_return: float = -0.50


def classify_potential_stage(
    row: dict[str, Any], thresholds: PotentialStageThresholds | None = None
) -> tuple[str, str]:
    cfg = thresholds or PotentialStageThresholds()
    if _feature_num(row, "history_hours") < cfg.min_history_hours:
        return "insufficient_history", "low"

    stealth = _is_stealth_base(row, cfg)
    falling_knife = stealth and _is_falling_knife(row, cfg)
    cooldown_downtrend = stealth and not falling_knife and _is_cooldown_downtrend(row, cfg)
    base_stable = _is_base_stable(row, cfg)
    reclaiming = _is_reclaiming(row, cfg)
    base_ready = stealth and base_stable and (
        _feature_num(row, "quote_volume_24h") >= cfg.min_volume_24h_usd
        and _feature_num(row, "source_coverage_score") >= 0.75
        and _feature_num(row, "drawdown_30d_pct") > -0.20
    )
    first_expansion = (
        stealth
        and _feature_num(row, "volume_spike_ratio_1h_20h") >= cfg.volume_spike_ratio
        and _feature_num(row, "prior_spike_count_5d") == 0
        and (_feature_num(row, "return_1h") > 0.0 or _feature_num(row, "return_24h") > 0.03)
        and _feature_num(row, "price_vs_vwap_24h_pct") >= 0.0
    )
    pump_flags = sum(
        [
            _feature_num(row, "prior_spike_count_5d") >= 2,
            abs(_feature_num(row, "return_24h")) >= 0.15,
            _feature_num(row, "range_width_90d_pct") >= 0.80,
            _feature_num(row, "volume_spike_ratio_1h_20h") >= cfg.volume_spike_ratio
            and _feature_num(row, "price_vs_vwap_24h_pct") < 0.0,
        ]
    )
    late_confirmation = any(
        [
            _feature_lt(row, "price_vs_vwap_24h_pct", 0.0),
            _feature_lt(row, "taker_buy_ratio", 0.45),
            _feature_lt(row, "depth_imbalance_25_mean", -0.10),
        ]
    )
    if (
        _feature_gte(row, "range_position_90d_pct", cfg.late_range_min_pct)
        or _feature_lte(row, "price_to_30d_high", 1.05)
    ) and late_confirmation:
        return "late_distribution_risk", "medium"
    if pump_flags >= 2:
        return "pump_chop", "medium"
    if falling_knife:
        return "falling_knife", "medium"
    if cooldown_downtrend:
        return "cooldown_downtrend", "medium"
    if first_expansion:
        return "first_expansion", "high" if base_ready else "medium"
    if base_ready:
        return "base_ready", "high"
    if stealth and reclaiming and not base_stable:
        return "early_reclaim", "medium"
    if stealth:
        return "stealth_base", "medium"
    if (
        _feature_num(row, "return_24h") > 0.03
        and _feature_num(row, "price_vs_vwap_24h_pct") > 0.0
        and _feature_num(row, "range_position_90d_pct") < 0.75
    ):
        return "controlled_lift", "medium"
    return "pump_chop" if pump_flags else "controlled_lift", "low"


def _is_stealth_base(row: dict[str, Any], cfg: PotentialStageThresholds) -> bool:
    flags = sum(
        [
            _feature_lte(row, "price_to_90d_low", cfg.near_low_ratio),
            _feature_lte(row, "range_position_90d_pct", cfg.range_low_max_pct),
            _feature_lte(row, "bb_width_percentile_90d", cfg.compression_pctile_max),
            _feature_lte(row, "volume_contraction_10d_90d", cfg.volume_contraction_max),
            not bool(row["new_low_15d"]),
        ]
    )
    return flags >= 3


def _is_base_stable(row: dict[str, Any], cfg: PotentialStageThresholds) -> bool:
    checks = []
    if _feature_has(row, "base_duration_hours"):
        checks.append(_feature_gte(row, "base_duration_hours", cfg.min_base_duration_hours))
    if _feature_has(row, "new_low_count_30d"):
        checks.append(_feature_lte(row, "new_low_count_30d", cfg.max_new_low_count_30d))
    if _feature_has(row, "ma_30d_slope_14d"):
        checks.append(_feature_gte(row, "ma_30d_slope_14d", cfg.max_ma30_down_slope))
    if not checks:
        return True
    reclaim_or_higher_low = any(
        [
            _feature_gte(row, "price_vs_vwap_24h_pct", 0.0),
            _feature_gte(row, "price_vs_ma_7d_pct", cfg.ma7_reclaim_min_pct),
            _feature_gte(row, "price_vs_ma_30d_pct", cfg.ma30_reclaim_min_pct),
            _feature_gte(row, "ma_7d_slope_7d", 0.0),
            _feature_gte(row, "higher_low_count_30d", 1),
        ]
    )
    return all(checks) and reclaim_or_higher_low


def _is_reclaiming(row: dict[str, Any], cfg: PotentialStageThresholds) -> bool:
    return any(
        [
            _feature_gte(row, "price_vs_vwap_24h_pct", 0.0),
            _feature_gte(row, "price_vs_ma_7d_pct", cfg.ma7_reclaim_min_pct),
            _feature_gte(row, "price_vs_ma_30d_pct", cfg.ma30_reclaim_min_pct),
            str(row["reclaim_state"] or "")
            in {"vwap_reclaim", "ma7_reclaim", "ma30_reclaim", "reclaim_hold"},
        ]
    )


def _is_falling_knife(row: dict[str, Any], cfg: PotentialStageThresholds) -> bool:
    return (
        _feature_has(row, "new_low_count_30d")
        and _feature_num(row, "new_low_count_30d") > cfg.max_new_low_count_30d
        and (
            _feature_lt(row, "ma_30d_slope_14d", cfg.max_ma30_down_slope)
            or _feature_lt(row, "price_vs_vwap_24h_pct", 0.0)
            or not _is_reclaiming(row, cfg)
        )
    )


def _is_cooldown_downtrend(row: dict[str, Any], cfg: PotentialStageThresholds) -> bool:
    strong_decline = any(
        [
            _feature_lt(row, "return_60d", cfg.strong_downtrend_60d_return),
            _feature_lt(row, "return_90d", cfg.strong_downtrend_90d_return),
        ]
    )
    immature = (
        _feature_has(row, "base_duration_hours")
        and _feature_num(row, "base_duration_hours") < cfg.min_base_duration_hours
    )
    still_down = _feature_lt(row, "ma_30d_slope_14d", cfg.max_ma30_down_slope)
    return (strong_decline or immature or still_down) and not _is_base_stable(row, cfg)


def _feature_num(row: dict[str, Any], key: str) -> float:
    value = row[key]
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _feature_num_or_none(row: dict[str, Any], key: str) -> float | None:
    value = row[key]
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _feature_has(row: dict[str, Any], key: str) -> bool:
    return _feature_num_or_none(row, key) is not None


def _feature_lte(row: dict[str, Any], key: str, threshold: float) -> bool:
    value = _feature_num_or_none(row, key)
    return value is not None and value <= threshold


def _feature_lt(row: dict[str, Any], key: str, threshold: float) -> bool:
    value = _feature_num_or_none(row, key)
    return value is not None and value < threshold


def _feature_gte(row: dict[str, Any], key: str, threshold: float) -> bool:
    value = _feature_num_or_none(row, key)
    return value is not None and value >= threshold


def rank_potential_candidates(
    potential_features: pl.DataFrame,
    *,
    strict_scores: pl.DataFrame | None = None,
    broad_candidates: pl.DataFrame | None = None,
    discovery: pl.DataFrame | None = None,
    config: PotentialScanConfig | None = None,
) -> pl.DataFrame:
    if potential_features.is_empty():
        return empty_potential_candidate_frame()
    cfg = config or PotentialScanConfig()
    latest = _latest_by_symbol(potential_features)
    strict = _strict_context(strict_scores if strict_scores is not None else pl.DataFrame())
    broad = _broad_context(broad_candidates if broad_candidates is not None else pl.DataFrame())
    disc = _discovery_context(discovery if discovery is not None else pl.DataFrame())
    frame = (
        latest.join(strict, on="symbol", how="left")
        .join(broad, on="symbol", how="left")
        .join(disc, on="symbol", how="left")
    )
    frame = _filter_potential_universe(frame, cfg)
    rows = []
    thresholds = PotentialStageThresholds(
        near_low_ratio=cfg.near_low_ratio,
        compression_pctile_max=cfg.compression_pctile_max,
        volume_contraction_max=cfg.volume_contraction_max,
        volume_spike_ratio=cfg.volume_spike_ratio,
        range_low_max_pct=cfg.range_low_max_pct,
        late_range_min_pct=cfg.late_range_min_pct,
        taker_buy_confirm_min=cfg.taker_buy_confirm_min,
        min_history_hours=cfg.min_history_hours,
        min_volume_24h_usd=cfg.min_volume_24h_usd,
        min_base_duration_hours=cfg.min_base_duration_hours,
        max_new_low_count_30d=cfg.max_new_low_count_30d,
        max_ma30_down_slope=cfg.max_ma30_down_slope,
        ma7_reclaim_min_pct=cfg.ma7_reclaim_min_pct,
        ma30_reclaim_min_pct=cfg.ma30_reclaim_min_pct,
        strong_downtrend_60d_return=cfg.strong_downtrend_60d_return,
        strong_downtrend_90d_return=cfg.strong_downtrend_90d_return,
    )
    for row in frame.to_dicts():
        stage, confidence = classify_potential_stage(row, thresholds)
        score, positives, risks = _score(row, stage, cfg)
        missing = _missing_evidence(row)
        confirmation = _confirmation_state(row, positives)
        risk_state = _risk_state(stage, risks, missing)
        action_state = _action_state(stage, confirmation)
        setup_quality = _setup_quality(stage, score)
        setup_pass_reason = _setup_pass_reason(row, cfg)
        funds_score, funds_pos, funds_neg, funds_state, evidence_families = _funds_evidence(row)
        next_needed = _next_confirmation_needed(row, funds_state, evidence_families)
        board_priority = _board_priority(
            stage,
            action_state,
            risk_state,
            funds_state,
            setup_quality,
            str(row["broad_reasons"] or ""),
        )
        rows.append(
            {
                "rank": 0,
                "timestamp": row["timestamp"],
                "symbol": row["symbol"],
                "base_ccy": row["base_ccy"] or _base_from_symbol(str(row["symbol"] or "")),
                "market_cap_usd": row["market_cap_usd"],
                "volume_24h_usd": row["volume_24h_usd"],
                "potential_score": score,
                "stage": stage,
                "stage_confidence": confidence,
                "setup_quality": setup_quality,
                "confirmation_state": confirmation,
                "risk_state": risk_state,
                "action_state": action_state,
                "board_priority": board_priority,
                "setup_pass_reason": setup_pass_reason,
                "funds_state": funds_state,
                "funds_score_total": funds_score,
                "funds_positive_reasons": ";".join(funds_pos),
                "funds_negative_reasons": ";".join(funds_neg),
                "next_confirmation_needed": ";".join(next_needed),
                "evidence_families_present": evidence_families,
                "price_to_90d_low": row["price_to_90d_low"],
                "bb_width_percentile_90d": row["bb_width_percentile_90d"],
                "range_position_90d_pct": row["range_position_90d_pct"],
                "base_duration_hours": row["base_duration_hours"],
                "new_low_count_30d": row["new_low_count_30d"],
                "higher_low_count_30d": row["higher_low_count_30d"],
                "price_vs_ma_7d_pct": row["price_vs_ma_7d_pct"],
                "price_vs_ma_30d_pct": row["price_vs_ma_30d_pct"],
                "ma_7d_slope_7d": row["ma_7d_slope_7d"],
                "ma_30d_slope_14d": row["ma_30d_slope_14d"],
                "reclaim_state": row["reclaim_state"] or "",
                "structure_block_reason": row["structure_block_reason"] or "",
                "volume_spike_ratio_1h_20h": row["volume_spike_ratio_1h_20h"],
                "first_volume_expansion": row["first_volume_expansion"],
                "price_vs_vwap_24h_pct": row["price_vs_vwap_24h_pct"],
                "taker_buy_ratio": row["taker_buy_ratio"],
                "open_interest_usd_change_24h": row["open_interest_usd_change_24h"],
                "net_exchange_flow": row["net_exchange_flow"],
                "flow_zscore": row["flow_zscore"],
                "whale_accumulation_ratio": row["whale_accumulation_ratio"],
                "depth_imbalance_25_mean": row["depth_imbalance_25_mean"],
                "large_trade_buy_ratio": row["large_trade_buy_ratio"],
                "mention_growth": row["mention_growth"],
                "strict_score_total": row["strict_score_total"],
                "strict_alert_level": row["strict_alert_level"] or "none",
                "broad_rank": row["broad_rank"],
                "broad_score": row["broad_score"],
                "broad_reasons": row["broad_reasons"] or "",
                "positive_reasons": ";".join(positives),
                "risk_reasons": ";".join([*risks, *missing]),
                "missing_evidence": ";".join(missing),
                "data_quality_warning": row["data_quality_warning"] or "",
                "_board_priority_sort": _board_priority_sort(board_priority),
            }
        )
    if not rows:
        return empty_potential_candidate_frame()
    rows = sorted(
        rows,
        key=lambda row: (
            row["_board_priority_sort"],
            -(row["funds_score_total"] or 0.0),
            -(row["potential_score"] or 0.0),
            -(row["broad_score"] or 0.0),
            str(row["symbol"]),
        ),
    )[: cfg.output_top_n]
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return pl.DataFrame(rows, schema=POTENTIAL_CANDIDATE_SCHEMA)


def _score(
    row: dict[str, Any], stage: str, cfg: PotentialScanConfig
) -> tuple[float, list[str], list[str]]:
    score = {
        "base_ready": 30.0,
        "stealth_base": 22.0,
        "early_reclaim": 24.0,
        "cooldown_downtrend": 6.0,
        "falling_knife": -22.0,
        "first_expansion": 35.0,
        "controlled_lift": 18.0,
        "pump_chop": -12.0,
        "late_distribution_risk": -25.0,
        "insufficient_history": -15.0,
    }.get(stage, 0.0)
    positives = [stage]
    risks: list[str] = []
    if _num(row, "price_to_90d_low") and _num(row, "price_to_90d_low") <= cfg.strong_near_low_ratio:
        score += 8.0
        positives.append("near_90d_low")
    if _num(row, "bb_width_percentile_90d") and _num(row, "bb_width_percentile_90d") <= 0.20:
        score += 8.0
        positives.append("compressed_volatility")
    if _num(row, "volume_contraction_10d_90d") and _num(row, "volume_contraction_10d_90d") <= 0.50:
        score += 6.0
        positives.append("volume_contraction")
    if not bool(row.get("new_low_15d")):
        score += 4.0
        positives.append("no_new_low_15d")
    if _num(row, "range_position_90d_pct") <= cfg.range_low_max_pct:
        score += 6.0
        positives.append("low_range_position")
    if _num(row, "base_duration_hours") >= cfg.min_base_duration_hours:
        score += 8.0
        positives.append("base_duration_7d")
    if (
        _has_num(row, "new_low_count_30d")
        and _num(row, "new_low_count_30d") <= cfg.max_new_low_count_30d
    ):
        score += 6.0
        positives.append("no_recent_lower_lows")
    if _num(row, "higher_low_count_30d") >= 1:
        score += 5.0
        positives.append("higher_low_structure")
    if _num(row, "price_vs_ma_7d_pct") >= cfg.ma7_reclaim_min_pct:
        score += 5.0
        positives.append("ma7_reclaim")
    if _num(row, "ma_30d_slope_14d") >= cfg.max_ma30_down_slope:
        score += 5.0
        positives.append("ma30_slope_flattening")
    if row.get("first_volume_expansion") is True:
        score += 12.0
        positives.append("first_volume_expansion")
    if _num(row, "volume_spike_ratio_1h_20h") >= cfg.volume_spike_ratio:
        score += 6.0
        positives.append("volume_spike")
    if _num(row, "price_vs_vwap_24h_pct") > 0.0:
        score += 5.0
        positives.append("above_vwap")
    if _num(row, "vwap_slope_24h") > 0.0:
        score += 5.0
        positives.append("vwap_slope_up")
    if _num(row, "taker_buy_ratio") >= cfg.taker_buy_confirm_min:
        score += 8.0
        positives.append("taker_buy_confirm")
    if _num(row, "large_trade_buy_ratio") >= 0.60:
        score += 6.0
        positives.append("large_trade_buy_confirm")
    if _num(row, "depth_imbalance_25_mean") >= 0.25:
        score += 6.0
        positives.append("depth_support")
    if _num(row, "open_interest_usd_change_24h") > 0.0:
        score += 4.0
        positives.append("oi_expanding")
    strict_score = _num(row, "strict_score_total")
    if strict_score > 0.0:
        score += min(strict_score * 0.15, 8.0)
        positives.append("strict_positive_context")
    if _num(row, "range_position_90d_pct") >= cfg.late_range_min_pct:
        score -= 12.0
        risks.append("near_resistance")
    if (
        _num(row, "volume_spike_ratio_1h_20h") >= cfg.volume_spike_ratio
        and _num(row, "price_vs_vwap_24h_pct") < 0.0
    ):
        score -= 10.0
        risks.append("below_vwap_after_spike")
    if _num(row, "taker_buy_ratio") and _num(row, "taker_buy_ratio") < 0.45:
        score -= 8.0
        risks.append("taker_sell_dominance")
    if _num(row, "depth_imbalance_25_mean") and _num(row, "depth_imbalance_25_mean") < -0.10:
        score -= 8.0
        risks.append("weak_depth")
    if _num(row, "prior_spike_count_5d") >= 2:
        score -= 8.0
        risks.append("repeated_prior_spikes")
    if (
        _has_num(row, "base_duration_hours")
        and _num(row, "base_duration_hours") < cfg.min_base_duration_hours
    ):
        score -= 8.0
        risks.append("base_too_short")
    if (
        _has_num(row, "new_low_count_30d")
        and _num(row, "new_low_count_30d") > cfg.max_new_low_count_30d
    ):
        score -= 10.0
        risks.append("active_lower_lows")
    if (
        _has_num(row, "ma_30d_slope_14d")
        and _num(row, "ma_30d_slope_14d") < cfg.max_ma30_down_slope
    ):
        score -= 8.0
        risks.append("ma30_downtrend")
    if str(row.get("reclaim_state") or "") == "below_ma":
        score -= 6.0
        risks.append("no_reclaim")
    if stage == "falling_knife":
        risks.append("falling_knife")
    if stage == "insufficient_history":
        risks.append("insufficient_history")
    return max(score, -50.0), positives, risks


def _confirmation_state(row: dict[str, Any], positives: list[str]) -> str:
    confirmations = {
        "above_vwap",
        "taker_buy_confirm",
        "large_trade_buy_confirm",
        "depth_support",
        "oi_expanding",
    }
    count = len(confirmations.intersection(positives))
    if count >= 3:
        return "confirmed"
    if count >= 1:
        return "partial"
    if _missing_evidence(row):
        return "missing_optional_context"
    return "unconfirmed"


def _risk_state(stage: str, risks: list[str], missing: list[str]) -> str:
    if stage in {"late_distribution_risk", "pump_chop", "falling_knife"}:
        return "elevated"
    if risks:
        return "mixed"
    if missing:
        return "missing_confirmation"
    return "clean"


def _action_state(stage: str, confirmation: str) -> str:
    if stage == "insufficient_history":
        return "data_blocked"
    if stage in {"late_distribution_risk", "pump_chop", "falling_knife"}:
        return "avoid_late"
    if stage == "cooldown_downtrend":
        return "wait_pullback"
    if stage == "first_expansion" or (
        stage == "base_ready" and confirmation in {"confirmed", "partial"}
    ):
        return "review_now"
    if stage in {"stealth_base", "base_ready", "early_reclaim"}:
        return "watch_base"
    return "wait_pullback"


def _setup_quality(stage: str, score: float) -> str:
    if stage in {"pump_chop", "late_distribution_risk", "insufficient_history", "falling_knife"}:
        return "weak"
    if stage == "cooldown_downtrend":
        return "cooldown"
    if score >= 55.0:
        return "strong"
    if score >= 30.0:
        return "medium"
    return "early"


def _setup_pass_reason(row: dict[str, Any], cfg: PotentialScanConfig) -> str:
    near_low = (
        _has_num(row, "price_to_90d_low")
        and _num(row, "price_to_90d_low") <= cfg.near_low_ratio
    )
    compressed = (
        _has_num(row, "bb_width_percentile_90d")
        and _num(row, "bb_width_percentile_90d") < cfg.compression_pctile_max
    )
    if near_low and compressed:
        return "near_low_and_compressed"
    if near_low:
        return "near_low"
    if compressed:
        return "compressed"
    return "none"


def _funds_evidence(row: dict[str, Any]) -> tuple[float, list[str], list[str], str, str]:
    score = 0.0
    positives: list[str] = []
    negatives: list[str] = []
    families = _evidence_families(row)
    positive_families: set[str] = set()
    if _has_num(row, "flow_zscore") and _num(row, "flow_zscore") <= -3.0:
        score += 30.0
        positives.append("onchain_outflow")
        positive_families.add("onchain")
    elif _has_num(row, "net_exchange_flow") and _num(row, "net_exchange_flow") < 0.0:
        score += 30.0
        positives.append("onchain_outflow")
        positive_families.add("onchain")
    if _has_num(row, "whale_accumulation_ratio") and _num(row, "whale_accumulation_ratio") >= 0.60:
        score += 20.0
        positives.append("whale_accumulation")
        positive_families.add("whale")
    if _has_num(row, "depth_imbalance_25_mean") and _num(row, "depth_imbalance_25_mean") >= 0.25:
        score += 15.0
        positives.append("book_support")
        positive_families.add("book")
    elif _has_num(row, "depth_imbalance_25_mean") and _num(row, "depth_imbalance_25_mean") >= 0.05:
        score += 8.0
        positives.append("book_support_partial")
        positive_families.add("book")
    if _has_num(row, "large_trade_buy_ratio") and _num(row, "large_trade_buy_ratio") >= 0.60:
        score += 15.0
        positives.append("large_buying")
        positive_families.add("trade")
    if _has_num(row, "taker_buy_ratio") and _num(row, "taker_buy_ratio") >= 0.60:
        score += 15.0
        positives.append("taker_buying")
        positive_families.add("trade")
    elif _has_num(row, "taker_buy_ratio") and _num(row, "taker_buy_ratio") >= 0.55:
        score += 8.0
        positives.append("taker_buying_partial")
        positive_families.add("trade")
    if (
        _has_num(row, "open_interest_usd_change_24h")
        and _num(row, "open_interest_usd_change_24h") > 0.0
    ):
        score += 10.0
        positives.append("oi_expanding")
        positive_families.add("derivatives")
    if _has_num(row, "mention_growth") and _num(row, "mention_growth") <= 1.20:
        score += 10.0
        positives.append("message_quiet")
        positive_families.add("message")
    major_negative = False
    if _has_num(row, "flow_zscore") and _num(row, "flow_zscore") >= 3.0:
        score -= 30.0
        negatives.append("exchange_inflow")
        major_negative = True
    elif _has_num(row, "net_exchange_flow") and _num(row, "net_exchange_flow") > 0.0:
        score -= 30.0
        negatives.append("exchange_inflow")
        major_negative = True
    if _has_num(row, "depth_imbalance_25_mean") and _num(row, "depth_imbalance_25_mean") <= -0.10:
        score -= 15.0
        negatives.append("weak_depth")
    if _has_num(row, "taker_buy_ratio") and _num(row, "taker_buy_ratio") < 0.45:
        score -= 15.0
        negatives.append("sell_dominance")
    elif _has_num(row, "large_trade_buy_ratio") and _num(row, "large_trade_buy_ratio") < 0.45:
        score -= 15.0
        negatives.append("sell_dominance")
    if _message_overheat(row):
        score -= 20.0
        negatives.append("message_overheat")
        major_negative = True
    family_text = ";".join(families)
    if not families:
        state = "funds_missing"
    elif major_negative:
        state = "funds_rejected"
    elif score > 60.0 and len(positive_families) >= 2:
        state = "funds_confirmed"
    elif (score >= 25.0 and len(positive_families) >= 2) or (
        score >= 35.0 and bool(positive_families) and not major_negative
    ) or (
        "onchain_outflow" in positives and not negatives
    ):
        state = "funds_building"
    else:
        state = "funds_weak"
    return score, positives, negatives, state, family_text


def _evidence_families(row: dict[str, Any]) -> list[str]:
    families = []
    if _has_num(row, "flow_zscore") or _has_num(row, "net_exchange_flow"):
        families.append("onchain")
    if _has_num(row, "whale_accumulation_ratio"):
        families.append("whale")
    if _has_num(row, "depth_imbalance_25_mean"):
        families.append("book")
    if _has_num(row, "large_trade_buy_ratio") or _has_num(row, "taker_buy_ratio"):
        families.append("trade")
    if _has_num(row, "open_interest_usd_change_24h"):
        families.append("derivatives")
    if _has_num(row, "mention_growth"):
        families.append("message")
    return families


def _message_overheat(row: dict[str, Any]) -> bool:
    return (
        _has_num(row, "mention_growth")
        and _num(row, "mention_growth") >= 3.0
        and _num(row, "emotion_news_ratio") > _num(row, "fundamental_news_ratio")
    )


def _next_confirmation_needed(
    row: dict[str, Any], funds_state: str, evidence_families: str
) -> list[str]:
    families = set(filter(None, evidence_families.split(";")))
    needed = []
    warning_text = str(row.get("data_quality_warning") or "")
    if "onchain" not in families:
        if "onchain_unavailable_address_book" in warning_text:
            needed.append("onchain_unavailable_address_book")
        elif "onchain_unavailable_token_mapping" in warning_text:
            needed.append("onchain_unavailable_token_mapping")
        else:
            needed.append("need_onchain_flow")
    if "whale" not in families:
        needed.append("need_whale_breadth")
    if "book" not in families:
        needed.append("need_book_support")
    if "trade" not in families or "sell_dominance" in _funds_negative_text(row, funds_state):
        needed.append("need_large_trade_buying")
    if "message" not in families:
        needed.append("need_message_quietness")
    if "insufficient_history" in warning_text:
        needed.append("need_history")
    return needed


def _funds_negative_text(row: dict[str, Any], funds_state: str) -> str:
    if (
        funds_state == "funds_rejected"
        and _has_num(row, "taker_buy_ratio")
        and _num(row, "taker_buy_ratio") < 0.45
    ):
        return "sell_dominance"
    return ""


def _board_priority(
    stage: str,
    action_state: str,
    risk_state: str,
    funds_state: str,
    setup_quality: str,
    broad_reasons: str,
) -> str:
    if action_state == "data_blocked":
        return "data_blocked"
    if stage == "falling_knife" or action_state == "avoid_late" or funds_state == "funds_rejected":
        return "D_avoid"
    if (
        action_state in {"review_now", "watch_base"}
        and funds_state in {"funds_confirmed", "funds_building"}
        and risk_state != "elevated"
    ):
        return "A_review"
    if action_state in {"review_now", "watch_base"} and funds_state in {
        "funds_missing",
        "funds_weak",
    }:
        return "B_watch"
    if stage == "cooldown_downtrend":
        return "C_context"
    if (
        action_state == "wait_pullback"
        or setup_quality == "early"
        or "coingecko_trending" in broad_reasons
    ):
        return "C_context"
    if stage in {"pump_chop", "late_distribution_risk"} or risk_state == "elevated":
        return "D_avoid"
    return "B_watch"


def _board_priority_sort(board_priority: str) -> int:
    return {
        "A_review": 0,
        "B_watch": 1,
        "C_context": 2,
        "D_avoid": 3,
        "data_blocked": 4,
    }.get(board_priority, 9)


def _missing_evidence(row: dict[str, Any]) -> list[str]:
    text = ";".join(
        str(row.get(col) or "")
        for col in ("missing_evidence", "negative_filters", "data_quality_warning")
    )
    wanted = ["onchain_missing", "whale_missing", "messages_missing"]
    return [item for item in wanted if item in text]


def _latest_by_symbol(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty() or "symbol" not in frame.columns:
        return pl.DataFrame(schema={"symbol": pl.String})
    if "timestamp" in frame.columns:
        frame = frame.sort(["symbol", "timestamp"])
    out = frame.group_by("symbol").last()
    renames = {}
    if "score_total" in out.columns:
        renames["score_total"] = "strict_score_total"
    if "alert_level" in out.columns:
        renames["alert_level"] = "strict_alert_level"
    return out.rename(renames)


def _strict_context(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty() or "symbol" not in frame.columns:
        return pl.DataFrame(
            schema={
                "symbol": pl.String,
                "strict_score_total": pl.Int64,
                "strict_alert_level": pl.String,
                "missing_evidence": pl.String,
            }
        )
    out = _latest_by_symbol(frame)
    return out.select(
        [
            pl.col("symbol"),
            pl.col("strict_score_total").cast(pl.Int64, strict=False),
            pl.col("strict_alert_level").cast(pl.String),
            pl.col("missing_evidence").cast(pl.String),
        ]
    )


def _broad_context(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty() or "okx_symbol" not in frame.columns:
        return pl.DataFrame(
            schema={
                "symbol": pl.String,
                "broad_rank": pl.Int64,
                "market_cap_usd": pl.Float64,
                "volume_24h_usd": pl.Float64,
                "broad_score": pl.Float64,
                "broad_reasons": pl.String,
            }
        )
    cols = [
        pl.col("okx_symbol").alias("symbol"),
        pl.col("rank").cast(pl.Int64, strict=False).alias("broad_rank"),
        pl.col("market_cap_usd").cast(pl.Float64, strict=False),
        pl.col("volume_24h_usd").cast(pl.Float64, strict=False),
        pl.col("broad_score").cast(pl.Float64, strict=False),
        pl.col("broad_reasons").cast(pl.String),
    ]
    return frame.filter(pl.col("okx_symbol").fill_null("") != "").select(cols).unique("symbol")


def _filter_potential_universe(frame: pl.DataFrame, cfg: PotentialScanConfig) -> pl.DataFrame:
    near_low = pl.col("price_to_90d_low") <= cfg.near_low_ratio
    compressed = pl.col("bb_width_percentile_90d") < cfg.compression_pctile_max
    eligible = near_low.fill_null(False) | compressed.fill_null(False)
    market_gate = (
        pl.col("market_cap_usd").is_null()
        | (
            (pl.col("market_cap_usd") > cfg.min_market_cap_usd)
            & (pl.col("market_cap_usd") < cfg.max_market_cap_usd)
            & (pl.col("volume_24h_usd") > cfg.min_volume_24h_usd)
        )
    )
    eligible = eligible & market_gate.fill_null(False)
    return frame.filter(eligible)


def _discovery_context(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty() or "symbol" not in frame.columns:
        return pl.DataFrame(schema={"symbol": pl.String, "base_ccy": pl.String})
    return frame.select([pl.col("symbol"), pl.col("base_ccy").cast(pl.String)]).unique("symbol")


def _base_from_symbol(symbol: str) -> str:
    return symbol.split("-")[0] if symbol else ""


def _num(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None:
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(number) else number


def _has_num(row: dict[str, Any], key: str) -> bool:
    value = row.get(key)
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def render_potential_report(candidates: pl.DataFrame) -> str:
    lines = [
        "# Altcoin Potential Trend Report",
        "",
        "Research-only trend readout. No trading action is authorized.",
        "",
    ]
    if candidates.is_empty():
        return "\n".join([*lines, "No ranked rows were evaluated.", ""])
    rows = candidates.to_dicts()
    counts = {state: sum(1 for row in rows if row.get("stage") == state) for state in _STAGES}
    data_blocked = sum(1 for row in rows if row.get("action_state") == "data_blocked")
    lines.extend(
        [
            "## Run Summary",
            "",
            f"- Total evaluated: {len(rows)}",
            f"- Base-ready: {counts['base_ready']}",
            f"- First-expansion: {counts['first_expansion']}",
            f"- Controlled-lift: {counts['controlled_lift']}",
            f"- Pump-chop/late-risk: {counts['pump_chop'] + counts['late_distribution_risk']}",
            f"- Data-blocked: {data_blocked}",
            "",
        ]
    )
    _section(lines, "A Review: Prepared Base + Funds Building/Confirmed", rows, {"A_review"})
    _section(lines, "B Watch: Stable Base But Needs Confirmation", rows, {"B_watch"})
    _section(
        lines,
        "C Context: Cooldown Downtrend / Early Reclaim",
        rows,
        {"C_context"},
        exclude_stages={"controlled_lift"},
    )
    _section(
        lines,
        "Wait Reset: Controlled Lift / Not Base",
        rows,
        set(),
        actions={"wait_pullback"},
        stages={"controlled_lift"},
    )
    _section(lines, "Avoid: Falling Knife, Late, Choppy, Distribution", rows, {"D_avoid"})
    _section(lines, "Data Blocked", rows, {"data_blocked"})
    _missing_section(lines, rows)
    return "\n".join(lines).rstrip() + "\n"


def _section(
    lines: list[str],
    title: str,
    rows: list[dict[str, object]],
    tiers: set[str],
    *,
    actions: set[str] | None = None,
    exclude_actions: set[str] | None = None,
    stages: set[str] | None = None,
    exclude_stages: set[str] | None = None,
) -> None:
    lines.extend([f"## {title}", ""])
    selected = [
        row
        for row in rows
        if (
            str(row.get("board_priority") or "") in tiers
            or (actions is not None and str(row.get("action_state") or "") in actions)
        )
        and not (
            exclude_actions is not None
            and str(row.get("action_state") or "") in exclude_actions
        )
        and (stages is None or str(row.get("stage") or "") in stages)
        and not (
            exclude_stages is not None and str(row.get("stage") or "") in exclude_stages
        )
    ]
    if not selected:
        if "A Review" in title:
            lines.extend(
                [
                    "No rows met A_review: requires prepared base/watch structure plus "
                    "funds_building or funds_confirmed.",
                    "",
                ]
            )
        else:
            lines.extend(["No rows.", ""])
        return
    for row in selected:
        lines.extend([_row_line(row), ""])


def _missing_section(lines: list[str], rows: list[dict[str, object]]) -> None:
    lines.extend(["## Confirmation Gaps", ""])
    selected = [
        row
        for row in rows
        if str(row.get("next_confirmation_needed") or row.get("missing_evidence") or "")
    ]
    if not selected:
        lines.extend(["No optional confirmation gaps reported.", ""])
        return
    for row in selected:
        need = row.get("next_confirmation_needed") or row.get("missing_evidence")
        lines.append(
            f"- {row.get('symbol')}: need={need}, tier={row.get('board_priority')}, "
            f"stage={row.get('stage')}, funds={row.get('funds_state')}"
        )
    lines.append("")


def _row_line(row: dict[str, object]) -> str:
    return (
        f"- {row.get('symbol')}: tier={row.get('board_priority')}, "
        f"stage={row.get('stage')}, "
        f"funds={row.get('funds_state')}/{_fmt(row.get('funds_score_total'))}, "
        f"cap={_money(row.get('market_cap_usd'))}, vol={_money(row.get('volume_24h_usd'))}, "
        f"gate={row.get('setup_pass_reason')}, "
        f"structure=price_to_low={_fmt(row.get('price_to_90d_low'))}, "
        f"bb={_fmt(row.get('bb_width_percentile_90d'))}, "
        f"range={_fmt(row.get('range_position_90d_pct'))}, "
        f"base={_hours(row.get('base_duration_hours'))}, "
        f"new_lows_30d={row.get('new_low_count_30d')}, "
        f"reclaim={row.get('reclaim_state') or 'n/a'}, "
        f"ma30_slope={_fmt(row.get('ma_30d_slope_14d'))}, "
        f"confirmations={_confirmations(row)}, "
        f"funds_pos={row.get('funds_positive_reasons') or 'none'}, "
        f"need={row.get('next_confirmation_needed') or 'none'}, "
        f"block={row.get('structure_block_reason') or 'none'}, "
        f"risks={row.get('risk_reasons') or row.get('funds_negative_reasons') or 'none'}"
    )


def _trend_label(row: dict[str, object]) -> str:
    stage = str(row.get("stage") or "")
    return {
        "first_expansion": "early_expansion",
        "base_ready": "base_ready",
        "stealth_base": "quiet_base",
        "controlled_lift": "controlled_lift",
        "pump_chop": "choppy_pump",
        "late_distribution_risk": "late_trend_risk",
        "insufficient_history": "data_limited",
    }.get(stage, "unclassified")


def _vwap_state(row: dict[str, object]) -> str:
    value = _float(row.get("price_vs_vwap_24h_pct"))
    if value is None:
        return "unknown"
    return "above" if value >= 0.0 else "below"


def _taker_state(row: dict[str, object]) -> str:
    value = _float(row.get("taker_buy_ratio"))
    if value is None:
        return "unknown"
    if value >= 0.65:
        return "buy_dominant"
    if value <= 0.45:
        return "sell_dominant"
    return "mixed"


def _oi_state(row: dict[str, object]) -> str:
    value = _float(row.get("open_interest_usd_change_24h"))
    if value is None:
        return "unknown"
    return "expanding" if value > 0.0 else "flat_or_contracting"


def _confirmations(row: dict[str, object]) -> str:
    confirmations = []
    if _vwap_state(row) == "above":
        confirmations.append("vwap_above")
    if _taker_state(row) == "buy_dominant":
        confirmations.append("taker_buy")
    if _oi_state(row) == "expanding":
        confirmations.append("oi_expanding")
    if _float(row.get("depth_imbalance_25_mean")) is not None:
        confirmations.append("book")
    if _float(row.get("large_trade_buy_ratio")) is not None:
        confirmations.append("large_trade")
    return ";".join(confirmations) if confirmations else "none"


def _money(value: object) -> str:
    number = _float(value)
    if number is None:
        return "n/a"
    if abs(number) >= 1_000_000_000:
        return f"{number / 1_000_000_000:.1f}B"
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.1f}M"
    if abs(number) >= 1_000:
        return f"{number / 1_000:.1f}K"
    return f"{number:.0f}"


def _hours(value: object) -> str:
    number = _float(value)
    if number is None:
        return "n/a"
    if number >= 24.0:
        return f"{number / 24.0:.0f}d"
    return f"{number:.0f}h"


def _fmt(value: object) -> str:
    number = _float(value)
    return "n/a" if number is None else f"{number:.2f}"


def _float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


_STAGES = (
    "stealth_base",
    "base_ready",
    "early_reclaim",
    "cooldown_downtrend",
    "falling_knife",
    "first_expansion",
    "controlled_lift",
    "pump_chop",
    "late_distribution_risk",
    "insufficient_history",
)


def build_potential_board(
    bars: pl.DataFrame,
    market: pl.DataFrame,
    okx: pl.DataFrame,
    context: pl.DataFrame,
    config: PotentialScanConfig,
) -> tuple[pl.DataFrame, str]:
    """Build one latest-row potential board and Markdown report.

    `market` is the broad/market metadata frame. CoinGecko trending is treated as
    attention annotation through `heat_source`/`broad_reasons`, not as a selector.
    """
    if bars.is_empty() or "symbol" not in bars.columns:
        board = empty_potential_board_frame()
        return board, render_potential_board(board)
    features = compute_potential_features_batch(
        bars,
        context,
        min_history_hours=config.min_history_hours,
        full_history_hours=config.full_history_hours,
        volume_spike_ratio=config.volume_spike_ratio,
        first_spike_lookback_hours=config.first_spike_lookback_hours,
    )
    candidates = rank_potential_candidates(
        features,
        broad_candidates=market,
        discovery=okx,
        config=config,
    )
    board = _ranked_rows_to_board(candidates, market)
    return board, render_potential_board(board)


def _ranked_rows_to_board(candidates: pl.DataFrame, market: pl.DataFrame) -> pl.DataFrame:
    if candidates.is_empty():
        return empty_potential_board_frame()
    frame = candidates
    if {"symbol", "timestamp"}.issubset(frame.columns):
        frame = frame.sort(["symbol", "timestamp"]).group_by("symbol").last()
    cols = set(frame.columns)
    market_cols = set(market.columns) if not market.is_empty() else set()
    if "price_change_pct_1h" not in cols or "price_change_pct_24h" not in cols:
        if {"okx_symbol", "price_change_pct_1h", "price_change_pct_24h"}.issubset(market_cols):
            provider = pl.col("provider") if "provider" in market_cols else pl.lit(None)
            heat_source = pl.col("heat_source") if "heat_source" in market_cols else pl.lit("")
            prices = market.select(
                [
                    pl.col("okx_symbol").alias("symbol"),
                    pl.col("price_change_pct_1h"),
                    pl.col("price_change_pct_24h"),
                    provider.alias("market_data_provider"),
                    heat_source.alias("attention_source"),
                ]
            ).filter(pl.col("symbol").fill_null("") != "")
            frame = frame.join(prices.unique("symbol"), on="symbol", how="left")
    market_provider = (
        pl.col("market_data_provider")
        if "market_data_provider" in frame.columns
        else pl.lit(None)
    )
    attention_source = (
        pl.col("attention_source") if "attention_source" in frame.columns else pl.lit(None)
    )
    out = frame.with_columns(
        [
            pl.coalesce([market_provider, pl.lit("unknown")]).alias("market_data_provider"),
            pl.coalesce(
                [
                    attention_source,
                    pl.when(
                        pl.col("broad_reasons").fill_null("").str.contains("coingecko_trending")
                    )
                    .then(pl.lit("coingecko_trending"))
                    .otherwise(pl.lit("")),
                ]
            ).alias("attention_source"),
            pl.lit(True).alias("okx_mapped"),
            pl.lit("passed").alias("gate_state"),
            pl.col("setup_pass_reason").fill_null("none").alias("gate_reason"),
            pl.col("stage").alias("structure_state"),
            pl.col("potential_score").alias("structure_score"),
            pl.col("structure_block_reason").fill_null("").alias("structure_blockers"),
            pl.col("funds_score_total").alias("funds_score"),
            pl.col("funds_positive_reasons").fill_null("").alias("funds_positive"),
            pl.col("funds_negative_reasons").fill_null("").alias("funds_negative"),
            pl.col("evidence_families_present").fill_null("").alias("evidence_families"),
            _bucket_expr().alias("board_bucket"),
            _decision_reason_expr().alias("decision_reason"),
            pl.col("next_confirmation_needed").fill_null("").alias("next_confirmation"),
            pl.col("data_quality_warning").fill_null("").alias("data_warning"),
        ]
    )
    for col, dtype in POTENTIAL_BOARD_SCHEMA.items():
        if col not in out.columns:
            out = out.with_columns(pl.lit(None).cast(dtype).alias(col))
        else:
            out = out.with_columns(pl.col(col).cast(dtype, strict=False).alias(col))
    return out.select(POTENTIAL_BOARD_SCHEMA.keys()).sort("rank")


def render_potential_board(board: pl.DataFrame) -> str:
    lines = [
        "# Potential Coin Board",
        "",
        "Research-only board. No trading action is authorized.",
        "",
        "Source list: OKX live swaps joined with CoinGecko market cap/volume. "
        "CoinGecko search-trending is annotation only.",
        "",
    ]
    if board.is_empty():
        return "\n".join([*lines, "No potential-board rows were evaluated.", ""])
    rows = board.to_dicts()
    counts = {
        bucket: sum(1 for row in rows if row.get("board_bucket") == bucket)
        for bucket in _BUCKETS
    }
    blockers = _token_counts(row.get("structure_blockers") for row in rows)
    attention = _token_counts(row.get("attention_source") for row in rows)
    lines.extend(
        [
            "## Funnel",
            "",
            f"- Board rows: {len(rows)}",
            f"- A_prepared: {counts['A_prepared']}",
            f"- B_watch: {counts['B_watch']}",
            f"- C_context: {counts['C_context']}",
            f"- D_reject: {counts['D_reject']}",
            f"- Top blockers: {_render_counts(blockers)}",
            f"- Attention sources: {_render_counts(attention)}",
            "",
        ]
    )
    if counts["A_prepared"] == 0:
        lines.extend(
            [
                "## Why No A_prepared",
                "",
                "No A_prepared rows because structure/funds filters did not jointly pass.",
                f"Top structure blockers: {_render_counts(blockers)}",
                "",
            ]
        )
    for bucket, title in (
        ("A_prepared", "A Prepared"),
        ("B_watch", "B Watch"),
        ("C_context", "C Context"),
        ("D_reject", "D Reject"),
    ):
        selected = [row for row in rows if row.get("board_bucket") == bucket]
        lines.extend([f"## {title}", ""])
        if not selected:
            lines.extend(["No rows.", ""])
            continue
        for row in selected:
            lines.extend([_row(row), ""])
    return "\n".join(lines).rstrip() + "\n"


def compact_potential_sources(manifest: pl.DataFrame) -> pl.DataFrame:
    schema = {
        "timestamp": pl.Int64,
        "source": pl.String,
        "provider": pl.String,
        "endpoint": pl.String,
        "status": pl.String,
        "rows": pl.Int64,
        "warning": pl.String,
        "elapsed_ms": pl.Int64,
    }
    if manifest.is_empty():
        return pl.DataFrame(schema=schema)
    keep_sources = {
        "coingecko_markets",
        "coingecko_trending",
        "discovery",
        "bars",
        "books",
        "trades",
        "funding",
        "open_interest_history",
        "taker_volume_contract",
        "long_short_ratio_contract",
    }
    frame = manifest.filter(pl.col("source").is_in(list(keep_sources)))
    for col, dtype in schema.items():
        if col == "elapsed_ms":
            frame = frame.with_columns(pl.lit(None).cast(dtype).alias(col))
        elif col not in frame.columns:
            frame = frame.with_columns(pl.lit(None).cast(dtype).alias(col))
        else:
            frame = frame.with_columns(pl.col(col).cast(dtype, strict=False).alias(col))
    return frame.select(schema.keys())


def _bucket_expr() -> pl.Expr:
    stage = pl.col("stage").fill_null("")
    funds = pl.col("funds_state").fill_null("")
    risk = pl.col("risk_state").fill_null("")
    return (
        pl.when(
            stage.is_in(
                ["falling_knife", "pump_chop", "late_distribution_risk", "insufficient_history"]
            )
        )
        .then(pl.lit("D_reject"))
        .when(
            stage.is_in(["base_ready", "stealth_base", "early_reclaim", "first_expansion"])
            & funds.is_in(["funds_building", "funds_confirmed"])
            & (risk != "elevated")
        )
        .then(pl.lit("A_prepared"))
        .when(stage.is_in(["base_ready", "stealth_base", "early_reclaim"]))
        .then(pl.lit("B_watch"))
        .when(stage.is_in(["cooldown_downtrend", "controlled_lift"]))
        .then(pl.lit("C_context"))
        .otherwise(pl.lit("D_reject"))
    )


def _decision_reason_expr() -> pl.Expr:
    return pl.concat_str(
        [
            pl.lit("structure="),
            pl.col("stage").fill_null("unknown"),
            pl.lit(";funds="),
            pl.col("funds_state").fill_null("unknown"),
            pl.lit(";blockers="),
            pl.col("structure_block_reason").fill_null("none"),
        ]
    )


def _token_counts(values: object) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        for token in str(value or "").split(";"):
            token = token.strip()
            if token:
                counts[token] = counts.get(token, 0) + 1
    return counts


def _render_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    ordered = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:8]
    return ";".join(f"{key}={value}" for key, value in ordered)


def _row(row: dict[str, object]) -> str:
    return (
        f"- {row.get('symbol')}: bucket={row.get('board_bucket')}, "
        f"structure={row.get('structure_state')}, funds={row.get('funds_state')}, "
        f"base={row.get('base_duration_hours')}h, new_lows_30d={row.get('new_low_count_30d')}, "
        f"reclaim={row.get('reclaim_state')}, blockers={row.get('structure_blockers') or 'none'}, "
        f"funds_pos={row.get('funds_positive') or 'none'}, "
        f"next={row.get('next_confirmation') or 'none'}"
    )


_BUCKETS = ("A_prepared", "B_watch", "C_context", "D_reject")



