"""Evaluation layer — formatting and comparison.

Primary metrics for sparse systems = trade-level stats. Calendar-bar Sharpe /
Sortino kept as secondary diagnostics only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import polars as pl

from qooi.core.metrics import EvalMetrics, as_float, compute_metrics, infer_periods_per_year

EMPTY_TRADE_STATS = {
    "trade_expectancy_pct": 0.0,
    "trade_expectancy_usd": 0.0,
    "median_trade_pct": 0.0,
    "trade_sharpe": 0.0,
}


@dataclass(frozen=True)
class FeatureDiagnostics:
    bars: int = 0
    usable_bars: int = 0
    warmup_bars: int = 0


@dataclass(frozen=True)
class SignalDiagnostics:
    nonzero_signal_bars: int = 0
    long_signal_bars: int = 0
    short_signal_bars: int = 0


@dataclass(frozen=True)
class BasketLifecycleDiagnostics:
    entry_signals: int = 0
    entry_actions: int = 0
    exit_actions: int = 0
    grid_actions: int = 0
    hedge_actions: int = 0
    recovery_actions: int = 0
    max_simultaneous_baskets: int = 0
    same_bar_exit_entry_count: int = 0
    blocked_entry_signals: int = 0
    duplicate_entry_suppressed: int = 0
    capacity_blocked_entries: int = 0
    sizing_blocked_entries: int = 0
    entry_acceptance_rate_pct: float = 0.0
    blocked_entry_reasons: dict[str, int] = field(default_factory=dict)
    min_contract_block_count: int = 0
    median_required_capital_for_min_contract: float = 0.0
    median_required_risk_pct_for_min_contract: float = 0.0
    blocked_by_risk_count: int = 0
    blocked_by_notional_count: int = 0
    action_event_count: int = 0
    stacked_entry_count: int = 0
    stacked_entry_net_pnl_usd: float = 0.0
    final_open_positions: int = 0


@dataclass(frozen=True)
class PortfolioRiskDiagnostics:
    avg_active_exposure: float = 0.0
    max_active_exposure: float = 0.0
    avg_notional_exposure_pct: float = 0.0
    max_notional_exposure_pct: float = 0.0
    fee_usd: float = 0.0
    stop_exit_count: int = 0
    stop_exit_net_pnl_usd: float = 0.0
    recovered_stop_exit_count: int = 0
    recovered_stop_exit_net_pnl_usd: float = 0.0
    recovery_net_pnl_usd: float = 0.0
    recovery_preempted_stop_count: int = 0
    recovery_preempted_time_count: int = 0
    recovery_preempted_trailing_count: int = 0
    recovery_unsized_actions: int = 0
    recovery_cap_breach_actions: int = 0
    recovery_blocked_actions: int = 0
    recovery_blocked_reasons: dict[str, int] = field(default_factory=dict)
    recovery_allowed_actions: int = 0
    recovery_notional_after_pct: float = 0.0
    ambiguous_stop_target_count: int = 0
    ambiguous_stop_net_pnl_usd: float = 0.0
    target_first_counterfactual_net_pnl_usd: float = 0.0
    ambiguity_impact_usd: float = 0.0
    drawdown_stop_pct: float | None = None
    stopped_early: bool = False


@dataclass(frozen=True)
class EngineDataAudit:
    bars: int = 0
    bars_processed: int = 0
    data_start: int | None = None
    data_end: int | None = None
    mark_to_market: bool = False


@dataclass(frozen=True)
class YieldAttribution:
    gross_pnl_usd: float = 0.0
    net_pnl_usd: float = 0.0
    fee_usd: float = 0.0
    gross_profit_usd: float = 0.0
    gross_loss_usd: float = 0.0
    fee_drag_pct: float = 0.0
    price_expectancy_pct: float = 0.0
    dollar_expectancy_usd: float = 0.0
    worst_side: str = "n/a"
    worst_side_expectancy_usd: float = 0.0
    worst_exit_reason: str = "n/a"
    worst_exit_expectancy_usd: float = 0.0
    worst_signal_id: str = "n/a"
    worst_signal_expectancy_usd: float = 0.0
    max_consecutive_losses: int = 0
    worst_5_trade_net_pnl_usd: float = 0.0
    size_weighted_expectancy_pct: float = 0.0
    avg_win_notional_pct_capital: float = 0.0
    avg_loss_notional_pct_capital: float = 0.0
    loss_to_win_notional_ratio: float = 0.0
    avg_win_contracts: float = 0.0
    avg_loss_contracts: float = 0.0


@dataclass(frozen=True)
class StopEffectiveness:
    stop_trades: int = 0
    stop_net_pnl_usd: float = 0.0
    stop_gross_loss_usd: float = 0.0
    stop_avg_pnl_pct: float = 0.0
    stop_avg_pnl_usd: float = 0.0
    stop_worst_pnl_usd: float = 0.0
    stop_avg_notional_pct: float = 0.0
    stop_loss_share_pct: float = 0.0
    worst_stop_side: str = "n/a"
    worst_stop_side_net_usd: float = 0.0
    worst_stop_signal_id: str = "n/a"
    worst_stop_signal_net_usd: float = 0.0
    worst_stop_side_share_pct: float = 0.0
    worst_stop_signal_share_pct: float = 0.0


@dataclass(frozen=True)
class DrawdownPathDiagnostics:
    entries_while_drawdown_count: int = 0
    entries_while_drawdown_pct: float = 0.0
    avg_entry_drawdown_pct: float = 0.0
    max_entry_drawdown_pct: float = 0.0
    max_notional_during_drawdown_pct: float = 0.0
    worst_drawdown_entry_signal_id: str = "n/a"
    worst_drawdown_side: str = "n/a"


@dataclass(frozen=True)
class BacktestDiagnostics:
    feature: FeatureDiagnostics = field(default_factory=FeatureDiagnostics)
    signal: SignalDiagnostics = field(default_factory=SignalDiagnostics)
    lifecycle: BasketLifecycleDiagnostics = field(default_factory=BasketLifecycleDiagnostics)
    risk: PortfolioRiskDiagnostics = field(default_factory=PortfolioRiskDiagnostics)
    audit: EngineDataAudit = field(default_factory=EngineDataAudit)
    bars: int = 0
    bars_processed: int = 0
    stopped_early: bool = False
    stop_bar_index: int | None = None
    nonzero_signal_bars: int = 0
    long_signal_bars: int = 0
    short_signal_bars: int = 0
    entries: int = 0
    exits: int = 0
    exit_reasons: dict[str, int] = field(default_factory=dict)
    avg_bars_held: float = 0.0
    avg_active_exposure: float = 0.0
    max_active_exposure: float = 0.0
    avg_notional_exposure_pct: float = 0.0
    max_notional_exposure_pct: float = 0.0
    final_open_positions: int = 0
    open_unrealized_pnl_usd: float = 0.0
    fee_usd: float = 0.0
    data_start: int | None = None
    data_end: int | None = None
    mark_to_market: bool = False
    drawdown_stop_pct: float | None = None

    @property
    def signal_bar_pct(self) -> float:
        return self.nonzero_signal_bars / self.bars * 100.0 if self.bars > 0 else 0.0


def _as_float(value: object, default: float = 0.0) -> float:
    return as_float(value, default)


def exit_family(reason: object) -> str:
    value = str(reason or "")
    if value in {"strategy_exit", "signal_zero", "thesis_failed", "signal_flip"}:
        return "strategy"
    if value in {"stop", "trailing_stop", "breakeven", "time", "global_drawdown_stop"}:
        return "risk_stop"
    if value.startswith("grid_level_") or value in {
        "martingale_reverse",
        "hedge_on_drawdown",
        "global_loss_limit",
    }:
        return "recovery"
    if value == "final_mark":
        return "mark"
    return "other"


def _trades_frame(trades: list[dict]) -> pl.DataFrame:
    if not trades:
        return pl.DataFrame(schema={"pnl": pl.Float64, "pnl_usd": pl.Float64})
    df = pl.DataFrame(trades)
    if "pnl" not in df.columns:
        df = df.with_columns(pl.lit(0.0).alias("pnl"))
    if "pnl_usd" not in df.columns:
        df = df.with_columns(pl.lit(0.0).alias("pnl_usd"))
    if "exit_family" not in df.columns:
        if "reason" in df.columns:
            families = [exit_family(reason) for reason in df["reason"].to_list()]
            df = df.with_columns(pl.Series("exit_family", families))
        else:
            df = df.with_columns(pl.lit("other").alias("exit_family"))
    return _with_regime_buckets(
        df.with_columns(
        pl.col("pnl").cast(pl.Float64),
        pl.col("pnl_usd").cast(pl.Float64),
        )
    )


def _with_regime_buckets(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    expressions: list[pl.Expr] = []
    if "entry_adx_14" in df.columns and "entry_adx_bucket" not in df.columns:
        expressions.append(
            pl.when(pl.col("entry_adx_14").cast(pl.Float64) <= 20.0)
            .then(pl.lit("low"))
            .when(pl.col("entry_adx_14").cast(pl.Float64) <= 35.0)
            .then(pl.lit("medium"))
            .otherwise(pl.lit("high"))
            .alias("entry_adx_bucket")
        )
    if "entry_volatility_ratio" in df.columns and "entry_volatility_bucket" not in df.columns:
        expressions.append(
            pl.when(pl.col("entry_volatility_ratio").cast(pl.Float64) < 0.75)
            .then(pl.lit("compressed"))
            .when(pl.col("entry_volatility_ratio").cast(pl.Float64) <= 1.5)
            .then(pl.lit("normal"))
            .otherwise(pl.lit("expanded"))
            .alias("entry_volatility_bucket")
        )
    if "entry_trend_return" in df.columns and "entry_trend_bucket" not in df.columns:
        expressions.append(
            pl.when(pl.col("entry_trend_return").cast(pl.Float64) <= -0.02)
            .then(pl.lit("downtrend"))
            .when(pl.col("entry_trend_return").cast(pl.Float64) >= 0.02)
            .then(pl.lit("uptrend"))
            .otherwise(pl.lit("flat"))
            .alias("entry_trend_bucket")
        )
    z_col = next(
        (
            col
            for col in ("entry_dynamic_z_score", "entry_robust_z_score", "entry_close_z_score")
            if col in df.columns
        ),
        "",
    )
    if z_col and "entry_zscore_bucket" not in df.columns:
        z_abs = pl.col(z_col).cast(pl.Float64).abs()
        expressions.append(
            pl.when(z_abs < 2.5)
            .then(pl.lit("moderate"))
            .when(z_abs < 3.5)
            .then(pl.lit("extreme"))
            .otherwise(pl.lit("tail"))
            .alias("entry_zscore_bucket")
        )
    if (
        "entry_bullish_liquidity_sweep" in df.columns
        and "entry_bearish_liquidity_sweep" in df.columns
        and "entry_liquidity_event" not in df.columns
    ):
        expressions.append(
            pl.when(pl.col("entry_bullish_liquidity_sweep").cast(pl.Float64) > 0.0)
            .then(pl.lit("bullish_reclaim"))
            .when(pl.col("entry_bearish_liquidity_sweep").cast(pl.Float64) > 0.0)
            .then(pl.lit("bearish_reclaim"))
            .otherwise(pl.lit("no_reclaim"))
            .alias("entry_liquidity_event")
        )
    if (
        "entry_liquidity_event_type" in df.columns
        and "entry_liquidity_event_type_bucket" not in df.columns
    ):
        expressions.append(
            pl.col("entry_liquidity_event_type")
            .fill_null("none")
            .alias("entry_liquidity_event_type_bucket")
        )
    if "entry_event_quality_score" in df.columns and "entry_event_quality_bucket" not in df.columns:
        quality = pl.col("entry_event_quality_score").cast(pl.Float64)
        expressions.append(
            pl.when(quality <= 0.0)
            .then(pl.lit("none"))
            .when(quality < 1.5)
            .then(pl.lit("low"))
            .when(quality < 2.5)
            .then(pl.lit("medium"))
            .otherwise(pl.lit("high"))
            .alias("entry_event_quality_bucket")
        )
    if (
        "entry_atr_percentile_100" in df.columns
        and "entry_atr_percentile_bucket" not in df.columns
    ):
        atr_pct = pl.col("entry_atr_percentile_100").cast(pl.Float64)
        expressions.append(
            pl.when(atr_pct.is_null())
            .then(pl.lit("unknown"))
            .when(atr_pct < 25.0)
            .then(pl.lit("low"))
            .when(atr_pct < 75.0)
            .then(pl.lit("normal"))
            .when(atr_pct < 90.0)
            .then(pl.lit("high"))
            .otherwise(pl.lit("extreme"))
            .alias("entry_atr_percentile_bucket")
        )
    if (
        "entry_key_level_proximity_bucket" in df.columns
        and "entry_key_level_proximity_context_bucket" not in df.columns
    ):
        expressions.append(
            pl.col("entry_key_level_proximity_bucket")
            .fill_null("breached_or_unknown")
            .alias("entry_key_level_proximity_context_bucket")
        )
    if "entry_z_pressure_side" in df.columns and "entry_z_pressure_side_bucket" not in df.columns:
        expressions.append(
            pl.col("entry_z_pressure_side").fill_null("none").alias("entry_z_pressure_side_bucket")
        )
    if (
        "entry_failed_bullish_sweep" in df.columns
        and "entry_failed_bearish_sweep" in df.columns
        and "entry_failed_sweep_bucket" not in df.columns
    ):
        expressions.append(
            pl.when(pl.col("entry_failed_bullish_sweep").cast(pl.Float64) > 0.0)
            .then(pl.lit("failed_bullish"))
            .when(pl.col("entry_failed_bearish_sweep").cast(pl.Float64) > 0.0)
            .then(pl.lit("failed_bearish"))
            .otherwise(pl.lit("no_failed_sweep"))
            .alias("entry_failed_sweep_bucket")
        )
    if (
        "entry_volume_impulse" in df.columns
        and "entry_bullish_liquidity_sweep" in df.columns
        and "entry_bearish_liquidity_sweep" in df.columns
        and "entry_volume_sweep_bucket" not in df.columns
    ):
        expressions.append(
            pl.when(
                (
                    (pl.col("entry_bullish_liquidity_sweep").cast(pl.Float64) > 0.0)
                    | (pl.col("entry_bearish_liquidity_sweep").cast(pl.Float64) > 0.0)
                )
                & (pl.col("entry_volume_impulse").cast(pl.Float64) > 0.0)
            )
            .then(pl.lit("volume_confirmed_reclaim"))
            .when(
                (pl.col("entry_bullish_liquidity_sweep").cast(pl.Float64) > 0.0)
                | (pl.col("entry_bearish_liquidity_sweep").cast(pl.Float64) > 0.0)
            )
            .then(pl.lit("unconfirmed_reclaim"))
            .otherwise(pl.lit("no_reclaim"))
            .alias("entry_volume_sweep_bucket")
        )
    if "entry_structure_trend_state" in df.columns and "entry_structure_bucket" not in df.columns:
        expressions.append(
            pl.col("entry_structure_trend_state")
            .fill_null("unknown")
            .alias("entry_structure_bucket")
        )
    if "entry_market_stage" in df.columns and "entry_market_stage_bucket" not in df.columns:
        expressions.append(
            pl.col("entry_market_stage").fill_null("unknown").alias("entry_market_stage_bucket")
        )
    if (
        "entry_market_stage_reason" in df.columns
        and "entry_market_stage_reason_bucket" not in df.columns
    ):
        expressions.append(
            pl.col("entry_market_stage_reason")
            .fill_null("none")
            .alias("entry_market_stage_reason_bucket")
        )
    if (
        "entry_stage_unknown_reason" in df.columns
        and "entry_stage_unknown_reason_bucket" not in df.columns
    ):
        expressions.append(
            pl.col("entry_stage_unknown_reason")
            .fill_null("none")
            .alias("entry_stage_unknown_reason_bucket")
        )
    if (
        "entry_near_range_high" in df.columns
        and "entry_near_range_low" in df.columns
        and "entry_range_location_bucket" not in df.columns
    ):
        expressions.append(
            pl.when(pl.col("entry_near_range_high").cast(pl.Float64) > 0.0)
            .then(pl.lit("near_range_high"))
            .when(pl.col("entry_near_range_low").cast(pl.Float64) > 0.0)
            .then(pl.lit("near_range_low"))
            .otherwise(pl.lit("range_mid_or_unknown"))
            .alias("entry_range_location_bucket")
        )
    mtf_bucket_sources = {
        "entry_mtf_state_key": "entry_mtf_state_bucket",
        "entry_mtf_structure_key": "entry_mtf_structure_bucket",
        "entry_mtf_stage_key": "entry_mtf_stage_bucket",
        "entry_mtf_event_state_key": "entry_mtf_event_state_bucket",
    }
    for source, target in mtf_bucket_sources.items():
        if source in df.columns and target not in df.columns:
            expressions.append(pl.col(source).cast(pl.Utf8).fill_null("unknown").alias(target))
    return df.with_columns(expressions) if expressions else df


def _equity_frame(
    equity: list[float],
    active_exposure: list[float] | None,
    timestamps: list[int] | None,
    signals: list[float] | None,
) -> pl.DataFrame:
    eq_series = pl.Series("portfolio_value", equity, dtype=pl.Float64)
    df = pl.DataFrame(
        {
            "portfolio_value": eq_series,
            "returns": eq_series.pct_change().fill_null(0.0),
        }
    )
    columns = []
    if active_exposure is not None and len(active_exposure) == len(equity):
        columns.append(pl.Series("active_exposure", active_exposure, dtype=pl.Float64))
    if timestamps is not None and len(timestamps) == len(equity):
        columns.append(pl.Series("timestamp", timestamps, dtype=pl.Int64))
    if signals is not None and len(signals) == len(equity):
        columns.append(pl.Series("signal", signals, dtype=pl.Float64))
    if columns:
        return df.with_columns(columns)
    return df


def _trade_stats(trades: pl.DataFrame) -> dict[str, float]:
    if trades.is_empty():
        return EMPTY_TRADE_STATS.copy()

    row = (
        trades.select(
            pl.col("pnl").mean().alias("mean_pnl"),
            pl.col("pnl").median().alias("median_pnl"),
            pl.col("pnl").std().alias("std_pnl"),
            pl.col("pnl_usd").mean().alias("mean_pnl_usd"),
            pl.len().alias("trade_count"),
        )
        .with_columns(
            pl.when((pl.col("std_pnl") > 0) & (pl.col("trade_count") > 0))
            .then(pl.col("mean_pnl") / pl.col("std_pnl") * pl.col("trade_count").sqrt())
            .otherwise(0.0)
            .alias("trade_sharpe")
        )
        .select(
            (pl.col("mean_pnl") * 100.0).alias("trade_expectancy_pct"),
            pl.col("mean_pnl_usd").alias("trade_expectancy_usd"),
            (pl.col("median_pnl") * 100.0).alias("median_trade_pct"),
            pl.col("trade_sharpe"),
        )
        .row(0, named=True)
    )
    return {key: _as_float(value) for key, value in row.items()}


def _series_values(frame: pl.DataFrame, preferred: str, fallback: str) -> pl.Series:
    if preferred in frame.columns:
        return frame[preferred].cast(pl.Float64)
    if fallback in frame.columns:
        return frame[fallback].cast(pl.Float64)
    return pl.Series([], dtype=pl.Float64)


def _worst_group(
    trades: pl.DataFrame,
    group_col: str,
    net_col: str,
    *,
    min_count: int = 2,
) -> tuple[str, float]:
    if trades.is_empty() or group_col not in trades.columns or net_col not in trades.columns:
        return "n/a", 0.0
    grouped = (
        trades.filter(pl.col(group_col).is_not_null())
        .group_by(group_col)
        .agg(
            pl.len().alias("count"),
            pl.col(net_col).cast(pl.Float64).mean().alias("mean_net"),
        )
        .filter(pl.col("count") >= min_count)
        .sort("mean_net")
    )
    if grouped.is_empty():
        return "n/a", 0.0
    row = grouped.row(0, named=True)
    return str(row[group_col] or "n/a"), round(_as_float(row["mean_net"]), 4)


def _max_consecutive_losses(values: list[float]) -> int:
    current = 0
    best = 0
    for value in values:
        if value <= 0.0:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _worst_rolling_sum(values: list[float], window: int = 5) -> float:
    if not values:
        return 0.0
    if len(values) <= window:
        return sum(values)
    return min(sum(values[i : i + window]) for i in range(0, len(values) - window + 1))


def _mean_series(series: pl.Series) -> float:
    return _as_float(series.mean()) if not series.is_empty() else 0.0


def _sum_negative_abs(series: pl.Series) -> float:
    return abs(_as_float(series.filter(series < 0).sum())) if not series.is_empty() else 0.0


def _worst_stop_group(
    stop_trades: pl.DataFrame,
    group_col: str,
    net_col: str,
) -> tuple[str, float, float]:
    if stop_trades.is_empty() or group_col not in stop_trades.columns:
        return "n/a", 0.0, 0.0
    grouped = (
        stop_trades.filter(pl.col(group_col).is_not_null())
        .group_by(group_col)
        .agg(pl.col(net_col).cast(pl.Float64).sum().alias("net_usd"))
        .sort("net_usd")
    )
    if grouped.is_empty():
        return "n/a", 0.0, 0.0
    total_loss = _sum_negative_abs(stop_trades[net_col].cast(pl.Float64))
    row = grouped.row(0, named=True)
    net = _as_float(row["net_usd"])
    share = abs(net) / total_loss * 100.0 if net < 0 and total_loss > 0 else 0.0
    return str(row[group_col] or "n/a"), round(net, 4), round(share, 2)


def compute_stop_effectiveness(trades: pl.DataFrame) -> StopEffectiveness:
    if trades.is_empty() or "reason" not in trades.columns:
        return StopEffectiveness()
    net_col = "net_pnl_usd" if "net_pnl_usd" in trades.columns else "pnl_usd"
    gross_col = "gross_pnl_usd" if "gross_pnl_usd" in trades.columns else net_col
    stop_trades = trades.filter(pl.col("reason") == "stop")
    if stop_trades.is_empty():
        return StopEffectiveness()
    stop_net = stop_trades[net_col].cast(pl.Float64)
    stop_gross = stop_trades[gross_col].cast(pl.Float64)
    all_gross = trades[gross_col].cast(pl.Float64)
    total_gross_loss = _sum_negative_abs(all_gross)
    stop_gross_loss = _sum_negative_abs(stop_gross)
    worst_side, worst_side_net, worst_side_share = _worst_stop_group(stop_trades, "side", net_col)
    worst_signal, worst_signal_net, worst_signal_share = _worst_stop_group(
        stop_trades, "signal_id", net_col
    )
    notional = (
        stop_trades["notional_pct_capital"].cast(pl.Float64)
        if "notional_pct_capital" in stop_trades.columns
        else pl.Series([], dtype=pl.Float64)
    )
    return StopEffectiveness(
        stop_trades=stop_trades.height,
        stop_net_pnl_usd=round(_as_float(stop_net.sum()), 4),
        stop_gross_loss_usd=round(stop_gross_loss, 4),
        stop_avg_pnl_pct=round(_mean_series(stop_trades["pnl"].cast(pl.Float64)) * 100.0, 4),
        stop_avg_pnl_usd=round(_mean_series(stop_net), 4),
        stop_worst_pnl_usd=round(_as_float(stop_net.min()), 4),
        stop_avg_notional_pct=round(_mean_series(notional), 4),
        stop_loss_share_pct=round(
            stop_gross_loss / total_gross_loss * 100.0 if total_gross_loss > 0 else 0.0, 2
        ),
        worst_stop_side=worst_side,
        worst_stop_side_net_usd=worst_side_net,
        worst_stop_signal_id=worst_signal,
        worst_stop_signal_net_usd=worst_signal_net,
        worst_stop_side_share_pct=worst_side_share,
        worst_stop_signal_share_pct=worst_signal_share,
    )


def compute_drawdown_path_diagnostics(trades: pl.DataFrame) -> DrawdownPathDiagnostics:
    if trades.is_empty() or "entry_drawdown_pct" not in trades.columns:
        return DrawdownPathDiagnostics()
    entry_dd = trades["entry_drawdown_pct"].cast(pl.Float64)
    in_dd = trades.filter(pl.col("entry_drawdown_pct").cast(pl.Float64) > 0)
    max_notional = 0.0
    notional_col = (
        "post_entry_total_notional_pct"
        if "post_entry_total_notional_pct" in trades.columns
        else "entry_total_notional_pct"
        if "entry_total_notional_pct" in trades.columns
        else ""
    )
    if not in_dd.is_empty() and notional_col:
        max_notional = _as_float(in_dd[notional_col].cast(pl.Float64).max())
    worst_signal = "n/a"
    worst_side = "n/a"
    if not in_dd.is_empty():
        max_row = in_dd.sort("entry_drawdown_pct", descending=True).row(0, named=True)
        worst_signal = str(max_row.get("entry_signal_id") or max_row.get("signal_id") or "n/a")
        worst_side = str(max_row.get("side") or "n/a")
    return DrawdownPathDiagnostics(
        entries_while_drawdown_count=in_dd.height,
        entries_while_drawdown_pct=round(in_dd.height / trades.height * 100.0, 2),
        avg_entry_drawdown_pct=round(_mean_series(entry_dd), 4),
        max_entry_drawdown_pct=round(_as_float(entry_dd.max()), 4),
        max_notional_during_drawdown_pct=round(max_notional, 4),
        worst_drawdown_entry_signal_id=worst_signal,
        worst_drawdown_side=worst_side,
    )


def compute_yield_attribution(
    trades: pl.DataFrame,
    trade_expectancy_pct: float,
    trade_expectancy_usd: float,
) -> YieldAttribution:
    if trades.is_empty():
        return YieldAttribution(
            price_expectancy_pct=trade_expectancy_pct,
            dollar_expectancy_usd=trade_expectancy_usd,
        )
    net = _series_values(trades, "net_pnl_usd", "pnl_usd")
    gross = _series_values(trades, "gross_pnl_usd", "pnl_usd")
    fees = trades["fee_usd"].cast(pl.Float64) if "fee_usd" in trades.columns else pl.Series([])
    net_col = "net_pnl_usd" if "net_pnl_usd" in trades.columns else "pnl_usd"
    gross_pnl = _as_float(gross.sum())
    net_pnl = _as_float(net.sum())
    fee_usd = _as_float(fees.sum()) if not fees.is_empty() else 0.0
    gross_profit = _as_float(gross.filter(gross > 0).sum())
    gross_loss = abs(_as_float(gross.filter(gross <= 0).sum()))
    fee_drag = fee_usd / gross_profit * 100.0 if gross_profit > 0 else 0.0
    net_values = [_as_float(value) for value in net.to_list()]
    worst_side, worst_side_exp = _worst_group(trades, "side", net_col)
    worst_exit, worst_exit_exp = _worst_group(trades, "reason", net_col)
    worst_signal, worst_signal_exp = _worst_group(trades, "signal_id", net_col)
    size_weighted_expectancy = 0.0
    avg_win_notional = 0.0
    avg_loss_notional = 0.0
    loss_to_win_notional = 0.0
    avg_win_contracts = 0.0
    avg_loss_contracts = 0.0
    if "entry_notional_usd" in trades.columns:
        notionals = trades["entry_notional_usd"].cast(pl.Float64)
        notional_sum = _as_float(notionals.sum())
        if notional_sum > 0 and "pnl" in trades.columns:
            weighted = (trades["pnl"].cast(pl.Float64) * notionals).sum()
            size_weighted_expectancy = _as_float(weighted) / notional_sum * 100.0
    if "notional_pct_capital" in trades.columns:
        notionals_pct = trades["notional_pct_capital"].cast(pl.Float64)
        avg_win_notional = _mean_series(notionals_pct.filter(net > 0))
        avg_loss_notional = _mean_series(notionals_pct.filter(net <= 0))
        if avg_win_notional > 0 and avg_loss_notional > 0:
            loss_to_win_notional = avg_loss_notional / avg_win_notional
    if "contracts" in trades.columns:
        contracts = trades["contracts"].cast(pl.Float64)
        avg_win_contracts = _mean_series(contracts.filter(net > 0))
        avg_loss_contracts = _mean_series(contracts.filter(net <= 0))
    return YieldAttribution(
        gross_pnl_usd=round(gross_pnl, 4),
        net_pnl_usd=round(net_pnl, 4),
        fee_usd=round(fee_usd, 4),
        gross_profit_usd=round(gross_profit, 4),
        gross_loss_usd=round(gross_loss, 4),
        fee_drag_pct=round(fee_drag, 2),
        price_expectancy_pct=trade_expectancy_pct,
        dollar_expectancy_usd=trade_expectancy_usd,
        worst_side=worst_side,
        worst_side_expectancy_usd=worst_side_exp,
        worst_exit_reason=worst_exit,
        worst_exit_expectancy_usd=worst_exit_exp,
        worst_signal_id=worst_signal,
        worst_signal_expectancy_usd=worst_signal_exp,
        max_consecutive_losses=_max_consecutive_losses(net_values),
        worst_5_trade_net_pnl_usd=round(_worst_rolling_sum(net_values), 4),
        size_weighted_expectancy_pct=round(size_weighted_expectancy, 4),
        avg_win_notional_pct_capital=round(avg_win_notional, 4),
        avg_loss_notional_pct_capital=round(avg_loss_notional, 4),
        loss_to_win_notional_ratio=round(loss_to_win_notional, 4),
        avg_win_contracts=round(avg_win_contracts, 4),
        avg_loss_contracts=round(avg_loss_contracts, 4),
    )


def _active_stats(equity: pl.DataFrame, periods_per_year: int) -> dict[str, float]:
    active_expr = (
        pl.col("active_exposure").abs() > 1e-12
        if "active_exposure" in equity.columns
        else pl.col("returns").abs() > 1e-12
    )
    total_bars = max(equity.height, 1)
    row = (
        equity.with_columns(active_expr.alias("is_active"))
        .filter(pl.col("is_active"))
        .select(
            (pl.len() / total_bars * 100.0).alias("active_bar_pct"),
            pl.col("returns").cast(pl.Float64).mean().alias("active_mean"),
            pl.col("returns").cast(pl.Float64).std().alias("active_std"),
        )
        .with_columns(
            pl.when(pl.col("active_std") > 0)
            .then(pl.col("active_mean") / pl.col("active_std") * periods_per_year**0.5)
            .otherwise(0.0)
            .alias("active_bar_sharpe")
        )
        .select("active_bar_pct", "active_bar_sharpe")
        .row(0, named=True)
    )
    return {key: _as_float(value) for key, value in row.items()}


@dataclass
class Report:
    label: str
    trades: pl.DataFrame
    equity: pl.DataFrame
    metrics: EvalMetrics
    active_bar_pct: float
    active_bar_sharpe: float
    trade_expectancy_pct: float
    trade_expectancy_usd: float
    median_trade_pct: float
    trade_sharpe: float
    unstable_annualization: bool
    yield_attribution: YieldAttribution = field(default_factory=YieldAttribution)
    stop_effectiveness: StopEffectiveness = field(default_factory=StopEffectiveness)
    drawdown_path: DrawdownPathDiagnostics = field(default_factory=DrawdownPathDiagnostics)
    diagnostics: BacktestDiagnostics | None = None
    metadata: tuple[str, ...] = ()

    @classmethod
    def from_raw(
        cls,
        trades: list[dict],
        equity: list[float],
        pair,
        *,
        label: str = "",
        active_exposure: list[float] | None = None,
        timestamps: list[int] | None = None,
        signals: list[float] | None = None,
        diagnostics: BacktestDiagnostics | None = None,
        metadata: Sequence[str] = (),
        periods_per_year: int | None = None,
    ) -> Report:
        t_df = _trades_frame(trades)
        eq_df = _equity_frame(equity, active_exposure, timestamps, signals)

        metrics = compute_metrics(eq_df, trades=t_df, periods_per_year=periods_per_year)
        trade_stats = _trade_stats(t_df)
        active_stats = _active_stats(eq_df, periods_per_year or infer_periods_per_year(eq_df))
        trade_expectancy_pct = round(trade_stats["trade_expectancy_pct"], 4)
        trade_expectancy_usd = round(trade_stats["trade_expectancy_usd"], 4)

        unstable = metrics.num_trades < 20 or active_stats["active_bar_pct"] < 10.0

        return cls(
            label=label or pair.asset.symbol,
            trades=t_df,
            equity=eq_df,
            metrics=metrics,
            active_bar_pct=round(active_stats["active_bar_pct"], 2),
            active_bar_sharpe=round(active_stats["active_bar_sharpe"], 4),
            trade_expectancy_pct=trade_expectancy_pct,
            trade_expectancy_usd=trade_expectancy_usd,
            median_trade_pct=round(trade_stats["median_trade_pct"], 4),
            trade_sharpe=round(trade_stats["trade_sharpe"], 4),
            unstable_annualization=unstable,
            yield_attribution=compute_yield_attribution(
                t_df,
                trade_expectancy_pct,
                trade_expectancy_usd,
            ),
            stop_effectiveness=compute_stop_effectiveness(t_df),
            drawdown_path=compute_drawdown_path_diagnostics(t_df),
            diagnostics=diagnostics,
            metadata=tuple(metadata),
        )

    def summary(self) -> str:
        m = self.metrics
        return (
            f"{self.label:30s} {m.num_trades:4d}tr  "
            f"{m.win_rate_pct:5.1f}%wr  {m.profit_factor:5.2f}pf  "
            f"Exp={self.trade_expectancy_pct:+6.2f}%  "
            f"TSh={self.trade_sharpe:+6.2f}  "
            f"ABSh={self.active_bar_sharpe:+6.2f}"
        )

    def table(self) -> str:
        m = self.metrics
        return (
            f"  Ret={m.total_return_pct:+7.2f}%  Exp={self.trade_expectancy_pct:+.2f}%  "
            f"Exp$={self.trade_expectancy_usd:+.2f}  MedT={self.median_trade_pct:+.2f}%  "
            f"TSharpe={self.trade_sharpe:+.2f}\n"
            f"  DD={m.max_drawdown_pct:5.1f}%  AvgDD={m.avg_drawdown_pct:.1f}%  "
            f"DDDays={m.drawdown_days:d}  ActiveBars={self.active_bar_pct:.1f}%  "
            f"ABSharpe={self.active_bar_sharpe:+.2f}\n"
            f"  Trades={m.num_trades:d}  WR={m.win_rate_pct:.1f}%  "
            f"AvgW={m.avg_win_pct:+.2f}%  AvgL={m.avg_loss_pct:+.2f}%  "
            f"P/L={m.profit_loss_ratio:.2f}  PF={m.profit_factor:.2f}\n"
            f"  CalSharpe={m.sharpe_ratio:+.2f}  CalSortino={m.sortino_ratio:+.2f}  "
            f"Ann={m.annual_return_pct:+.2f}%  Vol={m.annual_volatility_pct:.2f}%\n"
            f"  IC={m.ic_mean:+.4f}  IC_IR={m.ic_ir:+.2f}  IC+={m.ic_positive_pct:.0f}%  "
            f"UnstableAnn={'yes' if self.unstable_annualization else 'no'}"
        )

    def metric_sections(self) -> str:
        m = self.metrics
        d = self.diagnostics
        y = self.yield_attribution
        stop = self.stop_effectiveness
        dd_path = self.drawdown_path
        fee = d.fee_usd if d is not None else 0.0
        avg_notional = d.avg_notional_exposure_pct if d is not None else 0.0
        exposure_return = m.total_return_pct / avg_notional if avg_notional > 0 else 0.0
        metadata = "\n".join(f"  {item}" for item in self.metadata) or "  none"
        audit = "  diagnostics unavailable"
        if d is not None:
            audit = (
                f"  Bars={d.bars_processed}/{d.bars}  "
                f"StoppedEarly={'yes' if d.stopped_early else 'no'}  "
                f"OpenPos={d.final_open_positions}  Fees=${fee:.2f}\n"
                f"  DataStart={d.data_start or 'n/a'}  DataEnd={d.data_end or 'n/a'}  "
                f"MTM={'yes' if d.mark_to_market else 'no'}  "
                f"DDStop={d.drawdown_stop_pct if d.drawdown_stop_pct is not None else 'none'}"
            )
        return (
            "Run metadata\n"
            f"{metadata}\n"
            "Engine/Data Audit\n"
            f"{audit}\n"
            "Trade Metrics\n"
            f"  Trades={m.num_trades}  WR={m.win_rate_pct:.1f}%  PF={m.profit_factor:.2f}  "
            f"Exp={self.trade_expectancy_pct:+.2f}%  Exp$={self.trade_expectancy_usd:+.2f}  "
            f"MedT={self.median_trade_pct:+.2f}%\n"
            "Yield Attribution\n"
            f"  NetPnL=${y.net_pnl_usd:+.2f}  GrossPnL=${y.gross_pnl_usd:+.2f}  "
            f"Fees=${y.fee_usd:.2f}  FeeDrag={y.fee_drag_pct:.1f}%\n"
            f"  ExpPrice={y.price_expectancy_pct:+.2f}%  "
            f"ExpDollar=${y.dollar_expectancy_usd:+.2f}  "
            f"SizeWExp={y.size_weighted_expectancy_pct:+.2f}%  "
            f"MaxConsecLosses={y.max_consecutive_losses}  "
            f"Worst5Net=${y.worst_5_trade_net_pnl_usd:+.2f}\n"
            f"  AvgWinNotional={y.avg_win_notional_pct_capital:.1f}%cap  "
            f"AvgLossNotional={y.avg_loss_notional_pct_capital:.1f}%cap  "
            f"Loss/WinNotional={y.loss_to_win_notional_ratio:.2f}  "
            f"AvgWinCt={y.avg_win_contracts:.2f}  AvgLossCt={y.avg_loss_contracts:.2f}\n"
            f"  WorstExit={y.worst_exit_reason} Exp$={y.worst_exit_expectancy_usd:+.2f}  "
            f"WorstSide={y.worst_side} Exp$={y.worst_side_expectancy_usd:+.2f}  "
            f"WorstSignal={y.worst_signal_id} Exp$={y.worst_signal_expectancy_usd:+.2f}\n"
            "Stop Effectiveness\n"
            f"  StopTrades={stop.stop_trades}  StopNet=${stop.stop_net_pnl_usd:+.2f}  "
            f"StopAvg={stop.stop_avg_pnl_pct:+.2f}%/${stop.stop_avg_pnl_usd:+.2f}  "
            f"StopWorst=${stop.stop_worst_pnl_usd:+.2f}  "
            f"StopLossShare={stop.stop_loss_share_pct:.1f}%\n"
            f"  StopAvgNotional={stop.stop_avg_notional_pct:.1f}%cap  "
            f"WorstStopSide={stop.worst_stop_side} ${stop.worst_stop_side_net_usd:+.2f}  "
            f"WorstStopSignal={stop.worst_stop_signal_id} "
            f"${stop.worst_stop_signal_net_usd:+.2f}\n"
            f"{self.group_attribution()}\n"
            "Exposure Metrics\n"
            f"  ActiveBars={self.active_bar_pct:.1f}%  AvgNotional={avg_notional:.1f}%cap  "
            f"RetPerAvgNotional={exposure_return:+.2f}\n"
            "Equity Metrics\n"
            f"  Ret={m.total_return_pct:+.2f}%  DD={m.max_drawdown_pct:.1f}%  "
            f"AvgDD={m.avg_drawdown_pct:.1f}%  DDDays={m.drawdown_days}\n"
            "Drawdown Path Diagnostics\n"
            f"  EntriesInDD={dd_path.entries_while_drawdown_count} "
            f"({dd_path.entries_while_drawdown_pct:.1f}%)  "
            f"AvgEntryDD={dd_path.avg_entry_drawdown_pct:.1f}%  "
            f"MaxEntryDD={dd_path.max_entry_drawdown_pct:.1f}%  "
            f"MaxNotionalInDD={dd_path.max_notional_during_drawdown_pct:.1f}%  "
            f"WorstDDSignal={dd_path.worst_drawdown_entry_signal_id}  "
            f"WorstDDSide={dd_path.worst_drawdown_side}\n"
            "Annualized Diagnostics\n"
            f"  CalSharpe={m.sharpe_ratio:+.2f}  CalSortino={m.sortino_ratio:+.2f}  "
            f"Ann={m.annual_return_pct:+.2f}%  Vol={m.annual_volatility_pct:.2f}%  "
            f"UnstableAnn={'yes' if self.unstable_annualization else 'no'}"
        )

    def diagnostics_table(self) -> str:
        if self.diagnostics is None:
            return ""
        d = self.diagnostics
        lifecycle = d.lifecycle
        risk = d.risk
        dd_path = self.drawdown_path
        reasons = ", ".join(f"{k}:{v}" for k, v in sorted(d.exit_reasons.items())) or "none"
        return (
            f"  BarsProcessed={d.bars_processed}/{d.bars}  "
            f"StoppedEarly={'yes' if d.stopped_early else 'no'}  "
            f"SignalBars={d.nonzero_signal_bars}/{d.bars} ({d.signal_bar_pct:.1f}%)  "
            f"Long={d.long_signal_bars}  Short={d.short_signal_bars}\n"
            f"  Entries={d.entries}  Exits={d.exits}  AvgHold={d.avg_bars_held:.1f} bars  "
            f"ExitReasons={reasons}\n"
            f"  AvgContracts={d.avg_active_exposure:.2f}  "
            f"MaxContracts={d.max_active_exposure:.2f}  "
            f"AvgNotional={d.avg_notional_exposure_pct:.1f}%cap  "
            f"MaxNotional={d.max_notional_exposure_pct:.1f}%cap\n"
            f"  Fees=${d.fee_usd:.2f}  OpenPos={d.final_open_positions}  "
            f"OpenUnrealized=${d.open_unrealized_pnl_usd:.2f}\n"
            "Basket Lifecycle Diagnostics\n"
            f"  Signals entry={lifecycle.entry_signals}  "
            f"Actions enter={lifecycle.entry_actions} exit={lifecycle.exit_actions} "
            f"grid={lifecycle.grid_actions} hedge={lifecycle.hedge_actions} "
            f"recovery={lifecycle.recovery_actions}\n"
            f"  EntryAccept={lifecycle.entry_acceptance_rate_pct:.1f}%  "
            f"BlockedEntries={lifecycle.blocked_entry_signals}  "
            f"DuplicateSuppressed={lifecycle.duplicate_entry_suppressed}  "
            f"CapacityBlocked={lifecycle.capacity_blocked_entries}  "
            f"SizingBlocked={lifecycle.sizing_blocked_entries}\n"
            f"  BlockReasons={_format_counts(lifecycle.blocked_entry_reasons)}  "
            f"MaxBaskets={lifecycle.max_simultaneous_baskets}  "
            f"SameBarExitEntry={lifecycle.same_bar_exit_entry_count}  "
            f"FinalOpen={lifecycle.final_open_positions}\n"
            "Basket Mechanics Diagnostics\n"
            f"  ActionEvents={lifecycle.action_event_count}  "
            f"StackedEntries={lifecycle.stacked_entry_count}  "
            f"StackedEntryNet=${lifecycle.stacked_entry_net_pnl_usd:+.2f}\n"
            "Min-Contract Diagnostics\n"
            f"  MinContractBlocks={lifecycle.min_contract_block_count}  "
            f"MedianRequiredCapital=${lifecycle.median_required_capital_for_min_contract:.2f}  "
            f"MedianRequiredRiskPct={lifecycle.median_required_risk_pct_for_min_contract:.2f}%  "
            f"BlockedByRisk={lifecycle.blocked_by_risk_count}  "
            f"BlockedByNotional={lifecycle.blocked_by_notional_count}\n"
            "Risk Control Diagnostics\n"
            f"  StopExits={risk.stop_exit_count}  StopNet=${risk.stop_exit_net_pnl_usd:+.2f}  "
            f"RecoveredStops={risk.recovered_stop_exit_count}  "
            f"RecoveredStopNet=${risk.recovered_stop_exit_net_pnl_usd:+.2f}  "
            f"RecoveryNet=${risk.recovery_net_pnl_usd:+.2f}  "
            f"DDStop={risk.drawdown_stop_pct if risk.drawdown_stop_pct is not None else 'none'}  "
            f"StoppedEarly={'yes' if risk.stopped_early else 'no'}\n"
            "Recovery Mechanics Diagnostics\n"
            f"  PreemptedStop={risk.recovery_preempted_stop_count}  "
            f"PreemptedTime={risk.recovery_preempted_time_count}  "
            f"PreemptedTrailing={risk.recovery_preempted_trailing_count}  "
            f"UnsizedActions={risk.recovery_unsized_actions}  "
            f"CapBreaches={risk.recovery_cap_breach_actions}\n"
            f"  BlockedRecovery={risk.recovery_blocked_actions}  "
            f"AllowedRecovery={risk.recovery_allowed_actions}  "
            f"RecoveryNotionalAfter={risk.recovery_notional_after_pct:.1f}%cap  "
            f"BlockedReasons={_format_counts(risk.recovery_blocked_reasons)}\n"
            "Intrabar Ambiguity Diagnostics\n"
            f"  AmbiguousStopTarget={risk.ambiguous_stop_target_count}  "
            f"StopFirstNet=${risk.ambiguous_stop_net_pnl_usd:+.2f}  "
            f"TargetFirstNet=${risk.target_first_counterfactual_net_pnl_usd:+.2f}  "
            f"AmbiguityImpact=${risk.ambiguity_impact_usd:+.2f}\n"
            "Drawdown Path Diagnostics\n"
            f"  EntriesInDD={dd_path.entries_while_drawdown_count}  "
            f"AvgEntryDD={dd_path.avg_entry_drawdown_pct:.1f}%  "
            f"MaxEntryDD={dd_path.max_entry_drawdown_pct:.1f}%  "
            f"MaxNotionalInDD={dd_path.max_notional_during_drawdown_pct:.1f}%  "
            f"WorstDDSignal={dd_path.worst_drawdown_entry_signal_id}"
        )

    def group_attribution(self) -> str:
        parts = ["Group Attribution"]
        for group_col, title in (
            ("side", "By side"),
            ("exit_family", "By exit family"),
            ("reason", "By exit"),
            ("signal_id", "By signal"),
            ("entry_adx_bucket", "By entry ADX regime"),
            ("entry_volatility_bucket", "By entry volatility regime"),
            ("entry_trend_bucket", "By entry trend regime"),
            ("entry_zscore_bucket", "By entry Z-score magnitude"),
            ("entry_liquidity_event", "By entry liquidity event"),
            ("entry_liquidity_event_type_bucket", "By entry liquidity event type"),
            ("entry_event_quality_bucket", "By entry event quality"),
            ("entry_failed_sweep_bucket", "By entry failed sweep"),
            ("entry_volume_sweep_bucket", "By entry volume sweep"),
            ("entry_structure_bucket", "By entry structure"),
            ("entry_market_stage_bucket", "By entry market stage"),
            ("entry_market_stage_reason_bucket", "By entry market stage reason"),
            ("entry_stage_unknown_reason_bucket", "By entry unknown stage reason"),
            ("entry_range_location_bucket", "By entry range location"),
            ("entry_atr_percentile_bucket", "By entry ATR percentile"),
            ("entry_key_level_proximity_bucket", "By entry key-level proximity"),
            ("entry_z_pressure_side_bucket", "By entry Z pressure side"),
        ):
            table = format_group_attribution(self.trades, group_col)
            if table:
                parts.append(title)
                parts.append(table)
        for left_col, right_col, title in (
            ("side", "entry_structure_bucket", "By side x structure"),
            (
                "entry_market_stage_bucket",
                "entry_liquidity_event_type_bucket",
                "By stage x event",
            ),
            ("entry_z_pressure_side_bucket", "side", "By Z pressure x side"),
        ):
            table = format_pair_attribution(self.trades, left_col, right_col)
            if table:
                parts.append(title)
                parts.append(table)
        return "\n".join(parts) if len(parts) > 1 else "Group Attribution\n  n/a"


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    n_cols = len(headers)
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def _pad(cell: str, width: int) -> str:
        return cell + " " * (width - len(cell))

    lines = ["  ".join(_pad(headers[i], col_widths[i]) for i in range(n_cols))]
    lines.append("  ".join("-" * col_widths[i] for i in range(n_cols)))
    for row in rows:
        lines.append("  ".join(_pad(str(row[i]), col_widths[i]) for i in range(n_cols)))
    return "\n".join(lines)


def _format_counts(counts: dict[str, int]) -> str:
    return ",".join(f"{key}:{value}" for key, value in sorted(counts.items())) or "none"


def format_group_attribution(trades: pl.DataFrame, group_col: str) -> str:
    if trades.is_empty() or group_col not in trades.columns:
        return ""
    net_col = "net_pnl_usd" if "net_pnl_usd" in trades.columns else "pnl_usd"
    gross_col = "gross_pnl_usd" if "gross_pnl_usd" in trades.columns else net_col
    aggregations = [
        pl.len().alias("trades"),
        (pl.col("pnl") > 0).mean().mul(100.0).alias("wr"),
        pl.col("pnl").mean().mul(100.0).alias("exp_pct"),
        pl.col(net_col).cast(pl.Float64).mean().alias("exp_usd"),
        pl.col(net_col).cast(pl.Float64).sum().alias("net_usd"),
        pl.when(pl.col(gross_col).cast(pl.Float64) > 0)
        .then(pl.col(gross_col).cast(pl.Float64))
        .otherwise(0.0)
        .sum()
        .alias("gross_win"),
        pl.when(pl.col(gross_col).cast(pl.Float64) <= 0)
        .then(-pl.col(gross_col).cast(pl.Float64))
        .otherwise(0.0)
        .sum()
        .alias("gross_loss"),
    ]
    if "notional_pct_capital" in trades.columns:
        aggregations.append(
            pl.col("notional_pct_capital").cast(pl.Float64).mean().alias("avg_notional")
        )
    else:
        aggregations.append(pl.lit(0.0).alias("avg_notional"))
    grouped = (
        trades.filter(pl.col(group_col).is_not_null())
        .group_by(group_col)
        .agg(aggregations)
        .with_columns(
            pl.when(pl.col("gross_loss") > 0)
            .then(pl.col("gross_win") / pl.col("gross_loss"))
            .otherwise(float("inf"))
            .alias("pf")
        )
        .sort("net_usd")
    )
    if grouped.is_empty():
        return ""
    rows = []
    for row in grouped.iter_rows(named=True):
        rows.append(
            [
                str(row[group_col] or "n/a"),
                str(row["trades"]),
                f"{_as_float(row['wr']):.0f}",
                f"{_as_float(row['pf']):.2f}",
                f"{_as_float(row['exp_pct']):+.2f}",
                f"{_as_float(row['exp_usd']):+.2f}",
                f"{_as_float(row['net_usd']):+.2f}",
                f"{_as_float(row['avg_notional']):.1f}",
            ]
        )
    return format_table(["Group", "Trades", "WR%", "PF", "Exp%", "Exp$", "Net$", "AvgNot%"], rows)


def format_pair_attribution(
    trades: pl.DataFrame,
    left_col: str,
    right_col: str,
    *,
    limit: int = 8,
) -> str:
    if trades.is_empty() or not {left_col, right_col} <= set(trades.columns):
        return ""
    net_col = "net_pnl_usd" if "net_pnl_usd" in trades.columns else "pnl_usd"
    grouped = (
        trades.filter(pl.col(left_col).is_not_null() & pl.col(right_col).is_not_null())
        .group_by(left_col, right_col)
        .agg(
            pl.len().alias("trades"),
            (pl.col("pnl") > 0).mean().mul(100.0).alias("wr"),
            pl.col("pnl").mean().mul(100.0).alias("exp_pct"),
            pl.col(net_col).cast(pl.Float64).mean().alias("exp_usd"),
            pl.col(net_col).cast(pl.Float64).sum().alias("net_usd"),
        )
        .sort("net_usd")
        .head(limit)
    )
    if grouped.is_empty():
        return ""
    rows = []
    for row in grouped.iter_rows(named=True):
        rows.append(
            [
                str(row[left_col] or "n/a"),
                str(row[right_col] or "n/a"),
                str(row["trades"]),
                f"{_as_float(row['wr']):.0f}",
                f"{_as_float(row['exp_pct']):+.2f}",
                f"{_as_float(row['exp_usd']):+.2f}",
                f"{_as_float(row['net_usd']):+.2f}",
            ]
        )
    return format_table(["Left", "Right", "Trades", "WR%", "Exp%", "Exp$", "Net$"], rows)


def compare(*reports: Report) -> str:
    headers = ["Label", "Trades", "WR%", "PF", "Exp%", "TSh", "ABSh", "Ret%", "CalSh"]
    rows = []
    for r in reports:
        m = r.metrics
        rows.append(
            [
                r.label,
                str(m.num_trades),
                f"{m.win_rate_pct:.0f}",
                f"{m.profit_factor:.2f}",
                f"{r.trade_expectancy_pct:+.2f}",
                f"{r.trade_sharpe:+.2f}",
                f"{r.active_bar_sharpe:+.2f}",
                f"{m.total_return_pct:+.2f}",
                f"{m.sharpe_ratio:+.2f}",
            ]
        )
    return format_table(headers, rows)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_finite(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if finite:
        return _mean(finite)
    if any(math.isinf(value) and value > 0 for value in values):
        return float("inf")
    return 0.0


def format_strategy_summary(reports: Sequence[Report]) -> str:
    total_trades = sum(r.metrics.num_trades for r in reports)
    avg_return = _mean([r.metrics.total_return_pct for r in reports])
    avg_expectancy = _mean([r.trade_expectancy_pct for r in reports])
    best = max(reports, key=lambda r: r.metrics.total_return_pct)
    worst = min(reports, key=lambda r: r.metrics.total_return_pct)
    return (
        "Current strategy metrics\n"
        f"  Pairs={len(reports)}  Trades={total_trades}  "
        f"AvgRet={avg_return:+.2f}%  AvgExp={avg_expectancy:+.2f}%\n"
        f"  Best={best.label} {best.metrics.total_return_pct:+.2f}%  "
        f"Worst={worst.label} {worst.metrics.total_return_pct:+.2f}%"
    )


def format_strategy_recommendations(strategy: str, reports: Sequence[Report]) -> str:
    if not reports:
        return "Strategy recommendation\n  none"

    total_trades = sum(r.metrics.num_trades for r in reports)
    avg_expectancy = _mean([r.trade_expectancy_pct for r in reports])
    avg_pf = _mean_finite([r.metrics.profit_factor for r in reports])
    worst_dd = max(r.metrics.max_drawdown_pct for r in reports)
    unstable = sum(1 for r in reports if r.unstable_annualization)
    losing = avg_expectancy < 0.0 or avg_pf < 1.0
    sparse = total_trades < 20
    recovery_experimental = any(
        r.diagnostics is not None
        and (
            r.diagnostics.lifecycle.recovery_actions > 0
            or r.diagnostics.risk.recovery_blocked_actions > 0
            or r.diagnostics.risk.recovery_unsized_actions > 0
            or r.diagnostics.risk.recovery_cap_breach_actions > 0
        )
        for r in reports
    )

    lines = ["Strategy recommendation"]
    if recovery_experimental:
        lines.append(
            "  RECOVERY_EXPERIMENTAL: do not rank this run as a candidate baseline; "
            "use NoRecovery for candidate selection."
        )
        lines.append(
            "  Recovery actions were active or blocked by risk gates; compare only as a "
            "mechanics stress test."
        )
    elif losing:
        lines.append(
            f"  Reject current {strategy} baseline for allocation: AvgExp={avg_expectancy:+.2f}% "
            f"and finite AvgPF={avg_pf:.2f}."
        )
        lines.append(
            "  Next tuning should improve entry quality first; do not loosen filters "
            "or add recovery while expectancy is negative."
        )
    else:
        lines.append(
            f"  Candidate baseline: AvgExp={avg_expectancy:+.2f}% and finite AvgPF={avg_pf:.2f}."
        )
        lines.append(
            "  Validate with rolling or walk-forward before tuning exits or increasing exposure."
        )

    if sparse:
        lines.append(
            f"  Sample is sparse ({total_trades} trades); treat calendar "
            "Sharpe/Sortino as tertiary."
        )
    if worst_dd > 25.0:
        lines.append(
            f"  Worst drawdown is high ({worst_dd:.1f}%); cap exposure before "
            "increasing basket count."
        )
    if unstable:
        lines.append(
            f"  Unstable annualization on {unstable}/{len(reports)} symbols; "
            "prioritize PF, expectancy, drawdown, and OOS windows."
        )
    if strategy.startswith("momentum_burst"):
        lines.append(
            "  Current evidence favors comparing against rsi_bounce_reversion "
            "before more momentum sweeps."
        )
    return "\n".join(lines)


def _exposure_adjusted_return(report: Report) -> float:
    exposure_pct = (
        report.diagnostics.avg_notional_exposure_pct if report.diagnostics is not None else 0.0
    )
    if exposure_pct <= 0.0:
        return 0.0
    return report.metrics.total_return_pct / exposure_pct


def _is_data_incomplete(report: Report) -> bool:
    return any(item == "data_quality=data_incomplete" for item in report.metadata)


def format_symbol_rankings(reports: Sequence[Report]) -> str:
    rankable = [
        report
        for report in reports
        if report.metrics.num_trades > 0 and not _is_data_incomplete(report)
    ]
    if not rankable:
        return "Symbol rankings\n  none"

    skipped = len(reports) - len(rankable)
    skip_note = f"\n  RankingSkipped={skipped} diagnostic-only/no-trade reports" if skipped else ""
    best_return = max(rankable, key=lambda r: r.metrics.total_return_pct)
    worst_return = min(rankable, key=lambda r: r.metrics.total_return_pct)
    best_pf = max(rankable, key=lambda r: r.metrics.profit_factor)
    worst_pf = min(rankable, key=lambda r: r.metrics.profit_factor)
    best_exposure_adj = max(rankable, key=_exposure_adjusted_return)
    worst_exposure_adj = min(rankable, key=_exposure_adjusted_return)
    return (
        "Symbol rankings\n"
        f"  Return best={best_return.label} {best_return.metrics.total_return_pct:+.2f}%  "
        f"worst={worst_return.label} {worst_return.metrics.total_return_pct:+.2f}%\n"
        f"  PF best={best_pf.label} {best_pf.metrics.profit_factor:.2f}  "
        f"worst={worst_pf.label} {worst_pf.metrics.profit_factor:.2f}\n"
        f"  ExposureAdj best={best_exposure_adj.label} "
        f"{_exposure_adjusted_return(best_exposure_adj):+.2f}  "
        f"worst={worst_exposure_adj.label} "
        f"{_exposure_adjusted_return(worst_exposure_adj):+.2f}"
        f"{skip_note}"
    )


def format_comparability_warnings(reports: Sequence[Report]) -> str:
    if len(reports) < 2:
        return ""
    ranges = {
        (
            r.diagnostics.data_start if r.diagnostics is not None else None,
            r.diagnostics.data_end if r.diagnostics is not None else None,
        )
        for r in reports
    }
    dd_stops = {
        r.diagnostics.drawdown_stop_pct if r.diagnostics is not None else None for r in reports
    }
    metadata = {r.metadata for r in reports}
    warnings = []
    if len(ranges) > 1:
        warnings.append("data ranges differ")
    if len(dd_stops) > 1:
        warnings.append("drawdown stop settings differ")
    if len(metadata) > 1:
        warnings.append("run metadata differs")
    return "Comparability warning: " + "; ".join(warnings) if warnings else ""


def format_benchmark_report(
    *,
    mode: str,
    benchmark_results: Sequence[tuple[str, Sequence[Report]]],
    diagnostics: bool = False,
) -> str:
    rows = []
    for name, reports in benchmark_results:
        if not reports:
            continue
        total_trades = sum(r.metrics.num_trades for r in reports)
        avg_return = _mean([r.metrics.total_return_pct for r in reports])
        avg_pf = _mean_finite([r.metrics.profit_factor for r in reports])
        avg_expectancy = _mean([r.trade_expectancy_pct for r in reports])
        avg_active = _mean([r.active_bar_pct for r in reports])
        avg_notional = _mean(
            [
                r.diagnostics.avg_notional_exposure_pct if r.diagnostics is not None else 0.0
                for r in reports
            ]
        )
        max_notional = max(
            (r.diagnostics.max_notional_exposure_pct if r.diagnostics is not None else 0.0)
            for r in reports
        )
        rows.append(
            [
                name,
                str(len(reports)),
                str(total_trades),
                f"{avg_return:+.2f}",
                f"{avg_pf:.2f}",
                f"{avg_expectancy:+.2f}",
                f"{avg_active:.1f}",
                f"{avg_notional:.1f}",
                f"{max_notional:.1f}",
            ]
        )

    lines = [
        f"Mode: {mode}",
        "Benchmark: strategy variants",
        format_table(
            [
                "Variant",
                "Pairs",
                "Trades",
                "AvgRet%",
                "AvgPF",
                "Exp%",
                "ActBar%",
                "AvgNot%",
                "MaxNot%",
            ],
            rows,
        ),
    ]
    for name, reports in benchmark_results:
        if not reports:
            continue
        lines.append(f"\n{name}")
        warning = format_comparability_warnings(reports)
        if warning:
            lines.append(warning)
        lines.append(compare(*reports))
        lines.append(format_symbol_rankings(reports))
        if diagnostics:
            for report in reports:
                if report.diagnostics is None:
                    continue
                lines.append(f"\n{report.label}")
                lines.append(report.diagnostics_table())
    return "\n".join(lines)


def format_backtest_report(
    *,
    mode: str,
    strategy: str,
    reports: Sequence[Report],
    detail: bool = True,
    diagnostics: bool = False,
    signal_diagnostics: Sequence[tuple[str, dict[str, float]]] = (),
) -> str:
    lines = [
        f"Mode: {mode}",
        f"Strategy: {strategy}",
    ]
    warning = format_comparability_warnings(reports)
    if warning:
        lines.append(warning)
    lines.extend([compare(*reports), "", format_strategy_summary(reports)])
    lines.extend(["", format_strategy_recommendations(strategy, reports)])
    if detail:
        for report in reports:
            lines.append(f"\n{report.label}")
            lines.append(report.metric_sections())
            if diagnostics and report.diagnostics is not None:
                lines.append(report.diagnostics_table())
    elif diagnostics:
        lines.append("\nDiagnostics")
        for report in reports:
            if report.diagnostics is None:
                continue
            lines.append(f"\n{report.label}")
            lines.append(report.diagnostics_table())
    if diagnostics and signal_diagnostics:
        lines.append("\nSignal filter diagnostics")
        for label, values in signal_diagnostics:
            lines.append(format_signal_diagnostics(label, values))
    lines.append("")
    lines.append(format_symbol_rankings(reports))
    return "\n".join(lines)


def format_signal_diagnostics(label: str, diagnostics: dict[str, float]) -> str:
    parts = [f"{key}={value:.1f}" for key, value in sorted(diagnostics.items())]
    return f"  {label}: " + "  ".join(parts)

