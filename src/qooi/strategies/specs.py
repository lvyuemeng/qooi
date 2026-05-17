"""Composable strategy specifications and execution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import polars as pl

import qooi.strategies.conditions as c
from qooi.strategies.features import (
    FeatureFn,
    add_liquidity_sweep_features,
    add_macd_histogram,
    add_momentum_return,
    add_none_context_diagnostics,
    add_price_structure,
    add_price_structure_stage_features,
    add_trend_maturity,
    add_utc_hour,
    add_volume_average,
)
from qooi.strategies.indicators import add_indicators, compute_flow_pipeline_frame
from qooi.strategies.semantics import LiquidityEvent, StructureState

Direction = Literal[-1, 1]


@dataclass(frozen=True)
class SignalRule:
    name: str
    direction: Direction
    condition: pl.Expr


@dataclass(frozen=True)
class HoldPolicy:
    exit_when: pl.Expr | None = None
    exit_long_when: pl.Expr | None = None
    exit_short_when: pl.Expr | None = None
    max_bars: int | None = None


@dataclass(frozen=True)
class StrategySpec:
    name: str
    required_columns: tuple[str, ...]
    features: tuple[FeatureFn, ...]
    entries: tuple[SignalRule, ...]
    filters: tuple[pl.Expr, ...] = ()
    hold: HoldPolicy = HoldPolicy()
    continuous_entries: bool = False


@dataclass(frozen=True)
class FlowPipelineSpec:
    name: str = "flow_pipeline"
    threshold: float = 0.25

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return compute_flow_pipeline_frame(df, threshold=self.threshold)


StrategyBehavior = StrategySpec | FlowPipelineSpec


def momentum_burst_spec(
    *,
    mom_bars: int = 6,
    mom_threshold: float = 0.003,
    ema_fast: int = 20,
    ema_mid: int = 50,
    ema_slow: int = 200,
    trend_maturity: int = 12,
    volume_mult: float = 1.1,
    adx_threshold: float = 15.0,
    include_session_filter: bool = True,
    include_volume_filter: bool = True,
    max_bars: int | None = None,
    name: str = "momentum_burst",
) -> StrategySpec:
    filters = [c.adx_above(adx_threshold)]
    if include_session_filter:
        filters.append(c.session_between(8, 22))
    filters.append(c.trend_mature(trend_maturity))
    if include_volume_filter:
        filters.append(c.volume_spike(volume_mult))
    return StrategySpec(
        name=name,
        required_columns=(
            "timestamp",
            "close",
            "high",
            "low",
            "vol",
            "atr_14",
            "adx_14",
            f"ema_{ema_fast}",
            f"ema_{ema_mid}",
            f"ema_{ema_slow}",
        ),
        features=(
            add_momentum_return(mom_bars),
            add_volume_average(20),
            add_trend_maturity(ema_mid=ema_mid, ema_slow=ema_slow),
            add_utc_hour(),
            add_price_structure(5, 20),
        ),
        entries=(
            SignalRule(
                "long_momentum_burst",
                1,
                c.uptrend(ema_mid, ema_slow)
                & c.momentum_gt(mom_threshold)
                & c.above_ema(ema_fast)
                & c.higher_low_structure(),
            ),
            SignalRule(
                "short_momentum_burst",
                -1,
                c.downtrend(ema_mid, ema_slow)
                & c.momentum_lt(-mom_threshold)
                & c.below_ema(ema_fast)
                & c.lower_high_structure(),
            ),
        ),
        filters=tuple(filters),
        hold=HoldPolicy(
            exit_long_when=~c.uptrend(ema_mid, ema_slow),
            exit_short_when=~c.downtrend(ema_mid, ema_slow),
            max_bars=max_bars,
        ),
    )


def ema_trend_baseline_spec(
    *,
    ema_mid: int = 50,
    ema_slow: int = 200,
    name: str = "ema_trend_baseline",
) -> StrategySpec:
    """Always-in EMA direction baseline without session, volume, or structure gates."""
    return StrategySpec(
        name=name,
        required_columns=(
            "timestamp",
            "close",
            "high",
            "low",
            "atr_14",
            "adx_14",
            f"ema_{ema_mid}",
            f"ema_{ema_slow}",
        ),
        features=(),
        entries=(
            SignalRule("long_ema_trend", 1, c.uptrend(ema_mid, ema_slow)),
            SignalRule("short_ema_trend", -1, c.downtrend(ema_mid, ema_slow)),
        ),
        filters=(),
        continuous_entries=True,
        hold=HoldPolicy(exit_when=~(c.uptrend(ema_mid, ema_slow) | c.downtrend(ema_mid, ema_slow))),
    )


def rsi_bounce_reversion_spec(
    *,
    rsi_period: int = 14,
    rsi_oversold: float = 30.0,
    rsi_bounce: float = 25.0,
    rsi_confirmation: float = 20.0,
    rsi_exit: float = 50.0,
    ema_mid: int = 50,
    ema_slow: int = 200,
) -> StrategySpec:
    return StrategySpec(
        name="rsi_bounce_reversion",
        required_columns=(
            "timestamp",
            "close",
            "high",
            "low",
            "atr_14",
            "adx_14",
            f"ema_{ema_mid}",
            f"ema_{ema_slow}",
            f"rsi_{rsi_period}",
        ),
        features=(
            add_trend_maturity(ema_mid=ema_mid, ema_slow=ema_slow),
            add_utc_hour(),
            add_price_structure(5, 20),
        ),
        entries=(
            SignalRule(
                "long_rsi_bounce",
                1,
                c.uptrend(ema_mid, ema_slow)
                & c.rsi_cross_from_oversold(
                    rsi_period=rsi_period, oversold=rsi_oversold, bounce=rsi_bounce
                )
                & c.rsi_bounce_held(rsi_period=rsi_period, confirmation=rsi_confirmation)
                & c.higher_low_structure(),
            ),
        ),
        filters=(c.adx_above(20.0), c.session_between(8, 22)),
        hold=HoldPolicy(
            exit_when=(~c.uptrend(ema_mid, ema_slow)) | c.rsi_above(threshold=rsi_exit)
        ),
    )


def rsi_macd_trend_spec(
    *,
    rsi_period: int = 14,
    long_rsi: float = 55.0,
    short_rsi: float = 45.0,
    ema_mid: int = 50,
    ema_slow: int = 200,
) -> StrategySpec:
    return StrategySpec(
        name="rsi_macd_trend",
        required_columns=(
            "timestamp",
            "close",
            "high",
            "low",
            "atr_14",
            "adx_14",
            "ema_12",
            "ema_26",
            f"ema_{ema_mid}",
            f"ema_{ema_slow}",
            f"rsi_{rsi_period}",
        ),
        features=(add_macd_histogram(),),
        entries=(
            SignalRule(
                "long_rsi_macd_trend",
                1,
                c.uptrend(ema_mid, ema_slow)
                & c.rsi_above(rsi_period=rsi_period, threshold=long_rsi)
                & c.macd_hist_above(),
            ),
            SignalRule(
                "short_rsi_macd_trend",
                -1,
                c.downtrend(ema_mid, ema_slow)
                & c.rsi_below(rsi_period=rsi_period, threshold=short_rsi)
                & c.macd_hist_below(),
            ),
        ),
        filters=(c.adx_above(15.0),),
        hold=HoldPolicy(
            exit_long_when=(~c.uptrend(ema_mid, ema_slow)) | c.macd_hist_below(),
            exit_short_when=(~c.downtrend(ema_mid, ema_slow)) | c.macd_hist_above(),
        ),
    )


def flow_pipeline_spec(*, threshold: float = 0.25) -> FlowPipelineSpec:
    return FlowPipelineSpec(threshold=threshold)


def _failed_breakout_entry_conditions(
    *,
    event_quality_min: float,
    require_volume_impulse: bool,
) -> tuple[pl.Expr, pl.Expr]:
    volume_gate = (
        pl.col("volume_impulse").fill_null(False)
        if require_volume_impulse
        else pl.lit(True)
    )
    quality_gate = pl.col("event_quality_score").cast(pl.Float64) >= event_quality_min
    long_entry = (
        (pl.col("liquidity_event_type") == LiquidityEvent.FAILED_BREAKOUT_LOW)
        & pl.col("failed_breakout_low").fill_null(False)
        & pl.col("prior_liquidity_low").is_not_null()
        & quality_gate
        & volume_gate
    )
    short_entry = (
        (pl.col("liquidity_event_type") == LiquidityEvent.FAILED_BREAKOUT_HIGH)
        & pl.col("failed_breakout_high").fill_null(False)
        & pl.col("prior_liquidity_high").is_not_null()
        & quality_gate
        & volume_gate
    )
    return long_entry, short_entry


def _structural_feature_stack() -> tuple[FeatureFn, ...]:
    return (
        add_liquidity_sweep_features(),
        add_none_context_diagnostics(),
        add_price_structure_stage_features(),
    )


def structure_event_reversal_v1_spec(
    *,
    event_quality_min: float = 1.5,
    require_volume_impulse: bool = True,
    include_reclaim_sweeps: bool = False,
    max_bars: int = 8,
    name: str = "structure_event_reversal_v1",
) -> StrategySpec:
    """Failed-breakout reversal strategy for falsifying structural-event edge."""
    if include_reclaim_sweeps:
        raise ValueError(
            "include_reclaim_sweeps is not implemented for structure_event_reversal_v1"
        )

    long_entry, short_entry = _failed_breakout_entry_conditions(
        event_quality_min=event_quality_min,
        require_volume_impulse=require_volume_impulse,
    )
    return StrategySpec(
        name=name,
        required_columns=("timestamp", "open", "high", "low", "close", "atr_14"),
        features=_structural_feature_stack(),
        entries=(
            SignalRule("long_failed_breakout_low", 1, long_entry),
            SignalRule("short_failed_breakout_high", -1, short_entry),
        ),
        hold=HoldPolicy(
            exit_long_when=pl.col("breakout_acceptance_low").fill_null(False),
            exit_short_when=pl.col("breakout_acceptance_high").fill_null(False),
            max_bars=max_bars,
        ),
    )


def structure_event_trend_aligned_v1_spec(
    *,
    event_quality_min: float = 1.5,
    require_volume_impulse: bool = True,
    include_reclaim_sweeps: bool = False,
    max_bars: int = 8,
    name: str = "structure_event_trend_aligned_v1",
) -> StrategySpec:
    """Trend-aligned failed-breakout pullback strategy."""
    if include_reclaim_sweeps:
        raise ValueError(
            "include_reclaim_sweeps is not implemented for structure_event_trend_aligned_v1"
        )

    long_base, short_base = _failed_breakout_entry_conditions(
        event_quality_min=event_quality_min,
        require_volume_impulse=require_volume_impulse,
    )
    long_entry = long_base & (pl.col("structure_trend_state") == StructureState.UPTREND)
    short_entry = short_base & (pl.col("structure_trend_state") == StructureState.DOWNTREND)
    return StrategySpec(
        name=name,
        required_columns=("timestamp", "open", "high", "low", "close", "atr_14"),
        features=_structural_feature_stack(),
        entries=(
            SignalRule("long_trend_aligned_failed_breakout_low", 1, long_entry),
            SignalRule("short_trend_aligned_failed_breakout_high", -1, short_entry),
        ),
        hold=HoldPolicy(
            exit_long_when=pl.col("breakout_acceptance_low").fill_null(False),
            exit_short_when=pl.col("breakout_acceptance_high").fill_null(False),
            max_bars=max_bars,
        ),
    )


def structure_event_trend_aligned_mtf_confirm_v1_spec(
    *,
    event_quality_min: float = 1.5,
    require_volume_impulse: bool = True,
    include_reclaim_sweeps: bool = False,
    max_bars: int = 8,
    name: str = "structure_event_trend_aligned_mtf_confirm_v1",
) -> StrategySpec:
    """Trend-aligned structural events gated by immediate M15 confirmation."""
    if include_reclaim_sweeps:
        raise ValueError(
            "include_reclaim_sweeps is not implemented for "
            "structure_event_trend_aligned_mtf_confirm_v1"
        )

    long_base, short_base = _failed_breakout_entry_conditions(
        event_quality_min=event_quality_min,
        require_volume_impulse=require_volume_impulse,
    )
    long_entry = (
        long_base
        & (pl.col("structure_trend_state") == StructureState.UPTREND)
        & pl.col("m15_confirm_long").fill_null(False)
    )
    short_entry = (
        short_base
        & (pl.col("structure_trend_state") == StructureState.DOWNTREND)
        & pl.col("m15_confirm_short").fill_null(False)
    )
    return StrategySpec(
        name=name,
        required_columns=(
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "atr_14",
            "m15_confirm_long",
            "m15_confirm_short",
            "m15_confirm_available",
        ),
        features=_structural_feature_stack(),
        entries=(
            SignalRule("long_trend_aligned_mtf_confirm_failed_breakout_low", 1, long_entry),
            SignalRule("short_trend_aligned_mtf_confirm_failed_breakout_high", -1, short_entry),
        ),
        hold=HoldPolicy(
            exit_long_when=pl.col("breakout_acceptance_low").fill_null(False),
            exit_short_when=pl.col("breakout_acceptance_high").fill_null(False),
            max_bars=max_bars,
        ),
    )


def compute_signal_frame(df: pl.DataFrame, strategy: StrategyBehavior) -> pl.DataFrame:
    """Return ``df`` with a behavior-computed ``signal`` column."""
    if not isinstance(strategy, (StrategySpec, FlowPipelineSpec)):
        raise TypeError("strategy must be a strategy behavior object")
    if isinstance(strategy, FlowPipelineSpec):
        return _with_flow_signal_columns(strategy.compute(df))
    df = add_indicators(df)
    return apply_strategy_spec(df, strategy)


def latest_signal(df: pl.DataFrame, strategy: StrategyBehavior) -> float:
    signal_df = compute_signal_frame(df, strategy)
    if signal_df.is_empty() or "signal" not in signal_df.columns:
        return 0.0
    value = signal_df["signal"][-1]
    return float(value or 0.0)


def strategy_signal_diagnostics(df: pl.DataFrame, strategy: StrategyBehavior) -> dict[str, float]:
    """Return signal/filter pass-rate diagnostics for a strategy behavior."""
    signal_columns = {"raw_entry_signal", "entry_signal", "position_signal", "exit_signal"}
    signal_df = df if signal_columns.issubset(df.columns) else compute_signal_frame(df, strategy)
    bars = float(signal_df.height)
    if bars == 0:
        return {"bars": 0.0, "signal_pct": 0.0}

    if isinstance(strategy, FlowPipelineSpec):
        diagnostics: dict[str, float] = {
            "bars": bars,
            "held_signal_pct": _expr_pct(signal_df, pl.col("signal") != 0),
        }
        if "ofi_flow_score" in signal_df.columns:
            diagnostics["ofi_threshold_pct"] = _expr_pct(
                signal_df, pl.col("ofi_flow_score").abs() >= strategy.threshold
            )
        if "regime_score" in signal_df.columns:
            diagnostics["regime_gate_pass_pct"] = _expr_pct(
                signal_df, pl.col("regime_score").abs() <= 0.7
            )
        return diagnostics

    diagnostics = {
        "bars": bars,
        "held_signal_pct": _expr_pct(signal_df, pl.col("position_signal") != 0),
        "long_held_signal_pct": _expr_pct(signal_df, pl.col("position_signal") > 0),
        "short_held_signal_pct": _expr_pct(signal_df, pl.col("position_signal") < 0),
        "raw_entry_any_pct": _expr_pct(signal_df, pl.col("raw_entry_signal") != 0),
        "entry_event_pct": _expr_pct(signal_df, pl.col("entry_signal") != 0),
    }

    work = add_indicators(df)
    for feature in strategy.features:
        work = feature(work)
    for idx, expr in enumerate(strategy.filters):
        diagnostics[f"filter_{idx}_pct"] = _expr_pct(work, expr.fill_null(False))

    filter_expr = pl.lit(True)
    for expr in strategy.filters:
        filter_expr = filter_expr & expr.fill_null(False)
    diagnostics["all_filters_pct"] = _expr_pct(work, filter_expr)

    for rule in strategy.entries:
        diagnostics[f"entry_{rule.name}_pct"] = _expr_pct(
            work, (filter_expr & rule.condition).fill_null(False)
        )
    if "volatility_regime" in work.columns:
        diagnostics["high_volatility_regime_pct"] = _expr_pct(work, pl.col("volatility_regime") > 0)
    if "signal_strength" in signal_df.columns:
        strength = signal_df.filter(pl.col("entry_signal") != 0)["signal_strength"].drop_nulls()
        if not strength.is_empty():
            strength_mean = cast(float | None, strength.mean())
            strength_median = cast(float | None, strength.median())
            diagnostics["signal_strength_avg"] = float(strength_mean or 0.0)
            diagnostics["signal_strength_median"] = float(strength_median or 0.0)
            diagnostics["entry_strength_low_pct"] = _expr_pct(
                signal_df, (pl.col("entry_signal") != 0) & (pl.col("signal_strength") < 0.34)
            )
            diagnostics["entry_strength_mid_pct"] = _expr_pct(
                signal_df,
                (pl.col("entry_signal") != 0)
                & pl.col("signal_strength").is_between(0.34, 0.67, closed="left"),
            )
            diagnostics["entry_strength_high_pct"] = _expr_pct(
                signal_df, (pl.col("entry_signal") != 0) & (pl.col("signal_strength") >= 0.67)
            )
    if "signal_id" in signal_df.columns:
        ids = (
            signal_df.filter(pl.col("entry_signal") != 0)["signal_id"]
            .drop_nulls()
            .unique()
            .to_list()
        )
        for signal_id in ids:
            if signal_id:
                key = str(signal_id).replace(" ", "_")
                diagnostics[f"entry_count_{key}"] = float(
                    signal_df.filter(
                        (pl.col("entry_signal") != 0) & (pl.col("signal_id") == signal_id)
                    ).height
                )
    return diagnostics


def apply_strategy_spec(df: pl.DataFrame, spec: StrategySpec) -> pl.DataFrame:
    if df.is_empty():
        return _with_empty_signals(df)
    missing = set(spec.required_columns) - set(df.columns)
    if missing:
        return _with_empty_signals(df)

    work = df
    for feature in spec.features:
        work = feature(work)

    if not spec.entries:
        return _with_empty_signals(work)

    filter_expr = pl.lit(True)
    for expr in spec.filters:
        filter_expr = filter_expr & expr.fill_null(False)

    entry_expr = pl.lit(0.0)
    signal_id_expr = pl.lit("")
    for rule in reversed(spec.entries):
        match_expr = (filter_expr & rule.condition).fill_null(False)
        entry_expr = pl.when(match_expr).then(float(rule.direction)).otherwise(entry_expr)
        signal_id_expr = pl.when(match_expr).then(pl.lit(rule.name)).otherwise(signal_id_expr)

    work = work.with_columns(
        entry_expr.alias("raw_entry_signal"),
        signal_id_expr.alias("signal_id"),
    )
    exit_any_expr = _exit_expr(spec.hold.exit_when)
    exit_long_expr = _exit_expr(spec.hold.exit_long_when, fallback=exit_any_expr)
    exit_short_expr = _exit_expr(spec.hold.exit_short_when, fallback=exit_any_expr)
    work = work.with_columns(
        exit_long_expr.alias("_exit_long_signal"),
        exit_short_expr.alias("_exit_short_signal"),
    )
    position, exit_events = _hold_signal(
        work["raw_entry_signal"].to_list(),
        work["_exit_long_signal"].to_list(),
        work["_exit_short_signal"].to_list(),
        spec,
    )
    entry_events = _entry_events(work["raw_entry_signal"].to_list(), position, spec)
    strength_expr = pl.when(pl.col("raw_entry_signal") != 0).then(1.0).otherwise(0.0)
    return (
        work.with_columns(
            pl.Series("entry_signal", entry_events, dtype=pl.Float64),
            pl.Series("position_signal", position, dtype=pl.Float64),
            pl.Series("exit_signal", exit_events, dtype=pl.Boolean),
            strength_expr.alias("signal_strength"),
            pl.Series("signal", position, dtype=pl.Float64),
        )
        .drop("_exit_long_signal", "_exit_short_signal")
    )


def _exit_expr(expr: pl.Expr | None, *, fallback: pl.Expr | None = None) -> pl.Expr:
    if expr is not None:
        return expr.fill_null(False)
    if fallback is not None:
        return fallback.fill_null(False)
    return pl.lit(False)


def _expr_pct(df: pl.DataFrame, expr: pl.Expr) -> float:
    if df.is_empty():
        return 0.0
    value = df.select(expr.cast(pl.Int64).sum()).item()
    return float(value or 0.0) / df.height * 100.0


def _with_empty_signals(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.lit(0.0).alias("raw_entry_signal"),
        pl.lit(0.0).alias("entry_signal"),
        pl.lit(0.0).alias("position_signal"),
        pl.lit(False).alias("exit_signal"),
        pl.lit(0.0).alias("signal_strength"),
        pl.lit("").alias("signal_id"),
        pl.lit(0.0).alias("signal"),
    )


def _with_flow_signal_columns(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty() or "signal" not in df.columns:
        return _with_empty_signals(df)
    signed_signal = pl.col("signal").sign().cast(pl.Float64)
    raw_entries = [float(value or 0.0) for value in df["signal"].sign().to_list()]
    entry_events = _entry_events(raw_entries, raw_entries, StrategySpec("flow", (), (), ()))
    return df.with_columns(
        signed_signal.alias("raw_entry_signal"),
        pl.Series("entry_signal", entry_events, dtype=pl.Float64),
        signed_signal.alias("position_signal"),
        pl.lit(False).alias("exit_signal"),
        pl.col("signal").abs().clip(0.0, 1.0).alias("signal_strength"),
        pl.when(pl.col("signal") != 0)
        .then(pl.lit("flow_pipeline"))
        .otherwise(pl.lit(""))
        .alias("signal_id"),
    )


def _entry_events(
    raw_entries: list[float], position: list[float], spec: StrategySpec
) -> list[float]:
    if spec.continuous_entries:
        return [float(v or 0.0) for v in raw_entries]
    events: list[float] = []
    prev_pos = 0.0
    for raw, pos in zip(raw_entries, position, strict=False):
        raw_val = float(raw or 0.0)
        pos_val = float(pos or 0.0)
        if raw_val != 0.0 and (prev_pos == 0.0 or raw_val != prev_pos):
            events.append(raw_val)
        else:
            events.append(0.0)
        prev_pos = pos_val
    return events


def _hold_signal(
    entries: list[float],
    long_exits: list[bool],
    short_exits: list[bool],
    spec: StrategySpec,
) -> tuple[list[float], list[bool]]:
    signal: list[float] = []
    exit_events: list[bool] = []
    pos = 0.0
    bars_held = 0
    max_bars = spec.hold.max_bars
    for entry, long_exit, short_exit in zip(entries, long_exits, short_exits, strict=False):
        entry_val = float(entry or 0.0)
        exit_now = (pos > 0.0 and bool(long_exit)) or (pos < 0.0 and bool(short_exit))
        did_exit = False
        if pos != 0.0:
            bars_held += 1
            if exit_now or (max_bars is not None and bars_held >= max_bars):
                pos = 0.0
                bars_held = 0
                did_exit = True
            elif entry_val != 0.0 and entry_val != pos:
                pos = entry_val
                bars_held = 0
        elif entry_val != 0.0:
            pos = entry_val
            bars_held = 0
        exit_events.append(did_exit)
        signal.append(pos)
    return signal, exit_events
