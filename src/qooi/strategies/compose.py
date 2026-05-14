"""Apply composable strategy specs to market data."""

from __future__ import annotations

import polars as pl

from qooi.strategies.indicators import add_indicators
from qooi.strategies.specs import StrategySpec, resolve_spec


def compute_signal_frame(df: pl.DataFrame, spec_or_name: StrategySpec | str) -> pl.DataFrame:
    """Return ``df`` with a composed strategy ``signal`` column."""
    spec = resolve_spec(spec_or_name) if isinstance(spec_or_name, str) else spec_or_name
    if spec.name == "flow_pipeline":
        from qooi.core.indicators import compute_dataframe

        return compute_dataframe(df, threshold=0.25)
    df = add_indicators(df)
    return apply_strategy_spec(df, spec)


def latest_signal(df: pl.DataFrame, spec_or_name: StrategySpec | str) -> float:
    signal_df = compute_signal_frame(df, spec_or_name)
    if signal_df.is_empty() or "signal" not in signal_df.columns:
        return 0.0
    value = signal_df["signal"][-1]
    return float(value or 0.0)


def apply_strategy_spec(df: pl.DataFrame, spec: StrategySpec) -> pl.DataFrame:
    if df.is_empty():
        return df.with_columns(pl.lit(0.0).alias("signal"))
    missing = set(spec.required_columns) - set(df.columns)
    if missing:
        return df.with_columns(pl.lit(0.0).alias("signal"))

    work = df
    for feature in spec.features:
        work = feature(work)

    if not spec.entries:
        return work.with_columns(pl.lit(0.0).alias("signal"))

    filter_expr = pl.lit(True)
    for expr in spec.filters:
        filter_expr = filter_expr & expr.fill_null(False)

    entry_expr = pl.lit(0.0)
    for rule in reversed(spec.entries):
        entry_expr = pl.when((filter_expr & rule.condition).fill_null(False)).then(
            float(rule.direction)
        ).otherwise(entry_expr)

    work = work.with_columns(entry_expr.alias("entry_signal"))
    exit_expr = (
        spec.hold.exit_when.fill_null(False)
        if spec.hold.exit_when is not None
        else pl.lit(False)
    )
    work = work.with_columns(exit_expr.alias("exit_signal"))
    signal = _hold_signal(work["entry_signal"].to_list(), work["exit_signal"].to_list(), spec)
    return work.with_columns(pl.Series("signal", signal, dtype=pl.Float64)).drop(
        "entry_signal", "exit_signal"
    )


def _hold_signal(entries: list[float], exits: list[bool], spec: StrategySpec) -> list[float]:
    signal: list[float] = []
    pos = 0.0
    bars_held = 0
    max_bars = spec.hold.max_bars
    for entry, exit_now in zip(entries, exits, strict=False):
        entry_val = float(entry or 0.0)
        if pos != 0.0:
            bars_held += 1
            if exit_now or (max_bars is not None and bars_held >= max_bars):
                pos = 0.0
                bars_held = 0
            elif entry_val != 0.0 and entry_val != pos:
                pos = entry_val
                bars_held = 0
        elif entry_val != 0.0:
            pos = entry_val
            bars_held = 0
        signal.append(pos)
    return signal
