"""Rule primitive diagnostics derived from behavior-state taxonomy."""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass

import polars as pl

from qooi.research.artifacts import empty_frame, ensure_columns
from qooi.research.behavior_tables import (
    STATE_TRANSITION_CHAIN_SCHEMA,
    build_state_transition_chains,
)

RULE_PRIMITIVE_SIGNAL_SCHEMA: dict[str, pl.DataType] = {
    "context_kind": pl.Utf8,
    "context_value": pl.Utf8,
    "taxonomy_label": pl.Utf8,
    "rule_name": pl.Utf8,
    "symbol": pl.Utf8,
    "timeframe": pl.Utf8,
    "timestamp": pl.Int64,
    "state_column": pl.Utf8,
    "horizon": pl.Int64,
    "side": pl.Utf8,
    "signal_close": pl.Float64,
    "ema_fast": pl.Float64,
    "ema_slow": pl.Float64,
    "prior_breakout_high": pl.Float64,
    "prior_breakout_low": pl.Float64,
    "split": pl.Utf8,
}

RULE_PRIMITIVE_TRADE_SCHEMA: dict[str, pl.DataType] = {
    "context_kind": pl.Utf8,
    "context_value": pl.Utf8,
    "taxonomy_label": pl.Utf8,
    "rule_name": pl.Utf8,
    "symbol": pl.Utf8,
    "timeframe": pl.Utf8,
    "horizon": pl.Int64,
    "side": pl.Utf8,
    "signal_timestamp": pl.Int64,
    "entry_timestamp": pl.Int64,
    "exit_timestamp": pl.Int64,
    "entry_close": pl.Float64,
    "exit_close": pl.Float64,
    "market_return_pct": pl.Float64,
    "side_return_pct": pl.Float64,
    "transaction_cost_bps": pl.Float64,
    "nonoverlap": pl.Boolean,
    "split": pl.Utf8,
}

RULE_PRIMITIVE_SUMMARY_SCHEMA: dict[str, pl.DataType] = {
    "context_kind": pl.Utf8,
    "context_value": pl.Utf8,
    "taxonomy_label": pl.Utf8,
    "rule_name": pl.Utf8,
    "symbol": pl.Utf8,
    "horizon": pl.Int64,
    "trades": pl.Int64,
    "mean_return_pct": pl.Float64,
    "median_return_pct": pl.Float64,
    "win_rate_pct": pl.Float64,
    "trade_sharpe": pl.Float64,
    "max_drawdown_pct": pl.Float64,
    "matched_buy_hold_mean_pct": pl.Float64,
    "mean_excess_return_pct": pl.Float64,
}


@dataclass(frozen=True)
class RulePrimitiveConfig:
    horizons: tuple[int, ...] = (1,)
    ema_fast: int = 9
    ema_slow: int = 21
    compression_lookback: int = 20
    breakout_lookback: int = 20


def build_rule_primitive_signals(
    market: pl.DataFrame,
    taxonomy: pl.DataFrame,
    state_column: str,
    *,
    config: RulePrimitiveConfig | None = None,
) -> pl.DataFrame:
    if market.is_empty() or taxonomy.is_empty() or state_column not in market.columns:
        return _empty_frame(RULE_PRIMITIVE_SIGNAL_SCHEMA)
    cfg = config or RulePrimitiveConfig()
    prepared = _indicator_market(market, cfg)
    chain_lengths = tuple(
        sorted(
            {
                int(v)
                for v in taxonomy.get_column("ngram_length").drop_nulls().to_list()
                if int(v) > 1
            }
        )
    )
    chains = (
        build_state_transition_chains(prepared, state_column, chain_lengths)
        if chain_lengths
        else _empty_frame(STATE_TRANSITION_CHAIN_SCHEMA)
    )
    rows = []
    for tax in taxonomy.to_dicts():
        label = str(tax.get("taxonomy_label"))
        if label in {"avoid", "wide_chop", "rare_transition", "informative_transition"}:
            continue
        horizon = int(tax.get("horizon") or (cfg.horizons[0] if cfg.horizons else 1))
        kind = str(tax.get("context_kind") or "state")
        context_value = str(tax.get("context_value"))
        symbol = tax.get("symbol")
        timeframe = tax.get("timeframe")
        candidates = _context_signal_rows(
            prepared, chains, state_column, kind, context_value, symbol, timeframe
        )
        for row in candidates:
            if (
                label == "trend_smooth"
                and row.get("ema_fast") is not None
                and row.get("ema_slow") is not None
                and float(row["close"]) > float(row["ema_slow"])
                and float(row["ema_fast"]) > float(row["ema_slow"])
            ):
                rule_name = "ema_trend_follow"
            elif (
                label in {"narrow_compression", "breakout_prone"}
                and row.get("prior_breakout_high") is not None
                and float(row["close"]) > float(row["prior_breakout_high"])
            ):
                rule_name = "compression_breakout"
            else:
                continue
            rows.append(
                {
                    "context_kind": kind,
                    "context_value": context_value,
                    "taxonomy_label": label,
                    "rule_name": rule_name,
                    "symbol": row.get("symbol"),
                    "timeframe": row.get("timeframe"),
                    "timestamp": row.get("timestamp"),
                    "state_column": state_column,
                    "horizon": horizon,
                    "side": "long",
                    "signal_close": row.get("close"),
                    "ema_fast": row.get("ema_fast"),
                    "ema_slow": row.get("ema_slow"),
                    "prior_breakout_high": row.get("prior_breakout_high"),
                    "prior_breakout_low": row.get("prior_breakout_low"),
                    "split": row.get("split"),
                }
            )
    return _ensure_columns(pl.DataFrame(rows), RULE_PRIMITIVE_SIGNAL_SCHEMA)


def build_rule_primitive_trades(
    signals: pl.DataFrame, market: pl.DataFrame, *, transaction_cost_bps: float
) -> pl.DataFrame:
    if signals.is_empty() or market.is_empty():
        return _empty_frame(RULE_PRIMITIVE_TRADE_SCHEMA)
    market_lookup = _market_lookup(market)
    accepted: dict[tuple[str, str, str, str, str, int, str], int] = {}
    cost_pct = transaction_cost_bps / 100.0
    rows = []
    for signal in signals.sort(
        ["symbol", "context_kind", "context_value", "rule_name", "timestamp"]
    ).to_dicts():
        horizon = int(signal.get("horizon") or 0)
        if horizon <= 0:
            continue
        symbol = str(signal.get("symbol"))
        timeframe = str(signal.get("timeframe"))
        market_rows = market_lookup.get((symbol, timeframe)) or market_lookup.get(
            (symbol, "unknown")
        )
        if market_rows is None:
            continue
        entry_index = bisect_right(market_rows["timestamps"], int(signal["timestamp"]))
        exit_index = entry_index + horizon
        if exit_index >= len(market_rows["timestamps"]):
            continue
        entry_timestamp = market_rows["timestamps"][entry_index]
        exit_timestamp = market_rows["timestamps"][exit_index]
        group_key = (
            symbol,
            timeframe,
            str(signal.get("context_kind")),
            str(signal.get("context_value")),
            str(signal.get("rule_name")),
            horizon,
            str(signal.get("side")),
        )
        if entry_timestamp <= accepted.get(group_key, -1):
            continue
        entry_close = market_rows["closes"][entry_index]
        exit_close = market_rows["closes"][exit_index]
        if entry_close in (None, 0) or exit_close is None:
            continue
        market_return = (float(exit_close) - float(entry_close)) / float(entry_close) * 100.0
        side_return = -market_return if signal.get("side") == "short" else market_return
        accepted[group_key] = exit_timestamp
        rows.append(
            {
                **{
                    key: signal.get(key)
                    for key in (
                        "context_kind",
                        "context_value",
                        "taxonomy_label",
                        "rule_name",
                        "symbol",
                        "timeframe",
                        "horizon",
                        "side",
                        "split",
                    )
                },
                "signal_timestamp": int(signal["timestamp"]),
                "entry_timestamp": entry_timestamp,
                "exit_timestamp": exit_timestamp,
                "entry_close": float(entry_close),
                "exit_close": float(exit_close),
                "market_return_pct": market_return,
                "side_return_pct": side_return - cost_pct,
                "transaction_cost_bps": transaction_cost_bps,
                "nonoverlap": True,
            }
        )
    return _ensure_columns(pl.DataFrame(rows), RULE_PRIMITIVE_TRADE_SCHEMA)


def summarize_rule_primitives(trades: pl.DataFrame) -> pl.DataFrame:
    if trades.is_empty():
        return _empty_frame(RULE_PRIMITIVE_SUMMARY_SCHEMA)
    rows = []
    for key, group in trades.partition_by(
        ["context_kind", "context_value", "taxonomy_label", "rule_name", "symbol", "horizon"],
        as_dict=True,
    ).items():
        context_kind, context_value, taxonomy_label, rule_name, symbol, horizon = (
            key if isinstance(key, tuple) else (key, None, None, None, None, None)
        )
        returns = [float(value) for value in group.get_column("side_return_pct").drop_nulls()]
        market_returns = [
            float(value) for value in group.get_column("market_return_pct").drop_nulls()
        ]
        rows.append(
            {
                "context_kind": context_kind,
                "context_value": context_value,
                "taxonomy_label": taxonomy_label,
                "rule_name": rule_name,
                "symbol": symbol,
                "horizon": horizon,
                "trades": len(returns),
                "mean_return_pct": _mean(returns),
                "median_return_pct": _median(returns),
                "win_rate_pct": _win_rate(returns),
                "trade_sharpe": _sharpe_like(returns),
                "max_drawdown_pct": _max_drawdown(returns),
                "matched_buy_hold_mean_pct": _mean(market_returns),
                "mean_excess_return_pct": _mean(
                    [r - m for r, m in zip(returns, market_returns, strict=False)]
                ),
            }
        )
    return _ensure_columns(pl.DataFrame(rows), RULE_PRIMITIVE_SUMMARY_SCHEMA)


def build_rule_primitive_baselines(
    market: pl.DataFrame,
    horizons: tuple[int, ...],
    *,
    transaction_cost_bps: float = 0.0,
    config: RulePrimitiveConfig | None = None,
) -> pl.DataFrame:
    if market.is_empty():
        return _empty_frame(RULE_PRIMITIVE_SUMMARY_SCHEMA)
    cfg = config or RulePrimitiveConfig(horizons=horizons or (1,))
    prepared = _indicator_market(market, cfg)
    rows = []
    for row in prepared.to_dicts():
        for horizon in _positive_unique_ints(horizons):
            if (
                row.get("ema_fast") is not None
                and row.get("ema_slow") is not None
                and row.get("close") is not None
                and float(row["close"]) > float(row["ema_slow"])
                and float(row["ema_fast"]) > float(row["ema_slow"])
            ):
                rows.append(
                    {
                        "context_kind": "baseline",
                        "context_value": "all",
                        "taxonomy_label": "unconditioned",
                        "rule_name": "ema_trend_follow",
                        "symbol": row.get("symbol"),
                        "timeframe": row.get("timeframe"),
                        "timestamp": row.get("timestamp"),
                        "state_column": None,
                        "horizon": horizon,
                        "side": "long",
                        "signal_close": row.get("close"),
                        "ema_fast": row.get("ema_fast"),
                        "ema_slow": row.get("ema_slow"),
                        "prior_breakout_high": row.get("prior_breakout_high"),
                        "prior_breakout_low": row.get("prior_breakout_low"),
                        "split": row.get("split"),
                    }
                )
            if (
                row.get("prior_breakout_high") is not None
                and row.get("close") is not None
                and float(row["close"]) > float(row["prior_breakout_high"])
            ):
                rows.append(
                    {
                        "context_kind": "baseline",
                        "context_value": "all",
                        "taxonomy_label": "unconditioned",
                        "rule_name": "compression_breakout",
                        "symbol": row.get("symbol"),
                        "timeframe": row.get("timeframe"),
                        "timestamp": row.get("timestamp"),
                        "state_column": None,
                        "horizon": horizon,
                        "side": "long",
                        "signal_close": row.get("close"),
                        "ema_fast": row.get("ema_fast"),
                        "ema_slow": row.get("ema_slow"),
                        "prior_breakout_high": row.get("prior_breakout_high"),
                        "prior_breakout_low": row.get("prior_breakout_low"),
                        "split": row.get("split"),
                    }
                )
    signals = _ensure_columns(pl.DataFrame(rows), RULE_PRIMITIVE_SIGNAL_SCHEMA)
    return summarize_rule_primitives(
        build_rule_primitive_trades(signals, market, transaction_cost_bps=transaction_cost_bps)
    )


def _market_lookup(market_frame: pl.DataFrame) -> dict[tuple[str, str], dict[str, list[object]]]:
    if market_frame.is_empty() or not {"timestamp", "close"} <= set(market_frame.columns):
        return {}
    work = _market_with_group_columns(market_frame)
    out: dict[tuple[str, str], dict[str, list[object]]] = {}
    for key, group in (
        work.sort(["symbol", "timeframe", "timestamp"])
        .partition_by(["symbol", "timeframe"], as_dict=True)
        .items()
    ):
        symbol, timeframe = key if isinstance(key, tuple) else (key, "unknown")
        out[(str(symbol), str(timeframe))] = {
            "timestamps": [int(value) for value in group.get_column("timestamp").to_list()],
            "closes": group.get_column("close").to_list(),
        }
    return out


def _market_with_group_columns(market: pl.DataFrame) -> pl.DataFrame:
    work = market
    if "symbol" not in work.columns:
        work = work.with_columns(pl.lit("unknown").alias("symbol"))
    if "timeframe" not in work.columns:
        work = work.with_columns(pl.lit("unknown").alias("timeframe"))
    return work


def _positive_unique_ints(values: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(dict.fromkeys(int(value) for value in values if int(value) > 0))


def _indicator_market(market: pl.DataFrame, config: RulePrimitiveConfig) -> pl.DataFrame:
    work = market
    if "symbol" not in work.columns:
        work = work.with_columns(pl.lit("unknown").alias("symbol"))
    if "timeframe" not in work.columns:
        work = work.with_columns(pl.lit("unknown").alias("timeframe"))
    group = ["symbol", "timeframe"]
    work = work.sort([*group, "timestamp"])
    ema_fast = pl.col("close").ewm_mean(span=config.ema_fast, adjust=False)
    ema_slow = pl.col("close").ewm_mean(span=config.ema_slow, adjust=False)
    prior_high = (
        pl.col("high").rolling_max(window_size=config.breakout_lookback, min_samples=1).shift(1)
    )
    prior_low = (
        pl.col("low").rolling_min(window_size=config.breakout_lookback, min_samples=1).shift(1)
    )
    if group:
        ema_fast = ema_fast.over(group)
        ema_slow = ema_slow.over(group)
        prior_high = prior_high.over(group)
        prior_low = prior_low.over(group)
    return work.with_columns(
        ema_fast.alias("ema_fast"),
        ema_slow.alias("ema_slow"),
        prior_high.alias("prior_breakout_high"),
        prior_low.alias("prior_breakout_low"),
    )


def _context_signal_rows(
    market: pl.DataFrame,
    chains: pl.DataFrame,
    state_column: str,
    kind: str,
    context_value: str,
    symbol: object,
    timeframe: object,
) -> list[dict[str, object]]:
    if kind == "chain":
        if chains.is_empty():
            return []
        selected = chains.filter(pl.col("context_value") == context_value)
        if symbol is not None:
            selected = selected.filter(pl.col("symbol") == symbol)
        if timeframe is not None:
            selected = selected.filter(pl.col("timeframe") == timeframe)
        keys = selected.select("symbol", "timeframe", "timestamp").unique()
        return market.join(keys, on=["symbol", "timeframe", "timestamp"], how="inner").to_dicts()
    selected = market.filter(pl.col(state_column).cast(pl.Utf8) == context_value)
    if symbol is not None:
        selected = selected.filter(pl.col("symbol") == symbol)
    if timeframe is not None:
        selected = selected.filter(pl.col("timeframe") == timeframe)
    return selected.to_dicts()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sum(values: list[float]) -> float:
    return sum(values) if values else 0.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(max(int(round((len(ordered) - 1) * q)), 0), len(ordered) - 1)
    return ordered[index]


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _win_rate(values: list[float]) -> float:
    return sum(1 for value in values if value > 0) / len(values) * 100.0 if values else 0.0


def _sharpe_like(values: list[float]) -> float:
    std = _std(values)
    return _mean(values) / std if std > 0 else 0.0


def _max_drawdown(values: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity - peak)
    return max_drawdown


def _ensure_columns(frame: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return ensure_columns(frame, schema)


def _empty_frame(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return empty_frame(schema)
