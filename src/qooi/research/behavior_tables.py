"""Behavior-state diagnostics, taxonomy, and rule primitive research tables."""

from __future__ import annotations

import math
from collections import Counter

import polars as pl

from qooi.research.artifacts import empty_frame, ensure_columns

STATE_DIAGNOSTIC_SCHEMA: dict[str, pl.DataType] = {
    "context_kind": pl.Utf8,
    "context_value": pl.Utf8,
    "symbol": pl.Utf8,
    "timeframe": pl.Utf8,
    "state_column": pl.Utf8,
    "horizon": pl.Int64,
    "rows": pl.Int64,
    "forward_return_mean_pct": pl.Float64,
    "forward_return_abs_mean_pct": pl.Float64,
    "forward_volatility_pct": pl.Float64,
    "forward_efficiency_ratio": pl.Float64,
    "forward_range_pct": pl.Float64,
    "breakout_up_rate_pct": pl.Float64,
    "breakout_down_rate_pct": pl.Float64,
    "volume_change_mean_pct": pl.Float64,
    "direction_bias_pct": pl.Float64,
    "market_beta_proxy": pl.Float64,
    "split": pl.Utf8,
    "descriptive_only": pl.Boolean,
}

STATE_TRANSITION_CHAIN_SCHEMA: dict[str, pl.DataType] = {
    "context_kind": pl.Utf8,
    "context_value": pl.Utf8,
    "chain_value": pl.Utf8,
    "symbol": pl.Utf8,
    "timeframe": pl.Utf8,
    "timestamp": pl.Int64,
    "state_column": pl.Utf8,
    "state_source": pl.Utf8,
    "ngram_length": pl.Int64,
    "from_state": pl.Utf8,
    "to_state": pl.Utf8,
    "previous_state": pl.Utf8,
    "next_state": pl.Utf8,
    "chain_count": pl.Int64,
    "symbol_chain_count": pl.Int64,
    "split": pl.Utf8,
    "invalid_state_present": pl.Boolean,
}

STATE_CHAIN_INFORMATION_SCHEMA: dict[str, pl.DataType] = {
    "context_kind": pl.Utf8,
    "context_value": pl.Utf8,
    "symbol": pl.Utf8,
    "timeframe": pl.Utf8,
    "state_column": pl.Utf8,
    "ngram_length": pl.Int64,
    "horizon": pl.Int64,
    "rows": pl.Int64,
    "next_state_entropy_bits": pl.Float64,
    "conditional_next_state_entropy_bits": pl.Float64,
    "information_gain_bits": pl.Float64,
    "normalized_information_gain": pl.Float64,
    "return_bucket_mutual_information_bits": pl.Float64,
    "vol_bucket_mutual_information_bits": pl.Float64,
    "breakout_bucket_mutual_information_bits": pl.Float64,
    "top_next_state": pl.Utf8,
    "top_next_state_probability": pl.Float64,
    "effective_next_state_count": pl.Float64,
    "surprisal_bits": pl.Float64,
    "sufficient": pl.Boolean,
}

STATE_TAXONOMY_SCHEMA: dict[str, pl.DataType] = {
    "context_kind": pl.Utf8,
    "context_value": pl.Utf8,
    "symbol": pl.Utf8,
    "timeframe": pl.Utf8,
    "state_column": pl.Utf8,
    "ngram_length": pl.Int64,
    "horizon": pl.Int64,
    "taxonomy_label": pl.Utf8,
    "rows": pl.Int64,
    "information_score": pl.Float64,
    "descriptive_only": pl.Boolean,
    "reason": pl.Utf8,
}

DEFAULT_INVALID_VALUES = ("warmup", "unknown", "data_error")
_BULLISH_EVENTS = {"failed_breakout_low", "bullish_reclaim", "breakout_acceptance_high"}
_BEARISH_EVENTS = {"failed_breakout_high", "bearish_reclaim", "breakout_acceptance_low"}


def summarize_state_diagnostics(
    market: pl.DataFrame,
    state_column: str,
    horizons: tuple[int, ...],
    *,
    split: str = "test",
) -> pl.DataFrame:
    if market.is_empty() or state_column not in market.columns:
        return _empty_frame(STATE_DIAGNOSTIC_SCHEMA)
    work = _market_with_group_columns(market).sort("symbol", "timeframe", "timestamp")
    if split != "all" and "split" in work.columns:
        work = work.filter(pl.col("split") == split)
    volume_column = (
        "volume" if "volume" in work.columns else "vol" if "vol" in work.columns else None
    )
    frames = []
    group = ["symbol", "timeframe"]
    for horizon in _positive_unique_ints(horizons):
        future_close = pl.col("close").shift(-horizon).over(group)
        future_high = pl.max_horizontal(
            [pl.col("high").shift(-offset).over(group) for offset in range(1, horizon + 1)]
        )
        future_low = pl.min_horizontal(
            [pl.col("low").shift(-offset).over(group) for offset in range(1, horizon + 1)]
        )
        prior_high = pl.max_horizontal(
            [pl.col("high").shift(offset).over(group) for offset in range(0, horizon)]
        )
        prior_low = pl.min_horizontal(
            [pl.col("low").shift(offset).over(group) for offset in range(0, horizon)]
        )
        path_distance = pl.sum_horizontal(
            [
                (
                    pl.col("close").shift(-offset).over(group)
                    - pl.col("close").shift(1 - offset).over(group)
                ).abs()
                for offset in range(1, horizon + 1)
            ]
        )
        if volume_column:
            future_volume = pl.mean_horizontal(
                [
                    pl.col(volume_column).shift(-offset).over(group)
                    for offset in range(1, horizon + 1)
                ]
            )
            prior_volume = pl.mean_horizontal(
                [pl.col(volume_column).shift(offset).over(group) for offset in range(0, horizon)]
            )
            volume_change = (
                pl.when(prior_volume > 0)
                .then((future_volume - prior_volume) / prior_volume * 100.0)
                .otherwise(None)
            )
        else:
            volume_change = pl.lit(None, dtype=pl.Float64)
        observations = (
            work.with_columns(
                pl.lit("state").alias("context_kind"),
                pl.col(state_column).cast(pl.Utf8).alias("context_value"),
                pl.lit(state_column).alias("state_column"),
                pl.lit(horizon).alias("horizon"),
                future_close.alias("_future_close"),
                future_high.alias("_future_high"),
                future_low.alias("_future_low"),
                prior_high.alias("_prior_high"),
                prior_low.alias("_prior_low"),
                path_distance.alias("_path_distance"),
                volume_change.alias("volume_change_pct"),
                pl.lit(split).alias("split"),
            )
            .filter(
                pl.col("context_value").is_not_null()
                & pl.col("_future_close").is_not_null()
                & pl.col("close").is_not_null()
                & (pl.col("close") != 0)
            )
            .with_columns(
                ((pl.col("_future_close") - pl.col("close")) / pl.col("close") * 100.0).alias(
                    "forward_return_pct"
                ),
                (
                    (pl.col("_future_high") - pl.col("_future_low"))
                    / pl.col("close")
                    * 100.0
                ).alias(
                    "forward_range_pct"
                ),
                (pl.col("_future_high") > pl.col("_prior_high")).alias("breakout_up"),
                (pl.col("_future_low") < pl.col("_prior_low")).alias("breakout_down"),
            )
            .with_columns(
                pl.col("forward_return_pct").abs().alias("forward_abs_return_pct"),
                pl.when(pl.col("_path_distance") > 0)
                .then((pl.col("_future_close") - pl.col("close")).abs() / pl.col("_path_distance"))
                .otherwise(None)
                .alias("forward_efficiency_ratio"),
            )
        )
        frames.append(observations)
    if not frames:
        return _empty_frame(STATE_DIAGNOSTIC_SCHEMA)
    observations = pl.concat(frames, how="diagonal_relaxed")
    rows = observations.group_by(
        "context_kind",
        "context_value",
        "symbol",
        "timeframe",
        "state_column",
        "horizon",
        "split",
    ).agg(
        pl.len().alias("rows"),
        pl.col("forward_return_pct").mean().alias("forward_return_mean_pct"),
        pl.col("forward_abs_return_pct").mean().alias("forward_return_abs_mean_pct"),
        pl.col("forward_return_pct").std().fill_null(0.0).alias("forward_volatility_pct"),
        pl.col("forward_efficiency_ratio").mean().alias("forward_efficiency_ratio"),
        pl.col("forward_range_pct").mean().alias("forward_range_pct"),
        (pl.col("breakout_up").mean() * 100.0).alias("breakout_up_rate_pct"),
        (pl.col("breakout_down").mean() * 100.0).alias("breakout_down_rate_pct"),
        pl.col("volume_change_pct").mean().alias("volume_change_mean_pct"),
        ((pl.col("forward_return_pct") > 0).mean() * 100.0).alias("direction_bias_pct"),
        pl.lit(None, dtype=pl.Float64).alias("market_beta_proxy"),
        pl.lit(True).alias("descriptive_only"),
    )
    return _ensure_columns(rows, STATE_DIAGNOSTIC_SCHEMA)


def build_state_transition_chains(
    market: pl.DataFrame, state_column: str, ngram_lengths: tuple[int, ...]
) -> pl.DataFrame:
    if market.is_empty() or state_column not in market.columns:
        return _empty_frame(STATE_TRANSITION_CHAIN_SCHEMA)
    work = _market_with_group_columns(market).sort("symbol", "timeframe", "timestamp")
    group = ["symbol", "timeframe"]
    frames = []
    for length in _positive_unique_ints(ngram_lengths):
        values = [
            pl.col(state_column).shift(offset).over(group).cast(pl.Utf8)
            for offset in range(length - 1, -1, -1)
        ]
        ready = pl.all_horizontal([expr.is_not_null() for expr in values])
        chain_value = pl.concat_str(values, separator="->")
        invalid_state = pl.any_horizontal([expr.is_in(DEFAULT_INVALID_VALUES) for expr in values])
        previous_state = values[-2] if length > 1 else pl.lit(None, dtype=pl.Utf8)
        frames.append(
            work.with_columns(
                pl.lit("state" if length == 1 else "chain").alias("context_kind"),
                chain_value.alias("chain_value"),
                pl.lit(state_column).alias("state_column"),
                pl.lit("vq_rssm").alias("state_source"),
                pl.lit(length).alias("ngram_length"),
                values[0].alias("from_state"),
                values[-1].alias("to_state"),
                previous_state.alias("previous_state"),
                pl.col(state_column).shift(-1).over(group).cast(pl.Utf8).alias("next_state"),
                invalid_state.alias("invalid_state_present"),
                ready.alias("_ready"),
            )
            .filter(pl.col("_ready"))
            .with_columns(
                pl.when(pl.lit(length == 1))
                .then(pl.col("to_state"))
                .otherwise(pl.col("chain_value"))
                .alias("context_value"),
            )
        )
    non_empty = [frame for frame in frames if not frame.is_empty()]
    if not non_empty:
        return _empty_frame(STATE_TRANSITION_CHAIN_SCHEMA)
    frame = pl.concat(non_empty, how="diagonal_relaxed")
    counts = frame.group_by("context_kind", "context_value", "ngram_length").agg(
        pl.len().alias("chain_count")
    )
    symbol_counts = frame.group_by("symbol", "context_kind", "context_value", "ngram_length").agg(
        pl.len().alias("symbol_chain_count")
    )
    return _ensure_columns(
        frame.join(counts, on=["context_kind", "context_value", "ngram_length"], how="left").join(
            symbol_counts,
            on=["symbol", "context_kind", "context_value", "ngram_length"],
            how="left",
        ),
        STATE_TRANSITION_CHAIN_SCHEMA,
    )


def summarize_state_chain_information(
    chains: pl.DataFrame, market: pl.DataFrame, horizons: tuple[int, ...]
) -> pl.DataFrame:
    if chains.is_empty() or market.is_empty():
        return _empty_frame(STATE_CHAIN_INFORMATION_SCHEMA)
    market_index = _market_row_index(market)
    observations = []
    for row in chains.to_dicts():
        key = (str(row.get("symbol")), str(row.get("timeframe")), int(row.get("timestamp")))
        indexed = market_index.get(key)
        if indexed is None:
            continue
        group_rows, index = indexed
        close = group_rows[index].get("close")
        if close in (None, 0):
            continue
        for horizon in _positive_unique_ints(horizons):
            if index + horizon >= len(group_rows):
                continue
            future_close = group_rows[index + horizon].get("close")
            if future_close is None:
                continue
            window = group_rows[index + 1 : index + horizon + 1]
            ret = (float(future_close) - float(close)) / float(close) * 100.0
            high_values = [float(item["high"]) for item in window if item.get("high") is not None]
            low_values = [float(item["low"]) for item in window if item.get("low") is not None]
            prior = group_rows[max(0, index - horizon + 1) : index + 1]
            prior_highs = [float(item["high"]) for item in prior if item.get("high") is not None]
            prior_lows = [float(item["low"]) for item in prior if item.get("low") is not None]
            observations.append(
                {
                    **row,
                    "horizon": horizon,
                    "return_bucket": _return_bucket(ret),
                    "vol_bucket": _vol_bucket(
                        (max(high_values) - min(low_values)) / float(close) * 100.0
                        if high_values and low_values
                        else 0.0
                    ),
                    "breakout_bucket": _breakout_bucket(
                        high_values, low_values, prior_highs, prior_lows
                    ),
                }
            )
    return _chain_information_rows(observations)


def classify_state_taxonomy(diagnostics: pl.DataFrame) -> pl.DataFrame:
    if diagnostics.is_empty():
        return _empty_frame(STATE_TAXONOMY_SCHEMA)
    rows = []
    for row in diagnostics.to_dicts():
        rows_count = int(row.get("rows") or 0)
        kind = str(row.get("context_kind") or "state")
        min_rows = 20 if kind == "chain" else 10
        information = float(
            row.get("normalized_information_gain") or row.get("information_score") or 0.0
        )
        if rows_count < min_rows:
            label, reason = "avoid", "insufficient_rows"
        elif kind == "chain" and information >= 0.25:
            label, reason = "informative_transition", "high_information_gain"
        else:
            direction_bias = float(row.get("direction_bias_pct") or 50.0)
            efficiency = float(row.get("forward_efficiency_ratio") or 0.0)
            range_pct = float(row.get("forward_range_pct") or 0.0)
            volatility = float(row.get("forward_volatility_pct") or 0.0)
            breakout_rate = max(
                float(row.get("breakout_up_rate_pct") or 0.0),
                float(row.get("breakout_down_rate_pct") or 0.0),
            )
            if efficiency >= 0.55 and abs(direction_bias - 50.0) >= 15.0:
                label, reason = "trend_smooth", "efficient_directional"
            elif breakout_rate >= 40.0:
                label, reason = "breakout_prone", "frequent_future_breakouts"
            elif range_pct <= 1.5 and volatility <= 1.0:
                label, reason = "narrow_compression", "low_range_low_volatility"
            elif range_pct >= 3.0 and efficiency < 0.35:
                label, reason = "wide_chop", "wide_inefficient_range"
            elif abs(direction_bias - 50.0) >= 10.0:
                label, reason = "trend_choppy", "directional_low_efficiency"
            else:
                label, reason = "avoid", "weak_or_conflicting_diagnostics"
        rows.append(
            {
                "context_kind": kind,
                "context_value": str(row.get("context_value")),
                "symbol": row.get("symbol"),
                "timeframe": row.get("timeframe"),
                "state_column": row.get("state_column"),
                "ngram_length": int(row.get("ngram_length") or (1 if kind == "state" else 0)),
                "horizon": row.get("horizon"),
                "taxonomy_label": label,
                "rows": rows_count,
                "information_score": information,
                "descriptive_only": bool(row.get("descriptive_only", True)),
                "reason": reason,
            }
        )
    return _ensure_columns(pl.DataFrame(rows), STATE_TAXONOMY_SCHEMA)


def _market_with_group_columns(market: pl.DataFrame) -> pl.DataFrame:
    work = market
    if "symbol" not in work.columns:
        work = work.with_columns(pl.lit("unknown").alias("symbol"))
    if "timeframe" not in work.columns:
        work = work.with_columns(pl.lit("unknown").alias("timeframe"))
    return work


def _iter_market_groups(market: pl.DataFrame):
    if market.is_empty() or "timestamp" not in market.columns:
        return
    work = market
    if "symbol" not in work.columns:
        work = work.with_columns(pl.lit("unknown").alias("symbol"))
    if "timeframe" not in work.columns:
        work = work.with_columns(pl.lit("unknown").alias("timeframe"))
    for key, group in (
        work.sort(["symbol", "timeframe", "timestamp"])
        .partition_by(["symbol", "timeframe"], as_dict=True)
        .items()
    ):
        symbol, timeframe = key if isinstance(key, tuple) else (key, "unknown")
        yield (str(symbol), str(timeframe)), group.to_dicts()


def _market_row_index(
    market: pl.DataFrame,
) -> dict[tuple[str, str, int], tuple[list[dict[str, object]], int]]:
    out = {}
    for (_symbol, _timeframe), rows in _iter_market_groups(market):
        for index, row in enumerate(rows):
            if row.get("timestamp") is not None:
                out[(str(row.get("symbol")), str(row.get("timeframe")), int(row["timestamp"]))] = (
                    rows,
                    index,
                )
    return out


def _positive_unique_ints(values: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(dict.fromkeys(int(value) for value in values if int(value) > 0))


def _return_bucket(value: float) -> str:
    if value > 0.1:
        return "positive"
    if value < -0.1:
        return "negative"
    return "flat"


def _vol_bucket(value: float) -> str:
    if value >= 3.0:
        return "high"
    if value >= 1.0:
        return "medium"
    return "low"


def _breakout_bucket(
    high_values: list[float],
    low_values: list[float],
    prior_highs: list[float],
    prior_lows: list[float],
) -> str:
    if high_values and prior_highs and max(high_values) > max(prior_highs):
        return "up_breakout"
    if low_values and prior_lows and min(low_values) < min(prior_lows):
        return "down_breakout"
    return "no_breakout"


def _chain_information_rows(observations: list[dict[str, object]]) -> pl.DataFrame:
    if not observations:
        return _empty_frame(STATE_CHAIN_INFORMATION_SCHEMA)
    rows = []
    by_horizon: dict[int, list[dict[str, object]]] = {}
    for item in observations:
        by_horizon.setdefault(int(item["horizon"]), []).append(item)
    for horizon, horizon_rows in by_horizon.items():
        next_states = [
            str(item["next_state"]) for item in horizon_rows if item.get("next_state") is not None
        ]
        next_entropy = _entropy_values(next_states)
        return_entropy = _entropy_values([str(item["return_bucket"]) for item in horizon_rows])
        vol_entropy = _entropy_values([str(item["vol_bucket"]) for item in horizon_rows])
        breakout_entropy = _entropy_values([str(item["breakout_bucket"]) for item in horizon_rows])
        groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
        for item in horizon_rows:
            key = (
                item["context_kind"],
                item["context_value"],
                item["symbol"],
                item["timeframe"],
                item["state_column"],
                item["ngram_length"],
            )
            groups.setdefault(key, []).append(item)
        for key, group in groups.items():
            context_kind, context_value, symbol, timeframe, state_column, ngram_length = key
            group_next_states = [
                str(item["next_state"]) for item in group if item.get("next_state") is not None
            ]
            conditional_entropy = _entropy_values(group_next_states)
            information_gain = max(next_entropy - conditional_entropy, 0.0)
            return_information = max(
                return_entropy - _entropy_values([str(item["return_bucket"]) for item in group]),
                0.0,
            )
            vol_information = max(
                vol_entropy - _entropy_values([str(item["vol_bucket"]) for item in group]),
                0.0,
            )
            breakout_information = max(
                breakout_entropy
                - _entropy_values([str(item["breakout_bucket"]) for item in group]),
                0.0,
            )
            counts = Counter(group_next_states)
            top_next_state, top_count = counts.most_common(1)[0] if counts else (None, 0)
            rows.append(
                {
                    "context_kind": context_kind,
                    "context_value": context_value,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "state_column": state_column,
                    "ngram_length": ngram_length,
                    "horizon": horizon,
                    "rows": len(group),
                    "next_state_entropy_bits": next_entropy,
                    "conditional_next_state_entropy_bits": conditional_entropy,
                    "information_gain_bits": information_gain,
                    "normalized_information_gain": information_gain / max(next_entropy, 1e-12),
                    "return_bucket_mutual_information_bits": return_information,
                    "vol_bucket_mutual_information_bits": vol_information,
                    "breakout_bucket_mutual_information_bits": breakout_information,
                    "top_next_state": top_next_state,
                    "top_next_state_probability": top_count / len(group_next_states)
                    if group_next_states
                    else None,
                    "effective_next_state_count": 2.0**conditional_entropy,
                    "surprisal_bits": -math.log2(top_count / len(group_next_states))
                    if top_count and group_next_states
                    else None,
                    "sufficient": len(group) >= 30,
                }
            )
    return _ensure_columns(pl.DataFrame(rows), STATE_CHAIN_INFORMATION_SCHEMA)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _entropy_values(values: list[str]) -> float:
    total = len(values)
    if not values:
        return 0.0
    counts = Counter(values)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _ensure_columns(frame: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return ensure_columns(frame, schema)


def _empty_frame(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return empty_frame(schema)
