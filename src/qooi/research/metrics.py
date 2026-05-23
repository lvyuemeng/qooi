"""Shared metric kernels for research pattern scoring."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable

import polars as pl

from qooi.research.contracts import METRIC_TABLE_SCHEMA, empty_frame, ensure_columns


def entropy(frame: pl.DataFrame, column: str) -> float:
    if frame.is_empty() or column not in frame.columns:
        return 0.0
    counts = Counter(value for value in frame.get_column(column).to_list() if value is not None)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def mutual_information(frame: pl.DataFrame, x: str, y: str) -> float:
    if frame.is_empty() or not {x, y} <= set(frame.columns):
        return 0.0
    return entropy(frame, x) + entropy(frame, y) - _joint_entropy(frame, (x, y))


def conditional_mutual_information(frame: pl.DataFrame, x: str, y: str, z: str) -> float:
    if frame.is_empty() or not {x, y, z} <= set(frame.columns):
        return 0.0
    return (
        _joint_entropy(frame, (x, z))
        + _joint_entropy(frame, (y, z))
        - entropy(frame, z)
        - _joint_entropy(frame, (x, y, z))
    )


def summarize_returns(outcomes: pl.DataFrame, group_cols: Iterable[str]) -> pl.DataFrame:
    if outcomes.is_empty():
        return empty_frame(METRIC_TABLE_SCHEMA)
    groups = list(dict.fromkeys(group_cols))
    if "side_return_pct" not in outcomes.columns:
        raise ValueError("OutcomeTable missing side_return_pct")
    summarized = outcomes.group_by(groups).agg(
        pl.len().alias("rows"),
        pl.col("side_return_pct").mean().alias("mean_side_return_pct"),
        pl.when(pl.col("side_return_pct") > 0)
        .then(pl.col("side_return_pct"))
        .otherwise(0.0)
        .sum()
        .alias("positive_sum"),
        pl.when(pl.col("side_return_pct") < 0)
        .then(pl.col("side_return_pct"))
        .otherwise(0.0)
        .sum()
        .abs()
        .alias("negative_sum_abs"),
        (pl.col("side_return_pct") > 0).sum().alias("positive_rows"),
        (pl.col("side_return_pct") < 0).sum().alias("negative_rows"),
        pl.when(pl.col("side_return_pct") > 0)
        .then(pl.col("side_return_pct"))
        .otherwise(None)
        .mean()
        .fill_null(0.0)
        .alias("positive_mean"),
        pl.when(pl.col("side_return_pct") < 0)
        .then(pl.col("side_return_pct"))
        .otherwise(None)
        .mean()
        .abs()
        .fill_null(0.0)
        .alias("negative_mean_abs"),
        pl.when(pl.col("side_return_pct") < 0)
        .then(pl.col("side_return_pct") ** 2)
        .otherwise(0.0)
        .mean()
        .alias("downside_variance"),
        pl.col("invalid_state_present").fill_null(False).any().alias("invalid_state_present"),
    )
    out = summarized.with_columns(
        (pl.col("positive_rows") / pl.col("rows") * 100.0).alias("positive_rate"),
        (pl.col("negative_rows") / pl.col("rows") * 100.0).alias("negative_rate"),
        pl.when(pl.col("negative_sum_abs") > 0)
        .then(pl.col("positive_sum") / pl.col("negative_sum_abs"))
        .when(pl.col("positive_sum") > 0)
        .then(999.0)
        .otherwise(0.0)
        .alias("omega_ratio"),
        pl.when(pl.col("negative_mean_abs") > 0)
        .then(pl.col("positive_mean") / pl.col("negative_mean_abs"))
        .when(pl.col("positive_mean") > 0)
        .then(999.0)
        .otherwise(0.0)
        .alias("pwpr"),
        pl.when(pl.col("downside_variance") > 0)
        .then(pl.col("mean_side_return_pct") / pl.col("downside_variance").sqrt())
        .otherwise(0.0)
        .alias("sortino_zero"),
        pl.lit(None, dtype=pl.Float64).alias("transition_information"),
        pl.lit(None, dtype=pl.Float64).alias("conditional_transition_information"),
        pl.lit(None, dtype=pl.Float64).alias("normalized_transition_information"),
        pl.lit(None, dtype=pl.Float64).alias("normalized_conditional_transition_information"),
        pl.lit("none").alias("bias_warning"),
        pl.lit(True).alias("sufficient"),
    )
    return ensure_columns(out, METRIC_TABLE_SCHEMA)


def summarize_transition_information(
    frame: pl.DataFrame, *, state_column: str = "state_value", condition_column: str = "event_value"
) -> pl.DataFrame:
    if frame.is_empty() or state_column not in frame.columns:
        return empty_frame(METRIC_TABLE_SCHEMA)
    group_cols = [
        column for column in ("symbol", "timeframe", "state_column") if column in frame.columns
    ]
    if not group_cols:
        group_cols = ["state_column"] if "state_column" in frame.columns else []
    rows = []
    groups = frame.partition_by(group_cols, as_dict=True) if group_cols else {(): frame}
    for key, group in groups.items():
        key_values = key if isinstance(key, tuple) else (key,)
        ti = _transition_information(group, state_column)
        cti = (
            conditional_mutual_information(group, state_column, "_prev_state", condition_column)
            if condition_column in group.columns
            else 0.0
        )
        base = {column: value for column, value in zip(group_cols, key_values, strict=False)}
        rows.append(
            {
                **base,
                "pattern_id": "transition-information|" + "|".join(str(v) for v in key_values),
                "pattern_family": "transition_information",
                "pattern_source": "metrics",
                "horizon": None,
                "side": None,
                "rows": group.height,
                "transition_information": ti,
                "conditional_transition_information": cti,
                "normalized_transition_information": ti / max(entropy(group, state_column), 1e-12),
                "normalized_conditional_transition_information": cti
                / max(entropy(group, state_column), 1e-12),
                "bias_warning": "sparse" if group.height < 100 else "none",
                "sufficient": group.height >= 100,
                "invalid_state_present": False,
            }
        )
    return ensure_columns(pl.DataFrame(rows), METRIC_TABLE_SCHEMA)


def _transition_information(frame: pl.DataFrame, state_column: str) -> float:
    if "_prev_state" not in frame.columns:
        work = frame.with_columns(pl.col(state_column).shift(1).alias("_prev_state"))
    else:
        work = frame
    return mutual_information(
        work.filter(pl.col("_prev_state").is_not_null()), state_column, "_prev_state"
    )


def _joint_entropy(frame: pl.DataFrame, columns: tuple[str, ...]) -> float:
    values = [tuple(row) for row in frame.select(columns).drop_nulls().iter_rows()]
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values())
