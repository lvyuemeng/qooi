"""Generic known-at-close research pattern and outcome pipeline."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping

import polars as pl

from qooi.research.artifacts import ArtifactBundle, empty_frame, ensure_columns

RESEARCH_FRAME_SCHEMA: dict[str, pl.DataType] = {
    "symbol": pl.Utf8,
    "timeframe": pl.Utf8,
    "timestamp": pl.Int64,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "state_source": pl.Utf8,
    "state_column": pl.Utf8,
    "state_value": pl.Utf8,
    "event_column": pl.Utf8,
    "event_value": pl.Utf8,
}

PATTERN_TABLE_SCHEMA: dict[str, pl.DataType] = {
    "pattern_id": pl.Utf8,
    "pattern_family": pl.Utf8,
    "pattern_source": pl.Utf8,
    "symbol": pl.Utf8,
    "timeframe": pl.Utf8,
    "timestamp": pl.Int64,
    "state_source": pl.Utf8,
    "state_column": pl.Utf8,
    "pattern_value": pl.Utf8,
    "event_value": pl.Utf8,
    "side": pl.Utf8,
    "ngram_length": pl.Int64,
    "invalid_state_present": pl.Boolean,
}

OUTCOME_TABLE_SCHEMA: dict[str, pl.DataType] = {
    **PATTERN_TABLE_SCHEMA,
    "horizon": pl.Int64,
    "forward_return_pct": pl.Float64,
    "side_return_pct": pl.Float64,
    "forward_direction": pl.Utf8,
}

METRIC_TABLE_SCHEMA: dict[str, pl.DataType] = {
    "pattern_id": pl.Utf8,
    "pattern_family": pl.Utf8,
    "pattern_source": pl.Utf8,
    "symbol": pl.Utf8,
    "horizon": pl.Int64,
    "side": pl.Utf8,
    "rows": pl.Int64,
    "positive_rate": pl.Float64,
    "negative_rate": pl.Float64,
    "positive_mean": pl.Float64,
    "negative_mean_abs": pl.Float64,
    "omega_ratio": pl.Float64,
    "pwpr": pl.Float64,
    "sortino_zero": pl.Float64,
    "mean_side_return_pct": pl.Float64,
    "transition_information": pl.Float64,
    "conditional_transition_information": pl.Float64,
    "normalized_transition_information": pl.Float64,
    "normalized_conditional_transition_information": pl.Float64,
    "bias_warning": pl.Utf8,
    "sufficient": pl.Boolean,
    "invalid_state_present": pl.Boolean,
}

SCORED_PATTERN_SCHEMA: dict[str, pl.DataType] = {
    **METRIC_TABLE_SCHEMA,
    "passes_candidate_gate": pl.Boolean,
    "gate_failure_reasons": pl.Utf8,
    "passes_promotion_gate": pl.Boolean,
    "promotion_failure_reasons": pl.Utf8,
    "sufficient_symbols": pl.Int64,
    "symbol_direction_agreement_pct": pl.Float64,
    "time_splits": pl.Int64,
    "sufficient_time_splits": pl.Int64,
    "time_split_sign_agreement_pct": pl.Float64,
    "time_stable": pl.Boolean,
}

DEFAULT_INVALID_VALUES = ("warmup", "unknown", "data_error")
_BULLISH_EVENTS = {"failed_breakout_low", "bullish_reclaim", "breakout_acceptance_high"}
_BEARISH_EVENTS = {"failed_breakout_high", "bearish_reclaim", "breakout_acceptance_low"}


def normalize_research_frame(
    frame: pl.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    state_columns: Iterable[str],
    event_column: str,
    context_columns: Iterable[str] = (),
    state_source: str = "classifier",
) -> pl.DataFrame:
    """Convert a wide known-at-close frame into long state/event rows."""
    if frame.is_empty():
        return _empty_frame(RESEARCH_FRAME_SCHEMA)
    base_cols = [
        column
        for column in ("timestamp", "open", "high", "low", "close", "split")
        if column in frame
    ]
    context_cols = [column for column in context_columns if column in frame]
    states = [column for column in state_columns if column in frame]
    if not states:
        return _empty_frame(RESEARCH_FRAME_SCHEMA)
    work = frame.with_columns(
        pl.lit(symbol).alias("symbol"),
        pl.lit(timeframe).alias("timeframe"),
        pl.lit(event_column).alias("event_column"),
        pl.col(event_column).cast(pl.Utf8).alias("event_value")
        if event_column in frame.columns
        else pl.lit(None, dtype=pl.Utf8).alias("event_value"),
    )
    rows = []
    for state_column in states:
        selected = work.select(
            "symbol",
            "timeframe",
            *base_cols,
            pl.lit(state_source).alias("state_source"),
            pl.lit(state_column).alias("state_column"),
            pl.col(state_column).cast(pl.Utf8).alias("state_value"),
            "event_column",
            "event_value",
            *context_cols,
        )
        rows.append(_ensure_columns(selected, RESEARCH_FRAME_SCHEMA))
    return _concat_research_frames(rows)


def materialize_transition_patterns(
    research_frame: pl.DataFrame, spec: Mapping[str, object] | None = None
) -> pl.DataFrame:
    if research_frame.is_empty():
        return _empty_frame(PATTERN_TABLE_SCHEMA)
    ngram_lengths = tuple(int(v) for v in (spec or {}).get("ngram_lengths", (2,)))
    source = str((spec or {}).get("pattern_source", "transition"))
    sort_cols = ["symbol", "timeframe", "state_source", "state_column", "timestamp"]
    work = research_frame.filter(pl.col("state_value").is_not_null()).sort(
        [column for column in sort_cols if column in research_frame.columns]
    )
    frames = []
    group = ["symbol", "timeframe", "state_source", "state_column"]
    for length in ngram_lengths:
        if length < 2:
            continue
        value_exprs = [
            pl.col("state_value").shift(offset).over(group) for offset in range(length - 1, -1, -1)
        ]
        ngram = pl.concat_str([expr.cast(pl.Utf8) for expr in value_exprs], separator="->")
        ready = pl.all_horizontal([expr.is_not_null() for expr in value_exprs])
        family = "transition" if length == 2 else "transition_ngram"
        frame = work.with_columns(
            ngram.alias("pattern_value"),
            ready.alias("_ready"),
        ).filter(pl.col("_ready"))
        optional_cols = [column for column in ("split",) if column in frame.columns]
        frame = frame.select(
            pl.lit(family).alias("pattern_family"),
            pl.lit(source).alias("pattern_source"),
            "symbol",
            "timeframe",
            "timestamp",
            *optional_cols,
            "state_source",
            "state_column",
            "pattern_value",
            "event_value",
            pl.lit(None, dtype=pl.Utf8).alias("side"),
            pl.lit(length).alias("ngram_length"),
            _invalid_expr("pattern_value", _invalid_values(spec)).alias("invalid_state_present"),
        )
        frames.append(_with_pattern_id(frame))
    return _concat_patterns(frames)


def materialize_state_patterns(research_frame: pl.DataFrame, source: str) -> pl.DataFrame:
    if research_frame.is_empty():
        return _empty_frame(PATTERN_TABLE_SCHEMA)
    work = research_frame.filter(pl.col("state_value").is_not_null())
    if work.is_empty():
        return _empty_frame(PATTERN_TABLE_SCHEMA)
    optional_cols = [column for column in ("split",) if column in work.columns]
    frame = work.select(
        pl.lit("state").alias("pattern_family"),
        pl.lit(source).alias("pattern_source"),
        "symbol",
        "timeframe",
        "timestamp",
        *optional_cols,
        "state_source",
        "state_column",
        pl.col("state_value").alias("pattern_value"),
        "event_value",
        pl.lit(None, dtype=pl.Utf8).alias("side"),
        pl.lit(1).alias("ngram_length"),
        pl.lit(False).alias("invalid_state_present"),
    )
    return _with_pattern_id(frame)


def _materialize_none_event_context_patterns(
    research_frame: pl.DataFrame, spec: Mapping[str, object] | None = None
) -> pl.DataFrame:
    if research_frame.is_empty():
        return _empty_frame(PATTERN_TABLE_SCHEMA)
    context_columns = [
        column for column in (spec or {}).get("context_columns", ()) if column in research_frame
    ]
    if not context_columns:
        return _empty_frame(PATTERN_TABLE_SCHEMA)
    source = str((spec or {}).get("pattern_source", "none_event_context"))
    rows = []
    none_rows = research_frame.filter(pl.col("event_value") == "none")
    for column in context_columns:
        rows.append(
            _with_pattern_id(
                none_rows.select(
                    pl.lit("none_event_context").alias("pattern_family"),
                    pl.lit(source).alias("pattern_source"),
                    "symbol",
                    "timeframe",
                    "timestamp",
                    "state_source",
                    pl.lit(column).alias("state_column"),
                    pl.col(column).cast(pl.Utf8).alias("pattern_value"),
                    "event_value",
                    pl.lit(None, dtype=pl.Utf8).alias("side"),
                    pl.lit(1).alias("ngram_length"),
                    _invalid_expr(column, _invalid_values(spec)).alias("invalid_state_present"),
                )
            )
        )
    return _concat_patterns(rows)


def attach_forward_outcomes(
    patterns: pl.DataFrame, market_frame: pl.DataFrame, horizons: Iterable[int]
) -> pl.DataFrame:
    if patterns.is_empty():
        return _empty_frame(OUTCOME_TABLE_SCHEMA)
    if market_frame.is_empty() or "close" not in market_frame.columns:
        return _empty_frame(OUTCOME_TABLE_SCHEMA)
    keys = [column for column in ("symbol", "timeframe", "timestamp") if column in patterns.columns]
    market = market_frame
    if "symbol" not in market.columns and "symbol" in patterns.columns:
        market = market.with_columns(pl.lit(patterns.select("symbol").item(0, 0)).alias("symbol"))
    if "timeframe" not in market.columns and "timeframe" in patterns.columns:
        market = market.with_columns(
            pl.lit(patterns.select("timeframe").item(0, 0)).alias("timeframe")
        )
    sort_cols = [
        column for column in ("symbol", "timeframe", "timestamp") if column in market.columns
    ]
    market = market.sort(sort_cols) if sort_cols else market.sort("timestamp")
    frames = []
    group_cols = [column for column in ("symbol", "timeframe") if column in market.columns]
    for horizon in horizons:
        future_close = (
            pl.col("close").shift(-int(horizon)).over(group_cols)
            if group_cols
            else pl.col("close").shift(-int(horizon))
        )
        returns = market.with_columns(
            pl.lit(int(horizon)).alias("horizon"),
            ((future_close - pl.col("close")) / pl.col("close") * 100.0).alias(
                "forward_return_pct"
            ),
        ).with_columns(
            pl.when(pl.col("forward_return_pct") > 0)
            .then(pl.lit("up"))
            .when(pl.col("forward_return_pct") < 0)
            .then(pl.lit("down"))
            .otherwise(pl.lit("flat"))
            .alias("forward_direction")
        )
        joined = patterns.join(
            returns.select(*keys, "horizon", "forward_return_pct", "forward_direction"),
            on=keys,
            how="left",
        )
        frames.append(joined)
    return _attach_side_returns(pl.concat(frames, how="diagonal_relaxed"))


def filter_evaluation_outcomes(
    outcomes: pl.DataFrame,
    *,
    returns_split: str = "test",
    transaction_cost_bps: float = 0.0,
) -> pl.DataFrame:
    if outcomes.is_empty():
        return outcomes
    work = outcomes.filter(pl.col("forward_return_pct").is_not_null())
    if returns_split != "all" and "split" in work.columns:
        work = work.filter(pl.col("split") == returns_split)
    cost_pct = transaction_cost_bps / 100.0
    return work.with_columns(
        (pl.col("side_return_pct") - cost_pct).alias("side_return_pct"),
        pl.lit(transaction_cost_bps).alias("transaction_cost_bps"),
        pl.lit(returns_split).alias("returns_split"),
    )


def summarize_returns(outcomes: pl.DataFrame, group_cols: Iterable[str]) -> pl.DataFrame:
    if outcomes.is_empty():
        return _empty_frame(METRIC_TABLE_SCHEMA)
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
    return _ensure_columns(out, METRIC_TABLE_SCHEMA)


def summarize_transition_information(
    frame: pl.DataFrame, *, state_column: str = "state_value", condition_column: str = "event_value"
) -> pl.DataFrame:
    if frame.is_empty() or state_column not in frame.columns:
        return _empty_frame(METRIC_TABLE_SCHEMA)
    group_cols = [
        column for column in ("symbol", "timeframe", "state_column") if column in frame.columns
    ]
    if not group_cols:
        group_cols = ["state_column"] if "state_column" in frame.columns else []
    rows = []
    groups = frame.partition_by(group_cols, as_dict=True) if group_cols else {(): frame}
    for key, group in groups.items():
        key_values = key if isinstance(key, tuple) else (key,)
        group = group.filter(pl.col(state_column).is_not_null())
        if group.is_empty():
            continue
        ti = _transition_information(group, state_column)
        cti = (
            _conditional_mutual_information(group, state_column, "_prev_state", condition_column)
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
                "normalized_transition_information": ti / max(_entropy(group, state_column), 1e-12),
                "normalized_conditional_transition_information": cti
                / max(_entropy(group, state_column), 1e-12),
                "bias_warning": "sparse" if group.height < 100 else "none",
                "sufficient": group.height >= 100,
                "invalid_state_present": False,
            }
        )
    return _ensure_columns(pl.DataFrame(rows), METRIC_TABLE_SCHEMA)


def summarize_state_info(
    frame: pl.DataFrame,
    state_col: str,
    group_cols: tuple[str, ...],
) -> pl.DataFrame:
    if frame.is_empty() or state_col not in frame.columns:
        return pl.DataFrame()
    groups = [column for column in group_cols if column in frame.columns]
    work = frame.filter(pl.col(state_col).is_not_null())
    if work.is_empty():
        return pl.DataFrame()
    partitions = work.partition_by(groups, as_dict=True) if groups else {(): work}
    rows = []
    for key, group in partitions.items():
        key_values = key if isinstance(key, tuple) else (key,)
        states = [str(value) for value in group.get_column(state_col).to_list()]
        pairs = list(zip(states[:-1], states[1:], strict=False))
        entropy = _entropy_values(states)
        conditional_entropy = _conditional_entropy_values(pairs)
        mutual_information = max(entropy - conditional_entropy, 0.0)
        rows.append(
            {
                **{column: value for column, value in zip(groups, key_values, strict=False)},
                "rows": len(states),
                "active_states": len(set(states)),
                "state_entropy_bits": entropy,
                "conditional_entropy_bits": conditional_entropy,
                "mutual_information_bits": mutual_information,
                "normalized_mutual_information": mutual_information / max(entropy, 1e-12),
                "effective_states": 2.0**entropy,
                "statistical_complexity_proxy": entropy,
                "predictive_information_proxy": mutual_information,
            }
        )
    return pl.DataFrame(rows)


def with_transition_path_scores(graph: pl.DataFrame) -> pl.DataFrame:
    if graph.is_empty() or "transition_probability" not in graph.columns:
        return pl.DataFrame()
    surprisal = -(pl.col("transition_probability").log() / math.log(2))
    return graph.with_columns(
        surprisal.alias("surprisal_bits"),
        (pl.col("transition_probability") * surprisal).alias("information_contribution_bits"),
    )


def _project_top_transitions(
    scored_graph: pl.DataFrame, *, order_by: str, label: str, n: int
) -> pl.DataFrame:
    if scored_graph.is_empty() or order_by not in scored_graph.columns:
        return pl.DataFrame()
    return (
        scored_graph.sort(order_by, descending=True)
        .head(n)
        .with_columns(pl.lit(label).alias("path_kind"))
    )


def project_transition_paths(graph: pl.DataFrame) -> pl.DataFrame:
    scored = with_transition_path_scores(graph)
    paths = [
        _project_top_transitions(
            scored,
            order_by="transition_probability",
            label="high_probability",
            n=25,
        ),
        _project_top_transitions(
            scored,
            order_by="surprisal_bits",
            label="low_probability_high_information",
            n=25,
        ),
    ]
    non_empty = [path for path in paths if not path.is_empty()]
    return pl.concat(non_empty, how="diagonal_relaxed") if non_empty else pl.DataFrame()


def apply_candidate_gate(metrics: pl.DataFrame, thresholds: Mapping[str, object]) -> pl.DataFrame:
    min_rows = int(thresholds.get("min_rows", 30))
    omega = float(thresholds.get("omega_threshold", 1.5))
    pwpr = float(thresholds.get("pwpr_threshold", 2.0))
    scored = metrics.with_columns(
        (
            (pl.col("rows") >= min_rows)
            & (~pl.col("invalid_state_present").fill_null(False))
            & (pl.col("omega_ratio").fill_null(0.0) > omega)
            & (pl.col("pwpr").fill_null(0.0) > pwpr)
            & (pl.col("mean_side_return_pct").fill_null(0.0) > 0)
        ).alias("passes_candidate_gate"),
        pl.concat_str(
            [
                _reason(pl.col("rows") < min_rows, "rows,"),
                _reason(pl.col("invalid_state_present").fill_null(False), "invalid_state,"),
                _reason(pl.col("omega_ratio").fill_null(0.0) <= omega, "omega,"),
                _reason(pl.col("pwpr").fill_null(0.0) <= pwpr, "pwpr,"),
                _reason(pl.col("mean_side_return_pct").fill_null(0.0) <= 0, "direction,"),
            ]
        )
        .str.strip_chars_end(",")
        .alias("gate_failure_reasons"),
        pl.lit(False).alias("passes_promotion_gate"),
        pl.lit("unscored").alias("promotion_failure_reasons"),
        pl.lit(0).alias("sufficient_symbols"),
        pl.lit(0.0).alias("symbol_direction_agreement_pct"),
        pl.lit(0).alias("time_splits"),
        pl.lit(0).alias("sufficient_time_splits"),
        pl.lit(0.0).alias("time_split_sign_agreement_pct"),
        pl.lit(False).alias("time_stable"),
    )
    return _ensure_columns(scored, SCORED_PATTERN_SCHEMA)


def _apply_promotion_gate(scored: pl.DataFrame, thresholds: Mapping[str, object]) -> pl.DataFrame:
    min_rows = int(thresholds.get("promotion_min_rows", 50))
    min_symbols = int(thresholds.get("promotion_min_symbols", 3))
    min_splits = int(thresholds.get("promotion_min_time_splits", 2))
    symbol_agreement = float(thresholds.get("promotion_symbol_agreement_pct", 67.0))
    time_agreement = float(thresholds.get("promotion_time_agreement_pct", 100.0))
    out = scored.with_columns(
        (
            (pl.col("sufficient_time_splits") >= min_splits)
            & (pl.col("time_split_sign_agreement_pct") >= time_agreement)
        ).alias("time_stable")
    ).with_columns(
        (
            pl.col("passes_candidate_gate")
            & (pl.col("rows") >= min_rows)
            & (pl.col("sufficient_symbols") >= min_symbols)
            & (pl.col("symbol_direction_agreement_pct") >= symbol_agreement)
            & pl.col("time_stable")
        ).alias("passes_promotion_gate"),
        pl.concat_str(
            [
                _reason(~pl.col("passes_candidate_gate"), "candidate_gate,"),
                _reason(pl.col("rows") < min_rows, "promotion_rows,"),
                _reason(pl.col("sufficient_symbols") < min_symbols, "symbols,"),
                _reason(
                    pl.col("symbol_direction_agreement_pct") < symbol_agreement, "symbol_agreement,"
                ),
                _reason(pl.col("sufficient_time_splits") < min_splits, "time_splits,"),
                _reason(
                    pl.col("time_split_sign_agreement_pct") < time_agreement, "time_agreement,"
                ),
            ]
        )
        .str.strip_chars_end(",")
        .alias("promotion_failure_reasons"),
    )
    return _ensure_columns(out, SCORED_PATTERN_SCHEMA)


def project_transition_graph(patterns: pl.DataFrame) -> pl.DataFrame:
    if patterns.is_empty():
        return pl.DataFrame()
    transitions = patterns.filter(pl.col("pattern_family") == "transition")
    if transitions.is_empty():
        return pl.DataFrame()
    split = transitions.with_columns(
        pl.col("pattern_value")
        .str.split_exact("->", 1)
        .struct.rename_fields(["source_state", "target_state"])
        .alias("_edge")
    ).unnest("_edge")
    counts = split.group_by(
        "symbol",
        "timeframe",
        "state_column",
        "source_state",
        "target_state",
        "invalid_state_present",
    ).agg(pl.len().alias("rows"))
    source_totals = counts.group_by("symbol", "timeframe", "state_column", "source_state").agg(
        pl.col("rows").sum().alias("source_rows")
    )
    return counts.join(
        source_totals, on=["symbol", "timeframe", "state_column", "source_state"]
    ).with_columns(
        pl.lit("state-transition-graph").alias("artifact"),
        (pl.col("rows") / pl.col("source_rows")).alias("transition_probability"),
    )


def project_pattern_quality(scored: pl.DataFrame, families: tuple[str, ...] = ()) -> pl.DataFrame:
    if scored.is_empty() or not families:
        return scored
    return scored.filter(pl.col("pattern_family").is_in(families))


def _project_promotion_candidates(scored: pl.DataFrame) -> pl.DataFrame:
    if scored.is_empty() or "passes_promotion_gate" not in scored.columns:
        return scored.head(0)
    return scored.filter(pl.col("passes_promotion_gate").fill_null(False))


def build_transition_bundle(
    prepared_frames: Iterable[pl.DataFrame],
    *,
    frame_specs: Iterable[Mapping[str, object]],
    horizons: tuple[int, ...],
    thresholds: Mapping[str, object],
) -> ArtifactBundle:
    research_frames = []
    market_frames = []
    for frame, spec in zip(prepared_frames, frame_specs, strict=False):
        market_frames.append(frame)
        research_frames.append(
            normalize_research_frame(
                frame,
                symbol=str(spec["symbol"]),
                timeframe=str(spec["timeframe"]),
                state_columns=tuple(spec["state_columns"]),
                event_column=str(spec.get("event_column", "liquidity_event_type")),
                context_columns=tuple(spec.get("context_columns", ())),
            )
        )
    research_frame = _concat_research_frames(research_frames)
    transition_patterns = materialize_transition_patterns(
        research_frame, {"ngram_lengths": thresholds.get("ngram_lengths", (2, 3))}
    )
    none_patterns = _materialize_none_event_context_patterns(
        research_frame, {"context_columns": thresholds.get("none_context_columns", ())}
    )
    all_patterns = _concat_patterns([transition_patterns, none_patterns])
    market = pl.concat(market_frames, how="diagonal_relaxed") if market_frames else pl.DataFrame()
    outcome_table = attach_forward_outcomes(all_patterns, market, horizons)
    metric_table = summarize_returns(
        outcome_table,
        ["pattern_id", "pattern_family", "pattern_source", "symbol", "horizon", "side"],
    )
    scored = apply_candidate_gate(metric_table, thresholds)
    bundle_tables = {
        "state-transition-graph.csv": project_transition_graph(all_patterns),
        "transition-information.csv": summarize_transition_information(research_frame),
        "transition-ngram-quality.csv": project_pattern_quality(
            scored, ("transition", "transition_ngram")
        ),
        "none-event-context-quality.csv": project_pattern_quality(scored, ("none_event_context",)),
        "scored-patterns.csv": scored,
        "promotion-candidates.csv": _project_promotion_candidates(scored),
    }
    return ArtifactBundle(
        "transition-discovery",
        bundle_tables,
        summary=(f"patterns={all_patterns.height}", f"scored={scored.height}"),
    )


def _ensure_columns(frame: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return ensure_columns(frame, schema)


def _empty_frame(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return empty_frame(schema)


def _concat_research_frames(frames: Iterable[pl.DataFrame]) -> pl.DataFrame:
    non_empty = [frame for frame in frames if not frame.is_empty()]
    if not non_empty:
        return _empty_frame(RESEARCH_FRAME_SCHEMA)
    return _ensure_columns(pl.concat(non_empty, how="diagonal_relaxed"), RESEARCH_FRAME_SCHEMA)


def _concat_patterns(patterns: Iterable[pl.DataFrame]) -> pl.DataFrame:
    non_empty = [frame for frame in patterns if not frame.is_empty()]
    if not non_empty:
        return _empty_frame(PATTERN_TABLE_SCHEMA)
    return _ensure_columns(pl.concat(non_empty, how="diagonal_relaxed"), PATTERN_TABLE_SCHEMA)


def _with_pattern_id(frame: pl.DataFrame) -> pl.DataFrame:
    work = frame.with_columns(
        pl.concat_str(
            [
                pl.col("pattern_family"),
                pl.col("pattern_source"),
                pl.col("state_source"),
                pl.col("state_column"),
                pl.col("pattern_value").fill_null("null"),
                pl.col("event_value").fill_null("null"),
                pl.col("ngram_length").cast(pl.Utf8),
            ],
            separator="|",
        ).alias("pattern_id")
    )
    return _ensure_columns(work, PATTERN_TABLE_SCHEMA)


def _invalid_values(spec: Mapping[str, object] | None) -> tuple[str, ...]:
    values = (spec or {}).get("invalid_values", DEFAULT_INVALID_VALUES)
    return tuple(str(value) for value in values)


def _invalid_expr(column: str, invalid_values: tuple[str, ...]) -> pl.Expr:
    expr = pl.lit(False)
    for value in invalid_values:
        expr = expr | pl.col(column).cast(pl.Utf8).str.contains(value, literal=True)
    return expr


def _side_from_event(event_value: pl.Expr | None = None) -> pl.Expr:
    expr = event_value if isinstance(event_value, pl.Expr) else pl.col("event_value")
    return (
        pl.when(expr.is_in(_BULLISH_EVENTS))
        .then(pl.lit("long"))
        .when(expr.is_in(_BEARISH_EVENTS))
        .then(pl.lit("short"))
        .otherwise(None)
    )


def _attach_side_returns(outcomes: pl.DataFrame) -> pl.DataFrame:
    if outcomes.is_empty():
        return _empty_frame(OUTCOME_TABLE_SCHEMA)
    work = outcomes.with_columns(
        pl.coalesce([pl.col("side"), _side_from_event(pl.col("event_value"))]).alias("side")
    ).with_columns(
        pl.when(pl.col("side") == "short")
        .then(-pl.col("forward_return_pct"))
        .otherwise(pl.col("forward_return_pct"))
        .alias("side_return_pct")
    )
    return _ensure_columns(work, OUTCOME_TABLE_SCHEMA)


def _entropy(frame: pl.DataFrame, column: str) -> float:
    if frame.is_empty() or column not in frame.columns:
        return 0.0
    counts = Counter(value for value in frame.get_column(column).to_list() if value is not None)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _entropy_values(values: list[str]) -> float:
    total = len(values)
    if total == 0:
        return 0.0
    counts = Counter(values)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _conditional_entropy_values(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return 0.0
    total = len(pairs)
    sources = {source for source, _target in pairs}
    out = 0.0
    for source in sources:
        targets = [target for prev, target in pairs if prev == source]
        out += len(targets) / total * _entropy_values(targets)
    return out


def _mutual_information(frame: pl.DataFrame, x: str, y: str) -> float:
    if frame.is_empty() or not {x, y} <= set(frame.columns):
        return 0.0
    return _entropy(frame, x) + _entropy(frame, y) - _joint_entropy(frame, (x, y))


def _conditional_mutual_information(frame: pl.DataFrame, x: str, y: str, z: str) -> float:
    if frame.is_empty() or not {x, y, z} <= set(frame.columns):
        return 0.0
    return (
        _joint_entropy(frame, (x, z))
        + _joint_entropy(frame, (y, z))
        - _entropy(frame, z)
        - _joint_entropy(frame, (x, y, z))
    )


def _transition_information(frame: pl.DataFrame, state_column: str) -> float:
    if "_prev_state" not in frame.columns:
        work = frame.with_columns(pl.col(state_column).shift(1).alias("_prev_state"))
    else:
        work = frame
    return _mutual_information(
        work.filter(pl.col("_prev_state").is_not_null()), state_column, "_prev_state"
    )


def _joint_entropy(frame: pl.DataFrame, columns: tuple[str, ...]) -> float:
    values = [tuple(row) for row in frame.select(columns).drop_nulls().iter_rows()]
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _reason(condition: pl.Expr, text: str) -> pl.Expr:
    return pl.when(condition).then(pl.lit(text)).otherwise(pl.lit(""))
