"""Potential scanner kline path and realized-transition history."""

from __future__ import annotations

import polars as pl

KLINE_PATH_HISTORY_SCHEMA = {
    "symbol": pl.String,
    "timeframe": pl.String,
    "bar_close_ms": pl.Int64,
    "direction_hint": pl.String,
    "regime_state": pl.String,
    "structure_state": pl.String,
    "core_context": pl.String,
    "state_age_bars": pl.UInt32,
    "state_age_bucket": pl.String,
    "event_state": pl.String,
    "event_age_bars": pl.UInt32,
    "event_age_bucket": pl.String,
    "range_state": pl.String,
    "vol_state": pl.String,
    "state_changed": pl.Boolean,
    "event_changed": pl.Boolean,
    "fresh_event": pl.Boolean,
    "transition_kind": pl.String,
    "extreme_range": pl.String,
    "extreme_vol": pl.String,
    "compression_state": pl.String,
    "expansion_state": pl.String,
    "transition_path": pl.String,
    "full_context": pl.String,
}

REALIZED_TRANSITION_SCHEMA = {
    "symbol": pl.String,
    "timeframe": pl.String,
    "bar_close_ms": pl.Int64,
    "outcome_horizon": pl.Int64,
    "terminal_direction": pl.String,
    "terminal_regime_state": pl.String,
    "terminal_structure_state": pl.String,
    "terminal_core_context": pl.String,
    "terminal_transition_kind": pl.String,
    "direction_changed": pl.Boolean,
    "regime_changed": pl.Boolean,
    "structure_changed": pl.Boolean,
    "core_context_changed": pl.Boolean,
    "event_fired": pl.Boolean,
    "returned_to_origin": pl.Boolean,
    "time_to_direction_change_bars": pl.Int64,
    "time_to_core_change_bars": pl.Int64,
    "transition_count": pl.Int64,
}


def kline_path_history_frame(config, frames: dict[tuple[str, str], pl.DataFrame]) -> pl.DataFrame:
    histories = []
    for timeframe in config.timeframes:
        for (symbol, frame_timeframe), frame in frames.items():
            if frame_timeframe != timeframe or frame.is_empty():
                continue
            rows = frame
            if rows.is_empty() or rows.get_column("missing_flag").all():
                continue
            histories.append(kline_path_rows(rows, config.transition.ngram_length))
    if not histories:
        return pl.DataFrame(schema=KLINE_PATH_HISTORY_SCHEMA)
    return pl.concat(histories, how="vertical_relaxed").select(*KLINE_PATH_HISTORY_SCHEMA.keys())


def kline_path_rows(rows: pl.DataFrame, ngram_length: int) -> pl.DataFrame:
    ngram_length = max(2, ngram_length)
    state_changed = (pl.col("state_key") != pl.col("state_key").shift(1).over("symbol")).fill_null(
        True
    )
    event_changed = (
        pl.col("context_event") != pl.col("context_event").shift(1).over("symbol")
    ).fill_null(True)
    return (
        rows.sort("symbol", "timestamp")
        .with_columns(
            pl.col("state_key").str.split("|").list.get(0, null_on_oob=True).alias("regime_state"),
            pl.col("state_key")
            .str.split("|")
            .list.get(1, null_on_oob=True)
            .alias("structure_state"),
            pl.col("state_key").str.split("|").list.get(2, null_on_oob=True).alias("range_state"),
            pl.col("state_key").str.split("|").list.get(3, null_on_oob=True).alias("vol_state"),
            pl.concat_str("state_key", "context_event", separator="|").alias("full_context"),
            pl.concat_str(
                [
                    pl.col("state_key").shift(offset).over("symbol")
                    for offset in range(ngram_length - 1, 0, -1)
                ]
                + [pl.col("state_key")],
                separator=" -> ",
            ).alias("transition_path"),
            state_changed.alias("state_changed"),
            event_changed.alias("event_changed"),
            state_changed.cast(pl.Int64).cum_sum().over("symbol").alias("state_run"),
            event_changed.cast(pl.Int64).cum_sum().over("symbol").alias("event_run"),
        )
        .with_columns(
            pl.col("direction_hint").fill_null("unknown_direction"),
            pl.col("regime_state").fill_null("unknown_regime"),
            pl.col("structure_state").fill_null("unknown_structure"),
            pl.col("range_state").fill_null("range_unknown"),
            pl.col("vol_state").fill_null("vol_unknown"),
            pl.col("context_event").fill_null("none_event").alias("event_state"),
        )
        .with_columns(
            pl.concat_str("regime_state", "structure_state", separator="|").alias("core_context"),
            (~pl.col("event_state").str.starts_with("none")).alias("fresh_event"),
            pl.when(pl.col("state_changed") & pl.col("event_changed"))
            .then(pl.lit("state_and_event_transition"))
            .when(pl.col("state_changed"))
            .then(pl.lit("state_transition"))
            .when(pl.col("event_changed") & ~pl.col("event_state").str.starts_with("none"))
            .then(pl.lit("event_trigger"))
            .otherwise(pl.lit("same_context"))
            .alias("transition_kind"),
            pl.when(pl.col("range_state") == "range_tight")
            .then(pl.lit("tight_extreme"))
            .when(pl.col("range_state") == "range_wide")
            .then(pl.lit("wide_extreme"))
            .otherwise(pl.lit("normal_range"))
            .alias("extreme_range"),
            pl.when(pl.col("vol_state") == "vol_low")
            .then(pl.lit("low_vol_extreme"))
            .when(pl.col("vol_state") == "vol_high")
            .then(pl.lit("high_vol_extreme"))
            .when(pl.col("vol_state") == "vol_unknown")
            .then(pl.lit("vol_unknown"))
            .otherwise(pl.lit("normal_vol"))
            .alias("extreme_vol"),
            pl.when(pl.col("range_state") == "range_tight")
            .then(pl.lit("compressed"))
            .otherwise(pl.lit("not_compressed"))
            .alias("compression_state"),
            pl.when(pl.col("range_state") == "range_wide")
            .then(pl.lit("expanded"))
            .otherwise(pl.lit("not_expanded"))
            .alias("expansion_state"),
        )
        .with_columns(
            pl.cum_count("state_key")
            .over("symbol", "state_run")
            .cast(pl.UInt32)
            .alias("state_age_bars"),
            pl.cum_count("context_event")
            .over("symbol", "event_run")
            .cast(pl.UInt32)
            .alias("event_age_bars"),
        )
        .with_columns(
            _age_bucket_expr("state_age_bars").alias("state_age_bucket"),
            _age_bucket_expr("event_age_bars").alias("event_age_bucket"),
        )
        .select(
            pl.col("symbol").cast(pl.String),
            pl.col("scale").alias("timeframe"),
            pl.col("timestamp").alias("bar_close_ms"),
            "direction_hint",
            "regime_state",
            "structure_state",
            "core_context",
            "state_age_bars",
            "state_age_bucket",
            "event_state",
            "event_age_bars",
            "event_age_bucket",
            "range_state",
            "vol_state",
            "state_changed",
            "event_changed",
            "fresh_event",
            "transition_kind",
            "extreme_range",
            "extreme_vol",
            "compression_state",
            "expansion_state",
            "transition_path",
            "full_context",
        )
    )


def _age_bucket_expr(column: str) -> pl.Expr:
    age = pl.col(column).cast(pl.UInt32, strict=False)
    return (
        pl.when(age <= 1)
        .then(pl.lit("new"))
        .when(age <= 3)
        .then(pl.lit("young"))
        .when(age <= 12)
        .then(pl.lit("mature"))
        .otherwise(pl.lit("stale"))
    )


def realized_transition_frame(
    kline_history: pl.DataFrame, horizons: tuple[int, ...]
) -> pl.DataFrame:
    if kline_history.is_empty() or not horizons:
        return pl.DataFrame(schema=REALIZED_TRANSITION_SCHEMA)
    frames = []
    base = kline_history.sort("symbol", "timeframe", "bar_close_ms")
    for horizon in sorted({int(horizon) for horizon in horizons if int(horizon) > 0}):
        direction_change_time = pl.min_horizontal(
            *(
                pl.when(
                    pl.col("direction_hint").shift(-offset).over("symbol", "timeframe")
                    != pl.col("direction_hint")
                )
                .then(offset)
                .otherwise(None)
                for offset in range(1, horizon + 1)
            )
        )
        core_change_time = pl.min_horizontal(
            *(
                pl.when(
                    pl.col("core_context").shift(-offset).over("symbol", "timeframe")
                    != pl.col("core_context")
                )
                .then(offset)
                .otherwise(None)
                for offset in range(1, horizon + 1)
            )
        )
        transition_count = sum(
            (
                pl.col("core_context").shift(-offset).over("symbol", "timeframe")
                != pl.col("core_context").shift(-(offset - 1)).over("symbol", "timeframe")
            ).cast(pl.Int64)
            for offset in range(1, horizon + 1)
        )
        frames.append(
            base.with_columns(
                pl.lit(horizon).alias("outcome_horizon"),
                pl.col("direction_hint")
                .shift(-horizon)
                .over("symbol", "timeframe")
                .alias("terminal_direction"),
                pl.col("regime_state")
                .shift(-horizon)
                .over("symbol", "timeframe")
                .alias("terminal_regime_state"),
                pl.col("structure_state")
                .shift(-horizon)
                .over("symbol", "timeframe")
                .alias("terminal_structure_state"),
                pl.col("core_context")
                .shift(-horizon)
                .over("symbol", "timeframe")
                .alias("terminal_core_context"),
                pl.col("transition_kind")
                .shift(-horizon)
                .over("symbol", "timeframe")
                .alias("terminal_transition_kind"),
                direction_change_time.alias("time_to_direction_change_bars"),
                core_change_time.alias("time_to_core_change_bars"),
                transition_count.alias("transition_count"),
            )
            .with_columns(
                (pl.col("terminal_direction") != pl.col("direction_hint")).alias(
                    "direction_changed"
                ),
                (pl.col("terminal_regime_state") != pl.col("regime_state")).alias("regime_changed"),
                (pl.col("terminal_structure_state") != pl.col("structure_state")).alias(
                    "structure_changed"
                ),
                (pl.col("terminal_core_context") != pl.col("core_context")).alias(
                    "core_context_changed"
                ),
                pl.concat_list(
                    [
                        ~pl.col("event_state")
                        .shift(-offset)
                        .over("symbol", "timeframe")
                        .str.starts_with("none")
                        for offset in range(1, horizon + 1)
                    ]
                )
                .list.any()
                .alias("event_fired"),
                (
                    (pl.col("terminal_core_context") == pl.col("core_context"))
                    & (pl.col("time_to_core_change_bars").is_not_null())
                ).alias("returned_to_origin"),
            )
            .select(*REALIZED_TRANSITION_SCHEMA.keys())
        )
    if not frames:
        return pl.DataFrame(schema=REALIZED_TRANSITION_SCHEMA)
    return pl.concat(frames, how="vertical_relaxed").select(*REALIZED_TRANSITION_SCHEMA.keys())
