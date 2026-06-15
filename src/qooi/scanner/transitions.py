"""Known-at-close transition discovery."""

from __future__ import annotations

from typing import cast

import polars as pl

from qooi.scanner import (
    PotentialScanConfig,
    SourceStateRow,
    StateDirection,
    TransitionAnalysis,
    TransitionEdge,
    TransitionInsight,
    TransitionPattern,
    UnsupportedTransitionPath,
    missing_state,
)


def compute_transition_insights(
    config: PotentialScanConfig,
    symbols: tuple[str, ...],
    frames: dict[tuple[str, str], pl.DataFrame],
    state_frames: dict[tuple[str, str], pl.DataFrame],
) -> TransitionAnalysis:
    insights = {
        symbol: TransitionInsight(
            symbol,
            missing_state(symbol, "transition", "transition_pattern_missing"),
            (),
        )
        for symbol in symbols
    }
    edges: list[TransitionEdge] = []
    unsupported: list[UnsupportedTransitionPath] = []
    symbol_index = {symbol: index for index, symbol in enumerate(symbols)}
    for timeframe in config.timeframes:
        classified = _classified_symbol_frame(config, symbols, frames, state_frames, timeframe)
        if classified.is_empty():
            continue
        work = _transition_work_frame(classified, config)
        rows = work.filter(
            pl.col("prev_state").is_not_null()
            & pl.col("transition_path").is_not_null()
            & pl.col("forward_return_pct").is_not_null()
        )
        if rows.is_empty():
            continue
        transition_counts = _transition_counts(rows)
        rows_sorted = rows.sort("symbol", "timestamp")
        long_rows = rows_sorted.group_by("symbol").tail(
            max(config.transition.long_window, config.transition.min_count)
        )
        recent_rows = rows_sorted.group_by("symbol").tail(
            max(config.transition.recent_window, config.transition.min_count)
        )
        long_counts = _transition_counts(long_rows).rename(
            {"transition_probability": "long_transition_probability"}
        )
        recent_counts = _transition_counts(recent_rows).rename(
            {"transition_probability": "recent_transition_probability"}
        )
        event_information = _transition_information_frame(rows)
        edges.extend(transition_edges(timeframe, transition_counts, event_information))
        pattern_frame = _pattern_frame(rows, config)
        if pattern_frame.is_empty():
            current = (
                work.filter(
                    pl.col("transition_path").is_not_null()
                    & pl.col("contextual_event").is_not_null()
                )
                .sort("timestamp")
                .group_by("symbol")
                .tail(1)
            )
            unsupported.extend(
                UnsupportedTransitionPath(
                    str(row["symbol"]),
                    timeframe,
                    str(row["transition_path"]),
                    str(row["contextual_event"]),
                    "no_supported_patterns_for_timeframe",
                )
                for row in current.iter_rows(named=True)
            )
            continue
        patterns = (
            pattern_frame.join(
                transition_counts,
                on=("prev_state", "state_key"),
                how="left",
            )
            .join(
                long_counts.select("prev_state", "state_key", "long_transition_probability"),
                on=("prev_state", "state_key"),
                how="left",
            )
            .join(
                recent_counts.select("prev_state", "state_key", "recent_transition_probability"),
                on=("prev_state", "state_key"),
                how="left",
            )
            .join(
                event_information.select(
                    "prev_state",
                    "state_key",
                    "contextual_event",
                    "transition_information_bits",
                    "conditional_transition_information_bits",
                ),
                on=("prev_state", "state_key", "contextual_event"),
                how="left",
            )
            .with_columns(
                (
                    pl.col("recent_transition_probability").fill_null(0.0)
                    - pl.col("long_transition_probability").fill_null(0.0)
                ).alias("probability_delta")
            )
        )
        current = (
            work.filter(
                pl.col("transition_path").is_not_null() & pl.col("contextual_event").is_not_null()
            )
            .sort("timestamp")
            .group_by("symbol")
            .tail(1)
            .select("symbol", "timestamp", "transition_path", "contextual_event")
            .join(patterns, on=("transition_path", "contextual_event"), how="left")
            .with_columns(
                pl.when(pl.col("direction") == "bullish")
                .then(pl.col("p_up"))
                .when(pl.col("direction") == "bearish")
                .then(pl.col("p_down"))
                .otherwise(0.0)
                .alias("directional_probability"),
                pl.max_horizontal(
                    pl.col("transition_information_bits").fill_null(0.0),
                    pl.col("conditional_transition_information_bits").fill_null(0.0),
                ).alias("information_bits"),
            )
            .with_columns(
                (
                    pl.col("count").is_not_null()
                    & (pl.col("count") >= config.transition.min_count)
                    & (
                        pl.col("transition_probability").fill_null(0.0)
                        >= config.transition.min_probability
                    )
                    & (
                        pl.col("recent_transition_probability").fill_null(0.0)
                        >= config.transition.min_probability
                    )
                    & (
                        pl.col("probability_delta").fill_null(-1.0)
                        >= config.transition.min_probability_delta
                    )
                    & (
                        pl.col("directional_probability").fill_null(0.0)
                        >= config.transition.min_directional_probability
                    )
                    & (pl.col("reward_risk").fill_null(0.0) >= config.transition.min_reward_risk)
                    & (
                        pl.col("loss_stop_pct").fill_null(float("inf"))
                        <= config.transition.max_tail_loss_pct
                    )
                    & (pl.col("information_bits") >= config.transition.min_information_bits)
                    & pl.col("direction").is_in(["bullish", "bearish"])
                ).alias("gate_pass")
            )
        )
        unsupported.extend(
            UnsupportedTransitionPath(
                str(row["symbol"]),
                timeframe,
                str(row["transition_path"]),
                str(row["contextual_event"]),
                "current_path_below_support_or_quality"
                if row["count"] is None
                else "current_path_below_probability_or_outcome_quality",
            )
            for row in current.filter(~pl.col("gate_pass")).iter_rows(named=True)
        )
        for row in current.filter(pl.col("gate_pass")).iter_rows(named=True):
            pattern = TransitionPattern(
                symbol=str(row["symbol"]),
                timeframe=timeframe,
                path=str(row["transition_path"]),
                event=str(row["contextual_event"]),
                count=int(row["count"]),
                transition_probability=float(row["transition_probability"] or 0.0),
                win_rate=float(row["win_rate"] or 0.0),
                average_forward_return_pct=float(row["average_forward_return_pct"] or 0.0),
                omega=float(row["omega"] or 0.0),
                pwpr=float(row["pwpr"] or 0.0),
                transition_information_bits=float(row["transition_information_bits"] or 0.0),
                conditional_transition_information_bits=float(
                    row["conditional_transition_information_bits"] or 0.0
                ),
                direction=cast(StateDirection, str(row["direction"])),
                recent_transition_probability=float(row["recent_transition_probability"] or 0.0),
                long_transition_probability=float(row["long_transition_probability"] or 0.0),
                probability_delta=float(row["probability_delta"] or 0.0),
                p_up=float(row["p_up"] or 0.0),
                p_down=float(row["p_down"] or 0.0),
                median_forward_return_pct=float(row["median_forward_return_pct"] or 0.0),
                q10_forward_return_pct=float(row["q10_forward_return_pct"] or 0.0),
                q25_forward_return_pct=float(row["q25_forward_return_pct"] or 0.0),
                q75_forward_return_pct=float(row["q75_forward_return_pct"] or 0.0),
                q90_forward_return_pct=float(row["q90_forward_return_pct"] or 0.0),
                q25_forward_min_return_pct=float(row["q25_forward_min_return_pct"] or 0.0),
                q75_forward_max_return_pct=float(row["q75_forward_max_return_pct"] or 0.0),
                loss_stop_pct=float(row["loss_stop_pct"] or 0.0),
                profit_stop_pct=float(row["profit_stop_pct"] or 0.0),
                reward_risk=float(row["reward_risk"] or 0.0),
                symbol_count=int(row["symbol_count"] or 0),
                effective_count=float(row["effective_count"] or 0.0),
                suggestion=str(row["suggestion"] or "watch"),
            )
            previous = insights[pattern.symbol]
            current_state = previous.current
            if (
                current_state.direction == "missing"
                or pattern.transition_probability > current_state.confidence
            ):
                current_state = SourceStateRow(
                    symbol=pattern.symbol,
                    family="transition",
                    timestamp=int(row["timestamp"]) if row["timestamp"] is not None else None,
                    state=pattern.path,
                    direction=pattern.direction,
                    confidence=min(0.95, max(0.05, pattern.transition_probability)),
                    evidence=_transition_evidence_text(pattern, ""),
                    missing_reason="",
                    stale=False,
                )
            insights[pattern.symbol] = transition_insight(
                pattern.symbol, current_state, (*previous.patterns, pattern)
            )
    insights = {
        symbol: transition_insight(symbol, insight.current, insight.patterns)
        for symbol, insight in sorted(insights.items(), key=lambda item: symbol_index[item[0]])
    }
    return TransitionAnalysis(insights, tuple(edges), tuple(unsupported))


def _classified_symbol_frame(
    config: PotentialScanConfig,
    symbols: tuple[str, ...],
    frames: dict[tuple[str, str], pl.DataFrame],
    state_frames: dict[tuple[str, str], pl.DataFrame],
    timeframe: str,
) -> pl.DataFrame:
    symbol_frames = []
    for symbol in symbols:
        frame = frames.get((symbol, timeframe), pl.DataFrame())
        states = state_frames.get((symbol, timeframe), pl.DataFrame())
        if (
            frame.is_empty()
            or states.is_empty()
            or frame.height < config.transition.horizon + config.transition.ngram_length + 1
        ):
            continue
        if "volume" not in frame.columns and "vol" in frame.columns:
            frame = frame.with_columns(pl.col("vol").alias("volume"))
        if "volume" not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=pl.Float64).alias("volume"))
        symbol_frames.append(
            frame.with_columns(pl.lit(symbol).alias("symbol"), pl.lit(timeframe).alias("timeframe"))
            .select("symbol", "timeframe", "timestamp", "open", "high", "low", "close", "volume")
            .join(
                states.select(
                    "symbol",
                    "timestamp",
                    "state_key",
                    pl.col("context_event").alias("contextual_event"),
                ),
                on=("symbol", "timestamp"),
                how="inner",
            )
        )
    if not symbol_frames:
        return pl.DataFrame()
    return pl.concat(symbol_frames, how="vertical_relaxed").sort("symbol", "timestamp")


def _transition_work_frame(classified: pl.DataFrame, config: PotentialScanConfig) -> pl.DataFrame:
    horizon = max(1, config.transition.mae_mfe_horizon)
    ngram_length = max(2, config.transition.ngram_length)
    forward_max_high = pl.max_horizontal(
        *(pl.col("high").shift(-offset).over("symbol") for offset in range(1, horizon + 1))
    )
    forward_min_low = pl.min_horizontal(
        *(pl.col("low").shift(-offset).over("symbol") for offset in range(1, horizon + 1))
    )
    return classified.with_columns(
        pl.col("state_key").cast(pl.String),
        pl.col("contextual_event").cast(pl.String),
        pl.col("state_key").shift(1).over("symbol").alias("prev_state"),
        pl.concat_str(
            [
                pl.col("state_key").shift(offset).over("symbol")
                for offset in range(ngram_length - 1, 0, -1)
            ]
            + [pl.col("state_key")],
            separator=" -> ",
        ).alias("transition_path"),
        (
            (
                pl.col("close").shift(-config.transition.horizon).over("symbol") / pl.col("close")
                - 1.0
            )
            * 100.0
        ).alias("forward_return_pct"),
        ((forward_max_high / pl.col("close") - 1.0) * 100.0).alias("forward_max_return_pct"),
        ((forward_min_low / pl.col("close") - 1.0) * 100.0).alias("forward_min_return_pct"),
    )


def _transition_counts(rows: pl.DataFrame) -> pl.DataFrame:
    return (
        rows.group_by("prev_state", "state_key")
        .len()
        .with_columns(
            (pl.col("len") / pl.col("len").sum().over("prev_state")).alias("transition_probability")
        )
    )


def _pattern_frame(rows: pl.DataFrame, config: PotentialScanConfig) -> pl.DataFrame:
    return (
        rows.group_by("transition_path", "contextual_event", "prev_state", "state_key")
        .agg(
            pl.len().alias("count"),
            pl.col("symbol").n_unique().alias("symbol_count"),
            (pl.col("forward_return_pct") > config.transition.return_threshold_pct)
            .mean()
            .alias("win_rate"),
            (pl.col("forward_return_pct") > config.transition.return_threshold_pct)
            .mean()
            .alias("p_up"),
            (pl.col("forward_return_pct") < -config.transition.return_threshold_pct)
            .mean()
            .alias("p_down"),
            pl.col("forward_return_pct").mean().alias("average_forward_return_pct"),
            pl.col("forward_return_pct").median().alias("median_forward_return_pct"),
            pl.col("forward_return_pct").quantile(0.10).alias("q10_forward_return_pct"),
            pl.col("forward_return_pct").quantile(0.25).alias("q25_forward_return_pct"),
            pl.col("forward_return_pct").quantile(0.75).alias("q75_forward_return_pct"),
            pl.col("forward_return_pct").quantile(0.90).alias("q90_forward_return_pct"),
            pl.col("forward_min_return_pct").quantile(0.25).alias("q25_forward_min_return_pct"),
            pl.col("forward_max_return_pct").quantile(0.75).alias("q75_forward_max_return_pct"),
            pl.col("forward_return_pct")
            .filter(pl.col("forward_return_pct") > 0.0)
            .sum()
            .alias("gains"),
            (-pl.col("forward_return_pct"))
            .filter(pl.col("forward_return_pct") < 0.0)
            .sum()
            .alias("losses"),
        )
        .filter(pl.col("count") >= config.transition.min_count)
        .with_columns(
            (pl.col("count") / max(1, config.transition.horizon)).alias("effective_count"),
            (
                pl.col("gains")
                / pl.when(pl.col("losses") > 0.0).then(pl.col("losses")).otherwise(1.0)
            ).alias("omega"),
        )
        .with_columns(
            ((pl.col("win_rate") * pl.col("omega")) / (1.0 + pl.col("omega"))).alias("pwpr"),
            pl.when(pl.col("average_forward_return_pct") > config.transition.return_threshold_pct)
            .then(pl.lit("bullish"))
            .when(pl.col("average_forward_return_pct") < -config.transition.return_threshold_pct)
            .then(pl.lit("bearish"))
            .otherwise(pl.lit("neutral"))
            .alias("direction"),
        )
        .with_columns(
            pl.when(pl.col("direction") == "bullish")
            .then((-pl.col("q25_forward_min_return_pct")).clip(0.0))
            .when(pl.col("direction") == "bearish")
            .then(pl.col("q75_forward_max_return_pct").clip(0.0))
            .otherwise(0.0)
            .alias("loss_stop_pct"),
            pl.when(pl.col("direction") == "bullish")
            .then(pl.col("q75_forward_max_return_pct").clip(0.0))
            .when(pl.col("direction") == "bearish")
            .then((-pl.col("q25_forward_min_return_pct")).clip(0.0))
            .otherwise(0.0)
            .alias("profit_stop_pct"),
        )
        .with_columns(
            (
                pl.col("profit_stop_pct")
                / pl.when(pl.col("loss_stop_pct") > 0.0)
                .then(pl.col("loss_stop_pct"))
                .otherwise(None)
            )
            .fill_null(0.0)
            .alias("reward_risk"),
        )
        .with_columns(
            pl.when(pl.col("reward_risk") >= config.transition.min_reward_risk)
            .then(pl.lit("rapid_trend_watch"))
            .when(pl.col("p_up") + pl.col("p_down") >= 0.75)
            .then(pl.lit("volatility_expansion_watch"))
            .otherwise(pl.lit("insufficient_evidence"))
            .alias("suggestion"),
        )
        .sort(["pwpr", "count"], descending=[True, True])
    )


def transition_edges(
    timeframe: str, transition_counts: pl.DataFrame, information: pl.DataFrame
) -> tuple[TransitionEdge, ...]:
    if transition_counts.is_empty():
        return ()
    joined = transition_counts.join(information, on=("prev_state", "state_key"), how="left")
    rows = []
    for row in joined.iter_rows(named=True):
        rows.append(
            TransitionEdge(
                timeframe=timeframe,
                prev_state=str(row["prev_state"]),
                state=str(row["state_key"]),
                event=str(row["contextual_event"]),
                count=int(row["len"]),
                transition_probability=float(row["transition_probability"]),
                transition_information_bits=float(row["transition_information_bits"] or 0.0),
                conditional_transition_information_bits=float(
                    row["conditional_transition_information_bits"] or 0.0
                ),
            )
        )
    return tuple(rows)


def transition_insight(
    symbol: str, current: SourceStateRow, patterns: tuple[TransitionPattern, ...]
) -> TransitionInsight:
    directional = tuple(
        pattern for pattern in patterns if pattern.direction in {"bullish", "bearish"}
    )
    if not directional:
        return TransitionInsight(symbol, current, patterns, "none", 0.0)
    bullish = sum(1 for pattern in directional if pattern.direction == "bullish")
    bearish = sum(1 for pattern in directional if pattern.direction == "bearish")
    consensus = "bullish" if bullish > bearish else "bearish" if bearish > bullish else "mixed"
    rank_score = max(
        pattern.transition_probability
        + pattern.transition_information_bits
        + pattern.conditional_transition_information_bits
        for pattern in directional
    )
    base_evidence = current.evidence.split("; consensus=", 1)[0]
    current = (
        current
        if current.direction == "missing"
        else SourceStateRow(
            current.symbol,
            current.family,
            current.timestamp,
            current.state,
            current.direction,
            current.confidence,
            f"{base_evidence}; consensus={consensus}",
            current.missing_reason,
            current.stale,
        )
    )
    return TransitionInsight(symbol, current, patterns, consensus, rank_score)


def _transition_evidence_text(pattern: TransitionPattern, consensus: str) -> str:
    text = (
        f"timeframe={pattern.timeframe}; path={pattern.path}; event={pattern.event}; "
        f"p={pattern.transition_probability:.3f}; "
        f"p_recent={pattern.recent_transition_probability:.3f}; "
        f"p_long={pattern.long_transition_probability:.3f}; "
        f"p_delta={pattern.probability_delta:.3f}; "
        f"p_up={pattern.p_up:.3f}; p_down={pattern.p_down:.3f}; "
        f"expected_return={pattern.average_forward_return_pct:.3f}; "
        f"median_return={pattern.median_forward_return_pct:.3f}; "
        f"loss_stop={pattern.loss_stop_pct:.3f}; profit_stop={pattern.profit_stop_pct:.3f}; "
        f"rr={pattern.reward_risk:.3f}; "
        f"win_rate={pattern.win_rate:.3f}; omega={pattern.omega:.3f}; pwpr={pattern.pwpr:.3f}; "
        f"mi={pattern.transition_information_bits:.3f}; "
        f"cmi={pattern.conditional_transition_information_bits:.3f}; "
        f"suggestion={pattern.suggestion}"
    )
    return f"{text}; consensus={consensus}" if consensus else text


def _transition_information_frame(rows: pl.DataFrame) -> pl.DataFrame:
    total = rows.height
    if total == 0:
        return pl.DataFrame()
    return (
        rows.group_by("prev_state", "state_key", "contextual_event")
        .len()
        .with_columns(
            pl.col("len").sum().over("prev_state", "state_key").alias("pair_count"),
            pl.col("len").sum().over("prev_state").alias("prev_count"),
            pl.col("len").sum().over("state_key").alias("state_count"),
            pl.col("len").sum().over("contextual_event").alias("event_count"),
            pl.col("len").sum().over("prev_state", "contextual_event").alias("prev_event_count"),
            pl.col("len").sum().over("state_key", "contextual_event").alias("state_event_count"),
        )
        .with_columns(
            (pl.col("pair_count") / total).alias("p_pair"),
            (pl.col("prev_count") / total).alias("p_prev"),
            (pl.col("state_count") / total).alias("p_state"),
            (pl.col("event_count") / total).alias("p_event"),
            (pl.col("len") / pl.col("event_count")).alias("p_pair_given_event"),
            (pl.col("prev_event_count") / pl.col("event_count")).alias("p_prev_given_event"),
            (pl.col("state_event_count") / pl.col("event_count")).alias("p_state_given_event"),
        )
        .with_columns(
            (
                pl.col("p_pair")
                * (pl.col("p_pair") / (pl.col("p_prev") * pl.col("p_state"))).log(2)
            ).alias("transition_information_bits"),
            (
                pl.col("p_event")
                * pl.col("p_pair_given_event")
                * (
                    pl.col("p_pair_given_event")
                    / (pl.col("p_prev_given_event") * pl.col("p_state_given_event"))
                ).log(2)
            ).alias("conditional_transition_information_bits"),
        )
    )
