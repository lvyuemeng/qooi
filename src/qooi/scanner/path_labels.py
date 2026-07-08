"""Path-prototype labels and legacy tail policy helpers."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from qooi.scanner.config import ExtremeTailConfig


@dataclass(frozen=True)
class PathLabelSpec:
    """Path-prototype label thresholds and class/timing weights."""

    smooth_return_pct: float = 3.0
    smooth_adverse_tolerance_pct: float = 1.5
    chop_up_pct: float = 4.0
    chop_down_pct: float = 4.0
    fake_breakout_pct: float = 3.0
    timing_lambda: float = 0.1
    calm_weight: float = 0.5
    smooth_up_weight: float = 1.0
    smooth_down_weight: float = 1.0
    chop_weight: float = 1.5
    fake_breakout_weight: float = 0.1
    trend_weight_floor: float = 0.05
    trend_weight_cap: float = 5.0


_PATH_LABEL_NAMES = {
    0: "calm",
    1: "smooth_up",
    2: "smooth_down",
    3: "chop",
    4: "fake_breakout",
}


def _empty_path_labels() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "symbol": pl.String,
            "decision_bar_close_ms": pl.Int64,
            "horizon_hours": pl.Int64,
            "path_label": pl.Int64,
            "path_label_name": pl.String,
            "sample_weight": pl.Float64,
            "trend_cleanliness": pl.Float64,
            "risk_adjusted_path_weight": pl.Float64,
            "first_touch_hours": pl.Float64,
            "final_return": pl.Float64,
            "max_up_return": pl.Float64,
            "max_down_return": pl.Float64,
            "first_touch_side": pl.String,
            "path_reason": pl.String,
        }
    )


def make_path_labels(
    outcomes: pl.DataFrame,
    horizons: tuple[int, ...],
    spec: PathLabelSpec,
) -> pl.DataFrame:
    """Build multiclass path-prototype labels at symbol × decision × horizon grain."""
    if outcomes.is_empty():
        return _empty_path_labels()
    required = {
        "symbol",
        "decision_bar_close_ms",
        "outcome_horizon",
        "forward_return_pct",
        "forward_max_return_pct",
        "forward_min_return_pct",
    }
    missing = sorted(required - set(outcomes.columns))
    if missing:
        raise ValueError(f"make_path_labels requires outcome columns: {', '.join(missing)}")
    horizon_values = [int(horizon) for horizon in horizons]
    if not horizon_values:
        return _empty_path_labels()
    frame = outcomes.filter(pl.col("outcome_horizon").is_in(horizon_values))
    if frame.is_empty():
        return _empty_path_labels()
    duplicate_count = (
        frame.group_by("symbol", "decision_bar_close_ms", "outcome_horizon")
        .agg(pl.len().alias("n"))
        .filter(pl.col("n") > 1)
        .height
    )
    if duplicate_count:
        raise ValueError(
            "make_path_labels requires one outcome row per symbol × decision × horizon; "
            "call path_label_outcome_frame first"
        )
    time_to_max = (
        pl.col("time_to_max_bar").cast(pl.Float64)
        if "time_to_max_bar" in frame.columns
        else pl.lit(None, dtype=pl.Float64)
    )
    time_to_min = (
        pl.col("time_to_min_bar").cast(pl.Float64)
        if "time_to_min_bar" in frame.columns
        else pl.lit(None, dtype=pl.Float64)
    )
    final_return = pl.col("forward_return_pct").cast(pl.Float64).fill_null(0.0)
    max_up = pl.col("forward_max_return_pct").cast(pl.Float64).fill_null(0.0)
    max_down = pl.col("forward_min_return_pct").cast(pl.Float64).fill_null(0.0)
    up_adverse = (-max_down).clip(0.0, None)
    down_adverse = max_up.clip(0.0, None)
    up_cleanliness = ((max_up - up_adverse).clip(0.0, None) / (1.0 + up_adverse)).fill_nan(0.0)
    down_cleanliness = (
        ((-max_down) - down_adverse).clip(0.0, None) / (1.0 + down_adverse)
    ).fill_nan(0.0)
    up_touch = max_up >= float(spec.smooth_return_pct)
    down_touch = max_down <= -float(spec.smooth_return_pct)
    max_time = time_to_max.fill_null(float("inf"))
    min_time = time_to_min.fill_null(float("inf"))
    first_up = up_touch & (~down_touch | (max_time <= min_time))
    first_down = down_touch & (~up_touch | (min_time < max_time))
    first_touch_side = (
        pl.when(first_up)
        .then(pl.lit("up"))
        .when(first_down)
        .then(pl.lit("down"))
        .when(up_touch & down_touch)
        .then(pl.lit("tie"))
        .otherwise(pl.lit("none"))
    )
    first_touch_hours = (
        pl.when(first_up)
        .then(time_to_max)
        .when(first_down)
        .then(time_to_min)
        .otherwise(pl.lit(None, dtype=pl.Float64))
    )
    fake_breakout = (
        (final_return > 0.0) & first_down & (max_down <= -float(spec.fake_breakout_pct))
    ) | ((final_return < 0.0) & first_up & (max_up >= float(spec.fake_breakout_pct)))
    chop = (max_up >= float(spec.chop_up_pct)) & (max_down <= -float(spec.chop_down_pct))
    smooth_up = (
        (final_return >= float(spec.smooth_return_pct))
        & (max_down > -float(spec.smooth_adverse_tolerance_pct))
        & ~first_down
    )
    smooth_down = (
        (final_return <= -float(spec.smooth_return_pct))
        & (max_up < float(spec.smooth_adverse_tolerance_pct))
        & ~first_up
    )
    path_label = (
        pl.when(fake_breakout)
        .then(pl.lit(4))
        .when(chop)
        .then(pl.lit(3))
        .when(smooth_up)
        .then(pl.lit(1))
        .when(smooth_down)
        .then(pl.lit(2))
        .otherwise(pl.lit(0))
    )
    path_reason = (
        pl.when(fake_breakout)
        .then(pl.lit("opposite_first_reversal"))
        .when(chop)
        .then(pl.lit("both_side_excursion"))
        .when(smooth_up)
        .then(pl.lit("smooth_up_path"))
        .when(smooth_down)
        .then(pl.lit("smooth_down_path"))
        .otherwise(pl.lit("no_path_prototype"))
    )
    return (
        frame.with_columns(
            pl.col("outcome_horizon").cast(pl.Int64).alias("horizon_hours"),
            final_return.alias("final_return"),
            max_up.alias("max_up_return"),
            max_down.alias("max_down_return"),
            first_touch_side.alias("first_touch_side"),
            first_touch_hours.alias("first_touch_hours"),
            path_label.cast(pl.Int64).alias("path_label"),
            path_reason.alias("path_reason"),
        )
        .with_columns(
            pl.col("path_label")
            .replace_strict(_PATH_LABEL_NAMES, return_dtype=pl.String)
            .alias("path_label_name"),
            pl.when(pl.col("path_label") == 1)
            .then(up_cleanliness)
            .when(pl.col("path_label") == 2)
            .then(down_cleanliness)
            .otherwise(pl.lit(0.0))
            .clip(float(spec.trend_weight_floor), float(spec.trend_weight_cap))
            .alias("risk_adjusted_path_weight"),
            pl.when(pl.col("path_label") == 1)
            .then(up_cleanliness)
            .when(pl.col("path_label") == 2)
            .then(down_cleanliness)
            .otherwise(pl.lit(0.0))
            .alias("trend_cleanliness"),
        )
        .with_columns(
            pl.when(pl.col("path_label") == 1)
            .then(
                pl.lit(float(spec.smooth_up_weight))
                * (-pl.lit(float(spec.timing_lambda)) * pl.col("first_touch_hours")).exp()
                * pl.col("risk_adjusted_path_weight")
            )
            .when(pl.col("path_label") == 2)
            .then(
                pl.lit(float(spec.smooth_down_weight))
                * (-pl.lit(float(spec.timing_lambda)) * pl.col("first_touch_hours")).exp()
                * pl.col("risk_adjusted_path_weight")
            )
            .when(pl.col("path_label") == 3)
            .then(pl.lit(float(spec.chop_weight)))
            .when(pl.col("path_label") == 4)
            .then(pl.lit(float(spec.fake_breakout_weight)))
            .otherwise(pl.lit(float(spec.calm_weight)))
            .alias("sample_weight"),
        )
        .select(
            "symbol",
            "decision_bar_close_ms",
            "horizon_hours",
            "path_label",
            "path_label_name",
            "sample_weight",
            "trend_cleanliness",
            "risk_adjusted_path_weight",
            "first_touch_hours",
            "final_return",
            "max_up_return",
            "max_down_return",
            "first_touch_side",
            "path_reason",
        )
    )


def path_label_outcome_frame(outcomes: pl.DataFrame, horizons: tuple[int, ...]) -> pl.DataFrame:
    """Select the minimal outcome contract consumed by make_path_labels."""
    if outcomes.is_empty():
        return outcomes
    frame = outcomes.filter(pl.col("outcome_horizon").is_in([int(h) for h in horizons]))
    if frame.is_empty():
        return frame
    if "source_family" not in frame.columns:
        return frame.unique(
            subset=["symbol", "decision_bar_close_ms", "outcome_horizon"],
            keep="first",
            maintain_order=True,
        )
    return (
        frame.with_columns(pl.col("source_family").is_null().cast(pl.Int8).alias("_market_outcome"))
        .sort(
            ["symbol", "decision_bar_close_ms", "outcome_horizon", "_market_outcome"],
            descending=[False, False, False, True],
        )
        .unique(
            subset=["symbol", "decision_bar_close_ms", "outcome_horizon"],
            keep="first",
            maintain_order=True,
        )
        .drop("_market_outcome")
    )


@dataclass(frozen=True)
class TailEventPolicy:
    """Fold-local tail-event policy with reference fitting and label application."""

    extreme: ExtremeTailConfig

    def reference_frame(self, outcomes: pl.DataFrame) -> pl.DataFrame:
        """Fit side/horizon reference rows from outcome-known training rows."""
        if outcomes.is_empty() or "outcome_horizon" not in outcomes.columns:
            return _empty_reference_frame()
        side_frames: list[pl.DataFrame] = []
        if "forward_max_return_pct" in outcomes.columns:
            side_frames.append(
                outcomes.select(
                    "outcome_horizon",
                    pl.lit("up").alias("side"),
                    pl.col("forward_max_return_pct")
                    .cast(pl.Float64)
                    .fill_null(0.0)
                    .clip(0.0, None)
                    .alias("tail_depth_pct"),
                )
            )
        if "forward_min_return_pct" in outcomes.columns:
            side_frames.append(
                outcomes.select(
                    "outcome_horizon",
                    pl.lit("down").alias("side"),
                    pl.col("forward_min_return_pct")
                    .cast(pl.Float64)
                    .abs()
                    .fill_null(0.0)
                    .alias("tail_depth_pct"),
                )
            )
        if not side_frames:
            return _empty_reference_frame()
        depths = pl.concat(side_frames, how="diagonal_relaxed")
        material_floor = float(self.extreme.material_floor_pct)
        quantile = float(self.extreme.quantile)
        reference = depths.group_by("outcome_horizon", "side").agg(
            pl.len().alias("sample_count"),
            pl.lit("universe_horizon").alias("reference_scope"),
            pl.lit(self.extreme.method).alias("policy_method"),
            pl.lit(material_floor).alias("material_floor_pct"),
            pl.lit(quantile).alias("quantile"),
            pl.lit(float(self.extreme.min_event_rate)).alias("min_event_rate"),
            pl.lit(float(self.extreme.max_event_rate)).alias("max_event_rate"),
            pl.col("tail_depth_pct")
            .quantile(quantile, interpolation="nearest")
            .alias("quantile_depth_pct"),
            pl.col("tail_depth_pct").sort().alias("reference_depths"),
        )
        if self.extreme.method == "fixed_pct":
            threshold = pl.lit(material_floor)
        elif self.extreme.method == "empirical_quantile":
            threshold = pl.col("quantile_depth_pct")
        else:
            threshold = pl.max_horizontal(pl.lit(material_floor), pl.col("quantile_depth_pct"))
        return reference.with_columns(threshold.alias("event_depth_threshold_pct")).sort(
            "outcome_horizon", "side"
        )

    def label_paths(
        self,
        outcome_frame: pl.DataFrame,
        reference: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Label tail magnitude, relative event policy, path behavior, and utility."""
        if outcome_frame.is_empty():
            return outcome_frame
        tail_reference = reference if reference is not None else self.reference_frame(outcome_frame)
        labeled = self._with_depth_and_rank(outcome_frame, tail_reference)
        if "tail_event_up" not in labeled.columns or "tail_event_down" not in labeled.columns:
            return labeled
        return self._with_path_behavior(labeled)

    def behavior_target_frame(
        self, labeled_outcomes: pl.DataFrame, *, direction: str
    ) -> pl.DataFrame:
        """Build the strict first behavior target from labeled path outcomes."""
        if direction != "up":
            raise ValueError("behavior target clean_up_actionable is only defined for up")
        schema = {
            "symbol": pl.String,
            "decision_bar_close_ms": pl.Int64,
            "outcome_horizon": pl.Int64,
            "behavior_side": pl.String,
            "behavior_target": pl.String,
            "behavior_actionable": pl.Boolean,
            "behavior_false_direction": pl.Boolean,
            "behavior_utility_margin": pl.Float64,
            "behavior_path_state": pl.String,
            "behavior_actionability": pl.String,
            "behavior_blocker": pl.String,
        }
        if labeled_outcomes.is_empty():
            return pl.DataFrame(schema=schema)
        required = {
            "symbol",
            "decision_bar_close_ms",
            "outcome_horizon",
            "tail_touch_up",
            "tail_touch_down",
            "first_touch_side",
            "path_state",
            "path_actionability",
            "path_blocker",
            "path_utility_margin_up",
        }
        missing = sorted(required - set(labeled_outcomes.columns))
        if missing:
            raise ValueError(f"behavior target requires labeled path columns: {', '.join(missing)}")
        clean_up_actionable = (
            pl.col("tail_touch_up").fill_null(False).cast(pl.Boolean)
            & ~pl.col("tail_touch_down").fill_null(False).cast(pl.Boolean)
            & (pl.col("first_touch_side") == "up")
            & (pl.col("path_state") == "clean_up")
            & (pl.col("path_actionability") == "tradable_up")
            & (pl.col("path_utility_margin_up").fill_null(0.0).cast(pl.Float64) > 0.0)
        )
        false_direction = (pl.col("first_touch_side") == "down") | (
            pl.col("path_state") == "clean_down"
        )
        blocker = (
            pl.when(pl.col("path_state") == "clean_down")
            .then(pl.lit("opposite_clean_path"))
            .otherwise(pl.col("path_blocker").fill_null(""))
        )
        return labeled_outcomes.select(
            "symbol",
            "decision_bar_close_ms",
            "outcome_horizon",
            pl.lit("up").alias("behavior_side"),
            pl.lit("clean_up_actionable").alias("behavior_target"),
            clean_up_actionable.alias("behavior_actionable"),
            false_direction.alias("behavior_false_direction"),
            pl.col("path_utility_margin_up")
            .fill_null(0.0)
            .cast(pl.Float64)
            .alias("behavior_utility_margin"),
            pl.col("path_state").alias("behavior_path_state"),
            pl.col("path_actionability").alias("behavior_actionability"),
            blocker.alias("behavior_blocker"),
        )

    def _with_depth_and_rank(
        self, outcome_frame: pl.DataFrame, reference: pl.DataFrame
    ) -> pl.DataFrame:
        labeled = outcome_frame.with_row_index("_tail_row_id")
        if "forward_max_return_pct" in labeled.columns:
            labeled = labeled.with_columns(
                pl.col("forward_max_return_pct")
                .cast(pl.Float64)
                .fill_null(0.0)
                .clip(0.0, None)
                .alias("tail_depth_up_pct")
            )
            labeled = self._join_side_rank(labeled, reference, side="up")
        if "forward_min_return_pct" in labeled.columns:
            labeled = labeled.with_columns(
                pl.col("forward_min_return_pct")
                .cast(pl.Float64)
                .abs()
                .fill_null(0.0)
                .alias("tail_depth_down_pct")
            )
            labeled = self._join_side_rank(labeled, reference, side="down")
        return labeled.drop("_tail_row_id")

    def _join_side_rank(
        self, frame: pl.DataFrame, reference: pl.DataFrame, *, side: str
    ) -> pl.DataFrame:
        depth_col = f"tail_depth_{side}_pct"
        rank_col = f"tail_rank_{side}"
        threshold_col = f"tail_event_threshold_{side}_pct"
        event_col = f"tail_event_{side}"
        if reference.is_empty():
            return frame.with_columns(
                pl.lit(0.0).alias(rank_col),
                pl.lit(float(self.extreme.material_floor_pct)).alias(threshold_col),
                pl.lit(False).alias(event_col),
            )
        side_reference = reference.filter(pl.col("side") == side).select(
            "outcome_horizon",
            "reference_depths",
            "sample_count",
            pl.col("event_depth_threshold_pct").alias(threshold_col),
        )
        thresholds = side_reference.select("outcome_horizon", threshold_col)
        ranked = frame.join(thresholds, on="outcome_horizon", how="left").join(
            self._empirical_rank_from_reference(
                frame, side_reference, depth_col=depth_col, rank_col=rank_col
            ),
            on="_tail_row_id",
            how="left",
        )
        if self.extreme.method == "fixed_pct":
            event = pl.col(depth_col) >= pl.col(threshold_col)
        elif self.extreme.method == "empirical_quantile":
            event = pl.col(rank_col).fill_null(0.0) >= float(self.extreme.quantile)
        else:
            event = (pl.col(depth_col) >= float(self.extreme.material_floor_pct)) & (
                pl.col(rank_col).fill_null(0.0) >= float(self.extreme.quantile)
            )
        return ranked.with_columns(
            pl.col(rank_col).fill_null(0.0),
            pl.col(threshold_col).fill_null(float(self.extreme.material_floor_pct)),
            event.fill_null(False).alias(event_col),
            pl.lit(self.extreme.method).alias("tail_event_policy"),
        )

    def _empirical_rank_from_reference(
        self,
        frame: pl.DataFrame,
        reference: pl.DataFrame,
        *,
        depth_col: str,
        rank_col: str,
    ) -> pl.DataFrame:
        if frame.is_empty() or reference.is_empty() or "reference_depths" not in reference.columns:
            return frame.select("_tail_row_id").with_columns(pl.lit(0.0).alias(rank_col))
        rank_frames: list[pl.DataFrame] = []
        for horizon_frame in frame.partition_by("outcome_horizon", maintain_order=True):
            horizon = horizon_frame.get_column("outcome_horizon").item(0)
            horizon_reference = (
                reference.filter(pl.col("outcome_horizon") == horizon)
                .select("reference_depths", "sample_count")
                .head(1)
                .explode("reference_depths")
                .drop_nulls("reference_depths")
                .sort("reference_depths")
                .with_columns(
                    (pl.int_range(pl.len()) + 1)
                    .cast(pl.Float64)
                    .truediv(pl.col("sample_count").cast(pl.Float64))
                    .alias(rank_col)
                )
                .rename({"reference_depths": "_reference_depth"})
                .select("_reference_depth", rank_col)
            )
            if horizon_reference.is_empty():
                rank_frames.append(
                    horizon_frame.select("_tail_row_id").with_columns(pl.lit(0.0).alias(rank_col))
                )
                continue
            rank_frames.append(
                horizon_frame.select("_tail_row_id", depth_col)
                .sort(depth_col)
                .join_asof(
                    horizon_reference,
                    left_on=depth_col,
                    right_on="_reference_depth",
                    strategy="backward",
                )
                .select("_tail_row_id", pl.col(rank_col).fill_null(0.0))
            )
        rank_lookup = (
            pl.concat(rank_frames, how="diagonal_relaxed") if rank_frames else pl.DataFrame()
        )
        if rank_lookup.is_empty():
            return frame.select("_tail_row_id").with_columns(pl.lit(0.0).alias(rank_col))
        return rank_lookup

    def _with_path_behavior(self, labeled: pl.DataFrame) -> pl.DataFrame:
        up = pl.col("tail_event_up").fill_null(False).cast(pl.Boolean)
        down = pl.col("tail_event_down").fill_null(False).cast(pl.Boolean)
        up_threshold = pl.col("tail_event_threshold_up_pct").fill_null(
            float(self.extreme.material_floor_pct)
        )
        down_threshold = pl.col("tail_event_threshold_down_pct").fill_null(
            float(self.extreme.material_floor_pct)
        )
        retention = (
            pl.col("close_retention_ratio").cast(pl.Float64).clip(0.0, 1.0)
            if "close_retention_ratio" in labeled.columns
            else pl.lit(1.0)
        )
        efficiency = (
            pl.col("path_efficiency").cast(pl.Float64).clip(0.0, 1.0)
            if "path_efficiency" in labeled.columns
            else pl.lit(1.0)
        )
        max_speed = (
            1.0 / (1.0 + pl.col("time_to_max_bar").cast(pl.Float64).fill_null(0.0)).sqrt()
            if "time_to_max_bar" in labeled.columns
            else pl.lit(1.0)
        )
        min_speed = (
            1.0 / (1.0 + pl.col("time_to_min_bar").cast(pl.Float64).fill_null(0.0)).sqrt()
            if "time_to_min_bar" in labeled.columns
            else pl.lit(1.0)
        )
        max_drawdown_penalty = (
            pl.col("post_max_drawdown_pct").cast(pl.Float64).fill_null(0.0).clip(0.0, None)
            if "post_max_drawdown_pct" in labeled.columns
            else pl.lit(0.0)
        )
        min_rebound_penalty = (
            pl.col("post_min_rebound_pct").cast(pl.Float64).fill_null(0.0).clip(0.0, None)
            if "post_min_rebound_pct" in labeled.columns
            else pl.lit(0.0)
        )
        up_excess = pl.when(up).then(pl.col("tail_depth_up_pct") - up_threshold).otherwise(None)
        down_excess = (
            pl.when(down).then(pl.col("tail_depth_down_pct") - down_threshold).otherwise(None)
        )
        up_utility = (
            pl.when(up)
            .then(
                (
                    (pl.col("tail_depth_up_pct") - up_threshold)
                    * retention
                    * efficiency
                    * max_speed
                    - 0.1 * max_drawdown_penalty
                ).clip(0.0, None)
            )
            .otherwise(0.0)
        )
        down_utility = (
            pl.when(down)
            .then(
                (
                    (pl.col("tail_depth_down_pct") - down_threshold)
                    * retention
                    * efficiency
                    * min_speed
                    - 0.1 * min_rebound_penalty
                ).clip(0.0, None)
            )
            .otherwise(0.0)
        )
        path_labeled = labeled.with_columns(
            up.alias("tail_up"),
            down.alias("tail_down"),
            (up | down).alias("tail_any"),
            (up & down).alias("tail_both"),
            up_excess.alias("tail_exceedance_value_up"),
            down_excess.alias("tail_exceedance_value_down"),
            up_utility.alias("tail_utility_up"),
            down_utility.alias("tail_utility_down"),
        )
        return _with_path_state_columns(path_labeled)


def _empty_reference_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "outcome_horizon": pl.Int64,
            "side": pl.String,
            "sample_count": pl.Int64,
            "reference_scope": pl.String,
            "policy_method": pl.String,
            "material_floor_pct": pl.Float64,
            "quantile": pl.Float64,
            "min_event_rate": pl.Float64,
            "max_event_rate": pl.Float64,
            "quantile_depth_pct": pl.Float64,
            "reference_depths": pl.List(pl.Float64),
            "event_depth_threshold_pct": pl.Float64,
        }
    )


def _with_path_state_columns(labeled: pl.DataFrame) -> pl.DataFrame:
    up = pl.col("tail_up").fill_null(False).cast(pl.Boolean)
    down = pl.col("tail_down").fill_null(False).cast(pl.Boolean)
    up_utility = pl.col("tail_utility_up").fill_null(0.0).cast(pl.Float64)
    down_utility = pl.col("tail_utility_down").fill_null(0.0).cast(pl.Float64)
    up_margin = up_utility - down_utility
    down_margin = down_utility - up_utility
    time_to_max = (
        pl.col("time_to_max_bar").cast(pl.Float64).fill_null(0.0)
        if "time_to_max_bar" in labeled.columns
        else pl.lit(0.0)
    )
    time_to_min = (
        pl.col("time_to_min_bar").cast(pl.Float64).fill_null(0.0)
        if "time_to_min_bar" in labeled.columns
        else pl.lit(0.0)
    )
    efficient_path = (
        pl.col("path_efficiency").cast(pl.Float64).fill_null(0.0) >= 0.0
        if "path_efficiency" in labeled.columns
        else pl.lit(True)
    )
    first_up = up & (~down | (time_to_max < time_to_min))
    first_down = down & (~up | (time_to_min < time_to_max))
    first_touch = (
        pl.when(first_up)
        .then(pl.lit("up"))
        .when(first_down)
        .then(pl.lit("down"))
        .when(up & down)
        .then(pl.lit("tie"))
        .otherwise(pl.lit("none"))
    )
    clean_up = up & ~down & efficient_path & (up_utility >= 0.0)
    clean_down = down & ~up & efficient_path & (down_utility >= 0.0)
    path_state = (
        pl.when(clean_up)
        .then(pl.lit("clean_up"))
        .when(clean_down)
        .then(pl.lit("clean_down"))
        .when(up & down & first_up)
        .then(pl.lit("up_first_both"))
        .when(up & down & first_down)
        .then(pl.lit("down_first_both"))
        .when(up & down)
        .then(pl.lit("chop_both"))
        .otherwise(pl.lit("none"))
    )
    path_actionability = (
        pl.when(clean_up & (up_margin >= 0.0))
        .then(pl.lit("tradable_up"))
        .when(clean_down & (down_margin >= 0.0))
        .then(pl.lit("tradable_down"))
        .when(up & down)
        .then(pl.lit("reversal_watch"))
        .when(up | down)
        .then(pl.lit("gray_zone"))
        .otherwise(pl.lit("no_action"))
    )
    return labeled.with_columns(
        up.alias("tail_any_up"),
        down.alias("tail_any_down"),
        up.alias("tail_touch_up"),
        down.alias("tail_touch_down"),
        (up | down).alias("tail_touch_any"),
        (up & down).alias("tail_touch_both"),
        pl.when(up & down)
        .then(pl.lit("both"))
        .when(up)
        .then(pl.lit("up"))
        .when(down)
        .then(pl.lit("down"))
        .otherwise(pl.lit("none"))
        .alias("tail_state"),
        first_touch.alias("first_touch_side"),
        path_state.alias("path_state"),
        path_actionability.alias("path_actionability"),
        pl.when(path_actionability == "no_action")
        .then(pl.lit("no_tail_touch"))
        .when(path_actionability == "gray_zone")
        .then(pl.lit("weak_path_utility"))
        .when(path_actionability == "reversal_watch")
        .then(pl.lit("both_or_mixed_path"))
        .otherwise(pl.lit(""))
        .alias("path_blocker"),
        up_utility.alias("path_utility_up"),
        down_utility.alias("path_utility_down"),
        up_margin.alias("tail_utility_margin_up"),
        down_margin.alias("tail_utility_margin_down"),
        up_margin.alias("path_utility_margin_up"),
        down_margin.alias("path_utility_margin_down"),
    )


def tailtree_label_distribution_frame(labeled_outcomes: pl.DataFrame) -> pl.DataFrame:
    """Summarize orthogonal tail-state prevalence by horizon."""
    if labeled_outcomes.is_empty() or "tail_state" not in labeled_outcomes.columns:
        return pl.DataFrame()
    total_by_horizon = labeled_outcomes.group_by("outcome_horizon").agg(
        pl.len().alias("horizon_row_count")
    )
    return (
        labeled_outcomes.group_by("outcome_horizon", "tail_state")
        .agg(
            pl.len().alias("row_count"),
            pl.col("tail_up").fill_null(False).sum().alias("tail_up_count"),
            pl.col("tail_down").fill_null(False).sum().alias("tail_down_count"),
            pl.col("tail_any").fill_null(False).sum().alias("tail_any_count"),
            pl.col("tail_both").fill_null(False).sum().alias("tail_both_count"),
            (pl.col("tail_state") == "up").sum().alias("tail_state_up_count"),
            (pl.col("tail_state") == "down").sum().alias("tail_state_down_count"),
            pl.col("tail_utility_up").fill_null(0.0).mean().alias("tail_utility_up_mean"),
            pl.col("tail_utility_down").fill_null(0.0).mean().alias("tail_utility_down_mean"),
            pl.col("tail_utility_margin_up")
            .fill_null(0.0)
            .mean()
            .alias("tail_utility_margin_up_mean"),
        )
        .join(total_by_horizon, on="outcome_horizon", how="left")
        .with_columns((pl.col("row_count") / pl.col("horizon_row_count")).alias("class_rate"))
        .drop("horizon_row_count")
        .sort("outcome_horizon", "tail_state")
    )
