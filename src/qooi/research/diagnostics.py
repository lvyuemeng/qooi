"""Reduced research-evaluation diagnostics API."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from qooi.core.evaluate import format_table


@dataclass(frozen=True)
class ClassifierHealthResult:
    frame: pl.DataFrame
    text: str


@dataclass(frozen=True)
class JointForwardQualityResult:
    frame: pl.DataFrame
    text: str


@dataclass(frozen=True)
class TradeRecordControlResult:
    frame: pl.DataFrame
    text: str


HEALTH_SCHEMA = {
    "artifact": pl.Utf8,
    "label": pl.Utf8,
    "health_check": pl.Utf8,
    "status": pl.Utf8,
    "value": pl.Float64,
    "threshold": pl.Float64,
    "reason": pl.Utf8,
}

JOINT_SCHEMA = {
    "artifact": pl.Utf8,
    "symbol": pl.Utf8,
    "horizon": pl.Int64,
    "configuration_name": pl.Utf8,
    "configuration_kind": pl.Utf8,
    "bucket_family": pl.Utf8,
    "joint_group": pl.Utf8,
    "joint_group_columns": pl.Utf8,
    "liquidity_event_type": pl.Utf8,
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
    "bucket_mean_side_return_pct": pl.Float64,
    "global_mean_side_return_pct": pl.Float64,
    "shrinkage_weight": pl.Float64,
    "shrunk_mean_side_return_pct": pl.Float64,
    "shrunk_positive_rate": pl.Float64,
    "shrunk_omega_proxy": pl.Float64,
    "rank_raw_omega": pl.Int64,
    "rank_shrunk_omega_proxy": pl.Int64,
    "rank_delta": pl.Int64,
    "directional_bias": pl.Utf8,
    "invalid_state_present": pl.Boolean,
    "passes_candidate_gate": pl.Boolean,
    "gate_failure_reasons": pl.Utf8,
    "sufficient_symbols": pl.Int64,
    "symbol_direction_agreement_pct": pl.Float64,
    "time_splits": pl.Int64,
    "sufficient_time_splits": pl.Int64,
    "time_split_sign_agreement_pct": pl.Float64,
    "time_stable": pl.Boolean,
    "configuration_role": pl.Utf8,
    "source_columns": pl.Utf8,
    "component_count": pl.Int64,
    "bucket_count": pl.Int64,
    "valid_bucket_count": pl.Int64,
    "invalid_bucket_count": pl.Int64,
    "invalid_bucket_pct": pl.Float64,
    "median_bucket_rows": pl.Float64,
    "p10_bucket_rows": pl.Float64,
    "p90_bucket_rows": pl.Float64,
    "entropy": pl.Float64,
    "normalized_entropy": pl.Float64,
    "compression_ratio_vs_raw_mtf": pl.Float64,
    "coverage_rows": pl.Int64,
    "coverage_pct": pl.Float64,
    "dominant_bucket_pct": pl.Float64,
    "transition_changed_rate": pl.Float64,
    "self_transition_pct": pl.Float64,
    "intrinsic_quality_bucket": pl.Utf8,
    "intrinsic_warnings": pl.Utf8,
    "total_buckets": pl.Int64,
    "sufficient_buckets": pl.Int64,
    "candidate_gate_buckets": pl.Int64,
    "median_rows": pl.Float64,
    "p90_rows": pl.Float64,
    "median_omega": pl.Float64,
    "p90_omega": pl.Float64,
    "time_stable_buckets": pl.Int64,
    "cross_asset_consistent_buckets": pl.Int64,
    "invalid_state_buckets": pl.Int64,
    "merge_policy": pl.Utf8,
    "connection_family": pl.Utf8,
    "raw_bucket_count": pl.Int64,
    "reduced_bucket_count": pl.Int64,
    "compression_ratio": pl.Float64,
    "information_retention_proxy": pl.Float64,
    "best_bucket_omega": pl.Float64,
    "median_bucket_omega": pl.Float64,
    "merge_decision": pl.Utf8,
    "decision_reason": pl.Utf8,
}

CONTROL_SCHEMA = {
    "artifact": pl.Utf8,
    "base_feature": pl.Utf8,
    "base_value": pl.Utf8,
    "modulator_feature": pl.Utf8,
    "modulator_value": pl.Utf8,
    "base_trades": pl.Int64,
    "conditional_trades": pl.Int64,
    "base_expectancy": pl.Float64,
    "conditional_expectancy": pl.Float64,
    "delta_expectancy": pl.Float64,
    "classification": pl.Utf8,
    "sufficient_base": pl.Boolean,
    "sufficient_cell": pl.Boolean,
    "significant": pl.Boolean,
}

_BULLISH = {"failed_breakout_low", "bullish_reclaim", "breakout_acceptance_high"}
_BEARISH = {"failed_breakout_high", "bearish_reclaim", "breakout_acceptance_low"}


def classifier_health(frame: pl.DataFrame, *, label: str = "") -> ClassifierHealthResult:
    required = ("structure_trend_state", "market_stage", "structure_reason", "stage_unknown_reason")
    present = [column for column in required if column in frame.columns]
    rows = [
        _row(
            HEALTH_SCHEMA,
            artifact="classifier-health",
            label=label,
            health_check="required_classifier_columns",
            status="pass" if len(present) == len(required) else "fail",
            value=len(present) / max(len(required), 1) * 100.0,
            threshold=100.0,
            reason=f"present={len(present)}/{len(required)} rows={frame.height}",
        )
    ]
    for column in ("market_stage", "structure_trend_state", "liquidity_event_type"):
        if column in frame.columns and frame.height:
            unique = int(frame.select(pl.col(column).n_unique()).item() or 0)
            rows.append(
                _row(
                    HEALTH_SCHEMA,
                    artifact="classifier-health",
                    label=label,
                    health_check=f"{column}_cardinality",
                    status="warn" if unique > max(20, frame.height // 5) else "pass",
                    value=float(unique),
                    threshold=float(max(20, frame.height // 5)),
                    reason=f"unique={unique}",
                )
            )
    out = pl.DataFrame(rows, schema=HEALTH_SCHEMA)
    text = f"{label}\n" if label else ""
    text += format_table(
        ["Health check", "Status", "Reason"],
        [[r["health_check"], r["status"], r["reason"]] for r in rows],
    )
    return ClassifierHealthResult(out, text)


def add_market_state_reductions(frame: pl.DataFrame) -> pl.DataFrame:
    work = frame
    if "market_stage" in work.columns and "market_stage_reduced" not in work.columns:
        work = work.with_columns(pl.col("market_stage").cast(pl.Utf8).alias("market_stage_reduced"))
    if "h4_market_stage" in work.columns and "h4_market_stage_reduced" not in work.columns:
        work = work.with_columns(
            pl.col("h4_market_stage").cast(pl.Utf8).alias("h4_market_stage_reduced")
        )
    if "d1_market_stage" in work.columns and "d1_market_stage_reduced" not in work.columns:
        work = work.with_columns(
            pl.col("d1_market_stage").cast(pl.Utf8).alias("d1_market_stage_reduced")
        )
    for source, target in (
        ("h4_structure_trend_state", "h4_structure_trend_state"),
        ("d1_structure_trend_state", "d1_structure_trend_state"),
    ):
        if source in work.columns and target not in work.columns:
            work = work.with_columns(pl.col(source).cast(pl.Utf8).alias(target))
    return work


def add_forward_outcomes(
    frame: pl.DataFrame, *, symbol: str, horizons: tuple[int, ...], timeframe: str = ""
) -> pl.DataFrame:
    if frame.is_empty() or "close" not in frame.columns:
        return frame.with_columns(pl.lit(symbol).alias("symbol"))
    sort_cols = [column for column in ("symbol", "timestamp") if column in frame.columns]
    work = frame.sort(sort_cols) if sort_cols else frame
    exprs = [pl.lit(symbol).alias("symbol")]
    if timeframe and "timeframe" not in work.columns:
        exprs.append(pl.lit(timeframe).alias("timeframe"))
    for horizon in horizons:
        future_close = pl.col("close").shift(-horizon)
        ret = (future_close - pl.col("close")) / pl.col("close") * 100.0
        exprs.extend(
            [
                ret.alias(f"fwd_{horizon}_return_pct"),
                pl.when(ret > 0)
                .then(pl.lit("up"))
                .when(ret < 0)
                .then(pl.lit("down"))
                .otherwise(pl.lit("flat"))
                .alias(f"fwd_{horizon}_direction"),
            ]
        )
    return work.with_columns(exprs)


def joint_forward_quality(
    frame: pl.DataFrame,
    *,
    horizons: tuple[int, ...],
    min_rows: int,
    transition_min_rows: int,
    omega_threshold: float,
    pwpr_threshold: float,
    prior_strength: int,
    invalid_values: tuple[str, ...],
) -> JointForwardQualityResult:
    out = _rank(
        _joint_frame(
            frame,
            horizons=horizons,
            min_rows=min_rows,
            transition_min_rows=transition_min_rows,
            omega_threshold=omega_threshold,
            pwpr_threshold=pwpr_threshold,
            prior_strength=prior_strength,
            invalid_values=invalid_values,
        )
    )
    text = _joint_text(out)
    return JointForwardQualityResult(out, text)


def trade_record_control(
    trades: pl.DataFrame,
    *,
    min_base_trades: int,
    min_cell_trades: int,
    practical_delta_threshold: float,
) -> TradeRecordControlResult:
    out = _trade_record_control_frame(
        trades,
        min_base_trades=min_base_trades,
        min_cell_trades=min_cell_trades,
        practical_delta_threshold=practical_delta_threshold,
    )
    text = "Trade-record control\n" + format_table(
        ["Metric", "Value"],
        [["rows", str(out.height)], ["significant", str(_bool_sum(out, "significant"))]],
    )
    return TradeRecordControlResult(out, text)


def _trade_record_control_frame(
    trades: pl.DataFrame,
    *,
    min_base_trades: int,
    min_cell_trades: int,
    practical_delta_threshold: float,
) -> pl.DataFrame:
    if trades.is_empty():
        return pl.DataFrame(schema=CONTROL_SCHEMA)
    value_col = "net_pnl_usd" if "net_pnl_usd" in trades.columns else "pnl_usd"
    work = _normalize_trade_aliases(trades)
    bases = [column for column in ("entry_market_stage_bucket", "side") if column in work.columns]
    mods = [
        column
        for column in ("entry_d1_structure_trend_state", "entry_d1_market_stage")
        if column in work.columns
    ]
    frames = []
    for base in bases:
        base_stats = work.group_by(base).agg(
            pl.len().alias("base_trades"),
            pl.col(value_col).cast(pl.Float64).mean().alias("base_expectancy"),
        )
        for mod in mods:
            frame = (
                work.group_by(base, mod)
                .agg(
                    pl.len().alias("conditional_trades"),
                    pl.col(value_col).cast(pl.Float64).mean().alias("conditional_expectancy"),
                )
                .join(base_stats, on=base)
                .with_columns(
                    pl.lit("trade-record-modulation").alias("artifact"),
                    pl.lit(base).alias("base_feature"),
                    pl.col(base).cast(pl.Utf8).alias("base_value"),
                    pl.lit(mod).alias("modulator_feature"),
                    pl.col(mod).cast(pl.Utf8).alias("modulator_value"),
                    (pl.col("conditional_expectancy") - pl.col("base_expectancy")).alias(
                        "delta_expectancy"
                    ),
                )
                .with_columns(
                    (pl.col("base_trades") >= min_base_trades).alias("sufficient_base"),
                    (pl.col("conditional_trades") >= min_cell_trades).alias("sufficient_cell"),
                )
                .with_columns(
                    (
                        pl.col("sufficient_base")
                        & pl.col("sufficient_cell")
                        & (pl.col("delta_expectancy").abs() >= practical_delta_threshold)
                    ).alias("significant"),
                    pl.when(pl.col("sufficient_base") & pl.col("sufficient_cell"))
                    .then(pl.lit("control"))
                    .otherwise(pl.lit("insufficient"))
                    .alias("classification"),
                )
            )
            frames.append(_select_schema(frame, CONTROL_SCHEMA))
    return (
        pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame(schema=CONTROL_SCHEMA)
    )


def _joint_frame(
    frame: pl.DataFrame,
    *,
    horizons: tuple[int, ...],
    min_rows: int,
    transition_min_rows: int,
    omega_threshold: float,
    pwpr_threshold: float,
    prior_strength: int,
    invalid_values: tuple[str, ...],
) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(schema=JOINT_SCHEMA)
    work = _with_event_side(add_market_state_reductions(frame))
    work = _with_connection_columns(_with_transition_columns(work))
    specs = _configuration_specs(work)
    frames: list[pl.DataFrame] = [_intrinsic_frame(work, specs, invalid_values, min_rows)]
    for horizon in horizons:
        ret_col = f"fwd_{horizon}_return_pct"
        if ret_col not in work.columns:
            continue
        scored = work.filter(
            pl.col("side").is_not_null() & pl.col(ret_col).is_not_null()
        ).with_columns(
            pl.when(pl.col("side") == "short")
            .then(-pl.col(ret_col).cast(pl.Float64))
            .otherwise(pl.col(ret_col).cast(pl.Float64))
            .alias("side_return_pct")
        )
        if scored.is_empty():
            continue
        quality_frames = [
            _quality_frame(
                scored,
                spec,
                horizon=horizon,
                min_rows=min_rows if spec["kind"] != "transition" else transition_min_rows,
                omega_threshold=omega_threshold if spec["kind"] != "transition" else 3.0,
                pwpr_threshold=pwpr_threshold,
                prior_strength=prior_strength,
                invalid_values=invalid_values,
            )
            for spec in specs
            if set(spec["columns"]) <= set(scored.columns)
        ]
        quality = _concat_frames(quality_frames)
        frames.extend(
            [
                quality,
                _comparison_frame(quality, horizon),
                _inner_connection_frame(quality, horizon),
            ]
        )
    return _concat_frames(frames)


def _configuration_specs(frame: pl.DataFrame) -> list[dict[str, object]]:
    candidates = [
        ("config_reduced_d1_event", "reduced_static", ("d1_market_stage_reduced",)),
        ("config_reduced_d1_structure_event", "reduced_static", ("d1_structure_trend_state",)),
        (
            "config_reduced_d1_h4_h1_event",
            "reduced_static",
            ("d1_structure_trend_state", "h4_market_stage_reduced", "market_stage_reduced"),
        ),
        ("config_static_raw_mtf", "static", ("mtf_structure_key",)),
        (
            "config_dynamic_stage_transition_event",
            "transition",
            ("market_stage_reduced_transition",),
        ),
        (
            "config_dynamic_inner_connection_event",
            "inner_connection",
            ("reduced_inner_connection_path",),
        ),
    ]
    return [
        {"name": name, "kind": kind, "columns": cols, "family": name.removeprefix("config_")}
        for name, kind, cols in candidates
        if set(cols) <= set(frame.columns)
    ]


def _quality_frame(
    frame: pl.DataFrame,
    spec: dict[str, object],
    *,
    horizon: int,
    min_rows: int,
    omega_threshold: float,
    pwpr_threshold: float,
    prior_strength: int,
    invalid_values: tuple[str, ...],
) -> pl.DataFrame:
    group_expr = pl.concat_str(
        [pl.col(c).cast(pl.Utf8).fill_null("unknown") for c in spec["columns"]], separator="|"
    )
    global_mean = float(frame.select(pl.col("side_return_pct").mean()).item() or 0.0)
    artifact = (
        "transition-event-quality" if spec["kind"] == "transition" else "joint-forward-quality"
    )
    grouped = (
        frame.with_columns(group_expr.alias("joint_group"))
        .group_by("symbol", "joint_group", "liquidity_event_type", "side")
        .agg(
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
        )
        .with_columns(
            pl.lit(artifact).alias("artifact"),
            pl.lit(horizon).alias("horizon"),
            pl.lit(spec["name"]).alias("configuration_name"),
            pl.lit(spec["kind"]).alias("configuration_kind"),
            pl.lit(spec["family"]).alias("bucket_family"),
            pl.lit(",".join(spec["columns"])).alias("joint_group_columns"),
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
            pl.col("mean_side_return_pct").alias("bucket_mean_side_return_pct"),
            pl.lit(global_mean).alias("global_mean_side_return_pct"),
            (pl.col("rows") / (pl.col("rows") + prior_strength)).alias("shrinkage_weight"),
            pl.when(pl.col("mean_side_return_pct") > 0)
            .then(pl.lit("up"))
            .when(pl.col("mean_side_return_pct") < 0)
            .then(pl.lit("down"))
            .otherwise(pl.lit("flat"))
            .alias("directional_bias"),
            _invalid_expr("joint_group", invalid_values).alias("invalid_state_present"),
            pl.lit(1).alias("sufficient_symbols"),
            pl.when(pl.col("mean_side_return_pct") > 0)
            .then(100.0)
            .otherwise(0.0)
            .alias("symbol_direction_agreement_pct"),
            pl.lit(1).alias("time_splits"),
            pl.when(pl.col("rows") >= min_rows)
            .then(1)
            .otherwise(0)
            .alias("sufficient_time_splits"),
            pl.when(pl.col("mean_side_return_pct") > 0)
            .then(100.0)
            .otherwise(0.0)
            .alias("time_split_sign_agreement_pct"),
            ((pl.col("rows") >= min_rows) & (pl.col("mean_side_return_pct") > 0)).alias(
                "time_stable"
            ),
        )
        .with_columns(
            (
                pl.col("shrinkage_weight") * pl.col("mean_side_return_pct")
                + (1.0 - pl.col("shrinkage_weight")) * pl.col("global_mean_side_return_pct")
            ).alias("shrunk_mean_side_return_pct"),
            (
                (pl.col("positive_rows") + prior_strength * 0.5)
                / (pl.col("rows") + prior_strength)
                * 100.0
            ).alias("shrunk_positive_rate"),
            (pl.col("shrinkage_weight") * pl.col("omega_ratio")).alias("shrunk_omega_proxy"),
        )
        .with_columns(
            (
                (pl.col("rows") >= min_rows)
                & (~pl.col("invalid_state_present"))
                & (pl.col("omega_ratio") > omega_threshold)
                & (pl.col("pwpr") > pwpr_threshold)
                & (pl.col("mean_side_return_pct") > 0)
            ).alias("passes_candidate_gate"),
            pl.concat_str(
                [
                    pl.when(pl.col("rows") < min_rows).then(pl.lit("rows,")).otherwise(pl.lit("")),
                    pl.when(pl.col("invalid_state_present"))
                    .then(pl.lit("invalid_state,"))
                    .otherwise(pl.lit("")),
                    pl.when(pl.col("omega_ratio") <= omega_threshold)
                    .then(pl.lit("omega,"))
                    .otherwise(pl.lit("")),
                    pl.when(pl.col("pwpr") <= pwpr_threshold)
                    .then(pl.lit("pwpr,"))
                    .otherwise(pl.lit("")),
                    pl.when(pl.col("mean_side_return_pct") <= 0)
                    .then(pl.lit("direction,"))
                    .otherwise(pl.lit("")),
                ]
            )
            .str.strip_chars_end(",")
            .alias("gate_failure_reasons"),
        )
    )
    return _select_schema(grouped, JOINT_SCHEMA)


def _intrinsic_frame(
    frame: pl.DataFrame,
    specs: list[dict[str, object]],
    invalid_values: tuple[str, ...],
    min_rows: int,
) -> pl.DataFrame:
    raw_count = _bucket_count(frame, ("mtf_structure_key",)) or 1
    frames = []
    for spec in specs:
        cols = spec["columns"]
        counts = (
            frame.with_columns(
                pl.concat_str(
                    [pl.col(c).cast(pl.Utf8).fill_null("unknown") for c in cols], separator="|"
                ).alias("bucket")
            )
            .group_by("bucket")
            .agg(pl.len().alias("rows"))
            .with_columns((pl.col("rows") / frame.height).alias("p"))
        )
        stats = (
            counts.select(
                pl.len().alias("bucket_count"),
                _invalid_expr("bucket", invalid_values).sum().alias("invalid_bucket_count"),
                pl.col("rows").median().alias("median_bucket_rows"),
                pl.col("rows").quantile(0.1).alias("p10_bucket_rows"),
                pl.col("rows").quantile(0.9).alias("p90_bucket_rows"),
                (-(pl.col("p") * pl.col("p").log(2)).sum()).alias("entropy"),
                (pl.col("rows").max() / frame.height * 100.0).alias("dominant_bucket_pct"),
            )
            .with_columns(
                pl.lit("configuration-intrinsic-quality").alias("artifact"),
                pl.lit(spec["name"]).alias("configuration_name"),
                pl.lit(spec["kind"]).alias("configuration_kind"),
                pl.lit(spec["family"]).alias("bucket_family"),
                pl.lit(spec["kind"]).alias("configuration_role"),
                pl.lit(",".join(cols)).alias("source_columns"),
                pl.lit(len(cols)).alias("component_count"),
                pl.lit(frame.height).alias("coverage_rows"),
                pl.lit(100.0).alias("coverage_pct"),
                (pl.col("bucket_count") / raw_count).alias("compression_ratio_vs_raw_mtf"),
                (pl.col("invalid_bucket_count") / pl.col("bucket_count") * 100.0).alias(
                    "invalid_bucket_pct"
                ),
                (pl.col("bucket_count") - pl.col("invalid_bucket_count")).alias(
                    "valid_bucket_count"
                ),
                (
                    pl.col("entropy") / pl.max_horizontal(pl.col("bucket_count"), pl.lit(2)).log(2)
                ).alias("normalized_entropy"),
                pl.lit(
                    _transition_changed_rate_native(frame, cols[0])
                    if spec["kind"] == "transition"
                    else 0.0
                ).alias("transition_changed_rate"),
                pl.lit(0.0).alias("self_transition_pct"),
            )
            .with_columns(
                pl.when(
                    (pl.col("median_bucket_rows") >= min_rows)
                    & (pl.col("normalized_entropy") >= 0.2)
                    & (pl.col("invalid_bucket_count") == 0)
                )
                .then(pl.lit("high"))
                .when(pl.col("median_bucket_rows") >= min_rows)
                .then(pl.lit("medium"))
                .otherwise(pl.lit("low"))
                .alias("intrinsic_quality_bucket"),
                pl.concat_str(
                    [
                        pl.when(pl.col("median_bucket_rows") < min_rows)
                        .then(pl.lit("sparse,"))
                        .otherwise(pl.lit("")),
                        pl.when(pl.col("invalid_bucket_count") > 0)
                        .then(pl.lit("invalid,"))
                        .otherwise(pl.lit("")),
                    ]
                )
                .str.strip_chars_end(",")
                .replace("", "none")
                .alias("intrinsic_warnings"),
            )
        )
        frames.append(_select_schema(stats, JOINT_SCHEMA))
    return _concat_frames(frames)


def _comparison_frame(frame: pl.DataFrame, horizon: int) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(schema=JOINT_SCHEMA)
    out = (
        frame.group_by("bucket_family")
        .agg(
            pl.len().alias("total_buckets"),
            (pl.col("rows") >= 1).sum().alias("sufficient_buckets"),
            pl.col("passes_candidate_gate").fill_null(False).sum().alias("candidate_gate_buckets"),
            pl.col("rows").median().alias("median_rows"),
            pl.col("rows").quantile(0.9).alias("p90_rows"),
            pl.col("omega_ratio").median().alias("median_omega"),
            pl.col("omega_ratio").quantile(0.9).alias("p90_omega"),
            pl.col("time_stable").fill_null(False).sum().alias("time_stable_buckets"),
            (pl.col("symbol_direction_agreement_pct") >= 75.0)
            .sum()
            .alias("cross_asset_consistent_buckets"),
            pl.col("invalid_state_present").fill_null(False).sum().alias("invalid_state_buckets"),
        )
        .with_columns(
            pl.lit("joint-reduction-comparison").alias("artifact"),
            pl.lit(horizon).alias("horizon"),
        )
    )
    return _select_schema(out, JOINT_SCHEMA)


def _inner_connection_frame(frame: pl.DataFrame, horizon: int) -> pl.DataFrame:
    inner = (
        frame.filter(pl.col("configuration_kind") == "inner_connection")
        if not frame.is_empty()
        else frame
    )
    if inner.is_empty():
        return pl.DataFrame(schema=JOINT_SCHEMA)
    out = inner.select(
        pl.lit("inner-connection-reduction-quality").alias("artifact"),
        pl.lit(horizon).alias("horizon"),
        pl.lit("merge_adjacent").alias("merge_policy"),
        pl.lit("stage_connection").alias("connection_family"),
        pl.len().alias("raw_bucket_count"),
        pl.col("joint_group").n_unique().alias("reduced_bucket_count"),
        pl.lit(1.0).alias("compression_ratio"),
        pl.col("passes_candidate_gate")
        .fill_null(False)
        .mean()
        .alias("information_retention_proxy"),
        pl.col("passes_candidate_gate").fill_null(False).sum().alias("candidate_gate_buckets"),
        pl.col("time_stable").fill_null(False).sum().alias("time_stable_buckets"),
        pl.lit(100.0).alias("symbol_direction_agreement_pct"),
        pl.col("omega_ratio").max().alias("best_bucket_omega"),
        pl.col("omega_ratio").median().alias("median_bucket_omega"),
    ).with_columns(
        pl.when(pl.col("candidate_gate_buckets") > 0)
        .then(pl.lit("merge"))
        .otherwise(pl.lit("reject"))
        .alias("merge_decision"),
        pl.lit("diagnostic_only").alias("decision_reason"),
    )
    return _select_schema(out, JOINT_SCHEMA)


def _with_event_side(frame: pl.DataFrame) -> pl.DataFrame:
    if "liquidity_event_type" not in frame.columns:
        return frame.with_columns(pl.lit(None).cast(pl.Utf8).alias("side"))
    return frame.with_columns(
        pl.when(pl.col("liquidity_event_type").is_in(_BULLISH))
        .then(pl.lit("long"))
        .when(pl.col("liquidity_event_type").is_in(_BEARISH))
        .then(pl.lit("short"))
        .otherwise(None)
        .alias("side")
    )


def _with_transition_columns(frame: pl.DataFrame) -> pl.DataFrame:
    sort_cols = [column for column in ("symbol", "timestamp") if column in frame.columns]
    work = frame.sort(sort_cols) if sort_cols else frame
    exprs = []
    for column in ("market_stage_reduced", "h4_market_stage_reduced", "d1_market_stage_reduced"):
        if column in work.columns:
            prev = (
                pl.col(column).shift(1).over("symbol")
                if "symbol" in work.columns
                else pl.col(column).shift(1)
            )
            exprs.append(
                pl.concat_str(
                    [prev.cast(pl.Utf8), pl.lit("->"), pl.col(column).cast(pl.Utf8)]
                ).alias(f"{column}_transition")
            )
    return work.with_columns(exprs) if exprs else work


def _with_connection_columns(frame: pl.DataFrame) -> pl.DataFrame:
    exprs = []
    if {"d1_market_stage_reduced", "h4_market_stage_reduced", "market_stage_reduced"} <= set(
        frame.columns
    ):
        exprs.append(
            pl.concat_str(
                [
                    pl.col("d1_market_stage_reduced").cast(pl.Utf8),
                    pl.lit("->"),
                    pl.col("h4_market_stage_reduced").cast(pl.Utf8),
                    pl.lit("|"),
                    pl.col("h4_market_stage_reduced").cast(pl.Utf8),
                    pl.lit("->"),
                    pl.col("market_stage_reduced").cast(pl.Utf8),
                ]
            ).alias("reduced_inner_connection_path")
        )
    return frame.with_columns(exprs) if exprs else frame


def _rank(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty() or "omega_ratio" not in frame.columns:
        return frame
    rankable = frame.with_row_index("_idx")
    ranks = rankable.filter(
        pl.col("artifact").is_in(["joint-forward-quality", "transition-event-quality"])
    ).select(
        "_idx",
        pl.col("omega_ratio")
        .rank("ordinal", descending=True)
        .cast(pl.Int64)
        .alias("rank_raw_omega"),
        pl.col("shrunk_omega_proxy")
        .rank("ordinal", descending=True)
        .cast(pl.Int64)
        .alias("rank_shrunk_omega_proxy"),
    )
    return (
        rankable.join(ranks, on="_idx", how="left", suffix="_new")
        .with_columns(
            pl.coalesce([pl.col("rank_raw_omega_new"), pl.col("rank_raw_omega")]).alias(
                "rank_raw_omega"
            ),
            pl.coalesce(
                [pl.col("rank_shrunk_omega_proxy_new"), pl.col("rank_shrunk_omega_proxy")]
            ).alias("rank_shrunk_omega_proxy"),
        )
        .with_columns(
            (pl.col("rank_raw_omega") - pl.col("rank_shrunk_omega_proxy")).alias("rank_delta")
        )
        .drop(["_idx", "rank_raw_omega_new", "rank_shrunk_omega_proxy_new"])
    )


def _normalize_trade_aliases(trades: pl.DataFrame) -> pl.DataFrame:
    work = trades
    if "entry_market_stage_bucket" not in work.columns and "entry_market_stage" in work.columns:
        work = work.with_columns(pl.col("entry_market_stage").alias("entry_market_stage_bucket"))
    if "side" in work.columns:
        work = work.with_columns(
            pl.when(pl.col("side").is_in(["buy", "long"]))
            .then(pl.lit("long"))
            .when(pl.col("side").is_in(["sell", "short"]))
            .then(pl.lit("short"))
            .otherwise(pl.col("side").cast(pl.Utf8))
            .alias("side")
        )
    return work


def _row(schema: dict[str, pl.DataType], **values: object) -> dict[str, object]:
    return {column: values.get(column) for column in schema}


def _select_schema(frame: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if frame.is_empty() and not frame.columns:
        return pl.DataFrame(schema=schema)
    additions = [
        pl.lit(None).cast(dtype).alias(column)
        for column, dtype in schema.items()
        if column not in frame.columns
    ]
    work = frame.with_columns(additions) if additions else frame
    return work.select([pl.col(column).cast(dtype) for column, dtype in schema.items()])


def _concat_frames(frames: list[pl.DataFrame]) -> pl.DataFrame:
    frames = [frame for frame in frames if not frame.is_empty()]
    return (
        pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame(schema=JOINT_SCHEMA)
    )


def _invalid_expr(column: str, invalid_values: tuple[str, ...]) -> pl.Expr:
    if not invalid_values:
        return pl.lit(False)
    return (
        pl.col(column)
        .cast(pl.Utf8)
        .str.split("|")
        .list.eval(pl.element().is_in(invalid_values))
        .list.any()
    )


def _transition_changed_rate_native(frame: pl.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return 0.0
    result = frame.select(
        pl.col(column)
        .cast(pl.Utf8)
        .str.split_exact("->", 1)
        .struct.rename_fields(["from_state", "to_state"])
        .struct.unnest()
    ).select((pl.col("from_state") != pl.col("to_state")).mean())
    return float(result.item() or 0.0) * 100.0


def _joint_text(frame: pl.DataFrame) -> str:
    if frame.is_empty():
        return "Joint forward quality\nno rows"
    artifacts = frame.group_by("artifact").agg(pl.len().alias("rows")).sort("artifact")
    return "Joint forward quality\n" + format_table(
        ["Artifact", "Rows"],
        [[str(r["artifact"]), str(r["rows"])] for r in artifacts.iter_rows(named=True)],
    )


def _bool_sum(frame: pl.DataFrame, column: str) -> int:
    if frame.is_empty() or column not in frame.columns:
        return 0
    return int(frame.select(pl.col(column).fill_null(False).cast(pl.Int64).sum()).item() or 0)


def _bucket_count(frame: pl.DataFrame, columns: tuple[str, ...]) -> int:
    if not set(columns) <= set(frame.columns):
        return 0
    return frame.select(pl.struct(list(columns)).n_unique()).item() or 0
