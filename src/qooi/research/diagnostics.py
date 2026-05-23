"""Reduced research-evaluation diagnostics facade."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from qooi.core.evaluate import format_table


@dataclass(frozen=True)
class ClassifierHealthResult:
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
    return work


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
    return work.select([pl.col(column).cast(dtype) for column, dtype in schema])


def _bool_sum(frame: pl.DataFrame, column: str) -> int:
    if frame.is_empty() or column not in frame.columns:
        return 0
    return int(frame.select(pl.col(column).fill_null(False).cast(pl.Int64).sum()).item() or 0)
