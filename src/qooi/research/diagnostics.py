"""Layered diagnostics for strategy improvement workflows."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypedDict

import polars as pl

from qooi.core.evaluate import Report, format_table
from qooi.research.config import RiskGateConfig
from qooi.strategies import StrategyBehavior, StrategySpec
from qooi.strategies.semantics import (
    ClassifierColumn,
    ClassifierDiagnosticName,
    DiagnosticColumn,
    LiquidityEvent,
    LossCause,
)


@dataclass(frozen=True)
class DiagnosticRow:
    layer: str
    name: str
    summary: str
    severity: str = "info"


@dataclass(frozen=True)
class DiagnosticTable:
    name: str
    frame: pl.DataFrame


@dataclass(frozen=True)
class ClassifierDiagnostics:
    label: str
    rows: tuple[DiagnosticRow, ...]
    tables: tuple[DiagnosticTable, ...]


@dataclass(frozen=True)
class StateAttributionDiagnostics:
    label: str
    rows: tuple[DiagnosticRow, ...]
    tables: tuple[DiagnosticTable, ...] = ()


class _UnknownConsistencySummary(TypedDict):
    consistency: str
    none: str
    raw_unknown: str
    severity: str
    none_severity: str
    raw_unknown_watch: bool


class ClassifierDiagnosticsBuilder:
    def evaluate(self, label: str, frame: pl.DataFrame) -> ClassifierDiagnostics:
        rows: list[DiagnosticRow] = []
        tables: list[DiagnosticTable] = []
        row_count = frame.height
        required = (
            ClassifierColumn.STRUCTURE_TREND_STATE,
            ClassifierColumn.MARKET_STAGE,
            ClassifierColumn.STRUCTURE_REASON,
            ClassifierColumn.STAGE_UNKNOWN_REASON,
        )
        present = [column for column in required if column in frame.columns]
        coverage_pct = len(present) / max(len(required), 1) * 100.0
        coverage_summary = (
            f"rows={row_count} required_columns={len(present)}/{len(required)} "
            f"coverage={coverage_pct:.1f}%"
        )
        rows.append(
            DiagnosticRow(
                "classifier",
                ClassifierDiagnosticName.COVERAGE,
                coverage_summary,
                "info" if len(present) == len(required) else "warn",
            )
        )

        distribution_specs = (
            (ClassifierDiagnosticName.STAGE_DISTRIBUTION, ClassifierColumn.MARKET_STAGE),
            (ClassifierDiagnosticName.TREND_DISTRIBUTION, ClassifierColumn.STRUCTURE_TREND_STATE),
            (ClassifierDiagnosticName.REASON_DISTRIBUTION, ClassifierColumn.STAGE_UNKNOWN_REASON),
            ("Structure reason distribution", ClassifierColumn.STRUCTURE_REASON),
        )
        distribution_frames = []
        for name, column in distribution_specs:
            table = _classifier_distribution_table(frame, column)
            distribution_frames.append(table)
            rows.append(DiagnosticRow("classifier", name, _summarize_distribution(table, column)))
        tables.append(
            DiagnosticTable(
                "distribution",
                _concat_or_empty(
                    distribution_frames,
                    ["dimension", "value", "count", "pct"],
                ),
            )
        )

        unknown_table = _classifier_unknown_consistency_table(frame)
        unknown_summary = _summarize_unknown_consistency(unknown_table)
        rows.append(
            DiagnosticRow(
                "classifier",
                ClassifierDiagnosticName.UNKNOWN_REASON_CONSISTENCY,
                unknown_summary["consistency"],
                unknown_summary["severity"],
            )
        )
        rows.append(
            DiagnosticRow(
                "classifier",
                ClassifierDiagnosticName.RESOLVED_NONE_AUDIT,
                unknown_summary["none"],
                unknown_summary["none_severity"],
            )
        )
        rows.append(
            DiagnosticRow(
                "classifier",
                ClassifierDiagnosticName.RAW_UNKNOWN_ATTRIBUTION,
                unknown_summary["raw_unknown"],
                "warn" if unknown_summary["raw_unknown_watch"] else "info",
            )
        )
        tables.append(DiagnosticTable("unknown_consistency", unknown_table))

        threshold_table = _classifier_threshold_table(frame)
        rows.append(
            DiagnosticRow(
                "classifier",
                ClassifierDiagnosticName.THRESHOLD_DISTRIBUTION,
                _summarize_threshold_table(threshold_table),
            )
        )
        tables.append(DiagnosticTable("threshold", threshold_table))

        matrix_frames = [
            _classifier_matrix_table(
                frame,
                ClassifierColumn.STRUCTURE_TREND_STATE,
                ClassifierColumn.MARKET_STAGE,
            ),
            _classifier_matrix_table(
                frame,
                ClassifierColumn.MARKET_STAGE,
                ClassifierColumn.STAGE_UNKNOWN_REASON,
            ),
        ]
        rows.append(
            DiagnosticRow(
                "classifier",
                ClassifierDiagnosticName.STRUCTURE_STAGE_MATRIX,
                _summarize_matrix(matrix_frames[0]),
            )
        )
        rows.append(
            DiagnosticRow(
                "classifier",
                ClassifierDiagnosticName.STAGE_REASON_MATRIX,
                _summarize_matrix(matrix_frames[1]),
            )
        )
        tables.append(
            DiagnosticTable(
                "matrix",
                _concat_or_empty(
                    matrix_frames,
                    [
                        "row_dimension",
                        "row_value",
                        "col_dimension",
                        "col_value",
                        "count",
                        "pct",
                    ],
                ),
            )
        )

        state_columns = [
            column
            for column in ("mtf_state_key", "mtf_structure_key", "mtf_stage_key")
            if column in frame.columns
        ]
        rows.append(
            DiagnosticRow(
                "classifier",
                ClassifierDiagnosticName.MTF_STATE_CARDINALITY,
                _summarize_state_cardinality(frame, state_columns),
            )
        )
        transition_frames = [
            _classifier_transition_table(frame, column) for column in state_columns
        ]
        transition_table = _concat_or_empty(
            transition_frames,
            ["state_column", "from_state", "to_state", "count", "pct", "self_transition"],
        )
        rows.append(
            DiagnosticRow(
                "classifier",
                ClassifierDiagnosticName.MTF_STATE_TRANSITION_SUMMARY,
                _summarize_transitions(transition_table),
            )
        )
        rows.append(
            DiagnosticRow(
                "classifier",
                ClassifierDiagnosticName.MTF_STATE_TRANSITION_MATRIX,
                _summarize_matrix_like_count(transition_table),
            )
        )
        tables.append(DiagnosticTable("transition", transition_table))

        dwell_table = _concat_or_empty(
            [_classifier_dwell_table(frame, column) for column in state_columns],
            ["state_column", "state", "runs", "median_dwell", "mean_dwell", "max_dwell"],
        )
        rows.append(
            DiagnosticRow(
                "classifier",
                ClassifierDiagnosticName.MTF_STATE_DWELL_DISTRIBUTION,
                _summarize_matrix_like_count(dwell_table, count_col="runs"),
            )
        )
        tables.append(DiagnosticTable("dwell", dwell_table))

        time_table = _concat_or_empty(
            [_classifier_time_distribution_table(frame, column) for column in state_columns],
            [
                "state_column",
                "state",
                "time_bucket",
                "count",
                "pct_of_state",
                "pct_of_bucket",
            ],
        )
        rows.append(
            DiagnosticRow(
                "classifier",
                ClassifierDiagnosticName.MTF_STATE_TIME_DISTRIBUTION,
                _summarize_matrix_like_count(time_table),
            )
        )
        tables.append(DiagnosticTable("time_distribution", time_table))
        rows.append(
            DiagnosticRow(
                "classifier",
                ClassifierDiagnosticName.MTF_RIGHT_EDGE_DRIFT,
                _format_mtf_right_edge_drift_row(frame)[1],
            )
        )
        return ClassifierDiagnostics(label, tuple(rows), tuple(tables))


class StateAttributionDiagnosticsBuilder:
    def evaluate(
        self,
        label: str,
        report: Report,
        frame: pl.DataFrame,
    ) -> StateAttributionDiagnostics:
        return StateAttributionDiagnostics(
            label,
            tuple(
                DiagnosticRow("state_attribution", name, summary)
                for name, summary in state_diagnostics_rows(report, frame)
            ),
        )


def evaluate_classifier_frame(label: str, frame: pl.DataFrame) -> ClassifierDiagnostics:
    return ClassifierDiagnosticsBuilder().evaluate(label, frame)


def evaluate_state_attribution(
    label: str,
    report: Report,
    frame: pl.DataFrame,
) -> StateAttributionDiagnostics:
    return StateAttributionDiagnosticsBuilder().evaluate(label, report, frame)


def format_classifier_diagnostics(diagnostics: ClassifierDiagnostics) -> str:
    return f"{diagnostics.label}\n" + format_table(
        ["Classifier diagnostic", "Severity", "Summary"],
        [[row.name, row.severity, row.summary] for row in diagnostics.rows],
    )


def format_state_attribution(diagnostics: StateAttributionDiagnostics) -> str:
    return f"{diagnostics.label}\n" + format_table(
        ["State diagnostic", "Summary"],
        [[row.name, row.summary] for row in diagnostics.rows],
    )


def classifier_diagnostics_export_frame(diagnostics: ClassifierDiagnostics) -> pl.DataFrame:
    records = [
        {
            "label": diagnostics.label,
            "artifact": "row",
            "table": "rows",
            "layer": row.layer,
            "name": row.name,
            "severity": row.severity,
            "summary": row.summary,
            "field": "",
            "value": "",
        }
        for row in diagnostics.rows
    ]
    for table in diagnostics.tables:
        for record in table.frame.iter_rows(named=True):
            for field, value in record.items():
                records.append(
                    {
                        "label": diagnostics.label,
                        "artifact": "table",
                        "table": table.name,
                        "layer": "classifier",
                        "name": table.name,
                        "severity": "info",
                        "summary": "",
                        "field": str(field),
                        "value": "" if value is None else str(value),
                    }
                )
    return pl.DataFrame(records)


def _concat_or_empty(frames: list[pl.DataFrame], columns: list[str]) -> pl.DataFrame:
    frames = [frame for frame in frames if not frame.is_empty()]
    if not frames:
        return pl.DataFrame({column: [] for column in columns})
    return pl.concat(frames, how="diagonal_relaxed")


def _classifier_distribution_table(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    if frame.is_empty() or column not in frame.columns:
        return pl.DataFrame({"dimension": [], "value": [], "count": [], "pct": []})
    total = max(frame.height, 1)
    return (
        frame.with_columns(pl.col(column).cast(pl.Utf8).fill_null("data_error"))
        .group_by(column)
        .agg(pl.len().alias("count"))
        .with_columns(
            pl.lit(column).alias("dimension"),
            pl.col(column).alias("value"),
            (pl.col("count") / total * 100.0).alias("pct"),
        )
        .select("dimension", "value", "count", "pct")
        .sort(["count", "value"], descending=[True, False])
    )


def _summarize_distribution(table: pl.DataFrame, column: str) -> str:
    if table.is_empty():
        return f"{column}=n/a"
    parts = [
        f"{row['value']}:{int(row['count'])}/{float(row['pct'] or 0.0):.1f}%"
        for row in table.head(5).iter_rows(named=True)
    ]
    return f"{column}=" + ",".join(parts)


def _classifier_unknown_consistency_table(frame: pl.DataFrame) -> pl.DataFrame:
    columns = [
        ClassifierColumn.MARKET_STAGE,
        ClassifierColumn.STRUCTURE_TREND_STATE,
        ClassifierColumn.STRUCTURE_REASON,
        ClassifierColumn.MARKET_STAGE_REASON,
        ClassifierColumn.STAGE_UNKNOWN_REASON,
    ]
    output_columns = [*columns, "count", "pct", "verdict"]
    if frame.is_empty() or not set(columns) <= set(frame.columns):
        return pl.DataFrame({column: [] for column in output_columns})

    unknown_stages = ["warmup", "wide_range", "transition", "data_error"]
    resolved_stages = [
        "markup",
        "markdown",
        "accumulation",
        "distribution_or_reversal",
        "range",
        "trend_continuation",
    ]
    total = max(frame.height, 1)
    return (
        frame.with_columns(
            *(pl.col(column).cast(pl.Utf8).fill_null("data_error") for column in columns)
        )
        .group_by(columns)
        .agg(pl.len().alias("count"))
        .with_columns(
            (pl.col("count") / total * 100.0).alias("pct"),
            pl.when(
                pl.col(ClassifierColumn.MARKET_STAGE).is_in(unknown_stages)
                & (pl.col(ClassifierColumn.STAGE_UNKNOWN_REASON) == "none")
            )
            .then(pl.lit("contradiction_unknown_stage_none"))
            .when(
                pl.col(ClassifierColumn.MARKET_STAGE).is_in(resolved_stages)
                & (pl.col(ClassifierColumn.STAGE_UNKNOWN_REASON) != "none")
            )
            .then(pl.lit("contradiction_resolved_stage_unknown_reason"))
            .when(
                (pl.col(ClassifierColumn.STAGE_UNKNOWN_REASON) == "data_error")
                & (pl.col(ClassifierColumn.MARKET_STAGE) != "data_error")
                & (pl.col(ClassifierColumn.STRUCTURE_REASON) != "data_error")
                & (pl.col(ClassifierColumn.MARKET_STAGE_REASON) != "data_error")
            )
            .then(pl.lit("contradiction_data_error_reason"))
            .when(
                (pl.col(ClassifierColumn.STRUCTURE_TREND_STATE) == "unknown")
                & (pl.col(ClassifierColumn.STAGE_UNKNOWN_REASON) == "none")
            )
            .then(pl.lit("watch_raw_unknown_resolved_none"))
            .otherwise(pl.lit("ok"))
            .alias("verdict"),
        )
        .select(output_columns)
        .sort(["verdict", "count"], descending=[False, True])
    )


def _summarize_unknown_consistency(table: pl.DataFrame) -> _UnknownConsistencySummary:
    if table.is_empty():
        return {
            "consistency": "n/a",
            "none": "n/a",
            "raw_unknown": "n/a",
            "severity": "warn",
            "none_severity": "warn",
            "raw_unknown_watch": False,
        }
    total = int(table.select(pl.col("count").sum()).item() or 0)
    contradiction_count = int(
        table.filter(pl.col("verdict").str.starts_with("contradiction"))
        .select(pl.col("count").sum())
        .item()
        or 0
    )
    raw_unknown_none = int(
        table.filter(pl.col("verdict") == "watch_raw_unknown_resolved_none")
        .select(pl.col("count").sum())
        .item()
        or 0
    )
    none_total = int(
        table.filter(pl.col(ClassifierColumn.STAGE_UNKNOWN_REASON) == "none")
        .select(pl.col("count").sum())
        .item()
        or 0
    )
    none_contradictions = int(
        table.filter(
            (pl.col(ClassifierColumn.STAGE_UNKNOWN_REASON) == "none")
            & (pl.col("verdict").str.starts_with("contradiction"))
        )
        .select(pl.col("count").sum())
        .item()
        or 0
    )
    top_watch = table.filter(pl.col("verdict") != "ok").sort("count", descending=True).head(3)
    watch_parts = [
        f"{row['verdict']}:{int(row['count'])}/{float(row['pct'] or 0.0):.1f}%"
        for row in top_watch.iter_rows(named=True)
    ]
    severity = "fail" if contradiction_count else "info"
    return {
        "consistency": (
            f"contradictions={contradiction_count}/{total} "
            f"({contradiction_count / max(total, 1) * 100.0:.2f}%) "
            f"watch={','.join(watch_parts) or 'none'}"
        ),
        "none": (
            f"none_rows={none_total}/{total} ({none_total / max(total, 1) * 100.0:.1f}%) "
            f"none_contradictions={none_contradictions} "
            f"none_is_resolved_marker={str(none_contradictions == 0).lower()}"
        ),
        "raw_unknown": (
            f"raw_unknown_with_none={raw_unknown_none}/{total} "
            f"({raw_unknown_none / max(total, 1) * 100.0:.2f}%)"
        ),
        "severity": severity,
        "none_severity": "fail" if none_contradictions else "info",
        "raw_unknown_watch": raw_unknown_none > 0,
    }


def _classifier_matrix_table(frame: pl.DataFrame, row_col: str, col_col: str) -> pl.DataFrame:
    empty = {
        "row_dimension": [],
        "row_value": [],
        "col_dimension": [],
        "col_value": [],
        "count": [],
        "pct": [],
    }
    if frame.is_empty() or not {row_col, col_col} <= set(frame.columns):
        return pl.DataFrame(empty)
    total = max(frame.height, 1)
    return (
        frame.with_columns(
            pl.col(row_col).cast(pl.Utf8).fill_null("data_error"),
            pl.col(col_col).cast(pl.Utf8).fill_null("data_error"),
        )
        .group_by(row_col, col_col)
        .agg(pl.len().alias("count"))
        .with_columns(
            pl.lit(row_col).alias("row_dimension"),
            pl.col(row_col).alias("row_value"),
            pl.lit(col_col).alias("col_dimension"),
            pl.col(col_col).alias("col_value"),
            (pl.col("count") / total * 100.0).alias("pct"),
        )
        .select("row_dimension", "row_value", "col_dimension", "col_value", "count", "pct")
        .sort(["count", "row_value", "col_value"], descending=[True, False, False])
    )


def _summarize_matrix(table: pl.DataFrame) -> str:
    if table.is_empty():
        return "n/a"
    return ",".join(
        f"{row['row_value']}x{row['col_value']}:{int(row['count'])}"
        for row in table.head(5).iter_rows(named=True)
    )


def _classifier_threshold_table(frame: pl.DataFrame) -> pl.DataFrame:
    numeric_col = ClassifierColumn.RANGE_WIDTH_ATR_THRESHOLD
    if frame.is_empty() or numeric_col not in frame.columns:
        return pl.DataFrame(
            {"metric": [], "value": [], "source": [], "ready": [], "rows": []}
        )
    source_col = ClassifierColumn.RANGE_WIDTH_THRESHOLD_SOURCE
    ready_col = ClassifierColumn.RANGE_WIDTH_THRESHOLD_READY
    stats = frame.select(
        pl.col(numeric_col).cast(pl.Float64).quantile(0.25).alias("q25"),
        pl.col(numeric_col).cast(pl.Float64).median().alias("median"),
        pl.col(numeric_col).cast(pl.Float64).quantile(0.75).alias("q75"),
    ).row(0, named=True)
    rows = [
        {
            "metric": metric,
            "value": float(value) if value is not None else None,
            "source": "quantile",
            "ready": None,
            "rows": frame.height,
        }
        for metric, value in stats.items()
    ]
    if source_col in frame.columns:
        for row in _classifier_distribution_table(frame, source_col).iter_rows(named=True):
            rows.append(
                {
                    "metric": "source",
                    "value": float(row["pct"] or 0.0),
                    "source": str(row["value"]),
                    "ready": None,
                    "rows": int(row["count"] or 0),
                }
            )
    if ready_col in frame.columns:
        ready = _expr_count(frame, pl.col(ready_col).cast(pl.Boolean).fill_null(False))
        rows.append(
            {
                "metric": "ready_pct",
                "value": ready / max(frame.height, 1) * 100.0,
                "source": "readiness",
                "ready": True,
                "rows": ready,
            }
        )
    return pl.DataFrame(rows)


def _summarize_threshold_table(table: pl.DataFrame) -> str:
    if table.is_empty():
        return "n/a"
    return ",".join(
        f"{row['metric']}={float(row['value'] or 0.0):.2f}/{row['source']}"
        for row in table.head(6).iter_rows(named=True)
    )


def _summarize_state_cardinality(frame: pl.DataFrame, columns: list[str]) -> str:
    if frame.is_empty() or not columns:
        return "n/a"
    parts = []
    for column in columns:
        parts.append(f"{column}={int(frame.select(pl.col(column).n_unique()).item() or 0)}")
    return " ".join(parts)


def _classifier_transition_table(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    empty = {
        "state_column": [],
        "from_state": [],
        "to_state": [],
        "count": [],
        "pct": [],
        "self_transition": [],
    }
    values = _state_key_values(frame, column)
    if len(values) < 2:
        return pl.DataFrame(empty)
    transitions = [
        {"from_state": prev, "to_state": curr, "self_transition": prev == curr}
        for prev, curr in zip(values, values[1:], strict=False)
    ]
    total = max(len(transitions), 1)
    return (
        pl.DataFrame(transitions)
        .group_by("from_state", "to_state", "self_transition")
        .agg(pl.len().alias("count"))
        .with_columns(
            pl.lit(column).alias("state_column"),
            (pl.col("count") / total * 100.0).alias("pct"),
        )
        .select("state_column", "from_state", "to_state", "count", "pct", "self_transition")
        .sort(["count", "from_state", "to_state"], descending=[True, False, False])
    )


def _summarize_transitions(table: pl.DataFrame) -> str:
    if table.is_empty():
        return "n/a"
    total = int(table.select(pl.col("count").sum()).item() or 0)
    changed = int(
        table.filter(~pl.col("self_transition")).select(pl.col("count").sum()).item() or 0
    )
    return f"transitions={changed}/{total} changed_rate={changed / max(total, 1) * 100.0:.1f}%"


def _summarize_matrix_like_count(table: pl.DataFrame, count_col: str = "count") -> str:
    if table.is_empty() or count_col not in table.columns:
        return "n/a"
    observations = int(table.select(pl.col(count_col).sum()).item() or 0)
    return f"groups={table.height} observations={observations}"


def _classifier_dwell_table(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    empty = {
        "state_column": [],
        "state": [],
        "runs": [],
        "median_dwell": [],
        "mean_dwell": [],
        "max_dwell": [],
    }
    runs = _state_runs(_state_key_values(frame, column))
    if not runs:
        return pl.DataFrame(empty)
    rows = []
    for state in sorted({state for state, _length in runs}):
        dwell = [length for run_state, length in runs if run_state == state]
        dwell_sorted = sorted(dwell)
        rows.append(
            {
                "state_column": column,
                "state": state,
                "runs": len(dwell),
                "median_dwell": dwell_sorted[len(dwell_sorted) // 2],
                "mean_dwell": sum(dwell) / max(len(dwell), 1),
                "max_dwell": max(dwell),
            }
        )
    return pl.DataFrame(rows).sort(["runs", "state"], descending=[True, False])


def _classifier_time_distribution_table(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    empty = {
        "state_column": [],
        "state": [],
        "time_bucket": [],
        "count": [],
        "pct_of_state": [],
        "pct_of_bucket": [],
    }
    if frame.is_empty() or column not in frame.columns:
        return pl.DataFrame(empty)
    bucket_expr = (
        (pl.col("timestamp").cast(pl.Int64) // (30 * 86_400_000)).cast(pl.Utf8)
        if "timestamp" in frame.columns
        else (pl.int_range(0, pl.len()) // 720).cast(pl.Utf8)
    )
    grouped = (
        frame.with_columns(
            pl.col(column).cast(pl.Utf8).fill_null("data_error").alias("state"),
            bucket_expr.alias("time_bucket"),
        )
        .group_by("state", "time_bucket")
        .agg(pl.len().alias("count"))
    )
    if grouped.is_empty():
        return pl.DataFrame(empty)
    state_totals = grouped.group_by("state").agg(pl.col("count").sum().alias("state_total"))
    bucket_totals = grouped.group_by("time_bucket").agg(
        pl.col("count").sum().alias("bucket_total")
    )
    return (
        grouped.join(state_totals, on="state")
        .join(bucket_totals, on="time_bucket")
        .with_columns(
            pl.lit(column).alias("state_column"),
            (pl.col("count") / pl.col("state_total") * 100.0).alias("pct_of_state"),
            (pl.col("count") / pl.col("bucket_total") * 100.0).alias("pct_of_bucket"),
        )
        .select("state_column", "state", "time_bucket", "count", "pct_of_state", "pct_of_bucket")
        .sort(["count", "state", "time_bucket"], descending=[True, False, False])
    )


@dataclass(frozen=True)
class ReportStatus:
    status: str
    reasons: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.status in ("PASS", "WARN")


@dataclass(frozen=True)
class CandidateGateConfig:
    target_max_dd_pct: float = 5.0
    hard_max_dd_pct: float = 8.0
    max_loss_concentration_pct: float = 60.0
    max_ambiguity_impact_pct: float = 25.0


@dataclass(frozen=True)
class CandidateReportStatus:
    status: str
    classification: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class BucketSummary:
    key: tuple[str, ...]
    trades: int
    net: float
    expectancy: float


@dataclass(frozen=True)
class TimeClusterSummary:
    segments: int
    rows: tuple[BucketSummary, ...]
    best_share_pct: float
    clustered: bool


@dataclass(frozen=True)
class StructuralEventOpportunity:
    failed_breakout_low: int
    failed_breakout_high: int
    bullish_reclaim: int
    bearish_reclaim: int
    breakout_acceptance_high: int
    breakout_acceptance_low: int
    after_volume: int
    after_quality: int
    after_both: int


@dataclass(frozen=True)
class ReclaimExtensionStatus:
    include_reclaim_sweeps: bool
    reclaim_extension_trades: int


def report_status(report: Report, gates: RiskGateConfig) -> ReportStatus:
    reasons: list[str] = []
    if _metadata_value(report, "data_quality") == "data_incomplete":
        incomplete_reason = _metadata_value(report, "data_incomplete_reason")
        if incomplete_reason == "listing_age":
            reasons.append("DATA_INCOMPLETE_LISTING_AGE")
        else:
            reasons.append("DATA_INCOMPLETE")
    metrics = report.metrics
    diagnostics = report.diagnostics
    y = report.yield_attribution
    stop = report.stop_effectiveness
    if metrics.num_trades < gates.min_trades:
        reasons.append("SPARSE")
    if gates.min_pf > 0 and metrics.profit_factor < gates.min_pf:
        reasons.append("PF_LOW")
    if (
        gates.min_expectancy_pct is not None
        and report.trade_expectancy_pct <= gates.min_expectancy_pct
    ):
        reasons.append("EXP_LOW")
    if gates.max_dd_pct is not None and metrics.max_drawdown_pct > gates.max_dd_pct:
        reasons.append("DD_HIGH")
    if (
        gates.max_notional_exposure_pct is not None
        and diagnostics is not None
        and diagnostics.max_notional_exposure_pct > gates.max_notional_exposure_pct
    ):
        reasons.append("NOTIONAL_HIGH")
    if diagnostics is not None and diagnostics.bars_processed < diagnostics.bars:
        reasons.append("STOPPED_EARLY")
    if report.trade_expectancy_pct > 0 and (
        report.trade_expectancy_usd <= 0 or metrics.total_return_pct <= 0
    ):
        reasons.append("YIELD_DIVERGENCE")
    sparse_floor = gates.min_trades if gates.min_trades > 0 else 20
    if math.isinf(metrics.profit_factor) and metrics.num_trades < sparse_floor:
        reasons.append("SPARSE_INF_PF")
    if y.fee_drag_pct > 20.0 and y.gross_profit_usd > 0:
        reasons.append("FEE_DRAG")
    if y.worst_exit_reason != "n/a" and y.worst_exit_expectancy_usd < 0:
        reasons.append("EXIT_LEAK")
    if y.worst_side != "n/a" and y.worst_side_expectancy_usd < 0:
        reasons.append("SIDE_LEAK")
    if y.worst_signal_id != "n/a" and y.worst_signal_expectancy_usd < 0:
        reasons.append("SIGNAL_LEAK")
    if y.loss_to_win_notional_ratio > 1.25:
        reasons.append("LOSS_OVERSIZED")
    if stop.stop_trades > 0 and stop.stop_loss_share_pct > 60.0:
        reasons.append("STOP_DOMINATED_LOSSES")
    if stop.worst_stop_signal_share_pct > 60.0:
        reasons.append("STOP_SIGNAL_LEAK")
    if stop.worst_stop_side_share_pct > 60.0:
        reasons.append("STOP_SIDE_LEAK")
    if diagnostics is not None:
        risk = diagnostics.risk
        lifecycle = diagnostics.lifecycle
        if metrics.max_drawdown_pct > 25.0 and (
            risk.stop_exit_count > 0 or risk.drawdown_stop_pct is not None
        ):
            reasons.append("DD_CONTROL_FAILED")
        if (
            risk.stop_exit_count > 0
            and risk.stop_exit_net_pnl_usd < 0
            and metrics.max_drawdown_pct > 25.0
        ):
            reasons.append("STOP_LOSS_INEFFECTIVE")
        if lifecycle.max_simultaneous_baskets > 1 and metrics.max_drawdown_pct > 25.0:
            reasons.append("BASKET_STACKING_RISK")
        if lifecycle.recovery_actions > 0 and risk.recovery_net_pnl_usd < 0:
            reasons.append("RECOVERY_LOSS_AMPLIFIED")
        if lifecycle.recovery_actions > 0 or risk.recovery_blocked_actions > 0:
            reasons.append("RECOVERY_EXPERIMENTAL")
        if lifecycle.blocked_entry_signals > 0:
            reasons.append("ENTRY_BLOCKED")
        if (
            gates.min_execution_acceptance_pct > 0
            and lifecycle.entry_signals > 0
            and lifecycle.entry_acceptance_rate_pct < gates.min_execution_acceptance_pct
            and lifecycle.min_contract_block_count
            / max(lifecycle.blocked_entry_signals, 1)
            >= 0.5
        ):
            reasons.append("EXECUTION_INFEASIBLE")
        if risk.recovery_cap_breach_actions > 0:
            reasons.append("RECOVERY_CAP_BREACH")
        if risk.recovery_unsized_actions > 0:
            reasons.append("RECOVERY_UNSIZED")
        if risk.recovery_preempted_stop_count > 0:
            reasons.append("RECOVERY_PREEMPTED_STOP")
        if risk.ambiguous_stop_target_count > 0:
            reasons.append("INTRABAR_AMBIGUITY")
    if not reasons:
        return ReportStatus("PASS", ())
    fail_reasons = {
        "PF_LOW",
        "EXP_LOW",
        "DD_HIGH",
        "NOTIONAL_HIGH",
        "LOSS_OVERSIZED",
        "YIELD_DIVERGENCE",
        "DD_CONTROL_FAILED",
        "STOP_LOSS_INEFFECTIVE",
        "BASKET_STACKING_RISK",
        "RECOVERY_LOSS_AMPLIFIED",
        "EXECUTION_INFEASIBLE",
        "STOP_DOMINATED_LOSSES",
        "RECOVERY_CAP_BREACH",
        "DATA_INCOMPLETE",
        "DATA_INCOMPLETE_LISTING_AGE",
    }
    if any(reason in reasons for reason in fail_reasons):
        return ReportStatus("FAIL", tuple(reasons))
    return ReportStatus("WARN", tuple(reasons))


def candidate_report_status(
    report: Report,
    gates: RiskGateConfig,
    candidate_gates: CandidateGateConfig = CandidateGateConfig(),
) -> CandidateReportStatus:
    reasons: list[str] = []
    operational = report_status(report, gates)
    diagnostic_reasons = {
        "EXECUTION_INFEASIBLE",
        "DATA_INCOMPLETE",
        "DATA_INCOMPLETE_LISTING_AGE",
    }
    classification = (
        "DIAGNOSTIC_ONLY"
        if any(reason in operational.reasons for reason in diagnostic_reasons)
        else "FEASIBLE"
    )
    if report.metrics.num_trades <= 0:
        reasons.append("SPARSE")
    if report.metrics.max_drawdown_pct > candidate_gates.hard_max_dd_pct:
        reasons.append("DD_HARD_FAIL")
    elif report.metrics.max_drawdown_pct > candidate_gates.target_max_dd_pct:
        reasons.append("DD_TARGET_HIGH")
    if (
        report.stop_effectiveness.worst_stop_signal_share_pct
        > candidate_gates.max_loss_concentration_pct
    ):
        reasons.append("STOP_SIGNAL_CONCENTRATED")
    if (
        report.stop_effectiveness.worst_stop_side_share_pct
        > candidate_gates.max_loss_concentration_pct
    ):
        reasons.append("STOP_SIDE_CONCENTRATED")
    if report.diagnostics is not None:
        risk = report.diagnostics.risk
        net_pnl = abs(report.yield_attribution.net_pnl_usd)
        ambiguity_pct = (
            abs(risk.ambiguity_impact_usd) / net_pnl * 100.0 if net_pnl > 1e-9 else 0.0
        )
        if (
            risk.ambiguous_stop_target_count > 0
            and ambiguity_pct > candidate_gates.max_ambiguity_impact_pct
        ):
            reasons.append("AMBIGUITY_MATERIAL")
    fail_reasons = {
        "DD_TARGET_HIGH",
        "DD_HARD_FAIL",
        "STOP_SIGNAL_CONCENTRATED",
        "STOP_SIDE_CONCENTRATED",
    }
    if classification == "DIAGNOSTIC_ONLY":
        diagnostic_status_reasons = [
            reason for reason in operational.reasons if reason in diagnostic_reasons
        ]
        diagnostic_report_reasons = [*diagnostic_status_reasons, *reasons]
        return CandidateReportStatus(
            "WARN",
            classification,
            tuple(dict.fromkeys(diagnostic_report_reasons or ["DIAGNOSTIC_ONLY"])),
        )
    if any(reason in reasons for reason in fail_reasons):
        return CandidateReportStatus("FAIL", classification, tuple(reasons))
    if reasons:
        return CandidateReportStatus("WARN", classification, tuple(reasons))
    return CandidateReportStatus("PASS", classification, ())


def _feasible_reports(reports: list[Report], gates: RiskGateConfig) -> list[Report]:
    return [
        report
        for report in reports
        if candidate_report_status(report, gates).classification == "FEASIBLE"
    ]


def _trade_bucket_summary(reports: list[Report], bucket_col: str) -> tuple[int, int, float]:
    frames = []
    for report in reports:
        if not report.trades.is_empty() and bucket_col in report.trades.columns:
            frames.append(report.trades)
    if not frames:
        return 0, 0, 0.0
    trades = pl.concat(frames, how="diagonal_relaxed")
    net_col = "net_pnl_usd" if "net_pnl_usd" in trades.columns else "pnl_usd"
    grouped = (
        trades.filter(pl.col(bucket_col).is_not_null())
        .group_by(bucket_col)
        .agg(
            pl.len().alias("trades"),
            pl.col(net_col).cast(pl.Float64).mean().alias("expectancy"),
        )
    )
    if grouped.is_empty():
        return 0, 0, 0.0
    qualified = grouped.filter(pl.col("trades") >= 2)
    if qualified.is_empty():
        return grouped.height, 0, 0.0
    negative = qualified.filter(pl.col("expectancy") < 0).height
    worst = float(qualified["expectancy"].min() or 0.0)
    return grouped.height, negative, worst


def _strategy_key(report: Report) -> str:
    for item in report.metadata:
        if not item.startswith("strategy_args="):
            continue
        args = item.removeprefix("strategy_args=")
        for part in args.split(","):
            if part.startswith("strategy="):
                return part.removeprefix("strategy=")
    return "reports"


def _metadata_value(report: Report, key: str) -> str:
    prefix = f"{key}="
    for item in report.metadata:
        if item.startswith(prefix):
            return item.removeprefix(prefix)
    return ""


def format_candidate_status_table(
    reports: list[Report],
    gates: RiskGateConfig,
    candidate_gates: CandidateGateConfig = CandidateGateConfig(),
) -> str:
    rows = []
    for report in reports:
        status = candidate_report_status(report, gates, candidate_gates)
        rows.append(
            [
                status.status,
                status.classification,
                report.label,
                str(report.metrics.num_trades),
                f"{report.metrics.max_drawdown_pct:.1f}",
                f"{candidate_gates.target_max_dd_pct:.1f}",
                f"{candidate_gates.hard_max_dd_pct:.1f}",
                f"{report.stop_effectiveness.stop_loss_share_pct:.1f}",
                ",".join(status.reasons) or "-",
            ]
        )
    return format_table(
        [
            "Status",
            "Class",
            "Label",
            "Trades",
            "DD%",
            "TargetDD%",
            "HardDD%",
            "StopLoss%",
            "Reasons",
        ],
        rows,
    )


def format_cross_run_consistency(
    reports: list[Report],
    gates: RiskGateConfig,
) -> str:
    grouped_reports: dict[str, list[Report]] = {}
    for report in reports:
        grouped_reports.setdefault(_strategy_key(report), []).append(report)
    if len(grouped_reports) > 1:
        sections = []
        for strategy, strategy_reports in grouped_reports.items():
            sections.append(
                f"{strategy}\n" + _format_cross_run_consistency_group(strategy_reports, gates)
            )
        return "\n\n".join(sections)
    return _format_cross_run_consistency_group(reports, gates)


def _format_cross_run_consistency_group(
    reports: list[Report],
    gates: RiskGateConfig,
) -> str:
    feasible = _feasible_reports(reports, gates)
    diagnostic_only = len(reports) - len(feasible)
    rows = []
    for report in reports:
        status = candidate_report_status(report, gates)
        rows.append(
            [
                report.label,
                status.classification,
                f"{report.metrics.profit_factor:.2f}",
                f"{report.trade_expectancy_pct:+.2f}",
                f"{report.metrics.max_drawdown_pct:.1f}",
                f"{report.stop_effectiveness.stop_loss_share_pct:.1f}",
                f"{report.diagnostics.lifecycle.entry_acceptance_rate_pct:.1f}"
                if report.diagnostics is not None
                else "n/a",
            ]
        )
    reasons = []
    if len(feasible) < 2:
        reasons.append("INSUFFICIENT_FEASIBLE_ASSETS")
    if len(feasible) >= 2:
        negative_expectancy = sum(1 for report in feasible if report.trade_expectancy_pct < 0.0)
        low_pf = sum(1 for report in feasible if report.metrics.profit_factor < 1.0)
        if negative_expectancy > 0 or low_pf > 0:
            reasons.append("CROSS_ASSET_INCONSISTENT")
    trend_buckets, negative_trends, worst_trend_exp = _trade_bucket_summary(
        feasible, "entry_trend_bucket"
    )
    volatility_buckets, negative_volatility, worst_vol_exp = _trade_bucket_summary(
        feasible, "entry_volatility_bucket"
    )
    if trend_buckets >= 2 and negative_trends > 0:
        reasons.append("CROSS_TREND_INCONSISTENT")
    if volatility_buckets >= 2 and negative_volatility > 0:
        reasons.append("CROSS_VOLATILITY_INCONSISTENT")
    worst_side_structure = _worst_pair_bucket(feasible, ("side", "entry_structure_bucket"))
    worst_stage_event = _worst_pair_bucket(
        feasible,
        ("entry_market_stage_bucket", "entry_liquidity_event_type_bucket"),
    )
    status = "PASS" if not reasons else "FAIL"
    summary = (
        f"Cross-run consistency: {status}  feasible={len(feasible)}  "
        f"diagnostic_only={diagnostic_only}  reasons={','.join(reasons) or '-'}  "
        f"worst_trend_exp=${worst_trend_exp:+.2f}  worst_vol_exp=${worst_vol_exp:+.2f}  "
        f"worst_side_structure={worst_side_structure}  "
        f"worst_stage_event={worst_stage_event}"
    )
    return summary + "\n" + format_table(
        ["Label", "Class", "PF", "Exp%", "DD%", "StopLoss%", "EntryAccept%"], rows
    )


def _hypothesis_rejection_reasons(reports: list[Report]) -> list[str]:
    reasons: list[str] = []
    if not _reclaim_supported(reports):
        reasons.append("RECLAIM_NOT_ROBUST")
    if _eth_only_positive_structure(reports):
        reasons.append("ETH_ONLY_STRUCTURE_EDGE")
        reasons.append("ASSET_SPECIFIC_PATTERN")
    if _none_event_drawdown(reports):
        reasons.append("NONE_EVENT_UNEXPLAINED_LOSS")
    if _none_event_dominant(reports):
        reasons.append("NONE_EVENT_DOMINANT")
    if _none_context_inconsistent(reports):
        reasons.append("NONE_CONTEXT_INCONSISTENT")
    if _none_profit_clustered(reports):
        reasons.append("NONE_PROFIT_CLUSTERED")
    if not _accepted_breakout_consistent(reports):
        reasons.append("ACCEPTED_BREAKOUT_MIXED")
    if _stage_event_inconsistent(reports):
        reasons.append("STAGE_EVENT_INCONSISTENT")
    return reasons


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _bucket_expectancy(report: Report, column: str, bucket: str) -> tuple[int, float]:
    if report.trades.is_empty() or column not in report.trades.columns:
        return 0, 0.0
    net_col = "net_pnl_usd" if "net_pnl_usd" in report.trades.columns else "pnl_usd"
    frame = report.trades.filter(pl.col(column) == bucket)
    if frame.is_empty():
        return 0, 0.0
    value = frame.select(pl.col(net_col).cast(pl.Float64).mean()).item()
    return frame.height, float(value or 0.0)


def _reclaim_supported(reports: list[Report]) -> bool:
    positive_assets = 0
    for report in reports:
        bullish_count, bullish_exp = _bucket_expectancy(
            report,
            "entry_liquidity_event_type_bucket",
            "bullish_reclaim",
        )
        bearish_count, bearish_exp = _bucket_expectancy(
            report,
            "entry_liquidity_event_type_bucket",
            "bearish_reclaim",
        )
        if bullish_count + bearish_count >= 5 and bullish_exp + bearish_exp > 0.0:
            positive_assets += 1
    return positive_assets >= 2


def _accepted_breakout_consistent(reports: list[Report]) -> bool:
    signs = []
    for report in reports:
        high_count, high_exp = _bucket_expectancy(
            report,
            "entry_liquidity_event_type_bucket",
            "breakout_acceptance_high",
        )
        low_count, low_exp = _bucket_expectancy(
            report,
            "entry_liquidity_event_type_bucket",
            "breakout_acceptance_low",
        )
        if high_count + low_count >= 3:
            signs.append((high_exp + low_exp) > 0.0)
    return len(signs) >= 2 and (all(signs) or not any(signs))


def _eth_only_positive_structure(reports: list[Report]) -> bool:
    positives = []
    for report in reports:
        label = report.label.upper()
        _, uptrend_exp = _bucket_expectancy(report, "entry_structure_bucket", "uptrend")
        if uptrend_exp > 0.0:
            positives.append("ETH" if "ETH" in label else label)
    return positives == ["ETH"]


def _none_event_drawdown(reports: list[Report]) -> bool:
    for report in reports:
        count, exp = _bucket_expectancy(report, "entry_liquidity_event_type_bucket", "none")
        if count >= 1 and exp < 0.0:
            return True
    return False


def _none_event_dominant(reports: list[Report]) -> bool:
    for report in reports:
        if report.trades.is_empty():
            continue
        none_trades = _none_event_trades(report.trades)
        if none_trades.height / max(report.trades.height, 1) > 0.60:
            return True
    return False


def _none_context_inconsistent(reports: list[Report]) -> bool:
    signs: dict[str, set[bool]] = {}
    for report in reports:
        none_trades = _none_event_trades(report.trades)
        if none_trades.is_empty():
            continue
        for column in (
            "entry_atr_percentile_bucket",
            "entry_key_level_proximity_bucket",
            "entry_z_pressure_side_bucket",
        ):
            if column not in none_trades.columns:
                continue
            net_col = "net_pnl_usd" if "net_pnl_usd" in none_trades.columns else "pnl_usd"
            grouped = none_trades.group_by(column).agg(
                pl.len().alias("trades"),
                pl.col(net_col).cast(pl.Float64).mean().alias("expectancy"),
            )
            for row in grouped.iter_rows(named=True):
                if int(row["trades"] or 0) < 2:
                    continue
                key = f"{column}:{row[column] or 'unknown'}"
                signs.setdefault(key, set()).add(float(row["expectancy"] or 0.0) > 0.0)
    return any(len(values) > 1 for values in signs.values())


def _none_profit_clustered(reports: list[Report]) -> bool:
    for report in reports:
        stats = _none_time_clustering_stats(_none_event_trades(report.trades))
        if stats["clustered"] == "yes":
            return True
    return False


def _stage_event_inconsistent(reports: list[Report]) -> bool:
    negative = 0
    positive = 0
    for report in reports:
        _, value = _worst_pair_bucket_value(
            report,
            ("entry_market_stage_bucket", "entry_liquidity_event_type_bucket"),
        )
        if value < 0.0:
            negative += 1
        elif value > 0.0:
            positive += 1
    return negative > 0 and positive > 0


def _worst_pair_bucket(reports: list[Report], columns: tuple[str, str]) -> str:
    worst_label = "n/a"
    worst_value = 0.0
    for report in reports:
        label, value = _worst_pair_bucket_value(report, columns)
        if label != "n/a" and (worst_label == "n/a" or value < worst_value):
            worst_label = f"{report.label}:{label}"
            worst_value = value
    return f"{worst_label}/${worst_value:+.2f}" if worst_label != "n/a" else "n/a"


def _worst_pair_bucket_value(report: Report, columns: tuple[str, str]) -> tuple[str, float]:
    if report.trades.is_empty() or not set(columns) <= set(report.trades.columns):
        return "n/a", 0.0
    net_col = "net_pnl_usd" if "net_pnl_usd" in report.trades.columns else "pnl_usd"
    grouped = (
        report.trades.group_by(list(columns))
        .agg(
            pl.len().alias("trades"),
            pl.col(net_col).cast(pl.Float64).mean().alias("expectancy"),
        )
        .filter(pl.col("trades") >= 2)
        .sort("expectancy")
    )
    if grouped.is_empty():
        return "n/a", 0.0
    row = grouped.row(0, named=True)
    label = "/".join(str(row[column] or "n/a") for column in columns)
    return label, float(row["expectancy"] or 0.0)


def feature_layer(frame: pl.DataFrame) -> tuple[str, str]:
    candidates = [
        col
        for col in (
            "volatility_ratio",
            "volatility_regime",
            "prior_liquidity_high",
            "prior_liquidity_low",
            "bullish_liquidity_sweep",
            "bearish_liquidity_sweep",
            "failed_bullish_sweep",
            "failed_bearish_sweep",
            "breakout_acceptance_high",
            "breakout_acceptance_low",
            "failed_breakout_high",
            "failed_breakout_low",
            "event_quality_score",
            "liquidity_event_type",
            "volume_impulse",
            "sweep_distance_atr",
            "structure_trend_state",
            "market_stage",
            "structure_reason",
            "market_stage_reason",
            "stage_unknown_reason",
            "range_width_atr",
            "range_compression",
            "near_range_high",
            "near_range_low",
        )
        if col in frame.columns
    ]
    if not candidates:
        return "FEATURE", "n/a"
    parts = []
    for col in candidates:
        null_pct = frame[col].null_count() / max(frame.height, 1) * 100.0
        parts.append(f"{col}:null={null_pct:.1f}%")
    return "FEATURE", "; ".join(parts)


def signal_layer(values: dict[str, float]) -> tuple[str, str]:
    keys = (
        "raw_entry_any_pct",
        "entry_event_pct",
        "held_signal_pct",
        "long_held_signal_pct",
        "short_held_signal_pct",
        "high_volatility_regime_pct",
    )
    parts = [f"{key}={values[key]:.1f}" for key in keys if key in values]
    return "SIGNAL", " ".join(parts) if parts else "n/a"


def lifecycle_layer(report: Report) -> tuple[str, str]:
    d = report.diagnostics
    if d is None:
        return "LIFECYCLE", "n/a"
    reasons = ",".join(f"{key}:{value}" for key, value in sorted(d.exit_reasons.items())) or "none"
    lifecycle = d.lifecycle
    block_reasons = ",".join(
        f"{key}:{value}" for key, value in sorted(lifecycle.blocked_entry_reasons.items())
    ) or "none"
    return (
        "LIFECYCLE",
        f"entry_signals={lifecycle.entry_signals} entries={d.entries} "
        f"accept={lifecycle.entry_acceptance_rate_pct:.1f}% exits={d.exits} "
        f"grid={lifecycle.grid_actions} "
        f"hedge={lifecycle.hedge_actions} recovery={lifecycle.recovery_actions} "
        f"max_baskets={lifecycle.max_simultaneous_baskets} "
        f"blocked_entries={lifecycle.blocked_entry_signals} "
        f"duplicate={lifecycle.duplicate_entry_suppressed} "
        f"capacity_blocked={lifecycle.capacity_blocked_entries} "
        f"sizing_blocked={lifecycle.sizing_blocked_entries} "
        f"min_contract_blocks={lifecycle.min_contract_block_count} "
        f"median_required_capital=${lifecycle.median_required_capital_for_min_contract:.2f} "
        f"median_required_risk_pct={lifecycle.median_required_risk_pct_for_min_contract:.2f}% "
        f"block_reasons={block_reasons} "
        f"same_bar_exit_entry={lifecycle.same_bar_exit_entry_count} "
        f"avg_hold={d.avg_bars_held:.1f} reasons={reasons}",
    )


def sizing_layer(report: Report) -> tuple[str, str]:
    d = report.diagnostics
    if d is None:
        return "SIZING", "n/a"
    y = report.yield_attribution
    risk = d.risk
    return (
        "SIZING",
        f"avg_contracts={d.avg_active_exposure:.2f} max_contracts={d.max_active_exposure:.2f} "
        f"avg_notional={d.avg_notional_exposure_pct:.1f}% "
        f"max_notional={d.max_notional_exposure_pct:.1f}% "
        f"size_w_exp={y.size_weighted_expectancy_pct:+.2f}% "
        f"loss/win_notional={y.loss_to_win_notional_ratio:.2f} "
        f"net=${y.net_pnl_usd:+.2f} fee_drag={y.fee_drag_pct:.1f}% "
        f"stops={risk.stop_exit_count} stop_net=${risk.stop_exit_net_pnl_usd:+.2f} "
        f"recovered_stops={risk.recovered_stop_exit_count} "
        f"stop_loss_share={report.stop_effectiveness.stop_loss_share_pct:.1f}% "
        f"recovery_preempted_stop={risk.recovery_preempted_stop_count} "
        f"recovery_cap_breach={risk.recovery_cap_breach_actions} "
        f"recovery_net=${risk.recovery_net_pnl_usd:+.2f}",
    )


def strategy_development_rows(
    report: Report,
    signal_frame: pl.DataFrame,
    strategy: StrategyBehavior,
) -> list[list[str]]:
    rows: list[list[str]] = []
    rows.append(_signal_development_row(report, signal_frame, strategy))
    rows.append(_structure_development_row(report, signal_frame, strategy))
    rows.append(_liquidity_development_row(report, signal_frame, strategy))
    rows.extend(_structural_event_rows(report, signal_frame, strategy))
    rows.append(_none_coverage_row(report))
    rows.append(_structural_overlap_row(report))
    rows.append(_cross_structure_liquidity_row(report))
    rows.append(_interpretability_note_row(report))
    rows.extend(_filter_attrition_rows(signal_frame, strategy))
    rows.append(_trade_path_row(report, signal_frame))
    rows.append(_loss_cause_row(report))
    rows.append(_loss_confidence_row(report))
    rows.append(_production_design_row(report))
    return rows


def format_strategy_development_summary(
    label: str,
    report: Report,
    signal_frame: pl.DataFrame,
    strategy: StrategyBehavior,
) -> str:
    return f"{label}\n" + format_table(
        ["Diagnosis", "Summary"],
        strategy_development_rows(report, signal_frame, strategy),
    )


def state_diagnostics_rows(report: Report, signal_frame: pl.DataFrame) -> list[list[str]]:
    return [
        _format_mtf_confirmation_attribution_row(report),
        _format_mtf_context_coverage_row(signal_frame),
        _format_mtf_higher_timeframe_state_row(signal_frame),
        _format_structure_unknown_breakdown_row(signal_frame),
        _format_mtf_state_cardinality_row(signal_frame),
        _format_mtf_state_stability_row(signal_frame),
        _format_mtf_right_edge_drift_row(signal_frame),
        _format_mtf_state_attribution_row(report),
        _format_mtf_state_event_row(report),
        _format_mtf_state_separation_row(report),
        _format_mtf_state_operability_row(report),
        _format_mtf_state_time_consistency_row(report),
        _format_structural_side_event_stage_row(report),
        _format_structural_event_time_clustering_row(report),
    ]


def format_state_diagnostics_summary(
    label: str,
    report: Report,
    signal_frame: pl.DataFrame,
) -> str:
    return f"{label}\n" + format_table(
        ["State diagnostic", "Summary"],
        state_diagnostics_rows(report, signal_frame),
    )


def state_diagnostics_export_frame(
    label: str,
    report: Report,
    signal_frame: pl.DataFrame,
) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"label": label, "diagnostic": diagnostic, "summary": summary}
            for diagnostic, summary in state_diagnostics_rows(report, signal_frame)
        ]
    )


def _signal_development_row(
    report: Report,
    signal_frame: pl.DataFrame,
    strategy: StrategyBehavior,
) -> list[str]:
    bars = signal_frame.height
    raw_opportunities = _raw_opportunity_count(signal_frame, strategy)
    filtered_events = _expr_count(signal_frame, pl.col("entry_signal") != 0)
    accepted_entries = report.diagnostics.lifecycle.entry_actions if report.diagnostics else 0
    raw_to_filtered = filtered_events / raw_opportunities * 100.0 if raw_opportunities else 0.0
    filtered_to_accepted = accepted_entries / filtered_events * 100.0 if filtered_events else 0.0
    trade_density = report.metrics.num_trades / bars * 1000.0 if bars else 0.0
    return [
        "Signal opportunity",
        f"bars={bars} raw_opportunities={raw_opportunities} "
        f"entry_events={filtered_events} accepted_entries={accepted_entries} "
        f"raw_to_entry={raw_to_filtered:.1f}% entry_to_accept={filtered_to_accepted:.1f}% "
        f"trade_density_per_1000_bars={trade_density:.2f}",
    ]


def _liquidity_development_row(
    report: Report,
    signal_frame: pl.DataFrame,
    strategy: StrategyBehavior,
) -> list[str]:
    if not isinstance(strategy, StrategySpec):
        return ["Liquidity sweep opportunity", "n/a"]
    raw_expr = _raw_opportunity_expr(strategy)
    raw_opportunities = _expr_count(signal_frame, raw_expr)
    near_pool = _expr_count(
        signal_frame,
        raw_expr
        & (
            pl.col("swept_high").fill_null(False)
            | pl.col("swept_low").fill_null(False)
            if {"swept_high", "swept_low"} <= set(signal_frame.columns)
            else pl.lit(False)
        ),
    )
    reclaim = _expr_count(
        signal_frame,
        raw_expr
        & (
            pl.col("bullish_liquidity_sweep").fill_null(False)
            | pl.col("bearish_liquidity_sweep").fill_null(False)
            if {"bullish_liquidity_sweep", "bearish_liquidity_sweep"} <= set(signal_frame.columns)
            else pl.lit(False)
        ),
    )
    failed = _expr_count(
        signal_frame,
        raw_expr
        & (
            pl.col("failed_bullish_sweep").fill_null(False)
            | pl.col("failed_bearish_sweep").fill_null(False)
            if {"failed_bullish_sweep", "failed_bearish_sweep"} <= set(signal_frame.columns)
            else pl.lit(False)
        ),
    )
    accepted_breakout = _expr_count(
        signal_frame,
        raw_expr
        & (
            pl.col("breakout_acceptance_high").fill_null(False)
            | pl.col("breakout_acceptance_low").fill_null(False)
            if {"breakout_acceptance_high", "breakout_acceptance_low"}
            <= set(signal_frame.columns)
            else pl.lit(False)
        ),
    )
    failed_breakout = _expr_count(
        signal_frame,
        raw_expr
        & (
            pl.col("failed_breakout_high").fill_null(False)
            | pl.col("failed_breakout_low").fill_null(False)
            if {"failed_breakout_high", "failed_breakout_low"} <= set(signal_frame.columns)
            else pl.lit(False)
        ),
    )
    quality_avg = _filtered_mean(signal_frame, raw_expr, "event_quality_score")
    trade_parts = _sweep_trade_expectancy(report)
    return [
        "Liquidity sweep opportunity",
        f"raw_opportunities={raw_opportunities} near_liquidity={near_pool} "
        f"sweep_reclaim={reclaim} failed_sweep={failed} "
        f"accepted_breakout={accepted_breakout} failed_breakout={failed_breakout} "
        f"avg_event_quality={quality_avg:.2f} {trade_parts} {_event_trade_expectancy(report)}",
    ]


def _structural_event_rows(
    report: Report,
    signal_frame: pl.DataFrame,
    strategy: StrategyBehavior,
) -> list[list[str]]:
    opportunity = _structural_event_opportunity(signal_frame)
    return [
        _format_structural_event_opportunity_row(opportunity),
        _format_structural_event_quality_row(report, signal_frame),
        _format_structural_event_volume_row(report, signal_frame),
        _format_mtf_confirmation_opportunity_row(signal_frame),
        _format_mtf_confirmation_attribution_row(report),
        _format_mtf_confirmation_reason_row(signal_frame),
        _format_mtf_context_coverage_row(signal_frame),
        _format_mtf_higher_timeframe_state_row(signal_frame),
        _format_mtf_confirmation_htf_state_row(signal_frame),
        _format_structure_unknown_breakdown_row(signal_frame),
        _format_mtf_state_cardinality_row(signal_frame),
        _format_mtf_state_stability_row(signal_frame),
        _format_mtf_right_edge_drift_row(signal_frame),
        _format_mtf_state_attribution_row(report),
        _format_mtf_state_event_row(report),
        _format_mtf_state_separation_row(report),
        _format_mtf_state_operability_row(report),
        _format_mtf_state_time_consistency_row(report),
        _format_structural_event_attribution_row(report),
        _format_structural_side_structure_row(report),
        _format_structural_side_event_row(report),
        _format_structural_side_event_stage_row(report),
        _format_structural_quality_attribution_row(report),
        _format_structural_volume_attribution_row(report),
        _format_structural_event_time_clustering_row(report),
        _format_reclaim_extension_status_row(_reclaim_extension_status(report, strategy)),
    ]


def _structural_event_opportunity(
    signal_frame: pl.DataFrame,
    event_quality_min: float = 1.5,
) -> StructuralEventOpportunity:
    if signal_frame.is_empty():
        return StructuralEventOpportunity(0, 0, 0, 0, 0, 0, 0, 0, 0)
    failed_expr = _structural_failed_breakout_expr(signal_frame)
    volume_expr = (
        pl.col("volume_impulse").fill_null(False)
        if "volume_impulse" in signal_frame.columns
        else pl.lit(False)
    )
    quality_expr = (
        pl.col("event_quality_score").cast(pl.Float64) >= event_quality_min
        if "event_quality_score" in signal_frame.columns
        else pl.lit(False)
    )
    failed_low_expr = _structural_failed_breakout_low_expr(signal_frame)
    failed_high_expr = _structural_failed_breakout_high_expr(signal_frame)
    return StructuralEventOpportunity(
        failed_breakout_low=_expr_count(signal_frame, failed_low_expr),
        failed_breakout_high=_expr_count(signal_frame, failed_high_expr),
        bullish_reclaim=_event_count(signal_frame, LiquidityEvent.BULLISH_RECLAIM),
        bearish_reclaim=_event_count(signal_frame, LiquidityEvent.BEARISH_RECLAIM),
        breakout_acceptance_high=_event_count(
            signal_frame, LiquidityEvent.BREAKOUT_ACCEPTANCE_HIGH
        ),
        breakout_acceptance_low=_event_count(signal_frame, LiquidityEvent.BREAKOUT_ACCEPTANCE_LOW),
        after_volume=_expr_count(signal_frame, failed_expr & volume_expr),
        after_quality=_expr_count(signal_frame, failed_expr & quality_expr),
        after_both=_expr_count(signal_frame, failed_expr & volume_expr & quality_expr),
    )


def _format_structural_event_opportunity_row(summary: StructuralEventOpportunity) -> list[str]:
    return [
        "Structural event opportunity",
        f"failed_breakout={summary.failed_breakout_low + summary.failed_breakout_high} "
        f"low={summary.failed_breakout_low} high={summary.failed_breakout_high} "
        f"reclaim={summary.bullish_reclaim + summary.bearish_reclaim} "
        f"accepted_breakout={summary.breakout_acceptance_high + summary.breakout_acceptance_low} "
        f"after_volume={summary.after_volume} after_quality={summary.after_quality} "
        f"after_both={summary.after_both}",
    ]


def _format_structural_event_quality_row(
    report: Report,
    signal_frame: pl.DataFrame,
) -> list[str]:
    if "event_quality_score" not in signal_frame.columns:
        return ["Structural event quality", "n/a"]
    avg_quality = _filtered_mean(
        signal_frame,
        _structural_failed_breakout_expr(signal_frame),
        "event_quality_score",
    )
    summaries = _trade_bucket_summaries(
        report.trades,
        (DiagnosticColumn.ENTRY_EVENT_QUALITY_BUCKET,),
        limit=4,
    )
    return [
        "Structural event quality",
        f"failed_avg_quality={avg_quality:.2f} "
        f"{_format_bucket_summaries('quality_trade_expectancy', summaries)}",
    ]


def _format_structural_event_volume_row(
    report: Report,
    signal_frame: pl.DataFrame,
) -> list[str]:
    if "volume_impulse" not in signal_frame.columns:
        return ["Structural event volume impulse", "n/a"]
    failed_expr = _structural_failed_breakout_expr(signal_frame)
    failed_count = _expr_count(signal_frame, failed_expr)
    volume_count = _expr_count(
        signal_frame,
        failed_expr & pl.col("volume_impulse").fill_null(False),
    )
    summaries = _trade_bucket_summaries(report.trades, ("entry_volume_impulse",), limit=2)
    return [
        "Structural event volume impulse",
        f"failed_with_volume={volume_count}/{failed_count} "
        f"{_format_bucket_summaries('volume_trade_expectancy', summaries)}",
    ]


def _has_mtf_confirmation_columns(frame: pl.DataFrame) -> bool:
    return {
        "m15_confirm_long",
        "m15_confirm_short",
        "m15_confirm_reason",
        "m15_confirm_available",
    } <= set(frame.columns)


def _format_mtf_confirmation_opportunity_row(signal_frame: pl.DataFrame) -> list[str]:
    if not _has_mtf_confirmation_columns(signal_frame):
        return ["MTF confirmation opportunity", "n/a"]
    failed_expr = _structural_failed_breakout_expr(signal_frame)
    failed_low_expr = _structural_failed_breakout_low_expr(signal_frame)
    failed_high_expr = _structural_failed_breakout_high_expr(signal_frame)
    raw = _expr_count(signal_frame, failed_expr)
    available = _expr_count(
        signal_frame,
        failed_expr & pl.col("m15_confirm_available").fill_null(False),
    )
    long_pass = _expr_count(
        signal_frame,
        failed_low_expr & pl.col("m15_confirm_long").fill_null(False),
    )
    short_pass = _expr_count(
        signal_frame,
        failed_high_expr & pl.col("m15_confirm_short").fill_null(False),
    )
    passed = long_pass + short_pass
    pass_rate = passed / raw * 100.0 if raw else 0.0
    return [
        "MTF confirmation opportunity",
        f"raw_structural_candidates={raw} confirm_available={available} "
        f"confirm_pass_long={long_pass} confirm_pass_short={short_pass} "
        f"confirm_pass_rate={pass_rate:.1f}%",
    ]


def _format_mtf_confirmation_attribution_row(report: Report) -> list[str]:
    trades = report.trades
    if trades.is_empty() or "entry_m15_confirm_available" not in trades.columns:
        return ["MTF confirmation attribution", "n/a"]
    if "side" not in trades.columns or not {
        "entry_m15_confirm_long",
        "entry_m15_confirm_short",
    } <= set(trades.columns):
        confirmed = pl.col("entry_m15_confirm_available").cast(pl.Boolean).fill_null(False)
    else:
        confirmed = pl.when(pl.col("side").is_in(("long", "buy")))
        confirmed = confirmed.then(
            pl.col("entry_m15_confirm_long").cast(pl.Boolean).fill_null(False)
        )
        confirmed = confirmed.when(pl.col("side").is_in(("short", "sell"))).then(
            pl.col("entry_m15_confirm_short").cast(pl.Boolean).fill_null(False)
        )
        confirmed = confirmed.otherwise(False)
    work = trades.with_columns(
        pl.when(confirmed)
        .then(pl.lit("confirmed"))
        .otherwise(pl.lit("unconfirmed"))
        .alias("_mtf_confirm_bucket")
    )
    return [
        "MTF confirmation attribution",
        _format_attribution_table(work, ("_mtf_confirm_bucket",), "mtf_confirm_x_result", limit=2),
    ]


def _format_mtf_confirmation_reason_row(signal_frame: pl.DataFrame) -> list[str]:
    if not _has_mtf_confirmation_columns(signal_frame):
        return ["MTF confirmation reason", "n/a"]
    failed_expr = _structural_failed_breakout_expr(signal_frame)
    total = _expr_count(signal_frame, failed_expr)
    breakout = _expr_count(
        signal_frame,
        failed_expr & pl.col("m15_confirm_reason").str.contains("breakout").fill_null(False),
    )
    macd = _expr_count(
        signal_frame,
        failed_expr & pl.col("m15_confirm_reason").str.contains("macd").fill_null(False),
    )
    unavailable = _expr_count(
        signal_frame,
        failed_expr & ~pl.col("m15_confirm_available").fill_null(False),
    )
    return [
        "MTF confirmation reason",
        f"failed_breakout={total} breakout_confirm={breakout} "
        f"macd_confirm={macd} unavailable={unavailable}",
    ]


def _format_mtf_context_coverage_row(signal_frame: pl.DataFrame) -> list[str]:
    if signal_frame.is_empty():
        return ["MTF context coverage", "n/a"]
    parts = []
    for prefix in ("m15", "h4", "d1"):
        column = (
            f"{prefix}_confirm_available" if prefix == "m15" else f"{prefix}_context_available"
        )
        if column not in signal_frame.columns:
            parts.append(f"{prefix}_available=n/a")
            continue
        available = _expr_count(signal_frame, pl.col(column).cast(pl.Boolean).fill_null(False))
        rate = available / max(signal_frame.height, 1) * 100.0
        parts.append(f"{prefix}_available={available}/{signal_frame.height} ({rate:.1f}%)")
    return ["MTF context coverage", " ".join(parts)]


def _format_mtf_higher_timeframe_state_row(signal_frame: pl.DataFrame) -> list[str]:
    if signal_frame.is_empty():
        return ["MTF higher-timeframe state", "n/a"]
    parts = []
    for prefix in ("h4", "d1"):
        column = f"{prefix}_trend_state"
        if column not in signal_frame.columns:
            parts.append(f"{prefix}=n/a")
            continue
        counts = (
            signal_frame.filter(pl.col(column).is_not_null())
            .group_by(column)
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )
        if counts.is_empty():
            parts.append(f"{prefix}=none")
            continue
        state_counts = ",".join(
            f"{row[column]}:{row['count']}" for row in counts.iter_rows(named=True)
        )
        parts.append(f"{prefix}={state_counts}")
    return ["MTF higher-timeframe state", " ".join(parts)]


def _format_mtf_confirmation_htf_state_row(signal_frame: pl.DataFrame) -> list[str]:
    if not _has_mtf_confirmation_columns(signal_frame):
        return ["MTF confirmation x HTF state", "n/a"]
    failed_expr = _structural_failed_breakout_expr(signal_frame)
    confirmed_expr = pl.col("m15_confirm_long").fill_null(False) | pl.col(
        "m15_confirm_short"
    ).fill_null(False)
    parts = []
    for prefix in ("h4", "d1"):
        column = f"{prefix}_trend_state"
        if column not in signal_frame.columns:
            parts.append(f"{prefix}=n/a")
            continue
        work = signal_frame.filter(failed_expr & pl.col(column).is_not_null()).with_columns(
            pl.when(confirmed_expr)
            .then(pl.lit("confirmed"))
            .otherwise(pl.lit("unconfirmed"))
            .alias("_confirm_bucket")
        )
        if work.is_empty():
            parts.append(f"{prefix}=none")
            continue
        counts = work.group_by(column, "_confirm_bucket").agg(pl.len().alias("count"))
        state_counts = ",".join(
            f"{row[column]}:{row['_confirm_bucket']}:{row['count']}"
            for row in counts.iter_rows(named=True)
        )
        parts.append(f"{prefix}={state_counts}")
    return ["MTF confirmation x HTF state", " ".join(parts)]


def _format_structure_unknown_breakdown_row(signal_frame: pl.DataFrame) -> list[str]:
    if signal_frame.is_empty():
        return ["Structure stage unknown breakdown", "n/a"]
    columns = [
        column
        for column in ("market_stage", "market_stage_reason", "stage_unknown_reason")
        if column in signal_frame.columns
    ]
    if not columns:
        return ["Structure stage unknown breakdown", "n/a"]
    parts = []
    for column in columns:
        counts = (
            signal_frame.with_columns(pl.col(column).cast(pl.Utf8).fill_null("data_error"))
            .group_by(column)
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )
        labels = []
        for row in counts.head(6).iter_rows(named=True):
            count = int(row["count"] or 0)
            pct = count / max(signal_frame.height, 1) * 100.0
            labels.append(f"{row[column]}:{count}/{pct:.1f}%")
        raw_unknown = _expr_count(
            signal_frame,
            pl.col(column).cast(pl.Utf8).fill_null("data_error") == "unknown",
        )
        parts.append(f"{column}={','.join(labels)} raw_unknown={raw_unknown}")
    return ["Structure stage unknown breakdown", " ".join(parts)]


def _format_mtf_state_cardinality_row(signal_frame: pl.DataFrame) -> list[str]:
    if signal_frame.is_empty():
        return ["MTF state cardinality", "n/a"]
    parts = []
    for column in ("mtf_state_key", "mtf_structure_key", "mtf_stage_key"):
        if column not in signal_frame.columns:
            parts.append(f"{column}=n/a")
            continue
        unique_count = signal_frame.select(pl.col(column).n_unique()).item()
        value_counts = (
            signal_frame.with_columns(pl.col(column).cast(pl.Utf8).fill_null("data_error"))
            .group_by(column)
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )
        active = value_counts.filter(pl.col("count") >= 10).height
        verdict = "high_cardinality" if int(unique_count or 0) > 30 else "ok"
        parts.append(f"{column}={unique_count} active_ge10={active}/{verdict}")
    return ["MTF state cardinality", " ".join(parts)]


def _state_runs(values: list[str]) -> list[tuple[str, int]]:
    if not values:
        return []
    runs: list[tuple[str, int]] = []
    current = values[0]
    length = 1
    for value in values[1:]:
        if value == current:
            length += 1
            continue
        runs.append((current, length))
        current = value
        length = 1
    runs.append((current, length))
    return runs


def _state_key_values(signal_frame: pl.DataFrame, column: str = "mtf_state_key") -> list[str]:
    if signal_frame.is_empty() or column not in signal_frame.columns:
        return []
    return [str(value or "data_error") for value in signal_frame[column].to_list()]


def _format_mtf_state_stability_row(signal_frame: pl.DataFrame) -> list[str]:
    values = _state_key_values(signal_frame)
    if not values:
        return ["MTF state stability", "n/a"]
    transitions = sum(1 for prev, curr in zip(values, values[1:], strict=False) if prev != curr)
    transition_rate = transitions / max(len(values) - 1, 1) * 100.0
    runs = _state_runs(values)
    dwell_values = sorted(length for _state, length in runs)
    median_dwell = dwell_values[len(dwell_values) // 2] if dwell_values else 0
    mean_dwell = sum(dwell_values) / max(len(dwell_values), 1)
    verdict = "churn_high" if transition_rate > 35.0 or median_dwell <= 2 else "ok"
    return [
        "MTF state stability",
        f"transitions={transitions}/{max(len(values) - 1, 0)} rate={transition_rate:.1f}% "
        f"runs={len(runs)} median_dwell={median_dwell} mean_dwell={mean_dwell:.1f} {verdict}",
    ]


def _format_mtf_right_edge_drift_row(signal_frame: pl.DataFrame, edge_bars: int = 12) -> list[str]:
    values = _state_key_values(signal_frame)
    if len(values) < 3:
        return ["MTF right-edge drift", "n/a"]
    transitions = [1 if prev != curr else 0 for prev, curr in zip(values, values[1:], strict=False)]
    edge = transitions[-edge_bars:]
    history = transitions[: -edge_bars] if len(transitions) > edge_bars else transitions
    edge_rate = sum(edge) / max(len(edge), 1) * 100.0
    history_rate = sum(history) / max(len(history), 1) * 100.0
    drift_threshold = max(history_rate * 2.0, history_rate + 25.0)
    verdict = "right_edge_drift" if edge_rate > drift_threshold else "ok"
    return [
        "MTF right-edge drift",
        f"edge_bars={min(edge_bars, len(values) - 1)} edge_rate={edge_rate:.1f}% "
        f"history_rate={history_rate:.1f}% {verdict}",
    ]


def _format_mtf_state_attribution_row(report: Report) -> list[str]:
    trades = report.trades
    state_col = _first_column(trades, ("entry_mtf_state_bucket", "entry_mtf_state_key"))
    return [
        "MTF state attribution",
        _format_attribution_table(trades, (state_col,), "mtf_state", limit=6),
    ]


def _format_mtf_state_event_row(report: Report) -> list[str]:
    trades = _structural_event_trades(report.trades)
    state_col = _first_column(trades, ("entry_mtf_state_bucket", "entry_mtf_state_key"))
    event_col = _event_column(trades)
    return [
        "MTF state x event",
        _format_attribution_table(
            trades,
            (state_col, event_col, "side"),
            "mtf_state_x_event",
            limit=8,
        ),
    ]


def _format_mtf_state_separation_row(report: Report, min_trades: int = 10) -> list[str]:
    trades = _structural_event_trades(report.trades)
    state_col = _first_column(trades, ("entry_mtf_state_bucket", "entry_mtf_state_key"))
    event_col = _event_column(trades)
    if trades.is_empty() or not state_col or not event_col or not {state_col, event_col} <= set(
        trades.columns
    ):
        return ["MTF state separation", "n/a"]
    net_col = _net_col(trades)
    grouped = (
        trades.with_columns(
            pl.col(state_col).cast(pl.Utf8).fill_null("unknown"),
            pl.col(event_col).cast(pl.Utf8).fill_null("none"),
            pl.col("side").cast(pl.Utf8).fill_null("unknown")
            if "side" in trades.columns
            else pl.lit("unknown").alias("side"),
            pl.col(net_col).cast(pl.Float64).alias("_bucket_net"),
        )
        .group_by(state_col, event_col, "side")
        .agg(
            pl.len().alias("trades"),
            pl.col("_bucket_net").sum().alias("net"),
            pl.col("_bucket_net").mean().alias("expectancy"),
        )
    )
    actionable = grouped.filter(pl.col("trades") >= min_trades)
    positive = actionable.filter(pl.col("expectancy") > 0.0).height
    negative = actionable.filter(pl.col("expectancy") < 0.0).height
    sparse = grouped.filter(pl.col("trades") < min_trades).height
    best = "n/a"
    worst = "n/a"
    if not actionable.is_empty():
        best_row = actionable.sort("expectancy", descending=True).row(0, named=True)
        worst_row = actionable.sort("expectancy").row(0, named=True)
        best = _state_group_label(best_row, (state_col, event_col, "side"))
        worst = _state_group_label(worst_row, (state_col, event_col, "side"))
    return [
        "MTF state separation",
        f"groups={grouped.height} actionable_ge{min_trades}={actionable.height} "
        f"positive={positive} negative={negative} sparse={sparse} "
        f"best={best} worst={worst}",
    ]


def _format_mtf_state_operability_row(report: Report, min_trades: int = 10) -> list[str]:
    trades = _structural_event_trades(report.trades)
    state_col = _first_column(trades, ("entry_mtf_state_bucket", "entry_mtf_state_key"))
    if trades.is_empty() or not state_col or state_col not in trades.columns:
        return ["MTF state operability", "n/a"]
    net_col = _net_col(trades)
    grouped = (
        trades.with_columns(
            pl.col(state_col).cast(pl.Utf8).fill_null("data_error"),
            pl.col(net_col).cast(pl.Float64).alias("_bucket_net"),
        )
        .group_by(state_col)
        .agg(
            pl.len().alias("trades"),
            pl.col("_bucket_net").sum().alias("net"),
            pl.col("_bucket_net").mean().alias("expectancy"),
            (pl.col("_bucket_net") > 0.0).cast(pl.Int64).sum().alias("wins"),
            pl.when(pl.col("_bucket_net") > 0.0)
            .then(pl.col("_bucket_net"))
            .otherwise(0.0)
            .max()
            .alias("max_win"),
            pl.when(pl.col("_bucket_net") > 0.0)
            .then(pl.col("_bucket_net"))
            .otherwise(0.0)
            .sum()
            .alias("gross_profit"),
        )
        .sort("expectancy", descending=True)
    )
    candidates = grouped.filter((pl.col("trades") >= min_trades) & (pl.col("expectancy") > 0.0))
    if candidates.is_empty():
        return [
            "MTF state operability",
            f"candidate_ge{min_trades}=0 fragile_positive=0 robust_positive=0",
        ]
    fragile = 0
    robust = 0
    labels = []
    for row in candidates.head(4).iter_rows(named=True):
        trades_count = int(row["trades"] or 0)
        wins = int(row["wins"] or 0)
        win_rate = wins / max(trades_count, 1) * 100.0
        gross_profit = float(row["gross_profit"] or 0.0)
        max_win = float(row["max_win"] or 0.0)
        max_win_share = max_win / gross_profit * 100.0 if gross_profit > 1e-9 else 0.0
        is_fragile = win_rate < 30.0 or max_win_share > 50.0
        fragile += int(is_fragile)
        robust += int(not is_fragile)
        verdict = "fragile_positive" if is_fragile else "robust_positive"
        labels.append(
            f"{row[state_col]}:{trades_count}/wr={win_rate:.0f}%/max_win={max_win_share:.0f}%/{verdict}"
        )
    return [
        "MTF state operability",
        f"candidate_ge{min_trades}={candidates.height} fragile_positive={fragile} "
        f"robust_positive={robust} top={','.join(labels)}",
    ]


def _format_mtf_state_time_consistency_row(report: Report) -> list[str]:
    trades = _structural_event_trades(report.trades)
    state_col = _first_column(trades, ("entry_mtf_state_bucket", "entry_mtf_state_key"))
    if trades.is_empty() or not state_col or state_col not in trades.columns:
        return ["MTF state PnL time consistency", "n/a"]
    time_col = _first_column(trades, ("entry_ts", "timestamp"))
    if not time_col:
        return ["MTF state PnL time consistency", "n/a"]
    net_col = _net_col(trades)
    bucket_ms = 30 * 86_400_000
    grouped = (
        trades.with_columns(
            pl.col(state_col).cast(pl.Utf8).fill_null("data_error"),
            (pl.col(time_col).cast(pl.Int64) // bucket_ms).alias("_time_bucket"),
            pl.col(net_col).cast(pl.Float64).alias("_bucket_net"),
        )
        .group_by(state_col, "_time_bucket")
        .agg(
            pl.len().alias("trades"),
            pl.col("_bucket_net").sum().alias("net"),
        )
    )
    if grouped.is_empty():
        return ["MTF state PnL time consistency", "n/a"]
    state_totals = grouped.group_by(state_col).agg(
        pl.col("net").sum().alias("total_net"),
        pl.col("net").max().alias("best_bucket_net"),
        pl.len().alias("time_buckets"),
    )
    concentrated = state_totals.filter(
        (pl.col("total_net") > 0.0)
        & (pl.col("best_bucket_net") / pl.col("total_net") > 0.6)
    )
    labels = []
    for row in concentrated.head(3).iter_rows(named=True):
        share = float(row["best_bucket_net"] or 0.0) / max(float(row["total_net"] or 0.0), 1e-9)
        labels.append(f"{row[state_col]}:{share * 100.0:.0f}%/{row['time_buckets']}buckets")
    return [
        "MTF state PnL time consistency",
        f"states={state_totals.height} time_concentrated={concentrated.height} "
        f"top={','.join(labels) or 'none'}",
    ]


def _state_group_label(row: dict, columns: tuple[str, ...]) -> str:
    key = "/".join(str(row.get(column) or "unknown") for column in columns)
    trades = int(row.get("trades") or 0)
    expectancy = float(row.get("expectancy") or 0.0)
    return f"{key}:{trades}/${expectancy:+.2f}"


def _structural_event_attribution(report: Report) -> tuple[BucketSummary, ...]:
    return _trade_bucket_summaries(
        _structural_event_trades(report.trades),
        (DiagnosticColumn.ENTRY_LIQUIDITY_EVENT_TYPE_BUCKET,),
        limit=4,
    )


def _format_structural_event_attribution_row(report: Report) -> list[str]:
    return [
        "Structural event attribution",
        _format_bucket_summaries("event_trade_expectancy", _structural_event_attribution(report)),
    ]


def _format_structural_side_structure_row(report: Report) -> list[str]:
    trades = _structural_event_trades(report.trades)
    structure_col = _first_column(trades, ("entry_structure_bucket", "entry_structure_trend_state"))
    return [
        "Structural side x trend",
        _format_attribution_table(
            trades,
            ("side", structure_col),
            "side_x_trend",
            include_pf=True,
        ),
    ]


def _format_structural_side_event_row(report: Report) -> list[str]:
    trades = _structural_event_trades(report.trades)
    event_col = _event_column(trades)
    return [
        "Structural side x event",
        _format_attribution_table(trades, ("side", event_col), "side_x_event"),
    ]


def _format_structural_side_event_stage_row(report: Report) -> list[str]:
    trades = _structural_event_trades(report.trades)
    event_col = _event_column(trades)
    stage_col = _first_column(trades, ("entry_market_stage_bucket", "entry_market_stage"))
    return [
        "Structural side x event x stage",
        _format_attribution_table(
            trades,
            ("side", event_col, stage_col),
            "side_x_event_x_stage",
            limit=6,
        ),
    ]


def _format_structural_quality_attribution_row(report: Report) -> list[str]:
    trades = _structural_event_trades(report.trades)
    quality_col = _first_column(
        trades,
        (DiagnosticColumn.ENTRY_EVENT_QUALITY_BUCKET, DiagnosticColumn.ENTRY_EVENT_QUALITY_SCORE),
    )
    return [
        "Structural quality x result",
        _format_attribution_table(trades, (quality_col,), "quality_x_result"),
    ]


def _format_structural_volume_attribution_row(report: Report) -> list[str]:
    trades = _structural_event_trades(report.trades)
    return [
        "Structural volume x result",
        _format_attribution_table(trades, ("entry_volume_impulse",), "volume_x_result"),
    ]


def _first_column(frame: pl.DataFrame, columns: tuple[str, ...]) -> str:
    for column in columns:
        if column in frame.columns:
            return column
    return ""


def _format_attribution_table(
    frame: pl.DataFrame,
    columns: tuple[str, ...],
    label: str,
    *,
    limit: int = 4,
    include_pf: bool = False,
) -> str:
    columns = tuple(column for column in columns if column)
    if frame.is_empty() or not columns or not set(columns) <= set(frame.columns):
        return f"{label}=no trades"
    net_col = _net_col(frame)
    grouped = (
        frame.with_columns(
            *(pl.col(column).cast(pl.Utf8).fill_null("unknown") for column in columns),
            pl.col(net_col).cast(pl.Float64).alias("_bucket_net"),
        )
        .group_by(list(columns))
        .agg(
            pl.len().alias("trades"),
            pl.col("_bucket_net").sum().alias("net"),
            pl.col("_bucket_net").mean().alias("expectancy"),
            (pl.col("_bucket_net") > 0.0).cast(pl.Int64).sum().alias("wins"),
            pl.when(pl.col("_bucket_net") > 0.0)
            .then(pl.col("_bucket_net"))
            .otherwise(0.0)
            .sum()
            .alias("gross_profit"),
            pl.when(pl.col("_bucket_net") < 0.0)
            .then(-pl.col("_bucket_net"))
            .otherwise(0.0)
            .sum()
            .alias("gross_loss"),
        )
        .sort("net")
        .head(limit)
    )
    parts = []
    for row in grouped.iter_rows(named=True):
        key = "/".join(str(row[column] or "unknown") for column in columns)
        trades = int(row["trades"] or 0)
        net = float(row["net"] or 0.0)
        expectancy = float(row["expectancy"] or 0.0)
        win_rate = float(row["wins"] or 0.0) / max(trades, 1) * 100.0
        verdict = _attribution_verdict(net, expectancy, trades)
        text = f"{key}:{trades}/${net:+.2f}/${expectancy:+.2f}/wr={win_rate:.0f}%"
        if include_pf:
            gross_profit = float(row["gross_profit"] or 0.0)
            gross_loss = float(row["gross_loss"] or 0.0)
            pf = gross_profit / gross_loss if gross_loss > 1e-9 else float("inf")
            pf_text = "inf" if math.isinf(pf) else f"{pf:.2f}"
            text = f"{text}/pf={pf_text}"
        parts.append(f"{text}/{verdict}")
    return f"{label}=" + ",".join(parts)


def _attribution_verdict(net: float, expectancy: float, trades: int) -> str:
    if trades <= 0:
        return "no_trades"
    if net < 0.0 and expectancy < 0.0:
        return "loss_source"
    if net > 0.0 and expectancy > 0.0:
        return "profit_source"
    return "mixed"


def _structural_event_time_clustering(report: Report) -> TimeClusterSummary:
    trades = _structural_event_trades(report.trades)
    if trades.is_empty():
        return TimeClusterSummary(0, (), 0.0, False)
    sort_col = "entry_bar_index" if "entry_bar_index" in trades.columns else "entry_ts"
    ordered = trades.sort(sort_col) if sort_col in trades.columns else trades
    net_col = _net_col(ordered)
    segment_count = min(4, max(1, ordered.height))
    rows = []
    for idx in range(segment_count):
        start = idx * ordered.height // segment_count
        end = (idx + 1) * ordered.height // segment_count
        segment = ordered.slice(start, end - start)
        net = float(segment.select(pl.col(net_col).cast(pl.Float64).sum()).item() or 0.0)
        rows.append(
            BucketSummary(
                (f"q{idx + 1}",),
                segment.height,
                net,
                net / max(segment.height, 1),
            )
        )
    positive_total = sum(row.net for row in rows if row.net > 0.0)
    best_net = max((row.net for row in rows), default=0.0)
    if positive_total > 0.0 and best_net > 0.0:
        best_share = best_net / positive_total * 100.0
    else:
        total_abs = sum(abs(row.net) for row in rows)
        best_share = abs(best_net) / total_abs * 100.0 if total_abs > 0.0 else 0.0
    return TimeClusterSummary(
        segment_count,
        tuple(rows),
        best_share,
        best_share > 60.0 and segment_count > 1,
    )


def _format_structural_event_time_clustering_row(report: Report) -> list[str]:
    summary = _structural_event_time_clustering(report)
    if summary.segments <= 0:
        return ["Structural event time clustering", "no trades"]
    segment_text = ",".join(
        f"{row.key[0]}:{row.trades}/${row.net:+.2f}/${row.expectancy:+.2f}"
        for row in summary.rows
    )
    return [
        "Structural event time clustering",
        f"segments={summary.segments} {segment_text} "
        f"best_segment_net_share={summary.best_share_pct:.1f}% "
        f"clustered={_yes_no(summary.clustered)} "
        f"promotion_clustered={_yes_no(summary.best_share_pct > 80.0)}",
    ]


def _reclaim_extension_status(
    report: Report,
    strategy: StrategyBehavior,
) -> ReclaimExtensionStatus:
    include_reclaim = bool(getattr(strategy, "include_reclaim_sweeps", False))
    trades = report.trades
    reclaim_trades = 0
    event_col = _event_column(trades) if not trades.is_empty() else ""
    if event_col:
        reclaim_trades = trades.filter(
            pl.col(event_col).is_in(
                [LiquidityEvent.BULLISH_RECLAIM, LiquidityEvent.BEARISH_RECLAIM]
            )
        ).height
    return ReclaimExtensionStatus(include_reclaim, reclaim_trades)


def _format_reclaim_extension_status_row(summary: ReclaimExtensionStatus) -> list[str]:
    state = "enabled" if summary.include_reclaim_sweeps else "disabled"
    return [
        "Reclaim/sweep extension",
        f"{state} reclaim_extension_trades={summary.reclaim_extension_trades}",
    ]


def _reclaim_hypothesis_row(report: Report) -> list[str]:
    if report.trades.is_empty() or not {
        "entry_liquidity_event",
        "entry_liquidity_event_type_bucket",
    }.intersection(report.trades.columns):
        return ["Reclaim hypothesis", "n/a"]
    return [
        "Reclaim hypothesis",
        "shelved: reclaim is diagnostic-only until bullish/bearish reclaim expectancy "
        "is robust across assets; volume-confirmed reclaim is not production confirmation",
    ]


def _none_coverage_row(report: Report) -> list[str]:
    if report.trades.is_empty():
        return ["None-event coverage", "no trades"]
    none_trades = _none_event_trades(report.trades)
    if none_trades.is_empty():
        return [
            "None-event coverage",
            "none_event_trades=0 residual=0.0% event_coverage=good",
        ]
    net_col = "net_pnl_usd" if "net_pnl_usd" in none_trades.columns else "pnl_usd"
    total_net_col = "net_pnl_usd" if "net_pnl_usd" in report.trades.columns else "pnl_usd"
    none_net = float(none_trades.select(pl.col(net_col).cast(pl.Float64).sum()).item() or 0.0)
    total_net = float(
        report.trades.select(pl.col(total_net_col).cast(pl.Float64).sum()).item() or 0.0
    )
    residual_pct = none_trades.height / max(report.trades.height, 1) * 100.0
    if residual_pct > 60.0:
        coverage = "poor"
    elif residual_pct >= 30.0:
        coverage = "partial"
    else:
        coverage = "good"
    parts = [
        f"none_event_trades={none_trades.height}/{report.trades.height}",
        f"residual={residual_pct:.1f}%",
        f"event_coverage={coverage}",
        f"none_net=${none_net:+.2f}",
        f"total_net=${total_net:+.2f}",
        "none_is_residual_not_market_event=true",
    ]
    for column, label in (
        ("entry_market_stage_bucket", "stage"),
        ("entry_market_stage_reason_bucket", "stage_reason"),
        ("entry_stage_unknown_reason_bucket", "unknown_reason"),
        ("entry_structure_bucket", "structure"),
        ("side", "side"),
    ):
        breakdown = _top_bucket_net(none_trades, column, label)
        if breakdown:
            parts.append(breakdown)
    return ["None-event coverage", " ".join(parts)]


def decompose_none_events(report: Report) -> list[list[str]]:
    if report.trades.is_empty():
        return [["None trend alignment", "no trades"]]
    none_trades = _none_event_trades(report.trades)
    if none_trades.is_empty():
        return [["None trend alignment", "none_event_trades=0"]]
    return [
        _none_trend_alignment_row(none_trades),
        _none_volatility_context_row(none_trades),
        _none_key_level_proximity_row(none_trades),
        _none_z_pressure_row(none_trades),
        _none_time_clustering_row(none_trades),
    ]


def _none_trend_alignment_row(none_trades: pl.DataFrame) -> list[str]:
    rows = []
    for trade in none_trades.iter_rows(named=True):
        side = str(trade.get("side") or "")
        structure = str(trade.get("entry_structure_bucket") or "")
        trend = str(trade.get("entry_trend_bucket") or "")
        rows.append(
            {
                **trade,
                "none_trend_alignment": _trend_alignment_bucket(side, structure, trend),
            }
        )
    frame = pl.DataFrame(rows) if rows else none_trades
    parts = [
        _cross_bucket_expectancy_frame(
            none_trades,
            ("side", "entry_structure_bucket"),
            "none_side_x_structure",
        ),
        _cross_bucket_expectancy_frame(
            none_trades,
            ("side", "entry_trend_bucket"),
            "none_side_x_trend",
        ),
        _bucket_expectancy_frame(frame, "none_trend_alignment", "alignment"),
    ]
    return ["None trend alignment", " ".join(part for part in parts if part)]


def _none_volatility_context_row(none_trades: pl.DataFrame) -> list[str]:
    parts = [
        _bucket_expectancy_frame(none_trades, "entry_atr_percentile_bucket", "atr_percentile"),
        _bucket_expectancy_frame(none_trades, "entry_volatility_bucket", "volatility_regime"),
    ]
    return ["None volatility context", " ".join(part for part in parts if part) or "n/a"]


def _none_key_level_proximity_row(none_trades: pl.DataFrame) -> list[str]:
    text = _bucket_expectancy_frame(
        none_trades,
        "entry_key_level_proximity_bucket",
        "key_level_proximity",
    )
    return ["None key-level proximity", text or "n/a"]


def _none_z_pressure_row(none_trades: pl.DataFrame) -> list[str]:
    parts = [
        _cross_bucket_expectancy_frame(
            none_trades,
            ("entry_z_pressure_side_bucket", "side"),
            "none_z_pressure_x_side",
        ),
        _cross_bucket_expectancy_frame(
            none_trades,
            ("entry_zscore_bucket", "side"),
            "none_zscore_bucket_x_side",
        ),
    ]
    return ["None Z pressure", " ".join(part for part in parts if part) or "n/a"]


def _none_time_clustering_row(none_trades: pl.DataFrame) -> list[str]:
    stats = _none_time_clustering_stats(none_trades)
    return [
        "None time clustering",
        f"segments={stats['segments']} {stats['segment_text']} "
        f"best_segment_net_share={stats['best_share_pct']:.1f}% "
        f"clustered={stats['clustered']}",
    ]


def _trend_alignment_bucket(side: str, structure: str, trend_bucket: str) -> str:
    trend = structure or trend_bucket
    if trend in {"range", "flat"}:
        return "range_or_flat"
    if side == "buy" and trend == "uptrend":
        return "aligned"
    if side == "sell" and trend == "downtrend":
        return "aligned"
    if side == "buy" and trend == "downtrend":
        return "countertrend"
    if side == "sell" and trend == "uptrend":
        return "countertrend"
    return "unknown"


def _bucket_expectancy_frame(frame: pl.DataFrame, column: str, label: str) -> str:
    if frame.is_empty() or column not in frame.columns:
        return f"none_{label}=unknown"
    net_col = "net_pnl_usd" if "net_pnl_usd" in frame.columns else "pnl_usd"
    grouped = (
        frame.with_columns(pl.col(column).fill_null("unknown"))
        .group_by(column)
        .agg(
            pl.len().alias("trades"),
            pl.col(net_col).cast(pl.Float64).sum().alias("net"),
            pl.col(net_col).cast(pl.Float64).mean().alias("expectancy"),
        )
        .sort("net")
    )
    parts = [
        (
            f"{row[column]}:{int(row['trades'])}/"
            f"${float(row['net'] or 0.0):+.2f}/"
            f"${float(row['expectancy'] or 0.0):+.2f}"
        )
        for row in grouped.iter_rows(named=True)
    ]
    return f"none_{label}=" + ",".join(parts)


def _net_col(frame: pl.DataFrame) -> str:
    return "net_pnl_usd" if "net_pnl_usd" in frame.columns else "pnl_usd"


def _trade_bucket_summaries(
    frame: pl.DataFrame,
    columns: tuple[str, ...],
    *,
    min_count: int = 1,
    limit: int | None = None,
    sort_by: str = "net",
) -> tuple[BucketSummary, ...]:
    if frame.is_empty() or not set(columns) <= set(frame.columns):
        return ()
    net_col = _net_col(frame)
    if net_col not in frame.columns:
        return ()
    grouped = (
        frame.with_columns(
            *(pl.col(column).cast(pl.Utf8).fill_null("unknown") for column in columns)
        )
        .group_by(list(columns))
        .agg(
            pl.len().alias("trades"),
            pl.col(net_col).cast(pl.Float64).sum().alias("net"),
            pl.col(net_col).cast(pl.Float64).mean().alias("expectancy"),
        )
        .filter(pl.col("trades") >= min_count)
        .sort(sort_by)
    )
    if limit is not None:
        grouped = grouped.head(limit)
    return tuple(
        BucketSummary(
            key=tuple(str(row[column] or "unknown") for column in columns),
            trades=int(row["trades"] or 0),
            net=float(row["net"] or 0.0),
            expectancy=float(row["expectancy"] or 0.0),
        )
        for row in grouped.iter_rows(named=True)
    )


def _format_bucket_summaries(label: str, summaries: tuple[BucketSummary, ...]) -> str:
    if not summaries:
        return f"{label}=no trades"
    parts = [
        f"{'/'.join(row.key)}:{row.trades}/${row.net:+.2f}/${row.expectancy:+.2f}"
        for row in summaries
    ]
    return f"{label}=" + ",".join(parts)


def _cross_bucket_expectancy_frame(
    frame: pl.DataFrame,
    columns: tuple[str, str],
    label: str,
    *,
    limit: int = 3,
) -> str:
    if frame.is_empty() or not set(columns) <= set(frame.columns):
        return f"{label}=unknown"
    net_col = "net_pnl_usd" if "net_pnl_usd" in frame.columns else "pnl_usd"
    grouped = (
        frame.with_columns(
            pl.col(columns[0]).fill_null("unknown"),
            pl.col(columns[1]).fill_null("unknown"),
        )
        .group_by(list(columns))
        .agg(
            pl.len().alias("trades"),
            pl.col(net_col).cast(pl.Float64).mean().alias("expectancy"),
        )
        .sort("expectancy")
        .head(limit)
    )
    parts = []
    for row in grouped.iter_rows(named=True):
        key = "/".join(str(row[column] or "unknown") for column in columns)
        parts.append(f"{key}:{int(row['trades'])}/${float(row['expectancy'] or 0.0):+.2f}")
    return f"{label}=" + ",".join(parts)


def _none_time_clustering_stats(none_trades: pl.DataFrame) -> dict[str, object]:
    if none_trades.is_empty():
        return {"segments": 0, "segment_text": "none", "best_share_pct": 0.0, "clustered": "no"}
    sort_col = "entry_bar_index" if "entry_bar_index" in none_trades.columns else "entry_ts"
    ordered = none_trades.sort(sort_col) if sort_col in none_trades.columns else none_trades
    net_col = "net_pnl_usd" if "net_pnl_usd" in ordered.columns else "pnl_usd"
    segment_count = min(4, max(1, ordered.height))
    rows = []
    for idx in range(segment_count):
        start = idx * ordered.height // segment_count
        end = (idx + 1) * ordered.height // segment_count
        segment = ordered.slice(start, end - start)
        net = float(segment.select(pl.col(net_col).cast(pl.Float64).sum()).item() or 0.0)
        expectancy = net / max(segment.height, 1)
        rows.append((idx + 1, segment.height, net, expectancy))
    positive_total = sum(net for _, _, net, _ in rows if net > 0.0)
    best_net = max((net for _, _, net, _ in rows), default=0.0)
    if positive_total > 0.0 and best_net > 0.0:
        best_share = best_net / positive_total * 100.0
    else:
        total_abs = sum(abs(net) for _, _, net, _ in rows)
        best_share = abs(best_net) / total_abs * 100.0 if total_abs > 0.0 else 0.0
    text = ",".join(
        f"q{idx}:{count}/${net:+.2f}/${expectancy:+.2f}"
        for idx, count, net, expectancy in rows
    )
    return {
        "segments": segment_count,
        "segment_text": text,
        "best_share_pct": best_share,
        "clustered": "yes" if best_share > 60.0 and segment_count > 1 else "no",
    }


def _structural_overlap_row(report: Report) -> list[str]:
    if report.trades.is_empty():
        return ["Structural overlap audit", "no trades"]
    parts = [
        "overlap_not_confirmation=true",
        "attribution_not_causality=true",
    ]
    for columns, label in (
        (("entry_structure_bucket", "entry_market_stage_bucket"), "structure_x_stage"),
        (("entry_structure_bucket", "entry_market_stage_reason_bucket"), "structure_x_reason"),
        (("entry_market_stage_bucket", "entry_market_stage_reason_bucket"), "stage_x_reason"),
        (("entry_market_stage_bucket", "entry_stage_unknown_reason_bucket"), "stage_x_unknown"),
    ):
        overlap = _top_pair_count(report.trades, columns, label)
        if overlap:
            parts.append(overlap)
    return ["Structural overlap audit", " ".join(parts)]


def _interpretability_note_row(report: Report) -> list[str]:
    if report.trades.is_empty():
        return ["Interpretability guardrail", "no trades"]
    return [
        "Interpretability guardrail",
        "cross-bucket attribution identifies where PnL occurred, not why; "
        "trend_without_range_break is a stage-model residual reason, not a validated trend edge; "
        "ETH-only positive attribution is a rejection signal, not a strategy signal",
    ]


def _structure_development_row(
    report: Report,
    signal_frame: pl.DataFrame,
    strategy: StrategyBehavior,
) -> list[str]:
    if not isinstance(strategy, StrategySpec) or "market_stage" not in signal_frame.columns:
        return ["Structure opportunity", "n/a"]
    raw_expr = _raw_opportunity_expr(strategy)
    parts = [
        f"raw_opportunities={_expr_count(signal_frame, raw_expr)}",
        _opportunity_counts_by_column(signal_frame, raw_expr, "market_stage", "stage"),
        _opportunity_counts_by_column(
            signal_frame,
            raw_expr,
            "structure_trend_state",
            "structure",
        ),
        _stage_trade_expectancy(report),
    ]
    return ["Structure opportunity", " ".join(part for part in parts if part)]


def _structure_veto_rows(
    signal_frame: pl.DataFrame,
    strategy: StrategyBehavior,
) -> list[list[str]]:
    required = {
        "zbasket_pre_veto_long",
        "zbasket_pre_veto_short",
        "zbasket_vetoed_long",
        "zbasket_vetoed_short",
        "structure_veto_reason",
    }
    if not isinstance(strategy, StrategySpec) or not required <= set(signal_frame.columns):
        return [
            ["Structure veto attribution", "disabled"],
            ["Structure veto opportunity cost", "disabled"],
            ["Exhaustion exemption attribution", "disabled"],
        ]
    filter_expr = _strategy_filter_expr(strategy)
    pre_long = pl.col("zbasket_pre_veto_long").fill_null(False) & filter_expr
    pre_short = pl.col("zbasket_pre_veto_short").fill_null(False) & filter_expr
    veto_long = pl.col("zbasket_vetoed_long").fill_null(False) & filter_expr
    veto_short = pl.col("zbasket_vetoed_short").fill_null(False) & filter_expr
    vetoed = signal_frame.filter((veto_long | veto_short).fill_null(False))
    pre_long_count = _expr_count(signal_frame, pre_long)
    pre_short_count = _expr_count(signal_frame, pre_short)
    veto_long_count = _expr_count(signal_frame, veto_long)
    veto_short_count = _expr_count(signal_frame, veto_short)
    allowed = pre_long_count + pre_short_count - veto_long_count - veto_short_count
    parts = [
        f"pre_veto_long={pre_long_count}",
        f"pre_veto_short={pre_short_count}",
        f"raw_long_vetoed={veto_long_count}",
        f"raw_short_vetoed={veto_short_count}",
        f"allowed_after_veto={allowed}",
        _veto_count_by(vetoed, "structure_trend_state", "veto_by_structure"),
        _veto_count_by(vetoed, "market_stage", "veto_by_stage"),
        _veto_count_by(vetoed, "atr_percentile_bucket", "veto_by_atr"),
        _veto_count_by(vetoed, "structure_veto_reason", "veto_by_reason"),
    ]
    exemption = _veto_exemption_status(signal_frame)
    opportunity_cost = _structure_veto_opportunity_cost(signal_frame, strategy)
    return [
        ["Structure veto attribution", " ".join(part for part in parts if part)],
        ["Structure veto opportunity cost", opportunity_cost],
        ["Exhaustion exemption attribution", exemption],
    ]


def _strategy_filter_expr(strategy: StrategySpec) -> pl.Expr:
    filter_expr = pl.lit(True)
    for expr in strategy.filters:
        filter_expr = filter_expr & expr.fill_null(False)
    return filter_expr


def _veto_count_by(frame: pl.DataFrame, column: str, label: str, *, limit: int = 4) -> str:
    if frame.is_empty() or column not in frame.columns:
        return f"{label}=none"
    grouped = (
        frame.with_columns(pl.col(column).fill_null("unknown"))
        .group_by(column)
        .agg(pl.len().alias("count"))
        .sort(["count", column], descending=[True, False])
        .head(limit)
    )
    parts = [f"{row[column]}:{int(row['count'])}" for row in grouped.iter_rows(named=True)]
    return f"{label}=" + ",".join(parts)


def _veto_exemption_status(signal_frame: pl.DataFrame) -> str:
    if "structure_veto_exemption_status" not in signal_frame.columns:
        return "disabled"
    values = signal_frame["structure_veto_exemption_status"].drop_nulls().unique().to_list()
    return "status=" + (",".join(str(value) for value in values) if values else "disabled")


def _structure_veto_opportunity_cost(
    signal_frame: pl.DataFrame,
    strategy: StrategyBehavior,
    *,
    max_bars: int = 10,
) -> str:
    if not isinstance(strategy, StrategySpec):
        return "n/a"
    needed = {"close", "high", "low", "zbasket_vetoed_long", "zbasket_vetoed_short"}
    if signal_frame.is_empty() or not needed <= set(signal_frame.columns):
        return "n/a"
    filter_expr = _strategy_filter_expr(strategy)
    indexed = signal_frame.with_row_index("_row_idx")
    vetoed = indexed.filter(
        (
            (pl.col("zbasket_vetoed_long").fill_null(False))
            | (pl.col("zbasket_vetoed_short").fill_null(False))
        )
        & filter_expr
    )
    if vetoed.is_empty():
        return "vetoed_proxy_trades=0"
    outcomes: list[tuple[float, float, float]] = []
    for row in vetoed.iter_rows(named=True):
        idx = int(row["_row_idx"])
        entry_px = float(row.get("close") or 0.0)
        if entry_px <= 0.0:
            continue
        window = signal_frame.slice(idx + 1, min(max_bars, signal_frame.height - idx - 1))
        if window.is_empty():
            continue
        side_mult = 1.0 if bool(row.get("zbasket_vetoed_long")) else -1.0
        exit_px = float(window["close"][-1] or entry_px)
        high = float(window.select(pl.col("high").cast(pl.Float64).max()).item() or entry_px)
        low = float(window.select(pl.col("low").cast(pl.Float64).min()).item() or entry_px)
        directional = side_mult * (exit_px / entry_px - 1.0) * 100.0
        if side_mult > 0.0:
            mfe = (high / entry_px - 1.0) * 100.0
            mae = (low / entry_px - 1.0) * 100.0
        else:
            mfe = (entry_px / low - 1.0) * 100.0 if low > 0.0 else 0.0
            mae = (entry_px / high - 1.0) * 100.0 if high > 0.0 else 0.0
        outcomes.append((directional, mfe, mae))
    if not outcomes:
        return f"vetoed_proxy_trades={vetoed.height} proxy_available=0"
    return (
        f"vetoed_proxy_trades={len(outcomes)} "
        f"proxy_exp={_avg([value[0] for value in outcomes]):+.2f}% "
        f"proxy_mfe={_avg([value[1] for value in outcomes]):+.2f}% "
        f"proxy_mae={_avg([value[2] for value in outcomes]):+.2f}% "
        f"horizon_bars={max_bars}"
    )


def _filter_attrition_rows(
    signal_frame: pl.DataFrame, strategy: StrategyBehavior) -> list[list[str]]:
    if not isinstance(strategy, StrategySpec):
        return [["Filter attrition", "n/a for non-spec strategy"]]
    if not strategy.filters:
        return [["Filter attrition", "no explicit filters"]]
    raw_expr = _raw_opportunity_expr(strategy)
    before = _expr_count(signal_frame, raw_expr)
    if before <= 0:
        return [["Filter attrition", "no raw opportunities before filters"]]
    rows = []
    active_expr = raw_expr
    for idx, filter_expr in enumerate(strategy.filters):
        next_expr = active_expr & filter_expr.fill_null(False)
        after = _expr_count(signal_frame, next_expr)
        removed = before - after
        rows.append(
            [
                f"Filter {idx} attrition",
                f"before={before} after={after} removed={removed} "
                f"removed_pct={removed / before * 100.0 if before else 0.0:.1f}%",
            ]
        )
        active_expr = next_expr
        before = after
    return rows


def _trade_path_row(report: Report, signal_frame: pl.DataFrame) -> list[str]:
    path = _trade_path_stats(report, signal_frame)
    if path["trades"] <= 0:
        return ["Trade path", "no accepted trades"]
    return [
        "Trade path",
        f"trades={path['trades']:.0f} avg_mfe={path['avg_mfe_pct']:+.2f}% "
        f"avg_mae={path['avg_mae_pct']:+.2f}% loss_avg_mfe={path['loss_avg_mfe_pct']:+.2f}% "
        f"loss_avg_mae={path['loss_avg_mae_pct']:+.2f}% avg_bars_held={path['avg_bars_held']:.1f}",
    ]


def _loss_cause_row(report: Report) -> list[str]:
    if report.trades.is_empty():
        return ["Loss path", "no trades"]
    net_col = "net_pnl_usd" if "net_pnl_usd" in report.trades.columns else "pnl_usd"
    losses = report.trades.filter(pl.col(net_col).cast(pl.Float64) < 0)
    if losses.is_empty():
        return ["Loss path", "no losing trades"]
    counts: dict[str, int] = {}
    for row in losses.iter_rows(named=True):
        cause = _loss_cause(row)
        counts[cause] = counts.get(cause, 0) + 1
    parts = ",".join(f"{key}:{value}" for key, value in sorted(counts.items()))
    return ["Loss path", parts]


def _loss_confidence_row(report: Report) -> list[str]:
    if report.trades.is_empty():
        return ["Loss confidence", "no trades"]
    net_col = "net_pnl_usd" if "net_pnl_usd" in report.trades.columns else "pnl_usd"
    losses = report.trades.filter(pl.col(net_col).cast(pl.Float64) < 0)
    if losses.is_empty():
        return ["Loss confidence", "no losing trades"]
    counts = {"high": 0, "medium": 0, "low": 0}
    residual = 0
    for row in losses.iter_rows(named=True):
        cause = _loss_cause(row)
        confidence = _loss_cause_confidence(cause)
        counts[confidence] += 1
        if cause == "unclassified_none_event":
            residual += 1
    return [
        "Loss confidence",
        f"high={counts['high']} medium={counts['medium']} low={counts['low']} "
        f"low_confidence_residual={residual} "
        "unclassified_none_event_is_residual=true",
    ]


def _production_design_row(report: Report) -> list[str]:
    max_per_strategy_symbol = _metadata_value(report, "max_per_strategy_symbol")
    stacked_entries = report.diagnostics.lifecycle.stacked_entry_count if report.diagnostics else 0
    if max_per_strategy_symbol == "1":
        summary = "single-thesis basket mode; same-strategy concurrency intentionally disabled"
    elif max_per_strategy_symbol:
        summary = (
            f"concurrency cap={max_per_strategy_symbol}; stacked_entries={stacked_entries}; "
            "requires explicit multi-entry thesis to be production-relevant"
        )
    else:
        summary = (
            "concurrency semantics unknown; define production basket model before changing caps"
        )
    return ["Production basket design", summary]


def _raw_opportunity_count(signal_frame: pl.DataFrame, strategy: StrategyBehavior) -> int:
    if isinstance(strategy, StrategySpec):
        return _expr_count(signal_frame, _raw_opportunity_expr(strategy))
    if "raw_entry_signal" in signal_frame.columns:
        return _expr_count(signal_frame, pl.col("raw_entry_signal") != 0)
    if "signal" in signal_frame.columns:
        return _expr_count(signal_frame, pl.col("signal") != 0)
    return 0


def _raw_opportunity_expr(strategy: StrategySpec) -> pl.Expr:
    expr = pl.lit(False)
    for rule in strategy.entries:
        expr = expr | rule.condition.fill_null(False)
    return expr


def _expr_count(df: pl.DataFrame, expr: pl.Expr) -> int:
    if df.is_empty():
        return 0
    return int(df.select(expr.fill_null(False).cast(pl.Int64).sum()).item() or 0)


def _trade_path_stats(report: Report, signal_frame: pl.DataFrame) -> dict[str, float]:
    values: list[tuple[float, float, float, bool]] = []
    if report.trades.is_empty():
        return {"trades": 0.0}
    net_col = "net_pnl_usd" if "net_pnl_usd" in report.trades.columns else "pnl_usd"
    for trade in report.trades.iter_rows(named=True):
        if not {"entry_bar_index", "exit_bar_index", "entry_px", "side"} <= set(trade):
            continue
        entry_idx = int(trade.get("entry_bar_index") or 0)
        exit_idx = int(trade.get("exit_bar_index") or entry_idx)
        if entry_idx < 0 or exit_idx < entry_idx or entry_idx >= signal_frame.height:
            continue
        window = signal_frame.slice(
            entry_idx,
            min(exit_idx, signal_frame.height - 1) - entry_idx + 1,
        )
        if window.is_empty() or "high" not in window.columns or "low" not in window.columns:
            continue
        entry_px = float(trade.get("entry_px") or 0.0)
        if entry_px <= 0:
            continue
        high_value = window.select(pl.col("high").cast(pl.Float64).max()).item()
        low_value = window.select(pl.col("low").cast(pl.Float64).min()).item()
        high = float(high_value) if high_value is not None else entry_px
        low = float(low_value) if low_value is not None else entry_px
        side = str(trade.get("side") or "")
        if side == "sell":
            mfe = (entry_px - low) / entry_px * 100.0
            mae = (entry_px - high) / entry_px * 100.0
        else:
            mfe = (high - entry_px) / entry_px * 100.0
            mae = (low - entry_px) / entry_px * 100.0
        bars_held = float(trade.get("bars_held") or 0.0)
        is_loss = float(trade.get(net_col) or 0.0) < 0.0
        values.append((mfe, mae, bars_held, is_loss))
    if not values:
        return {"trades": 0.0}
    losses = [value for value in values if value[3]]
    return {
        "trades": float(len(values)),
        "avg_mfe_pct": _avg([value[0] for value in values]),
        "avg_mae_pct": _avg([value[1] for value in values]),
        "avg_bars_held": _avg([value[2] for value in values]),
        "loss_avg_mfe_pct": _avg([value[0] for value in losses]),
        "loss_avg_mae_pct": _avg([value[1] for value in losses]),
    }


def _none_event_trades(trades: pl.DataFrame) -> pl.DataFrame:
    if trades.is_empty():
        return trades
    if "entry_liquidity_event_type_bucket" in trades.columns:
        return trades.filter(pl.col("entry_liquidity_event_type_bucket") == "none")
    if "entry_liquidity_event_type" in trades.columns:
        return trades.filter(pl.col("entry_liquidity_event_type") == "none")
    if "entry_liquidity_event" in trades.columns:
        return trades.filter(pl.col("entry_liquidity_event") == "no_reclaim")
    return trades.clear()


def _structural_event_trades(trades: pl.DataFrame) -> pl.DataFrame:
    if trades.is_empty():
        return trades
    event_col = _event_column(trades)
    if not event_col:
        return trades.clear()
    return trades.filter(
        pl.col(event_col).is_in(
            [LiquidityEvent.FAILED_BREAKOUT_HIGH, LiquidityEvent.FAILED_BREAKOUT_LOW]
        )
    )


def _event_column(frame: pl.DataFrame) -> str:
    for column in (
        DiagnosticColumn.ENTRY_LIQUIDITY_EVENT_TYPE_BUCKET,
        DiagnosticColumn.ENTRY_LIQUIDITY_EVENT_TYPE,
        "entry_liquidity_event",
        "liquidity_event_type",
    ):
        if column in frame.columns:
            return column
    return ""


def _event_count(frame: pl.DataFrame, event: str) -> int:
    event_col = _event_column(frame)
    if event_col:
        return _expr_count(frame, pl.col(event_col) == event)
    flag_column = event if event in frame.columns else ""
    return _expr_count(frame, pl.col(flag_column).fill_null(False)) if flag_column else 0


def _structural_failed_breakout_expr(frame: pl.DataFrame) -> pl.Expr:
    return _structural_failed_breakout_low_expr(frame) | _structural_failed_breakout_high_expr(
        frame
    )


def _structural_failed_breakout_low_expr(frame: pl.DataFrame) -> pl.Expr:
    event_col = _event_column(frame)
    required = {event_col, "failed_breakout_low", "prior_liquidity_low"}
    if not event_col or not required <= set(frame.columns):
        return pl.lit(False)
    return (
        (pl.col(event_col) == LiquidityEvent.FAILED_BREAKOUT_LOW)
        & pl.col("failed_breakout_low").fill_null(False)
        & pl.col("prior_liquidity_low").is_not_null()
    )


def _structural_failed_breakout_high_expr(frame: pl.DataFrame) -> pl.Expr:
    event_col = _event_column(frame)
    required = {event_col, "failed_breakout_high", "prior_liquidity_high"}
    if not event_col or not required <= set(frame.columns):
        return pl.lit(False)
    return (
        (pl.col(event_col) == LiquidityEvent.FAILED_BREAKOUT_HIGH)
        & pl.col("failed_breakout_high").fill_null(False)
        & pl.col("prior_liquidity_high").is_not_null()
    )


def _top_bucket_net(frame: pl.DataFrame, column: str, label: str, *, limit: int = 3) -> str:
    if frame.is_empty() or column not in frame.columns:
        return ""
    net_col = "net_pnl_usd" if "net_pnl_usd" in frame.columns else "pnl_usd"
    grouped = (
        frame.filter(pl.col(column).is_not_null())
        .group_by(column)
        .agg(
            pl.len().alias("trades"),
            pl.col(net_col).cast(pl.Float64).sum().alias("net"),
        )
        .sort("net")
        .head(limit)
    )
    if grouped.is_empty():
        return ""
    parts = [
        f"{row[column]}:{int(row['trades'])}/${float(row['net'] or 0.0):+.2f}"
        for row in grouped.iter_rows(named=True)
    ]
    return f"none_by_{label}=" + ",".join(parts)


def _top_pair_count(
    frame: pl.DataFrame,
    columns: tuple[str, str],
    label: str,
    *,
    limit: int = 3,
) -> str:
    if frame.is_empty() or not set(columns) <= set(frame.columns):
        return ""
    grouped = (
        frame.filter(pl.col(columns[0]).is_not_null() & pl.col(columns[1]).is_not_null())
        .group_by(list(columns))
        .agg(pl.len().alias("trades"))
        .sort("trades", descending=True)
        .head(limit)
    )
    if grouped.is_empty():
        return ""
    total = max(frame.height, 1)
    parts = []
    for row in grouped.iter_rows(named=True):
        key = "/".join(str(row[column] or "n/a") for column in columns)
        pct = int(row["trades"]) / total * 100.0
        parts.append(f"{key}:{int(row['trades'])}/{pct:.1f}%")
    return f"{label}=" + ",".join(parts)


def _loss_cause(row: dict[str, object]) -> str:
    for detector in (
        _accepted_breakout_loss,
        _reclaim_failed_loss,
        _failed_breakout_loss,
        _volatility_expansion_loss,
        _countertrend_loss,
        _none_event_loss,
    ):
        cause = detector(row)
        if cause:
            return cause
    reason = str(row.get("reason") or "")
    if reason == "stop":
        return LossCause.STOP_NO_REVERSION
    if reason in {"time", "strategy_exit", "thesis_failed", "signal_zero"}:
        return LossCause.EXIT_MISMATCH_OR_NO_REVERSION
    return reason or LossCause.UNCLASSIFIED


def _loss_cause_confidence(cause: str) -> str:
    return _LOSS_CAUSE_CONFIDENCE.get(cause, "low")


_LOSS_CAUSE_CONFIDENCE = {
    LossCause.ACCEPTED_BREAKOUT_AGAINST_REVERSION: "high",
    LossCause.RECLAIM_FAILED: "high",
    LossCause.FAILED_BREAKOUT: "high",
    LossCause.TREND_CONTINUATION_AGAINST_REVERSION: "medium",
    LossCause.VOLATILITY_EXPANSION: "medium",
}


def _row_event(row: dict[str, object]) -> str:
    return str(
        row.get(DiagnosticColumn.ENTRY_LIQUIDITY_EVENT_TYPE_BUCKET)
        or row.get(DiagnosticColumn.ENTRY_LIQUIDITY_EVENT_TYPE)
        or row.get("entry_liquidity_event")
        or ""
    )


def _accepted_breakout_loss(row: dict[str, object]) -> str:
    event = _row_event(row)
    if (
        _truthy(row.get("entry_breakout_acceptance_high"))
        or event == LiquidityEvent.BREAKOUT_ACCEPTANCE_HIGH
    ):
        return LossCause.ACCEPTED_BREAKOUT_AGAINST_REVERSION
    if (
        _truthy(row.get("entry_breakout_acceptance_low"))
        or event == LiquidityEvent.BREAKOUT_ACCEPTANCE_LOW
    ):
        return LossCause.ACCEPTED_BREAKOUT_AGAINST_REVERSION
    return ""


def _reclaim_failed_loss(row: dict[str, object]) -> str:
    if _row_event(row) in {LiquidityEvent.BULLISH_RECLAIM, LiquidityEvent.BEARISH_RECLAIM}:
        return LossCause.RECLAIM_FAILED
    return ""


def _failed_breakout_loss(row: dict[str, object]) -> str:
    if (
        _truthy(row.get("entry_failed_bullish_sweep"))
        or _truthy(row.get("entry_failed_bearish_sweep"))
        or _truthy(row.get("entry_failed_breakout_high"))
        or _truthy(row.get("entry_failed_breakout_low"))
        or _row_event(row)
        in {LiquidityEvent.FAILED_BREAKOUT_HIGH, LiquidityEvent.FAILED_BREAKOUT_LOW}
    ):
        return LossCause.FAILED_BREAKOUT
    return ""


def _volatility_expansion_loss(row: dict[str, object]) -> str:
    if row.get("entry_volatility_bucket") == "expanded":
        return LossCause.VOLATILITY_EXPANSION
    return ""


def _countertrend_loss(row: dict[str, object]) -> str:
    side = str(row.get("side") or "")
    structure = str(row.get("entry_structure_bucket") or row.get("entry_trend_bucket") or "")
    if (side == "buy" and structure == "downtrend") or (
        side == "sell" and structure == "uptrend"
    ):
        return LossCause.TREND_CONTINUATION_AGAINST_REVERSION
    return ""


def _none_event_loss(row: dict[str, object]) -> str:
    if _row_event(row) in {LiquidityEvent.NONE, "no_reclaim", ""}:
        return LossCause.UNCLASSIFIED_NONE_EVENT
    return ""


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (float, int)):
        return float(value) > 0.0
    return bool(value)


def _sweep_trade_expectancy(report: Report) -> str:
    if report.trades.is_empty() or "entry_liquidity_event" not in report.trades.columns:
        return "trade_expectancy_by_sweep=n/a"
    net_col = "net_pnl_usd" if "net_pnl_usd" in report.trades.columns else "pnl_usd"
    grouped = (
        report.trades.group_by("entry_liquidity_event")
        .agg(
            pl.len().alias("trades"),
            pl.col(net_col).cast(pl.Float64).mean().alias("expectancy"),
        )
        .sort("entry_liquidity_event")
    )
    parts = [
        (
            f"{row['entry_liquidity_event']}:{int(row['trades'])}/"
            f"${float(row['expectancy'] or 0.0):+.2f}"
        )
        for row in grouped.iter_rows(named=True)
    ]
    return "trade_expectancy_by_sweep=" + ",".join(parts)


def _event_trade_expectancy(report: Report) -> str:
    bucket = "entry_liquidity_event_type_bucket"
    if report.trades.is_empty() or bucket not in report.trades.columns:
        return "trade_expectancy_by_event=n/a"
    net_col = "net_pnl_usd" if "net_pnl_usd" in report.trades.columns else "pnl_usd"
    grouped = (
        report.trades.group_by(bucket)
        .agg(
            pl.len().alias("trades"),
            pl.col(net_col).cast(pl.Float64).mean().alias("expectancy"),
        )
        .sort(bucket)
    )
    parts = [
        f"{row[bucket]}:{int(row['trades'])}/${float(row['expectancy'] or 0.0):+.2f}"
        for row in grouped.iter_rows(named=True)
    ]
    return "trade_expectancy_by_event=" + ",".join(parts)


def _cross_structure_liquidity_row(report: Report) -> list[str]:
    if report.trades.is_empty():
        return ["Cross-bucket attribution", "no trades"]
    parts = [
        _cross_bucket_expectancy(
            report,
            ("side", "entry_structure_bucket"),
            "side_x_structure",
        ),
        _cross_bucket_expectancy(
            report,
            ("entry_market_stage_bucket", "entry_liquidity_event_type_bucket"),
            "stage_x_event",
        ),
    ]
    return ["Cross-bucket attribution", " ".join(part for part in parts if part)]


def _cross_bucket_expectancy(
    report: Report,
    columns: tuple[str, str],
    label: str,
    *,
    limit: int = 3,
) -> str:
    if not set(columns) <= set(report.trades.columns):
        return f"{label}=n/a"
    net_col = "net_pnl_usd" if "net_pnl_usd" in report.trades.columns else "pnl_usd"
    grouped = (
        report.trades.group_by(list(columns))
        .agg(
            pl.len().alias("trades"),
            pl.col(net_col).cast(pl.Float64).mean().alias("expectancy"),
        )
        .sort("expectancy")
        .head(limit)
    )
    if grouped.is_empty():
        return f"{label}=none"
    parts = []
    for row in grouped.iter_rows(named=True):
        key = "/".join(str(row[column] or "n/a") for column in columns)
        parts.append(f"{key}:{int(row['trades'])}/${float(row['expectancy'] or 0.0):+.2f}")
    return f"{label}=" + ",".join(parts)


def _filtered_mean(signal_frame: pl.DataFrame, expr: pl.Expr, column: str) -> float:
    if column not in signal_frame.columns:
        return 0.0
    value = signal_frame.filter(expr.fill_null(False)).select(
        pl.col(column).cast(pl.Float64).mean()
    ).item()
    return float(value or 0.0)


def _opportunity_counts_by_column(
    signal_frame: pl.DataFrame,
    raw_expr: pl.Expr,
    column: str,
    label: str,
) -> str:
    if column not in signal_frame.columns:
        return f"{label}=n/a"
    grouped = (
        signal_frame.filter(raw_expr.fill_null(False))
        .group_by(column)
        .agg(pl.len().alias("count"))
        .sort(column)
    )
    if grouped.is_empty():
        return f"{label}=none"
    parts = [f"{row[column]}:{int(row['count'])}" for row in grouped.iter_rows(named=True)]
    return f"{label}=" + ",".join(parts)


def _stage_trade_expectancy(report: Report) -> str:
    bucket = "entry_market_stage_bucket"
    if report.trades.is_empty() or bucket not in report.trades.columns:
        return "trade_expectancy_by_stage=n/a"
    net_col = "net_pnl_usd" if "net_pnl_usd" in report.trades.columns else "pnl_usd"
    grouped = (
        report.trades.group_by(bucket)
        .agg(
            pl.len().alias("trades"),
            pl.col(net_col).cast(pl.Float64).mean().alias("expectancy"),
        )
        .sort(bucket)
    )
    parts = [
        (
            f"{row[bucket]}:{int(row['trades'])}/"
            f"${float(row['expectancy'] or 0.0):+.2f}"
        )
        for row in grouped.iter_rows(named=True)
    ]
    return "trade_expectancy_by_stage=" + ",".join(parts)


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def format_layer_summary(
    label: str,
    report: Report,
    signal_diagnostics: dict[str, float],
    signal_frame: pl.DataFrame | None = None,
) -> str:
    rows = []
    if signal_frame is not None:
        rows.append(list(feature_layer(signal_frame)))
    rows.append(list(signal_layer(signal_diagnostics)))
    rows.append(list(lifecycle_layer(report)))
    rows.append(list(sizing_layer(report)))
    return f"{label}\n" + format_table(["Layer", "Summary"], rows)


def format_status_table(reports: list[Report], gates: RiskGateConfig) -> str:
    rows = []
    for report in reports:
        status = report_status(report, gates)
        d = report.diagnostics
        rows.append(
            [
                status.status,
                report.label,
                str(report.metrics.num_trades),
                f"{report.metrics.profit_factor:.2f}",
                f"{report.trade_expectancy_pct:+.2f}",
                f"{report.metrics.max_drawdown_pct:.1f}",
                f"{d.max_notional_exposure_pct:.1f}" if d is not None else "n/a",
                ",".join(status.reasons) or "-",
            ]
        )
    return format_table(
        ["Status", "Label", "Trades", "PF", "Exp%", "DD%", "MaxNot%", "Reasons"],
        rows,
    )


def assert_reports_pass(reports: list[Report], gates: RiskGateConfig) -> None:
    failures = [report_status(report, gates) for report in reports]
    failed = [status for status in failures if status.status == "FAIL"]
    if gates.fail_on_risk and failed:
        reasons = "; ".join(",".join(status.reasons) for status in failed)
        raise SystemExit(f"risk gates failed: {reasons}")
