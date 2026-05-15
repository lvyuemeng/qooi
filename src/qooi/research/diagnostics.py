"""Layered diagnostics for strategy improvement workflows."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from qooi.core.evaluate import Report, format_table
from qooi.research.config import RiskGateConfig


@dataclass(frozen=True)
class ReportStatus:
    status: str
    reasons: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.status in ("PASS", "WARN")


def report_status(report: Report, gates: RiskGateConfig) -> ReportStatus:
    reasons: list[str] = []
    metrics = report.metrics
    diagnostics = report.diagnostics
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
    if not reasons:
        return ReportStatus("PASS", ())
    if any(reason in reasons for reason in ("PF_LOW", "EXP_LOW", "DD_HIGH", "NOTIONAL_HIGH")):
        return ReportStatus("FAIL", tuple(reasons))
    return ReportStatus("WARN", tuple(reasons))


def feature_layer(frame: pl.DataFrame) -> tuple[str, str]:
    candidates = [
        col
        for col in (
            "close_z_score",
            "ewma_z_score",
            "robust_z_score",
            "dynamic_z_score",
            "volatility_ratio",
            "volatility_regime",
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
        "dynamic_z_extreme_pct",
        "robust_z_extreme_pct",
        "high_volatility_regime_pct",
    )
    parts = [f"{key}={values[key]:.1f}" for key in keys if key in values]
    return "SIGNAL", " ".join(parts) if parts else "n/a"


def lifecycle_layer(report: Report) -> tuple[str, str]:
    d = report.diagnostics
    if d is None:
        return "LIFECYCLE", "n/a"
    reasons = ",".join(f"{key}:{value}" for key, value in sorted(d.exit_reasons.items())) or "none"
    return (
        "LIFECYCLE",
        f"entries={d.entries} exits={d.exits} avg_hold={d.avg_bars_held:.1f} reasons={reasons}",
    )


def sizing_layer(report: Report) -> tuple[str, str]:
    d = report.diagnostics
    if d is None:
        return "SIZING", "n/a"
    return (
        "SIZING",
        f"avg_contracts={d.avg_active_exposure:.2f} max_contracts={d.max_active_exposure:.2f} "
        f"avg_notional={d.avg_notional_exposure_pct:.1f}% "
        f"max_notional={d.max_notional_exposure_pct:.1f}%",
    )


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
