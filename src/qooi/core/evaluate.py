"""Evaluation layer — formatting and comparison.

Primary metrics for sparse systems = trade-level stats. Calendar-bar Sharpe /
Sortino kept as secondary diagnostics only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import polars as pl

from qooi.core.metrics import EvalMetrics, as_float, compute_metrics, infer_periods_per_year

EMPTY_TRADE_STATS = {
    "trade_expectancy_pct": 0.0,
    "trade_expectancy_usd": 0.0,
    "median_trade_pct": 0.0,
    "trade_sharpe": 0.0,
}


@dataclass(frozen=True)
class BacktestDiagnostics:
    bars: int = 0
    bars_processed: int = 0
    stopped_early: bool = False
    stop_bar_index: int | None = None
    nonzero_signal_bars: int = 0
    long_signal_bars: int = 0
    short_signal_bars: int = 0
    entries: int = 0
    exits: int = 0
    exit_reasons: dict[str, int] = field(default_factory=dict)
    avg_bars_held: float = 0.0
    avg_active_exposure: float = 0.0
    max_active_exposure: float = 0.0
    avg_notional_exposure_pct: float = 0.0
    max_notional_exposure_pct: float = 0.0
    final_open_positions: int = 0
    open_unrealized_pnl_usd: float = 0.0
    fee_usd: float = 0.0
    data_start: int | None = None
    data_end: int | None = None
    mark_to_market: bool = False
    drawdown_stop_pct: float | None = None

    @property
    def signal_bar_pct(self) -> float:
        return self.nonzero_signal_bars / self.bars * 100.0 if self.bars > 0 else 0.0


def _as_float(value: object, default: float = 0.0) -> float:
    return as_float(value, default)


def _trades_frame(trades: list[dict]) -> pl.DataFrame:
    if not trades:
        return pl.DataFrame(schema={"pnl": pl.Float64, "pnl_usd": pl.Float64})
    df = pl.DataFrame(trades)
    if "pnl" not in df.columns:
        df = df.with_columns(pl.lit(0.0).alias("pnl"))
    if "pnl_usd" not in df.columns:
        df = df.with_columns(pl.lit(0.0).alias("pnl_usd"))
    return df.with_columns(
        pl.col("pnl").cast(pl.Float64),
        pl.col("pnl_usd").cast(pl.Float64),
    )


def _equity_frame(
    equity: list[float],
    active_exposure: list[float] | None,
    timestamps: list[int] | None,
    signals: list[float] | None,
) -> pl.DataFrame:
    eq_series = pl.Series("portfolio_value", equity, dtype=pl.Float64)
    df = pl.DataFrame(
        {
            "portfolio_value": eq_series,
            "returns": eq_series.pct_change().fill_null(0.0),
        }
    )
    columns = []
    if active_exposure is not None and len(active_exposure) == len(equity):
        columns.append(pl.Series("active_exposure", active_exposure, dtype=pl.Float64))
    if timestamps is not None and len(timestamps) == len(equity):
        columns.append(pl.Series("timestamp", timestamps, dtype=pl.Int64))
    if signals is not None and len(signals) == len(equity):
        columns.append(pl.Series("signal", signals, dtype=pl.Float64))
    if columns:
        return df.with_columns(columns)
    return df


def _trade_stats(trades: pl.DataFrame) -> dict[str, float]:
    if trades.is_empty():
        return EMPTY_TRADE_STATS.copy()

    row = (
        trades.select(
            pl.col("pnl").mean().alias("mean_pnl"),
            pl.col("pnl").median().alias("median_pnl"),
            pl.col("pnl").std().alias("std_pnl"),
            pl.col("pnl_usd").mean().alias("mean_pnl_usd"),
            pl.len().alias("trade_count"),
        )
        .with_columns(
            pl.when((pl.col("std_pnl") > 0) & (pl.col("trade_count") > 0))
            .then(pl.col("mean_pnl") / pl.col("std_pnl") * pl.col("trade_count").sqrt())
            .otherwise(0.0)
            .alias("trade_sharpe")
        )
        .select(
            (pl.col("mean_pnl") * 100.0).alias("trade_expectancy_pct"),
            pl.col("mean_pnl_usd").alias("trade_expectancy_usd"),
            (pl.col("median_pnl") * 100.0).alias("median_trade_pct"),
            pl.col("trade_sharpe"),
        )
        .row(0, named=True)
    )
    return {key: _as_float(value) for key, value in row.items()}


def _active_stats(equity: pl.DataFrame, periods_per_year: int) -> dict[str, float]:
    active_expr = (
        pl.col("active_exposure").abs() > 1e-12
        if "active_exposure" in equity.columns
        else pl.col("returns").abs() > 1e-12
    )
    total_bars = max(equity.height, 1)
    row = (
        equity.with_columns(active_expr.alias("is_active"))
        .filter(pl.col("is_active"))
        .select(
            (pl.len() / total_bars * 100.0).alias("active_bar_pct"),
            pl.col("returns").cast(pl.Float64).mean().alias("active_mean"),
            pl.col("returns").cast(pl.Float64).std().alias("active_std"),
        )
        .with_columns(
            pl.when(pl.col("active_std") > 0)
            .then(pl.col("active_mean") / pl.col("active_std") * periods_per_year**0.5)
            .otherwise(0.0)
            .alias("active_bar_sharpe")
        )
        .select("active_bar_pct", "active_bar_sharpe")
        .row(0, named=True)
    )
    return {key: _as_float(value) for key, value in row.items()}


@dataclass
class Report:
    label: str
    trades: pl.DataFrame
    equity: pl.DataFrame
    metrics: EvalMetrics
    active_bar_pct: float
    active_bar_sharpe: float
    trade_expectancy_pct: float
    trade_expectancy_usd: float
    median_trade_pct: float
    trade_sharpe: float
    unstable_annualization: bool
    diagnostics: BacktestDiagnostics | None = None
    metadata: tuple[str, ...] = ()

    @classmethod
    def from_raw(
        cls,
        trades: list[dict],
        equity: list[float],
        pair,
        *,
        label: str = "",
        active_exposure: list[float] | None = None,
        timestamps: list[int] | None = None,
        signals: list[float] | None = None,
        diagnostics: BacktestDiagnostics | None = None,
        metadata: Sequence[str] = (),
        periods_per_year: int | None = None,
    ) -> Report:
        t_df = _trades_frame(trades)
        eq_df = _equity_frame(equity, active_exposure, timestamps, signals)

        metrics = compute_metrics(eq_df, trades=t_df, periods_per_year=periods_per_year)
        trade_stats = _trade_stats(t_df)
        active_stats = _active_stats(eq_df, periods_per_year or infer_periods_per_year(eq_df))

        unstable = metrics.num_trades < 20 or active_stats["active_bar_pct"] < 10.0

        return cls(
            label=label or pair.asset.symbol,
            trades=t_df,
            equity=eq_df,
            metrics=metrics,
            active_bar_pct=round(active_stats["active_bar_pct"], 2),
            active_bar_sharpe=round(active_stats["active_bar_sharpe"], 4),
            trade_expectancy_pct=round(trade_stats["trade_expectancy_pct"], 4),
            trade_expectancy_usd=round(trade_stats["trade_expectancy_usd"], 4),
            median_trade_pct=round(trade_stats["median_trade_pct"], 4),
            trade_sharpe=round(trade_stats["trade_sharpe"], 4),
            unstable_annualization=unstable,
            diagnostics=diagnostics,
            metadata=tuple(metadata),
        )

    def summary(self) -> str:
        m = self.metrics
        return (
            f"{self.label:30s} {m.num_trades:4d}tr  "
            f"{m.win_rate_pct:5.1f}%wr  {m.profit_factor:5.2f}pf  "
            f"Exp={self.trade_expectancy_pct:+6.2f}%  "
            f"TSh={self.trade_sharpe:+6.2f}  "
            f"ABSh={self.active_bar_sharpe:+6.2f}"
        )

    def table(self) -> str:
        m = self.metrics
        return (
            f"  Ret={m.total_return_pct:+7.2f}%  Exp={self.trade_expectancy_pct:+.2f}%  "
            f"Exp$={self.trade_expectancy_usd:+.2f}  MedT={self.median_trade_pct:+.2f}%  "
            f"TSharpe={self.trade_sharpe:+.2f}\n"
            f"  DD={m.max_drawdown_pct:5.1f}%  AvgDD={m.avg_drawdown_pct:.1f}%  "
            f"DDDays={m.drawdown_days:d}  ActiveBars={self.active_bar_pct:.1f}%  "
            f"ABSharpe={self.active_bar_sharpe:+.2f}\n"
            f"  Trades={m.num_trades:d}  WR={m.win_rate_pct:.1f}%  "
            f"AvgW={m.avg_win_pct:+.2f}%  AvgL={m.avg_loss_pct:+.2f}%  "
            f"P/L={m.profit_loss_ratio:.2f}  PF={m.profit_factor:.2f}\n"
            f"  CalSharpe={m.sharpe_ratio:+.2f}  CalSortino={m.sortino_ratio:+.2f}  "
            f"Ann={m.annual_return_pct:+.2f}%  Vol={m.annual_volatility_pct:.2f}%\n"
            f"  IC={m.ic_mean:+.4f}  IC_IR={m.ic_ir:+.2f}  IC+={m.ic_positive_pct:.0f}%  "
            f"UnstableAnn={'yes' if self.unstable_annualization else 'no'}"
        )

    def metric_sections(self) -> str:
        m = self.metrics
        d = self.diagnostics
        fee = d.fee_usd if d is not None else 0.0
        avg_notional = d.avg_notional_exposure_pct if d is not None else 0.0
        exposure_return = m.total_return_pct / avg_notional if avg_notional > 0 else 0.0
        metadata = "\n".join(f"  {item}" for item in self.metadata) or "  none"
        audit = "  diagnostics unavailable"
        if d is not None:
            audit = (
                f"  Bars={d.bars_processed}/{d.bars}  "
                f"StoppedEarly={'yes' if d.stopped_early else 'no'}  "
                f"OpenPos={d.final_open_positions}  Fees=${fee:.2f}\n"
                f"  DataStart={d.data_start or 'n/a'}  DataEnd={d.data_end or 'n/a'}  "
                f"MTM={'yes' if d.mark_to_market else 'no'}  "
                f"DDStop={d.drawdown_stop_pct if d.drawdown_stop_pct is not None else 'none'}"
            )
        return (
            "Run metadata\n"
            f"{metadata}\n"
            "Engine/Data Audit\n"
            f"{audit}\n"
            "Trade Metrics\n"
            f"  Trades={m.num_trades}  WR={m.win_rate_pct:.1f}%  PF={m.profit_factor:.2f}  "
            f"Exp={self.trade_expectancy_pct:+.2f}%  Exp$={self.trade_expectancy_usd:+.2f}  "
            f"MedT={self.median_trade_pct:+.2f}%\n"
            "Exposure Metrics\n"
            f"  ActiveBars={self.active_bar_pct:.1f}%  AvgNotional={avg_notional:.1f}%cap  "
            f"RetPerAvgNotional={exposure_return:+.2f}\n"
            "Equity Metrics\n"
            f"  Ret={m.total_return_pct:+.2f}%  DD={m.max_drawdown_pct:.1f}%  "
            f"AvgDD={m.avg_drawdown_pct:.1f}%  DDDays={m.drawdown_days}\n"
            "Annualized Diagnostics\n"
            f"  CalSharpe={m.sharpe_ratio:+.2f}  CalSortino={m.sortino_ratio:+.2f}  "
            f"Ann={m.annual_return_pct:+.2f}%  Vol={m.annual_volatility_pct:.2f}%  "
            f"UnstableAnn={'yes' if self.unstable_annualization else 'no'}"
        )

    def diagnostics_table(self) -> str:
        if self.diagnostics is None:
            return ""
        d = self.diagnostics
        reasons = ", ".join(f"{k}:{v}" for k, v in sorted(d.exit_reasons.items())) or "none"
        return (
            f"  BarsProcessed={d.bars_processed}/{d.bars}  "
            f"StoppedEarly={'yes' if d.stopped_early else 'no'}  "
            f"SignalBars={d.nonzero_signal_bars}/{d.bars} ({d.signal_bar_pct:.1f}%)  "
            f"Long={d.long_signal_bars}  Short={d.short_signal_bars}\n"
            f"  Entries={d.entries}  Exits={d.exits}  AvgHold={d.avg_bars_held:.1f} bars  "
            f"ExitReasons={reasons}\n"
            f"  AvgContracts={d.avg_active_exposure:.2f}  "
            f"MaxContracts={d.max_active_exposure:.2f}  "
            f"AvgNotional={d.avg_notional_exposure_pct:.1f}%cap  "
            f"MaxNotional={d.max_notional_exposure_pct:.1f}%cap\n"
            f"  Fees=${d.fee_usd:.2f}  OpenPos={d.final_open_positions}  "
            f"OpenUnrealized=${d.open_unrealized_pnl_usd:.2f}"
        )


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    n_cols = len(headers)
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def _pad(cell: str, width: int) -> str:
        return cell + " " * (width - len(cell))

    lines = ["  ".join(_pad(headers[i], col_widths[i]) for i in range(n_cols))]
    lines.append("  ".join("-" * col_widths[i] for i in range(n_cols)))
    for row in rows:
        lines.append("  ".join(_pad(str(row[i]), col_widths[i]) for i in range(n_cols)))
    return "\n".join(lines)


def compare(*reports: Report) -> str:
    headers = ["Label", "Trades", "WR%", "PF", "Exp%", "TSh", "ABSh", "Ret%", "CalSh"]
    rows = []
    for r in reports:
        m = r.metrics
        rows.append(
            [
                r.label,
                str(m.num_trades),
                f"{m.win_rate_pct:.0f}",
                f"{m.profit_factor:.2f}",
                f"{r.trade_expectancy_pct:+.2f}",
                f"{r.trade_sharpe:+.2f}",
                f"{r.active_bar_sharpe:+.2f}",
                f"{m.total_return_pct:+.2f}",
                f"{m.sharpe_ratio:+.2f}",
            ]
        )
    return format_table(headers, rows)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_finite(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if finite:
        return _mean(finite)
    if any(math.isinf(value) and value > 0 for value in values):
        return float("inf")
    return 0.0


def format_strategy_summary(reports: Sequence[Report]) -> str:
    total_trades = sum(r.metrics.num_trades for r in reports)
    avg_return = _mean([r.metrics.total_return_pct for r in reports])
    avg_expectancy = _mean([r.trade_expectancy_pct for r in reports])
    best = max(reports, key=lambda r: r.metrics.total_return_pct)
    worst = min(reports, key=lambda r: r.metrics.total_return_pct)
    return (
        "Current strategy metrics\n"
        f"  Pairs={len(reports)}  Trades={total_trades}  "
        f"AvgRet={avg_return:+.2f}%  AvgExp={avg_expectancy:+.2f}%\n"
        f"  Best={best.label} {best.metrics.total_return_pct:+.2f}%  "
        f"Worst={worst.label} {worst.metrics.total_return_pct:+.2f}%"
    )


def format_strategy_recommendations(strategy: str, reports: Sequence[Report]) -> str:
    if not reports:
        return "Strategy recommendation\n  none"

    total_trades = sum(r.metrics.num_trades for r in reports)
    avg_expectancy = _mean([r.trade_expectancy_pct for r in reports])
    avg_pf = _mean_finite([r.metrics.profit_factor for r in reports])
    worst_dd = max(r.metrics.max_drawdown_pct for r in reports)
    unstable = sum(1 for r in reports if r.unstable_annualization)
    losing = avg_expectancy < 0.0 or avg_pf < 1.0
    sparse = total_trades < 20

    lines = ["Strategy recommendation"]
    if losing:
        lines.append(
            f"  Reject current {strategy} baseline for allocation: AvgExp={avg_expectancy:+.2f}% "
            f"and finite AvgPF={avg_pf:.2f}."
        )
        lines.append(
            "  Next tuning should improve entry quality first; do not loosen filters "
            "or add recovery while expectancy is negative."
        )
    else:
        lines.append(
            f"  Candidate baseline: AvgExp={avg_expectancy:+.2f}% and finite AvgPF={avg_pf:.2f}."
        )
        lines.append(
            "  Validate with rolling or walk-forward before tuning exits or increasing exposure."
        )

    if sparse:
        lines.append(
            f"  Sample is sparse ({total_trades} trades); treat calendar "
            "Sharpe/Sortino as tertiary."
        )
    if worst_dd > 25.0:
        lines.append(
            f"  Worst drawdown is high ({worst_dd:.1f}%); cap exposure before "
            "increasing basket count."
        )
    if unstable:
        lines.append(
            f"  Unstable annualization on {unstable}/{len(reports)} symbols; "
            "prioritize PF, expectancy, drawdown, and OOS windows."
        )
    if strategy.startswith("momentum_burst"):
        lines.append(
            "  Current evidence favors comparing against rsi_bounce_reversion "
            "before more momentum sweeps."
        )
    return "\n".join(lines)


def _exposure_adjusted_return(report: Report) -> float:
    exposure_pct = (
        report.diagnostics.avg_notional_exposure_pct if report.diagnostics is not None else 0.0
    )
    if exposure_pct <= 0.0:
        return 0.0
    return report.metrics.total_return_pct / exposure_pct


def format_symbol_rankings(reports: Sequence[Report]) -> str:
    if not reports:
        return "Symbol rankings\n  none"

    best_return = max(reports, key=lambda r: r.metrics.total_return_pct)
    worst_return = min(reports, key=lambda r: r.metrics.total_return_pct)
    best_pf = max(reports, key=lambda r: r.metrics.profit_factor)
    worst_pf = min(reports, key=lambda r: r.metrics.profit_factor)
    best_exposure_adj = max(reports, key=_exposure_adjusted_return)
    worst_exposure_adj = min(reports, key=_exposure_adjusted_return)
    return (
        "Symbol rankings\n"
        f"  Return best={best_return.label} {best_return.metrics.total_return_pct:+.2f}%  "
        f"worst={worst_return.label} {worst_return.metrics.total_return_pct:+.2f}%\n"
        f"  PF best={best_pf.label} {best_pf.metrics.profit_factor:.2f}  "
        f"worst={worst_pf.label} {worst_pf.metrics.profit_factor:.2f}\n"
        f"  ExposureAdj best={best_exposure_adj.label} "
        f"{_exposure_adjusted_return(best_exposure_adj):+.2f}  "
        f"worst={worst_exposure_adj.label} "
        f"{_exposure_adjusted_return(worst_exposure_adj):+.2f}"
    )


def format_comparability_warnings(reports: Sequence[Report]) -> str:
    if len(reports) < 2:
        return ""
    ranges = {
        (
            r.diagnostics.data_start if r.diagnostics is not None else None,
            r.diagnostics.data_end if r.diagnostics is not None else None,
        )
        for r in reports
    }
    dd_stops = {
        r.diagnostics.drawdown_stop_pct if r.diagnostics is not None else None for r in reports
    }
    metadata = {r.metadata for r in reports}
    warnings = []
    if len(ranges) > 1:
        warnings.append("data ranges differ")
    if len(dd_stops) > 1:
        warnings.append("drawdown stop settings differ")
    if len(metadata) > 1:
        warnings.append("run metadata differs")
    return "Comparability warning: " + "; ".join(warnings) if warnings else ""


def format_benchmark_report(
    *,
    mode: str,
    benchmark_results: Sequence[tuple[str, Sequence[Report]]],
    diagnostics: bool = False,
) -> str:
    rows = []
    for name, reports in benchmark_results:
        if not reports:
            continue
        total_trades = sum(r.metrics.num_trades for r in reports)
        avg_return = _mean([r.metrics.total_return_pct for r in reports])
        avg_pf = _mean_finite([r.metrics.profit_factor for r in reports])
        avg_expectancy = _mean([r.trade_expectancy_pct for r in reports])
        avg_active = _mean([r.active_bar_pct for r in reports])
        avg_notional = _mean(
            [
                r.diagnostics.avg_notional_exposure_pct if r.diagnostics is not None else 0.0
                for r in reports
            ]
        )
        max_notional = max(
            (r.diagnostics.max_notional_exposure_pct if r.diagnostics is not None else 0.0)
            for r in reports
        )
        rows.append(
            [
                name,
                str(len(reports)),
                str(total_trades),
                f"{avg_return:+.2f}",
                f"{avg_pf:.2f}",
                f"{avg_expectancy:+.2f}",
                f"{avg_active:.1f}",
                f"{avg_notional:.1f}",
                f"{max_notional:.1f}",
            ]
        )

    lines = [
        f"Mode: {mode}",
        "Benchmark: strategy variants",
        format_table(
            [
                "Variant",
                "Pairs",
                "Trades",
                "AvgRet%",
                "AvgPF",
                "Exp%",
                "ActBar%",
                "AvgNot%",
                "MaxNot%",
            ],
            rows,
        ),
    ]
    for name, reports in benchmark_results:
        if not reports:
            continue
        lines.append(f"\n{name}")
        warning = format_comparability_warnings(reports)
        if warning:
            lines.append(warning)
        lines.append(compare(*reports))
        lines.append(format_symbol_rankings(reports))
        if diagnostics:
            for report in reports:
                if report.diagnostics is None:
                    continue
                lines.append(f"\n{report.label}")
                lines.append(report.diagnostics_table())
    return "\n".join(lines)


def format_backtest_report(
    *,
    mode: str,
    strategy: str,
    reports: Sequence[Report],
    detail: bool = True,
    diagnostics: bool = False,
    signal_diagnostics: Sequence[tuple[str, dict[str, float]]] = (),
) -> str:
    lines = [
        f"Mode: {mode}",
        f"Strategy: {strategy}",
    ]
    warning = format_comparability_warnings(reports)
    if warning:
        lines.append(warning)
    lines.extend([compare(*reports), "", format_strategy_summary(reports)])
    lines.extend(["", format_strategy_recommendations(strategy, reports)])
    if detail:
        for report in reports:
            lines.append(f"\n{report.label}")
            lines.append(report.metric_sections())
            if diagnostics and report.diagnostics is not None:
                lines.append(report.diagnostics_table())
    elif diagnostics:
        lines.append("\nDiagnostics")
        for report in reports:
            if report.diagnostics is None:
                continue
            lines.append(f"\n{report.label}")
            lines.append(report.diagnostics_table())
    if diagnostics and signal_diagnostics:
        lines.append("\nSignal filter diagnostics")
        for label, values in signal_diagnostics:
            lines.append(format_signal_diagnostics(label, values))
    lines.append("")
    lines.append(format_symbol_rankings(reports))
    return "\n".join(lines)


def format_signal_diagnostics(label: str, diagnostics: dict[str, float]) -> str:
    parts = [f"{key}={value:.1f}" for key, value in sorted(diagnostics.items())]
    return f"  {label}: " + "  ".join(parts)
