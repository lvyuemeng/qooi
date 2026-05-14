"""Evaluation layer — formatting and comparison.

Primary metrics for sparse systems = trade-level stats. Calendar-bar Sharpe /
Sortino kept as secondary diagnostics only.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from qooi.exchange.eval import EvalMetrics, compute_metrics

EMPTY_TRADE_STATS = {
    "trade_expectancy_pct": 0.0,
    "trade_expectancy_usd": 0.0,
    "median_trade_pct": 0.0,
    "trade_sharpe": 0.0,
}


def _as_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return default


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


def _equity_frame(equity: list[float], active_exposure: list[float] | None) -> pl.DataFrame:
    eq_series = pl.Series("portfolio_value", equity, dtype=pl.Float64)
    df = pl.DataFrame(
        {
            "portfolio_value": eq_series,
            "returns": eq_series.pct_change().fill_null(0.0),
        }
    )
    if active_exposure is not None and len(active_exposure) == len(equity):
        return df.with_columns(pl.Series("active_exposure", active_exposure, dtype=pl.Float64))
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

    @classmethod
    def from_raw(
        cls,
        trades: list[dict],
        equity: list[float],
        pair,
        *,
        label: str = "",
        active_exposure: list[float] | None = None,
        periods_per_year: int = 365 * 24,
    ) -> Report:
        t_df = _trades_frame(trades)
        eq_df = _equity_frame(equity, active_exposure)

        metrics = compute_metrics(eq_df, trades=t_df, periods_per_year=periods_per_year)
        trade_stats = _trade_stats(t_df)
        active_stats = _active_stats(eq_df, periods_per_year)

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
