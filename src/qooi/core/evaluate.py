"""Evaluation layer — formatting, comparison, and plotting.

Strategy-independent: takes raw trades + equity and produces formatted
output.  Works identically for any backtest style (single-run, walk-forward,
rolling window, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from qooi.exchange.eval import EvalMetrics, compute_metrics


@dataclass
class Report:
    label: str
    trades: pl.DataFrame
    equity: pl.DataFrame
    metrics: EvalMetrics

    @classmethod
    def from_raw(cls, trades: list[dict], equity: list[float], pair, *, label: str = "") -> Report:
        t_df = pl.DataFrame(trades) if trades else pl.DataFrame()
        eq_df = pl.DataFrame(
            {
                "portfolio_value": equity,
                "returns": pl.Series(equity).pct_change().fill_null(0.0),
            }
        )
        m = compute_metrics(eq_df, trades=t_df)
        return cls(
            label=label or f"{pair.asset.symbol} {pair.okx.strategy}",
            trades=t_df,
            equity=eq_df,
            metrics=m,
        )

    def summary(self) -> str:
        m = self.metrics
        return (
            f"{self.label:30s} {m.num_trades:4d}tr  "
            f"{m.total_return_pct:+7.2f}%  {m.win_rate_pct:5.1f}%wr  "
            f"{m.profit_factor:5.2f}pl  {m.sharpe_ratio:+6.2f}sh"
        )

    def table(self) -> str:
        m = self.metrics
        return (
            f"  Ret={m.total_return_pct:+7.2f}%  Ann={m.annual_return_pct:+.2f}%  "
            f"Vol={m.annual_volatility_pct:.2f}%  Sharpe={m.sharpe_ratio:+.2f}  "
            f"Sortino={m.sortino_ratio:+.2f}\n"
            f"  DD={m.max_drawdown_pct:5.1f}%  AvgDD={m.avg_drawdown_pct:.1f}%  "
            f"DDDays={m.drawdown_days:d}  Calmar={m.calmar_ratio:+.2f}\n"
            f"  Trades={m.num_trades:d}  WR={m.win_rate_pct:.1f}%  "
            f"AvgW={m.avg_win_pct:+.2f}%  AvgL={m.avg_loss_pct:+.2f}%  "
            f"PL={m.profit_factor:.2f}\n"
            f"  IC={m.ic_mean:+.4f}  IC_IR={m.ic_ir:+.2f}  IC+={m.ic_positive_pct:.0f}%"
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
    headers = ["Label", "Trades", "Ret%", "WR%", "PL", "Sharpe", "IC"]
    rows = []
    for r in reports:
        m = r.metrics
        rows.append(
            [
                r.label,
                str(m.num_trades),
                f"{m.total_return_pct:+.2f}",
                f"{m.win_rate_pct:.0f}",
                f"{m.profit_factor:.2f}",
                f"{m.sharpe_ratio:+.2f}",
                f"{m.ic_mean:+.4f}",
            ]
        )
    return format_table(headers, rows)
