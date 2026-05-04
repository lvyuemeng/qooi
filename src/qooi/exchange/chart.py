"""Standalone charting — backtest equity curve, price, signals, and returns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import polars as pl

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

plt.style.use("dark_background")

from qooi.exchange.eval import EvalMetrics  # noqa: E402


@dataclass
class ChartResult:
    chart_path: str | None = None


def plot_backtest(
    equity_curve: pl.DataFrame,
    metrics: EvalMetrics,
    out: str | None = None,
) -> ChartResult:
    times = pl.from_epoch(equity_curve["timestamp"], time_unit="ms").to_list()
    close = equity_curve["close"].to_list()
    equity = equity_curve["portfolio_value"].to_list()
    sig = equity_curve["signal"].to_list()
    rets = equity_curve["returns"].to_list()

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1, 1]}
    )

    ax1.plot(times, close, color="white", linewidth=1, label="Close")
    longs = [(times[i], close[i]) for i, v in enumerate(sig) if v == 1.0]
    shorts = [(times[i], close[i]) for i, v in enumerate(sig) if v == -1.0]
    if longs:
        ax1.scatter(
            [t for t, _ in longs],
            [c for _, c in longs],
            color="#00cc66",
            s=8,
            alpha=0.5,
            label="Long",
        )
    if shorts:
        ax1.scatter(
            [t for t, _ in shorts],
            [c for _, c in shorts],
            color="#ff3355",
            s=8,
            alpha=0.5,
            label="Short",
        )
    ax1.set_ylabel("Price (USDT)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.15)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    ax1.set_title(
        f"Sharpe={metrics.sharpe_ratio}  "
        f"Return={metrics.total_return_pct}%  DD={metrics.max_drawdown_pct}%  "
        f"Trades={metrics.num_trades}",
        fontsize=11,
    )

    ax2.plot(times, equity, color="#00aaff", linewidth=1.2)
    ax2.fill_between(times, equity[0], equity, alpha=0.1, color="#00aaff")
    ax2.set_ylabel("Equity (USDT)")
    ax2.grid(alpha=0.15)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    ax3.bar(times, rets, width=0.6, color="#8888ff", alpha=0.5)
    ax3.axhline(0, color="gray", linewidth=0.5)
    ax3.set_ylabel("Daily Return")
    ax3.grid(alpha=0.15)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    plt.tight_layout()
    if out is None:
        out_dir = Path("data/charts")
        out_dir.mkdir(parents=True, exist_ok=True)
        out = str(out_dir / "backtest.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)

    return ChartResult(chart_path=out)
