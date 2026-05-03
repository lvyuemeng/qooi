"""OKX pipeline — compose data loading, indicators, signals, backtest, and charting.

Each stage returns a ``Stage`` that hands off to the next. Typical usage::

    result = (
        Pipeline()
        .load("BTC-USDT", bar="1D", days=90)
        .indicators()
        .signal(sma_cross_signal(10, 30))
        .backtest(capital=10_000)
        .plot()
    )

    print(result.metrics)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import polars as pl

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

plt.style.use("dark_background")

from qooi.exchange.backtest import Backtest, BacktestResult  # noqa: E402
from qooi.exchange.eval import EvalMetrics, compute_metrics  # noqa: E402
from qooi.exchange.indicator import add_indicators  # noqa: E402
from qooi.exchange.store import CacheStore  # noqa: E402
from qooi.strategies import sma_cross_signal  # noqa: E402

SignalExpr = pl.Expr


@dataclass
class Stage:
    df: pl.DataFrame = field(default_factory=pl.DataFrame)
    result: BacktestResult | None = None
    eval: EvalMetrics | None = None
    chart_path: str | None = None


class Pipeline:
    """Functional data pipeline for crypto quant.

    Steps::

        Pipeline()                          # init
            .load(...)                      # fetch/cache OHLCV
            .indicators()                   # add SMA/RSI/ATR/Bollinger
            .signal(expr)                   # apply trading signal
            .backtest(...)                  # run vectorized backtest
            .plot(...)                      # save chart → return Stage
    """

    def __init__(self) -> None:
        self._s = Stage()
        self._cs = CacheStore()

    # ------------------------------------------------------------------
    # Stage 1 — Load
    # ------------------------------------------------------------------

    def load(self, symbol: str = "BTC-USDT", bar: str = "1D", days: int = 90) -> Pipeline:
        try:
            df = self._cs.load(symbol, bar=bar)
        except FileNotFoundError:
            df = self._cs.refresh(symbol, bar=bar, days=days, overwrite=True)
        self._s.df = df
        return self

    # ------------------------------------------------------------------
    # Stage 2 — Indicators
    # ------------------------------------------------------------------

    def indicators(self) -> Pipeline:
        self._s.df = add_indicators(self._s.df)
        return self

    # ------------------------------------------------------------------
    # Stage 3 — Signal
    # ------------------------------------------------------------------

    def signal(self, expr: SignalExpr | None = None) -> Pipeline:
        _expr: SignalExpr = expr if expr is not None else sma_cross_signal(10, 30)
        self._s.df = self._s.df.with_columns(_expr.alias("signal"))
        return self

    # ------------------------------------------------------------------
    # Stage 4 — Backtest
    # ------------------------------------------------------------------

    def backtest(
        self,
        capital: float = 10_000,
        commission: float = 0.001,
        slippage: float = 0.0005,
    ) -> Pipeline:
        bt = Backtest(
            data=self._s.df,
            signal_expr=pl.col("signal"),
            initial_capital=capital,
            commission_pct=commission,
            slippage_pct=slippage,
        )
        self._s.result = bt.run()
        return self

    # ------------------------------------------------------------------
    # Stage 5 — Plot
    # ------------------------------------------------------------------

    def plot(self, out: str | None = None) -> Stage:
        result = self._s.result
        if result is None:
            raise RuntimeError("call .backtest() before .plot()")

        df = result.equity_curve
        m = result.metrics
        times = pl.from_epoch(df["timestamp"], time_unit="ms").to_list()
        close = df["close"].to_list()
        equity = df["portfolio_value"].to_list()
        sig = df["signal"].to_list()
        rets = df["returns"].to_list()

        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1, 1]}
        )

        ax1.plot(times, close, color="white", linewidth=1, label="Close")
        ax1.scatter(
            [times[i] for i, v in enumerate(sig) if v == 1.0],
            [close[i] for i, v in enumerate(sig) if v == 1.0],
            color="#00cc66",
            s=8,
            alpha=0.5,
            label="Long",
        )
        ax1.scatter(
            [times[i] for i, v in enumerate(sig) if v == -1.0],
            [close[i] for i, v in enumerate(sig) if v == -1.0],
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
            f"Sharpe={m['sharpe_ratio']}  "
            f"Return={m['total_return_pct']}%  "
            f"DD={m['max_drawdown_pct']}%  "
            f"Trades={m['num_trades']}",
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
            out = str(out_dir / "pipeline_backtest.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        self._s.chart_path = out
        return self._s

    # ------------------------------------------------------------------
    # Stage 5 — Evaluate
    # ------------------------------------------------------------------

    def evaluate(self) -> Pipeline:
        result = self._s.result
        if result is None:
            raise RuntimeError("call .backtest() before .evaluate()")
        self._s.eval = compute_metrics(
            equity_curve=result.equity_curve,
            trades=result.trades,
        )
        return self

    # ------------------------------------------------------------------
    # Shortcuts
    # ------------------------------------------------------------------

    def run(
        self,
        symbol: str = "BTC-USDT",
        bar: str = "1D",
        days: int = 90,
        capital: float = 10_000,
        signal_expr: SignalExpr | None = None,
        plot_out: str | None = None,
    ) -> Stage:
        """Run the full pipeline in one call."""
        return (
            self.load(symbol, bar, days)
            .indicators()
            .signal(signal_expr)
            .backtest(capital=capital)
            .evaluate()
            .plot(out=plot_out)
        )
