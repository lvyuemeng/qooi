"""Backtest styles — walk-forward, rolling-window, cross-validation.

Strategy-independent: takes a trades_fn (DataFrame → (trades, equity)) and
runs it repeatedly under different slicing regimes.  Returns StyleResult
with aggregated OOS metrics and stability stats.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import polars as pl

from qooi.core.metrics import EvalMetrics, compute_metrics

BacktestFn = Callable[[pl.DataFrame], object]
TradesFn = Callable[[pl.DataFrame], tuple[list[dict], list[float]]]


@dataclass
class WindowSlice:
    label: str
    start: int
    end: int
    trades: list[dict]
    equity: list[float]
    metrics: EvalMetrics

    def summary(self) -> str:
        m = self.metrics
        return (
            f"  [{self.label:>7}]  [{self.start:5d}-{self.end:5d}]  "
            f"Ret={m.total_return_pct:+7.2f}%  Sharpe={m.sharpe_ratio:+.2f}  "
            f"DD={m.max_drawdown_pct:.1f}%  Trades={m.num_trades}"
        )


@dataclass
class StyleResult:
    label: str
    windows: list[WindowSlice]
    combined_metrics: EvalMetrics
    stability: dict[str, float]

    def summary(self) -> str:
        lines = [f"=== {self.label} ==="]
        for w in self.windows:
            lines.append(w.summary())
        m = self.combined_metrics
        lines.append(
            f"  OOS Sharpe={m.sharpe_ratio:+.2f}  "
            f"DD={m.max_drawdown_pct:.1f}%  "
            f"Trades={m.num_trades}  "
            f"Stability={self.stability}"
        )
        return "\n".join(lines)


def walk_forward(
    trades_fn: BacktestFn,
    df: pl.DataFrame,
    *,
    train_bars: int = 500,
    test_bars: int = 100,
    step_bars: int = 100,
    holdout_bars: int = 0,
    label: str = "walk_forward",
) -> StyleResult:
    n = df.height
    windows: list[WindowSlice] = []
    oos_trades: list[dict] = []
    oos_rets: list[float] = []

    start = 0
    while start + train_bars + test_bars + holdout_bars <= n:
        train_end = start + train_bars
        test_end = train_end + test_bars
        holdout_end = test_end + holdout_bars

        slices = [("train", start, train_end), ("test", train_end, test_end)]
        if holdout_bars > 0:
            slices.append(("holdout", test_end, holdout_end))

        for sl_label, lo, hi in slices:
            seg = df.slice(lo, hi - lo)
            if seg.height < 2:
                continue
            trades, equity, eq_df, td = _coerce_result(trades_fn(seg))
            m = compute_metrics(eq_df, trades=td)
            windows.append(
                WindowSlice(
                    label=sl_label,
                    start=lo,
                    end=hi,
                    trades=trades,
                    equity=equity,
                    metrics=m,
                )
            )
            if sl_label in ("test", "holdout"):
                oos_trades.extend(trades)
                n_eq = len(equity)
                if n_eq > 1:
                    oos_rets.extend([equity[i] / equity[i - 1] - 1 for i in range(1, n_eq)])

        start += step_bars

    oos_eq = _build_equity(oos_rets)
    oos_td = pl.DataFrame(oos_trades) if oos_trades else pl.DataFrame()
    combined = compute_metrics(oos_eq, trades=oos_td)
    stability = _stability(windows)
    return StyleResult(
        label=label,
        windows=windows,
        combined_metrics=combined,
        stability=stability,
    )


def rolling_window(
    trades_fn: BacktestFn,
    df: pl.DataFrame,
    *,
    lookback_bars: int = 500,
    step_bars: int = 100,
    label: str = "rolling_window",
) -> StyleResult:
    return walk_forward(
        trades_fn,
        df,
        train_bars=lookback_bars,
        test_bars=step_bars,
        step_bars=step_bars,
        holdout_bars=0,
        label=label,
    )


def cross_validate(
    trades_fn: BacktestFn,
    df: pl.DataFrame,
    *,
    folds: int = 5,
    label: str = "cross_validate",
) -> StyleResult:
    n = df.height
    fold_size = n // folds
    if fold_size < 2:
        raise ValueError(f"Not enough bars ({n}) for {folds} folds")

    windows: list[WindowSlice] = []
    oos_trades: list[dict] = []
    oos_rets: list[float] = []

    for k in range(folds):
        lo = k * fold_size
        hi = min((k + 1) * fold_size, n)
        seg = df.slice(lo, hi - lo)
        if seg.height < 2:
            continue
        trades, equity, eq_df, td = _coerce_result(trades_fn(seg))
        m = compute_metrics(eq_df, trades=td)
        windows.append(
            WindowSlice(
                label=f"fold_{k}",
                start=lo,
                end=hi,
                trades=trades,
                equity=equity,
                metrics=m,
            )
        )
        oos_trades.extend(trades)
        n_eq = len(equity)
        if n_eq > 1:
            oos_rets.extend([equity[i] / equity[i - 1] - 1 for i in range(1, n_eq)])

    oos_eq = _build_equity(oos_rets)
    oos_td = pl.DataFrame(oos_trades) if oos_trades else pl.DataFrame()
    combined = compute_metrics(oos_eq, trades=oos_td)
    stability = _stability(windows)
    return StyleResult(
        label=label,
        windows=windows,
        combined_metrics=combined,
        stability=stability,
    )


def _build_equity(returns: list[float], initial: float = 10000.0) -> pl.DataFrame:
    eq = [initial]
    for r in returns:
        eq.append(eq[-1] * (1.0 + r))
    return pl.DataFrame(
        {
            "portfolio_value": eq,
            "returns": [0.0] + returns,
        }
    )


def _coerce_result(result: object) -> tuple[list[dict], list[float], pl.DataFrame, pl.DataFrame]:
    if isinstance(result, tuple):
        trades = cast(list[dict], result[0])
        equity = cast(list[float], result[1])
        eq_df = pl.DataFrame(
            {
                "portfolio_value": equity,
                "returns": pl.Series(equity).pct_change().fill_null(0.0),
            }
        )
        td = pl.DataFrame(trades) if trades else pl.DataFrame()
        return trades, equity, eq_df, td
    trades_df = getattr(result, "trades")
    equity_df = getattr(result, "equity")
    trades = trades_df.to_dicts() if isinstance(trades_df, pl.DataFrame) else []
    equity = equity_df["portfolio_value"].to_list() if isinstance(equity_df, pl.DataFrame) else []
    return trades, equity, equity_df, trades_df


def _stability(windows: list[WindowSlice]) -> dict[str, float]:
    sharpe_train = [w.metrics.sharpe_ratio for w in windows if "train" in w.label]
    sharpe_test = [
        w.metrics.sharpe_ratio for w in windows if "test" in w.label or "fold" in w.label
    ]
    if not sharpe_test:
        return {}
    mean_test = statistics.mean(sharpe_test)
    std_test = statistics.stdev(sharpe_test) if len(sharpe_test) > 1 else 0.0
    paired = min(len(sharpe_train), len(sharpe_test))
    overfit = (
        sum(1 for i in range(paired) if sharpe_train[i] > sharpe_test[i]) / paired
        if paired > 0
        else 0.0
    )
    return {
        "mean_oos_sharpe": round(mean_test, 4),
        "std_oos_sharpe": round(std_test, 4),
        "overfit_ratio": round(overfit, 4),
    }
