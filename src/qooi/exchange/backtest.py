"""Vectorized backtester — dynamic position sizing, leverage, risk management."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import polars as pl

from qooi.exchange.eval import EvalMetrics, compute_metrics

SignalExpr = pl.Expr


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    trades: pl.DataFrame
    equity_curve: pl.DataFrame
    metrics: EvalMetrics

    def __str__(self) -> str:
        return str(self.metrics)


@dataclass
class WindowResult:
    label: str
    start: int
    end: int
    equity_curve: pl.DataFrame
    trades: pl.DataFrame
    metrics: EvalMetrics

    def __str__(self) -> str:
        return (
            f"  [{self.label:>7}]  Ret={self.metrics.total_return_pct:>7.2f}%  "
            f"Sharpe={self.metrics.sharpe_ratio:.2f}  DD={self.metrics.max_drawdown_pct:.1f}%  "
            f"Trades={self.metrics.num_trades}"
        )


@dataclass
class WalkForwardResult:
    windows: list[WindowResult]
    combined_oos_metrics: EvalMetrics
    stability_metrics: dict

    def __str__(self) -> str:
        parts = [str(self.combined_oos_metrics)]
        parts.append(f"  Combined OOS Sharpe:  {self.combined_oos_metrics.sharpe_ratio}")
        parts.append(f"  Stability:            {self.stability_metrics}")
        if self.windows:
            parts.append("\n  Walk-forward segments:")
            parts.extend(str(w) for w in self.windows)
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Cost, risk, config
# ---------------------------------------------------------------------------


@dataclass
class CostModel:
    slippage_pct: float = 0.0
    spread_pct: float = 0.0
    commission_pct: float = 0.00005  # OKX VIP0 maker: 0.005% per side
    market_impact_pct: float = 0.000
    short_borrow_rate: float = 0.0001

    @property
    def total_per_side(self) -> float:
        return self.slippage_pct + self.spread_pct + self.commission_pct


@dataclass
class RiskConfig:
    max_leverage: float = 1.0
    position_sizing: str = "fixed"
    max_risk_pct: float = 0.02
    atr_stop_mult: float = 3.0
    atr_target_mult: float = 6.0
    trailing_activation_mult: float = 2.0
    trailing_distance_mult: float = 2.0
    atr_col: str = "atr_14"


@dataclass
class WalkForwardConfig:
    train_windows: int = 3
    test_window: int = 1
    holdout_window: int = 1
    step: int = 1
    rebalance_bars: int = 20


# ---------------------------------------------------------------------------
# Single backtest
# ---------------------------------------------------------------------------


@dataclass
class Backtest:
    data: pl.DataFrame
    signal_expr: SignalExpr
    initial_capital: float = 10_000.0
    cost: CostModel = field(default_factory=CostModel)
    risk: RiskConfig = field(default_factory=RiskConfig)

    def run(self) -> BacktestResult:
        df = self.data.sort("timestamp").with_columns(self.signal_expr.alias("signal"))
        df = df.with_columns(pl.col("signal").forward_fill().fill_null(0.0))
        return self._run_single(df)

    # ------------------------------------------------------------------
    # Single backtest
    # ------------------------------------------------------------------

    def _run_single(self, df: pl.DataFrame) -> BacktestResult:
        n = len(df)
        close = df["close"].to_list()
        signal = df["signal"].to_list()
        atr = self._clean_atr(df, n)
        r = self.risk

        equity = [self.initial_capital]
        trade_log: list[dict] = []
        pos = [0.0]
        active: dict[float, dict] = {}

        stop_price = -1.0
        target_price = -1.0
        trailing_high = -1.0
        trailing_low = -1.0
        size = 0.0

        for i in range(1, n):
            p_prev = pos[-1]
            prev_eq = equity[-1]
            atr_i = atr[i - 1] if atr[i - 1] > 0 else atr[i] if atr[i] > 0 else 1.0
            p = self._clipped_signal(signal[i - 1])

            exit_reason: str | None = None
            if p_prev != 0 and stop_price > 0:
                exit_reason = self._check_exit(
                    p_prev,
                    close[i - 1],
                    stop_price,
                    target_price,
                    trailing_high,
                    trailing_low,
                    atr_i,
                    r,
                )

            if exit_reason:
                p = 0.0
            if exit_reason or (p == 0 and p_prev != 0):
                reason = exit_reason or "signal"
                trade_log.extend(
                    self._close_active(active, df["timestamp"][i - 1], close[i - 1], reason)
                )
                stop_price = target_price = -1.0
                trailing_high = trailing_low = -1.0
                active = {}

            if abs(p) > 0 and p != p_prev:
                entry_price = close[i - 1]
                impact = self.cost.market_impact_pct * abs(p - p_prev)
                entry_price, size, stop_price, target_price, trailing_high, trailing_low = (
                    self._enter_position(p, entry_price, impact, prev_eq, atr_i, r)
                )
                if size > 0:
                    active[p] = {
                        "entry_ts": df["timestamp"][i - 1],
                        "entry_price": entry_price,
                        "entry_equity": prev_eq,
                        "size": size,
                    }

            pos_current = size * (1.0 if p > 0 else -1.0) if abs(p) > 0 else 0.0
            daily_ret = self._bar_return(close[i], close[i - 1], pos_current)
            equity.append(prev_eq * (1.0 + daily_ret))
            pos.append(pos_current)

        trade_log.extend(self._close_active(active, df["timestamp"][-1], close[-1], "end"))

        eq_series = pl.Series(equity, dtype=pl.Float64)
        result_df = df.select(["timestamp", "close", "signal"]).with_columns(
            [
                pl.Series(pos).alias("position"),
                eq_series.alias("portfolio_value"),
                eq_series.pct_change().fill_null(0.0).alias("returns"),
            ]
        )
        trade_df = pl.DataFrame(trade_log) if trade_log else pl.DataFrame()
        return BacktestResult(
            trades=trade_df,
            equity_curve=result_df,
            metrics=compute_metrics(result_df, trades=trade_df),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _clean_atr(self, df: pl.DataFrame, n: int) -> list[float]:
        col = self.risk.atr_col
        raw = (
            df.get_column(col).fill_nan(0.0).fill_null(0.0)
            if col in df.columns
            else pl.Series([1.0] * n)
        )
        vals = raw.to_list()
        first = next((v for v in vals if v > 0), 1.0)
        return [v if v > 0 else first for v in vals]

    def _clipped_signal(self, raw: float) -> float:
        r = self.risk
        if r.max_leverage <= 0:
            return 0.0
        return max(-r.max_leverage, min(r.max_leverage, raw))

    @staticmethod
    def _check_exit(
        p_prev: float,
        cur_close: float,
        stop_price: float,
        target_price: float,
        trailing_high: float,
        trailing_low: float,
        atr_i: float,
        r: RiskConfig,
    ) -> str | None:
        if p_prev > 0:
            if cur_close <= stop_price:
                return "stop"
            if target_price > 0 and cur_close >= target_price:
                return "target"
            if trailing_high > 0:
                new_high = max(trailing_high, cur_close)
                if new_high - cur_close >= r.trailing_distance_mult * atr_i:
                    return "trailing_stop"
            return None
        if cur_close >= stop_price:
            return "stop"
        if target_price > 0 and cur_close <= target_price:
            return "target"
        if trailing_low > 0:
            new_low = min(trailing_low, cur_close)
            if cur_close - new_low >= r.trailing_distance_mult * atr_i:
                return "trailing_stop"
        return None

    def _close_active(self, active: dict, exit_ts, exit_price: float, reason: str) -> list[dict]:
        logs: list[dict] = []
        for sig_val, t in list(active.items()):
            net_price = (
                exit_price * (1.0 - self.cost.total_per_side)
                if sig_val > 0
                else exit_price * (1.0 + self.cost.total_per_side)
            )
            pnl = (net_price / t["entry_price"] - 1) * t["size"] * t["entry_equity"]
            logs.append(
                {
                    "entry_time": t["entry_ts"],
                    "exit_time": exit_ts,
                    "side": "long" if sig_val > 0 else "short",
                    "entry_price": t["entry_price"],
                    "exit_price": net_price,
                    "pnl": pnl,
                    "reason": reason,
                }
            )
        return logs

    def _enter_position(
        self,
        p: float,
        entry_price: float,
        impact: float,
        prev_eq: float,
        atr_i: float,
        r: RiskConfig,
    ):
        if p > 0:
            entry_price *= 1.0 + self.cost.total_per_side + impact
            stop = entry_price - r.atr_stop_mult * atr_i
            target = entry_price + r.atr_target_mult * atr_i
            trail_h = entry_price
            trail_l = -1.0
        else:
            entry_price *= 1.0 - self.cost.total_per_side - impact
            stop = entry_price + r.atr_stop_mult * atr_i
            target = entry_price - r.atr_target_mult * atr_i
            trail_h = -1.0
            trail_l = entry_price

        if r.position_sizing == "atr" and atr_i > 0:
            sz = r.max_risk_pct * prev_eq / (atr_i * r.atr_stop_mult)
            sz = min(sz, r.max_leverage)
        else:
            sz = min(abs(p), r.max_leverage) if r.max_leverage > 0 else 1.0
        sz = max(sz, 0.0)
        return entry_price, sz, stop, target, trail_h, trail_l

    def _bar_return(self, cur_close: float, prev_close: float, pos_current: float) -> float:
        if pos_current == 0:
            return 0.0
        ret = (cur_close / prev_close - 1) * pos_current
        if pos_current < 0:
            ret -= self.cost.short_borrow_rate
        return ret


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------


class WalkForwardBacktest:
    def __init__(
        self,
        config: WalkForwardConfig,
        backtest: Backtest,
    ) -> None:
        self._config = config
        self._backtest = backtest

    def run(self) -> WalkForwardResult:
        w = self._config
        df = self._backtest.data.sort("timestamp").with_columns(
            self._backtest.signal_expr.alias("signal")
        )
        df = df.with_columns(pl.col("signal").forward_fill().fill_null(0.0))

        window_total = w.train_windows + w.test_window + w.holdout_window
        total_bars = len(df)
        window_bars = w.rebalance_bars
        n_windows = total_bars // window_bars
        if n_windows < window_total + 1:
            last = self._backtest._run_single(df)
            return WalkForwardResult(
                windows=[],
                combined_oos_metrics=last.metrics,
                stability_metrics={"insufficient_windows": True},
            )

        windows: list[WindowResult] = []
        oos_returns: list[float] = []

        for start_win in range(0, n_windows - window_total + 1, w.step):
            train_end = (start_win + w.train_windows) * window_bars
            test_end = (start_win + w.train_windows + w.test_window) * window_bars
            holdout_end = (start_win + window_total) * window_bars

            for label, (lo, hi) in [
                ("train", (start_win * window_bars, train_end)),
                ("test", (train_end, test_end)),
                ("holdout", (test_end, holdout_end)),
            ]:
                seg = df.slice(lo, hi - lo)
                if seg.height < 2:
                    continue
                result = self._backtest._run_single(seg)
                win = WindowResult(
                    label=label,
                    start=lo,
                    end=hi,
                    equity_curve=result.equity_curve,
                    trades=result.trades,
                    metrics=result.metrics,
                )
                windows.append(win)
                if label == "test":
                    oos_returns.extend(result.equity_curve["returns"].to_list())

        if oos_returns:
            oos_eq = _returns_to_equity(oos_returns, self._backtest.initial_capital)
            combined_oos_metrics = compute_metrics(oos_eq)
        else:
            last = self._backtest._run_single(df)
            combined_oos_metrics = last.metrics

        stability = _compute_stability_metrics(windows)
        return WalkForwardResult(
            windows=windows,
            combined_oos_metrics=combined_oos_metrics,
            stability_metrics=stability,
        )


# ---------------------------------------------------------------------------
# Walk-forward helpers
# ---------------------------------------------------------------------------


def _returns_to_equity(returns: list[float], initial_capital: float = 10_000.0) -> pl.DataFrame:
    eq = [initial_capital]
    for r in returns:
        eq.append(eq[-1] * (1.0 + r))
    eq_series = pl.Series(eq, dtype=pl.Float64)
    return pl.DataFrame(
        {
            "portfolio_value": eq_series,
            "returns": eq_series.pct_change().fill_null(0.0),
            "position": pl.Series([0.0] * len(eq), dtype=pl.Float64),
            "signal": pl.Series([0.0] * len(eq), dtype=pl.Float64),
        }
    )


def _compute_stability_metrics(windows: list[WindowResult]) -> dict:
    test_sharpes = [w.metrics.sharpe_ratio for w in windows if w.label == "test"]
    train_sharpes = [w.metrics.sharpe_ratio for w in windows if w.label == "train"]

    if not test_sharpes:
        return {}

    import statistics

    mean_test = statistics.mean(test_sharpes)
    var_test = statistics.variance(test_sharpes) if len(test_sharpes) > 1 else 0.0
    overfit_count = sum(1 for t, v in zip(train_sharpes, test_sharpes) if t > v)
    total = min(len(train_sharpes), len(test_sharpes))
    overfit_ratio = overfit_count / total if total > 0 else 0.0

    return {
        "mean_test_sharpe": round(mean_test, 4),
        "std_test_sharpe": round(math.sqrt(var_test), 4) if var_test > 0 else 0.0,
        "overfit_ratio": round(overfit_ratio, 4),
    }
