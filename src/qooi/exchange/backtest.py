"""Unified backtester — single-asset, pair-spread, and multi-asset portfolio.

All engines share the same CostModel, EvalMetrics, and equity curve format.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import polars as pl

from qooi.exchange.eval import EvalMetrics, compute_metrics
from qooi.strategies.portfolio import (
    AssetSignalState,
    PortfolioLimits,
    allocate_portfolio_weights,
)

SignalExpr = pl.Expr


# ======================================================================
# Shared result types
# ======================================================================


@dataclass
class BacktestResult:
    trades: pl.DataFrame
    equity_curve: pl.DataFrame
    metrics: EvalMetrics

    def __str__(self) -> str:
        return str(self.metrics)


@dataclass
class PairBacktestResult(BacktestResult):
    """Pair-trading result — same schema as BacktestResult."""


@dataclass
class PortfolioBacktestResult(BacktestResult):
    weights: pl.DataFrame = field(default_factory=pl.DataFrame)


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


# ======================================================================
# Cost / Risk / Config
# ======================================================================


@dataclass
class CostModel:
    slippage_pct: float = 0.0
    spread_pct: float = 0.0
    commission_pct: float = 0.00005
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


# ======================================================================
# 1. Single-asset backtest
# ======================================================================


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

    def _run_single(self, df: pl.DataFrame) -> BacktestResult:
        n = len(df)
        close = df["close"].to_list()
        signal = df["signal"].to_list()
        atr = self._clean_atr(df, n)
        r = self.risk

        equity = [self.initial_capital]
        trade_log: list[dict] = []
        pos = [0.0]
        active: dict | None = None
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
            sign_flip = p_prev != 0 and p != 0 and (p_prev * p < 0)
            if active is not None and (exit_reason or (p == 0 and p_prev != 0) or sign_flip):
                reason = exit_reason or "signal"
                trade_log.append(
                    self._close_active(active, df["timestamp"][i - 1], close[i - 1], reason)
                )
                stop_price = target_price = -1.0
                trailing_high = trailing_low = -1.0
                active = None
                size = 0.0

            turnover = abs(p - p_prev)
            if turnover > 0:
                prev_eq *= max(
                    0.0, 1.0 - turnover * (self.cost.total_per_side + self.cost.market_impact_pct)
                )

            if abs(p) > 0 and p != p_prev:
                if active is not None and p_prev != 0 and p_prev * p > 0:
                    size = abs(p)
                    active["size"] = size
                else:
                    entry_price = close[i - 1]
                    entry_price, size, stop_price, target_price, trailing_high, trailing_low = (
                        self._enter_position(p, entry_price, 0.0, prev_eq, atr_i, r)
                    )
                    if size > 0:
                        active = {
                            "direction": 1 if p > 0 else -1,
                            "entry_ts": df["timestamp"][i - 1],
                            "entry_price": entry_price,
                            "entry_equity": prev_eq,
                            "size": size,
                        }
            elif p == 0:
                size = 0.0

            pos_current = size * (1.0 if p > 0 else -1.0) if abs(p) > 0 else 0.0
            daily_ret = self._bar_return(close[i], close[i - 1], pos_current)
            equity.append(prev_eq * (1.0 + daily_ret))
            pos.append(pos_current)

        if active is not None:
            trade_log.append(self._close_active(active, df["timestamp"][-1], close[-1], "end"))

        eq_series = pl.Series(equity, dtype=pl.Float64)
        result_df = df.select(["timestamp", "close", "signal"]).with_columns(
            [
                pl.Series(pos).alias("position"),
                eq_series.alias("portfolio_value"),
                eq_series.pct_change().fill_null(0.0).alias("returns"),
            ]
        )
        return BacktestResult(
            trades=pl.DataFrame(trade_log) if trade_log else pl.DataFrame(),
            equity_curve=result_df,
            metrics=compute_metrics(
                result_df, trades=pl.DataFrame(trade_log) if trade_log else pl.DataFrame()
            ),
        )

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
        p_prev, cur_close, stop_price, target_price, trailing_high, trailing_low, atr_i, r
    ):
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

    def _close_active(self, active, exit_ts, exit_price, reason):
        net_price = (
            exit_price * (1.0 - self.cost.total_per_side)
            if active["direction"] > 0
            else exit_price * (1.0 + self.cost.total_per_side)
        )
        pnl = (net_price / active["entry_price"] - 1) * active["size"] * active["entry_equity"]
        return {
            "entry_time": active["entry_ts"],
            "exit_time": exit_ts,
            "side": "long" if active["direction"] > 0 else "short",
            "entry_price": active["entry_price"],
            "exit_price": net_price,
            "pnl": pnl,
            "reason": reason,
        }

    def _enter_position(self, p, entry_price, impact, prev_eq, atr_i, r):
        if p > 0:
            entry_price *= 1.0 + self.cost.total_per_side + impact
            stop = entry_price - r.atr_stop_mult * atr_i
            target = entry_price + r.atr_target_mult * atr_i
            trail_h, trail_l = entry_price, -1.0
        else:
            entry_price *= 1.0 - self.cost.total_per_side - impact
            stop = entry_price + r.atr_stop_mult * atr_i
            target = entry_price - r.atr_target_mult * atr_i
            trail_h, trail_l = -1.0, entry_price

        if r.position_sizing == "atr" and atr_i > 0:
            sz = r.max_risk_pct * prev_eq / (atr_i * r.atr_stop_mult)
            sz = min(sz, r.max_leverage)
        else:
            sz = min(abs(p), r.max_leverage) if r.max_leverage > 0 else 1.0
        return entry_price, max(sz, 0.0), stop, target, trail_h, trail_l

    def _bar_return(self, cur_close, prev_close, pos_current):
        if pos_current == 0:
            return 0.0
        ret = (cur_close / prev_close - 1) * pos_current
        if pos_current < 0:
            ret -= self.cost.short_borrow_rate
        return ret


# ======================================================================
# 2. Pair-spread backtest (two-leg, hedge ratio aware)
# ======================================================================


def run_pair_backtest(
    df: pl.DataFrame,
    *,
    signal_col: str = "signal",
    hedge_col: str = "hedge_ratio",
    left_col: str = "close_left",
    right_col: str = "close_right",
    initial_capital: float = 10_000.0,
    commission_per_side: float = 0.00005,
) -> PairBacktestResult:
    """Two-leg spread PnL with rolling hedge ratio.

    ``df`` must contain ``timestamp``, ``signal`` (directional),
    ``hedge_ratio`` (beta of left vs right), and the left/right close
    columns.
    """
    if df.is_empty():
        empty = pl.DataFrame()
        m = compute_metrics(
            pl.DataFrame({"portfolio_value": [initial_capital], "returns": [0.0], "signal": [0.0]})
        )
        return PairBacktestResult(empty, empty, m)

    left = df[left_col].to_list()
    right = df[right_col].to_list()
    signal = df[signal_col].to_list()
    beta = df[hedge_col].to_list()
    ts = df["timestamp"].to_list()

    equity = [initial_capital]
    positions = [0.0]
    trades: list[dict] = []

    active, entry_left, entry_right, entry_beta, entry_equity, entry_ts = (
        0.0,
        0.0,
        0.0,
        1.0,
        initial_capital,
        ts[0],
    )

    for i in range(1, len(df)):
        prev_sig = signal[i - 1]
        prev_beta = beta[i - 1] if beta[i - 1] != 0 else 1.0
        prev_eq = equity[-1]

        if active != prev_sig:
            if active != 0.0:
                w_l = 1.0 / (1.0 + abs(entry_beta))
                w_r = abs(entry_beta) / (1.0 + abs(entry_beta))
                spread_ret = active * (
                    w_l * (left[i - 1] / entry_left - 1) - w_r * (right[i - 1] / entry_right - 1)
                )
                spread_ret -= 2 * commission_per_side
                trades.append(
                    {
                        "entry_time": entry_ts,
                        "exit_time": ts[i - 1],
                        "side": "long_spread" if active > 0 else "short_spread",
                        "entry_left": entry_left,
                        "entry_right": entry_right,
                        "exit_left": left[i - 1],
                        "exit_right": right[i - 1],
                        "hedge_ratio": entry_beta,
                        "pnl": spread_ret * entry_equity,
                        "reason": "signal",
                    }
                )
            active = prev_sig
            if active != 0.0:
                entry_left, entry_right, entry_beta, entry_equity, entry_ts = (
                    left[i - 1],
                    right[i - 1],
                    prev_beta,
                    prev_eq,
                    ts[i - 1],
                )
                prev_eq *= 1.0 - 2 * commission_per_side

        daily_ret = 0.0
        if active != 0.0:
            w_l = 1.0 / (1.0 + abs(prev_beta))
            w_r = abs(prev_beta) / (1.0 + abs(prev_beta))
            daily_ret = active * (
                w_l * (left[i] / left[i - 1] - 1) - w_r * (right[i] / right[i - 1] - 1)
            )

        equity.append(prev_eq * (1.0 + daily_ret))
        positions.append(active)

    if active != 0.0:
        w_l = 1.0 / (1.0 + abs(entry_beta))
        w_r = abs(entry_beta) / (1.0 + abs(entry_beta))
        spread_ret = (
            active * (w_l * (left[-1] / entry_left - 1) - w_r * (right[-1] / entry_right - 1))
            - 2 * commission_per_side
        )
        trades.append(
            {
                "entry_time": entry_ts,
                "exit_time": ts[-1],
                "side": "long_spread" if active > 0 else "short_spread",
                "entry_left": entry_left,
                "entry_right": entry_right,
                "exit_left": left[-1],
                "exit_right": right[-1],
                "hedge_ratio": entry_beta,
                "pnl": spread_ret * entry_equity,
                "reason": "end",
            }
        )

    eq = pl.Series(equity, dtype=pl.Float64)
    eq_c = df.select(["timestamp", signal_col]).with_columns(
        [
            pl.Series(positions).alias("position"),
            eq.alias("portfolio_value"),
            eq.pct_change().fill_null(0.0).alias("returns"),
            pl.col(signal_col).alias("signal"),
        ]
    )
    return PairBacktestResult(
        trades=pl.DataFrame(trades) if trades else pl.DataFrame(),
        equity_curve=eq_c,
        metrics=compute_metrics(eq_c, trades=pl.DataFrame(trades) if trades else pl.DataFrame()),
    )


# ======================================================================
# 3. Multi-asset portfolio backtest
# ======================================================================


def run_portfolio_backtest(
    frames: dict[str, pl.DataFrame],
    *,
    signal_col: str = "signal",
    close_col: str = "close",
    atr_col: str = "atr_14",
    initial_capital: float = 10_000.0,
    commission_per_side: float = 0.00005,
    portfolio_limits: PortfolioLimits | None = None,
    default_sharpe: float = 0.0,
    default_drawdown_pct: float = 25.0,
    metrics_by_symbol: dict[str, dict[str, float]] | None = None,
) -> PortfolioBacktestResult:
    """Backtest multiple assets under one shared equity curve.

    Each frame must contain ``timestamp``, ``close``, ``signal``.
    Allocation is decided per-bar via ``allocate_portfolio_weights``.
    """
    if not frames:
        empty = pl.DataFrame()
        metrics = compute_metrics(
            pl.DataFrame({"portfolio_value": [initial_capital], "returns": [0.0], "signal": [0.0]})
        )
        return PortfolioBacktestResult(empty, empty, empty, metrics)

    limits = portfolio_limits or PortfolioLimits()
    symbols = list(frames.keys())
    prepared: dict[str, pl.DataFrame] = {}

    for sym, frame in frames.items():
        cols = ["timestamp", close_col, signal_col]
        if atr_col in frame.columns:
            cols.append(atr_col)
        ren = {close_col: f"close__{sym}", signal_col: f"signal__{sym}"}
        if atr_col in frame.columns:
            ren[atr_col] = f"atr__{sym}"
        prepared[sym] = frame.select(cols).rename(ren).sort("timestamp")

    merged = None
    for sym in symbols:
        merged = (
            prepared[sym]
            if merged is None
            else merged.join(prepared[sym], on="timestamp", how="inner")
        )
    if merged is None or merged.is_empty():
        empty = pl.DataFrame()
        metrics = compute_metrics(
            pl.DataFrame({"portfolio_value": [initial_capital], "returns": [0.0], "signal": [0.0]})
        )
        return PortfolioBacktestResult(empty, empty, empty, metrics)

    close_map = {s: merged[f"close__{s}"].to_list() for s in symbols}
    signal_map = {s: merged[f"signal__{s}"].fill_nan(0).fill_null(0).to_list() for s in symbols}
    atr_map = {
        s: merged[f"atr__{s}"].fill_nan(0).fill_null(0).to_list()
        if f"atr__{s}" in merged.columns
        else [1.0] * merged.height
        for s in symbols
    }
    ts = merged["timestamp"].to_list()

    equity = [initial_capital]
    portfolio_sig = [0.0]
    weight_rows: list[dict] = []
    trades: list[dict] = []

    weights = {s: 0.0 for s in symbols}
    entry_price = {s: 0.0 for s in symbols}
    entry_eq = {s: initial_capital for s in symbols}
    entry_time = {s: ts[0] for s in symbols}
    loss_streak = {s: 0 for s in symbols}
    asset_stats = {s: {"sharpe": default_sharpe, "dd": default_drawdown_pct} for s in symbols}
    if metrics_by_symbol:
        for s, vals in metrics_by_symbol.items():
            if s in asset_stats:
                asset_stats[s]["sharpe"] = vals.get("sharpe", asset_stats[s]["sharpe"])
                asset_stats[s]["dd"] = vals.get("dd", asset_stats[s]["dd"])

    for i in range(1, merged.height):
        prev_eq = equity[-1]

        states = []
        for s in symbols:
            vol = atr_map[s][i] / max(close_map[s][i], 1e-9)
            states.append(
                AssetSignalState(
                    symbol=s,
                    score=float(signal_map[s][i - 1]),
                    volatility=max(vol, 1e-6),
                    sharpe=asset_stats[s]["sharpe"],
                    drawdown_pct=asset_stats[s]["dd"],
                    loss_streak=loss_streak[s],
                )
            )

        new_weights = allocate_portfolio_weights(states, limits)

        for s in symbols:
            old_w, new_w = weights[s], new_weights.get(s, 0.0)
            old_dir = 1 if old_w > 0 else (-1 if old_w < 0 else 0)
            new_dir = 1 if new_w > 0 else (-1 if new_w < 0 else 0)
            if old_dir and old_dir != new_dir:
                pnl = (
                    old_dir * (close_map[s][i - 1] / entry_price[s] - 1) * abs(old_w) * entry_eq[s]
                )
                trades.append(
                    {
                        "symbol": s,
                        "entry_time": entry_time[s],
                        "exit_time": ts[i - 1],
                        "side": "long" if old_dir > 0 else "short",
                        "entry_price": entry_price[s],
                        "exit_price": close_map[s][i - 1],
                        "weight": old_w,
                        "pnl": pnl,
                        "reason": "rebalance",
                    }
                )
                loss_streak[s] = loss_streak[s] + 1 if pnl < 0 else 0
            if old_dir == 0 and new_dir:
                entry_price[s], entry_eq[s], entry_time[s] = close_map[s][i - 1], prev_eq, ts[i - 1]

        turnover = sum(abs(new_weights.get(s, 0.0) - weights[s]) for s in symbols)
        prev_eq *= 1.0 - turnover * commission_per_side

        total_ret = 0.0
        for s in symbols:
            w = new_weights.get(s, 0.0)
            if w:
                total_ret += w * (close_map[s][i] / close_map[s][i - 1] - 1)

        curr_eq = prev_eq * (1.0 + total_ret)
        equity.append(curr_eq)
        portfolio_sig.append(sum(abs(new_weights.get(s, 0.0)) for s in symbols))
        weights = {s: new_weights.get(s, 0.0) for s in symbols}
        row = {"timestamp": ts[i]}
        row.update({f"weight__{s}": weights[s] for s in symbols})
        weight_rows.append(row)

    for s in symbols:
        w = weights[s]
        if w:
            d = 1 if w > 0 else -1
            pnl = d * (close_map[s][-1] / entry_price[s] - 1) * abs(w) * entry_eq[s]
            trades.append(
                {
                    "symbol": s,
                    "entry_time": entry_time[s],
                    "exit_time": ts[-1],
                    "side": "long" if d > 0 else "short",
                    "entry_price": entry_price[s],
                    "exit_price": close_map[s][-1],
                    "weight": w,
                    "pnl": pnl,
                    "reason": "end",
                }
            )

    eq = pl.Series(equity, dtype=pl.Float64)
    eq_c = pl.DataFrame(
        {
            "timestamp": ts,
            "signal": portfolio_sig,
            "position": portfolio_sig,
            "portfolio_value": eq,
            "returns": eq.pct_change().fill_null(0.0),
        }
    )
    return PortfolioBacktestResult(
        equity_curve=eq_c,
        trades=pl.DataFrame(trades) if trades else pl.DataFrame(),
        weights=pl.DataFrame(weight_rows) if weight_rows else pl.DataFrame(),
        metrics=compute_metrics(eq_c, trades=pl.DataFrame(trades) if trades else pl.DataFrame()),
    )


# ======================================================================
# Walk-forward backtest
# ======================================================================


class WalkForwardBacktest:
    def __init__(self, config: WalkForwardConfig, backtest: Backtest) -> None:
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
        oos_timestamps: list[int] = []

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
                    oos_timestamps.extend(result.equity_curve["timestamp"].to_list())

        if oos_returns:
            oos_eq = _returns_to_equity(
                oos_returns, self._backtest.initial_capital, timestamps=oos_timestamps
            )
            combined_oos_metrics = compute_metrics(oos_eq)
        else:
            last = self._backtest._run_single(df)
            combined_oos_metrics = last.metrics

        stability = _compute_stability_metrics(windows)
        return WalkForwardResult(
            windows=windows, combined_oos_metrics=combined_oos_metrics, stability_metrics=stability
        )


def _returns_to_equity(returns, initial_capital=10000.0, timestamps=None):
    eq, current = [], initial_capital
    for idx, ret in enumerate(returns):
        if idx:
            current *= 1.0 + ret
        eq.append(current)
    n = len(eq)
    data = {
        "portfolio_value": pl.Series(eq, dtype=pl.Float64),
        "returns": pl.Series(returns, dtype=pl.Float64)
        if returns
        else pl.Series([], dtype=pl.Float64),
        "position": pl.Series([0.0] * n, dtype=pl.Float64),
        "signal": pl.Series([0.0] * n, dtype=pl.Float64),
    }
    if timestamps is not None and len(timestamps) == n:
        data["timestamp"] = pl.Series(timestamps, dtype=pl.Int64)
    return pl.DataFrame(data)


def _compute_stability_metrics(windows):
    import statistics

    test_sharpes = [w.metrics.sharpe_ratio for w in windows if w.label == "test"]
    train_sharpes = [w.metrics.sharpe_ratio for w in windows if w.label == "train"]
    if not test_sharpes:
        return {}
    mean_test = statistics.mean(test_sharpes)
    var_test = statistics.variance(test_sharpes) if len(test_sharpes) > 1 else 0.0
    overfit_count = sum(1 for t, v in zip(train_sharpes, test_sharpes) if t > v)
    total = min(len(train_sharpes), len(test_sharpes))
    return {
        "mean_test_sharpe": round(mean_test, 4),
        "std_test_sharpe": round(math.sqrt(var_test), 4) if var_test > 0 else 0.0,
        "overfit_ratio": round(overfit_count / total, 4) if total > 0 else 0.0,
    }
