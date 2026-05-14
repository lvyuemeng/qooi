"""Backtest result types, cost model, pair-spread and portfolio backtests.

Single-asset pipeline backtest: use BacktestExecutor from qooi.core.executor.
Styles (walk-forward, rolling-window): use qooi.core.styles.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from qooi.exchange.eval import EvalMetrics, compute_metrics
from qooi.strategies.portfolio import (
    AssetSignalState,
    PortfolioLimits,
    allocate_portfolio_weights,
)


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
class CostModel:
    slippage_pct: float = 0.0
    spread_pct: float = 0.0
    commission_pct: float = 0.00005
    market_impact_pct: float = 0.000
    short_borrow_rate: float = 0.0001

    @property
    def total_per_side(self) -> float:
        return self.slippage_pct + self.spread_pct + self.commission_pct


# ======================================================================
# Pair-spread backtest (two-leg, hedge ratio aware)
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
# Multi-asset portfolio backtest
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
        return PortfolioBacktestResult(empty, empty, metrics)

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
        return PortfolioBacktestResult(empty, empty, metrics)

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
