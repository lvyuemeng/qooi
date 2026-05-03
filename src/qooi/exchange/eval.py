"""Strategy evaluation metrics — IC, IR, win rate, profit/loss ratio, drawdown, etc.

All functions work with the DataFrame produced by ``BacktestResult.equity_curve``
which has columns: timestamp, close, signal, position, portfolio_value, returns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class EvalMetrics:
    """Comprehensive strategy evaluation report."""

    # Return & risk
    total_return_pct: float
    annual_return_pct: float
    annual_volatility_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    avg_drawdown_pct: float
    drawdown_days: int

    # Trade statistics
    num_trades: int
    win_rate_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_loss_ratio: float
    expectancy: float
    profit_factor: float

    # Information coefficient
    ic_mean: float
    ic_std: float
    ic_ir: float
    ic_positive_pct: float

    factor_return_pct: float


def compute_metrics(
    equity_curve: pl.DataFrame,
    trades: pl.DataFrame | None = None,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 365,
) -> EvalMetrics:
    """Compute all evaluation metrics from a backtest result.

    Parameters
    ----------
    equity_curve:
        Must have columns: portfolio_value, returns (daily returns).
    trades:
        Optional trade log with columns: entry_time, pnl.
    risk_free_rate:
        Annual risk-free rate (default 2%).
    periods_per_year:
        Annualization factor (365 for daily, 365*24 for hourly, etc.)
    """
    rets = equity_curve["returns"].to_list()
    equity = equity_curve["portfolio_value"].to_list()

    # --- Basic return & risk ---
    total_ret = (equity[-1] / equity[0]) - 1
    n = len(rets)
    ann_factor = periods_per_year / n if n > 0 else 1.0
    ann_ret = (1 + total_ret) ** ann_factor - 1 if total_ret > -1 else -1.0

    ret_arr = np.array(rets, dtype=np.float64)
    std = float(np.nanstd(ret_arr)) if n > 1 else 0.0
    ann_vol = std * math.sqrt(periods_per_year) if std > 0 else 0.0
    excess = ann_ret - risk_free_rate
    sharpe = excess / ann_vol if ann_vol > 0 else 0.0

    # Sortino (downside deviation only)
    neg_rets = ret_arr[ret_arr < 0]
    downside = float(np.nanstd(neg_rets)) if len(neg_rets) > 1 else 0.0
    ann_downside = downside * math.sqrt(periods_per_year)
    sortino = excess / ann_downside if ann_downside > 0 else 0.0

    # Drawdown
    peaks = [equity[0]]
    dd_series = [0.0]
    for v in equity[1:]:
        p = max(peaks[-1], v)
        peaks.append(p)
        dd_series.append((p - v) / p if p > 0 else 0.0)
    max_dd = max(dd_series) if dd_series else 0.0
    avg_dd = float(np.mean(dd_series)) if dd_series else 0.0

    calmar = ann_ret / max_dd if max_dd > 0 else 0.0

    # Drawdown duration
    dd_days = 0
    for i, dd in enumerate(dd_series):
        if dd > 0.01:
            dd_days += 1
        else:
            dd_days = 0

    # --- Trade statistics ---
    num_trades = 0
    win_rate = 0.0
    avg_win = 0.0
    avg_loss = 0.0
    pl_ratio = 0.0
    expectancy = 0.0
    profit_factor = 0.0

    if trades is not None and not trades.is_empty():
        pnl_col = "pnl"
        if pnl_col not in trades.columns:
            # Try to compute PnL from signal changes if no trade log
            pnl_vals = []
        else:
            pnl_vals = trades[pnl_col].to_list()

        if len(pnl_vals) > 0:
            wins = [p for p in pnl_vals if p > 0]
            losses = [p for p in pnl_vals if p <= 0]
            num_trades = len(pnl_vals)
            n_wins = len(wins)
            n_losses = len(losses)
            win_rate = n_wins / num_trades if num_trades > 0 else 0.0
            avg_win = float(np.mean(wins)) if n_wins > 0 else 0.0
            avg_loss = abs(float(np.mean(losses))) if n_losses > 0 else 0.0
            pl_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
            expectancy = (win_rate * avg_win - (1 - win_rate) * avg_loss) if avg_loss > 0 else 0.0
            total_win = sum(wins)
            total_loss = abs(sum(losses))
            profit_factor = total_win / total_loss if total_loss > 0 else float("inf")
    else:
        # Infer trades from position changes
        # A trade starts when position changes value (enter or flip).
        # Trade PnL = cumulative equity return over the holding period.
        pos = equity_curve["position"].to_list()
        trade_rets = []
        i = 1
        while i < len(pos):
            if pos[i] != pos[i - 1]:
                entry = i
                entry_signal = pos[i]
                i += 1
                while i < len(pos) and pos[i] == entry_signal:
                    i += 1
                exit_ = min(i, len(pos))
                if exit_ > entry:
                    pnl = (equity[exit_ - 1] - equity[entry - 1]) / equity[entry - 1]
                    trade_rets.append(pnl)
                continue
            i += 1
        if trade_rets:
            wins = [r for r in trade_rets if r > 0]
            losses = [r for r in trade_rets if r <= 0]
            num_trades = len(trade_rets)
            win_rate = len(wins) / num_trades if num_trades > 0 else 0.0
            avg_win = float(np.mean(wins)) * 100 if wins else 0.0
            avg_loss = abs(float(np.mean(losses))) * 100 if losses else 0.0
            pl_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

    # --- Information Coefficient (time-series rank correlation, pure numpy) ---
    ic_values = []
    df = equity_curve.with_columns(pl.col("returns").shift(-1).alias("fwd_return")).drop_nulls(
        ["signal", "fwd_return"]
    )

    sig_arr = df["signal"].to_numpy()
    fwd_arr = df["fwd_return"].to_numpy()
    if len(sig_arr) > 10:
        window = min(60, len(sig_arr) // 2)
        for i in range(window, len(sig_arr)):
            try:
                _s = sig_arr[i - window : i]
                _f = fwd_arr[i - window : i]
                if np.nanstd(_s) == 0 or np.nanstd(_f) == 0:
                    continue
                s_rank = np.argsort(np.argsort(_s)).astype(np.float64)
                f_rank = np.argsort(np.argsort(_f)).astype(np.float64)
                s_rank -= s_rank.mean()
                f_rank -= f_rank.mean()
                rho = (s_rank * f_rank).sum() / math.sqrt(
                    (s_rank**2).sum() * (f_rank**2).sum() + 1e-10
                )
                ic_values.append(rho)
            except Exception:
                pass
    else:
        try:
            if np.nanstd(sig_arr) > 0 and np.nanstd(fwd_arr) > 0:
                _s = sig_arr
                _f = fwd_arr
                s_rank = np.argsort(np.argsort(_s)).astype(np.float64)
                f_rank = np.argsort(np.argsort(_f)).astype(np.float64)
                s_rank -= s_rank.mean()
                f_rank -= f_rank.mean()
                rho = (s_rank * f_rank).sum() / math.sqrt(
                    (s_rank**2).sum() * (f_rank**2).sum() + 1e-10
                )
                ic_values.append(rho)
        except Exception:
            pass

    ic_arr = np.array(ic_values) if ic_values else np.array([0.0])
    ic_mean = float(np.mean(ic_arr))
    ic_std = float(np.std(ic_arr)) if len(ic_arr) > 1 else 0.0
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
    ic_pos = float(np.sum(ic_arr > 0)) / len(ic_arr) * 100 if len(ic_arr) > 0 else 0.0

    # --- Factor-style metrics ---
    factor_ret = total_ret * 100

    return EvalMetrics(
        total_return_pct=round(total_ret * 100, 2),
        annual_return_pct=round(ann_ret * 100, 2),
        annual_volatility_pct=round(ann_vol * 100, 2),
        sharpe_ratio=round(sharpe, 2),
        sortino_ratio=round(sortino, 2),
        calmar_ratio=round(calmar, 2),
        max_drawdown_pct=round(max_dd * 100, 2),
        avg_drawdown_pct=round(avg_dd * 100, 2),
        drawdown_days=dd_days,
        num_trades=num_trades,
        win_rate_pct=round(win_rate * 100, 2),
        avg_win_pct=round(avg_win, 2),
        avg_loss_pct=round(avg_loss, 2),
        profit_loss_ratio=round(pl_ratio, 2),
        expectancy=round(expectancy, 4),
        profit_factor=round(profit_factor, 2),
        ic_mean=round(ic_mean, 4),
        ic_std=round(ic_std, 4),
        ic_ir=round(ic_ir, 2),
        ic_positive_pct=round(ic_pos, 1),
        factor_return_pct=round(factor_ret, 2),
    )
