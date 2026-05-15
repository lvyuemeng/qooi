"""Core evaluation metrics for equity/trade frames."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

import polars as pl


@dataclass
class EvalMetrics:
    total_return_pct: float
    annual_return_pct: float
    annual_volatility_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    avg_drawdown_pct: float
    drawdown_days: int
    num_trades: int
    win_rate_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_loss_ratio: float
    expectancy: float
    profit_factor: float
    ic_mean: float
    ic_std: float
    ic_ir: float
    ic_positive_pct: float
    factor_return_pct: float


def as_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def spearman_rho(x: pl.Series, y: pl.Series) -> float:
    if x.len() < 3:
        return 0.0
    rx = x.rank().to_numpy()
    ry = y.rank().to_numpy()
    rxm, rym = float(rx.mean()), float(ry.mean())
    d = rx - rxm
    e = ry - rym
    denom = math.sqrt((d * d).sum() * (e * e).sum())
    return float((d * e).sum()) / denom if denom > 1e-10 else 0.0


def infer_periods_per_year(equity_curve: pl.DataFrame, fallback: int = 365 * 24) -> int:
    """Infer annualization frequency from timestamp spacing."""
    if "timestamp" not in equity_curve.columns or equity_curve.height < 2:
        return fallback

    timestamps = equity_curve["timestamp"].to_list()
    deltas = [
        int(timestamps[i] - timestamps[i - 1])
        for i in range(1, len(timestamps))
        if (
            timestamps[i] is not None
            and timestamps[i - 1] is not None
            and timestamps[i] > timestamps[i - 1]
        )
    ]
    if not deltas:
        return fallback

    median_delta_ms = statistics.median(deltas)
    if median_delta_ms <= 0:
        return fallback
    return max(1, int(round((86_400_000.0 / median_delta_ms) * 365.0)))


def compute_metrics(
    equity_curve: pl.DataFrame,
    trades: pl.DataFrame | None = None,
    risk_free_rate: float = 0.02,
    periods_per_year: int | None = None,
) -> EvalMetrics:
    has_real_trades = trades is not None and not trades.is_empty() and "pnl" in trades.columns
    eq = equity_curve["portfolio_value"]
    rets = equity_curve["returns"]
    n = rets.len()
    effective_periods_per_year = periods_per_year or infer_periods_per_year(equity_curve)

    lv = as_float(eq.last())
    fv = as_float(eq.first(), 1.0)
    total_ret = lv / fv - 1 if fv != 0 else 0.0
    ann_factor = effective_periods_per_year / n if n > 0 else 1.0
    if total_ret > -1:
        ann_log_return = math.log1p(total_ret) * ann_factor
        ann_ret = math.expm1(min(ann_log_return, 700.0))
    else:
        ann_ret = -1.0

    std_val = rets.std()
    std = as_float(std_val) if n > 1 else 0.0
    ann_vol = std * math.sqrt(effective_periods_per_year) if std > 0 else 0.0
    excess = ann_ret - risk_free_rate
    sharpe = excess / ann_vol if ann_vol > 0 else 0.0

    neg_rets = rets.filter(rets < 0)
    ds = neg_rets.std()
    downside = as_float(ds) if neg_rets.len() > 1 else 0.0
    ann_downside = downside * math.sqrt(effective_periods_per_year)
    sortino = excess / ann_downside if ann_downside > 0 else 0.0

    peak = eq.cum_max()
    dd = (peak - eq) / peak
    md = dd.max()
    ad = dd.mean()
    assert md is not None and ad is not None
    max_dd = as_float(md)
    avg_dd = as_float(ad)
    calmar = ann_ret / max_dd if max_dd > 0 else 0.0

    dd_days = 0
    for v in dd:
        if v > 0.01:
            dd_days += 1
        else:
            dd_days = 0

    if has_real_trades:
        assert trades is not None
        pnl = trades["pnl"]
    elif "position" in equity_curve.columns:
        pos = equity_curve["position"]
        entry_equity: list[float] = []
        i = 1
        pos_arr = pos.to_list()
        eq_arr = eq.to_list()
        while i < len(pos_arr):
            if pos_arr[i] != pos_arr[i - 1]:
                j = i + 1
                while j < len(pos_arr) and pos_arr[j] == pos_arr[i]:
                    j += 1
                if j > i:
                    entry_equity.append((eq_arr[j - 1] - eq_arr[i - 1]) / eq_arr[i - 1])
                i = j
                continue
            i += 1
        pnl = pl.Series(entry_equity) if entry_equity else pl.Series([], dtype=pl.Float64)
    else:
        pnl = pl.Series([], dtype=pl.Float64)

    win_pnl = pnl.filter(pnl > 0)
    loss_pnl = pnl.filter(pnl <= 0)
    num_trades = pnl.len()
    nw = win_pnl.len()
    nl = loss_pnl.len()
    win_rate = nw / num_trades if num_trades > 0 else 0.0
    avg_win = as_float(win_pnl.mean()) * 100 if nw > 0 else 0.0
    avg_loss = abs(as_float(loss_pnl.mean())) * 100 if nl > 0 else 0.0
    pl_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
    total_win = float(win_pnl.sum())
    total_loss = abs(float(loss_pnl.sum()))
    profit_factor = total_win / total_loss if total_loss > 0 else float("inf")
    expectancy = (win_rate * avg_win - (1 - win_rate) * avg_loss) if avg_loss > 0 else 0.0

    has_signal = "signal" in equity_curve.columns and equity_curve["signal"].n_unique() > 1
    ic_values: list[float] = []
    if has_signal:
        ic_df = equity_curve.with_columns(rets.shift(-1).alias("fwd_return")).drop_nulls(
            ["signal", "fwd_return"]
        )
    else:
        ic_df = pl.DataFrame()
    if ic_df.height > 10:
        window = min(60, ic_df.height // 2)
        sig_s = ic_df["signal"]
        fwd_s = ic_df["fwd_return"]
        for i in range(window, ic_df.height):
            sw = sig_s.slice(i - window, window)
            fw = fwd_s.slice(i - window, window)
            if sw.std() == 0 or fw.std() == 0:
                continue
            ic_values.append(spearman_rho(sw, fw))
    elif ic_df.height > 3:
        rho = spearman_rho(ic_df["signal"], ic_df["fwd_return"])
        if rho != 0.0:
            ic_values.append(rho)

    ic_s = pl.Series(ic_values) if ic_values else pl.Series([0.0])
    ic_mean = as_float(ic_s.mean())
    ic_std = as_float(ic_s.std()) if ic_s.len() > 1 else 0.0
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
    ic_pos = float((ic_s > 0).sum()) / ic_s.len() * 100 if ic_s.len() > 0 else 0.0

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
        factor_return_pct=round(total_ret * 100, 2),
    )
