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
    atr_stop_mult: float = 2.0
    atr_target_mult: float = 3.0
    trailing_activation_mult: float = 2.0
    trailing_distance_mult: float = 1.0
    max_bars_held: int = 0  # 0 = hold indefinitely; N = exit after N bars in ACTIVE
    atr_col: str = "atr_14"
    ct_val: float = 1.0  # contract multiplier — 0.01 for BTC, 0.1 for ETH, 100 for XRP


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
    threshold: float | None = None  # per-asset threshold; None=use old 0.40/0.25/0.70
    ord_type: str = "limit"  # "limit" or "market"
    exit_mode: str = "signal_flip_only"
    # for stateful signals (ema_pullback*); "with_sl_tp" for flow_pipeline

    def run(self) -> BacktestResult:
        df = self.data.sort("timestamp").with_columns(self.signal_expr.alias("signal"))
        df = df.with_columns(pl.col("signal").forward_fill().fill_null(0.0))
        return self._run_shared(df)

    def _run_shared(self, df: pl.DataFrame) -> BacktestResult:
        """Backtest using legacy decide loop — OFI flow pipeline path."""
        from dataclasses import dataclass
        from enum import StrEnum

        from qooi.core.config import AssetConfig
        from qooi.core.indicators import SignalResult

        class _Action(StrEnum):
            ENTER = "enter"
            EXIT = "exit"
            HOLD = "hold"

        @dataclass
        class _Decision:
            action: _Action
            side: str = ""
            sz: int = 0
            entry_px: float = 0.0
            stop_px: float = 0.0
            target_px: float = 0.0
            detail: str = ""

        def _compute_sz(entry_px, stop_px, cfg):
            risk_per_ct = abs(entry_px - stop_px) * cfg.ct_val
            if risk_per_ct <= 0:
                return 0
            max_risk = cfg.capital * cfg.max_risk_pct
            sz = max(1, int(max_risk / risk_per_ct))
            notional_per_ct = cfg.ct_val * entry_px
            max_sz = int(cfg.capital * cfg.leverage / max(notional_per_ct, 1e-9))
            return max(1, min(sz, max_sz))

        def _compute_stop_target(side, entry_px, atr_val, cfg, regime_strength=0.0):
            d = 1 if side == "buy" else -1
            if regime_strength > 0.7:
                sm = cfg.atr_stop_mult * 0.5
                tm = cfg.atr_target_mult * 0.8
            elif regime_strength > 0.3:
                sm = cfg.atr_stop_mult * 0.75
                tm = cfg.atr_target_mult * 1.2
            else:
                sm = cfg.atr_stop_mult * 1.25
                tm = cfg.atr_target_mult * 0.6
            return (
                round(entry_px - d * sm * atr_val, 2),
                round(entry_px + d * tm * atr_val, 2),
            )

        def _decide_idle(signal, entry_px, side, cfg):
            if abs(signal.signal) < cfg.signal_threshold:
                return _Decision(action=_Action.HOLD, detail="weak_signal")
            if abs(signal.mom_fast) > 0.3 and signal.signal * signal.mom_fast < 0:
                return _Decision(action=_Action.HOLD, detail="momentum_opposing")
            if signal.vol_conf < 0.3:
                return _Decision(action=_Action.HOLD, detail="low_volume")
            sp, tp = _compute_stop_target(side, entry_px, signal.atr, cfg, signal.regime_strength)
            sz = _compute_sz(entry_px, sp, cfg)
            if sz < 1:
                return _Decision(action=_Action.HOLD, detail="insufficient_margin")
            return _Decision(
                action=_Action.ENTER,
                side=side,
                sz=sz,
                entry_px=round(entry_px, 2),
                stop_px=sp,
                target_px=tp,
            )

        def _decide_active(signal, pos_side, cfg, entry_px=0.0, mark_px=0.0, exit_mode=""):
            d = 1 if pos_side == "buy" else -1
            if signal.signal * d < 0:
                return _Decision(action=_Action.EXIT, side=pos_side, detail="signal_flipped")
            if exit_mode in ("with_sl_tp", "full") and entry_px > 0 and mark_px > 0:
                a = signal.atr if signal.atr > 0 else 50.0
                sm, tm = _compute_stop_target(pos_side, entry_px, a, cfg, signal.regime_strength)
                st = entry_px - d * sm * a
                tg = entry_px + d * tm * a
                if d * (st - mark_px) >= 0:
                    return _Decision(
                        action=_Action.EXIT, side=pos_side, detail="stop", stop_px=st, target_px=tg
                    )
                if d * (mark_px - tg) >= 0:
                    return _Decision(
                        action=_Action.EXIT,
                        side=pos_side,
                        detail="target",
                        stop_px=st,
                        target_px=tg,
                    )
            return _Decision(action=_Action.HOLD, detail="holding")

        n = len(df)
        close = df["close"].to_list()
        atr_col = self.risk.atr_col
        atr = df[atr_col].fill_nan(0).fill_null(0).to_list() if atr_col in df.columns else [1.0] * n
        atr = [v if v > 0 else 1.0 for v in atr]
        signal_col = df["signal"].to_list()
        regime_col = (
            df["regime_strength"].to_list() if "regime_strength" in df.columns else [0.0] * n
        )
        mom_col = df["regime_mom_fast"].to_list() if "regime_mom_fast" in df.columns else [0.0] * n
        vol_col = df["regime_vol_conf"].to_list() if "regime_vol_conf" in df.columns else [0.5] * n
        timestamp_col = df["timestamp"].to_list()

        cfg = AssetConfig(
            symbol="BACKTEST",
            sig_symbol="BACKTEST",
            timeframe="4h",
            capital=self.initial_capital,
            max_risk_pct=self.risk.max_risk_pct,
            leverage=self.risk.max_leverage,
            ct_val=self.risk.ct_val,
            signal_threshold=self.threshold or 0.25,
        )

        equity = [self.initial_capital]
        pos = [0.0]
        trades: list[dict] = []
        pos_side = ""
        entry_px = 0.0
        sz = 0
        entry_ts = 0

        for i in range(1, n):
            prev_eq = equity[-1]
            cur_close = close[i - 1]

            sig_val = signal_col[i - 1]
            sr = SignalResult(
                symbol="BACKTEST",
                timeframe="4h",
                timestamp=int(timestamp_col[i - 1]),
                signal=sig_val,
                flow=sig_val,
                threshold=cfg.signal_threshold,
                atr=atr[i - 1],
                regime_strength=regime_col[i - 1],
                mom_fast=mom_col[i - 1],
                vol_conf=vol_col[i - 1],
            )

            if not pos_side:
                side = "buy" if sig_val > 0 else "sell"
                d = _decide_idle(sr, cur_close, side, cfg)
                if d.action.value == "enter":
                    pos_side = d.side
                    entry_px = d.entry_px
                    sz = d.sz
                    entry_ts = int(timestamp_col[i - 1])
                    prev_eq *= 1.0 - self.cost.total_per_side
            else:
                d = _decide_active(
                    sr,
                    pos_side,
                    cfg,
                    entry_px=entry_px,
                    mark_px=cur_close,
                    exit_mode=self.exit_mode,
                )
                if d.action.value == "exit":
                    d_sign = 1 if pos_side == "buy" else -1
                    exit_px = cur_close * (1.0 - d_sign * self.cost.total_per_side)
                    pnl_pct = d_sign * (exit_px / entry_px - 1) if entry_px > 0 else 0.0
                    notional = sz * cfg.ct_val * entry_px
                    pnl_usd = pnl_pct * notional * cfg.leverage
                    trades.append(
                        {
                            "entry_time": entry_ts,
                            "exit_time": int(timestamp_col[i - 1]),
                            "side": "long" if d_sign > 0 else "short",
                            "entry_price": entry_px,
                            "exit_price": exit_px,
                            "pnl": pnl_usd,
                            "reason": d.detail,
                        }
                    )
                    prev_eq += pnl_usd
                    pos_side = ""
                    sz = 0

            pos_current = sz * (1 if pos_side == "buy" else -1) if pos_side else 0.0
            equity.append(prev_eq)
            pos.append(pos_current)

        eq_series = pl.Series(equity, dtype=pl.Float64)
        result_df = df.select(["timestamp", "close", "signal"]).with_columns(
            [
                pl.Series(pos).alias("position"),
                eq_series.alias("portfolio_value"),
                eq_series.pct_change().fill_null(0.0).alias("returns"),
            ]
        )
        return BacktestResult(
            trades=pl.DataFrame(trades) if trades else pl.DataFrame(),
            equity_curve=result_df,
            metrics=compute_metrics(
                result_df, trades=pl.DataFrame(trades) if trades else pl.DataFrame()
            ),
        )


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
            last = self._backtest._run_shared(df)
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
                result = self._backtest._run_shared(seg)
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
            last = self._backtest._run_shared(df)
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
