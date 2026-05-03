"""Vectorized backtester — dynamic position sizing, leverage, risk management."""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from qooi.exchange.eval import EvalMetrics, compute_metrics

SignalExpr = pl.Expr


@dataclass
class BacktestResult:
    trades: pl.DataFrame
    equity_curve: pl.DataFrame
    metrics: EvalMetrics
    walk_forward: list[EvalMetrics] | None = None


@dataclass
class CostModel:
    slippage_pct: float = 0.001
    spread_pct: float = 0.0005
    commission_pct: float = 0.001
    market_impact_pct: float = 0.000
    short_borrow_rate: float = 0.0001

    @property
    def total_per_side(self) -> float:
        return self.slippage_pct + self.spread_pct + self.commission_pct


@dataclass
class RiskConfig:
    """Dynamic position sizing and risk management.

    Signal interpretation:
      - Strategy outputs ``target_position`` in range [-max_leverage, +max_leverage].
      - +1.0 = 1x long, +2.0 = 2x leveraged long, -0.5 = 0.5x short, 0 = flat.

    Position sizing:
      ``position_sizing = "fixed"``: target_position is used directly (clipped).
      ``position_sizing = "atr"``:  target_position scaled so that
        ``size = max_risk_pct * capital / (atr_value * atr_stop_mult)``.

    Risk controls:
      - Stop-loss set at entry: ``entry_price ± atr_stop_mult * ATR``.
      - Take-profit: ``entry_price ± atr_target_mult * ATR``.
      - Trailing stop activates when price moves ``trailing_activation_mult * ATR``
        in profit, then follows at ``trailing_distance_mult * ATR`` behind.
    """

    max_leverage: float = 1.0
    position_sizing: str = "fixed"  # "fixed" | "atr"
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


@dataclass
class Backtest:
    data: pl.DataFrame
    signal_expr: SignalExpr
    initial_capital: float = 10_000.0
    cost: CostModel = field(default_factory=CostModel)
    risk: RiskConfig = field(default_factory=RiskConfig)
    walk: WalkForwardConfig | None = None

    def run(self) -> BacktestResult:
        df = self.data.sort("timestamp").with_columns(self.signal_expr.alias("signal"))
        df = df.with_columns(pl.col("signal").forward_fill().fill_null(0.0))
        if self.walk is not None:
            return self._run_walk_forward(df)
        return self._run_single(df)

    # ------------------------------------------------------------------
    # Single backtest
    # ------------------------------------------------------------------

    def _run_single(self, df: pl.DataFrame) -> BacktestResult:
        n = len(df)
        close = df["close"].to_list()
        signal = df["signal"].to_list()
        atr = df[self.risk.atr_col].to_list() if self.risk.atr_col in df.columns else [1.0] * n
        r = self.risk

        equity = [self.initial_capital]
        trade_log: list[dict] = []
        pos = [0.0]
        active: dict[float, dict] = {}
        stop_price = -1.0
        target_price = -1.0
        trailing_high = -1.0
        trailing_low = -1.0

        for i in range(1, n):
            p_prev = pos[-1]
            p_raw = signal[i - 1]
            prev_eq = equity[-1]
            a_prev = atr[i - 1] if atr[i - 1] is not None else 0.0
            a_curr = atr[i] if atr[i] is not None else 0.0
            atr_i = a_prev if a_prev > 0 else a_curr if a_curr > 0 else 1.0

            # --- Clip signal to max leverage ---
            p = max(-r.max_leverage, min(r.max_leverage, p_raw)) if r.max_leverage > 0 else 0.0

            # --- Check stop / target / trailing ---
            stopped_out = False
            if p_prev != 0 and stop_price > 0:
                if p_prev > 0:
                    if close[i - 1] <= stop_price:
                        p = 0.0
                        stopped_out = True
                    elif target_price > 0 and close[i - 1] >= target_price:
                        p = 0.0
                        stopped_out = True
                    else:
                        if trailing_high > 0:
                            trailing_high = max(trailing_high, close[i - 1])
                            if trailing_high - close[i - 1] >= r.trailing_distance_mult * atr_i:
                                p = 0.0
                                stopped_out = True
                else:
                    if close[i - 1] >= stop_price:
                        p = 0.0
                        stopped_out = True
                    elif target_price > 0 and close[i - 1] <= target_price:
                        p = 0.0
                        stopped_out = True
                    else:
                        if trailing_low > 0:
                            trailing_low = min(trailing_low, close[i - 1])
                            if close[i - 1] - trailing_low >= r.trailing_distance_mult * atr_i:
                                p = 0.0
                                stopped_out = True

            # --- Close position if stopped or signal changed ---
            if p == 0 and p_prev != 0:
                for sig_val, t in list(active.items()):
                    exit_price = close[i - 1]
                    if sig_val > 0:
                        exit_price *= 1.0 - self.cost.total_per_side
                    else:
                        exit_price *= 1.0 + self.cost.total_per_side
                    pnl = (exit_price / t["entry_price"] - 1) * t["size"] * t["entry_equity"]
                    reason = "stop" if stopped_out else "signal"
                    trade_log.append(
                        {
                            "entry_time": t["entry_ts"],
                            "exit_time": df["timestamp"][i - 1],
                            "side": "long" if sig_val > 0 else "short",
                            "entry_price": t["entry_price"],
                            "exit_price": exit_price,
                            "pnl": pnl,
                            "reason": reason,
                        }
                    )
                    del active[sig_val]
                stop_price = -1.0
                target_price = -1.0
                trailing_high = -1.0
                trailing_low = -1.0

            active = {}

            # --- Enter new position ---
            if abs(p) > 0 and p != p_prev:
                entry_price = close[i - 1]
                impact = self.cost.market_impact_pct * abs(p - p_prev)

                # Compute position size
                if r.position_sizing == "atr" and atr_i > 0:
                    base_size = r.max_risk_pct * prev_eq / (atr_i * r.atr_stop_mult)
                    size = min(abs(p), r.max_leverage) * base_size / prev_eq
                else:
                    size = abs(p) / r.max_leverage if r.max_leverage > 0 else 1.0

                if p > 0:
                    entry_price *= 1.0 + self.cost.total_per_side + impact
                    stop_price = entry_price - r.atr_stop_mult * atr_i
                    target_price = entry_price + r.atr_target_mult * atr_i
                    trailing_high = entry_price
                else:
                    entry_price *= 1.0 - self.cost.total_per_side - impact
                    stop_price = entry_price + r.atr_stop_mult * atr_i
                    target_price = entry_price - r.atr_target_mult * atr_i
                    trailing_low = entry_price

                size = max(size, 0.0)
                if size > 0:
                    active[p] = {
                        "entry_ts": df["timestamp"][i - 1],
                        "entry_price": entry_price,
                        "entry_equity": prev_eq,
                        "size": size,
                    }

            # --- Equity update ---
            pos_current = p
            daily_ret = 0.0
            if pos_current != 0:
                raw_ret = close[i] / close[i - 1] - 1
                daily_ret = raw_ret * pos_current
                if pos_current < 0:
                    daily_ret -= self.cost.short_borrow_rate

            equity.append(prev_eq * (1.0 + daily_ret))
            pos.append(pos_current)

        # Close any open trades at end
        if active:
            for sig_val, t in list(active.items()):
                pnl = (equity[-1] / t["entry_equity"] - 1) * t["size"] * t["entry_equity"]
                trade_log.append(
                    {
                        "entry_time": t["entry_ts"],
                        "exit_time": df["timestamp"][-1],
                        "side": "long" if sig_val > 0 else "short",
                        "entry_price": t["entry_price"],
                        "exit_price": close[-1],
                        "pnl": pnl,
                        "reason": "end",
                    }
                )

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
    # Walk-forward
    # ------------------------------------------------------------------

    def _run_walk_forward(self, df: pl.DataFrame) -> BacktestResult:
        assert self.walk is not None
        w = self.walk
        window_total = w.train_windows + w.test_window + w.holdout_window
        total_bars = len(df)
        window_bars = w.rebalance_bars
        n_windows = total_bars // window_bars
        if n_windows < window_total + 1:
            return self._run_single(df)
        walk_results: list[EvalMetrics] = []
        for start_win in range(0, n_windows - window_total + 1, w.step):
            train_end = (start_win + w.train_windows) * window_bars
            test_end = (start_win + w.train_windows + w.test_window) * window_bars
            holdout_end = (start_win + window_total) * window_bars
            for _label, (lo, hi) in [
                ("train", (start_win * window_bars, train_end)),
                ("test", (train_end, test_end)),
                ("holdout", (test_end, holdout_end)),
            ]:
                seg = df.slice(lo, hi - lo)
                if seg.height < 2:
                    continue
                result = self._run_single(seg)
                walk_results.append(compute_metrics(result.equity_curve, trades=result.trades))
        last_result = self._run_single(df)
        return BacktestResult(
            trades=last_result.trades,
            equity_curve=last_result.equity_curve,
            metrics=last_result.metrics,
            walk_forward=walk_results,
        )
