"""Vectorized backtester — proper slippage, spread, market impact, walk-forward."""

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
    """Realistic execution costs.

    Parameters
    ----------
    slippage_pct:
        Fraction of price lost per side (e.g. 0.001 = 0.1%).
        Applied to fill price: buy at close*(1+slippage), sell at close*(1-slippage).
    spread_pct:
        Half of the bid-ask spread as a fraction (e.g. 0.0005 = 0.05% half-spread).
        Added to slippage on each trade.
    commission_pct:
        Broker/exchange fee per trade (e.g. 0.001 = 0.1%).
    market_impact_pct:
        Additional cost when position size changes are large relative to volume.
        Applied as: impact = position_change * impact_pct.
    short_borrow_rate:
        Daily borrow cost for short positions as a fraction (e.g. 0.0001 = 0.01%/day).
    """

    slippage_pct: float = 0.001
    spread_pct: float = 0.0005
    commission_pct: float = 0.001
    market_impact_pct: float = 0.000
    short_borrow_rate: float = 0.0001

    @property
    def total_per_side(self) -> float:
        """Combined one-way cost fraction (slippage + spread + commission)."""
        return self.slippage_pct + self.spread_pct + self.commission_pct


@dataclass
class WalkForwardConfig:
    """Walk-forward analysis configuration.

    The full data is split into three periods:
      - Train (first N windows): signal is developed/observed.
      - Test (next 1 window): out-of-sample performance.
      - Holdout (last 1 window): final unseen validation.

    Rolling: the window slides forward by ``step`` bars each iteration.
    """

    train_windows: int = 3  # number of rebalance windows for training
    test_window: int = 1  # number of rebalance windows for out-of-sample
    holdout_window: int = 1  # number of rebalance windows for final validation
    step: int = 1  # windows to slide forward each iteration
    rebalance_bars: int = 20  # number of bars per rebalance window


@dataclass
class Backtest:
    """Vectorized backtest engine with realistic costs and walk-forward.

    Usage::

        bt = Backtest(df, signal_expr=..., cost=CostModel())
        result = bt.run()
        print(result.metrics)
    """

    data: pl.DataFrame
    signal_expr: SignalExpr
    initial_capital: float = 10_000.0
    cost: CostModel = field(default_factory=CostModel)
    walk: WalkForwardConfig | None = None

    def run(self) -> BacktestResult:
        df = self.data.sort("timestamp").with_columns(self.signal_expr.alias("signal"))
        df = df.with_columns(pl.col("signal").forward_fill().fill_null(0.0))

        if self.walk:
            return self._run_walk_forward(df)
        return self._run_single(df, label="full")

    # ------------------------------------------------------------------
    # Single backtest
    # ------------------------------------------------------------------

    def _run_single(self, df: pl.DataFrame, label: str = "full") -> BacktestResult:
        n = len(df)
        close = df["close"].to_list()
        signal = df["signal"].to_list()

        equity = [self.initial_capital]
        trade_log: list[dict] = []
        active: dict[float, dict] = {}
        pos = [0.0]

        for i in range(1, n):
            p_prev = pos[-1]
            p = signal[i - 1]  # execute at today's signal (yesterday's decision)
            prev_eq = equity[-1]

            # --- Exit existing positions if signal changed ---
            if p != p_prev and p_prev != 0:
                for sig_val, t in list(active.items()):
                    exit_price = close[i - 1]
                    if sig_val > 0:
                        exit_price *= 1.0 - self.cost.total_per_side
                    else:
                        exit_price *= 1.0 + self.cost.total_per_side
                    pnl = (exit_price / t["entry_price"] - 1) * t["size"] * t["entry_equity"]
                    trade_log.append(
                        {
                            "entry_time": t["entry_ts"],
                            "exit_time": df["timestamp"][i - 1],
                            "side": "long" if sig_val > 0 else "short",
                            "entry_price": t["entry_price"],
                            "exit_price": exit_price,
                            "pnl": pnl,
                        }
                    )
                    del active[sig_val]

            # --- Enter new positions ---
            if p != p_prev and p != 0:
                entry_price = close[i - 1]
                impact = self.cost.market_impact_pct * abs(p - p_prev)
                if p > 0:
                    entry_price *= 1.0 + self.cost.total_per_side + impact
                else:
                    entry_price *= 1.0 - self.cost.total_per_side - impact
                active[p] = {
                    "entry_ts": df["timestamp"][i - 1],
                    "entry_price": entry_price,
                    "entry_equity": prev_eq,
                    "size": p,
                }

            # --- Daily return with costs ---
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
            return self._run_single(df, label="full (too short for walk-forward)")

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
                result = self._run_single(seg, label=_label)
                walk_results.append(compute_metrics(result.equity_curve, trades=result.trades))

        last_result = self._run_single(df, label="walk_forward")

        return BacktestResult(
            trades=last_result.trades,
            equity_curve=last_result.equity_curve,
            metrics=last_result.metrics,
            walk_forward=walk_results,
        )
