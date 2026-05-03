"""Vectorized backtester for OHLCV data — built on Polars.

Produces a ``BacktestResult`` with:
  - trades: each filled trade
  - equity curve: daily portfolio value
  - metrics: sharpe, max drawdown, total return, etc.

Usage::

    result = Backtest(df, strategy).run()
    print(result.metrics)
    result.equity_curve.plot()
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass
class BacktestResult:
    trades: pl.DataFrame
    equity_curve: pl.DataFrame
    metrics: dict[str, float | str]


SignalExpr = pl.Expr


@dataclass
class Backtest:
    """Vectorized backtest engine.

    Parameters
    ----------
    data:
        OHLCV DataFrame with at least columns: timestamp, close.
        Must be sorted by timestamp ascending.
    signal_expr:
        Polars expression that evaluates to +1 (long), -1 (short), or 0 (flat).
        Applied within a ``with_columns`` context so it has access to all columns.
    initial_capital:
        Starting portfolio value in quote currency.
    commission_pct:
        Fee per trade as a fraction (e.g. 0.001 = 0.1%).
    slippage_pct:
        Slippage per trade as a fraction.
    """

    data: pl.DataFrame
    signal_expr: SignalExpr
    initial_capital: float = 10_000.0
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005

    def run(self) -> BacktestResult:
        df = self.data.sort("timestamp").with_columns(
            [
                pl.lit(0.0).alias("position"),
                pl.lit(0.0).alias("portfolio_value"),
                pl.lit(0.0).alias("returns"),
                pl.lit(0.0).alias("trade_return"),
            ]
        )

        # --- Generate signals ---
        df = df.with_columns(self.signal_expr.alias("signal"))

        # Forward-fill NaN signals (hold previous position)
        df = df.with_columns(pl.col("signal").forward_fill().fill_null(0.0))

        # --- Positions & trades ---
        df = df.with_columns(pl.col("signal").shift(1).fill_null(0.0).alias("position"))

        # Entry / exit events
        df = df.with_columns(
            (pl.col("position") != pl.col("position").shift(1).fill_null(0.0)).alias(
                "signal_change"
            )
        )

        # --- Portfolio simulation ---
        close = df["close"].to_list()
        pos = df["position"].to_list()
        n = len(close)

        equity = [self.initial_capital]
        trade_log: list[dict] = []
        active_trades: dict[float, dict] = {}  # signal_value -> trade_info

        for i in range(1, n):
            prev_equity = equity[-1]
            p = pos[i]
            p_prev = pos[i - 1]

            if p != p_prev:
                # Close existing trades
                for sig_val, t in list(active_trades.items()):
                    pnl = prev_equity - t["entry_equity"]
                    trade_log.append(
                        {
                            "entry_time": t["entry_ts"],
                            "exit_time": df["timestamp"][i],
                            "side": "long" if sig_val > 0 else "short",
                            "size_pct": abs(sig_val),
                            "commission": t["commission"],
                            "pnl": pnl,
                        }
                    )
                    del active_trades[sig_val]

                # Open new trades
                if p != 0:
                    cost = self.commission_pct + self.slippage_pct
                    commission_cost = prev_equity * cost
                    equity[-1] = prev_equity - commission_cost
                    active_trades[p] = {
                        "entry_ts": df["timestamp"][i],
                        "entry_equity": equity[-1],
                        "commission": commission_cost,
                    }
                    prev_equity = equity[-1]

            ret = (close[i] / close[i - 1] - 1) * p
            equity.append(prev_equity * (1 + ret))

        # Close any remaining open trades at end
        if active_trades:
            for sig_val, t in list(active_trades.items()):
                pnl = equity[-1] - t["entry_equity"]
                trade_log.append(
                    {
                        "entry_time": t["entry_ts"],
                        "exit_time": df["timestamp"][-1],
                        "side": "long" if sig_val > 0 else "short",
                        "size_pct": abs(sig_val),
                        "commission": t["commission"],
                        "pnl": pnl,
                    }
                )

        df = df.with_columns(pl.Series(equity).alias("portfolio_value"))
        df = df.with_columns(pl.col("portfolio_value").pct_change().fill_null(0.0).alias("returns"))

        # --- Metrics ---
        rets = df["returns"].to_list()
        total_ret = (equity[-1] / self.initial_capital) - 1
        ann_factor = 365 if self._bar_is_daily(df) else (365 * 24 * 60 * 60) // self._bar_ms(df)

        if ann_factor > 1 and len(rets) > 1:
            avg_ret = sum(rets) / len(rets)
            std_ret = (sum((r - avg_ret) ** 2 for r in rets) / (len(rets) - 1)) ** 0.5
            sharpe = (avg_ret / std_ret * (ann_factor**0.5)) if std_ret > 0 else 0.0
        else:
            sharpe = 0.0

        peaks = [equity[0]]
        dd = [0.0]
        for v in equity[1:]:
            peaks.append(max(peaks[-1], v))
            dd.append((peaks[-1] - v) / peaks[-1] if peaks[-1] > 0 else 0.0)
        max_dd = max(dd) if dd else 0.0

        trade_df = pl.DataFrame(trade_log) if trade_log else pl.DataFrame()

        return BacktestResult(
            trades=trade_df,
            equity_curve=df.select(
                [
                    "timestamp",
                    "close",
                    "signal",
                    "position",
                    "portfolio_value",
                    "returns",
                ]
            ),
            metrics={
                "total_return_pct": round(total_ret * 100, 2),
                "sharpe_ratio": round(sharpe, 2),
                "max_drawdown_pct": round(max_dd * 100, 2),
                "num_trades": len(trade_log),
                "initial_capital": self.initial_capital,
                "final_value": round(equity[-1], 2),
            },
        )

    @staticmethod
    def _bar_is_daily(df: pl.DataFrame) -> bool:
        if len(df) < 2:
            return True
        diff = df["timestamp"][1] - df["timestamp"][0]
        return abs(diff - 86_400_000) < 1000

    @staticmethod
    def _bar_ms(df: pl.DataFrame) -> int:
        if len(df) < 2:
            return 86_400_000
        return df["timestamp"][1] - df["timestamp"][0]
