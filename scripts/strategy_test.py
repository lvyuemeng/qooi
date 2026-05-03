"""Run EMA50/200 + VuManChu Swing Free strategy on BTC 4H/1D.

Usage:
    uv run python scripts/strategy_test.py
    uv run python scripts/strategy_test.py ETH-USDT 4H
"""

from __future__ import annotations

import sys

from qooi.exchange.backtest import CostModel
from qooi.exchange.pipeline import Pipeline
from qooi.strategies import ema_vumanchu_signal


def main() -> None:
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTC-USDT"
    bar = sys.argv[2] if len(sys.argv) > 2 else "1D"
    days = 360

    cost = CostModel(slippage_pct=0.001, spread_pct=0.0005, commission_pct=0.0005)

    print(f"EMA50/200 + VuManChu Swing Free  |  {symbol}  {bar}  ({days}d)")
    print("─" * 55)

    s = Pipeline().run(
        symbol=symbol,
        bar=bar,
        days=days,
        capital=10_000,
        signal_expr=ema_vumanchu_signal(require_close_above_ema_long=True),
        cost=cost,
        plot_out=f"data/charts/vumanchu_{symbol}_{bar}.png",
    )

    print(s.eval)
    print(f"  Chart: {s.chart_path}")

    # Walk-forward for robustness check
    from qooi.exchange.backtest import WalkForwardConfig

    wf = WalkForwardConfig(
        train_windows=6, test_window=2, holdout_window=1, step=1, rebalance_bars=30
    )
    s2 = Pipeline().run(
        symbol=symbol,
        bar=bar,
        days=days,
        capital=10_000,
        signal_expr=ema_vumanchu_signal(require_close_above_ema_long=True),
        cost=cost,
        walk_forward=wf,
    )

    if s2.result and s2.result.walk_forward:
        print("\n  Walk-forward segments:")
        for seg in s2.result.walk_forward:
            print(
                f"    Ret={seg.total_return_pct:>7.2f}%  "
                f"Sharpe={seg.sharpe_ratio:.2f}  DD={seg.max_drawdown_pct:.1f}%  "
                f"Trades={seg.num_trades}"
            )


if __name__ == "__main__":
    main()
