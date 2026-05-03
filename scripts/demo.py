"""Demo: SMA + Bollinger strategies with realistic costs + walk-forward.

Usage:
    uv run python scripts/demo.py
"""

from qooi.exchange.backtest import CostModel, WalkForwardConfig
from qooi.exchange.pipeline import Pipeline
from qooi.strategies import bollinger_signal, sma_cross_signal

COST = CostModel(
    slippage_pct=0.001, spread_pct=0.0005, commission_pct=0.0005, short_borrow_rate=0.0001
)
WF = WalkForwardConfig(train_windows=6, test_window=2, holdout_window=1, step=1, rebalance_bars=30)


def main() -> None:
    print("Data: BTC-USDT 1D (365 days)")

    s1 = Pipeline().run("BTC-USDT", "1D", 365, 10_000, sma_cross_signal(10, 30), cost=COST)
    print(f"\n== SMA(10,30) =={s1.eval}")

    s2 = Pipeline().run("BTC-USDT", "1D", 365, 10_000, bollinger_signal(20, 2), cost=COST)
    print(f"\n== BB(20,2) =={s2.eval}")

    s3 = Pipeline().run(
        "BTC-USDT", "1D", 365, 10_000, bollinger_signal(20, 2), cost=COST, walk_forward=WF
    )
    print(f"\n== BB(20,2) Walk-Forward =={s3.eval}")

    if s3.result and s3.result.walk_forward:
        print("\n  Walk-forward segments:")
        for seg in s3.result.walk_forward:
            print(
                f"    Ret={seg.total_return_pct:>7.2f}%  "
                f"Sharpe={seg.sharpe_ratio:.2f}  DD={seg.max_drawdown_pct:.1f}%  "
                f"Trades={seg.num_trades}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
