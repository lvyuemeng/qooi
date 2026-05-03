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


def print_result(name: str, s) -> None:
    if not s.eval:
        return
    e = s.eval
    print(f"\n{'=' * 55}")
    print(f"  {name}")
    print(f"{'=' * 55}")
    print(f"  Return:     {e.total_return_pct:>8.2f}%   Sharpe:    {e.sharpe_ratio:.2f}")
    print(f"  Max DD:     {e.max_drawdown_pct:>8.2f}%   Sortino:   {e.sortino_ratio:.2f}")
    print(f"  Win Rate:   {e.win_rate_pct:>8.2f}%   PL Ratio:  {e.profit_loss_ratio:.2f}")
    print(f"  IC Mean:    {e.ic_mean:>8.4f}   IC IR:     {e.ic_ir:.2f}")
    print(f"  Trades:     {e.num_trades:>8d}")
    if s.result and s.result.walk_forward:
        print("\n  Walk-forward segments:")
        for seg in s.result.walk_forward:
            print(
                f"    {seg['segment']:8s}  Ret={seg['total_return_pct']:>7.2f}%  "
                f"Sharpe={seg['sharpe_ratio']:.2f}  DD={seg['max_drawdown_pct']:.1f}%"
            )


def main() -> None:
    print("Data: BTC-USDT 1D (365 days)")

    s1 = Pipeline().run("BTC-USDT", "1D", 365, 10_000, sma_cross_signal(10, 30), cost=COST)
    print_result("SMA(10,30)", s1)

    s2 = Pipeline().run("BTC-USDT", "1D", 365, 10_000, bollinger_signal(20, 2), cost=COST)
    print_result("BB(20,2)", s2)

    s3 = Pipeline().run(
        "BTC-USDT", "1D", 365, 10_000, bollinger_signal(20, 2), cost=COST, walk_forward=WF
    )
    print_result("BB(20,2) Walk-Forward", s3)

    print("\nDone.")


if __name__ == "__main__":
    main()
