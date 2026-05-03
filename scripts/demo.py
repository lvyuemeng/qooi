"""Demo: run SMA, EMA, and Bollinger strategies with full evaluation.

Usage:
    uv run python scripts/demo.py
"""

from qooi.exchange.eval import EvalMetrics
from qooi.exchange.pipeline import Pipeline
from qooi.strategies import bollinger_signal, ema_cross_signal, sma_cross_signal


def print_eval(name: str, e: EvalMetrics) -> None:
    print(f"\n{'=' * 55}")
    print(f"  {name}")
    print(f"{'=' * 55}")
    print(
        f"  Return:         {e.total_return_pct:>8.2f}%    Ann. Return: {e.annual_return_pct:.2f}%"
    )
    print(
        f"  Ann. Volatility:{e.annual_volatility_pct:>8.2f}%    Sharpe:      {e.sharpe_ratio:.2f}"
    )
    print(f"  Sortino:        {e.sortino_ratio:>8.2f}    Calmar:      {e.calmar_ratio:.2f}")
    print(
        f"  Max DD:         {e.max_drawdown_pct:>8.2f}%    Avg DD:      {e.avg_drawdown_pct:.2f}%"
    )
    print(f"  DD Days:        {e.drawdown_days:>8d}")
    print(f"  Win Rate:       {e.win_rate_pct:>8.2f}%    PL Ratio:    {e.profit_loss_ratio:.2f}")
    print(f"  Profit Factor:  {e.profit_factor:>8.2f}    Expectancy:  {e.expectancy:.4f}")
    print(f"  IC Mean:        {e.ic_mean:>8.4f}    IC IR:       {e.ic_ir:.2f}")
    print(f"  IC Positive:    {e.ic_positive_pct:>8.1f}%    Trades:      {e.num_trades}")


def main() -> None:
    print("Loading/BTC-USDT 1D (90 days)...")

    print("\n--- SMA Crossover (10,30) ---")
    s = Pipeline().run(
        "BTC-USDT", "1D", days=90, capital=10_000, signal_expr=sma_cross_signal(10, 30)
    )
    print_eval("SMA(10,30)  BTC-USDT", s.eval)
    print(f"  Chart: {s.chart_path}")

    print("\n--- EMA Crossover (12,26) ---")
    s2 = Pipeline().run(
        "BTC-USDT", "1D", days=90, capital=10_000, signal_expr=ema_cross_signal(12, 26)
    )
    print_eval("EMA(12,26)  BTC-USDT", s2.eval)
    print(f"  Chart: {s2.chart_path}")

    print("\n--- Bollinger Mean-Reversion (20,2) ---")
    s3 = Pipeline().run(
        "BTC-USDT", "1D", days=90, capital=10_000, signal_expr=bollinger_signal(20, 2)
    )
    print_eval("BB(20,2)    BTC-USDT", s3.eval)
    print(f"  Chart: {s3.chart_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
