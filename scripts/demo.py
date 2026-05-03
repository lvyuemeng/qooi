"""One demo showing the entire OKX pipeline — cache, indicators, backtest, plot.

Usage:
    uv run python scripts/demo.py
"""

from qooi.exchange.pipeline import Pipeline
from qooi.strategies import sma_cross_signal, ema_cross_signal
from qooi.exchange.market import MarketData
from qooi.exchange.store import CacheStore


def demo_pipeline() -> None:
    print("=== Pipeline: BTC-USDT 1D SMA(10,30) ===")
    s = Pipeline().run(
        symbol="BTC-USDT",
        bar="1D",
        days=90,
        capital=10_000,
        signal_expr=sma_cross_signal(10, 30),
        plot_out="data/charts/BTC_SMA10_30.png",
    )
    m = s.result.metrics
    print(
        f"  Return: {m['total_return_pct']}%  Sharpe: {m['sharpe_ratio']}  DD: {m['max_drawdown_pct']}%"
    )
    print(f"  Chart: {s.chart_path}")

    print("\n=== Pipeline: ETH-USDT 1D EMA(12,26) ===")
    s2 = Pipeline().run(
        symbol="ETH-USDT",
        bar="1D",
        days=90,
        capital=10_000,
        signal_expr=ema_cross_signal(12, 26),
        plot_out="data/charts/ETH_EMA12_26.png",
    )
    m2 = s2.result.metrics
    print(
        f"  Return: {m2['total_return_pct']}%  Sharpe: {m2['sharpe_ratio']}  DD: {m2['max_drawdown_pct']}%"
    )
    print(f"  Chart: {s2.chart_path}")


def demo_cache() -> None:
    print("\n=== Cached datasets ===")
    cs = CacheStore()
    for c in cs.list_cached():
        print(f"  {c['inst_id']:20s}  {c['bar']:5s}  {c['size_kb']:>6s} KB")


def demo_ticker() -> None:
    print("\n=== Live ticker ===")
    md = MarketData()
    t = md.ticker("BTC-USDT")
    print(f"  BTC-USDT last: ${t['last'][0]}")
    t2 = md.ticker("ETH-USDT")
    print(f"  ETH-USDT last: ${t2['last'][0]}")


if __name__ == "__main__":
    demo_ticker()
    demo_cache()
    demo_pipeline()
