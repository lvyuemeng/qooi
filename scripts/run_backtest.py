"""Run backtest on cached OHLCV data, print metrics & save equity curve chart.

Usage:
    uv run python scripts/run_backtest.py                        # BTC-USDT 1D SMA(10,30)
    uv run python scripts/run_backtest.py ETH-USDT 4H --fast 5 --slow 20  # ETH 4H
    uv run python scripts/run_backtest.py SOL-USDT 1D --strategy ema      # EMA crossover
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from qooi.exchange.backtest import Backtest
from qooi.exchange.store import CacheStore
from qooi.strategies import sma_cross_signal, ema_cross_signal

plt.style.use("dark_background")


def plot_result(df: pl.DataFrame, symbol: str, bar: str, out: Path) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1, 1]}
    )

    times = pl.from_epoch(df["timestamp"], time_unit="ms").to_list()
    close = df["close"].to_list()
    equity = df["portfolio_value"].to_list()
    sig = df["signal"].to_list()
    rets = df["returns"].to_list()

    # --- Price + signals ---
    ax1.plot(times, close, color="white", linewidth=1, label="Close")
    long_mask = [i for i in range(len(sig)) if sig[i] == 1.0]
    short_mask = [i for i in range(len(sig)) if sig[i] == -1.0]
    if long_mask:
        ax1.scatter(
            [times[i] for i in long_mask],
            [close[i] for i in long_mask],
            color="#00cc66",
            s=8,
            alpha=0.5,
            label="Long",
        )
    if short_mask:
        ax1.scatter(
            [times[i] for i in short_mask],
            [close[i] for i in short_mask],
            color="#ff3355",
            s=8,
            alpha=0.5,
            label="Short",
        )
    ax1.set_title(
        f"{symbol}  ({bar}) — Backtest Results", fontsize=13, fontweight="bold"
    )
    ax1.set_ylabel("Price (USDT)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.15)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    # --- Equity curve ---
    ax2.plot(times, equity, color="#00aaff", linewidth=1.2)
    ax2.fill_between(times, equity[0], equity, alpha=0.1, color="#00aaff")
    ax2.set_ylabel("Equity (USDT)")
    ax2.grid(alpha=0.15)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    # --- Daily returns ---
    ax3.bar(times, rets, width=0.6, color="#8888ff", alpha=0.5)
    ax3.axhline(0, color="gray", linewidth=0.5)
    ax3.set_ylabel("Daily Return")
    ax3.set_xlabel("Date")
    ax3.grid(alpha=0.15)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    plt.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"  Chart saved: {out}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backtest on cached OHLCV data")
    parser.add_argument("symbol", nargs="?", default="BTC-USDT", help="Symbol")
    parser.add_argument("bar", nargs="?", default="1D", help="Candle size")
    parser.add_argument("--fast", type=int, default=10, help="Fast MA period")
    parser.add_argument("--slow", type=int, default=30, help="Slow MA period")
    parser.add_argument(
        "--strategy", default="sma", choices=["sma", "ema"], help="Strategy"
    )
    parser.add_argument("--capital", type=float, default=10_000, help="Initial capital")
    parser.add_argument(
        "--days", type=int, default=90, help="Lookback days if cache miss"
    )
    args = parser.parse_args()

    # --- Load data ---
    cs = CacheStore()
    try:
        df = cs.load(args.symbol, bar=args.bar)
    except FileNotFoundError:
        print(f"Cache miss — fetching {args.symbol} ({args.bar})...")
        df = cs.refresh(args.symbol, bar=args.bar, days=args.days, overwrite=True)

    # --- Run backtest ---
    signal_expr = (
        sma_cross_signal(args.fast, args.slow)
        if args.strategy == "sma"
        else ema_cross_signal(args.fast, args.slow)
    )

    bt = Backtest(df, signal_expr=signal_expr, initial_capital=args.capital)
    result = bt.run()

    # --- Print metrics ---
    m = result.metrics
    print(f"\n{'=' * 50}")
    print(
        f"  {args.symbol}  ({args.bar})  —  {args.strategy.upper()}({args.fast},{args.slow})"
    )
    print(f"{'=' * 50}")
    print(f"  Total return:    {m['total_return_pct']:>8.2f} %")
    print(f"  Sharpe ratio:    {m['sharpe_ratio']:>8.2f}")
    print(f"  Max drawdown:    {m['max_drawdown_pct']:>8.2f} %")
    print(f"  Number of trades: {m['num_trades']:>8d}")
    print(f"  Initial capital: ${m['initial_capital']:>8,.0f}")
    if m["final_value"] > 0:
        print(f"  Final value:     ${m['final_value']:>8,.0f}")
    print(f"{'=' * 50}\n")

    # --- Plot ---
    out_dir = Path("data/charts")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"backtest_{args.symbol}_{args.bar}_{args.strategy}.png"
    plot_result(result.equity_curve, args.symbol, args.bar, out)


if __name__ == "__main__":
    main()
