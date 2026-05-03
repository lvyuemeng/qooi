"""Plot OHLCV + indicators from local Parquet cache.

Usage:
    uv run python scripts/plot_okx.py                    # BTC-USDT 1D default
    uv run python scripts/plot_okx.py ETH-USDT 4H       # ETH 4H
    uv run python scripts/plot_okx.py SOL-USDT 1D --rsi  # with RSI subplot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import polars as pl

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from qooi.exchange.indicator import add_indicators, rsi, sma
from qooi.exchange.store import CacheStore

plt.style.use("dark_background")


def plot_candles(df: pl.DataFrame, symbol: str, bar: str, show_rsi: bool) -> None:
    df = df.with_columns(pl.from_epoch(pl.col("timestamp"), time_unit="ms").alias("dt"))
    df = df.sort("timestamp")

    if show_rsi:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]}
        )
        ax = ax1
    else:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax2 = None

    # --- Candlestick-like plot (line for close, fill for range) ---
    times = df["dt"].to_list()
    close = df["close"].to_list()
    high = df["high"].to_list()
    low = df["low"].to_list()

    ax.fill_between(times, low, high, alpha=0.15, color="gray")
    up_mask = [i for i in range(1, len(close)) if close[i] >= close[i - 1]]
    dn_mask = [i for i in range(1, len(close)) if close[i] < close[i - 1]]

    # Green for up, red for down
    for i in up_mask:
        ax.plot(
            [times[i], times[i]],
            [low[i], high[i]],
            color="#00cc66",
            linewidth=0.5,
            alpha=0.5,
        )
    for i in dn_mask:
        ax.plot(
            [times[i], times[i]],
            [low[i], high[i]],
            color="#ff3355",
            linewidth=0.5,
            alpha=0.5,
        )

    ax.plot(times, close, color="white", linewidth=1.2, label=f"{symbol} Close")

    # --- Indicators ---
    df_ind = add_indicators(df)
    ax.plot(
        times,
        df_ind["sma_20"].to_list(),
        color="#ffaa00",
        linewidth=0.8,
        alpha=0.8,
        label="SMA 20",
    )
    ax.plot(
        times,
        df_ind["sma_50"].to_list(),
        color="#00aaff",
        linewidth=0.8,
        alpha=0.8,
        label="SMA 50",
    )

    if "vol" in df.columns:
        vol = df["vol"].to_list()
        v_max = max(vol) if vol else 1
        v_scaled = [v / v_max * (max(high) - min(low)) * 0.15 for v in vol]
        ax.bar(
            times,
            v_scaled,
            bottom=min(low),
            width=0.6,
            color="#4444ff",
            alpha=0.25,
            label="Volume",
        )

    ax.set_title(f"{symbol}  ({bar})", fontsize=14, fontweight="bold")
    ax.set_ylabel("Price (USDT)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.15)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    # --- RSI subplot ---
    if show_rsi and ax2:
        rsi_vals = rsi(df).to_list()
        ax2.plot(times, rsi_vals, color="#cc66ff", linewidth=1)
        ax2.axhline(70, color="red", linestyle="--", alpha=0.4, linewidth=0.7)
        ax2.axhline(30, color="green", linestyle="--", alpha=0.4, linewidth=0.7)
        ax2.fill_between(times, 70, 100, alpha=0.08, color="red")
        ax2.fill_between(times, 0, 30, alpha=0.08, color="green")
        ax2.set_ylabel("RSI 14")
        ax2.set_ylim(0, 100)
        ax2.grid(alpha=0.15)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    plt.tight_layout()
    out_dir = Path("data/charts")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{symbol}_{bar}.png"
    fig.savefig(path, dpi=150)
    print(f"Saved chart: {path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot OHLCV + indicators")
    parser.add_argument(
        "symbol", nargs="?", default="BTC-USDT", help="Symbol (default: BTC-USDT)"
    )
    parser.add_argument(
        "bar", nargs="?", default="1D", help="Candle size (default: 1D)"
    )
    parser.add_argument("--rsi", action="store_true", help="Show RSI subplot")
    args = parser.parse_args()

    cs = CacheStore()
    try:
        df = cs.load(args.symbol, bar=args.bar)
    except FileNotFoundError:
        print(f"Cache miss. Fetching {args.symbol} ({args.bar})...")
        df = cs.refresh(args.symbol, bar=args.bar, days=90, overwrite=True)

    print(f"Plotting {args.symbol} ({args.bar}): {len(df)} rows")
    plot_candles(df, args.symbol, args.bar, show_rsi=args.rsi)


if __name__ == "__main__":
    main()
