"""Batch fetch & cache OHLCV data from OKX as local Parquet files.

Usage:
    uv run python scripts/cache_okx.py --refresh            # fetch & save
    uv run python scripts/cache_okx.py --list               # show cached
    uv run python scripts/cache_okx.py --clear              # delete all cache
"""

from __future__ import annotations

import argparse

from qooi.exchange.store import CacheStore


def main() -> None:
    parser = argparse.ArgumentParser(description="OKX OHLCV cache manager")
    parser.add_argument(
        "--refresh",
        nargs="*",
        default=None,
        help="Fetch and cache: pass symbols (default: BTC-USDT ETH-USDT SOL-USDT)",
    )
    parser.add_argument(
        "--bar", default="1D", help="Candle size: 1H, 4H, 1D (default: 1D)"
    )
    parser.add_argument(
        "--days", type=int, default=90, help="Lookback days (default: 90)"
    )
    parser.add_argument("--list", action="store_true", help="Show cached datasets")
    parser.add_argument("--clear", action="store_true", help="Delete all cache")
    args = parser.parse_args()

    cs = CacheStore()

    if args.list:
        cached = cs.list_cached()
        if not cached:
            print("No cached data.")
        else:
            print("Cached datasets:")
            for c in cached:
                print(f"  {c['inst_id']:20s}  {c['bar']:5s}  {c['size_kb']:>6s} KB")
        return

    if args.clear:
        n = cs.clear()
        print(f"Deleted {n} cached file(s).")
        return

    if args.refresh is not None:
        symbols = args.refresh or ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
        for sym in symbols:
            print(f"Fetching {sym} ({args.bar}, {args.days}d)...")
            df = cs.refresh(sym, bar=args.bar, days=args.days, overwrite=True)
            print(
                f"  → {len(df)} rows cached  ({df['timestamp'].min()} ~ {df['timestamp'].max()})"
            )

    print("Done.")


if __name__ == "__main__":
    main()
