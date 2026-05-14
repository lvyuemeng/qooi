"""Live trading entry point — GitHub Actions single invocation.

Four-layer architecture: Signal → Basket → Recovery → Exits → Executor.
Same pipeline used by backtest and live trading.

State management: loads soft accumulators from data/state/baskets.json
on startup, queries OKX for hard position/order truth, saves state after
execution.

Usage::

    uv run python scripts/trade.py test
    uv run python scripts/trade.py live [dry]
"""

from __future__ import annotations

import os
import sys

from qooi.core import process_bar
from qooi.core.config import PAIRS
from qooi.core.executor import LiveExecutor


def _run(dry_run: bool, env: str) -> None:
    os.environ["OKX_ENV"] = env
    from qooi.exchange.market import MarketData
    from qooi.exchange.trading import TradingClient, load_okx_env

    load_okx_env()
    tc = TradingClient()
    md = MarketData("okx")
    executor = LiveExecutor(tc, md)
    baskets = executor.load_state(PAIRS)

    for p in PAIRS:
        sym = p.asset.symbol

        bot = tc.signal_ensure_bot(p)
        if not bot:
            print(f"  {sym:20s}  skip (failed to ensure bot)")
            continue

        from qooi.exchange.indicator import add_indicators

        df = md.candles(p.asset.sig_symbol, timeframe=p.asset.timeframe, limit=500, cache=True)
        if df.is_empty():
            print(f"  {sym:20s}  skip (no data)")
            continue
        df = add_indicators(df)

        actions = process_bar(df, baskets, p)

        for a in actions:
            print(f"  {sym:20s} {a.action:10s} {a.side:5s} {a.reason}")

        executor.execute(actions, dry_run=dry_run)

    executor.save_state(baskets)


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    if cmd in ("test", "live"):
        dry = cmd == "live" and (len(sys.argv) <= 2 or sys.argv[2] != "live")
        _run(dry_run=dry, env=cmd)
    else:
        print("Usage: uv run python scripts/trade.py test|live [dry]")
        sys.exit(1)


if __name__ == "__main__":
    main()
