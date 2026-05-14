"""Live trading entry point — GitHub Actions single invocation.

Four-layer architecture: Signal → Basket → Recovery → Exits → Executor.
Same pipeline used by backtest and live trading.

State management: queries OKX for hard position/order truth and uses JSON
only for soft strategy state that OKX does not know.

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
from qooi.core.state import OkxStateProvider
from qooi.strategies import compute_signal_frame

DEFAULT_STRATEGY = os.environ.get("QOOI_STRATEGY", "momentum_burst")


def _run(dry_run: bool, env: str, strategy: str = DEFAULT_STRATEGY) -> None:
    os.environ["OKX_ENV"] = env
    from qooi.exchange.market import MarketData
    from qooi.exchange.trading import TradingClient, load_okx_env

    load_okx_env()
    tc = TradingClient()
    md = MarketData("okx")
    executor = LiveExecutor(tc, md)
    state = OkxStateProvider(tc)
    baskets = state.load(PAIRS, strategy_id=strategy)

    for p in PAIRS:
        sym = p.asset.symbol

        bot = tc.signal_ensure_bot(p, label=strategy)
        if not bot:
            print(f"  {sym:20s}  skip (failed to ensure bot)")
            continue

        df = md.candles(p.asset.sig_symbol, timeframe=p.asset.timeframe, limit=500, cache=True)
        if df.is_empty():
            print(f"  {sym:20s}  skip (no data)")
            continue
        df = compute_signal_frame(df, strategy)
        signal = float(df["signal"][-1]) if "signal" in df.columns else 0.0

        actions = process_bar(df, baskets, p, signal_src=signal, strategy_id=strategy)

        for a in actions:
            print(f"  {sym:20s} {a.action:10s} {a.side:5s} {a.reason}")

        executor.execute(actions, dry_run=dry_run)

    state.save_soft(baskets)


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    if cmd in ("test", "live"):
        dry = cmd == "live" and (len(sys.argv) <= 2 or sys.argv[2] != "live")
        strategy = os.environ.get("QOOI_STRATEGY", DEFAULT_STRATEGY)
        _run(dry_run=dry, env=cmd, strategy=strategy)
    else:
        print("Usage: uv run python scripts/trade.py test|live [dry]")
        sys.exit(1)


if __name__ == "__main__":
    main()
