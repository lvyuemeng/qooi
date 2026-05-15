"""Live trading entry point — GitHub Actions single invocation.

Four-layer architecture: Signal → Basket → Recovery → Exits → Executor.
Same pipeline used by backtest and live trading.

State management: queries OKX for position/order truth. GitHub Actions does
not persist local state, so live state must be reconstructable from OKX.

Usage::

    uv run python scripts/trade.py test
    uv run python scripts/trade.py live [dry]
"""

from __future__ import annotations

import os
import sys

from qooi.core import BarMarket, BarSignal, PipelineContext, process_bar
from qooi.core.basket import BasketBook
from qooi.core.config import PAIRS
from qooi.core.executor import LiveExecutor
from qooi.core.state import OkxStateProvider
from qooi.strategies import (
    StrategySpec,
    compute_signal_frame,
    momentum_burst_spec,
    rsi_bounce_reversion_spec,
)

DEFAULT_STRATEGY = os.environ.get("QOOI_STRATEGY", "momentum_burst")


def _strategy_from_env(value: str) -> StrategySpec:
    if value == "momentum_burst":
        return momentum_burst_spec()
    if value == "rsi_bounce_reversion":
        return rsi_bounce_reversion_spec()
    raise SystemExit(f"Unknown strategy: {value}")


def _run(dry_run: bool, env: str, strategy: StrategySpec) -> None:
    os.environ["OKX_ENV"] = env
    from qooi.exchange.market import MarketData
    from qooi.exchange.trading import TradingClient, load_okx_env

    load_okx_env()
    tc = TradingClient()
    md = MarketData("okx")
    executor = LiveExecutor(tc, md)
    state = OkxStateProvider(tc)
    baskets = state.load(PAIRS, strategy_id=strategy.name)
    book = BasketBook(baskets)

    for p in PAIRS:
        sym = p.asset.symbol

        bot = tc.signal_ensure_bot(p, label=strategy.name)
        if not bot:
            print(f"  {sym:20s}  skip (failed to ensure bot)")
            continue

        df = md.candles(p.asset.sig_symbol, timeframe=p.asset.timeframe, limit=500, cache=True)
        if df.is_empty():
            print(f"  {sym:20s}  skip (no data)")
            continue
        df = compute_signal_frame(df, strategy)
        signal = float(df["position_signal"][-1]) if "position_signal" in df.columns else 0.0
        entry = float(df["entry_signal"][-1]) if "entry_signal" in df.columns else signal
        exit_signal = bool(df["exit_signal"][-1]) if "exit_signal" in df.columns else False
        strength = float(df["signal_strength"][-1]) if "signal_strength" in df.columns else 1.0
        signal_id = (
            str(df["signal_id"][-1] or strategy.name)
            if "signal_id" in df.columns
            else strategy.name
        )
        context = PipelineContext(
            strategy_id=strategy.name,
            market=BarMarket.from_frame(df),
            signal=BarSignal(
                position=signal,
                entry=entry,
                exit=exit_signal,
                strength=strength,
                signal_id=signal_id,
            ),
        )

        actions = process_bar(df, book, p, context=context)

        for a in actions:
            print(f"  {sym:20s} {a.action:10s} {a.side:5s} {a.reason}")

        executor.execute(actions, dry_run=dry_run)
        for action in actions:
            # Live execution should eventually apply only broker-accepted actions.
            # Current signal-bot path treats a successful dispatch as accepted.
            if dry_run:
                continue
            book.apply_action(action)


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    if cmd in ("test", "live"):
        dry = cmd == "live" and (len(sys.argv) <= 2 or sys.argv[2] != "live")
        strategy = _strategy_from_env(os.environ.get("QOOI_STRATEGY", DEFAULT_STRATEGY))
        _run(dry_run=dry, env=cmd, strategy=strategy)
    else:
        print("Usage: uv run python scripts/trade.py test|live [dry]")
        sys.exit(1)


if __name__ == "__main__":
    main()
