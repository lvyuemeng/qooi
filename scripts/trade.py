"""Live trading entry point — GitHub Actions single invocation.

Auto-creates OKX signal bot on first run (idempotent), then computes
signal, decides, and pushes orders on every subsequent run.

Uses shared signal pipeline (qooi.core.signal) and decision engine
(qooi.core.decide) so backtest and live trade identically.

1H strategies:
  - momentum_1h → ETH (6-bar momentum burst + ADX + session filter)
  - rsi_reversion → SOL (oversold bounce in uptrend with RSI confirmation)

Position state: queried from OKX GET /signal/positions (server-side truth).

Usage::

    uv run python scripts/trade.py test
    uv run python scripts/trade.py live [dry]
"""

from __future__ import annotations

import os
import sys

from qooi.core.config import PAIRS


def _run(dry_run: bool, env: str) -> None:
    os.environ["OKX_ENV"] = env
    from qooi.core.decide import decide_active, decide_idle
    from qooi.core.signal import compute_momentum_1h, compute_rsi_reversion_1h
    from qooi.exchange.market import MarketData
    from qooi.exchange.trading import TradingClient, load_okx_env

    load_okx_env()
    tc = TradingClient()
    md = MarketData("okx")

    for p in PAIRS:
        sym = p.asset.symbol
        strat = p.okx.strategy

        # 0. Ensure bot exists (creates channel + order-algo if missing)
        bot = tc.signal_ensure_bot(p)
        if not bot:
            print(f"  {sym:20s}  skip (failed to ensure bot)")
            continue

        # 1. Compute signal via strategy-specific function
        if strat == "momentum_1h":
            signal = compute_momentum_1h(p.asset.sig_symbol)
        elif strat == "rsi_reversion":
            signal = compute_rsi_reversion_1h(p.asset.sig_symbol)
        else:
            print(f"  {sym:20s}  skip (unknown strategy: {strat})")
            continue

        if signal is None:
            print(f"  {sym:20s}  skip (no_signal)")
            continue

        # 2. Query OKX position state — server-side source of truth
        pos = tc.signal_query_position(bot, p)

        # 3. Decide (same functions as backtest)
        cfg = p.asset
        if not pos.has_position:
            obi = md.ob_snapshot(sym, limit=1)
            entry_px = obi.ask_price if obi else 0
            side = "buy" if signal.signal > 0 else "sell"
            if side == "sell" and obi:
                entry_px = obi.bid_price
            d = decide_idle(signal, entry_px, side, cfg)
        else:
            d = decide_active(signal, pos.side, cfg)

        print(
            f"  {sym:20s} strategy={strat} sig={signal.signal:+.0f} atr={signal.atr} "
            f"pos={pos.side if pos.has_position else 'flat'} action={d.action.value} {d.detail}"
        )

        if dry_run:
            continue

        # 4. Execute
        if d.action.value == "enter":
            try:
                tc.signal_execute_enter(d, bot.algo_id, bot.signal_chan_id, sym)
                print(
                    f"    ORDER {d.side} sz={d.sz} px={d.entry_px} sl={d.stop_px} tp={d.target_px}"
                )
            except Exception as e:
                print(f"    ORDER FAILED: {e}")

        elif d.action.value == "exit":
            try:
                tc.signal_execute_exit(bot.algo_id, bot.signal_chan_id, sym)
                print(f"    CLOSE ({d.detail})")
            except Exception as e:
                print(f"    CLOSE FAILED: {e}")


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
