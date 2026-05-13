"""One-time setup: create OKX signal channels and strategies.

Usage::

    uv run python scripts/setup_signal.py testnet
    uv run python scripts/setup_signal.py live

Creates signal channels + strategies for all configured pairs from
the canonical PAIRS list (src/qooi/core/config.py).
Idempotent — skips creation if channel/strategy already exists on OKX.
"""

from __future__ import annotations

import os
import sys

from qooi.core.config import PAIRS


def main():
    env_arg = sys.argv[1] if len(sys.argv) > 1 else "test"
    env = "test" if env_arg == "testnet" else env_arg
    os.environ["OKX_ENV"] = env

    from qooi.exchange.trading import TradingClient, load_okx_env

    load_okx_env()
    tc = TradingClient()

    # Pre-query active bots so we can skip channels that already exist
    existing: dict[str, dict] = {}
    try:
        pending = tc.signal_get_pending()
        for bot in pending.get("data", []):
            existing[bot.get("signalChanName", "")] = bot
    except Exception:
        pass

    for p in PAIRS:
        sym = p.asset.symbol
        name = p.chan_name
        print(f"--- {sym} ---")

        # 1. Create or reuse signal channel
        if name in existing:
            chan_id = existing[name].get("signalChanId", "")
            print(f"  signalChanId: {chan_id}  (existing)")
        else:
            desc = f"{p.okx.strategy} signal for {sym} {p.asset.timeframe}"
            chan = tc.signal_create(name, desc)
            chan_id = (
                chan.get("signalChanId", chan.get("data", [{}])[0].get("signalChanId", ""))
                if isinstance(chan, dict)
                else ""
            )
            print(f"  signalChanId: {chan_id}")

        # 2. Create or reuse signal strategy
        existing_algo = existing.get(name, {}).get("algoId", "")
        if existing_algo:
            algo_id = existing_algo
            print(f"  algoId: {algo_id}  (existing)")
        else:
            strat = tc.signal_create_order_algo(
                signal_chan_id=chan_id,
                inst_ids=[sym],
                lever=str(int(p.asset.leverage)),
                invest_amt=str(int(p.asset.capital)),
                entry_type="3",
                amt="",
                tp_pct=p.okx.tp_pct,
                sl_pct=p.okx.sl_pct,
                sub_ord_type="9",
                allow_multiple_entry=False,
            )
            strat_data = strat if isinstance(strat, dict) else {}
            algo_id = strat_data.get("algoId", strat_data.get("data", [{}])[0].get("algoId", ""))
            print(f"  algoId: {algo_id}")

    print("\nDone. Run: uv run python scripts/trade.py testnet")


if __name__ == "__main__":
    main()
