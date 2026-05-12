"""One-time setup: create OKX signal channels and strategies.

Usage::

    uv run python scripts/setup_signal.py testnet
    uv run python scripts/setup_signal.py live

Creates signal channels + strategies for all configured pairs.
Caches signalChanId/algoId to data/signal_bot_config.json for later use by trade.py.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PAIRS = {
    "test": [
        {
            "symbol": "ETH-USDT-SWAP",
            "sig_symbol": "ETH-USDT",
            "tf": "4h",
            "capital": 500,
            "leverage": "2",
            "risk_pct": 0.50,
            "ct_val": 0.1,
            "sig_threshold": 0.25,
            "tp_pct": "3.0",
            "sl_pct": "1.5",
        },
        {
            "symbol": "SOL-USDT-SWAP",
            "sig_symbol": "SOL-USDT",
            "tf": "4h",
            "capital": 200,
            "leverage": "3",
            "risk_pct": 0.50,
            "ct_val": 1.0,
            "sig_threshold": 0.35,
            "tp_pct": "4.0",
            "sl_pct": "2.0",
        },
    ],
    "live": [
        {
            "symbol": "ETH-USDT-SWAP",
            "sig_symbol": "ETH-USDT",
            "tf": "4h",
            "capital": 500,
            "leverage": "2",
            "risk_pct": 0.50,
            "ct_val": 0.1,
            "sig_threshold": 0.25,
            "tp_pct": "3.0",
            "sl_pct": "1.5",
        },
        {
            "symbol": "SOL-USDT-SWAP",
            "sig_symbol": "SOL-USDT",
            "tf": "4h",
            "capital": 200,
            "leverage": "3",
            "risk_pct": 0.50,
            "ct_val": 1.0,
            "sig_threshold": 0.35,
            "tp_pct": "4.0",
            "sl_pct": "2.0",
        },
    ],
}

CONFIG_PATH = Path(__file__).resolve().parent.parent / "data" / "signal_bot_config.json"


def main():
    env = sys.argv[1] if len(sys.argv) > 1 else "test"
    pairs = PAIRS.get(env, PAIRS["test"])
    os.environ["OKX_ENV"] = env

    from qooi.exchange.trading import TradingClient, load_okx_env

    load_okx_env()
    tc = TradingClient()

    configs: dict[str, dict] = {}

    for p in pairs:
        sym = p["symbol"]
        name = f"qooi-{sym.replace('-', '_')}"
        print(f"--- {sym} ---")

        # 1. Create signal channel
        chan = tc.signal_create(name, f"OFI flow signal for {sym} 4h")
        chan_id = (
            chan.get("signalChanId", chan.get("data", [{}])[0].get("signalChanId", ""))
            if isinstance(chan, dict)
            else ""
        )
        print(f"  signalChanId: {chan_id}")

        # 2. Create signal strategy
        strat = tc.signal_create_order_algo(
            signal_chan_id=chan_id,
            inst_ids=[sym],
            lever=p["leverage"],
            entry_type="3",  # fixed contracts
            amt="",  # derived from capital + ct_val per invocation
            tp_pct=p["tp_pct"],
            sl_pct=p["sl_pct"],
            sub_ord_type="9",  # TradingView signal
            allow_multiple_entry=False,
        )
        strat_data = strat if isinstance(strat, dict) else {}
        algo_id = strat_data.get("algoId", strat_data.get("data", [{}])[0].get("algoId", ""))
        print(f"  algoId: {algo_id}")

        configs[sym] = {
            "signal_chan_id": chan_id,
            "algo_id": algo_id,
            "tp_pct": p["tp_pct"],
            "sl_pct": p["sl_pct"],
            "leverage": p["leverage"],
        }

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(configs, indent=2))
    print(f"\nSaved to {CONFIG_PATH}")


if __name__ == "__main__":
    main()
