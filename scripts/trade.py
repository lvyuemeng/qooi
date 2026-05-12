"""Live trading entry point — GitHub Actions single invocation.

Uses shared signal pipeline (qooi.core.signal) and decision engine
(qooi.core.decide) so backtest and live trade identically.

Workflow:
  1. Load signal bot config (created once by setup_signal.py)
  2. For each pair: compute signal, query OKX state, decide, push

Usage::

    uv run python scripts/trade.py testnet
    uv run python scripts/trade.py live [dry]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent / "data" / "signal_bot_config.json"

PAIRS = [
    {
        "symbol": "ETH-USDT-SWAP",
        "sig_symbol": "ETH-USDT",
        "tf": "4h",
        "capital": 500,
        "leverage": 2.0,
        "ct_val": 0.1,
        "sig_threshold": 0.25,
    },
    {
        "symbol": "SOL-USDT-SWAP",
        "sig_symbol": "SOL-USDT",
        "tf": "4h",
        "capital": 200,
        "leverage": 3.0,
        "ct_val": 1.0,
        "sig_threshold": 0.35,
    },
    {
        "symbol": "BTC-USDT-SWAP",
        "sig_symbol": "BTC-USDT",
        "tf": "4h",
        "capital": 1000,
        "leverage": 2.0,
        "ct_val": 0.01,
        "sig_threshold": 0.25,
    },
]


def _run(dry_run: bool, env: str) -> None:
    os.environ["OKX_ENV"] = env
    from qooi.core.decide import AssetConfig, decide_active, decide_idle
    from qooi.core.signal import compute_single
    from qooi.exchange.market import MarketData
    from qooi.exchange.trading import TradingClient, load_okx_env

    load_okx_env()
    tc = TradingClient()
    md = MarketData("okx")
    configs = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}

    for p in PAIRS:
        sym = p["symbol"]
        cfg = configs.get(sym)
        if not cfg:
            print(f"  {sym:20s}  skip (no signal bot config)")
            continue

        # 1. Shared signal pipeline
        signal = compute_single(p["sig_symbol"], p["tf"], p["sig_threshold"])
        if signal is None:
            print(f"  {sym:20s}  skip (no_signal)")
            continue

        # 2. Query OKX signal bot state
        details = tc.signal_get_details(cfg["algo_id"])
        bot = details if isinstance(details, dict) else {}
        if isinstance(bot.get("data"), list) and bot["data"]:
            bot = bot["data"][0]
        frozen = float(bot.get("frozenBal", "0"))
        pos_side = cfg.get("_last_side", "")
        has_position = bool(pos_side) and frozen > 0

        # 3. Decide (same functions as backtest)
        asset_cfg = AssetConfig(
            symbol=sym,
            sig_symbol=p["sig_symbol"],
            timeframe=p["tf"],
            capital=p["capital"],
            leverage=p["leverage"],
            ct_val=p["ct_val"],
            signal_threshold=p["sig_threshold"],
            ord_type=p.get("ord_type", "limit"),
        )

        if not has_position:
            obi = md.ob_snapshot(sym, limit=1)
            entry_px = obi.ask_price if obi else 0
            side = "buy" if signal.signal > 0 else "sell"
            if side == "sell" and obi:
                entry_px = obi.bid_price
            d = decide_idle(signal, entry_px, side, asset_cfg)
        else:
            d = decide_active(signal, pos_side, asset_cfg)

        print(
            f"  {sym:20s} sig={signal.signal:+.4f} atr={signal.atr} "
            f"action={d.action.value} {d.detail}"
        )

        if dry_run:
            continue

        # 4. Execute
        if d.action.value == "enter":
            try:
                tc.signal_push_sub_order(
                    algo_id=cfg["algo_id"],
                    signal_chan_id=cfg["signal_chan_id"],
                    inst_id=sym,
                    side=d.side,
                    sz=str(int(d.sz)),
                    ord_type=asset_cfg.ord_type,
                    px=str(d.entry_px),
                    attach_algo_ords=[
                        {
                            "slTriggerPx": str(d.stop_px),
                            "slOrdPx": "-1",
                            "tpTriggerPx": str(d.target_px),
                            "tpOrdPx": "-1",
                            "cxlOnClosePos": "true",
                        }
                    ],
                )
                configs[sym]["_last_side"] = d.side
                CONFIG_PATH.write_text(json.dumps(configs, indent=2))
                print(
                    f"    ORDER {d.side} sz={d.sz} px={d.entry_px} sl={d.stop_px} tp={d.target_px}"
                )
            except Exception as e:
                print(f"    ORDER FAILED: {e}")

        elif d.action.value == "exit":
            try:
                tc.signal_close_position(
                    algo_id=cfg["algo_id"],
                    signal_chan_id=cfg["signal_chan_id"],
                    inst_id=sym,
                )
                configs[sym]["_last_side"] = ""
                CONFIG_PATH.write_text(json.dumps(configs, indent=2))
                print(f"    CLOSE ({d.detail})")
            except Exception as e:
                print(f"    CLOSE FAILED: {e}")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "testnet"
    if cmd in ("testnet", "live"):
        dry = cmd == "live" and (len(sys.argv) <= 2 or sys.argv[2] != "live")
        _run(dry_run=dry, env="test" if cmd == "testnet" else "live")
    else:
        print("Usage: uv run python scripts/trade.py testnet|live [dry]")
        sys.exit(1)


if __name__ == "__main__":
    main()
