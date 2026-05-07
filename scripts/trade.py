"""Trading script — single invocation for GitHub Actions workflows.

Usage::

    uv run python scripts/trade.py testnet
    uv run python scripts/trade.py live [dry]
"""

from __future__ import annotations

import os
import sys

from qooi.exchange.trading import (
    PortfolioConfig,
    PortfolioRunner,
    TradingClient,
    default_signal_source,
)

# Perpetual futures (swap) — positions persist on exchange, margin-efficient.
# Requires OKX account in "single-currency margin" or "multi-currency margin" mode.
# Error 51010 = account is in "simple" mode → switch in OKX settings.
#
# OKX swap contract sizes (ct_val):
#   ETH-USDT-SWAP: 0.1 ETH/ct,  min 1 ct
#   SOL-USDT-SWAP: 1 SOL/ct,    min 1 ct
#   BTC-USDT-SWAP: 0.01 BTC/ct, min 1 ct
TESTNET_PAIRS = [
    {"symbol": "ETH-USDT-SWAP", "tf": "4h", "capital": 500, "risk_pct": 0.50, "leverage": 2.0,
     "ct_val": 0.1, "td_mode": "cross", "sig_threshold": 0.25},
    {"symbol": "SOL-USDT-SWAP", "tf": "4h", "capital": 200, "risk_pct": 0.50, "leverage": 3.0,
     "ct_val": 1.0, "td_mode": "cross", "sig_threshold": 0.35},
    {"symbol": "BTC-USDT-SWAP", "tf": "4h", "capital": 1000, "risk_pct": 0.80, "leverage": 2.0,
     "ct_val": 0.01, "td_mode": "cross", "sig_threshold": 0.25},
]
LIVE_PAIRS = [
    {"symbol": "ETH-USDT-SWAP", "tf": "4h", "capital": 500, "risk_pct": 0.50, "leverage": 2.0,
     "ct_val": 0.1, "td_mode": "cross", "sig_threshold": 0.25},
    {"symbol": "SOL-USDT-SWAP", "tf": "4h", "capital": 200, "risk_pct": 0.50, "leverage": 3.0,
     "ct_val": 1.0, "td_mode": "cross", "sig_threshold": 0.35},
    {"symbol": "BTC-USDT-SWAP", "tf": "4h", "capital": 1000, "risk_pct": 0.80, "leverage": 2.0,
     "ct_val": 0.01, "td_mode": "cross", "sig_threshold": 0.25},
]


def _run(pairs: list[dict], dry_run: bool, env: str) -> None:
    """Run portfolio step.

    ``env`` must be ``"test"`` (OKX demo) or ``"live"`` (OKX production).
    The env-var is set *before* TradingClient imports so the correct
    credentials are loaded.
    """
    os.environ["OKX_ENV"] = env
    tc = TradingClient()

    # --- pre-flight: query once, reuse (was 5 API calls, now 3) ---
    pre_balance: list[dict] = []
    pre_positions: list[dict] = []
    pre_pending: list[dict] = []
    try:
        pre_balance = tc.balance()
        pre_positions = tc.positions()
        pre_pending = tc.pending()
    except Exception:
        pass

    pre_usdt = 0.0
    for b in pre_balance:
        if b.get("ccy") == "USDT":
            pre_usdt = float(b.get("availBal", 0))
            break
    pre_pos = len(pre_positions)

    print("=== pre-flight ===")
    for b in pre_balance:
        print(f"  {b.get('ccy','?'):6s} avail={b.get('availBal','?')} frozen={b.get('frozenBal','?')}")
    print(f"  positions: {pre_pos} open")
    print(f"  pending orders: {len(pre_pending)}")
    print("==================\n")

    max_frozen = 0.95 if env == "test" else 0.50  # testnet has frozen display artifacts
    config = PortfolioConfig(pairs=pairs, dry_run=dry_run, max_frozen_pct=max_frozen)
    runner = PortfolioRunner(config)
    src = default_signal_source()
    signals = runner.compute_all_signals(source=src)
    runner.step(source=src, signals=signals, tc=tc)

    runner.write_summary(tc, pre_usdt, pre_pos)


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "testnet"
    if cmd == "testnet":
        _run(TESTNET_PAIRS, dry_run=False, env="test")
    elif cmd == "live":
        dry = sys.argv[2] != "live" if len(sys.argv) > 2 else True
        _run(LIVE_PAIRS, dry_run=dry, env="live")
    else:
        print("Usage: uv run python scripts/trade.py testnet|live [dry]")
        sys.exit(1)


if __name__ == "__main__":
    main()
