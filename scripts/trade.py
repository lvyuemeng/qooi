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

# Perpetual swap instruments.
#   ctVal = contract value in base currency (OKX standard)
#   ETH-USDT-SWAP: 0.1 ETH/ct,  min 1 contract
#   SOL-USDT-SWAP: 1 SOL/ct,    min 1 contract
#   BTC-USDT-SWAP: 0.01 BTC/ct, min 1 contract
#
# Design: enter with LIMIT (signal verification), exit with MARKET (risk guarantee).
#   An unfilled limit order = prevented loss from bad signal.  Fill rate IS signal accuracy.
TESTNET_PAIRS = [
    {"symbol": "ETH-USDT-SWAP", "tf": "4h", "capital": 500, "risk_pct": 0.50, "leverage": 2.0,
     "ct_val": 0.1, "sig_threshold": 0.25},
    {"symbol": "SOL-USDT-SWAP", "tf": "4h", "capital": 200, "risk_pct": 0.50, "leverage": 3.0,
     "ct_val": 1.0, "sig_threshold": 0.35},
    {"symbol": "BTC-USDT-SWAP", "tf": "4h", "capital": 1000, "risk_pct": 0.80, "leverage": 2.0,
     "ct_val": 0.01, "sig_threshold": 0.25},
]
LIVE_PAIRS = [
    {"symbol": "ETH-USDT-SWAP", "tf": "4h", "capital": 500, "risk_pct": 0.50, "leverage": 2.0, "ct_val": 0.1},
]


def _run(pairs: list[dict], dry_run: bool, env: str) -> None:
    """Run portfolio step.

    ``env`` must be ``"test"`` (OKX demo) or ``"live"`` (OKX production).
    The env-var is set *before* TradingClient imports so the correct
    credentials are loaded.
    """
    os.environ["OKX_ENV"] = env
    tc = TradingClient()

    def _usdt() -> float:
        try:
            b = tc.balance("USDT")
            return float(b[0].get("availBal", 0)) if b else 0.0
        except Exception:
            return 0.0

    def _pos_count() -> int:
        try:
            return len(tc.positions())
        except Exception:
            return -1

    pre_usdt = _usdt()
    pre_pos = _pos_count()

    # --- pre-flight status ---
    print("=== pre-flight ===")
    try:
        for b in tc.balance():
            print(f"  {b.get('ccy','?'):6s} avail={b.get('availBal','?')} frozen={b.get('frozenBal','?')}")
    except Exception:
        print("  balance: unavailable")
    try:
        pos_list = tc.positions()
        print(f"  positions: {len(pos_list)} open")
    except Exception:
        print("  positions: unavailable")
    try:
        pend = tc.pending()
        print(f"  pending orders: {len(pend)}")
    except Exception:
        print("  pending orders: unavailable")
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
