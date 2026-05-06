"""Trading script — single invocation for GitHub Actions workflows.

Usage::

    uv run python scripts/trade.py testnet
    uv run python scripts/trade.py live [dry]
"""

from __future__ import annotations

import sys

from qooi.exchange.trading import (
    PortfolioConfig,
    PortfolioRunner,
    TradingClient,
    default_signal_source,
)

TESTNET_PAIRS = [
    {"symbol": "ETH-USDT", "tf": "4h", "capital": 100, "risk_pct": 0.03, "leverage": 1.0},
    {"symbol": "SOL-USDT", "tf": "4h", "capital": 50, "risk_pct": 0.05, "leverage": 1.0},
]
LIVE_PAIRS = [
    {"symbol": "ETH-USDT", "tf": "4h", "capital": 100, "risk_pct": 0.03, "leverage": 1.0},
]


def _run(pairs: list[dict], dry_run: bool) -> None:
    tc = TradingClient()

    def _usdt() -> float:
        b = tc.balance("USDT")
        return float(b[0].get("availBal", 0)) if b else 0.0

    pre_usdt = _usdt()
    pre_pos = len(tc.positions())

    config = PortfolioConfig(pairs=pairs, dry_run=dry_run)
    runner = PortfolioRunner(config)
    src = default_signal_source()
    runner.compute_all_signals(source=src)
    runner.step(source=src)

    runner.write_summary(tc, pre_usdt, pre_pos)


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "testnet"
    if cmd == "testnet":
        _run(TESTNET_PAIRS, dry_run=False)
    elif cmd == "live":
        dry = sys.argv[2] != "live" if len(sys.argv) > 2 else True
        _run(LIVE_PAIRS, dry_run=dry)
    else:
        print("Usage: uv run python scripts/trade.py testnet|live [dry]")
        sys.exit(1)


if __name__ == "__main__":
    main()
