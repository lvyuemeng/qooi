"""Pipeline backtest — ensemble comparison and per-strategy detail.

Usage:
    uv run python scripts/backtest.py
    uv run python scripts/backtest.py --mode base|grid|martingale|hedge
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

from qooi.core.config import PAIRS, RESEARCH_PAIRS
from qooi.core.evaluate import compare
from qooi.core.executor import BacktestExecutor
from qooi.core.recovery import RecoveryConfig, RecoveryKind

DEFAULT_STRATEGY = "momentum_burst"


def _cache_path(data_symbol: str, timeframe: str) -> str:
    symbol = data_symbol.replace("-", "_").replace("/", "_").upper()
    bar = timeframe.replace(" ", "").upper()
    return f"data/cache/{symbol}_{bar}.parquet"


def _mode_config(mode: str) -> RecoveryConfig:
    if mode == "grid":
        return RecoveryConfig(
            strategy=RecoveryKind.GRID, zone_atr=1.0, multiplier=2.0, max_levels=3
        )
    if mode == "martingale":
        return RecoveryConfig(strategy=RecoveryKind.MARTINGALE, zone_atr=1.0, max_levels=3)
    if mode == "hedge":
        return RecoveryConfig(strategy=RecoveryKind.HEDGE, zone_atr=1.0)
    return RecoveryConfig(strategy=RecoveryKind.NONE)


def main() -> None:
    mode = "base"
    strategy = DEFAULT_STRATEGY
    research = False
    symbol = ""
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--mode" and i + 1 < len(args):
            mode = args[i + 1]
            i += 2
        elif args[i] == "--strategy" and i + 1 < len(args):
            strategy = args[i + 1]
            i += 2
        elif args[i] in ("--research", "--all"):
            research = True
            i += 1
        elif args[i] == "--symbol" and i + 1 < len(args):
            symbol = args[i + 1]
            i += 2
        else:
            i += 1

    recovery_cfg = _mode_config(mode)
    pairs = RESEARCH_PAIRS if research else PAIRS
    if symbol:
        pairs = [p for p in pairs if p.asset.symbol == symbol or p.asset.sig_symbol == symbol]
    reports = []
    for pair in pairs:
        path = _cache_path(pair.asset.sig_symbol, pair.asset.timeframe)
        if not Path(path).exists():
            print(f"skip {pair.asset.symbol}: missing cache {path}")
            continue
        df = pl.read_parquet(path)
        bt = BacktestExecutor(initial_capital=pair.asset.capital, cost_pct=0.00005)
        report = bt.run_report(df, pair, recovery_cfg=recovery_cfg, strategy=strategy)
        reports.append(report)

    if not reports:
        print("No backtest reports generated")
        return

    print(f"Mode: {mode}")
    print(f"Strategy: {strategy}")
    print(compare(*reports))
    for report in reports:
        print(f"\n{report.label}")
        print(report.table())


if __name__ == "__main__":
    main()
