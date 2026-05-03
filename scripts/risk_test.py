"""Test dynamic risk management — ATR-based sizing, stop/target/trailing.

Usage:
    uv run python scripts/strategy_test.py BTC-USDT 1D
"""

from __future__ import annotations

import sys

from qooi.exchange.backtest import CostModel, RiskConfig
from qooi.exchange.pipeline import Pipeline
from qooi.strategies import ema_vumanchu_signal


def main() -> None:
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTC-USDT"
    bar = sys.argv[2] if len(sys.argv) > 2 else "1D"

    cost = CostModel(slippage_pct=0.001, spread_pct=0.0005, commission_pct=0.0005)

    # --- Test 1: fixed 1x, no risk mgmt (baseline) ---
    print(f"\n== Fixed 1x, no stop (baseline) | {symbol} {bar} ==")
    s = Pipeline().run(symbol, bar, 360, 10_000, ema_vumanchu_signal(), cost=cost)
    print(s.eval)

    # --- Test 2: 2x leverage, no risk mgmt ---
    print(f"\n== 2x leverage, no stop | {symbol} {bar} ==")
    r2 = RiskConfig(max_leverage=2.0, position_sizing="fixed")
    s2 = Pipeline().run(symbol, bar, 360, 10_000, ema_vumanchu_signal(), cost=cost, risk=r2)
    print(s2.eval)

    # --- Test 3: 1.5x, ATR sizing, stop/target/trailing ---
    print(f"\n== 1.5x, ATR sizing, stop/trailing | {symbol} {bar} ==")
    r3 = RiskConfig(
        max_leverage=1.5,
        position_sizing="atr",
        max_risk_pct=0.02,
        atr_stop_mult=3.0,
        atr_target_mult=6.0,
        trailing_activation_mult=2.0,
        trailing_distance_mult=2.0,
    )
    s3 = Pipeline().run(symbol, bar, 360, 10_000, ema_vumanchu_signal(), cost=cost, risk=r3)
    print(s3.eval)


if __name__ == "__main__":
    main()
