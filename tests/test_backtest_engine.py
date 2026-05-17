from __future__ import annotations

from qooi.core.executor import BacktestExecutor


def test_backtest_engine_constructs_executor():
    executor = BacktestExecutor(initial_capital=1000.0)

    assert executor._initial_capital == 1000.0
