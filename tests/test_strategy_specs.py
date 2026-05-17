from __future__ import annotations

from qooi.research.strategies import DEFAULT_STRATEGY, build_strategies


def test_strategy_registry_builds_default_strategy():
    selection = build_strategies((DEFAULT_STRATEGY,), None)

    assert len(selection.strategies) == 1
    assert selection.strategies[0].name == DEFAULT_STRATEGY
