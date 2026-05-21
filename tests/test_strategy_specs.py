from __future__ import annotations

from qooi.strategies.catalog import DEFAULT_STRATEGY, strategy_selection


def test_strategy_registry_builds_default_strategy():
    selection = strategy_selection(default=DEFAULT_STRATEGY)

    assert len(selection.strategies) == 1
    assert selection.strategies[0].name == DEFAULT_STRATEGY
