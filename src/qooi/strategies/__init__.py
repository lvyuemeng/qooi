"""Composable strategy package.

Strategies are built from reusable features, conditions, filters, and hold policies.
"""

from __future__ import annotations

from qooi.strategies.compose import apply_strategy_spec, compute_signal_frame, latest_signal
from qooi.strategies.flow_pipeline import (
    add_ofi_flow_columns,
    add_regime_features,
    apply_regime_gate,
)
from qooi.strategies.portfolio import (
    AssetSignalState,
    PortfolioLimits,
    allocate_portfolio_weights,
    qualify_asset,
)
from qooi.strategies.specs import (
    HoldPolicy,
    SignalRule,
    StrategySpec,
    momentum_burst_spec,
    resolve_spec,
    rsi_bounce_reversion_spec,
)

__all__ = [
    "apply_strategy_spec",
    "compute_signal_frame",
    "latest_signal",
    "HoldPolicy",
    "SignalRule",
    "StrategySpec",
    "momentum_burst_spec",
    "resolve_spec",
    "rsi_bounce_reversion_spec",
    "add_ofi_flow_columns",
    "add_regime_features",
    "apply_regime_gate",
    "AssetSignalState",
    "PortfolioLimits",
    "allocate_portfolio_weights",
    "qualify_asset",
]
