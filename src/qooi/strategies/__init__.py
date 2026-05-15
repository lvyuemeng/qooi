"""Composable strategy package.

Strategies are built from reusable features, conditions, filters, and hold policies.
"""

from __future__ import annotations

from qooi.strategies.indicators import (
    ORDER_BOOK_FEATURE_SCHEMA,
    IndicatorSources,
    add_ofi_flow_columns,
    add_regime_features,
    apply_regime_gate,
    attach_order_book_features,
    compute_flow_pipeline_frame,
    compute_indicator_frame,
    normalize_order_book_snapshots,
)
from qooi.strategies.portfolio import (
    AssetSignalState,
    PortfolioLimits,
    allocate_portfolio_weights,
    qualify_asset,
)
from qooi.strategies.specs import (
    FlowPipelineSpec,
    HoldPolicy,
    SignalRule,
    StrategyBehavior,
    StrategySpec,
    adaptive_zscore_mean_reversion_spec,
    apply_strategy_spec,
    compute_signal_frame,
    ema_trend_baseline_spec,
    flow_pipeline_spec,
    latest_signal,
    momentum_burst_spec,
    robust_zscore_mean_reversion_spec,
    rsi_bounce_reversion_spec,
    rsi_macd_trend_spec,
    strategy_signal_diagnostics,
    zscore_mean_reversion_spec,
)

__all__ = [
    "apply_strategy_spec",
    "compute_signal_frame",
    "latest_signal",
    "HoldPolicy",
    "SignalRule",
    "FlowPipelineSpec",
    "StrategyBehavior",
    "StrategySpec",
    "adaptive_zscore_mean_reversion_spec",
    "ema_trend_baseline_spec",
    "momentum_burst_spec",
    "robust_zscore_mean_reversion_spec",
    "rsi_bounce_reversion_spec",
    "rsi_macd_trend_spec",
    "zscore_mean_reversion_spec",
    "strategy_signal_diagnostics",
    "flow_pipeline_spec",
    "IndicatorSources",
    "ORDER_BOOK_FEATURE_SCHEMA",
    "add_ofi_flow_columns",
    "add_regime_features",
    "attach_order_book_features",
    "apply_regime_gate",
    "compute_indicator_frame",
    "compute_flow_pipeline_frame",
    "normalize_order_book_snapshots",
    "AssetSignalState",
    "PortfolioLimits",
    "allocate_portfolio_weights",
    "qualify_asset",
]
