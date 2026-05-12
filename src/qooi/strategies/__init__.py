"""Strategy functions — all signals used by the backtester.

Active 1H strategies: ``momentum_1h`` (trend burst) and ``rsi_reversion``
(oversold bounce).  ``flow_pipeline`` is the original 4H OFI flow signal.
``portfolio`` is used by the multi-asset backtester.
"""

from __future__ import annotations

from qooi.strategies.ema_pullback import (
    ema_pullback_signal,
    ema_pullback_signal_expr,
)
from qooi.strategies.ema_pullback_v2 import (
    ema_pullback_v2_signal,
    ema_pullback_v2_signal_expr,
)
from qooi.strategies.flow_pipeline import (
    add_ofi_flow_columns,
    add_regime_features,
    apply_regime_gate,
)
from qooi.strategies.momentum import (
    momentum_signal,
    momentum_signal_expr,
)
from qooi.strategies.momentum_1h import (
    momentum_1h_signal,
    momentum_1h_signal_expr,
)
from qooi.strategies.portfolio import (
    AssetSignalState,
    PortfolioLimits,
    allocate_portfolio_weights,
    qualify_asset,
)
from qooi.strategies.rsi_reversion import (
    rsi_reversion_signal,
    rsi_reversion_signal_expr,
)

__all__ = [
    "add_ofi_flow_columns",
    "add_regime_features",
    "apply_regime_gate",
    "AssetSignalState",
    "PortfolioLimits",
    "allocate_portfolio_weights",
    "qualify_asset",
    "ema_pullback_signal",
    "ema_pullback_signal_expr",
    "ema_pullback_v2_signal",
    "ema_pullback_v2_signal_expr",
    "momentum_signal",
    "momentum_signal_expr",
    "momentum_1h_signal",
    "momentum_1h_signal_expr",
    "rsi_reversion_signal",
    "rsi_reversion_signal_expr",
]
