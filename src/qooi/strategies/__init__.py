"""Strategy functions — all signals used by the backtester.

Re-exports from strategy modules so callers can do::

    from qooi.strategies import sma_cross_signal, trend_pullback_signal
"""

from qooi.strategies.flow_pipeline import (
    add_adaptive_threshold,
    add_ofi_flow_columns,
    add_regime_features,
    apply_adaptive_gate,
    apply_micro_confirmation,
)
from qooi.strategies.intraday import multi_factor_intraday_signal
from qooi.strategies.ma_cross import (
    bollinger_signal,
    ema_vumanchu_signal,
    sma_cross_signal,
)
from qooi.strategies.pairs import build_pair_frame, pair_spread_signal
from qooi.strategies.portfolio import (
    AssetSignalState,
    PortfolioLimits,
    allocate_portfolio_weights,
    qualify_asset,
)
from qooi.strategies.trend_pullback import trend_pullback_signal

__all__ = [
    "bollinger_signal",
    "AssetSignalState",
    "PortfolioLimits",
    "allocate_portfolio_weights",
    "ema_vumanchu_signal",
    "multi_factor_intraday_signal",
    "build_pair_frame",
    "pair_spread_signal",
    "pair_zscore_signal",
    "qualify_asset",
    "sma_cross_signal",
    "trend_pullback_signal",
    "add_regime_features",
    "add_ofi_flow_columns",
    "apply_micro_confirmation",
    "check_obi_alignment",
    "add_adaptive_threshold",
    "apply_adaptive_gate",
]
