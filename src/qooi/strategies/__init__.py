"""Strategy functions — all signals used by the backtester.

Re-exports from strategy modules so callers can do::

    from qooi.strategies import sma_cross_signal, trend_pullback_signal
"""

from qooi.strategies.adaptive_threshold import (
    add_adaptive_threshold,
    apply_adaptive_gate,
)
from qooi.strategies.intraday import (
    cvd_proxy_signal,
    ensemble_intraday_signal,
    multi_factor_intraday_signal,
    order_book_imbalance_signal,
    pair_zscore_signal,
)
from qooi.strategies.ma_cross import (
    bollinger_signal,
    ema_vumanchu_signal,
    sma_cross_signal,
)
from qooi.strategies.micro_confirmation import (
    add_ofi_flow_columns,
    apply_micro_confirmation,
    check_obi_alignment,
)
from qooi.strategies.pairs import build_pair_frame, pair_spread_signal
from qooi.strategies.portfolio import (
    AssetSignalState,
    PortfolioLimits,
    allocate_portfolio_weights,
    qualify_asset,
)
from qooi.strategies.regime import add_regime_features
from qooi.strategies.trend_pullback import trend_pullback_signal

__all__ = [
    "bollinger_signal",
    "AssetSignalState",
    "PortfolioLimits",
    "allocate_portfolio_weights",
    "cvd_proxy_signal",
    "ensemble_intraday_signal",
    "ema_vumanchu_signal",
    "multi_factor_intraday_signal",
    "order_book_imbalance_signal",
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
