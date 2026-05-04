"""Strategy functions — all signals used by the backtester.

Re-exports from strategy modules so callers can do::

    from qooi.strategies import sma_cross_signal, trend_pullback_signal
"""

from qooi.strategies.ma_cross import (
    bollinger_signal,
    ema_vumanchu_signal,
    sma_cross_signal,
)
from qooi.strategies.trend_pullback import trend_pullback_signal

__all__ = [
    "bollinger_signal",
    "ema_vumanchu_signal",
    "sma_cross_signal",
    "trend_pullback_signal",
]
