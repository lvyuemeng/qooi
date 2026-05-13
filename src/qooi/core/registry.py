"""Strategy registry — maps strategy names to signal functions.

Used by pipeline to dispatch signal computation without if/elif chains.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from qooi.core.indicators import (
    SignalResult,
    compute_momentum_1h,
    compute_rsi_reversion_1h,
    compute_single,
)

SignalFn = Callable[[str], SignalResult | None]


@dataclass
class Entry:
    fn: SignalFn
    desc: str

    def compute(self, symbol: str) -> SignalResult | None:
        return self.fn(symbol)


REGISTRY: dict[str, Entry] = {
    "momentum_1h": Entry(compute_momentum_1h, "1H momentum burst — directional persistence"),
    "rsi_reversion": Entry(
        compute_rsi_reversion_1h, "1H RSI mean-reversion — oversold bounce in uptrend"
    ),
    "flow_pipeline": Entry(
        lambda s: compute_single(s, "4h", 0.25), "4H OFI flow pipeline — order flow imbalance"
    ),
}


def resolve(name: str) -> Entry | None:
    return REGISTRY.get(name)
