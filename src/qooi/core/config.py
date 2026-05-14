"""Canonical pair configuration — single source of truth for all scripts.

Orthogonal layers:
  - AssetConfig — compute-time: sizing, stop/target, capital, leverage
  - OkxSignalConfig — OKX signal bot execution: TP/SL, strategy dispatch

Composition: PairConfig = AssetConfig + OkxSignalConfig.

Runtime discovery (from OKX API):
  - BotIdentity — algo_id + signal_chan_id from orders-algo-pending
  - PositionState — has_position + side from signal/positions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from qooi.core.indicators import SignalResult


@dataclass
class AssetConfig:
    symbol: str
    sig_symbol: str = ""
    timeframe: str = "4h"
    capital: float = 500.0
    max_risk_pct: float = 0.50
    leverage: float = 2.0
    ct_val: float = 0.1
    atr_stop_mult: float = 2.0
    atr_target_mult: float = 3.0
    signal_threshold: float = 0.25
    ord_type: str = "limit"


StrategyName = Literal["momentum_1h", "rsi_reversion", "flow_pipeline"]


@dataclass
class OkxSignalConfig:
    """OKX signal bot execution-layer config.

    Orthogonal to AssetConfig — only contains fields the OKX API needs.
    """

    strategy: StrategyName = "momentum_1h"
    tp_pct: str = "2.0"
    sl_pct: str = "2.5"

    def compute(self, symbol: str) -> SignalResult | None:
        """Compute signal via the strategy-specific function."""
        from qooi.core.indicators import (
            compute_momentum_1h,
            compute_rsi_reversion_1h,
            compute_single,
        )

        if self.strategy == "momentum_1h":
            return compute_momentum_1h(symbol)
        if self.strategy == "rsi_reversion":
            return compute_rsi_reversion_1h(symbol)
        if self.strategy == "flow_pipeline":
            return compute_single(symbol, "4h", 0.25)
        return None


@dataclass
class PairConfig:
    """Canonical pair — composes compute-layer + OKX execution-layer."""

    asset: AssetConfig
    okx: OkxSignalConfig

    @property
    def chan_name(self) -> str:
        return f"qooi-{self.asset.symbol.replace('-', '_')}"


@dataclass
class BotIdentity:
    """Runtime discovery from OKX orders-algo-pending."""

    algo_id: str = ""
    signal_chan_id: str = ""


@dataclass
class PositionState:
    """Runtime position query from OKX signal/positions."""

    has_position: bool = False
    side: str = ""


# ---- canonical pair list ---------------------------------------------------

PAIRS: list[PairConfig] = [
    PairConfig(
        asset=AssetConfig(
            symbol="ETH-USDT-SWAP",
            sig_symbol="ETH-USDT",
            timeframe="1H",
            capital=500,
            leverage=2.0,
            ct_val=0.1,
            signal_threshold=0.01,
        ),
        okx=OkxSignalConfig(strategy="momentum_1h", tp_pct="2.0", sl_pct="2.5"),
    ),
    PairConfig(
        asset=AssetConfig(
            symbol="SOL-USDT-SWAP",
            sig_symbol="SOL-USDT",
            timeframe="1H",
            capital=200,
            leverage=3.0,
            ct_val=1.0,
            signal_threshold=0.01,
        ),
        okx=OkxSignalConfig(strategy="rsi_reversion", tp_pct="2.0", sl_pct="2.0"),
    ),
]
