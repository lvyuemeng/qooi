"""Canonical pair configuration — single source of truth for all scripts.

Orthogonal layers:
  - AssetConfig — compute-time: sizing, stop/target, capital, leverage
  - OkxSignalConfig — OKX signal bot execution settings

Composition: PairConfig = AssetConfig + OkxSignalConfig.

Runtime discovery (from OKX API):
  - BotIdentity — algo_id + signal_chan_id from orders-algo-pending
  - PositionState — has_position + side from signal/positions
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AssetConfig:
    symbol: str
    sig_symbol: str = ""
    timeframe: str = "4h"
    capital: float = 500.0
    max_risk_pct: float = 0.50
    leverage: float = 2.0
    max_notional_pct_per_basket: float = 1.0
    min_contracts: int = 1
    ct_val: float = 0.1
    atr_stop_mult: float = 2.0
    atr_target_mult: float = 3.0
    signal_threshold: float = 0.25
    ord_type: str = "limit"


@dataclass
class OkxSignalConfig:
    """OKX signal bot execution-layer TP/SL config."""

    tp_pct: str = "2.0"
    sl_pct: str = "2.5"


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
        okx=OkxSignalConfig(tp_pct="2.0", sl_pct="2.5"),
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
        okx=OkxSignalConfig(tp_pct="2.0", sl_pct="2.0"),
    ),
    PairConfig(
        asset=AssetConfig(
            symbol="BTC-USDT-SWAP",
            sig_symbol="BTC-USDT",
            timeframe="1H",
            capital=500,
            leverage=2.0,
            ct_val=0.01,
            signal_threshold=0.01,
        ),
        okx=OkxSignalConfig(tp_pct="2.0", sl_pct="2.5"),
    ),
    PairConfig(
        asset=AssetConfig(
            symbol="XAU-USDT-SWAP",
            sig_symbol="XAU-USDT",
            timeframe="1H",
            capital=500,
            leverage=2.0,
            ct_val=0.01,
            signal_threshold=0.01,
        ),
        okx=OkxSignalConfig(tp_pct="2.0", sl_pct="2.5"),
    ),
]


RESEARCH_PAIRS: list[PairConfig] = [
    *PAIRS,
    PairConfig(
        asset=AssetConfig(
            symbol="XRP-USDT-SWAP",
            sig_symbol="XRP-USDT",
            timeframe="1H",
            capital=200,
            leverage=3.0,
            ct_val=1.0,
            signal_threshold=0.01,
        ),
        okx=OkxSignalConfig(tp_pct="2.0", sl_pct="2.5"),
    ),
    PairConfig(
        asset=AssetConfig(
            symbol="DOGE-USDT-SWAP",
            sig_symbol="DOGE-USDT",
            timeframe="1H",
            capital=200,
            leverage=3.0,
            ct_val=1000.0,
            signal_threshold=0.01,
        ),
        okx=OkxSignalConfig(tp_pct="2.0", sl_pct="2.5"),
    ),
    PairConfig(
        asset=AssetConfig(
            symbol="ADA-USDT-SWAP",
            sig_symbol="ADA-USDT",
            timeframe="1H",
            capital=200,
            leverage=3.0,
            ct_val=100.0,
            signal_threshold=0.01,
        ),
        okx=OkxSignalConfig(tp_pct="2.0", sl_pct="2.5"),
    ),
    PairConfig(
        asset=AssetConfig(
            symbol="AVAX-USDT-SWAP",
            sig_symbol="AVAX-USDT",
            timeframe="1H",
            capital=200,
            leverage=3.0,
            ct_val=1.0,
            signal_threshold=0.01,
        ),
        okx=OkxSignalConfig(tp_pct="2.0", sl_pct="2.5"),
    ),
    PairConfig(
        asset=AssetConfig(
            symbol="LINK-USDT-SWAP",
            sig_symbol="LINK-USDT",
            timeframe="1H",
            capital=200,
            leverage=3.0,
            ct_val=1.0,
            signal_threshold=0.01,
        ),
        okx=OkxSignalConfig(tp_pct="2.0", sl_pct="2.5"),
    ),
    PairConfig(
        asset=AssetConfig(
            symbol="LTC-USDT-SWAP",
            sig_symbol="LTC-USDT",
            timeframe="1H",
            capital=200,
            leverage=3.0,
            ct_val=1.0,
            signal_threshold=0.01,
        ),
        okx=OkxSignalConfig(tp_pct="2.0", sl_pct="2.5"),
    ),
    PairConfig(
        asset=AssetConfig(
            symbol="OP-USDT-SWAP",
            sig_symbol="OP-USDT",
            timeframe="1H",
            capital=200,
            leverage=3.0,
            ct_val=1.0,
            signal_threshold=0.01,
        ),
        okx=OkxSignalConfig(tp_pct="2.0", sl_pct="2.5"),
    ),
    PairConfig(
        asset=AssetConfig(
            symbol="ARB-USDT-SWAP",
            sig_symbol="ARB-USDT",
            timeframe="1H",
            capital=200,
            leverage=3.0,
            ct_val=1.0,
            signal_threshold=0.01,
        ),
        okx=OkxSignalConfig(tp_pct="2.0", sl_pct="2.5"),
    ),
]
