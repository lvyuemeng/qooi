"""Canonical research/backtest instrument configuration."""

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
    min_contracts: float = 1.0
    lot_size: float = 1.0
    tick_size: float = 0.01
    ct_val: float = 0.1
    atr_stop_mult: float = 2.0
    atr_target_mult: float = 3.0
    signal_threshold: float = 0.25


@dataclass
class PairConfig:
    """Canonical backtest pair."""

    asset: AssetConfig


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
            min_contracts=0.01,
            lot_size=0.01,
            tick_size=0.01,
            signal_threshold=0.01,
        )
    ),
    PairConfig(
        asset=AssetConfig(
            symbol="SOL-USDT-SWAP",
            sig_symbol="SOL-USDT",
            timeframe="1H",
            capital=200,
            leverage=3.0,
            ct_val=1.0,
            min_contracts=0.01,
            lot_size=0.01,
            tick_size=0.01,
            signal_threshold=0.01,
        )
    ),
    PairConfig(
        asset=AssetConfig(
            symbol="BTC-USDT-SWAP",
            sig_symbol="BTC-USDT",
            timeframe="1H",
            capital=500,
            leverage=2.0,
            ct_val=0.01,
            min_contracts=0.01,
            lot_size=0.01,
            tick_size=0.1,
            signal_threshold=0.01,
        )
    ),
    PairConfig(
        asset=AssetConfig(
            symbol="XAU-USDT-SWAP",
            sig_symbol="XAU-USDT",
            timeframe="1H",
            capital=500,
            leverage=2.0,
            ct_val=0.001,
            min_contracts=0.01,
            lot_size=0.01,
            tick_size=0.01,
            signal_threshold=0.01,
        )
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
        )
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
        )
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
        )
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
        )
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
        )
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
        )
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
        )
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
        )
    ),
]
