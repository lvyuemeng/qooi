"""Core instrument value types."""

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
