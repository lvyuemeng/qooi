"""Core configuration and instrument contracts."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


@dataclass(frozen=True)
class AssetConfig:
    symbol: str
    sig_symbol: str = ""
    timeframe: str = "1H"
    capital: float = 500.0
    leverage: float = 2.0
    ct_val: float = 1.0
    min_contracts: float = 1.0
    lot_size: float = 1.0
    tick_size: float = 0.01
    max_risk_pct: float = 0.01
    max_notional_pct_per_basket: float = 1.0
    signal_threshold: float = 0.0
    atr_stop_mult: float = 2.0
    atr_target_mult: float = 3.0

    def __post_init__(self) -> None:
        if not self.sig_symbol:
            object.__setattr__(self, "sig_symbol", self.symbol.replace("-SWAP", ""))


@dataclass(frozen=True)
class PairConfig:
    asset: AssetConfig


CORE_UNIVERSE: tuple[PairConfig, ...] = (
    PairConfig(
        AssetConfig(
            "ETH-USDT-SWAP",
            "ETH-USDT",
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
        AssetConfig(
            "SOL-USDT-SWAP",
            "SOL-USDT",
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
        AssetConfig(
            "BTC-USDT-SWAP",
            "BTC-USDT",
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
        AssetConfig(
            "XAU-USDT-SWAP",
            "XAU-USDT",
            capital=500,
            leverage=2.0,
            ct_val=0.001,
            min_contracts=0.01,
            lot_size=0.01,
            tick_size=0.01,
            signal_threshold=0.01,
        )
    ),
)

RESEARCH_UNIVERSE: tuple[PairConfig, ...] = (
    *CORE_UNIVERSE,
    PairConfig(
        AssetConfig(
            "BNB-USDT-SWAP",
            "BNB-USDT",
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
        AssetConfig(
            "XRP-USDT-SWAP",
            "XRP-USDT",
            capital=200,
            leverage=3.0,
            ct_val=1.0,
            signal_threshold=0.01,
        )
    ),
    PairConfig(
        AssetConfig(
            "DOGE-USDT-SWAP",
            "DOGE-USDT",
            capital=200,
            leverage=3.0,
            ct_val=1000.0,
            signal_threshold=0.01,
        )
    ),
    PairConfig(
        AssetConfig(
            "ADA-USDT-SWAP",
            "ADA-USDT",
            capital=200,
            leverage=3.0,
            ct_val=100.0,
            signal_threshold=0.01,
        )
    ),
    PairConfig(
        AssetConfig(
            "AVAX-USDT-SWAP",
            "AVAX-USDT",
            capital=200,
            leverage=3.0,
            ct_val=1.0,
            signal_threshold=0.01,
        )
    ),
    PairConfig(
        AssetConfig(
            "LINK-USDT-SWAP",
            "LINK-USDT",
            capital=200,
            leverage=3.0,
            ct_val=1.0,
            signal_threshold=0.01,
        )
    ),
    PairConfig(
        AssetConfig(
            "LTC-USDT-SWAP",
            "LTC-USDT",
            capital=200,
            leverage=3.0,
            ct_val=1.0,
            signal_threshold=0.01,
        )
    ),
    PairConfig(
        AssetConfig(
            "OP-USDT-SWAP", "OP-USDT", capital=200, leverage=3.0, ct_val=1.0, signal_threshold=0.01
        )
    ),
    PairConfig(
        AssetConfig(
            "ARB-USDT-SWAP",
            "ARB-USDT",
            capital=200,
            leverage=3.0,
            ct_val=1.0,
            signal_threshold=0.01,
        )
    ),
)

UNIVERSES: dict[str, tuple[PairConfig, ...]] = {
    "core": CORE_UNIVERSE,
    "research": RESEARCH_UNIVERSE,
}


def universe_pairs(name: str) -> tuple[PairConfig, ...]:
    if name in UNIVERSES:
        return UNIVERSES[name]
    raise ValueError(f"unknown universe {name!r}; expected one of: {', '.join(sorted(UNIVERSES))}")
