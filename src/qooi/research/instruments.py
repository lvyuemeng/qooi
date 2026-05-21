"""Research/application instrument universes."""

from __future__ import annotations

from qooi.core.instruments import AssetConfig, PairConfig

CORE_UNIVERSE: tuple[PairConfig, ...] = (
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
)


RESEARCH_UNIVERSE: tuple[PairConfig, ...] = (
    *CORE_UNIVERSE,
    PairConfig(
        asset=AssetConfig(
            symbol="BNB-USDT-SWAP",
            sig_symbol="BNB-USDT",
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
)


UNIVERSES: dict[str, tuple[PairConfig, ...]] = {
    "core": CORE_UNIVERSE,
    "research": RESEARCH_UNIVERSE,
}


def universe_pairs(name: str) -> tuple[PairConfig, ...]:
    try:
        return UNIVERSES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(UNIVERSES))
        raise ValueError(f"unknown universe {name!r}; expected one of: {choices}") from exc
