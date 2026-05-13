"""Canonical pair configuration — single source of truth for all scripts.

Orthogonal layers:
  - AssetConfig (decide.py) — compute-time: sizing, stop/target, decision
  - OkxSignalConfig — OKX signal bot execution: TP/SL, strategy dispatch

Composition: PairConfig = AssetConfig + OkxSignalConfig.

Runtime discovery (from OKX API):
  - BotIdentity — algo_id + signal_chan_id from orders-algo-pending
  - PositionState — has_position + side from signal/positions
"""

from __future__ import annotations

from dataclasses import dataclass

from qooi.core.decide import AssetConfig


@dataclass
class OkxSignalConfig:
    """OKX signal bot execution-layer config.

    Orthogonal to AssetConfig — only contains fields the OKX API needs.
    """

    strategy: str = "momentum_1h"
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


# ---- runtime discovery functions -------------------------------------------


def resolve_bot(tc, pair: PairConfig) -> BotIdentity | None:
    """Find OKX signal bot for this pair by chan_name.

    Queries orders-algo-pending and matches by signalChanName.
    """
    try:
        resp = tc.signal_get_pending()
        for bot in resp.get("data", []):
            if bot.get("signalChanName") == pair.chan_name:
                return BotIdentity(
                    algo_id=bot.get("algoId", ""),
                    signal_chan_id=bot.get("signalChanId", ""),
                )
    except Exception as e:
        print(f"    WARNING: resolve_bot({pair.chan_name}) failed: {e}")
    return None


def query_position(tc, bot: BotIdentity, pair: PairConfig) -> PositionState:
    """Query current position from OKX signal/positions.

    pos > 0 → long, pos < 0 → short, pos == "0" → flat.
    """
    try:
        resp = tc.signal_get_positions(bot.algo_id)
        for pos in resp.get("data", []):
            if pos.get("instId") == pair.asset.symbol:
                qty = str(pos.get("pos", "0"))
                if qty != "0" and qty not in ("", "nan", "None"):
                    p = float(qty)
                    if p > 0:
                        return PositionState(has_position=True, side="buy")
                    elif p < 0:
                        return PositionState(has_position=True, side="sell")
    except Exception:
        pass
    return PositionState()
