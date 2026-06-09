"""Portfolio-level risk and allocation helpers for intraday strategies.

These helpers sit above single-asset signal generation. They do not know
how a signal was produced; they only consume its latest score, recent
volatility and validation metrics, then return a scaled target weight.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AssetSignalState:
    symbol: str
    score: float
    volatility: float
    sharpe: float
    drawdown_pct: float
    loss_streak: int = 0


@dataclass
class PortfolioLimits:
    max_total_abs_weight: float = 2.0
    max_single_asset_weight: float = 0.75
    max_same_direction_weight: float = 1.25
    min_sharpe: float = 0.0
    max_drawdown_pct: float = 35.0


def qualify_asset(state: AssetSignalState, limits: PortfolioLimits) -> bool:
    """Simple pre-trade filter.

    Assets with unacceptable historical quality are skipped completely.
    """
    if abs(state.score) < 1e-9:
        return False
    if state.sharpe < limits.min_sharpe:
        return False
    if state.drawdown_pct > limits.max_drawdown_pct:
        return False
    if state.volatility <= 0:
        return False
    return True


def allocate_portfolio_weights(
    states: list[AssetSignalState],
    limits: PortfolioLimits = PortfolioLimits(),
) -> dict[str, float]:
    """Allocate normalized target weights across assets.

    Allocation principles:
    - stronger score → larger absolute weight
    - higher volatility → smaller weight
    - recent loss streak → multiplicative penalty
    - enforce total, per-asset and same-direction caps
    """
    qualified = [s for s in states if qualify_asset(s, limits)]
    if not qualified:
        return {}

    raw: dict[str, float] = {}
    for s in qualified:
        loss_factor = 1.0
        if s.loss_streak >= 3:
            loss_factor = 0.25
        elif s.loss_streak == 2:
            loss_factor = 0.5
        elif s.loss_streak == 1:
            loss_factor = 0.75

        inv_vol = 1.0 / max(s.volatility, 1e-9)
        weight = s.score * inv_vol * loss_factor
        # hard per-asset cap
        weight = max(-limits.max_single_asset_weight, min(limits.max_single_asset_weight, weight))
        raw[s.symbol] = weight

    if not raw:
        return {}

    # normalize to total absolute exposure
    total_abs = sum(abs(v) for v in raw.values())
    if total_abs > limits.max_total_abs_weight and total_abs > 0:
        scale = limits.max_total_abs_weight / total_abs
        raw = {k: v * scale for k, v in raw.items()}

    # cap same-direction exposure
    long_abs = sum(v for v in raw.values() if v > 0)
    short_abs = sum(-v for v in raw.values() if v < 0)
    if long_abs > limits.max_same_direction_weight and long_abs > 0:
        scale = limits.max_same_direction_weight / long_abs
        raw = {k: (v * scale if v > 0 else v) for k, v in raw.items()}
    if short_abs > limits.max_same_direction_weight and short_abs > 0:
        scale = limits.max_same_direction_weight / short_abs
        raw = {k: (v * scale if v < 0 else v) for k, v in raw.items()}

    # correlation-aware halving: when all non-zero scores point the same way,
    # diversification is weak → halve total exposure
    non_zero_dirs = [1 if v > 0 else (-1 if v < 0 else 0) for v in raw.values() if abs(v) > 1e-9]
    if len(non_zero_dirs) >= 2 and all(d == non_zero_dirs[0] for d in non_zero_dirs):
        raw = {k: v * 0.7 for k, v in raw.items()}

    return raw

