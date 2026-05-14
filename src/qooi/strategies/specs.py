"""Composable strategy specifications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

import qooi.strategies.conditions as c
from qooi.strategies.features import (
    FeatureFn,
    add_momentum_return,
    add_price_structure,
    add_trend_maturity,
    add_utc_hour,
    add_volume_average,
)

Direction = Literal[-1, 1]
StrategyName = Literal["momentum_burst", "rsi_bounce_reversion", "flow_pipeline"]


@dataclass(frozen=True)
class SignalRule:
    name: str
    direction: Direction
    condition: pl.Expr


@dataclass(frozen=True)
class HoldPolicy:
    exit_when: pl.Expr | None = None
    max_bars: int | None = None


@dataclass(frozen=True)
class StrategySpec:
    name: str
    required_columns: tuple[str, ...]
    features: tuple[FeatureFn, ...]
    entries: tuple[SignalRule, ...]
    filters: tuple[pl.Expr, ...] = ()
    hold: HoldPolicy = HoldPolicy()


def momentum_burst_spec(
    *,
    mom_bars: int = 6,
    mom_threshold: float = 0.003,
    ema_fast: int = 20,
    ema_mid: int = 50,
    ema_slow: int = 200,
    trend_maturity: int = 20,
    volume_mult: float = 1.5,
) -> StrategySpec:
    filters = (
        c.adx_above(20.0),
        c.session_between(8, 22),
        c.trend_mature(trend_maturity),
        c.volume_spike(volume_mult),
    )
    return StrategySpec(
        name="momentum_burst",
        required_columns=(
            "timestamp",
            "close",
            "high",
            "low",
            "vol",
            "atr_14",
            "adx_14",
            f"ema_{ema_fast}",
            f"ema_{ema_mid}",
            f"ema_{ema_slow}",
        ),
        features=(
            add_momentum_return(mom_bars),
            add_volume_average(20),
            add_trend_maturity(ema_mid=ema_mid, ema_slow=ema_slow),
            add_utc_hour(),
            add_price_structure(5, 20),
        ),
        entries=(
            SignalRule(
                "long_momentum_burst",
                1,
                c.uptrend(ema_mid, ema_slow)
                & c.momentum_gt(mom_threshold)
                & c.above_ema(ema_fast)
                & c.higher_low_structure(),
            ),
            SignalRule(
                "short_momentum_burst",
                -1,
                c.downtrend(ema_mid, ema_slow)
                & c.momentum_lt(-mom_threshold)
                & c.below_ema(ema_fast)
                & c.lower_high_structure(),
            ),
        ),
        filters=filters,
        hold=HoldPolicy(exit_when=~(c.uptrend(ema_mid, ema_slow) | c.downtrend(ema_mid, ema_slow))),
    )


def rsi_bounce_reversion_spec(
    *,
    rsi_period: int = 14,
    rsi_oversold: float = 30.0,
    rsi_bounce: float = 25.0,
    rsi_confirmation: float = 20.0,
    rsi_exit: float = 50.0,
    ema_mid: int = 50,
    ema_slow: int = 200,
) -> StrategySpec:
    return StrategySpec(
        name="rsi_bounce_reversion",
        required_columns=(
            "timestamp",
            "close",
            "high",
            "low",
            "atr_14",
            "adx_14",
            f"ema_{ema_mid}",
            f"ema_{ema_slow}",
            f"rsi_{rsi_period}",
        ),
        features=(
            add_trend_maturity(ema_mid=ema_mid, ema_slow=ema_slow),
            add_utc_hour(),
            add_price_structure(5, 20),
        ),
        entries=(
            SignalRule(
                "long_rsi_bounce",
                1,
                c.uptrend(ema_mid, ema_slow)
                & c.rsi_cross_from_oversold(
                    rsi_period=rsi_period, oversold=rsi_oversold, bounce=rsi_bounce
                )
                & c.rsi_bounce_held(rsi_period=rsi_period, confirmation=rsi_confirmation)
                & c.higher_low_structure(),
            ),
        ),
        filters=(c.adx_above(20.0), c.session_between(8, 22)),
        hold=HoldPolicy(
            exit_when=(~c.uptrend(ema_mid, ema_slow)) | c.rsi_above(threshold=rsi_exit)
        ),
    )


def resolve_spec(name: str) -> StrategySpec:
    if name == "momentum_burst":
        return momentum_burst_spec()
    if name == "rsi_bounce_reversion":
        return rsi_bounce_reversion_spec()
    if name == "flow_pipeline":
        return StrategySpec(name="flow_pipeline", required_columns=(), features=(), entries=())
    raise ValueError(f"Unknown strategy: {name}")
