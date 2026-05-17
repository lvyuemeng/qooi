"""Strategy registry and benchmark groups for research runs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

from qooi.strategies import (
    StrategyBehavior,
    ema_trend_baseline_spec,
    momentum_burst_spec,
    rsi_bounce_reversion_spec,
    rsi_macd_trend_spec,
    structure_event_reversal_v1_spec,
    structure_event_trend_aligned_mtf_confirm_v1_spec,
    structure_event_trend_aligned_v1_spec,
)

StrategyBuilder = Callable[[Any], StrategyBehavior]

DEFAULT_STRATEGY = "ema_trend_baseline"


@dataclass(frozen=True)
class StrategySelection:
    names: tuple[str, ...]
    strategies: tuple[StrategyBehavior, ...]


def _arg(args: Any, name: str, default: object) -> object:
    return getattr(args, name, default)


def _float_arg(args: Any, name: str, default: float) -> float:
    return float(cast(float | int | str, _arg(args, name, default)))


def _int_arg(args: Any, name: str, default: int) -> int:
    return int(cast(int | str, _arg(args, name, default)))


def _momentum(name: str, *, no_session: bool = False, soft_volume: bool = False) -> StrategyBuilder:
    def _build(args: Any) -> StrategyBehavior:
        return momentum_burst_spec(
            mom_threshold=_float_arg(args, "mom_threshold", 0.003),
            trend_maturity=_int_arg(args, "trend_maturity", 12),
            volume_mult=_float_arg(args, "volume_mult", 1.1),
            adx_threshold=_float_arg(args, "adx_threshold", 15.0),
            include_session_filter=not no_session,
            include_volume_filter=not soft_volume,
            name=name,
        )

    return _build


STRATEGY_REGISTRY: dict[str, StrategyBuilder] = {
    "momentum_burst": _momentum("momentum_burst"),
    "momentum_burst_no_session": _momentum("momentum_burst_no_session", no_session=True),
    "momentum_burst_soft_volume": _momentum("momentum_burst_soft_volume", soft_volume=True),
    "ema_trend_baseline": lambda _args: ema_trend_baseline_spec(),
    "rsi_bounce_reversion": lambda _args: rsi_bounce_reversion_spec(),
    "rsi_macd_trend": lambda _args: rsi_macd_trend_spec(),
    "structure_event_reversal_v1": lambda _args: structure_event_reversal_v1_spec(),
    "structure_event_trend_aligned_v1": lambda _args: structure_event_trend_aligned_v1_spec(),
    "structure_event_trend_aligned_mtf_confirm_v1": (
        lambda _args: structure_event_trend_aligned_mtf_confirm_v1_spec()
    ),
    "structure_event_reversal_no_vol_v1": lambda _args: structure_event_reversal_v1_spec(
        require_volume_impulse=False,
        name="structure_event_reversal_no_vol_v1",
    ),
    "structure_event_trend_aligned_no_vol_v1": lambda _args: structure_event_trend_aligned_v1_spec(
        require_volume_impulse=False,
        name="structure_event_trend_aligned_no_vol_v1",
    ),
}

STRATEGY_CHOICES = tuple(STRATEGY_REGISTRY)

BENCHMARK_GROUPS: dict[str, tuple[str, ...]] = {
    "baselines": (
        "ema_trend_baseline",
        "rsi_bounce_reversion",
        "rsi_macd_trend",
        "momentum_burst",
    ),
    "candidate": (
        "ema_trend_baseline",
        "rsi_bounce_reversion",
        "rsi_macd_trend",
        "momentum_burst",
        "structure_event_reversal_v1",
        "structure_event_trend_aligned_v1",
    ),
    "structure-development": (
        "structure_event_reversal_v1",
        "structure_event_trend_aligned_v1",
        "structure_event_trend_aligned_mtf_confirm_v1",
        "structure_event_reversal_no_vol_v1",
        "structure_event_trend_aligned_no_vol_v1",
        "ema_trend_baseline",
        "momentum_burst",
        "rsi_macd_trend",
    ),
    "all": STRATEGY_CHOICES,
}
BENCHMARK_GROUP_CHOICES = tuple(BENCHMARK_GROUPS)


def strategy_from_name(name: str, args: Any) -> StrategyBehavior:
    try:
        return STRATEGY_REGISTRY[name](args)
    except KeyError as exc:
        raise ValueError(f"Unknown strategy: {name}") from exc


def parse_strategy_names(value: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    unknown = [name for name in names if name not in STRATEGY_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown strategies: {', '.join(unknown)}")
    return names


def selected_strategy_names(args: Any) -> tuple[str, ...]:
    strategies = str(getattr(args, "strategies", "") or "")
    if strategies:
        return parse_strategy_names(strategies)
    if bool(getattr(args, "benchmark", False)):
        group = str(getattr(args, "benchmark_group", "baselines"))
        return BENCHMARK_GROUPS[group]
    return (str(getattr(args, "strategy", DEFAULT_STRATEGY)),)


def build_strategies(names: Sequence[str], args: Any) -> StrategySelection:
    return StrategySelection(
        names=tuple(names),
        strategies=tuple(strategy_from_name(name, args) for name in names),
    )


def strategy_args_metadata(args: Any, strategy_name: str | None = None) -> str:
    parts = [f"strategy={strategy_name or getattr(args, 'strategy', DEFAULT_STRATEGY)}"]
    for name in (
        "entry_z",
        "exit_z",
        "z_period",
        "ewma_span",
        "robust_period",
        "volatility_ratio_max",
        "adx_max",
        "zbasket_adx_max",
        "mom_threshold",
        "trend_maturity",
        "trend_period",
        "trend_return_max",
        "trend_return_min",
        "volume_mult",
        "adx_threshold",
    ):
        if hasattr(args, name):
            parts.append(f"{name}={getattr(args, name)}")
    return ",".join(parts)
