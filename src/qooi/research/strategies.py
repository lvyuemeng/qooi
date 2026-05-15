"""Strategy registry and benchmark groups for research runs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

from qooi.strategies import (
    StrategyBehavior,
    adaptive_zscore_mean_reversion_spec,
    ema_trend_baseline_spec,
    momentum_burst_spec,
    robust_zscore_mean_reversion_spec,
    rsi_bounce_reversion_spec,
    rsi_macd_trend_spec,
    zscore_mean_reversion_spec,
)

StrategyBuilder = Callable[[Any], StrategyBehavior]

DEFAULT_STRATEGY = "zscore_mean_reversion"


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


def _fixed_z(args: Any) -> StrategyBehavior:
    return zscore_mean_reversion_spec(
        z_period=_int_arg(args, "z_period", 20),
        entry_z=_float_arg(args, "entry_z", 2.0),
        exit_z=_float_arg(args, "exit_z", 0.25),
        adx_max=_float_arg(args, "adx_max", 25.0),
    )


def _adaptive_z(args: Any) -> StrategyBehavior:
    return adaptive_zscore_mean_reversion_spec(
        ewma_span=_int_arg(args, "ewma_span", 48),
        robust_period=_int_arg(args, "robust_period", 96),
        entry_z=_float_arg(args, "entry_z", 2.0),
        exit_z=_float_arg(args, "exit_z", 0.25),
        adx_max=_float_arg(args, "adx_max", 25.0),
        volatility_ratio_max=_float_arg(args, "volatility_ratio_max", 2.5),
    )


def _robust_z(args: Any) -> StrategyBehavior:
    return robust_zscore_mean_reversion_spec(
        robust_period=_int_arg(args, "robust_period", 96),
        entry_z=_float_arg(args, "entry_z", 2.0),
        exit_z=_float_arg(args, "exit_z", 0.25),
        adx_max=_float_arg(args, "adx_max", 25.0),
    )


STRATEGY_REGISTRY: dict[str, StrategyBuilder] = {
    "momentum_burst": _momentum("momentum_burst"),
    "momentum_burst_no_session": _momentum("momentum_burst_no_session", no_session=True),
    "momentum_burst_soft_volume": _momentum("momentum_burst_soft_volume", soft_volume=True),
    "ema_trend_baseline": lambda _args: ema_trend_baseline_spec(),
    "rsi_bounce_reversion": lambda _args: rsi_bounce_reversion_spec(),
    "zscore_mean_reversion": _fixed_z,
    "adaptive_zscore_mean_reversion": _adaptive_z,
    "robust_zscore_mean_reversion": _robust_z,
    "rsi_macd_trend": lambda _args: rsi_macd_trend_spec(),
}

STRATEGY_CHOICES = tuple(STRATEGY_REGISTRY)

BENCHMARK_GROUPS: dict[str, tuple[str, ...]] = {
    "zscore-family": (
        "zscore_mean_reversion",
        "adaptive_zscore_mean_reversion",
        "robust_zscore_mean_reversion",
    ),
    "baselines": (
        "ema_trend_baseline",
        "rsi_bounce_reversion",
        "zscore_mean_reversion",
    ),
    "candidate": (
        "zscore_mean_reversion",
        "adaptive_zscore_mean_reversion",
        "robust_zscore_mean_reversion",
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
        group = str(getattr(args, "benchmark_group", "zscore-family"))
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
        "mom_threshold",
        "trend_maturity",
        "volume_mult",
        "adx_threshold",
    ):
        if hasattr(args, name):
            parts.append(f"{name}={getattr(args, name)}")
    return ",".join(parts)
