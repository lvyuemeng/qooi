"""Explicit strategy specs and benchmark selections for research runs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from qooi.strategies.specs import (
    StrategyBehavior,
    ema_trend_baseline_spec,
    momentum_burst_spec,
    rsi_bounce_reversion_spec,
    rsi_macd_trend_spec,
    structure_event_reversal_v1_spec,
    structure_event_trend_aligned_v1_spec,
)

DEFAULT_STRATEGY = "ema_trend_baseline"


@dataclass(frozen=True)
class StrategySelection:
    strategies: tuple[StrategyBehavior, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(strategy.name for strategy in self.strategies)


def _strategy_specs() -> dict[str, StrategyBehavior]:
    return {
        "ema_trend_baseline": ema_trend_baseline_spec(),
        "rsi_bounce_reversion": rsi_bounce_reversion_spec(),
        "rsi_macd_trend": rsi_macd_trend_spec(),
        "momentum_burst": momentum_burst_spec(),
        "momentum_burst_no_session": momentum_burst_spec(
            include_session_filter=False,
            name="momentum_burst_no_session",
        ),
        "momentum_burst_soft_volume": momentum_burst_spec(
            include_volume_filter=False,
            name="momentum_burst_soft_volume",
        ),
        "structure_event_reversal_v1": structure_event_reversal_v1_spec(),
        "structure_event_trend_aligned_v1": structure_event_trend_aligned_v1_spec(),
        "structure_event_trend_aligned_no_range_v1": structure_event_trend_aligned_v1_spec(
            exclude_market_stages=("range",),
            exclude_market_stage_reasons=("compressed_mid_range",),
            name="structure_event_trend_aligned_no_range_v1",
        ),
        "structure_event_trend_aligned_no_range_longs_v1": structure_event_trend_aligned_v1_spec(
            exclude_long_market_stages=("range",),
            exclude_long_market_stage_reasons=("compressed_mid_range",),
            name="structure_event_trend_aligned_no_range_longs_v1",
        ),
        "structure_event_reversal_no_vol_v1": structure_event_reversal_v1_spec(
            require_volume_impulse=False,
            name="structure_event_reversal_no_vol_v1",
        ),
        "structure_event_trend_aligned_no_vol_v1": structure_event_trend_aligned_v1_spec(
            require_volume_impulse=False,
            name="structure_event_trend_aligned_no_vol_v1",
        ),
    }


STRATEGY_CHOICES = tuple(_strategy_specs())

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
        "structure_event_trend_aligned_no_range_v1",
        "structure_event_trend_aligned_no_range_longs_v1",
        "structure_event_reversal_no_vol_v1",
        "structure_event_trend_aligned_no_vol_v1",
        "ema_trend_baseline",
        "momentum_burst",
        "rsi_macd_trend",
    ),
    "all": STRATEGY_CHOICES,
}
BENCHMARK_GROUP_CHOICES = tuple(BENCHMARK_GROUPS)


def strategy_selection(
    labels: Sequence[str] = (),
    *,
    benchmark: bool = False,
    benchmark_group: str = "baselines",
    default: str = DEFAULT_STRATEGY,
) -> StrategySelection:
    specs = _strategy_specs()
    selected = BENCHMARK_GROUPS[benchmark_group] if benchmark else tuple(labels) or (default,)
    return StrategySelection(strategies=_specs_for_labels(specs, selected))


def _specs_for_labels(
    specs: dict[str, StrategyBehavior], labels: Sequence[str]
) -> tuple[StrategyBehavior, ...]:
    unknown = [label for label in labels if label not in specs]
    if unknown:
        raise ValueError(f"Unknown strategies: {', '.join(unknown)}")
    return tuple(specs[label] for label in labels)


def strategy_metadata(strategy: StrategyBehavior) -> str:
    return f"strategy={strategy.name}"

