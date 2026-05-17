"""Resolved research backtest configuration and sizing overrides."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from qooi.core.config import PAIRS, RESEARCH_PAIRS, PairConfig

PROFILE_CHOICES = ("research", "safe", "smoke", "live-like")
UNIVERSE_CHOICES = ("core", "research")
DATA_SOURCE_CHOICES = ("swap", "spot_signal_swap_exec", "spot")
STYLE_CHOICES = ("single", "rolling", "walk-forward", "cross-validate")


@dataclass(frozen=True)
class RiskGateConfig:
    min_coverage_pct: float = 0.0
    max_dd_pct: float | None = None
    max_notional_exposure_pct: float | None = None
    min_trades: int = 0
    min_pf: float = 0.0
    min_expectancy_pct: float | None = None
    min_execution_acceptance_pct: float = 0.0
    fail_on_risk: bool = False


@dataclass(frozen=True)
class SizingOverrideConfig:
    normalize: bool = False
    risk_pct: float | None = None
    max_notional_pct: float | None = None
    leverage: float | None = None
    capital: float | None = None
    min_contracts: float | None = None

    @property
    def active(self) -> bool:
        return self.normalize or any(
            value is not None
            for value in (
                self.risk_pct,
                self.max_notional_pct,
                self.leverage,
                self.capital,
                self.min_contracts,
            )
        )

    def metadata(self) -> tuple[str, ...]:
        profile = "normalized" if self.normalize else "configured"
        if not self.active:
            return ("sizing_profile=configured",)
        max_notional = self.max_notional_pct if self.max_notional_pct is not None else "pair"
        return (
            f"sizing_profile={profile}",
            f"risk_pct={self.risk_pct if self.risk_pct is not None else 'pair'}",
            f"max_notional_pct={max_notional}",
            f"leverage={self.leverage if self.leverage is not None else 'pair'}",
            f"capital={self.capital if self.capital is not None else 'pair'}",
            f"min_contracts={self.min_contracts if self.min_contracts is not None else 'pair'}",
        )


@dataclass(frozen=True)
class SignalDebugFilterConfig:
    side: str = "both"
    include_signal_ids: tuple[str, ...] = ()
    exclude_signal_ids: tuple[str, ...] = ()

    @property
    def active(self) -> bool:
        return (
            self.side != "both"
            or bool(self.include_signal_ids)
            or bool(self.exclude_signal_ids)
        )

    def metadata(self) -> tuple[str, ...]:
        if not self.active:
            return ("signal_debug_filter=none",)
        include = ",".join(self.include_signal_ids) or "none"
        exclude = ",".join(self.exclude_signal_ids) or "none"
        return (
            "signal_debug_filter=active",
            f"signal_side={self.side}",
            f"include_signal_id={include}",
            f"exclude_signal_id={exclude}",
        )


@dataclass(frozen=True)
class ResolvedBacktestConfig:
    profile: str
    days: int
    min_bars: int
    universe: str
    data_source: str
    style: str
    risk_gates: RiskGateConfig
    sizing: SizingOverrideConfig
    max_per_strategy_symbol: int = 3
    signal_filters: SignalDebugFilterConfig = field(default_factory=SignalDebugFilterConfig)

    def metadata(self) -> tuple[str, ...]:
        return (
            f"profile={self.profile}",
            f"days={self.days}",
            f"min_bars={self.min_bars}",
            f"universe={self.universe}",
            f"data_source={self.data_source}",
            f"style={self.style}",
            f"max_per_strategy_symbol={self.max_per_strategy_symbol}",
            *self.sizing.metadata(),
            *self.signal_filters.metadata(),
        )


def _csv_tuple(value: Any) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value or "").split(",") if item.strip())


def resolve_config(args: Any) -> ResolvedBacktestConfig:
    profile = str(getattr(args, "profile", "research"))
    days_arg = getattr(args, "days", None)
    min_bars_arg = getattr(args, "min_bars", None)
    min_coverage_arg = getattr(args, "min_coverage_pct", None)
    days = int(days_arg) if days_arg is not None else 730
    min_bars = int(min_bars_arg) if min_bars_arg is not None else 12000
    min_coverage = float(min_coverage_arg) if min_coverage_arg is not None else 0.0
    max_dd = getattr(args, "max_dd_pct", None)
    max_notional = getattr(args, "max_notional_exposure_pct", None)
    min_trades = int(getattr(args, "min_trades", 0))
    min_pf = float(getattr(args, "min_pf", 0.0))
    min_expectancy = getattr(args, "min_expectancy_pct", None)
    normalize = bool(getattr(args, "normalize_sizing", False))

    if profile == "smoke":
        days = int(days_arg) if days_arg is not None else 90
        min_bars = int(min_bars_arg) if min_bars_arg is not None else 1000
    elif profile in ("research", "safe"):
        min_coverage = min_coverage or 90.0

    if profile == "safe":
        normalize = True
        max_dd = 40.0 if max_dd is None else max_dd
        max_notional = 200.0 if max_notional is None else max_notional
        min_trades = max(min_trades, 30)
        min_pf = max(min_pf, 1.10)
        min_expectancy = 0.0 if min_expectancy is None else min_expectancy
    min_execution_acceptance = float(
        getattr(args, "min_execution_acceptance_pct", 0.0) or 0.0
    )
    if profile == "safe":
        min_execution_acceptance = max(min_execution_acceptance, 30.0)
    max_per_strategy_symbol = int(getattr(args, "max_per_strategy_symbol", 0) or 0)
    if max_per_strategy_symbol <= 0:
        max_per_strategy_symbol = 1 if profile == "safe" else 3

    sizing = SizingOverrideConfig(
        normalize=normalize,
        risk_pct=getattr(args, "risk_pct", None),
        max_notional_pct=getattr(args, "max_notional_pct", None),
        leverage=getattr(args, "leverage", None),
        capital=getattr(args, "capital", None),
        min_contracts=getattr(args, "min_contracts", None),
    )
    if sizing.normalize:
        sizing = SizingOverrideConfig(
            normalize=True,
            risk_pct=0.02 if sizing.risk_pct is None else sizing.risk_pct,
            max_notional_pct=1.0
            if sizing.max_notional_pct is None
            else sizing.max_notional_pct,
            leverage=2.0 if sizing.leverage is None else sizing.leverage,
            capital=sizing.capital,
            min_contracts=sizing.min_contracts,
        )

    long_only = bool(getattr(args, "long_only", False))
    short_only = bool(getattr(args, "short_only", False))
    if long_only and short_only:
        raise ValueError("--long-only and --short-only are mutually exclusive")
    signal_filters = SignalDebugFilterConfig(
        side="long" if long_only else "short" if short_only else "both",
        include_signal_ids=_csv_tuple(getattr(args, "include_signal_id", "")),
        exclude_signal_ids=_csv_tuple(getattr(args, "exclude_signal_id", "")),
    )

    return ResolvedBacktestConfig(
        profile=profile,
        days=days,
        min_bars=min_bars,
        universe=str(getattr(args, "universe", "core")),
        data_source=str(getattr(args, "data_source", "swap")),
        style=str(getattr(args, "style", "single")),
        risk_gates=RiskGateConfig(
            min_coverage_pct=min_coverage,
            max_dd_pct=float(max_dd) if max_dd is not None else None,
            max_notional_exposure_pct=float(max_notional) if max_notional is not None else None,
            min_trades=min_trades,
            min_pf=min_pf,
            min_expectancy_pct=float(min_expectancy) if min_expectancy is not None else None,
            min_execution_acceptance_pct=min_execution_acceptance,
            fail_on_risk=bool(getattr(args, "fail_on_risk", False)),
        ),
        sizing=sizing,
        max_per_strategy_symbol=max_per_strategy_symbol,
        signal_filters=signal_filters,
    )


def selected_pairs(args: Any, config: ResolvedBacktestConfig) -> tuple[PairConfig, ...]:
    universe = RESEARCH_PAIRS if config.universe == "research" else PAIRS
    symbol = str(getattr(args, "symbol", "") or "")
    exclude = {
        item.strip()
        for item in str(getattr(args, "exclude_symbol", "") or "").split(",")
        if item.strip()
    }
    pairs = [pair for pair in universe if pair.asset.symbol not in exclude]
    if symbol:
        pairs = [
            pair
            for pair in pairs
            if pair.asset.symbol == symbol or pair.asset.sig_symbol == symbol
        ]
    return tuple(apply_sizing_overrides(pair, config.sizing) for pair in pairs)


def apply_sizing_overrides(pair: PairConfig, sizing: SizingOverrideConfig) -> PairConfig:
    if not sizing.active:
        return pair
    asset = pair.asset
    updated = replace(
        asset,
        max_risk_pct=asset.max_risk_pct if sizing.risk_pct is None else sizing.risk_pct,
        max_notional_pct_per_basket=asset.max_notional_pct_per_basket
        if sizing.max_notional_pct is None
        else sizing.max_notional_pct,
        leverage=asset.leverage if sizing.leverage is None else sizing.leverage,
        capital=asset.capital if sizing.capital is None else sizing.capital,
        min_contracts=asset.min_contracts if sizing.min_contracts is None else sizing.min_contracts,
    )
    return replace(pair, asset=updated)


def risk_gate_metadata(gates: RiskGateConfig) -> tuple[str, ...]:
    max_notional = (
        gates.max_notional_exposure_pct
        if gates.max_notional_exposure_pct is not None
        else "none"
    )
    min_expectancy = (
        gates.min_expectancy_pct if gates.min_expectancy_pct is not None else "none"
    )
    return (
        f"min_coverage_pct={gates.min_coverage_pct}",
        f"max_dd_pct={gates.max_dd_pct if gates.max_dd_pct is not None else 'none'}",
        f"max_notional_exposure_pct={max_notional}",
        f"min_trades={gates.min_trades}",
        f"min_pf={gates.min_pf}",
        f"min_expectancy_pct={min_expectancy}",
        f"min_execution_acceptance_pct={gates.min_execution_acceptance_pct}",
    )
