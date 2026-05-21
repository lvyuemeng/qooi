"""Typed research command config and focused runtime value types."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from qooi.core.instruments import PairConfig
from qooi.research.instruments import universe_pairs
from qooi.strategies.catalog import BENCHMARK_GROUP_CHOICES, DEFAULT_STRATEGY, STRATEGY_CHOICES
from qooi.strategies.features import StructureClassifierConfig

PROFILE_CHOICES = ("research", "safe", "smoke", "live-like")
UNIVERSE_CHOICES = ("core", "research")
DATA_SOURCE_CHOICES = ("swap", "spot_signal_swap_exec", "spot")
STYLE_CHOICES = ("single", "rolling", "walk-forward", "cross-validate")

Profile = Literal["research", "safe", "smoke", "live-like"]
UniverseName = Literal["core", "research"]
DataSource = Literal["swap", "spot_signal_swap_exec", "spot"]
RunStyle = Literal["single", "rolling", "walk-forward", "cross-validate"]
DiagnosticMode = Literal[
    "backtest",
    "classifier",
    "state",
    "state-profitability",
    "state-filter-delta",
    "modulation-effect",
    "market-state-forward",
    "tradability",
    "research-evaluation",
]
ResearchOutputName = Literal[
    "classifier",
    "tradability",
    "market-state-forward",
    "market-state-modulation",
    "trade-record-modulation",
]
ClassifierProfile = Literal["default", "fixed", "rolling"]
RangeThresholdMode = Literal["rolling_quantile", "fixed"]
RangeThresholdFallback = Literal["fixed", "data_error"]
MarketStateDeltaMode = Literal["fixed", "cost_multiple"]
ModulationSeMethod = Literal["iid", "effective_n", "newey_west", "bootstrap"]


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunConfig(StrictConfigModel):
    profile: Profile = "research"
    universe: UniverseName = "core"
    data_source: DataSource = "swap"
    symbol: str = ""
    exclude_symbol: tuple[str, ...] = ()
    allow_swap_signal_fallback: bool = False
    show_status: bool = False


class CacheConfig(StrictConfigModel):
    audit: bool = False
    refresh: bool = False
    async_refresh: bool = False
    refresh_concurrency: int = 3
    refresh_full: bool = False
    days: int | None = None
    min_bars: int | None = None
    min_coverage_pct: float | None = None


class StrategyConfig(StrictConfigModel):
    mode: str = "base"
    strategy: str = DEFAULT_STRATEGY
    strategies: tuple[str, ...] = ()
    benchmark: bool = False
    benchmark_group: str = "baselines"
    long_only: bool = False
    short_only: bool = False
    include_signal_id: tuple[str, ...] = ()
    exclude_signal_id: tuple[str, ...] = ()
    style: RunStyle = "single"
    train_bars: int = 500
    test_bars: int = 100
    step_bars: int = 100
    folds: int = 5
    detail: bool = True
    diagnostics: bool = False
    explain_layers: bool = False

    @field_validator("strategy")
    @classmethod
    def _strategy_choice(cls, value: str) -> str:
        if value not in STRATEGY_CHOICES:
            choices = ", ".join(STRATEGY_CHOICES)
            raise ValueError(f"strategy.strategy={value!r} must be one of: {choices}")
        return value

    @field_validator("benchmark_group")
    @classmethod
    def _benchmark_group_choice(cls, value: str) -> str:
        if value not in BENCHMARK_GROUP_CHOICES:
            choices = ", ".join(BENCHMARK_GROUP_CHOICES)
            raise ValueError(
                f"strategy.benchmark_group={value!r} must be one of: {choices}"
            )
        return value

    @model_validator(mode="after")
    def _sides_are_exclusive(self) -> StrategyConfig:
        if self.long_only and self.short_only:
            raise ValueError("strategy.long_only and strategy.short_only are mutually exclusive")
        return self

    def signal_filters(self) -> SignalDebugFilterConfig:
        return SignalDebugFilterConfig(
            side="long" if self.long_only else "short" if self.short_only else "both",
            include_signal_ids=self.include_signal_id,
            exclude_signal_ids=self.exclude_signal_id,
        )


class SizingConfig(StrictConfigModel):
    normalize: bool = False
    risk_pct: float | None = None
    max_notional_pct: float | None = None
    leverage: float | None = None
    capital: float | None = None
    min_contracts: float | None = None
    max_per_strategy_symbol: int = 0


class RiskConfig(StrictConfigModel):
    max_dd_pct: float | None = None
    max_notional_exposure_pct: float | None = None
    min_trades: int = 0
    min_pf: float = 0.0
    min_expectancy_pct: float | None = None
    min_execution_acceptance_pct: float = 0.0
    fail_on_risk: bool = False


class ExitConfigRequest(StrictConfigModel):
    drawdown_stop_pct: float | None = None
    no_drawdown_stop: bool = False
    max_bars: int = 10
    stop_mult: float = 1.5
    target_mult: float = 1.3
    trail_mult: float = 2.0
    breakeven_after_target: bool = False
    loss_cooldown_bars: int = 0


class DiagnosticsConfig(StrictConfigModel):
    mode: DiagnosticMode = "backtest"
    export: str = ""
    export_dir: str = ""
    baseline_export: str = ""
    variant_export: str = ""


class ModulationEffectConfig(StrictConfigModel):
    base_columns: tuple[str, ...] = ()
    modulator_columns: tuple[str, ...] = ()
    min_base_trades: int = 20
    min_cell_trades: int = 10
    practical_delta_threshold: float = 0.15


class ResearchEvaluationConfig(StrictConfigModel):
    outputs: tuple[ResearchOutputName, ...] = (
        "classifier",
        "tradability",
        "market-state-forward",
        "market-state-modulation",
        "trade-record-modulation",
    )
    include_backtest_report: bool = True
    write_exports: bool = True
    fail_fast: bool = False
    modulation_effect: ModulationEffectConfig = Field(default_factory=ModulationEffectConfig)

    @field_validator("outputs")
    @classmethod
    def _outputs_non_empty(
        cls, value: tuple[ResearchOutputName, ...]
    ) -> tuple[ResearchOutputName, ...]:
        return tuple(dict.fromkeys(value)) or ("classifier",)


class ClassifierConfigRequest(StrictConfigModel):
    profile: ClassifierProfile = "default"
    swing_lookback: int | None = None
    range_lookback: int | None = None
    trend_window: int | None = None
    range_threshold_mode: RangeThresholdMode | None = None
    range_threshold_quantile: float | None = None
    range_threshold_window: int | None = None
    range_threshold_min_samples: int | None = None
    range_threshold_fallback: RangeThresholdFallback | None = None
    fixed_range_width_atr: float | None = None
    level_proximity_atr: float | None = None

    def to_structure_config(self) -> StructureClassifierConfig:
        if self.profile == "fixed":
            config = StructureClassifierConfig.fixed()
        elif self.profile == "rolling":
            config = StructureClassifierConfig.rolling_quantile()
        else:
            config = StructureClassifierConfig.default()

        threshold = config.range_width_threshold
        if self.range_threshold_mode is not None:
            threshold = replace(threshold, mode=self.range_threshold_mode)
        if self.fixed_range_width_atr is not None:
            threshold = replace(threshold, fixed_atr_max=float(self.fixed_range_width_atr))
        if self.range_threshold_quantile is not None:
            threshold = replace(threshold, quantile=float(self.range_threshold_quantile))
        if self.range_threshold_window is not None:
            threshold = replace(threshold, window=int(self.range_threshold_window))
        if self.range_threshold_min_samples is not None:
            threshold = replace(threshold, min_samples=int(self.range_threshold_min_samples))
        if self.range_threshold_fallback is not None:
            threshold = replace(threshold, fallback=self.range_threshold_fallback)

        updates: dict[str, object] = {"range_width_threshold": threshold}
        if self.swing_lookback is not None:
            updates["swing_lookback"] = int(self.swing_lookback)
        if self.range_lookback is not None:
            updates["range_lookback"] = int(self.range_lookback)
        if self.trend_window is not None:
            updates["trend_window"] = int(self.trend_window)
        if self.level_proximity_atr is not None:
            updates["level_proximity_atr"] = float(self.level_proximity_atr)
        return replace(config, **updates)

    def summary(self, resolved: StructureClassifierConfig | None = None) -> str:
        config = resolved or self.to_structure_config()
        threshold = config.range_width_threshold
        return (
            f"classifier=swing_lookback={config.swing_lookback} "
            f"range_lookback={config.range_lookback} trend_window={config.trend_window} "
            f"range_threshold_mode={threshold.mode} fixed_atr={threshold.fixed_atr_max:.2f} "
            f"quantile={threshold.quantile:.2f} window={threshold.window} "
            f"min_samples={threshold.min_samples} fallback={threshold.fallback} "
            f"level_proximity_atr={config.level_proximity_atr:.2f}"
        )


class MarketStateRobustnessConfig(StrictConfigModel):
    se_method: ModulationSeMethod = "iid"
    fdr: bool = False
    fdr_alpha: float = 0.1
    cohens_d_threshold: float = 0.2
    n_eff_min: float = 20.0


class MarketStateConfig(StrictConfigModel):
    horizons: tuple[int, ...] = (3, 5, 10)
    outcomes: tuple[str, ...] = ("return_pct",)
    min_rows: int = 30
    min_base_rows: int = 100
    min_cell_rows: int = 30
    delta_threshold_pct: float = 0.15
    delta_mode: MarketStateDeltaMode = "fixed"
    cost_pct: float = 0.10
    cost_multiple: float = 1.5
    base_columns: tuple[str, ...] = ()
    modulator_columns: tuple[str, ...] = ()
    time_splits: int = 2
    min_segment_months: int = 6
    min_segment_base_rows: int = 50
    min_segment_cell_rows: int = 15
    plot_dir: str = ""
    robustness: MarketStateRobustnessConfig = Field(default_factory=MarketStateRobustnessConfig)

    @field_validator("horizons")
    @classmethod
    def _horizons_positive(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(horizon <= 0 for horizon in value):
            raise ValueError("market_state.horizons values must be positive")
        return tuple(dict.fromkeys(value)) or (3, 5, 10)

    @field_validator("outcomes")
    @classmethod
    def _outcomes_non_empty(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return value or ("return_pct",)

    def delta_threshold(self) -> tuple[float, str]:
        if self.delta_mode == "cost_multiple":
            threshold = self.cost_pct * self.cost_multiple
            return (
                threshold,
                f"cost_multiple cost_pct={self.cost_pct:.2f} multiple={self.cost_multiple:.2f}",
            )
        return self.delta_threshold_pct, "fixed"


class ResearchCommandConfig(StrictConfigModel):
    run: RunConfig = Field(default_factory=RunConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    sizing: SizingConfig = Field(default_factory=SizingConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    exit: ExitConfigRequest = Field(default_factory=ExitConfigRequest)
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)
    classifier: ClassifierConfigRequest = Field(default_factory=ClassifierConfigRequest)
    market_state: MarketStateConfig = Field(default_factory=MarketStateConfig)
    research_evaluation: ResearchEvaluationConfig = Field(
        default_factory=ResearchEvaluationConfig
    )

    @property
    def days(self) -> int:
        if self.run.profile == "smoke":
            return int(self.cache.days) if self.cache.days is not None else 90
        return int(self.cache.days) if self.cache.days is not None else 730

    @property
    def min_bars(self) -> int:
        if self.run.profile == "smoke":
            return int(self.cache.min_bars) if self.cache.min_bars is not None else 1000
        return int(self.cache.min_bars) if self.cache.min_bars is not None else 12000

    @property
    def min_coverage_pct(self) -> float:
        if self.cache.min_coverage_pct is not None:
            return float(self.cache.min_coverage_pct)
        if self.run.profile in ("research", "safe"):
            return 90.0
        return 0.0

    @property
    def sizing_overrides(self) -> SizingOverrideConfig:
        sizing = SizingOverrideConfig(
            normalize=self.sizing.normalize or self.run.profile == "safe",
            risk_pct=self.sizing.risk_pct,
            max_notional_pct=self.sizing.max_notional_pct,
            leverage=self.sizing.leverage,
            capital=self.sizing.capital,
            min_contracts=self.sizing.min_contracts,
        )
        if not sizing.normalize:
            return sizing
        return SizingOverrideConfig(
            normalize=True,
            risk_pct=0.02 if sizing.risk_pct is None else sizing.risk_pct,
            max_notional_pct=1.0 if sizing.max_notional_pct is None else sizing.max_notional_pct,
            leverage=2.0 if sizing.leverage is None else sizing.leverage,
            capital=sizing.capital,
            min_contracts=sizing.min_contracts,
        )

    @property
    def risk_gates(self) -> RiskGateConfig:
        max_dd = self.risk.max_dd_pct
        max_notional = self.risk.max_notional_exposure_pct
        min_trades = int(self.risk.min_trades)
        min_pf = float(self.risk.min_pf)
        min_expectancy = self.risk.min_expectancy_pct
        min_execution_acceptance = float(self.risk.min_execution_acceptance_pct or 0.0)
        if self.run.profile == "safe":
            max_dd = 40.0 if max_dd is None else max_dd
            max_notional = 200.0 if max_notional is None else max_notional
            min_trades = max(min_trades, 30)
            min_pf = max(min_pf, 1.10)
            min_expectancy = 0.0 if min_expectancy is None else min_expectancy
            min_execution_acceptance = max(min_execution_acceptance, 30.0)
        return RiskGateConfig(
            min_coverage_pct=self.min_coverage_pct,
            max_dd_pct=float(max_dd) if max_dd is not None else None,
            max_notional_exposure_pct=float(max_notional) if max_notional is not None else None,
            min_trades=min_trades,
            min_pf=min_pf,
            min_expectancy_pct=float(min_expectancy) if min_expectancy is not None else None,
            min_execution_acceptance_pct=min_execution_acceptance,
            fail_on_risk=self.risk.fail_on_risk,
        )

    @property
    def signal_filters(self) -> SignalDebugFilterConfig:
        return self.strategy.signal_filters()

    @property
    def max_per_strategy_symbol(self) -> int:
        configured = int(self.sizing.max_per_strategy_symbol or 0)
        if configured > 0:
            return configured
        return 1 if self.run.profile == "safe" else 3

    def pairs(self) -> tuple[PairConfig, ...]:
        exclude = set(self.run.exclude_symbol)
        pairs = [
            pair for pair in universe_pairs(self.run.universe) if pair.asset.symbol not in exclude
        ]
        if self.run.symbol:
            pairs = [
                pair
                for pair in pairs
                if pair.asset.symbol == self.run.symbol or pair.asset.sig_symbol == self.run.symbol
            ]
        return tuple(apply_sizing_overrides(pair, self.sizing_overrides) for pair in pairs)

    def metadata(self) -> tuple[str, ...]:
        return (
            f"profile={self.run.profile}",
            f"days={self.days}",
            f"min_bars={self.min_bars}",
            f"universe={self.run.universe}",
            f"data_source={self.run.data_source}",
            f"style={self.strategy.style}",
            f"max_per_strategy_symbol={self.max_per_strategy_symbol}",
            *self.sizing_overrides.metadata(),
            *self.signal_filters.metadata(),
        )


def load_research_command_config(path: Path) -> ResearchCommandConfig:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return ResearchCommandConfig.model_validate(data)


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
