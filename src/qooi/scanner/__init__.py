"""Potential scanner package shared expressions and contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import polars as pl

from qooi.exchange.store import HistoryCoverage
from qooi.scanner.classifiers import StateDirection
from qooi.scanner.config import EvidenceConfig, ReviewConfig, SourceConfig, TransitionConfig
from qooi.sources.context import SourceContextResult


def pct_change_expr(next_col: str, base_col: str) -> pl.Expr:
    base = pl.when(pl.col(base_col).abs() > 1e-12).then(pl.col(base_col)).otherwise(None)
    return (pl.col(next_col) - pl.col(base_col)) / base * 100.0


def outcome_bucket_expr(return_threshold_pct: float) -> pl.Expr:
    return (
        pl.when(pl.col("forward_return_pct") > return_threshold_pct)
        .then(pl.lit("up"))
        .when(pl.col("forward_return_pct") < -return_threshold_pct)
        .then(pl.lit("down"))
        .otherwise(pl.lit("flat"))
    )


def entropy_term(col: str) -> pl.Expr:
    probability = pl.col(col).cast(pl.Float64)
    return pl.when(probability > 0.0).then(-probability * probability.log(2)).otherwise(0.0)


def entropy_expr(up_col: str, down_col: str, flat_col: str) -> pl.Expr:
    return entropy_term(up_col) + entropy_term(down_col) + entropy_term(flat_col)


DERIVATIVE_FAMILIES = ("funding", "open_interest", "taker_volume", "long_short_ratios")


class PotentialScanConfig(Protocol):
    output: Path
    symbols: tuple[str, ...]
    universe: str
    bar: str
    timeframes: tuple[str, ...]
    days: int
    refresh_mode: Literal["incremental", "cache_only", "force"]
    fetch_concurrency: int
    source: SourceConfig
    transition: TransitionConfig
    review: ReviewConfig
    evidence: EvidenceConfig


@dataclass(frozen=True)
class PotentialUniverse:
    symbols: tuple[str, ...]
    discovery: pl.DataFrame
    selection_note: str
    missing_reason: str
    eligible_count: int = 0


@dataclass(frozen=True)
class PotentialArtifacts:
    report: Path
    diagnostics_dir: Path
    states_dir: Path


@dataclass(frozen=True)
class BarFetchResult:
    frames: dict[tuple[str, str], pl.DataFrame]
    state_frames: dict[tuple[str, str], pl.DataFrame]
    coverage: tuple[HistoryCoverage, ...]


@dataclass(frozen=True)
class SourceStateRow:
    symbol: str
    family: str
    timestamp: int | None
    state: str
    direction: StateDirection
    confidence: float
    evidence: str
    missing_reason: str
    stale: bool


@dataclass(frozen=True)
class TransitionPattern:
    symbol: str
    timeframe: str
    path: str
    event: str
    count: int
    transition_probability: float
    win_rate: float
    average_forward_return_pct: float
    omega: float
    pwpr: float
    transition_information_bits: float
    conditional_transition_information_bits: float
    direction: StateDirection
    recent_transition_probability: float = 0.0
    long_transition_probability: float = 0.0
    probability_delta: float = 0.0
    p_up: float = 0.0
    p_down: float = 0.0
    median_forward_return_pct: float = 0.0
    q10_forward_return_pct: float = 0.0
    q25_forward_return_pct: float = 0.0
    q75_forward_return_pct: float = 0.0
    q90_forward_return_pct: float = 0.0
    q25_forward_min_return_pct: float = 0.0
    q75_forward_max_return_pct: float = 0.0
    loss_stop_pct: float = 0.0
    profit_stop_pct: float = 0.0
    reward_risk: float = 0.0
    symbol_count: int = 0
    effective_count: float = 0.0
    suggestion: str = "watch"


@dataclass(frozen=True)
class TransitionInsight:
    symbol: str
    current: SourceStateRow
    patterns: tuple[TransitionPattern, ...]
    consensus: str = "none"
    rank_score: float = 0.0


@dataclass(frozen=True)
class TransitionEdge:
    timeframe: str
    prev_state: str
    state: str
    event: str
    count: int
    transition_probability: float
    transition_information_bits: float
    conditional_transition_information_bits: float


@dataclass(frozen=True)
class UnsupportedTransitionPath:
    symbol: str
    timeframe: str
    path: str
    event: str
    reason: str


@dataclass(frozen=True)
class TransitionAnalysis:
    insights: dict[str, TransitionInsight]
    edges: tuple[TransitionEdge, ...]
    unsupported: tuple[UnsupportedTransitionPath, ...]


@dataclass(frozen=True)
class SymbolStateBundle:
    symbol: str
    kline: SourceStateRow
    transition: SourceStateRow
    books: SourceStateRow
    trades: SourceStateRow
    derivatives: SourceStateRow
    context: SourceStateRow
    coverage_notes: tuple[str, ...]
    transition_patterns: tuple[TransitionPattern, ...] = ()


@dataclass(frozen=True)
class ScanDecision:
    symbol: str
    group: str
    direction: str
    confidence: str
    transition_evidence: str
    structure_evidence: str
    flow_evidence: str
    liquidity_evidence: str
    derivatives_evidence: str
    context_evidence: str
    missing_evidence: tuple[str, ...]
    contradictory_evidence: tuple[str, ...]
    block_reason: str
    review_caveat: str


@dataclass
class ReportInputs:
    config: PotentialScanConfig
    artifacts: PotentialArtifacts
    universe: PotentialUniverse
    bars: BarFetchResult
    context: SourceContextResult
    transitions: TransitionAnalysis
    bundles: tuple[SymbolStateBundle, ...]
    decisions: tuple[ScanDecision, ...]
    report_sections: tuple = ()


def missing_state(
    symbol: str, family: str, reason: str, *, blocked: bool = False
) -> SourceStateRow:
    return SourceStateRow(
        symbol=symbol,
        family=family,
        timestamp=None,
        state=reason,
        direction="blocked" if blocked else "missing",
        confidence=0.0,
        evidence=reason,
        missing_reason=reason,
        stale=False,
    )


def float_value(value: object) -> float:
    converted = float_or_none(value)
    return converted if converted is not None else 0.0


def float_or_none(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def fmt(value: object) -> str:
    converted = float_or_none(value)
    return "n/a" if converted is None else f"{converted:.4f}"


def max_timestamp(frame: pl.DataFrame) -> int | None:
    if frame.is_empty() or "timestamp" not in frame.columns:
        return None
    value = frame.get_column("timestamp").drop_nulls().max()
    return int(value) if value is not None else None


def _transition_pattern_passes(config: PotentialScanConfig, pattern: TransitionPattern) -> bool:
    information = max(
        pattern.transition_information_bits,
        pattern.conditional_transition_information_bits,
    )
    directional_probability = pattern.p_up if pattern.direction == "bullish" else pattern.p_down
    tail_loss = pattern.loss_stop_pct
    return (
        pattern.count >= config.transition.min_count
        and pattern.transition_probability >= config.transition.min_probability
        and directional_probability >= config.transition.min_directional_probability
        and pattern.reward_risk >= config.transition.min_reward_risk
        and tail_loss <= config.transition.max_tail_loss_pct
        and information >= config.transition.min_information_bits
        and pattern.direction in {"bullish", "bearish"}
    )


def best_transition_pattern(
    config: PotentialScanConfig, patterns: tuple[TransitionPattern, ...]
) -> TransitionPattern | None:
    eligible = [pattern for pattern in patterns if _transition_pattern_passes(config, pattern)]
    return max(eligible, key=lambda pattern: pattern.transition_probability, default=None)


def transition_consensus_passes(patterns: tuple[TransitionPattern, ...], direction: str) -> bool:
    directional = tuple(
        pattern for pattern in patterns if pattern.direction in {"bullish", "bearish"}
    )
    if not directional:
        return False
    opposite = "bearish" if direction == "bullish" else "bullish"
    confirmations = {pattern.timeframe for pattern in directional if pattern.direction == direction}
    contradictions = {pattern.timeframe for pattern in directional if pattern.direction == opposite}
    return (
        bool(confirmations & {"1H", "4H"})
        and "1D" not in contradictions
        and "15m" not in contradictions
    )


def _rank_transition_symbols(
    config: PotentialScanConfig, transitions: dict[str, TransitionInsight]
) -> tuple[str, ...]:
    ranked = sorted(
        (
            insight
            for insight in transitions.values()
            if best_transition_pattern(config, insight.patterns) is not None
        ),
        key=lambda insight: insight.rank_score,
        reverse=True,
    )
    return tuple(insight.symbol for insight in ranked[: config.transition.context_limit])


def context_symbols(
    config: PotentialScanConfig,
    symbols: tuple[str, ...],
    transitions: dict[str, TransitionInsight],
) -> tuple[str, ...]:
    if config.symbols or config.transition.context_scope == "all_scanned":
        return symbols
    return _rank_transition_symbols(config, transitions)
