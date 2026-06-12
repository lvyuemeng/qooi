"""Deterministic source-state scoring and research review decisions.

This module owns the single supported potential-scanner source evaluation path:
known-at-close Polars source frames become explainable source states, and source
states may authorize only research report groups. Kline/transition evidence is
multi-timeframe; books, trades, derivatives, and context are source-native
temporal evidence, not separate model scales or trading signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import polars as pl

from qooi.exchange.store import HistoryCoverage
from qooi.scanner import (
    DERIVATIVE_FAMILIES,
    PotentialScanConfig,
    ScanDecision,
    SourceStateRow,
    SymbolStateBundle,
    TransitionInsight,
    best_transition_pattern,
    float_or_none,
    float_value,
    fmt,
    max_timestamp,
    missing_state,
    transition_consensus_passes,
)
from qooi.scanner.classifiers import StateDirection
from qooi.sources.context import SourceAvailability, SourceContextResult


def compute_kline_states(
    config: PotentialScanConfig,
    symbols: tuple[str, ...],
    state_frames: dict[tuple[str, str], pl.DataFrame],
    frames: dict[tuple[str, str], pl.DataFrame],
    coverage: tuple[HistoryCoverage, ...],
) -> dict[str, SourceStateRow]:
    coverage_by_symbol = {item.inst_id: item for item in coverage if item.bar == config.bar}
    states: dict[str, SourceStateRow] = {}
    for symbol in symbols:
        frame = frames.get((symbol, config.bar), pl.DataFrame())
        item = coverage_by_symbol.get(symbol)
        if frame.is_empty() or item is None or item.actual_bars <= 0:
            states[symbol] = SourceStateRow(
                symbol,
                "kline",
                None,
                "missing_bars",
                "blocked",
                0.0,
                "bar frame missing or empty",
                "bars_missing",
                False,
            )
            continue
        state_frame = state_frames.get((symbol, config.bar), pl.DataFrame())
        if state_frame.is_empty():
            states[symbol] = missing_state(symbol, "kline", "kline_state_missing", blocked=True)
            continue
        state_latest = state_frame.tail(1)
        state_key = str(state_latest.select("state_key").item() or "unknown|unknown")
        state_parts = state_key.split("|")
        stage = state_parts[0] if state_parts else "unknown"
        structure = state_parts[1] if len(state_parts) > 1 else "unknown"
        range_state = state_parts[2] if len(state_parts) > 2 else "range_unknown"
        vol_state = state_parts[3] if len(state_parts) > 3 else "vol_unknown"
        context_event = str(state_latest.select("context_event").item() or "unknown")
        direction = str(state_latest.select("direction_hint").item() or "missing")
        confidence = 0.8 if direction in {"bullish", "bearish"} else 0.45
        evidence = (
            f"stage={stage}; structure={structure}; event={context_event}; "
            f"state_key={state_key}; range_state={range_state}; vol_state={vol_state}; "
            f"close={fmt(frame.tail(1).select('close').item())}"
        )
        timestamp = state_latest.select("timestamp").item()
        states[symbol] = SourceStateRow(
            symbol,
            "kline",
            int(timestamp) if timestamp is not None else None,
            f"{structure}/{stage}",
            cast(StateDirection, direction),
            confidence,
            evidence,
            "" if direction != "blocked" else context_event,
            False,
        )
    return states


def compute_source_states(
    config: PotentialScanConfig,
    symbols: tuple[str, ...],
    kline_states: dict[str, SourceStateRow],
    transitions: dict[str, TransitionInsight],
    coverage: tuple[HistoryCoverage, ...],
    context: SourceContextResult,
) -> tuple[SymbolStateBundle, ...]:
    coverage_notes = {item.inst_id: item.notes for item in coverage}
    source_frames = _source_frames_by_family_symbol(context)
    source_availability = {(row.symbol, row.family): row for row in context.availability}
    bundles: list[SymbolStateBundle] = []
    for symbol in symbols:
        insight = transitions.get(
            symbol,
            TransitionInsight(
                symbol, missing_state(symbol, "transition", "transition_pattern_missing"), ()
            ),
        )
        bundles.append(
            SymbolStateBundle(
                symbol=symbol,
                kline=kline_states[symbol],
                transition=insight.current,
                books=_book_state(config, symbol, source_frames, source_availability),
                trades=_trade_state(config, symbol, source_frames, source_availability),
                derivatives=_derivative_state(config, symbol, source_frames),
                context=_message_context_state(config, symbol, source_frames),
                coverage_notes=coverage_notes.get(symbol, ()),
                transition_patterns=insight.patterns,
            )
        )
    return tuple(bundles)


def _source_frames_by_family_symbol(
    context: SourceContextResult,
) -> dict[tuple[str, str], pl.DataFrame]:
    frames: dict[tuple[str, str], pl.DataFrame] = {}
    for family, frame in context.frames.items():
        if frame.is_empty() or "symbol" not in frame.columns:
            continue
        for key, symbol_frame in frame.partition_by("symbol", as_dict=True).items():
            symbol = key[0] if isinstance(key, tuple) else key
            frames[(family, str(symbol))] = symbol_frame
    return frames


def _book_state(
    config: PotentialScanConfig,
    symbol: str,
    source_frames: dict[tuple[str, str], pl.DataFrame],
    source_availability: dict[tuple[str, str], SourceAvailability],
) -> SourceStateRow:
    availability = source_availability.get((symbol, "books"))
    if availability is not None and availability.status == "disabled":
        return missing_state(symbol, "books", "books_disabled", blocked=True)
    frame = source_frames.get(("books", symbol), pl.DataFrame())
    if frame.is_empty():
        return missing_state(symbol, "books", "books_missing")
    row = frame.sort("timestamp").tail(1)
    bid = float_value(row.select("ob_bid_price").item())
    ask = float_value(row.select("ob_ask_price").item())
    mid = (bid + ask) / 2.0 if bid and ask else 0.0
    spread_bps = ((ask - bid) / mid * 10_000.0) if mid > 0.0 else None
    imbalance = _first_float(row, ("ob_imbalance_25", "ob_imbalance_10", "ob_imbalance_5"))
    bid_depth = _first_float(row, ("ob_bid_vol_25", "ob_bid_vol_10", "ob_bid_vol_5", "ob_bid_vol"))
    ask_depth = _first_float(row, ("ob_ask_vol_25", "ob_ask_vol_10", "ob_ask_vol_5", "ob_ask_vol"))
    if spread_bps is not None and spread_bps > 80.0:
        direction: StateDirection = "blocked"
        state = "liquidity_fragile"
    elif imbalance is not None and imbalance >= 0.2:
        direction = "bullish"
        state = "bid_support"
    elif imbalance is not None and imbalance <= -0.2:
        direction = "bearish"
        state = "ask_pressure"
    else:
        direction = "neutral"
        state = "balanced_book"
    evidence = (
        f"imbalance={fmt(imbalance)}; spread_bps={fmt(spread_bps)}; "
        f"bid_depth={fmt(bid_depth)}; ask_depth={fmt(ask_depth)}"
    )
    timestamp = row.select("timestamp").item()
    return SourceStateRow(
        symbol,
        "books",
        int(timestamp) if timestamp is not None else None,
        state,
        direction,
        0.65,
        evidence,
        "",
        False,
    )


def _trade_state(
    config: PotentialScanConfig,
    symbol: str,
    source_frames: dict[tuple[str, str], pl.DataFrame],
    source_availability: dict[tuple[str, str], SourceAvailability],
) -> SourceStateRow:
    availability = source_availability.get((symbol, "trades"))
    if availability is not None and availability.status == "disabled":
        return missing_state(symbol, "trades", "trades_disabled", blocked=True)
    frame = source_frames.get(("trades", symbol), pl.DataFrame())
    if frame.is_empty():
        return missing_state(symbol, "trades", "trades_missing")
    rows = frame.sort("timestamp").tail(max(20, min(config.trade_limit, 100)))
    buy = _sum_side(rows, "buy")
    sell = _sum_side(rows, "sell")
    ratio = buy / sell if sell > 0.0 else (buy if buy > 0.0 else 1.0)
    if ratio >= 1.25:
        direction: StateDirection = "bullish"
        state = "aggressive_buy_dominance"
    elif ratio <= 0.8:
        direction = "bearish"
        state = "aggressive_sell_dominance"
    else:
        direction = "neutral"
        state = "balanced_trade_flow"
    timestamp = rows.tail(1).select("timestamp").item()
    evidence = f"buy_notional={buy:.2f}; sell_notional={sell:.2f}; buy_sell_ratio={ratio:.2f}"
    return SourceStateRow(
        symbol,
        "trades",
        int(timestamp) if timestamp is not None else None,
        state,
        direction,
        0.6,
        evidence,
        "",
        False,
    )


def _derivative_state(
    config: PotentialScanConfig,
    symbol: str,
    source_frames: dict[tuple[str, str], pl.DataFrame],
) -> SourceStateRow:
    funding = source_frames.get(("funding", symbol), pl.DataFrame())
    oi = source_frames.get(("open_interest", symbol), pl.DataFrame())
    taker = source_frames.get(("taker_volume", symbol), pl.DataFrame())
    ratios = source_frames.get(("long_short_ratios", symbol), pl.DataFrame())
    missing = []
    for family, frame in (
        ("funding", funding),
        ("open_interest", oi),
        ("taker_volume", taker),
        ("long_short_ratios", ratios),
    ):
        if family in config.disabled_sources:
            missing.append(f"{family}_disabled")
        elif frame.is_empty():
            missing.append(f"{family}_missing")
    if len(missing) == len(DERIVATIVE_FAMILIES):
        return missing_state(symbol, "derivatives", ";".join(missing))
    funding_latest = _latest_float(funding, "funding_rate")
    oi_delta = _tail_delta(oi, "open_interest_usd", "open_interest")
    buy = _latest_float(taker, "taker_buy_volume")
    sell = _latest_float(taker, "taker_sell_volume")
    taker_ratio = buy / sell if buy is not None and sell and sell > 0.0 else None
    position_ratio = _latest_float(ratios, "top_trader_long_short_position_ratio")
    bullish = (
        oi_delta is not None and oi_delta > 0.0 and taker_ratio is not None and taker_ratio > 1.1
    )
    bearish = (
        oi_delta is not None and oi_delta > 0.0 and taker_ratio is not None and taker_ratio < 0.9
    )
    if bullish:
        direction: StateDirection = "bullish"
        state = "oi_expansion_buy_pressure"
    elif bearish:
        direction = "bearish"
        state = "oi_expansion_sell_pressure"
    else:
        direction = "neutral"
        state = "mixed_derivatives"
    evidence = (
        f"funding={fmt(funding_latest)}; oi_delta={fmt(oi_delta)}; "
        f"taker_buy_sell_ratio={fmt(taker_ratio)}; top_position_ratio={fmt(position_ratio)}; "
        f"missing={','.join(missing) if missing else 'none'}"
    )
    timestamp = max(
        (
            ts
            for ts in (
                max_timestamp(funding),
                max_timestamp(oi),
                max_timestamp(taker),
                max_timestamp(ratios),
            )
            if ts is not None
        ),
        default=None,
    )
    return SourceStateRow(
        symbol, "derivatives", timestamp, state, direction, 0.55, evidence, ";".join(missing), False
    )


def _message_context_state(
    config: PotentialScanConfig,
    symbol: str,
    source_frames: dict[tuple[str, str], pl.DataFrame],
) -> SourceStateRow:
    if "messages" in config.disabled_sources:
        return missing_state(symbol, "context", "messages_disabled", blocked=True)
    messages = source_frames.get(("messages", symbol), pl.DataFrame())
    classifications = source_frames.get(("message_classifications", symbol), pl.DataFrame())
    if messages.is_empty() and classifications.is_empty():
        return missing_state(symbol, "context", "messages_missing")
    rows = classifications if not classifications.is_empty() else messages
    latest = rows.sort("timestamp").tail(1)
    timestamp = latest.select("timestamp").item()
    counts = (
        value_counts(classifications, "message_type")
        if not classifications.is_empty()
        else "unclassified"
    )
    return SourceStateRow(
        symbol,
        "context",
        int(timestamp) if timestamp is not None else None,
        "context_available",
        "neutral",
        0.4,
        f"messages={messages.height}; classifications={classifications.height}; types={counts}",
        "",
        False,
    )


def scan_review_decisions(
    config: PotentialScanConfig, bundles: tuple[SymbolStateBundle, ...]
) -> list[ScanDecision]:
    decisions: list[ScanDecision] = []
    for bundle in bundles:
        transition_state = bundle.transition
        best_pattern = best_transition_pattern(config, bundle.transition_patterns)
        transition_supported = best_pattern is not None and transition_state.direction in {
            "bullish",
            "bearish",
        }
        consensus_supported = transition_supported and transition_consensus_passes(
            bundle.transition_patterns, transition_state.direction
        )
        families = (bundle.kline, bundle.books, bundle.trades, bundle.derivatives, bundle.context)
        missing = tuple(row.family for row in families if row.direction in {"missing", "blocked"})
        contradictions = _contradictions(bundle)

        kline_blocked = bundle.kline.direction in {"missing", "blocked"}
        transition_missing = transition_state.direction in {"missing", "neutral"}
        unsupported_dir = (
            bundle.transition.direction in {"bullish", "bearish"} and not transition_supported
        )
        no_consensus = transition_supported and not consensus_supported
        many_conflicts = len(contradictions) >= 2
        context_required = (
            config.require_context_for_review and bundle.context.direction == "missing"
        )

        source_families = (bundle.books, bundle.trades, bundle.derivatives, bundle.context)
        confirming = [row for row in source_families if row.direction == transition_state.direction]
        fully_supported = (
            consensus_supported
            and transition_state.direction in {"bullish", "bearish"}
            and confirming
        )

        rules = (
            (
                kline_blocked,
                _D(
                    group="watch" if bundle.kline.direction == "missing" else "blocked",
                    direction="undecided" if bundle.kline.direction == "missing" else "blocked",
                    confidence="low" if bundle.kline.direction == "missing" else "blocked",
                    reason=bundle.kline.missing_reason or "kline_state_missing",
                ),
            ),
            (transition_missing, _D(reason="transition_path_missing_or_neutral")),
            (
                unsupported_dir,
                _D(
                    direction=bundle.transition.direction,
                    reason="transition_quality_below_threshold",
                ),
            ),
            (
                no_consensus,
                _D(
                    direction=bundle.transition.direction,
                    reason="transition_consensus_missing_or_contradicted",
                ),
            ),
            (
                many_conflicts,
                _D(
                    group="blocked",
                    direction="blocked",
                    confidence="blocked",
                    reason="contradictory_source_evidence",
                ),
            ),
            (
                bool(contradictions),
                _D(
                    direction=transition_state.direction,
                    reason="contradictory_source_evidence",
                ),
            ),
            (
                context_required,
                _D(
                    direction=transition_state.direction,
                    reason="context_missing",
                ),
            ),
            (
                fully_supported,
                _D(
                    group=transition_state.direction,
                    direction=transition_state.direction,
                    confidence=confidence(confirming, missing),
                    reason="",
                ),
            ),
            (
                len(missing) >= len(families),
                _D(
                    group="blocked",
                    direction="blocked",
                    confidence="blocked",
                    reason="all_non_kline_sources_missing",
                ),
            ),
            (
                transition_state.direction in {"bullish", "bearish"},
                _D(
                    direction=transition_state.direction,
                    reason="needs_source_confirmation",
                ),
            ),
            (
                True,
                _D(
                    direction="undecided",
                    reason="needs_directional_confirmation",
                ),
            ),
        )
        for condition, d in rules:
            if condition:
                decisions.append(
                    _decision(
                        bundle,
                        d.group,
                        d.direction,
                        d.confidence,
                        missing,
                        contradictions,
                        d.reason,
                    )
                )
                break
    return decisions


@dataclass(frozen=True)
class _D:
    group: str = "watch"
    direction: str = "undecided"
    confidence: str = "low"
    reason: str = ""


def confidence(confirming: list[SourceStateRow], missing: tuple[str, ...]) -> str:
    if len(confirming) >= 2 and len(missing) <= 1:
        return "high"
    if confirming:
        return "medium"
    return "low"


def _sum_side(frame: pl.DataFrame, side: str) -> float:
    if frame.is_empty() or "side" not in frame.columns:
        return 0.0
    value_col = "notional_usd" if "notional_usd" in frame.columns else "size"
    if value_col not in frame.columns:
        return 0.0
    value = frame.filter(pl.col("side") == side).get_column(value_col).drop_nulls().sum()
    return float(value or 0.0)


def _latest_float(frame: pl.DataFrame, column: str) -> float | None:
    if frame.is_empty() or column not in frame.columns:
        return None
    return float_or_none(frame.sort("timestamp").tail(1).get_column(column)[0])


def _tail_delta(frame: pl.DataFrame, preferred: str, fallback: str) -> float | None:
    column = (
        preferred if preferred in frame.columns else fallback if fallback in frame.columns else ""
    )
    if frame.is_empty() or not column:
        return None
    values = frame.sort("timestamp").tail(2).get_column(column).drop_nulls().to_list()
    if len(values) < 2:
        return None
    return float_value(values[-1]) - float_value(values[0])


def _first_float(row: pl.DataFrame, columns: tuple[str, ...]) -> float | None:
    for column in columns:
        if column in row.columns:
            value = float_or_none(row.select(column).item())
            if value is not None:
                return value
    return None


def value_counts(frame: pl.DataFrame, column: str) -> str:
    if frame.is_empty() or column not in frame.columns:
        return "none"
    counted = frame.group_by(column).len().sort("len", descending=True).head(3)
    return ",".join(
        f"{counted.select(column).row(index)[0]}={counted.select('len').row(index)[0]}"
        for index in range(counted.height)
    )


def _contradictions(bundle: SymbolStateBundle) -> tuple[str, ...]:
    thesis = (
        bundle.transition if bundle.transition.direction in {"bullish", "bearish"} else bundle.kline
    )
    if thesis.direction not in {"bullish", "bearish"}:
        return ()
    opposite = "bearish" if thesis.direction == "bullish" else "bullish"
    return tuple(
        row.family
        for row in (bundle.books, bundle.trades, bundle.derivatives, bundle.context)
        if row.direction == opposite
    )


def _decision(
    bundle: SymbolStateBundle,
    group: str,
    direction: str,
    confidence_bucket: str,
    missing: tuple[str, ...],
    contradictions: tuple[str, ...],
    block_reason: str,
) -> ScanDecision:
    review_caveat = "blocked until required research evidence is current"
    if direction == "bullish":
        review_caveat = "bullish evidence weakens below latest classified range support"
    elif direction == "bearish":
        review_caveat = "bearish evidence weakens above latest classified range resistance"
    elif group == "watch":
        review_caveat = "confirmation required before research follow-up"
    return ScanDecision(
        symbol=bundle.symbol,
        group=group,
        direction=direction,
        confidence=confidence_bucket,
        transition_evidence=bundle.transition.evidence,
        structure_evidence=bundle.kline.evidence,
        flow_evidence=bundle.trades.evidence,
        liquidity_evidence=bundle.books.evidence,
        derivatives_evidence=bundle.derivatives.evidence,
        context_evidence=bundle.context.evidence,
        missing_evidence=missing,
        contradictory_evidence=contradictions,
        block_reason=block_reason,
        review_caveat=review_caveat,
    )
