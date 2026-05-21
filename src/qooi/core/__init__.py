"""Pure basket pipeline shared by live trading and backtesting."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

import polars as pl

from qooi.core.basket import (
    ActionKind,
    Basket,
    BasketAction,
    BasketBook,
    BasketSnapshot,
    ExitConfig,
    ExitReason,
    SizingDecision,
    TrailTracker,
    evaluate_exits,
    evaluate_hard_exits,
)
from qooi.core.instruments import PairConfig
from qooi.core.recovery import NoRecovery, RecoveryPolicy
from qooi.core.recovery import evaluate as evaluate_recovery


@dataclass(frozen=True)
class BarMarket:
    close: float
    high: float
    low: float
    atr: float
    timestamp: int
    bar_index: int

    @classmethod
    def from_frame(cls, df: pl.DataFrame, bar_idx: int | None = None) -> BarMarket:
        idx = df.height - 1 if bar_idx is None else bar_idx
        return cls(
            close=float(df["close"][idx]),
            high=float(df["high"][idx]),
            low=float(df["low"][idx]),
            atr=float(df["atr_14"][idx] or 1.0),
            timestamp=int(df["timestamp"][idx]) if "timestamp" in df.columns else idx,
            bar_index=idx,
        )


@dataclass(frozen=True)
class BarSignal:
    position: float = 0.0
    entry: float = 0.0
    exit: bool = False
    strength: float = 1.0
    signal_id: str = ""
    zscore: float | None = None
    zscore_delta: float | None = None
    short_momentum_return: float | None = None
    lower_wick_ratio: float | None = None
    upper_wick_ratio: float | None = None
    volatility_ratio: float | None = None
    trend_return: float | None = None
    adx: float | None = None


@dataclass(frozen=True)
class PipelinePolicy:
    close_on_neutral_signal: bool = False
    flip_policy: Literal["ignore", "close_same_strategy_opposite", "reverse"] = "ignore"
    require_thesis_continuation: bool = False


@dataclass(frozen=True)
class PipelineContext:
    strategy_id: str
    market: BarMarket
    signal: BarSignal = BarSignal()
    policy: PipelinePolicy = PipelinePolicy()


def process_bar(
    df: pl.DataFrame,
    baskets: list[Basket] | BasketBook,
    pair: PairConfig,
    exit_cfg: ExitConfig | None = None,
    recovery_cfg: RecoveryPolicy | None = None,
    *,
    context: PipelineContext | None = None,
) -> list[BasketAction]:
    """Return pure basket action proposals for one bar.

    ``process_bar`` does not mutate baskets or action proposals. The executor decides
    which proposals are accepted, accounts fills, then applies lifecycle state through
    ``BasketBook``.
    """
    exit_cfg = exit_cfg or ExitConfig()
    recovery_cfg = recovery_cfg or NoRecovery()
    market = context.market if context else BarMarket.from_frame(df)
    signal = context.signal if context else BarSignal()
    policy = context.policy if context else PipelinePolicy()
    strategy_id = context.strategy_id if context else "default"
    sym = pair.asset.symbol
    book = baskets if isinstance(baskets, BasketBook) else BasketBook(baskets)
    actions: list[BasketAction] = []
    terminal_ids: set[str] = set()

    for basket in list(book.active_for_strategy(sym, strategy_id)):
        hold_action = _evaluate_hold_thesis(basket, market, signal, policy, book.snapshot(basket))
        if hold_action is not None:
            actions.append(hold_action)
            terminal_ids.add(basket.basket_id)
            continue

        hard_exit_action = _evaluate_hard_basket_exit(
            basket, market, signal, exit_cfg, book.snapshot(basket)
        )
        if hard_exit_action is not None:
            actions.append(hard_exit_action)
            terminal_ids.add(basket.basket_id)
            continue

        recovery_actions = _build_recovery_actions(
            book,
            basket,
            pair,
            market,
            signal,
            book.snapshot(basket),
            evaluate_recovery(
                basket,
                market.close,
                market.atr,
                recovery_cfg,
                basket.recovery_level,
                ct_val=pair.asset.ct_val,
                signal_position=signal.position,
                signal_entry=signal.entry,
                zscore=signal.zscore,
                zscore_delta=signal.zscore_delta,
                short_momentum_return=signal.short_momentum_return,
                lower_wick_ratio=signal.lower_wick_ratio,
                upper_wick_ratio=signal.upper_wick_ratio,
                volatility_ratio=signal.volatility_ratio,
                trend_return=signal.trend_return,
                adx=signal.adx,
            ),
        )
        if recovery_actions:
            actions.extend(recovery_actions)
            terminal_ids.update(
                a.basket_id for a in recovery_actions if a.action == ActionKind.EXIT
            )
            continue

        exit_action = _evaluate_basket_exit(basket, market, signal, exit_cfg, book.snapshot(basket))
        if exit_action is not None:
            actions.append(exit_action)
            terminal_ids.add(basket.basket_id)

    entry_actions = _build_flip_and_entry_actions(
        book,
        sym,
        strategy_id,
        pair,
        market,
        signal,
        policy,
        terminal_ids,
    )
    actions.extend(entry_actions)
    return actions


def _evaluate_hold_thesis(
    basket: Basket,
    market: BarMarket,
    signal: BarSignal,
    policy: PipelinePolicy,
    snapshot: BasketSnapshot,
) -> BasketAction | None:
    if signal.exit:
        return _exit_action(basket, market, signal, snapshot, ExitReason.STRATEGY_EXIT.value)
    if policy.close_on_neutral_signal and signal.position == 0:
        return _exit_action(basket, market, signal, snapshot, ExitReason.SIGNAL_ZERO.value)
    if policy.require_thesis_continuation:
        direction = 1 if basket.side == "buy" else -1
        if signal.position * direction <= 0:
            return _exit_action(basket, market, signal, snapshot, ExitReason.THESIS_FAILED.value)
    return None


def _build_recovery_actions(
    book: BasketBook,
    basket: Basket,
    pair: PairConfig,
    market: BarMarket,
    signal: BarSignal,
    snapshot: BasketSnapshot,
    proposals: list[BasketAction],
) -> list[BasketAction]:
    actions: list[BasketAction] = []
    for proposal in proposals:
        if proposal.action == ActionKind.EXIT:
            actions.append(_exit_action(basket, market, signal, snapshot, proposal.reason))
        elif proposal.action == ActionKind.ENTER:
            sz, stop_px, target_px, sizing = _size_recovery_proposal(
                book, basket, pair, market, signal, proposal
            )
            actions.append(
                BasketAction(
                    basket_id=proposal.basket_id,
                    action=ActionKind.ENTER,
                    side=proposal.side,
                    sz=sz,
                    px=proposal.px or market.close,
                    entry_px=proposal.entry_px or proposal.px or market.close,
                    stop_px=stop_px,
                    target_px=target_px,
                    reason=proposal.reason,
                    signal_id=proposal.signal_id or signal.signal_id,
                    signal_strength=signal.strength,
                    sizing=sizing,
                    snapshot=snapshot,
                )
            )
        else:
            sz, stop_px, target_px, sizing = _size_recovery_proposal(
                book, basket, pair, market, signal, proposal
            )
            actions.append(
                BasketAction(
                    basket_id=proposal.basket_id,
                    action=proposal.action,
                    side=proposal.side,
                    sz=sz,
                    px=proposal.px or market.close,
                    entry_px=proposal.entry_px or proposal.px or market.close,
                    stop_px=stop_px,
                    target_px=target_px,
                    reason=proposal.reason,
                    signal_id=proposal.signal_id or signal.signal_id,
                    signal_strength=signal.strength,
                    sizing=sizing,
                    snapshot=snapshot,
                )
            )
    return actions


def _size_recovery_proposal(
    book: BasketBook,
    basket: Basket,
    pair: PairConfig,
    market: BarMarket,
    signal: BarSignal,
    proposal: BasketAction,
) -> tuple[float, float, float, SizingDecision | None]:
    stop_px, target_px = _recovery_stop_target(book, basket, pair, market, proposal)
    sizing = proposal.sizing or _recovery_sizing(
        book,
        basket,
        pair,
        market,
        signal,
        proposal,
        stop_px,
    )
    sz = sizing.contracts if sizing is not None else proposal.sz
    return sz, stop_px, target_px, sizing


def _recovery_sizing(
    book: BasketBook,
    basket: Basket,
    pair: PairConfig,
    market: BarMarket,
    signal: BarSignal,
    proposal: BasketAction,
    stop_px: float,
) -> SizingDecision:
    if proposal.action not in {ActionKind.ADD_GRID, ActionKind.HEDGE, ActionKind.ENTER}:
        return SizingDecision(0, 0.0, 0.0, 0.0, 0.0, 0.0, "none", "not_sizable")

    entry_px = proposal.entry_px or proposal.px or market.close
    sizing = book.policy.size_decision(
        entry_px,
        stop_px,
        pair.asset,
        signal_strength=signal.strength,
    )
    requested_sz = _round_down_to_lot(proposal.sz, pair.asset)
    if sizing.contracts <= 0 or requested_sz <= 0:
        min_reason = f"below_min_contracts_{pair.asset.min_contracts:g}"
        return replace(
            sizing,
            contracts=0.0,
            blocked_reason=sizing.blocked_reason or min_reason,
        )
    capped_sz = _round_down_to_lot(min(requested_sz, sizing.contracts), pair.asset)
    if capped_sz + 1e-9 < float(getattr(pair.asset, "min_contracts", 1.0)):
        return replace(
            sizing,
            contracts=0.0,
            blocked_reason=f"below_min_contracts_{pair.asset.min_contracts:g}",
        )
    return replace(sizing, contracts=capped_sz)


def _recovery_stop_target(
    book: BasketBook,
    basket: Basket,
    pair: PairConfig,
    market: BarMarket,
    proposal: BasketAction,
) -> tuple[float, float]:
    side = proposal.side or basket.side
    entry_px = proposal.entry_px or proposal.px or market.close
    stop_px = proposal.stop_px
    target_px = proposal.target_px
    if proposal.action == ActionKind.ADD_GRID:
        stop_px = _valid_recovery_stop(side, entry_px, basket.stop_px)
        target_px = target_px or basket.target_px
    if stop_px <= 0:
        stop_px, computed_target_px = book.policy.compute_stop_target(
            side,
            entry_px,
            market.atr,
            pair.asset,
        )
        target_px = target_px or computed_target_px
    return stop_px, target_px


def _valid_recovery_stop(side: str, entry_px: float, stop_px: float) -> float:
    if stop_px <= 0:
        return 0.0
    if side == "buy" and stop_px < entry_px:
        return stop_px
    if side == "sell" and stop_px > entry_px:
        return stop_px
    return 0.0


def _round_down_to_lot(size: float, cfg) -> float:
    lot_size = max(float(getattr(cfg, "lot_size", 1.0)), 1e-9)
    lots = math.floor(max(float(size or 0.0), 0.0) / lot_size + 1e-9)
    return round(lots * lot_size, 10)


def _evaluate_basket_exit(
    basket: Basket,
    market: BarMarket,
    signal: BarSignal,
    exit_cfg: ExitConfig,
    snapshot: BasketSnapshot,
) -> BasketAction | None:
    trail = TrailTracker(
        trail_high=basket.trail_high,
        trail_low=basket.trail_low,
        target_hit=basket.target_hit,
    )
    action = evaluate_exits(
        basket,
        market.close,
        market.high,
        market.low,
        market.atr,
        trail,
        exit_cfg,
        skip_trailing=basket.recovery_activated and basket.recovery_level > 0,
    )
    if action is None:
        return None
    return _exit_action(basket, market, signal, snapshot, action.reason, px=action.px)


def _evaluate_hard_basket_exit(
    basket: Basket,
    market: BarMarket,
    signal: BarSignal,
    exit_cfg: ExitConfig,
    snapshot: BasketSnapshot,
) -> BasketAction | None:
    action = evaluate_hard_exits(
        basket,
        market.close,
        market.high,
        market.low,
        market.atr,
        exit_cfg,
    )
    if action is None:
        return None
    return _exit_action(basket, market, signal, snapshot, action.reason, px=action.px)


def _build_flip_and_entry_actions(
    book: BasketBook,
    symbol: str,
    strategy_id: str,
    pair: PairConfig,
    market: BarMarket,
    signal: BarSignal,
    policy: PipelinePolicy,
    terminal_ids: set[str],
) -> list[BasketAction]:
    if signal.entry == 0:
        return []
    actions: list[BasketAction] = []
    side = "buy" if signal.entry > 0 else "sell"
    if policy.flip_policy in ("close_same_strategy_opposite", "reverse"):
        for basket in list(book.active_for_strategy(symbol, strategy_id)):
            if basket.basket_id in terminal_ids or basket.side == side:
                continue
            snapshot = book.snapshot(basket)
            actions.append(
                _exit_action(basket, market, signal, snapshot, ExitReason.SIGNAL_FLIP.value)
            )
            terminal_ids.add(basket.basket_id)

    if not book.can_open(symbol, strategy_id) and not terminal_ids:
        return actions

    stop_px, target_px = book.policy.compute_stop_target(side, market.close, market.atr, pair.asset)
    sizing = book.policy.size_decision(
        market.close,
        stop_px,
        pair.asset,
        signal_strength=signal.strength,
    )
    if sizing.contracts <= 0:
        return actions
    direction = "long" if side == "buy" else "short"
    sequence = len(book.baskets) + len(actions) + 1
    basket_id = f"{symbol}-{strategy_id}-{direction}-{market.timestamp}-{sequence}"
    snapshot = BasketSnapshot(
        basket_id=basket_id,
        symbol=symbol,
        strategy=strategy_id,
        side=side,
        entry_px=market.close,
        current_sz=sizing.contracts,
        stop_px=stop_px,
        target_px=target_px,
        bars_in_pos=0,
        recovery_level=0,
        recovery_activated=False,
    )
    actions.append(
        BasketAction(
            basket_id=basket_id,
            action=ActionKind.ENTER,
            side=side,
            sz=sizing.contracts,
            px=market.close,
            entry_px=market.close,
            stop_px=stop_px,
            target_px=target_px,
            reason=ExitReason.SIGNAL_ENTRY.value,
            signal_id=signal.signal_id,
            signal_strength=signal.strength,
            sizing=sizing,
            snapshot=snapshot,
        )
    )
    return actions


def _exit_action(
    basket: Basket,
    market: BarMarket,
    signal: BarSignal,
    snapshot: BasketSnapshot,
    reason: str,
    *,
    px: float = 0.0,
) -> BasketAction:
    return BasketAction(
        basket_id=basket.basket_id,
        action=ActionKind.EXIT,
        side=basket.side,
        sz=basket.current_sz,
        px=px or market.close,
        entry_px=basket.entry_px,
        stop_px=basket.stop_px,
        target_px=basket.target_px,
        reason=reason,
        fraction=1.0,
        bars_held=basket.bars_in_pos,
        signal_id=signal.signal_id,
        signal_strength=signal.strength,
        snapshot=snapshot,
    )
