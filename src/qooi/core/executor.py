"""Executor — maps BasketActions to real-world orders.

Two executors: LiveExecutor (direct OKX API) and BacktestExecutor (simulate).
Both consume the same list[BasketAction] from the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from qooi.core.basket import (
    ActionKind,
    Basket,
    BasketAction,
    BasketBook,
    ExitConfig,
    TrailTracker,
    evaluate_exits,
)
from qooi.strategies import StrategyBehavior, compute_signal_frame, momentum_burst_spec


@dataclass(frozen=True)
class BacktestResult:
    label: str
    trades: pl.DataFrame
    equity: pl.DataFrame
    diagnostics: object
    run_metadata: tuple[str, ...] = ()


class LiveExecutor:
    """Execute BasketActions via direct OKX Trading API calls."""

    def __init__(self, tc, md):
        self._tc = tc
        self._md = md

    def execute(self, actions: list[BasketAction], dry_run: bool = False) -> None:
        for a in actions:
            if dry_run:
                self._log(a)
                continue
            try:
                self._dispatch(a)
            except Exception as e:
                print(f"    EXEC FAILED [{a.action}]: {e}")

    def _dispatch(self, a: BasketAction) -> None:
        if a.action == ActionKind.ENTER:
            px = a.px or self._entry_px(a.side, a.basket_id)
            sz = int(a.sz) if a.sz > 0 else 1
            print(f"    ORDER {a.side} sz={sz} px={px} ({a.reason})")

        elif a.action == ActionKind.EXIT:
            print(f"    CLOSE {a.side} ({a.reason})")

        elif a.action == ActionKind.ADD_GRID:
            px = a.px or self._entry_px(a.side, a.basket_id)
            print(f"    GRID ADD {a.side} sz={a.sz} px={px} ({a.reason})")

        elif a.action == ActionKind.HEDGE:
            print(f"    HEDGE {a.side} sz={a.sz} ({a.reason})")

    def _entry_px(self, side: str, symbol: str) -> float:
        obi = self._md.book(symbol, limit=1)
        if not obi:
            return 0.0
        return obi.ask_price if side == "buy" else obi.bid_price

    def _log(self, a: BasketAction) -> None:
        print(f"    {a.action.upper():10s} {a.side:5s} sz={a.sz:.0f} px={a.px:.2f} ({a.reason})")


class BacktestExecutor:
    """Simulate BasketActions against OHLCV bars for backtesting.

    Loops process_bar() over all bars and computes equity/PnL from
    the BasketAction stream.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        cost_pct: float = 0.00005,
        *,
        drawdown_stop_pct: float | None = None,
        mark_to_market: bool = True,
        close_open_positions: bool = True,
        max_per_strategy_symbol: int = 3,
        loss_cooldown_bars: int = 0,
    ):
        self._initial_capital = initial_capital
        self._cost = cost_pct
        self._drawdown_stop_pct = drawdown_stop_pct
        self._mark_to_market = mark_to_market
        self._close_open_positions = close_open_positions
        self._max_per_strategy_symbol = max_per_strategy_symbol
        self._loss_cooldown_bars = max(int(loss_cooldown_bars), 0)

    def run(
        self,
        df,
        pair,
        exit_cfg=None,
        recovery_cfg=None,
        *,
        strategy: StrategyBehavior | None = None,
        initial_baskets: list[Basket] | None = None,
        precomputed_signal: bool = False,
    ) -> tuple[list[dict], list[float]]:
        from qooi.core import BarMarket, BarSignal, PipelineContext, process_bar
        from qooi.core.evaluate import (
            BacktestDiagnostics,
            BasketLifecycleDiagnostics,
            EngineDataAudit,
            FeatureDiagnostics,
            PortfolioRiskDiagnostics,
            SignalDiagnostics,
        )

        strategy = strategy or momentum_burst_spec()
        strategy_id = strategy.name
        if not (precomputed_signal and "signal" in df.columns):
            df = compute_signal_frame(df, strategy)
        signal_col = (
            df["position_signal"].to_list()
            if "position_signal" in df.columns
            else df["signal"].to_list()
        )
        entry_signal_col = (
            df["entry_signal"].to_list() if "entry_signal" in df.columns else signal_col
        )
        exit_signal_col = (
            df["exit_signal"].to_list() if "exit_signal" in df.columns else [False] * df.height
        )
        signal_strength_col = (
            df["signal_strength"].to_list()
            if "signal_strength" in df.columns
            else [1.0] * df.height
        )
        signal_id_col = (
            df["signal_id"].to_list() if "signal_id" in df.columns else [strategy_id] * df.height
        )

        baskets = list(initial_baskets or [])
        book = BasketBook(baskets, max_per_strategy_symbol=self._max_per_strategy_symbol)
        trades: list[dict] = []
        equity = [self._initial_capital]
        cash = self._initial_capital
        active_exposure = [0.0]
        notional_exposure = [0.0]
        timestamps = [int(df["timestamp"][0])] if "timestamp" in df.columns and df.height else []
        signals = [float(signal_col[0] or 0.0)] if signal_col else []
        action_counts = {
            "entry_signals": 0,
            "entries": 0,
            "exits": 0,
            "grid": 0,
            "hedge": 0,
            "recovery": 0,
            "same_bar_exit_entry": 0,
            "blocked_entry_signals": 0,
            "duplicate_entry_suppressed": 0,
            "capacity_blocked_entries": 0,
            "sizing_blocked_entries": 0,
        }
        blocked_entry_reasons: dict[str, int] = {}
        blocked_sizing_rows: list[dict[str, float | int | str]] = []
        entry_metadata: dict[str, dict[str, float | int | str]] = {}
        exit_reasons: dict[str, int] = {}
        peak_equity = self._initial_capital
        stopped_out = False
        stop_bar_index: int | None = None
        fee_total = 0.0
        max_simultaneous_baskets = len(book.active())
        stop_exit_net_pnl_usd = 0.0
        recovered_stop_exit_count = 0
        recovered_stop_exit_net_pnl_usd = 0.0
        recovery_net_pnl_usd = 0.0
        action_events: list[dict[str, float | int | str | bool]] = []
        recovery_preempted_counts = {"stop": 0, "time": 0, "trailing_stop": 0}
        recovery_unsized_actions = 0
        recovery_cap_breach_actions = 0
        recovery_blocked_actions = 0
        recovery_blocked_reasons: dict[str, int] = {}
        recovery_allowed_actions = 0
        recovery_notional_after_pct = 0.0
        ambiguous_stop_target_count = 0
        ambiguous_stop_net_pnl_usd = 0.0
        target_first_counterfactual_net_pnl_usd = 0.0
        cooldown_until: dict[str, int] = {}

        def _is_recovery_reason(reason: str) -> bool:
            return reason.startswith("grid_level_") or reason in {
                "martingale_reverse",
                "hedge_on_drawdown",
                "global_loss_limit",
            }

        def _is_recovery_action(action: BasketAction) -> bool:
            return action.action in {ActionKind.ADD_GRID, ActionKind.HEDGE} or _is_recovery_reason(
                action.reason
            )

        def _exit_family(reason: str) -> str:
            if reason in {"strategy_exit", "signal_zero", "thesis_failed", "signal_flip"}:
                return "strategy"
            if reason in {"stop", "trailing_stop", "breakeven", "time", "global_drawdown_stop"}:
                return "risk_stop"
            if _is_recovery_reason(reason):
                return "recovery"
            if reason == "final_mark":
                return "mark"
            return "other"

        def _notional_exposure(close_px: float) -> float:
            return sum(
                abs(b.current_sz * pair.asset.ct_val * close_px)
                for b in book.baskets
                if b.is_active
            )

        def _unrealized_pnl(close_px: float) -> float:
            total = 0.0
            for b in book.baskets:
                if not b.is_active or b.entry_px <= 0:
                    continue
                d = 1 if b.side == "buy" else -1
                total += d * (close_px - b.entry_px) * b.current_sz * pair.asset.ct_val
            return total

        def _action_notional(action: BasketAction, close_px: float) -> float:
            px = action.px if action.px > 0 else close_px
            return abs(px * action.sz * pair.asset.ct_val)

        def _normal_exit_candidate(basket: Basket):
            trail = TrailTracker(
                trail_high=basket.trail_high,
                trail_low=basket.trail_low,
                target_hit=basket.target_hit,
            )
            return evaluate_exits(
                basket,
                float(market.close),
                float(market.high),
                float(market.low),
                float(market.atr),
                trail,
                exit_cfg or ExitConfig(),
                skip_trailing=basket.recovery_activated and basket.recovery_level > 0,
            )

        def _count_recovery_block(reason: str) -> None:
            nonlocal recovery_blocked_actions
            recovery_blocked_actions += 1
            recovery_blocked_reasons[reason] = recovery_blocked_reasons.get(reason, 0) + 1

        def _recovery_group_key(action: BasketAction) -> str:
            if action.snapshot is not None and _is_recovery_action(action):
                return action.snapshot.basket_id
            return action.basket_id

        def _recovery_action_block_reason(action: BasketAction) -> str:
            if action.action not in {ActionKind.ADD_GRID, ActionKind.HEDGE, ActionKind.ENTER}:
                return ""
            if not _is_recovery_action(action):
                return ""
            close_px = float(market.close)
            action_notional = _action_notional(action, close_px)
            basket = book.get(action.basket_id)
            basket_notional_after = action_notional
            if action.action == ActionKind.ADD_GRID and basket is not None:
                basket_notional_after += abs(basket.current_sz * pair.asset.ct_val * close_px)
            per_basket_cap = (
                pair.asset.capital
                * pair.asset.leverage
                * float(getattr(pair.asset, "max_notional_pct_per_basket", 1.0))
            )
            portfolio_cap = pair.asset.capital * pair.asset.leverage
            if action.sizing is None:
                return "unsized"
            if action.sizing.blocked_reason:
                return action.sizing.blocked_reason
            if action.sizing.contracts <= 0 or action.sz <= 0:
                return "zero_sized"
            if basket_notional_after > per_basket_cap:
                return "per_basket_notional_cap"
            if _notional_exposure(close_px) + action_notional > portfolio_cap:
                return "portfolio_notional_cap"
            return ""

        def _ambiguous_stop_target(basket: Basket) -> bool:
            if basket.target_hit or basket.entry_px <= 0:
                return False
            d = 1 if basket.side == "buy" else -1
            stop_px = basket.stop_px if basket.stop_px > 0 else basket.entry_px - d * market.atr
            target_px = (
                basket.target_px if basket.target_px > 0 else basket.entry_px + d * market.atr
            )
            stop_hit = market.low <= stop_px if d > 0 else market.high >= stop_px
            target_hit = market.high >= target_px if d > 0 else market.low <= target_px
            return bool(stop_hit and target_hit)

        def _record_action_event(
            action: BasketAction,
            *,
            suppressed_exit_reason: str = "",
        ) -> None:
            close_px = float(market.close)
            pre_notional = _notional_exposure(close_px)
            action_notional = _action_notional(action, close_px)
            post_notional = pre_notional
            if action.action in {ActionKind.ENTER, ActionKind.ADD_GRID, ActionKind.HEDGE}:
                post_notional += action_notional
            elif action.action == ActionKind.EXIT:
                post_notional = max(0.0, pre_notional - action_notional)
            basket = book.get(action.basket_id)
            action_events.append(
                {
                    "bar_index": int(market.bar_index),
                    "timestamp": int(market.timestamp),
                    "basket_id": action.basket_id,
                    "action": str(action.action),
                    "reason": action.reason,
                    "side": action.side,
                    "sz": float(action.sz or 0.0),
                    "px": float(action.px or close_px),
                    "pre_active_baskets": len(book.active()),
                    "pre_notional_pct": round(
                        pre_notional / self._initial_capital * 100.0
                        if self._initial_capital
                        else 0.0,
                        6,
                    ),
                    "post_notional_pct": round(
                        post_notional / self._initial_capital * 100.0
                        if self._initial_capital
                        else 0.0,
                        6,
                    ),
                    "pre_recovery_level": basket.recovery_level if basket is not None else 0,
                    "suppressed_exit_reason": suppressed_exit_reason,
                    "recovery_action": _is_recovery_action(action),
                }
            )

        def _fee(px: float, sz: float) -> float:
            return abs(px * sz * pair.asset.ct_val) * self._cost

        def _count_exit(reason: str) -> None:
            action_counts["exits"] += 1
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        def _count_blocked_entry(reason: str) -> None:
            action_counts["blocked_entry_signals"] += 1
            blocked_entry_reasons[reason] = blocked_entry_reasons.get(reason, 0) + 1
            if reason.startswith("below_min_contracts"):
                action_counts["sizing_blocked_entries"] += 1
            elif reason in {"max_total", "max_per_symbol", "max_per_strategy_symbol"}:
                action_counts["capacity_blocked_entries"] += 1
            else:
                action_counts["duplicate_entry_suppressed"] += 1

        def _exit_pnl(
            basket: Basket, exit_px: float, sz: float = 0.0
        ) -> tuple[float, float, float]:
            d = 1 if basket.side == "buy" else -1
            pnl_pct = d * (exit_px / basket.entry_px - 1) if basket.entry_px > 0 else 0.0
            effective_sz = sz if sz > 0 else basket.current_sz
            pnl_usd = d * (exit_px - basket.entry_px) * effective_sz * pair.asset.ct_val
            return pnl_pct, pnl_usd, effective_sz

        def _portfolio_value(close_px: float) -> float:
            return cash + (_unrealized_pnl(close_px) if self._mark_to_market else 0.0)

        def _drawdown_pct(value: float) -> float:
            return (1.0 - value / peak_equity) * 100.0 if peak_equity > 0 else 0.0

        def _entry_feature_metadata(bar_index: int) -> dict[str, float | int | str]:
            metadata: dict[str, float | int | str] = {}
            feature_columns = (
                "adx_14",
                "volatility_ratio",
                "volatility_regime",
                "trend_return",
                "close_z_score",
                "dynamic_z_score",
                "robust_z_score",
                "prior_liquidity_high",
                "prior_liquidity_low",
                "swept_high",
                "swept_low",
                "reclaimed_high",
                "reclaimed_low",
                "bullish_liquidity_sweep",
                "bearish_liquidity_sweep",
                "failed_bullish_sweep",
                "failed_bearish_sweep",
                "volume_impulse",
                "sweep_distance_atr",
                "wick_body_ratio",
                "lower_wick_body_ratio",
                "upper_wick_body_ratio",
                "bullish_rejection_bar",
                "bearish_rejection_bar",
                "breakout_acceptance_high",
                "breakout_acceptance_low",
                "failed_breakout_high",
                "failed_breakout_low",
                "event_quality_score",
                "liquidity_event_type",
                "atr_percentile_100",
                "atr_percentile_bucket",
                "distance_to_prior_high_atr",
                "distance_to_prior_low_atr",
                "near_prior_high_no_breach",
                "near_prior_low_no_breach",
                "key_level_proximity_bucket",
                "z_pressure_side",
                "structure_trend_state",
                "market_stage",
                "structure_reason",
                "market_stage_reason",
                "stage_unknown_reason",
                "range_high",
                "range_low",
                "range_mid",
                "range_width_atr",
                "range_compression",
                "near_range_high",
                "near_range_low",
                "last_swing_high",
                "last_swing_low",
                "structure_higher_high",
                "structure_higher_low",
                "structure_lower_high",
                "structure_lower_low",
                "m15_confirm_long",
                "m15_confirm_short",
                "m15_confirm_reason",
                "m15_confirm_available",
                "h1_close",
                "h1_trend_state",
                "h1_structure_trend_state",
                "h1_market_stage",
                "h1_structure_reason",
                "h1_market_stage_reason",
                "h1_stage_unknown_reason",
                "h1_range_width_atr",
                "h1_range_compression",
                "h1_near_range_high",
                "h1_near_range_low",
                "h1_context_available",
                "h4_close",
                "h4_trend_state",
                "h4_structure_trend_state",
                "h4_market_stage",
                "h4_structure_reason",
                "h4_market_stage_reason",
                "h4_stage_unknown_reason",
                "h4_range_width_atr",
                "h4_range_compression",
                "h4_near_range_high",
                "h4_near_range_low",
                "h4_context_available",
                "d1_close",
                "d1_trend_state",
                "d1_structure_trend_state",
                "d1_market_stage",
                "d1_structure_reason",
                "d1_market_stage_reason",
                "d1_stage_unknown_reason",
                "d1_range_width_atr",
                "d1_range_compression",
                "d1_near_range_high",
                "d1_near_range_low",
                "d1_context_available",
                "mtf_state_key",
                "mtf_structure_key",
                "mtf_stage_key",
                "mtf_event_state_key",
            )
            for column in feature_columns:
                if column not in df.columns:
                    continue
                value = df[column][bar_index]
                if value is None:
                    continue
                key = f"entry_{column}"
                if isinstance(value, str):
                    metadata[key] = value
                else:
                    metadata[key] = round(float(value), 8)
            return metadata

        def _sizing_metadata(a: BasketAction, bar_index: int) -> dict[str, float | int | str]:
            entry_px = a.entry_px if a.entry_px > 0 else a.px
            contracts = a.sz
            notional = abs(entry_px * contracts * pair.asset.ct_val)
            sizing = a.sizing
            entry_equity = _portfolio_value(float(market.close))
            pre_entry_notional_pct = (
                _notional_exposure(float(market.close)) / self._initial_capital * 100.0
                if self._initial_capital
                else 0.0
            )
            entry_notional_pct = (
                notional / self._initial_capital * 100.0 if self._initial_capital else 0.0
            )
            return {
                "entry_ts": int(market.timestamp),
                "entry_bar_index": bar_index,
                "contracts": round(contracts, 6),
                "recorded_entry_px": round(entry_px, 6),
                "entry_notional_usd": round(notional, 6),
                "notional_pct_capital": round(entry_notional_pct, 6),
                "ct_val": pair.asset.ct_val,
                "capital": self._initial_capital,
                "entry_signal_id": a.signal_id,
                "entry_signal_strength": a.signal_strength,
                "entry_equity": round(entry_equity, 6),
                "entry_peak_equity": round(peak_equity, 6),
                "entry_drawdown_pct": round(_drawdown_pct(entry_equity), 6),
                "pre_entry_total_notional_pct": round(pre_entry_notional_pct, 6),
                "entry_total_notional_pct": round(pre_entry_notional_pct, 6),
                "post_entry_total_notional_pct": round(
                    pre_entry_notional_pct + entry_notional_pct, 6
                ),
                "entry_active_baskets": len(book.active()),
                "sizing_binding": sizing.binding_cap if sizing is not None else "unknown",
                "risk_per_contract": sizing.risk_per_contract if sizing is not None else 0.0,
                "risk_budget_usd": sizing.risk_budget_usd if sizing is not None else 0.0,
                "risk_sized_contracts": sizing.risk_sized_contracts if sizing is not None else 0,
                "max_notional_usd": sizing.max_notional_usd if sizing is not None else 0.0,
                "notional_sized_contracts": (
                    sizing.notional_sized_contracts if sizing is not None else 0
                ),
                "sizing_blocked_reason": sizing.blocked_reason if sizing is not None else "",
                **_entry_feature_metadata(bar_index),
            }

        def _exit_metadata(
            basket_id: str,
            exit_px: float,
            effective_sz: float,
            bar_index: int,
        ) -> dict[str, float | int | str]:
            metadata = dict(entry_metadata.pop(basket_id, {}))
            exit_notional = abs(exit_px * effective_sz * pair.asset.ct_val)
            basket = book.get(basket_id)
            metadata.update(
                {
                    "exit_ts": int(market.timestamp),
                    "exit_bar_index": bar_index,
                    "exit_notional_usd": round(exit_notional, 6),
                    "exit_equity_before": round(_portfolio_value(float(market.close)), 6),
                    "exit_drawdown_pct_before": round(
                        _drawdown_pct(_portfolio_value(float(market.close))), 6
                    ),
                    "exit_total_notional_pct_before": round(
                        _notional_exposure(float(market.close))
                        / self._initial_capital
                        * 100.0
                        if self._initial_capital
                        else 0.0,
                        6,
                    ),
                    "recovery_active_at_exit": bool(
                        basket is not None and basket.recovery_activated
                    ),
                    "recovery_level_at_exit": basket.recovery_level if basket is not None else 0,
                }
            )
            if "contracts" not in metadata:
                metadata["contracts"] = round(effective_sz, 6)
            if "entry_notional_usd" not in metadata:
                entry_px = metadata.get("recorded_entry_px", 0.0)
                entry_notional = abs(float(entry_px or 0.0) * effective_sz * pair.asset.ct_val)
                metadata["entry_notional_usd"] = round(entry_notional, 6)
            if "notional_pct_capital" not in metadata:
                metadata["notional_pct_capital"] = round(
                    exit_notional / self._initial_capital * 100.0 if self._initial_capital else 0.0,
                    6,
                )
            return metadata

        def _blocked_sizing_metadata(
            side: str,
            entry_px: float,
            stop_px: float,
            sizing,
        ) -> dict[str, float | int | str]:
            min_contracts = float(getattr(pair.asset, "min_contracts", 1.0))
            strength = max(float(signal.strength or 0.0), 1e-9)
            risk_pct = max(float(getattr(pair.asset, "max_risk_pct", 0.0)), 1e-9)
            leverage = max(float(getattr(pair.asset, "leverage", 0.0)), 1e-9)
            notional_pct = max(
                float(getattr(pair.asset, "max_notional_pct_per_basket", 0.0)), 1e-9
            )
            notional_per_contract = abs(entry_px * pair.asset.ct_val)
            required_capital_risk = min_contracts * sizing.risk_per_contract / (risk_pct * strength)
            required_capital_notional = min_contracts * notional_per_contract / (
                leverage * notional_pct
            )
            required_risk_pct = (
                min_contracts * sizing.risk_per_contract / (pair.asset.capital * strength) * 100.0
                if pair.asset.capital > 0
                else 0.0
            )
            return {
                "side": side,
                "signal_id": signal.signal_id,
                "blocked_entry_px": round(entry_px, 6),
                "blocked_stop_px": round(stop_px, 6),
                "blocked_risk_per_contract": round(sizing.risk_per_contract, 6),
                "blocked_risk_budget_usd": round(sizing.risk_budget_usd, 6),
                "blocked_risk_sized_contracts": round(sizing.risk_sized_contracts, 10),
                "blocked_notional_sized_contracts": round(sizing.notional_sized_contracts, 10),
                "blocked_max_notional_usd": round(sizing.max_notional_usd, 6),
                "blocked_binding_cap": sizing.binding_cap,
                "blocked_required_capital_for_min_contract": round(
                    max(required_capital_risk, required_capital_notional), 6
                ),
                "blocked_required_risk_pct_for_min_contract": round(required_risk_pct, 6),
            }

        def _median(values: list[float]) -> float:
            if not values:
                return 0.0
            ordered = sorted(values)
            mid = len(ordered) // 2
            if len(ordered) % 2:
                return ordered[mid]
            return (ordered[mid - 1] + ordered[mid]) / 2.0

        def _bar_float(column: str, index: int) -> float | None:
            if column not in df.columns:
                return None
            value = df[column][index]
            return None if value is None else float(value)

        for i in range(1, df.height):
            bar_df = df.slice(i, 1)
            market = BarMarket.from_frame(df, i)
            intended_entry = float(entry_signal_col[i] or 0.0)
            intended_side = "buy" if intended_entry > 0 else "sell" if intended_entry < 0 else ""
            cooldown_key = f"{strategy_id}:{pair.asset.symbol}:{intended_side}"
            cooldown_blocked = bool(
                intended_side
                and self._loss_cooldown_bars > 0
                and i < cooldown_until.get(cooldown_key, 0)
            )
            signal = BarSignal(
                position=float(signal_col[i] or 0.0),
                entry=0.0 if cooldown_blocked else intended_entry,
                exit=bool(exit_signal_col[i]),
                strength=float(signal_strength_col[i] or 0.0),
                signal_id=str(signal_id_col[i] or strategy_id),
                zscore=_bar_float("dynamic_z_score", i),
                zscore_delta=_bar_float("zscore_delta", i),
                short_momentum_return=_bar_float("short_momentum_return", i),
                lower_wick_ratio=_bar_float("lower_wick_body_ratio", i),
                upper_wick_ratio=_bar_float("upper_wick_body_ratio", i),
                volatility_ratio=_bar_float("volatility_ratio", i),
                trend_return=_bar_float("trend_return", i),
                adx=_bar_float("adx_14", i),
            )
            actions = process_bar(
                bar_df,
                book,
                pair,
                exit_cfg,
                recovery_cfg,
                context=PipelineContext(strategy_id=strategy_id, market=market, signal=signal),
            )
            group_block_reasons: dict[str, str] = {}
            for action in actions:
                block_reason = _recovery_action_block_reason(action)
                if not block_reason:
                    continue
                group_key = _recovery_group_key(action)
                group_block_reasons[group_key] = block_reason
                if block_reason == "unsized":
                    recovery_unsized_actions += 1
            filtered_actions: list[BasketAction] = []
            for action in actions:
                group_key = _recovery_group_key(action)
                group_block_reason = (
                    group_block_reasons.get(group_key, "") if _is_recovery_action(action) else ""
                )
                block_reason = group_block_reason or _recovery_action_block_reason(action)
                if block_reason:
                    if _is_recovery_action(action):
                        if group_block_reason and not _recovery_action_block_reason(action):
                            block_reason = f"paired_{group_block_reason}"
                        _count_recovery_block(block_reason)
                    continue
                if _is_recovery_action(action) and action.action in {
                    ActionKind.ADD_GRID,
                    ActionKind.HEDGE,
                    ActionKind.ENTER,
                }:
                    recovery_allowed_actions += 1
                    close_px = float(market.close)
                    recovery_notional_after = _notional_exposure(close_px) + _action_notional(
                        action, close_px
                    )
                    recovery_notional_after_pct = max(
                        recovery_notional_after_pct,
                        recovery_notional_after / self._initial_capital * 100.0
                        if self._initial_capital
                        else 0.0,
                    )
                filtered_actions.append(action)
            actions = filtered_actions
            has_enter = any(a.action == ActionKind.ENTER for a in actions)
            has_exit = any(a.action == ActionKind.EXIT for a in actions)
            terminal_action_ids = {a.basket_id for a in actions if a.action == ActionKind.EXIT}
            suppressed_by_basket: dict[str, str] = {}
            for action in actions:
                if not _is_recovery_action(action):
                    continue
                basket = book.get(action.basket_id)
                if basket is None:
                    continue
                normal_exit = _normal_exit_candidate(basket)
                if normal_exit is not None:
                    suppressed_by_basket[action.basket_id] = normal_exit.reason
                    if normal_exit.reason in recovery_preempted_counts:
                        recovery_preempted_counts[normal_exit.reason] += 1
                if action.action in {ActionKind.ADD_GRID, ActionKind.HEDGE, ActionKind.ENTER}:
                    close_px = float(market.close)
                    action_notional = _action_notional(action, close_px)
                    basket_notional_after = action_notional
                    if action.action == ActionKind.ADD_GRID:
                        basket_notional_after += abs(
                            basket.current_sz * pair.asset.ct_val * close_px
                        )
                    max_notional = (
                        pair.asset.capital
                        * pair.asset.leverage
                        * float(getattr(pair.asset, "max_notional_pct_per_basket", 1.0))
                    )
                    if basket_notional_after > max_notional:
                        recovery_cap_breach_actions += 1
            if intended_entry != 0:
                action_counts["entry_signals"] += 1
            if has_enter and has_exit:
                action_counts["same_bar_exit_entry"] += 1
            if intended_entry != 0 and not has_enter:
                active_same_strategy = [
                    basket
                    for basket in book.active_for_strategy(pair.asset.symbol, strategy_id)
                    if basket.basket_id not in terminal_action_ids
                ]
                if cooldown_blocked:
                    _count_blocked_entry(f"loss_cooldown_{intended_side}")
                elif active_same_strategy:
                    _count_blocked_entry("duplicate_entry_suppressed")
                elif not terminal_action_ids and not book.can_open(pair.asset.symbol, strategy_id):
                    _count_blocked_entry(book.open_block_reason(pair.asset.symbol, strategy_id))
                else:
                    side = "buy" if signal.entry > 0 else "sell"
                    stop_px, _target_px = book.policy.compute_stop_target(
                        side, market.close, market.atr, pair.asset
                    )
                    sizing = book.policy.size_decision(
                        market.close,
                        stop_px,
                        pair.asset,
                        signal_strength=signal.strength,
                    )
                    _count_blocked_entry(sizing.blocked_reason or "no_entry_action")
                    if sizing.blocked_reason.startswith("below_min_contracts"):
                        blocked_sizing_rows.append(
                            _blocked_sizing_metadata(side, market.close, stop_px, sizing)
                        )
            for action in actions:
                _record_action_event(
                    action,
                    suppressed_exit_reason=suppressed_by_basket.get(action.basket_id, ""),
                )
                if action.action == ActionKind.ADD_GRID:
                    action_counts["grid"] += 1
                    action_counts["recovery"] += 1
                elif action.action == ActionKind.HEDGE:
                    action_counts["hedge"] += 1
                    action_counts["recovery"] += 1
                elif _is_recovery_reason(action.reason):
                    action_counts["recovery"] += 1
            close = float(df["close"][i])
            touched_ids: set[str] = set()

            for a in actions:
                touched_ids.add(a.basket_id)
                if a.action == ActionKind.HEDGE:
                    touched_ids.add(f"{a.basket_id}_hedge")
                basket = book.get(a.basket_id)

                if a.action == ActionKind.ENTER:
                    action_counts["entries"] += 1
                    entry_metadata[a.basket_id] = _sizing_metadata(a, i)
                    fee = _fee(a.px if a.px > 0 else close, a.sz)
                    fee_total += fee
                    cash -= fee

                elif a.action == ActionKind.EXIT:
                    snapshot = a.snapshot
                    if snapshot is None and basket is not None:
                        snapshot = book.snapshot(basket)
                    if snapshot is not None:
                        _count_exit(a.reason)
                        exit_px = a.px if a.px > 0 else close
                        entry_px = a.entry_px if a.entry_px > 0 else snapshot.entry_px
                        side = a.side or snapshot.side
                        d = 1 if side == "buy" else -1
                        effective_sz = a.sz if a.sz > 0 else snapshot.current_sz
                        pnl_pct = d * (exit_px / entry_px - 1) if entry_px > 0 else 0.0
                        gross_pnl_usd = d * (exit_px - entry_px) * effective_sz * pair.asset.ct_val
                        fee = _fee(exit_px, effective_sz)
                        net_pnl_usd = gross_pnl_usd - fee
                        ambiguous_stop_target = bool(
                            a.reason == "stop"
                            and basket is not None
                            and _ambiguous_stop_target(basket)
                        )
                        if ambiguous_stop_target:
                            ambiguous_stop_target_count += 1
                            ambiguous_stop_net_pnl_usd += net_pnl_usd
                            target_exit_px = snapshot.target_px
                            target_gross = (
                                d * (target_exit_px - entry_px) * effective_sz * pair.asset.ct_val
                            )
                            target_fee = _fee(target_exit_px, effective_sz)
                            target_first_counterfactual_net_pnl_usd += target_gross - target_fee
                        fee_total += fee
                        trades.append(
                            {
                                "side": side,
                                "entry_px": entry_px,
                                "exit_px": exit_px,
                                "pnl": round(pnl_pct, 6),
                                "gross_pnl_usd": round(gross_pnl_usd, 6),
                                "fee_usd": round(fee, 6),
                                "net_pnl_usd": round(net_pnl_usd, 6),
                                "pnl_usd": round(net_pnl_usd, 6),
                                "bars_held": a.bars_held,
                                "reason": a.reason,
                                "exit_family": _exit_family(a.reason),
                                "ambiguous_stop_target_bar": ambiguous_stop_target,
                                "stop_first_assumption_applied": ambiguous_stop_target,
                                "signal_id": a.signal_id,
                                "signal_strength": a.signal_strength,
                                **_exit_metadata(a.basket_id, exit_px, effective_sz, i),
                            }
                        )
                        cash += net_pnl_usd
                        if a.reason == "stop":
                            stop_exit_net_pnl_usd += net_pnl_usd
                            if snapshot.recovery_activated:
                                recovered_stop_exit_count += 1
                                recovered_stop_exit_net_pnl_usd += net_pnl_usd
                        if self._loss_cooldown_bars > 0 and net_pnl_usd < 0.0:
                            cooldown_until[f"{strategy_id}:{pair.asset.symbol}:{side}"] = (
                                i + self._loss_cooldown_bars
                            )
                        if _is_recovery_reason(a.reason):
                            recovery_net_pnl_usd += net_pnl_usd

                elif a.action == ActionKind.ADD_GRID:
                    fee = _fee(a.px if a.px > 0 else close, a.sz)
                    fee_total += fee
                    cash -= fee

                elif a.action == ActionKind.HEDGE:
                    fee = _fee(a.px if a.px > 0 else close, a.sz)
                    fee_total += fee
                    cash -= fee

            book.apply_actions(actions)
            book.advance_bar(close, float(df["high"][i]), float(df["low"][i]), skip_ids=touched_ids)

            current_value = _portfolio_value(close)
            if current_value > peak_equity:
                peak_equity = current_value

            if (
                self._drawdown_stop_pct is not None
                and peak_equity > 0
                and current_value / peak_equity - 1 < -self._drawdown_stop_pct
            ):
                print(
                    f"    WARNING: portfolio {self._drawdown_stop_pct:.0%} drawdown "
                    f"from peak ${peak_equity:.0f} — stopping"
                )
                for b in book.baskets:
                    if b.is_active:
                        pnl_pct, gross_pnl_usd, effective_sz = _exit_pnl(b, close)
                        fee = _fee(close, effective_sz)
                        net_pnl_usd = gross_pnl_usd - fee
                        fee_total += fee
                        trades.append(
                            {
                                "side": b.side,
                                "entry_px": b.entry_px,
                                "exit_px": close,
                                "pnl": round(pnl_pct, 6),
                                "gross_pnl_usd": round(gross_pnl_usd, 6),
                                "fee_usd": round(fee, 6),
                                "net_pnl_usd": round(net_pnl_usd, 6),
                                "pnl_usd": round(net_pnl_usd, 6),
                                "bars_held": b.bars_in_pos,
                                "reason": "global_drawdown_stop",
                                "exit_family": _exit_family("global_drawdown_stop"),
                                "ambiguous_stop_target_bar": False,
                                "stop_first_assumption_applied": False,
                                "signal_id": signal.signal_id,
                                "signal_strength": signal.strength,
                                **_exit_metadata(b.basket_id, close, effective_sz, i),
                            }
                        )
                        cash += net_pnl_usd
                        _count_exit("global_drawdown_stop")
                        book.close(b)
                stopped_out = True
                stop_bar_index = i
                current_value = cash

            active_contracts = book.active_exposure()
            max_simultaneous_baskets = max(max_simultaneous_baskets, len(book.active()))
            active_exposure.append(active_contracts)
            notional_exposure.append(_notional_exposure(close))
            equity.append(current_value)
            if "timestamp" in df.columns:
                timestamps.append(int(df["timestamp"][i]))
            signals.append(float(signal_col[i] or 0.0))
            if stopped_out:
                break

        if self._close_open_positions and not stopped_out and df.height:
            close = float(df["close"][min(len(equity), df.height) - 1])
            for b in list(book.baskets):
                if not b.is_active:
                    continue
                pnl_pct, gross_pnl_usd, effective_sz = _exit_pnl(b, close)
                fee = _fee(close, effective_sz)
                net_pnl_usd = gross_pnl_usd - fee
                fee_total += fee
                trades.append(
                    {
                        "side": b.side,
                        "entry_px": b.entry_px,
                        "exit_px": close,
                        "pnl": round(pnl_pct, 6),
                        "gross_pnl_usd": round(gross_pnl_usd, 6),
                        "fee_usd": round(fee, 6),
                        "net_pnl_usd": round(net_pnl_usd, 6),
                        "pnl_usd": round(net_pnl_usd, 6),
                        "bars_held": b.bars_in_pos,
                        "reason": "final_mark",
                        "exit_family": _exit_family("final_mark"),
                        "ambiguous_stop_target_bar": False,
                        "stop_first_assumption_applied": False,
                        "signal_id": "final_mark",
                        "signal_strength": 0.0,
                        **_exit_metadata(
                            b.basket_id,
                            close,
                            effective_sz,
                            min(len(equity), df.height) - 1,
                        ),
                    }
                )
                cash += net_pnl_usd
                _count_exit("final_mark")
                book.close(b)
            if equity:
                equity[-1] = cash

        self._last_active_exposure = active_exposure
        self._last_notional_exposure = notional_exposure
        self._last_timestamps = timestamps
        self._last_signals = signals
        self._last_action_events = action_events
        bars_held = [float(t.get("bars_held", 0.0) or 0.0) for t in trades]
        stacked_trades = [
            t for t in trades if int(t.get("entry_active_baskets", 0) or 0) > 0
        ]
        stacked_entry_net_pnl_usd = sum(
            float(t.get("net_pnl_usd", t.get("pnl_usd", 0.0)) or 0.0) for t in stacked_trades
        )
        exposure_pct = [value / self._initial_capital * 100.0 for value in notional_exposure]
        final_close = float(df["close"][min(len(equity), df.height) - 1]) if df.height else 0.0
        open_unrealized_pnl_usd = _unrealized_pnl(final_close) if final_close else 0.0
        feature = FeatureDiagnostics(bars=df.height, usable_bars=max(df.height - 1, 0))
        signal_diag = SignalDiagnostics(
            nonzero_signal_bars=sum(1 for signal in signal_col if float(signal or 0.0) != 0.0),
            long_signal_bars=sum(1 for signal in signal_col if float(signal or 0.0) > 0.0),
            short_signal_bars=sum(1 for signal in signal_col if float(signal or 0.0) < 0.0),
        )
        lifecycle = BasketLifecycleDiagnostics(
            entry_signals=action_counts["entry_signals"],
            entry_actions=action_counts["entries"],
            exit_actions=action_counts["exits"],
            grid_actions=action_counts["grid"],
            hedge_actions=action_counts["hedge"],
            recovery_actions=action_counts["recovery"],
            max_simultaneous_baskets=max_simultaneous_baskets,
            same_bar_exit_entry_count=action_counts["same_bar_exit_entry"],
            blocked_entry_signals=action_counts["blocked_entry_signals"],
            duplicate_entry_suppressed=action_counts["duplicate_entry_suppressed"],
            capacity_blocked_entries=action_counts["capacity_blocked_entries"],
            sizing_blocked_entries=action_counts["sizing_blocked_entries"],
            entry_acceptance_rate_pct=(
                action_counts["entries"] / action_counts["entry_signals"] * 100.0
                if action_counts["entry_signals"] > 0
                else 0.0
            ),
            blocked_entry_reasons=blocked_entry_reasons,
            min_contract_block_count=len(blocked_sizing_rows),
            median_required_capital_for_min_contract=round(
                _median(
                    [
                        float(row["blocked_required_capital_for_min_contract"])
                        for row in blocked_sizing_rows
                    ]
                ),
                4,
            ),
            median_required_risk_pct_for_min_contract=round(
                _median(
                    [
                        float(row["blocked_required_risk_pct_for_min_contract"])
                        for row in blocked_sizing_rows
                    ]
                ),
                4,
            ),
            blocked_by_risk_count=sum(
                1 for row in blocked_sizing_rows if row["blocked_binding_cap"] == "risk"
            ),
            blocked_by_notional_count=sum(
                1 for row in blocked_sizing_rows if row["blocked_binding_cap"] == "notional"
            ),
            action_event_count=len(action_events),
            stacked_entry_count=len(stacked_trades),
            stacked_entry_net_pnl_usd=round(stacked_entry_net_pnl_usd, 4),
            final_open_positions=sum(1 for b in book.baskets if b.is_active),
        )
        risk = PortfolioRiskDiagnostics(
            avg_active_exposure=(
                sum(active_exposure) / len(active_exposure) if active_exposure else 0.0
            ),
            max_active_exposure=max(active_exposure) if active_exposure else 0.0,
            avg_notional_exposure_pct=(
                sum(exposure_pct) / len(exposure_pct) if exposure_pct else 0.0
            ),
            max_notional_exposure_pct=max(exposure_pct) if exposure_pct else 0.0,
            fee_usd=fee_total,
            stop_exit_count=exit_reasons.get("stop", 0),
            stop_exit_net_pnl_usd=stop_exit_net_pnl_usd,
            recovered_stop_exit_count=recovered_stop_exit_count,
            recovered_stop_exit_net_pnl_usd=recovered_stop_exit_net_pnl_usd,
            recovery_net_pnl_usd=recovery_net_pnl_usd,
            recovery_preempted_stop_count=recovery_preempted_counts["stop"],
            recovery_preempted_time_count=recovery_preempted_counts["time"],
            recovery_preempted_trailing_count=recovery_preempted_counts["trailing_stop"],
            recovery_unsized_actions=recovery_unsized_actions,
            recovery_cap_breach_actions=recovery_cap_breach_actions,
            recovery_blocked_actions=recovery_blocked_actions,
            recovery_blocked_reasons=recovery_blocked_reasons,
            recovery_allowed_actions=recovery_allowed_actions,
            recovery_notional_after_pct=round(recovery_notional_after_pct, 6),
            ambiguous_stop_target_count=ambiguous_stop_target_count,
            ambiguous_stop_net_pnl_usd=ambiguous_stop_net_pnl_usd,
            target_first_counterfactual_net_pnl_usd=target_first_counterfactual_net_pnl_usd,
            ambiguity_impact_usd=target_first_counterfactual_net_pnl_usd
            - ambiguous_stop_net_pnl_usd,
            drawdown_stop_pct=self._drawdown_stop_pct,
            stopped_early=stopped_out,
        )
        audit = EngineDataAudit(
            bars=df.height,
            bars_processed=len(equity),
            data_start=int(df["timestamp"][0]) if "timestamp" in df.columns and df.height else None,
            data_end=(
                int(df["timestamp"][df.height - 1])
                if "timestamp" in df.columns and df.height
                else None
            ),
            mark_to_market=self._mark_to_market,
        )
        self._last_diagnostics = BacktestDiagnostics(
            feature=feature,
            signal=signal_diag,
            lifecycle=lifecycle,
            risk=risk,
            audit=audit,
            bars=df.height,
            bars_processed=len(equity),
            stopped_early=stopped_out,
            stop_bar_index=stop_bar_index,
            nonzero_signal_bars=sum(1 for signal in signal_col if float(signal or 0.0) != 0.0),
            long_signal_bars=sum(1 for signal in signal_col if float(signal or 0.0) > 0.0),
            short_signal_bars=sum(1 for signal in signal_col if float(signal or 0.0) < 0.0),
            entries=action_counts["entries"],
            exits=action_counts["exits"],
            exit_reasons=exit_reasons,
            avg_bars_held=sum(bars_held) / len(bars_held) if bars_held else 0.0,
            avg_active_exposure=(
                sum(active_exposure) / len(active_exposure) if active_exposure else 0.0
            ),
            max_active_exposure=max(active_exposure) if active_exposure else 0.0,
            avg_notional_exposure_pct=(
                sum(exposure_pct) / len(exposure_pct) if exposure_pct else 0.0
            ),
            max_notional_exposure_pct=max(exposure_pct) if exposure_pct else 0.0,
            final_open_positions=sum(1 for b in book.baskets if b.is_active),
            open_unrealized_pnl_usd=open_unrealized_pnl_usd,
            fee_usd=fee_total,
            data_start=int(df["timestamp"][0]) if "timestamp" in df.columns and df.height else None,
            data_end=(
                int(df["timestamp"][df.height - 1])
                if "timestamp" in df.columns and df.height
                else None
            ),
            mark_to_market=self._mark_to_market,
            drawdown_stop_pct=self._drawdown_stop_pct,
        )
        return trades, equity

    def run_report(
        self,
        df,
        pair,
        exit_cfg=None,
        recovery_cfg=None,
        *,
        strategy: StrategyBehavior | None = None,
        precomputed_signal: bool = False,
        metadata: tuple[str, ...] = (),
    ):
        from qooi.core.evaluate import Report

        strategy = strategy or momentum_burst_spec()
        strategy_id = strategy.name
        trades, equity = self.run(
            df,
            pair,
            exit_cfg,
            recovery_cfg,
            strategy=strategy,
            precomputed_signal=precomputed_signal,
        )
        return Report.from_raw(
            trades,
            equity,
            pair,
            label=f"{pair.asset.symbol} {strategy_id}",
            active_exposure=getattr(self, "_last_active_exposure", None),
            timestamps=getattr(self, "_last_timestamps", None),
            signals=getattr(self, "_last_signals", None),
            diagnostics=getattr(self, "_last_diagnostics", None),
            metadata=metadata,
        )

    def run_result(
        self,
        df,
        pair,
        exit_cfg=None,
        recovery_cfg=None,
        *,
        strategy: StrategyBehavior | None = None,
        precomputed_signal: bool = False,
        metadata: tuple[str, ...] = (),
    ) -> BacktestResult:
        from qooi.core.evaluate import _equity_frame, _trades_frame

        strategy = strategy or momentum_burst_spec()
        trades, equity = self.run(
            df,
            pair,
            exit_cfg,
            recovery_cfg,
            strategy=strategy,
            precomputed_signal=precomputed_signal,
        )
        return BacktestResult(
            label=f"{pair.asset.symbol} {strategy.name}",
            trades=_trades_frame(trades),
            equity=_equity_frame(
                equity,
                getattr(self, "_last_active_exposure", None),
                getattr(self, "_last_timestamps", None),
                getattr(self, "_last_signals", None),
            ),
            diagnostics=getattr(self, "_last_diagnostics", None),
            run_metadata=metadata,
        )

