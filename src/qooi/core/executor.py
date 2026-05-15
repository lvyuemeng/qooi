"""Executor — maps BasketActions to real-world orders.

Two executors: LiveExecutor (direct OKX API) and BacktestExecutor (simulate).
Both consume the same list[BasketAction] from the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from qooi.core.basket import ActionKind, Basket, BasketAction, BasketBook
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
        obi = self._md.ob_snapshot(symbol, limit=1)
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
    ):
        self._initial_capital = initial_capital
        self._cost = cost_pct
        self._drawdown_stop_pct = drawdown_stop_pct
        self._mark_to_market = mark_to_market
        self._close_open_positions = close_open_positions

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
        from qooi.core.evaluate import BacktestDiagnostics

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
        book = BasketBook(baskets)
        trades: list[dict] = []
        equity = [self._initial_capital]
        cash = self._initial_capital
        active_exposure = [0.0]
        notional_exposure = [0.0]
        timestamps = [int(df["timestamp"][0])] if "timestamp" in df.columns and df.height else []
        signals = [float(signal_col[0] or 0.0)] if signal_col else []
        action_counts = {"entries": 0, "exits": 0}
        exit_reasons: dict[str, int] = {}
        peak_equity = self._initial_capital
        stopped_out = False
        stop_bar_index: int | None = None
        fee_total = 0.0

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

        def _fee(px: float, sz: float) -> float:
            return abs(px * sz * pair.asset.ct_val) * self._cost

        def _count_exit(reason: str) -> None:
            action_counts["exits"] += 1
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

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

        for i in range(1, df.height):
            bar_df = df.slice(i, 1)
            market = BarMarket.from_frame(df, i)
            signal = BarSignal(
                position=float(signal_col[i] or 0.0),
                entry=float(entry_signal_col[i] or 0.0),
                exit=bool(exit_signal_col[i]),
                strength=float(signal_strength_col[i] or 0.0),
                signal_id=str(signal_id_col[i] or strategy_id),
            )
            actions = process_bar(
                bar_df,
                book,
                pair,
                exit_cfg,
                recovery_cfg,
                context=PipelineContext(strategy_id=strategy_id, market=market, signal=signal),
            )
            close = float(df["close"][i])
            touched_ids: set[str] = set()

            for a in actions:
                touched_ids.add(a.basket_id)
                if a.action == ActionKind.HEDGE:
                    touched_ids.add(f"{a.basket_id}_hedge")
                basket = book.get(a.basket_id)

                if a.action == ActionKind.ENTER:
                    action_counts["entries"] += 1
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
                                "signal_id": a.signal_id,
                                "signal_strength": a.signal_strength,
                            }
                        )
                        cash += net_pnl_usd

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
                            }
                        )
                        cash += net_pnl_usd
                        _count_exit("global_drawdown_stop")
                        book.close(b)
                stopped_out = True
                stop_bar_index = i
                current_value = cash

            active_contracts = book.active_exposure()
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
        bars_held = [float(t.get("bars_held", 0.0) or 0.0) for t in trades]
        exposure_pct = [value / self._initial_capital * 100.0 for value in notional_exposure]
        final_close = float(df["close"][min(len(equity), df.height) - 1]) if df.height else 0.0
        open_unrealized_pnl_usd = _unrealized_pnl(final_close) if final_close else 0.0
        self._last_diagnostics = BacktestDiagnostics(
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
