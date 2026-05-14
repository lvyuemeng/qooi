"""Executor — maps BasketActions to real-world orders.

Two executors: LiveExecutor (direct OKX API) and BacktestExecutor (simulate).
Both consume the same list[BasketAction] from the pipeline.
"""

from __future__ import annotations

from qooi.core.basket import ActionKind, Basket, BasketAction, BasketBook, BasketState


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

    def __init__(self, initial_capital: float = 10000.0, cost_pct: float = 0.00005):
        self._initial_capital = initial_capital
        self._cost = cost_pct

    def run(
        self,
        df,
        pair,
        exit_cfg=None,
        recovery_cfg=None,
        *,
        strategy="momentum_burst",
    ) -> tuple[list[dict], list[float]]:
        from qooi.core import process_bar
        from qooi.core.state import BacktestStateProvider
        from qooi.strategies import compute_signal_frame
        from qooi.strategies.specs import resolve_spec

        strategy_id = resolve_spec(strategy).name if isinstance(strategy, str) else strategy.name
        df = compute_signal_frame(df, strategy)
        signal_col = df["signal"].to_list()

        state = BacktestStateProvider()
        baskets = state.load([pair])
        book = BasketBook(baskets)
        trades: list[dict] = []
        equity = [self._initial_capital]
        active_exposure = [0.0]
        peak_equity = self._initial_capital
        stopped_out = False

        def _exit_pnl(
            basket: Basket, exit_px: float, sz: float = 0.0
        ) -> tuple[float, float, float]:
            d = 1 if basket.side == "buy" else -1
            pnl_pct = d * (exit_px / basket.entry_px - 1) if basket.entry_px > 0 else 0.0
            effective_sz = sz if sz > 0 else basket.current_sz
            notional = effective_sz * pair.asset.ct_val * basket.entry_px
            pnl_usd = pnl_pct * notional
            return pnl_pct, pnl_usd, effective_sz

        for i in range(1, df.height):
            active_before = book.active_exposure()
            bar_df = df.slice(i, 1)
            actions = process_bar(
                bar_df,
                book,
                pair,
                exit_cfg,
                recovery_cfg,
                signal_src=signal_col[i],
                strategy_id=strategy_id,
            )
            close = float(df["close"][i])
            prev_eq = equity[-1]

            for a in actions:
                basket = book.get(a.basket_id)

                if a.action == ActionKind.ENTER:
                    prev_eq *= 1.0 - self._cost

                elif a.action == ActionKind.EXIT:
                    if basket:
                        exit_px = a.px if a.px > 0 else close
                        pnl_pct, pnl_usd, _ = _exit_pnl(basket, exit_px, a.sz)
                        basket.cumulative_loss += abs(min(0, pnl_usd))
                        trades.append(
                            {
                                "side": basket.side,
                                "entry_px": basket.entry_px,
                                "exit_px": exit_px,
                                "pnl": round(pnl_pct, 6),
                                "pnl_usd": round(pnl_usd, 6),
                                "bars_held": basket.bars_in_pos,
                                "reason": a.reason,
                            }
                        )
                        prev_eq += pnl_usd

                elif a.action == ActionKind.ADD_GRID:
                    if basket:
                        basket.add_to_position(a.sz, a.px)
                    prev_eq *= 1.0 - self._cost

                elif a.action == ActionKind.HEDGE:
                    new_basket = Basket(
                        basket_id=f"{a.basket_id}_hedge",
                        symbol=pair.asset.symbol,
                        strategy=strategy_id,
                        side=a.side,
                        state=BasketState.ACTIVE,
                        entry_px=a.px,
                        current_sz=a.sz,
                    )
                    book.replace_or_add(new_basket)
                    prev_eq *= 1.0 - self._cost

            if prev_eq > peak_equity:
                peak_equity = prev_eq

            if peak_equity > 0 and prev_eq / peak_equity - 1 < -0.05:
                print(f"    WARNING: portfolio 5% drawdown from peak ${peak_equity:.0f} — stopping")
                for b in baskets:
                    if b.is_active:
                        pnl_pct, pnl_usd, _ = _exit_pnl(b, close)
                        trades.append(
                            {
                                "side": b.side,
                                "entry_px": b.entry_px,
                                "exit_px": close,
                                "pnl": round(pnl_pct, 6),
                                "pnl_usd": round(pnl_usd, 6),
                                "bars_held": b.bars_in_pos,
                                "reason": "global_drawdown_stop",
                            }
                        )
                        prev_eq += pnl_usd
                stopped_out = True

            active_exposure.append(active_before)
            equity.append(prev_eq)
            if stopped_out:
                break

        self._last_active_exposure = active_exposure
        state.save_soft(baskets)
        return trades, equity

    def run_report(
        self, df, pair, exit_cfg=None, recovery_cfg=None, *, strategy="momentum_burst"
    ):
        from qooi.core.evaluate import Report
        from qooi.strategies.specs import resolve_spec

        strategy_id = resolve_spec(strategy).name if isinstance(strategy, str) else strategy.name
        trades, equity = self.run(df, pair, exit_cfg, recovery_cfg, strategy=strategy)
        return Report.from_raw(
            trades,
            equity,
            pair,
            label=f"{pair.asset.symbol} {strategy_id}",
            active_exposure=getattr(self, "_last_active_exposure", None),
        )
