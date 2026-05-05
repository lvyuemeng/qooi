"""Live strategy executor — signal pipeline + paper trader + order lifecycle.

Wires together: OHLCV fetch → indicator → ensemble signal → OFI flow →
adaptive gate → limit order placement → fill tracking → cancel/timeout.

All events logged as structured JSONL for post-trade evaluation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from qooi.exchange.indicator import add_indicators
from qooi.exchange.market import MarketData
from qooi.exchange.paper_trader import OkxPaperTrader
from qooi.strategies.flow_pipeline import (
    add_adaptive_threshold,
    add_ofi_flow_columns,
    add_regime_features,
    apply_adaptive_gate,
    apply_micro_confirmation,
)
from qooi.strategies.intraday import multi_factor_intraday_signal

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "logs"


@dataclass
class OrderState:
    """Live order tracking — created when we place, updated on fill/cancel."""

    ord_id: str = ""
    inst_id: str = ""
    side: str = ""
    sz: float = 0.0
    px: float = 0.0
    placed_at: float = 0.0
    filled_sz: float = 0.0
    filled_px: float = 0.0
    status: str = "placed"
    reason: str = ""
    signal_value: float = 0.0
    obi_value: float = 0.0
    flow_value: float = 0.0


class LiveExecutor:
    """Run strategy on each bar close, place limit orders, track fills.

    Parameters:
        symbol: e.g. "BTC-USDT-SWAP"
        timeframe: "4h" or "1h"
        initial_capital: USDT
        max_position_pct: max % of capital to risk per trade
        post_only: use post_only limit orders (maker fee 0.005%)
        limit_timeout_sec: cancel unfilled limit after this many seconds
        sleep_sec: poll interval while waiting for fill
        log_dir: where to write JSONL logs
        dry_run: compute signal but don't send orders
    """

    def __init__(
        self,
        symbol: str = "BTC-USDT-SWAP",
        timeframe: str = "4h",
        initial_capital: float = 1000.0,
        max_position_pct: float = 0.05,
        post_only: bool = True,
        limit_timeout_sec: int = 120,
        sleep_sec: int = 10,
        log_dir: str | None = None,
        dry_run: bool = True,
    ) -> None:
        self._symbol = symbol
        self._timeframe = timeframe
        self._capital = initial_capital
        self._max_pos = max_position_pct
        self._post_only = post_only
        self._timeout = limit_timeout_sec
        self._sleep = sleep_sec
        self._dry = dry_run

        self._md = MarketData()
        self._trader: OkxPaperTrader | None = None
        if not dry_run:
            import os

            from dotenv import load_dotenv

            load_dotenv()
            self._trader = OkxPaperTrader(
                os.getenv("OKX_API_KEY", ""),
                os.getenv("OKX_SECRET_KEY", ""),
                os.getenv("OKX_PASSPHRASE", ""),
            )

        self._log_path = Path(log_dir or LOG_DIR)
        self._log_path.mkdir(parents=True, exist_ok=True)
        self._active: OrderState | None = None
        self._trade_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self) -> OrderState | None:
        """Run one bar cycle: fetch → signal → order → track → log.

        Returns the OrderState if an order was placed, None otherwise.
        """
        self._log_event("cycle_start", {})

        # 1. Fetch data
        df = self._md.candles(self._symbol, timeframe=self._timeframe, limit=500)
        if df.is_empty():
            self._log_event("error", {"msg": "empty candles"})
            return None

        # 2. Compute indicators & signal
        df = add_indicators(df)
        df = add_regime_features(df)
        df = multi_factor_intraday_signal(df)
        df = add_ofi_flow_columns(df)
        df = apply_micro_confirmation(df)
        df = add_adaptive_threshold(df)
        df = apply_adaptive_gate(df)

        signal = float(df["signal"][-1] or 0.0)
        obi = self._md.ob_snapshot(self._symbol)
        flow_val = float(df["ofi_flow_score"][-1] or 0.0)

        self._log_event(
            "signal",
            {
                "signal": round(signal, 4),
                "obi_5": round(obi.imbalance_5, 4),
                "obi_25": round(obi.imbalance_25, 4),
                "ofi_flow": round(flow_val, 4),
            },
        )

        # 3. Cancel stale orders / manage existing position
        if self._active and self._active.status == "placed":
            elapsed = time.time() - self._active.placed_at
            if elapsed > self._timeout:
                self._cancel_order("timeout")

        # 4. Entry gate
        if abs(signal) < 0.25:
            self._log_event("skip", {"reason": "weak_signal", "signal": round(signal, 4)})
            return None

        if self._active and self._active.status in ("placed", "partial_fill"):
            # Don't stack orders
            self._log_event("skip", {"reason": "order_outstanding", "ord_id": self._active.ord_id})
            return None

        # 5. Calculate size & place order
        side = "buy" if signal > 0 else "sell"
        entry_px = obi.ask_price if side == "buy" else obi.bid_price
        risk = self._capital * self._max_pos * abs(signal)
        sz = risk / entry_px if entry_px > 0 else 0.001

        if sz < 0.00001:
            self._log_event("skip", {"reason": "size_too_small", "sz": round(sz, 8)})
            return None

        order = self._place_limit(side, sz, entry_px, signal, obi, flow_val)
        return order

    def get_status(self) -> dict:
        """Current executor state for monitoring."""
        return {
            "symbol": self._symbol,
            "timeframe": self._timeframe,
            "capital": self._capital,
            "dry_run": self._dry,
            "active_order": {
                "ord_id": self._active.ord_id,
                "status": self._active.status,
                "side": self._active.side,
                "sz": self._active.sz,
                "px": self._active.px,
                "filled_sz": self._active.filled_sz,
            }
            if self._active
            else None,
            "trade_count": self._trade_count,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _place_limit(
        self,
        side: str,
        sz: float,
        px: float,
        signal: float,
        obi: object,
        flow: float,
    ) -> OrderState | None:
        sz = round(sz, 8)
        px = round(px, 1)

        if self._dry:
            self._active = OrderState(
                inst_id=self._symbol,
                side=side,
                sz=sz,
                px=px,
                placed_at=time.time(),
                signal_value=signal,
                obi_value=obi.imbalance_5,
                flow_value=flow,
                ord_id="dry_" + str(int(time.time())),
                status="simulated",
            )
            self._log_event("order", self._order_dict(self._active))
            self._trade_count += 1
            return self._active

        try:
            if side == "buy":
                resp = self._trader.limit_buy(self._symbol, sz, px)
            else:
                resp = self._trader.limit_sell(self._symbol, sz, px)

            ord_data = resp.get("data", [{}])[0]
            self._active = OrderState(
                ord_id=ord_data.get("ordId", ""),
                inst_id=self._symbol,
                side=side,
                sz=sz,
                px=px,
                placed_at=time.time(),
                signal_value=signal,
                obi_value=obi.imbalance_5,
                flow_value=flow,
                status="placed",
            )
            self._log_event("order", self._order_dict(self._active))
            self._trade_count += 1
            return self._active

        except Exception as e:
            self._log_event("order_error", {"error": str(e), "side": side, "sz": sz, "px": px})
            return None

    def _cancel_order(self, reason: str) -> None:
        if not self._active:
            return
        self._active.status = "cancelled"
        self._active.reason = reason
        self._log_event("cancel", self._order_dict(self._active))

        if not self._dry and self._trader:
            self._trader._api.cancel_order(instId=self._symbol, ordId=self._active.ord_id)

        self._active = None

    def _order_dict(self, o: OrderState) -> dict:
        return {
            "ord_id": o.ord_id,
            "inst_id": o.inst_id,
            "side": o.side,
            "sz": o.sz,
            "px": o.px,
            "placed_at": o.placed_at,
            "filled_sz": o.filled_sz,
            "filled_px": o.filled_px,
            "status": o.status,
            "reason": o.reason,
            "signal": round(o.signal_value, 4),
            "obi": round(o.obi_value, 4),
            "ofi_flow": round(o.flow_value, 4),
        }

    def _log_event(self, event: str, data: dict) -> None:
        record = {
            "ts": int(time.time() * 1000),
            "event": event,
            "symbol": self._symbol,
            "tf": self._timeframe,
            **data,
        }
        fname = self._log_path / f"exec_{self._symbol.replace('-', '_')}_{self._timeframe}.jsonl"
        with open(fname, "a") as f:
            f.write(json.dumps(record) + "\n")
