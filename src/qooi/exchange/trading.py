"""OKX trading — orders, signals, execution, portfolio management.

Typed data layer using pydantic — no raw dicts in public API.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

from qooi.exchange.backtest import CostModel, RiskConfig

load_dotenv()

# === constants ===============================================================

LOG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

SIGNAL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "signals"
SIGNAL_DIR.mkdir(parents=True, exist_ok=True)


def _env(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise RuntimeError(f"Missing {key}. Set in .env or export {key}=...")
    return val


# ============================================================================
# 1. Trading client
# ============================================================================


class TradingClient:
    """Place/cancel orders, query balance and positions.

    Reads ``OKX_API_KEY``, ``OKX_SECRET_KEY``, ``OKX_PASSPHRASE``.
    Add ``_LIVE`` suffix for production keys (testnet = default)."""

    def __init__(self, live: bool = False) -> None:
        from okx.Account import AccountAPI
        from okx.Trade import TradeAPI

        suffix = "_LIVE" if live else ""
        flag = "0" if live else "1"
        k = _env(f"OKX_API_KEY{suffix}")
        s = _env(f"OKX_SECRET_KEY{suffix}")
        p = _env(f"OKX_PASSPHRASE{suffix}")
        self._trade = TradeAPI(k, s, p, flag=flag, debug=False)
        self._account = AccountAPI(k, s, p, flag=flag, debug=False)

    def place(
        self,
        inst_id: str,
        side: str,
        sz: str,
        ord_type: str = "post_only",
        px: str | None = None,
        td_mode: str = "cash",
    ) -> dict:
        params = {"instId": inst_id, "side": side, "ordType": ord_type, "sz": sz, "tdMode": td_mode}
        if px:
            params["px"] = px
        return _check(self._trade.place_order(**params))

    def cancel(self, inst_id: str, ord_id: str) -> dict:
        return _check(self._trade.cancel_order(instId=inst_id, ordId=ord_id))

    def pending(self) -> list:
        return self._trade.get_order_list().get("data", [])

    def balance(self, ccy: str | None = None) -> list:
        params = {}
        if ccy:
            params["ccy"] = ccy
        return _check(self._account.get_account_balance(**params)).get("details", [])

    def positions(self) -> list:
        return _check(self._account.get_positions()).get("data", [])


def _check(resp: dict) -> dict:
    if resp.get("code") != "0":
        raise RuntimeError(f"OKX error: {resp.get('msg', resp)}")
    return resp.get("data", [{}])[0] if resp.get("data") else resp


# ============================================================================
# 2. Typed data models (pydantic — JSON + human readable)
# ============================================================================


class SignalResult(BaseModel):
    """Output of a signal pipeline — the only format the executor consumes."""

    symbol: str
    timeframe: str
    timestamp: int
    signal: float
    flow: float = 0.0
    computed_at: int = 0

    @classmethod
    def from_dataframe(cls, symbol: str, timeframe: str, df) -> SignalResult:
        signal_val = float(df["signal"][-1] or 0.0)
        flow_val = float(df["ofi_flow_score"][-1] or 0.0) if "ofi_flow_score" in df.columns else 0.0
        return cls(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=int(df["timestamp"][-1]),
            signal=round(signal_val, 4),
            flow=round(flow_val, 4),
            computed_at=int(time.time()),
        )


class OrderRecord(BaseModel):
    """Immutable snapshot of an order for logging."""

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
    signal: float = 0.0
    obi: float = 0.0
    ofi_flow: float = 0.0

    @classmethod
    def from_state(cls, o) -> OrderRecord:
        return cls(
            ord_id=o.ord_id,
            inst_id=o.inst_id,
            side=o.side,
            sz=o.sz,
            px=o.px,
            placed_at=o.placed_at,
            filled_sz=o.filled_sz,
            filled_px=o.filled_px,
            status=o.status,
            reason=o.reason,
            signal=round(o.signal_value, 4),
            obi=round(o.obi_value, 4),
            ofi_flow=round(o.flow_value, 4),
        )


class LogLine(BaseModel):
    """One event line — JSONL record + human readable string."""

    ts: int = 0
    event: str = ""
    symbol: str = ""
    tf: str = ""
    payload: dict = {}

    def __str__(self) -> str:
        t = time.strftime("%H:%M:%S", time.localtime(self.ts / 1000))
        s, tf = self.symbol, self.tf
        p = self.payload
        match self.event:
            case "cycle_start":
                return f"[{t}] {s} {tf} -- cycle start"
            case "skip":
                return f"[{t}] {s} {tf} skip ({p.get('reason', '?')})"
            case "signal":
                sig = p.get("signal", 0)
                obi = p.get("obi_5", 0)
                flow = p.get("ofi_flow", 0)
                return f"[{t}] {s} {tf} | sig={sig:+.3f} obi={obi:+.3f} flow={flow:+.3f}"
            case "order":
                sd = p.get("side", "?")
                z = p.get("sz", 0)
                x = p.get("px", 0)
                oid = p.get("ord_id", "?")
                return f"[{t}] {s} {tf} | ORDER {sd:4s} sz={z:.6f} px={x:.1f} id={oid}"
            case "cancel":
                return f"[{t}] {s} {tf} | CANCEL {p.get('reason', '?')}"
            case "order_error" | "error":
                return f"[{t}] {s} {tf} | ERROR {p.get('error', p.get('msg', '?'))}"
            case _:
                return json.dumps(p, default=str)[:120]


# ============================================================================
# 3. Order state (mutable lifecycle tracker)
# ============================================================================


class OrderState:
    """Mutable tracker — created on place, updated on fill/cancel."""

    __slots__ = (
        "ord_id",
        "inst_id",
        "side",
        "sz",
        "px",
        "placed_at",
        "filled_sz",
        "filled_px",
        "status",
        "reason",
        "signal_value",
        "obi_value",
        "flow_value",
    )

    def __init__(self) -> None:
        self.ord_id = ""
        self.inst_id = ""
        self.side = ""
        self.sz = 0.0
        self.px = 0.0
        self.placed_at = 0.0
        self.filled_sz = 0.0
        self.filled_px = 0.0
        self.status = "placed"
        self.reason = ""
        self.signal_value = 0.0
        self.obi_value = 0.0
        self.flow_value = 0.0


# ============================================================================
# 4. Signal source protocol
# ============================================================================

SignalSource = Callable[[str, str], SignalResult | None]


def default_signal_source() -> SignalSource:
    """Factory → multi-factor intraday ensemble."""

    def _src(symbol: str, timeframe: str) -> SignalResult | None:
        from qooi.exchange.indicator import add_indicators
        from qooi.exchange.market import MarketData
        from qooi.strategies.flow_pipeline import (
            add_adaptive_threshold,
            add_ofi_flow_columns,
            add_regime_features,
            apply_adaptive_gate,
            apply_micro_confirmation,
        )
        from qooi.strategies.intraday import multi_factor_intraday_signal

        df = MarketData().candles(symbol, timeframe=timeframe, limit=500)
        if df.is_empty():
            return None
        df = add_indicators(df)
        for fn in (
            add_regime_features,
            multi_factor_intraday_signal,
            add_ofi_flow_columns,
            apply_micro_confirmation,
            add_adaptive_threshold,
            apply_adaptive_gate,
        ):
            df = fn(df)
        return SignalResult.from_dataframe(symbol, timeframe, df)

    return _src


# ============================================================================
# 5. Live executor
# ============================================================================


class LiveExecutor:
    """Consume SignalResult → place/manage limit orders → log.

    signal_source=None → reads from data/signals/ (file-based fallback).
    signal_source=callable → computes directly (recommended).
    """

    def __init__(
        self,
        symbol: str = "BTC-USDT",
        timeframe: str = "4h",
        initial_capital: float = 1000.0,
        max_position_pct: float = 0.05,
        leverage: float = 1.0,
        post_only: bool = True,
        limit_timeout_sec: int = 120,
        dry_run: bool = True,
        live: bool = False,
        risk: RiskConfig | None = None,
        cost: CostModel | None = None,
    ) -> None:
        self._symbol = symbol
        self._tf = timeframe
        self._capital = initial_capital
        self._leverage = leverage
        self._max_pos = max_position_pct
        self._post_only = post_only
        self._timeout = limit_timeout_sec
        self._dry = dry_run
        self._risk = risk or RiskConfig(atr_stop_mult=2.0, atr_target_mult=3.0)
        self._cost = cost or CostModel()

        from qooi.exchange.market import MarketData

        self._md = MarketData()
        self._client: TradingClient | None = None if dry_run else TradingClient(live=live)
        self._active: OrderState | None = None
        self._trade_count = 0
        self._equity = [initial_capital]
        self._stop_price = -1.0
        self._target_price = -1.0
        self._trail_high = -1.0
        self._trail_low = -1.0

    # -- public ---------------------------------------------------------------

    def step(self, signal_source: SignalSource | None = None) -> OrderState | None:
        self._log("cycle_start")
        sr: SignalResult | None = None

        if signal_source:
            sr = signal_source(self._symbol, self._tf)
            if sr:
                # persist for audit trail
                (SIGNAL_DIR / f"{self._symbol.replace('-', '_')}_{self._tf}.json").write_text(
                    sr.model_dump_json(indent=2)
                )
        else:
            sf = SIGNAL_DIR / f"{self._symbol.replace('-', '_')}_{self._tf}.json"
            if sf.exists():
                sr = SignalResult.model_validate(json.loads(sf.read_text()))

        if not sr:
            return self._log("skip", {"reason": "no_signal"})

        bar_ms = {"1h": 3600000, "4h": 14400000, "1d": 86400000}.get(self._tf, 14400000)
        if time.time() * 1000 - sr.timestamp > bar_ms * 1.5:
            return self._log("skip", {"reason": "stale_signal"})

        obi = self._md.ob_snapshot(self._symbol)
        self._log(
            "signal", {"signal": sr.signal, "obi_5": round(obi.imbalance_5, 4), "ofi_flow": sr.flow}
        )

        # --- stop-loss / target / trailing (RiskConfig-integrated) ---
        exit_reason: str | None = None
        cur_close = (
            obi.ask_price
            if self._active and self._active.side == "buy"
            else obi.bid_price
            if self._active
            else 0
        )
        r = self._risk
        atr_est = obi.ask_price * 0.02  # rough ATR estimate from 2% volatility

        if self._active and self._stop_price > 0:
            direction = 1 if self._active.side == "buy" else -1
            if direction > 0:
                if cur_close <= self._stop_price:
                    exit_reason = "stop"
                elif self._target_price > 0 and cur_close >= self._target_price:
                    exit_reason = "target"
                elif self._trail_high > 0:
                    self._trail_high = max(self._trail_high, cur_close)
                    if self._trail_high - cur_close >= r.trailing_distance_mult * atr_est:
                        exit_reason = "trailing_stop"
            else:
                if cur_close >= self._stop_price:
                    exit_reason = "stop"
                elif self._target_price > 0 and cur_close <= self._target_price:
                    exit_reason = "target"
                elif self._trail_low > 0:
                    self._trail_low = min(self._trail_low, cur_close)
                    if cur_close - self._trail_low >= r.trailing_distance_mult * atr_est:
                        exit_reason = "trailing_stop"

        if exit_reason and self._active:
            self._cancel(exit_reason)

        if (
            self._active
            and self._active.status == "placed"
            and time.time() - self._active.placed_at > self._timeout
        ):
            self._cancel("timeout")

        if abs(sr.signal) < 0.25:
            return self._log("skip", {"reason": "weak_signal"})

        if self._active and self._active.status in ("placed", "partial_fill"):
            return self._log("skip", {"reason": "order_outstanding"})

        side = "buy" if sr.signal > 0 else "sell"
        if not self._dry and self._client:
            pos = self._client.positions()
            if any(p.get("instId") == self._symbol and float(p.get("pos", "0")) != 0 for p in pos):
                if side == "buy":
                    return self._log("skip", {"reason": "already_long"})

        entry_px = obi.ask_price if side == "buy" else obi.bid_price
        size_coeff = self._capital * self._max_pos * abs(sr.signal) * self._leverage
        sz = size_coeff / max(entry_px, 1)
        if sz < 0.00001:
            return self._log("skip", {"reason": "size_too_small"})

        return self._place(side, sz, entry_px, sr.signal, obi, sr.flow)

    def status(self) -> dict:
        return {
            "symbol": self._symbol,
            "tf": self._tf,
            "capital": self._capital,
            "leverage": self._leverage,
            "dry": self._dry,
            "trades": self._trade_count,
            "active": self._active.ord_id if self._active else None,
        }

    @staticmethod
    def report(symbol: str = "ETH-USDT", timeframe: str = "4h") -> dict:
        """Read trade logs and compute P&L / Sharpe / Win Rate via compute_metrics."""
        import polars as pl

        from qooi.exchange.eval import compute_metrics

        fname = LOG_DIR / f"exec_{symbol.replace('-', '_')}_{timeframe}.jsonl"
        if not fname.exists():
            return {"error": f"no log file: {fname}"}

        lines = []
        with open(fname) as f:
            for line in f:
                try:
                    lines.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

        orders = [o for o in lines if o.get("event") == "order"]
        if not orders:
            return {"error": "no trades yet"}

        # Build trade log for compute_metrics
        trades = []
        entry = {}
        for o in orders:
            payload = o.get("payload", {})
            if o.get("event") != "order":
                continue
            if not entry:
                entry = {
                    "entry_ts": o["ts"],
                    "entry_px": payload.get("px", 0),
                    "side": payload.get("side", ""),
                }
            else:
                exit_px = payload.get("px", 0)
                pnl = (exit_px / entry["entry_px"] - 1) * (1 if entry["side"] == "long" else -1)
                trades.append(
                    {
                        "entry_time": entry["entry_ts"],
                        "exit_time": o["ts"],
                        "side": entry["side"],
                        "entry_price": entry["entry_px"],
                        "exit_price": exit_px,
                        "pnl": pnl,
                        "reason": "signal",
                    }
                )
                entry = {"entry_ts": o["ts"], "entry_px": exit_px, "side": payload.get("side", "")}

        trade_df = pl.DataFrame(trades) if trades else pl.DataFrame()
        eq = pl.Series(
            [1000] + [1000 * (1 + (t["exit_price"] / t["entry_price"] - 1)) for t in trades],
            dtype=pl.Float64,
        )
        eq_df = pl.DataFrame({"portfolio_value": eq, "returns": eq.pct_change().fill_null(0.0)})
        m = compute_metrics(eq_df, trades=trade_df)

        return {
            "trades": m.num_trades,
            "win_rate_pct": m.win_rate_pct,
            "sharpe": m.sharpe_ratio,
            "total_return_pct": m.total_return_pct,
            "max_drawdown_pct": m.max_drawdown_pct,
        }

    # -- internals ------------------------------------------------------------

    def _place(
        self, side: str, sz: float, px: float, signal: float, obi, flow: float
    ) -> OrderState | None:
        sz, px = round(sz, 8), round(px, 1)
        r = self._risk
        atr_est = px * 0.02

        if side == "buy":
            self._stop_price = px - r.atr_stop_mult * atr_est
            self._target_price = px + r.atr_target_mult * atr_est
            self._trail_high = px
        else:
            self._stop_price = px + r.atr_stop_mult * atr_est
            self._target_price = px - r.atr_target_mult * atr_est
            self._trail_low = px
        st = OrderState()
        st.inst_id = self._symbol
        st.side = side
        st.sz = sz
        st.px = px
        st.placed_at = time.time()
        st.signal_value = signal
        st.obi_value = obi.imbalance_5
        st.flow_value = flow

        if self._dry:
            st.ord_id = "dry_" + str(int(time.time()))
            st.status = "simulated"
            self._active = st
            self._log_order()
            self._trade_count += 1
            return self._active

        try:
            otype = "post_only" if self._post_only else "limit"
            resp = self._client.place(self._symbol, side, str(sz), ord_type=otype, px=str(px))
            st.ord_id = resp.get("ordId", "")
            st.status = "placed"
            self._active = st
            self._log_order()
            self._trade_count += 1
            return self._active
        except Exception as e:
            self._log("order_error", {"error": str(e)})
            return None

    def _cancel(self, reason: str) -> None:
        if not self._active:
            return
        self._active.status, self._active.reason = "cancelled", reason
        self._log_order()
        if not self._dry and self._client:
            self._client.cancel(self._symbol, self._active.ord_id)
        self._active = None

    def _log_order(self) -> None:
        if self._active:
            self._log(
                "order" if self._active.status == "placed" else "cancel",
                OrderRecord.from_state(self._active).model_dump(),
            )

    def _log(self, event: str, payload: dict | None = None) -> None:
        line = LogLine(
            ts=int(time.time() * 1000),
            event=event,
            symbol=self._symbol,
            tf=self._tf,
            payload=payload or {},
        )
        fname = LOG_DIR / f"exec_{self._symbol.replace('-', '_')}_{self._tf}.jsonl"
        with open(fname, "a") as f:
            f.write(line.model_dump_json() + "\n")
        print(str(line))
        return None  # used as return from step() when skipping


# ============================================================================
# 6. Portfolio runner
# ============================================================================


class PortfolioConfig(BaseModel):
    pairs: list[dict]
    dry_run: bool = True
    post_only: bool = True
    limit_timeout_sec: int = 120


class PortfolioRunner:
    """Multi-asset deployment."""

    def __init__(self, config: PortfolioConfig) -> None:
        self._config = config
        self._executors: dict[str, LiveExecutor] = {
            f"{p['symbol']}_{p.get('tf', '4h')}": LiveExecutor(
                symbol=p["symbol"],
                timeframe=p.get("tf", "4h"),
                initial_capital=p.get("capital", 100),
                max_position_pct=p.get("risk_pct", 0.03),
                leverage=p.get("leverage", 1.0),
                post_only=config.post_only,
                limit_timeout_sec=config.limit_timeout_sec,
                dry_run=config.dry_run,
            )
            for p in config.pairs
        }

    def step(self, source: SignalSource | None = None) -> dict:
        results = {}
        t = time.strftime("%Y-%m-%d %H:%M:%S")
        d = self._config.dry_run
        print(f"\n{'=' * 60}\n  Portfolio {t}  Dry: {d}\n{'=' * 60}")
        for key, exe in self._executors.items():
            results[key] = exe.step(signal_source=source)
        self._summary(results)
        return results

    def compute_all_signals(self, source: SignalSource | None = None) -> list[SignalResult]:
        src = source or default_signal_source()
        out = []
        for p in self._config.pairs:
            s = src(p["symbol"], p.get("tf", "4h"))
            if s:
                out.append(s)
                print(f"  {p['symbol']:15s} sig={s.signal:+.4f}")
        return out

    def status(self) -> dict:
        return {k: e.status() for k, e in self._executors.items()}

    def _summary(self, results: dict) -> None:
        txt, jl = LOG_DIR / "portfolio_summary.txt", LOG_DIR / "portfolio.jsonl"
        placed = sum(1 for r in results.values() if r and r.status == "placed")
        lines = [f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]"]
        for k, r in results.items():
            if r and r.status == "placed":
                lines.append(
                    f"  {k:20s} {r.side.upper():4s} sz={r.sz:.6f} sig={r.signal_value:+.3f}"
                )
            elif r:
                lines.append(f"  {k:20s} {r.status}")
            else:
                lines.append(f"  {k:20s} no_signal")
        lines.append(f"  Placed: {placed}/{len(results)}")
        with open(txt, "a") as f:
            f.write("\n".join(lines) + "\n\n")
        with open(jl, "a") as f:
            f.write(
                LogLine(
                    ts=int(time.time() * 1000),
                    event="portfolio",
                    symbol="*",
                    tf="*",
                    payload={"placed": placed, "total": len(results)},
                ).model_dump_json()
                + "\n"
            )
