"""OKX trading �?orders, signals, execution, portfolio management.

Typed data layer using pydantic �?no raw dicts in public-facing API.
Side-effectful operations (IO, network) are separated from pure data models.
"""

from __future__ import annotations

import enum
import io
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

from qooi.exchange.backtest import CostModel, RiskConfig

# ============================================================================
# 0. Environment
# ============================================================================


def load_okx_env(env_path: str | None = None) -> None:
    path = None
    if env_path:
        path = Path(env_path)
    elif os.getenv("OKX_ENV"):
        inferred = (
            Path(__file__).resolve().parent.parent.parent.parent / f".env.{os.getenv('OKX_ENV')}"
        )
        if inferred.exists():
            path = inferred
    else:
        default = Path(".env")
        if default.exists():
            path = default
    load_dotenv(path, override=True) if path else load_dotenv(override=True)


LOG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
SIGNAL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "signals"
SIGNAL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 1. Trading client
# ============================================================================


class TradingClient:
    _RETRY_ATTEMPTS = 3
    _RETRY_DELAY = 1.0

    def __init__(self) -> None:
        from okx.Account import AccountAPI
        from okx.Trade import TradeAPI

        if not os.getenv("OKX_API_KEY") and not os.getenv("OKX_API_KEY_TEST"):
            load_okx_env()
        flag = os.getenv("OKX_FLAG", "1")
        k = os.getenv("OKX_API_KEY") or os.getenv("OKX_API_KEY_TEST", "")
        s = os.getenv("OKX_SECRET_KEY") or os.getenv("OKX_SECRET_KEY_TEST", "")
        p = os.getenv("OKX_PASSPHRASE") or os.getenv("OKX_PASSPHRASE_TEST", "")
        if not k:
            raise RuntimeError("Missing OKX_API_KEY �?call load_okx_env() or set envvars")
        self._trade = TradeAPI(k, s, p, flag=flag, debug=False)
        self._account = AccountAPI(k, s, p, flag=flag, debug=False)

    @staticmethod
    def _okx(resp: dict, key: str = "data") -> dict:
        if resp.get("code") != "0":
            code = resp.get("code", "?")
            msg = resp.get("msg", str(resp))
            data = resp.get("data", [])
            raise RuntimeError(f"OKX error [{code}]: {msg}  data={data}")
        return resp.get(key, [{}])[0] if resp.get(key) else {}

    @staticmethod
    def _retry(fn, *args, **kwargs):
        """Call fn with retry on network errors (3 attempts, backoff)."""
        attempts = TradingClient._RETRY_ATTEMPTS
        last_err = None
        for attempt in range(attempts):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_err = e
                if attempt < attempts - 1:
                    time.sleep(TradingClient._RETRY_DELAY * (attempt + 1))
        if last_err is not None:
            raise last_err

    def place(
        self,
        inst_id: str,
        side: str,
        sz: str,
        ord_type: str = "post_only",
        px: str | None = None,
        td_mode: str = "isolated",
    ) -> dict:
        params = {"instId": inst_id, "side": side, "ordType": ord_type, "sz": sz, "tdMode": td_mode}
        if px:
            params["px"] = px
        return self._okx(self._trade.place_order(**params))

    def cancel(self, inst_id: str, ord_id: str) -> dict:
        return self._okx(self._trade.cancel_order(instId=inst_id, ordId=ord_id))

    def amend(
        self, inst_id: str, ord_id: str, new_sz: str | None = None, new_px: str | None = None
    ) -> dict:
        """Amend an unfilled limit order — change price and/or size."""
        params: dict = {"instId": inst_id, "ordId": ord_id}
        if new_sz is not None:
            params["newSz"] = new_sz
        if new_px is not None:
            params["newPx"] = new_px
        return self._okx(self._trade.amend_order(**params))

    def pending(self) -> list:
        return self._okx(self._trade.get_order_list()).get("orders", [])

    def balance(self, ccy: str | None = None) -> list:
        p = {} if not ccy else {"ccy": ccy}
        return self._okx(self._account.get_account_balance(**p)).get("details", [])

    def positions(self) -> list:
        return self._okx(self._account.get_positions()).get("posData", [])


# ============================================================================
# 2. Pure data models �?all pydantic, zero side-effects
# ============================================================================

# -- signals ---------------------------------------------------------------


class SignalResult(BaseModel):
    symbol: str
    timeframe: str
    timestamp: int
    signal: float
    flow: float = 0.0
    threshold: float = 0.0  # signal threshold used for entry gating (per-asset percentile)
    computed_at: int = 0

    @classmethod
    def from_dataframe(cls, symbol: str, timeframe: str, df) -> SignalResult:
        sv = float(df["signal"][-1] or 0.0)
        fv = float(df["ofi_flow_score"][-1] or 0.0) if "ofi_flow_score" in df.columns else 0.0
        return cls(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=int(df["timestamp"][-1]),
            signal=round(sv, 4),
            flow=round(fv, 4),
            computed_at=int(time.time()),
        )


# -- log payloads ----------------------------------------------------------


class SkipPayload(BaseModel):
    reason: str = ""


class SignalPayload(BaseModel):
    signal: float = 0.0
    obi_5: float = 0.0
    ofi_flow: float = 0.0


class CancelPayload(BaseModel):
    reason: str = ""
    ord_id: str = ""


class ErrorPayload(BaseModel):
    error: str = ""


class OrderPayload(BaseModel):
    """Single order lifecycle �?placed �?filled/cancelled.  Also the mutable
    state tracker used by LiveExecutor internally (no separate OrderState)."""

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


LogPayload = SignalPayload | OrderPayload | SkipPayload | CancelPayload | ErrorPayload


# -- enums (must be before Decision which references them) ------------------


class FillStatus(enum.StrEnum):
    PLACED = "placed"
    PARTIAL = "partial_fill"
    FILLED = "filled"
    SIMULATED = "simulated"


class State(enum.StrEnum):
    """Executor lifecycle — IDLE → PENDING → ACTIVE → IDLE."""

    IDLE = "idle"
    PENDING = "pending"
    ACTIVE = "active"


@dataclass
class SyncFacts:
    """Pure query result — no state mutation."""

    position: dict | None = None  # exchange position data
    order: dict | None = None  # newest exchange order
    duplicates: list[dict] | None = None  # extra orders to cancel
    log_order: dict | None = None  # log-recovered order
    api_ok: bool = False


@dataclass
class FillFacts:
    """Pure query result — no state mutation."""

    filled: bool = False
    partial: bool = False
    missing: bool = False
    filled_px: float = 0.0
    filled_sz: float = 0.0


@dataclass
class Decision:
    """Pure output of the state machine — no side effects."""

    action: str  # "enter" | "exit" | "amend" | "skip"
    new_state: State = State.IDLE  # next state — ALWAYS set
    side: str = ""
    sz: float = 0.0
    entry_px: float = 0.0
    exit_px: float = 0.0
    detail: str = ""

    # --- dynamic risk adjustments (set alongside action) ---
    new_stop: float | None = None
    new_target: float | None = None
    scale_out_pct: float = 0.0
    amend_px: float | None = None
    amend_sz: float | None = None

    @classmethod
    def exit(cls, reason: str, px: float, new_state: State = State.IDLE) -> Decision:
        return cls(action="exit", new_state=new_state, detail=reason, exit_px=px)

    @classmethod
    def enter(cls, side: str, sz: float, entry_px: float) -> Decision:
        return cls(action="enter", new_state=State.PENDING, side=side, sz=sz, entry_px=entry_px)

    @classmethod
    def skip(cls, reason: str, new_state: State = State.IDLE) -> Decision:
        return cls(action="skip", new_state=new_state, detail=reason)

    @classmethod
    def amend(
        cls,
        reason: str,
        new_state: State = State.PENDING,
        *,
        new_stop: float | None = None,
        new_target: float | None = None,
        scale_out_pct: float = 0.0,
        amend_px: float | None = None,
        amend_sz: float | None = None,
    ) -> Decision:
        return cls(
            action="amend",
            new_state=new_state,
            detail=reason,
            new_stop=new_stop,
            new_target=new_target,
            scale_out_pct=scale_out_pct,
            amend_px=amend_px,
            amend_sz=amend_sz,
        )


class LogLine(BaseModel):
    ts: int = 0
    event: str = ""
    symbol: str = ""
    tf: str = ""
    payload: LogPayload = SkipPayload()

    def __str__(self) -> str:
        t = time.strftime("%H:%M:%S", time.localtime(self.ts / 1000))
        s, tf = self.symbol, self.tf
        p = self.payload
        match self.event:
            case "cycle_start":
                return f"[{t}] {s} {tf} -- cycle start"
            case "skip":
                return f"[{t}] {s} {tf} skip ({getattr(p, 'reason', '?')})"
            case "signal":
                sig = getattr(p, "signal", 0)
                obi = getattr(p, "obi_5", 0)
                flow = getattr(p, "ofi_flow", 0)
                return f"[{t}] {s} {tf} | sig={sig:+.3f} obi={obi:+.3f} flow={flow:+.3f}"
            case "order":
                sd = getattr(p, "side", "?")
                z = getattr(p, "sz", 0)
                x = getattr(p, "px", 0)
                oid = getattr(p, "ord_id", "?")
                return f"[{t}] {s} {tf} | ORDER {sd:4s} sz={z:.6f} px={x:.1f} id={oid}"
            case "cancel":
                oid = getattr(p, "ord_id", "")
                return f"[{t}] {s} {tf} | CANCEL {getattr(p, 'reason', '?')} id={oid}"
            case "order_error" | "error":
                return f"[{t}] {s} {tf} | ERROR {getattr(p, 'error', '?')}"
            case _:
                return p.model_dump_json()


# -- position state --------------------------------------------------------


class PositionState(BaseModel):
    """Bundles active order + risk management into one state object.

    Replaces the scattered scalar fields (stop_price, target_price, trail_high,
    trail_low) that previously lived on LiveExecutor itself."""

    order: OrderPayload = OrderPayload()
    stop_price: float = -1.0
    target_price: float = -1.0
    trail_high: float = -1.0
    trail_low: float = -1.0
    fill_status: FillStatus = FillStatus.PLACED
    entry_price: float = 0.0
    entry_ts: int = 0
    bars_held: int = 0

    @classmethod
    def enter_long(cls, entry_px: float, atr: float, risk: RiskConfig, ts_ms: int) -> PositionState:
        stop = entry_px - risk.atr_stop_mult * atr
        target = entry_px + risk.atr_target_mult * atr
        return cls(
            stop_price=stop,
            target_price=target,
            trail_high=entry_px,
            entry_price=entry_px,
            entry_ts=ts_ms,
            bars_held=0,
            fill_status=FillStatus.PLACED,
        )

    @classmethod
    def enter_short(
        cls, entry_px: float, atr: float, risk: RiskConfig, ts_ms: int
    ) -> PositionState:
        stop = entry_px + risk.atr_stop_mult * atr
        target = entry_px - risk.atr_target_mult * atr
        return cls(
            stop_price=stop,
            target_price=target,
            trail_low=entry_px,
            entry_price=entry_px,
            entry_ts=ts_ms,
            bars_held=0,
            fill_status=FillStatus.PLACED,
        )

    def check_exit(self, cur_close: float, atr: float, risk: RiskConfig) -> str | None:
        """Return 'stop' / 'target' / 'trailing_stop' / None."""
        d = 1 if self.order.side == "buy" else -1
        if d > 0:
            if cur_close <= self.stop_price:
                return "stop"
            if self.target_price > 0 and cur_close >= self.target_price:
                return "target"
            if self.trail_high > 0:
                self.trail_high = max(self.trail_high, cur_close)
                if self.trail_high - cur_close >= risk.trailing_distance_mult * atr:
                    return "trailing_stop"
        else:
            if cur_close >= self.stop_price:
                return "stop"
            if self.target_price > 0 and cur_close <= self.target_price:
                return "target"
            if self.trail_low > 0:
                self.trail_low = min(self.trail_low, cur_close)
                if cur_close - self.trail_low >= risk.trailing_distance_mult * atr:
                    return "trailing_stop"
        return None


class AssetReport(BaseModel):
    error: str | None = None
    trades: int = 0
    win_rate_pct: float = 0.0
    sharpe: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0


class Summary(BaseModel):
    """Portfolio state snapshot for CI / monitoring."""

    pre_usdt: float = 0.0
    post_usdt: float = 0.0
    usdt_change: float = 0.0
    total_upl: float = 0.0
    total_margin: float = 0.0
    positions: list[dict] = []
    assets: dict[str, AssetReport] = {}

    @classmethod
    def from_runner(cls, runner, tc: TradingClient, pre_usdt: float, pre_pos: int) -> Summary:
        # --- real-time position snapshot ---
        positions: list[dict] = []
        total_upl = 0.0
        total_margin = 0.0
        try:
            for p in tc.positions():
                inst = p.get("instId", "?")
                pos_sz = float(p.get("pos", 0))
                upl = float(p.get("upl", 0))
                margin = float(p.get("margin", 0))
                avg_px = float(p.get("avgPx", 0))
                mark_px = float(p.get("markPx", 0))
                side = "long" if pos_sz > 0 else "short"
                positions.append(
                    {
                        "inst": inst,
                        "side": side,
                        "sz": abs(pos_sz),
                        "avg_px": avg_px,
                        "mark_px": mark_px,
                        "upl": upl,
                        "margin": margin,
                    }
                )
                total_upl += upl
                total_margin += margin
        except Exception:
            pass

        # --- USDT balance ---
        try:
            post_usdt = float(tc.balance("USDT")[0].get("availBal", 0))
        except Exception:
            post_usdt = 0.0

        pairs = [
            (p.get("exec_symbol", p.get("sig_symbol", "")), p.get("tf", "4h"))
            for p in runner.config.pairs
        ]
        return cls(
            pre_usdt=pre_usdt,
            post_usdt=post_usdt,
            usdt_change=post_usdt - pre_usdt,
            total_upl=total_upl,
            total_margin=total_margin,
            positions=positions,
            assets={s: asset_report(s, tf) for s, tf in pairs},
        )

    def write_to(self, fh: io.TextIOBase | None = None, tc: TradingClient | None = None) -> None:
        if fh is None:
            fh = sys.stdout
        fh.write("## qooi Portfolio\n\n")

        # --- total portfolio value ---
        total_value = self.post_usdt + self.total_margin + self.total_upl
        yield_pct = (total_value / self.pre_usdt - 1) * 100 if self.pre_usdt > 0 else 0.0
        fh.write(f"Total: ${total_value:,.2f} ({yield_pct:+.2f}% since inception)\n")

        # --- USDT + margin breakdown ---
        free = self.post_usdt
        deployed = self.total_margin
        fh.write(f"  USDT free:   ${free:,.2f}\n")
        fh.write(f"  USDT margin: ${deployed:,.2f}\n")

        # --- open positions ---
        if self.positions:
            fh.write("\n  Positions:\n")
            for p in self.positions:
                pnl_pct = (p["upl"] / p["margin"] * 100) if p["margin"] > 0 else 0.0
                fh.write(
                    f"    {p['inst']:20s} {p['side']:5s} "
                    f"{p['sz']}ct @ {p['avg_px']:,.1f} → {p['mark_px']:,.1f}  "
                    f"upl=${p['upl']:+,.2f} ({pnl_pct:+.1f}%)\n"
                )

        # --- per-symbol trading stats ---
        fh.write("\n")
        for sym, rpt in self.assets.items():
            if rpt.error:
                fh.write(f"**{sym}**: {rpt.error}\n")
            else:
                fh.write(
                    f"**{sym}**: T={rpt.trades} WR={rpt.win_rate_pct:.0f}% "
                    f"Shp={rpt.sharpe:.2f} Ret={rpt.total_return_pct:.1f}% "
                    f"DD={rpt.max_drawdown_pct:.1f}%\n"
                )
        fh.write("\n> Full logs in artifact.\n")


# ============================================================================
# 3. Signal source
# ============================================================================

SignalSource = Callable[[str, str], SignalResult | None]


def default_signal_source(sig_threshold: float = 0.35) -> SignalSource:
    # Share a single MarketData instance across all calls to avoid
    # re-initializing ccxt for every symbol (3× ~10s → 1× ~12s).
    from qooi.exchange.market import MarketData

    _md = MarketData("okx")

    def _src(symbol: str, timeframe: str) -> SignalResult | None:
        from qooi.exchange.indicator import add_indicators
        from qooi.strategies.flow_pipeline import (
            add_ofi_flow_columns,
            add_regime_features,
        )

        df = _md.candles(symbol, timeframe=timeframe, limit=500, cache=True)
        if df.is_empty():
            return None
        df = add_indicators(df)
        df = add_regime_features(df)
        df = add_ofi_flow_columns(df)

        # OFI flow as signal with magnitude filter.
        # Default threshold = 0.35. Optimal per-asset:
        #   SOL: 0.35, BTC: 0.25, XRP: 0.45 (data-limited)
        ofi_col = df["ofi_flow_score"]
        threshold = sig_threshold
        ofi = float(ofi_col[-1])
        sig_val = round(ofi, 4) if abs(ofi) >= threshold else 0.0

        return SignalResult(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=int(df["timestamp"][-1]),
            signal=sig_val,
            flow=round(ofi, 4),
            threshold=round(threshold, 4),
            computed_at=int(time.time()),
        )

    return _src


# ============================================================================
# 4. Live executor
# ============================================================================


class LiveExecutor:
    # -- adaptive threshold defaults (scaled relative to asset threshold) -----
    _ADAPTIVE_BASE_RATIO: float = 1.00  # base = threshold × ratio
    _ADAPTIVE_MIN_RATIO: float = 0.625  # min  = threshold × ratio  (0.25/0.40)
    _ADAPTIVE_MAX_RATIO: float = 1.75  # max  = threshold × ratio  (0.70/0.40)
    _ADAPTIVE_LOOKBACK: int = 50

    def __init__(
        self,
        symbol: str = "BTC-USDT",
        timeframe: str = "4h",
        initial_capital: float = 1000.0,
        max_position_pct: float = 0.05,
        leverage: float = 1.0,
        ct_val: float = 1.0,
        td_mode: str = "isolated",
        post_only: bool = True,
        ord_type: str = "",
        limit_timeout_sec: int = 0,  # 0 = derive from timeframe (2x bar duration)
        dry_run: bool = True,
        risk: RiskConfig | None = None,
        cost: CostModel | None = None,
    ) -> None:
        self._symbol = symbol
        self._tf = timeframe
        self._capital = initial_capital
        self._leverage = leverage
        self._max_pos = max_position_pct
        self._post_only = post_only
        self._ct_val = ct_val  # contract value in base currency (ETH=0.1, SOL=1, BTC=0.01)
        self._td_mode = td_mode  # "isolated" = each position independent margin
        self._ord_type = ord_type  # "" = derive from post_only, else "market"/"limit"/"post_only"
        # Timeframe-aware timeout: 0 = derive from bar duration, else explicit
        bar_map = {"1h": 3600, "4h": 14400, "1d": 86400}
        bar_duration = bar_map.get(timeframe, 14400)
        self._timeout = limit_timeout_sec if limit_timeout_sec > 0 else (bar_duration * 2)
        self._dry = dry_run
        self._risk = risk or RiskConfig()
        self._cost = cost or CostModel()
        from qooi.exchange.market import MarketData

        self._md = MarketData()
        self._client: TradingClient | None = None if dry_run else TradingClient()
        self._state: State = State.IDLE
        self._position: PositionState | None = None
        self._trade_count = 0
        self._equity = [initial_capital]
        self._loss_streak = 0
        self._pnl_ema = 0.0
        self._error_streak = 0
        self._signal_threshold: float = 0.40  # default, overridden by signal

    # -- public ---------------------------------------------------------------

    def sync(self) -> SyncFacts:
        """Reconcile local state with exchange reality — pure query, no mutation.

        Returns SyncFacts.  State changes happen in _decide().

        When API returns 0 pending orders/positions, also reads the log.
        This prevents duplicates when the testnet API hasn't propagated
        an order from a previous run yet (API succeeds but order invisible).
        """
        if self._dry or not self._client:
            return SyncFacts()

        try:
            pending = TradingClient._retry(self._client.pending)
            my_orders = [o for o in pending if o.get("instId") == self._symbol]
            positions = TradingClient._retry(self._client.positions)
            my_positions = [p for p in positions if p.get("instId") == self._symbol]
            # When API shows nothing, also check the log for recent orders
            log_order = None
            if not my_positions and not my_orders:
                log_order = self._read_log_order()
            return SyncFacts(
                position=my_positions[0] if my_positions else None,
                order=my_orders[0] if my_orders else None,
                duplicates=my_orders[1:] if len(my_orders) > 1 else None,
                log_order=log_order,
                api_ok=True,
            )
        except Exception:
            return SyncFacts(log_order=self._read_log_order(), api_ok=False)

    def step(
        self, signal_source: SignalSource | None = None, signal: SignalResult | None = None
    ) -> OrderPayload | None:
        """One bar cycle: sync → fetch → decide → execute.

        If ``signal`` is provided (pre-computed), skips the fetch step.
        """
        facts = self.sync()
        self._log("cycle_start")
        sr = signal if signal is not None else self._fetch_signal(signal_source)
        if not sr:
            return self._log_skip("no_signal")
        if sr.threshold > 0:
            self._signal_threshold = sr.threshold
        bar_ms = {"1h": 3600000, "4h": 14400000, "1d": 86400000}.get(self._tf, 14400000)
        if time.time() * 1000 - sr.timestamp > bar_ms * 1.5:
            return self._log_skip("stale_signal")

        # Optimize: only fetch order book when we might actually trade.
        need_ob = self._state != State.IDLE or abs(sr.signal) >= self._entry_threshold()
        obi = self._md.ob_snapshot(self._symbol) if need_ob else None
        payload = SignalPayload(
            signal=sr.signal,
            obi_5=round(obi.imbalance_5, 4) if obi else 0,
            ofi_flow=sr.flow,
        )
        self._log("signal", payload)

        fill = self.check_fill()
        decision = self._decide(sr, obi, facts, fill)
        return self._execute(decision, sr, obi)

    # -- state machine (ALL state changes happen here) -------------------------

    def _decide(self, sr: SignalResult, obi, facts: SyncFacts, fill: FillFacts) -> Decision:
        """Central state dispatch.  SyncFacts + FillFacts consumed here.
        
        ALL state transitions happen in this method.
        """
        # --- resolve sync facts → set state + position ---
        if facts.position:
            self._adopt_position(facts.position)
            self._state = State.ACTIVE
        elif facts.order:
            self._adopt_order(facts.order)
            self._state = State.PENDING
        elif facts.log_order:
            self._adopt_order(facts.log_order)
            self._state = State.PENDING
            self._log_skip("resumed_from_logs")
        if facts.duplicates:
            self._cancel_dupes(facts.duplicates)

        # --- resolve fill facts → update fill_status ---
        if fill.missing and self._position:
            self._position = None
            self._state = State.IDLE
        elif fill.filled and self._position:
            self._position.fill_status = FillStatus.FILLED
            self._position.order.filled_sz = fill.filled_sz
            self._position.order.filled_px = fill.filled_px
            self._position.entry_price = fill.filled_px
        elif fill.partial and self._position:
            self._position.fill_status = FillStatus.PARTIAL

        # --- dispatch ---
        match self._state:
            case State.IDLE:
                return self._decide_idle(sr, obi)
            case State.PENDING:
                return self._decide_pending(sr, obi)
            case State.ACTIVE:
                return self._decide_active(sr, obi)
        return Decision.skip("unknown_state")

    def _decide_idle(self, sr: SignalResult, obi) -> Decision:
        """IDLE: check if signal is strong enough to enter."""
        threshold = self._entry_threshold()
        if abs(sr.signal) < threshold:
            return Decision.skip(f"weak_signal ({abs(sr.signal):.3f} < {threshold:.2f})")
        ml = self._risk.max_leverage
        clipped = max(-ml, min(ml, sr.signal)) if ml > 0 else 0.0
        if clipped == 0.0:
            return Decision.skip("clipped_to_zero")
        side = "buy" if clipped > 0 else "sell"
        entry_px = obi.ask_price if side == "buy" else obi.bid_price
        sz = self._compute_size(clipped, entry_px)
        if sz < 0.00001:
            return Decision.skip("size_too_small")
        required = sz * self._ct_val * entry_px / max(self._leverage, 1)
        free = self._free_usdt()
        if free < required * 1.2:
            return Decision.skip(f"insufficient_margin (need ${required:.0f}, have ${free:.0f})")
        return Decision.enter(side, sz, entry_px)

    def _decide_pending(self, sr: SignalResult, obi) -> Decision:
        """PENDING: check if order filled, or if we should amend/cancel."""
        if not self._position:
            return Decision.skip("position_lost")

        if self._position.fill_status == FillStatus.FILLED:
            self._position.bars_held = 0
            return Decision.amend("order_filled", new_state=State.ACTIVE)
        if self._position.fill_status == FillStatus.PARTIAL:
            return Decision.skip("partial_fill", new_state=State.PENDING)

        # --- dynamic risk: time decay ---
        age = time.time() - self._position.order.placed_at
        if age > self._timeout:
            px = obi.bid_price if self._position.order.side == "buy" else obi.ask_price
            return Decision.exit("timeout", px)

        # --- dynamic risk: signal decay ---
        entry_sig = self._position.order.signal
        if entry_sig and sr.signal * entry_sig < 0:
            px = obi.bid_price if self._position.order.side == "buy" else obi.ask_price
            return Decision.exit("signal_flipped", px)
        if entry_sig and abs(sr.signal) < abs(entry_sig) * 0.5:
            new_sz = self._position.order.sz * 0.5
            if new_sz >= 0.00001:
                return Decision.amend("signal_weakened", amend_sz=new_sz)
            px = obi.bid_price if self._position.order.side == "buy" else obi.ask_price
            return Decision.exit("signal_weakened_to_zero", px)

        # --- dynamic risk: price chasing ---
        is_buy = self._position.order.side == "buy"
        market_px = obi.ask_price if is_buy else obi.bid_price
        distance_pct = abs(market_px - self._position.order.px) / max(self._position.order.px, 1e-9)
        if distance_pct > 0.005:
            new_px = round(market_px * (0.998 if is_buy else 1.002), 1)
            return Decision.amend("price_chase", amend_px=new_px)

        return Decision.skip("order_outstanding", new_state=State.PENDING)

    def _decide_active(self, sr: SignalResult, obi) -> Decision:
        """ACTIVE: manage position risk — stops, targets, trails."""
        if not self._position:
            return Decision.skip("position_lost")

        self._position.bars_held += 1
        cur_close = obi.ask_price if self._position.order.side == "buy" else obi.bid_price
        atr_est = obi.ask_price * 0.02
        d = 1 if self._position.order.side == "buy" else -1

        if self._risk.max_bars_held > 0 and self._position.bars_held >= self._risk.max_bars_held:
            return Decision.exit("time", cur_close)

        # D1: signal-based exit
        if d * sr.signal < 0:
            return Decision.exit("signal_flipped", cur_close)

        # Hard exits (stop / target / trailing)
        reason = self._position.check_exit(cur_close, atr_est, self._risk)
        if reason:
            return Decision.exit(reason, cur_close)

        # Breakeven: move stop to entry once profit > 1×ATR
        unreal_pnl_pct = d * (cur_close / self._position.entry_price - 1)
        atr_pct = atr_est / max(self._position.entry_price, 1e-9)
        if unreal_pnl_pct > atr_pct and self._position.stop_price != self._position.entry_price:
            return Decision.amend("breakeven_stop", new_stop=self._position.entry_price,
                                 new_state=State.ACTIVE)

        # Continuous trailing after breakeven (replaces old 2×ATR activation gap)
        if self._position.stop_price == self._position.entry_price:
            trail_dist = self._risk.trailing_distance_mult * atr_est
            new_stop = cur_close - d * trail_dist
            if d * (new_stop - self._position.stop_price) > 0:
                self._position.stop_price = new_stop
                if d > 0:
                    self._position.trail_high = max(self._position.trail_high, cur_close)
                else:
                    self._position.trail_low = min(self._position.trail_low, cur_close)

        # Scale-out at 50% of target
        if self._position.target_price > 0:
            halfway = self._position.entry_price + d * 0.5 * abs(
                self._position.target_price - self._position.entry_price
            )
            if d * (cur_close - halfway) > 0:
                return Decision.amend("scale_out_50pct", scale_out_pct=0.5,
                                     new_state=State.ACTIVE)

        return Decision.skip("holding", new_state=State.ACTIVE)

    # -- execute (side effects only, no state mutation except d.new_state) ------

    def _execute(self, d: Decision, sr: SignalResult, obi) -> OrderPayload | None:
        """Apply the decision — side effects only.  State set from d.new_state."""
        self._state = d.new_state

        if d.action == "enter":
            return self._place(d.side, d.sz, d.entry_px, sr.signal, obi, sr.flow)

        if d.action == "exit":
            if self._position and self._position.fill_status == FillStatus.FILLED:
                exit_side = "sell" if self._position.order.side == "buy" else "buy"
                d_sign = 1 if self._position.order.side == "buy" else -1
                pnl = d_sign * (d.exit_px / self._position.order.px - 1)
                self._equity.append(self._equity[-1] * (1 + pnl))
                self._pnl_ema = self._ema_update(self._pnl_ema, pnl, self._ADAPTIVE_LOOKBACK)
                if pnl < 0:
                    self._loss_streak += 1
                else:
                    self._loss_streak = 0
                self._place(exit_side, self._position.order.sz, d.exit_px,
                            -self._position.order.signal, None, 0, force_market=True)
            else:
                d_sign = 1 if (self._position and self._position.order.side == "buy") else -1
                if self._position and self._position.order.px > 0:
                    pnl = d_sign * (d.exit_px / self._position.order.px - 1)
                    self._equity.append(self._equity[-1] * (1 + pnl))
                    self._pnl_ema = self._ema_update(self._pnl_ema, pnl, self._ADAPTIVE_LOOKBACK)
                    if pnl < 0:
                        self._loss_streak += 1
                    else:
                        self._loss_streak = 0
            self._cancel(d.detail)
            return None

        if d.action == "amend":
            self._apply_amend(d, sr.signal)
            return None

        if d.action == "skip":
            self._log_skip(d.detail)

        return None

    # -- adaptive threshold --------------------------------------------------

    @staticmethod
    def _thresh(pnl_ema: float, base: float, lo: float, hi: float) -> float:
        """Adaptive entry threshold from real PnL history."""
        if abs(pnl_ema) < 0.001 or pnl_ema > 0.02:
            return lo
        if pnl_ema > 0.005:
            return base - (pnl_ema - 0.005) / 0.015 * (base - lo)
        if pnl_ema > -0.005:
            return base
        return max(lo, min(hi, base + (abs(pnl_ema) - 0.005) / 0.02 * (hi - base)))

    @staticmethod
    def _ema_update(prev: float, cur: float, period: int) -> float:
        alpha = 2.0 / (period + 1)
        return alpha * cur + (1 - alpha) * prev

    def _free_usdt(self) -> float:
        """Query free USDT balance.  Returns 0 on any error."""
        if not self._client:
            return float("inf")  # dry run — no margin constraint
        try:
            for b in self._client.balance():
                if b.get("ccy") == "USDT":
                    return float(b.get("availBal", 0))
        except Exception:
            pass
        return 0.0

    def _entry_threshold(self) -> float:
        """Current adaptive entry threshold based on real PnL EMA.
        Scaled relative to the asset's signal threshold (ETH≈0.4, BTC≈0.03)."""
        t = self._signal_threshold
        base = t * self._ADAPTIVE_BASE_RATIO
        lo = t * self._ADAPTIVE_MIN_RATIO
        hi = t * self._ADAPTIVE_MAX_RATIO
        return self._thresh(self._pnl_ema, base, lo, hi)

    # -- sync helpers ---------------------------------------------------------

    def _adopt_position(self, pos_data: dict) -> None:
        """Build PositionState from exchange position data — no state change."""
        side = "buy" if float(pos_data.get("pos", 0)) > 0 else "sell"
        px = float(pos_data.get("avgPx", 0))
        sz = abs(float(pos_data.get("pos", 0)))
        atr_est = px * 0.02
        parent = PositionState.enter_long if side == "buy" else PositionState.enter_short
        self._position = parent(px, atr_est, self._risk, int(time.time() * 1000))
        self._position.order = OrderPayload(
            inst_id=self._symbol, side=side, sz=sz, px=px, status="filled", ord_id=""
        )
        self._position.entry_price = px
        self._position.fill_status = FillStatus.FILLED

    def _adopt_order(self, ord_data: dict) -> None:
        """Build PositionState from order data — no state change."""
        ctime_ms = float(ord_data.get("cTime", 0))
        placed_at = ctime_ms / 1000.0 if ctime_ms > 0 else time.time()
        entry_sig = float(ord_data.get("signal", 0))
        self._position = PositionState(
            order=OrderPayload(
                inst_id=self._symbol,
                side=ord_data.get("side", ""),
                sz=float(ord_data.get("sz", 0)),
                px=float(ord_data.get("px", 0)),
                ord_id=ord_data.get("ordId", ""),
                placed_at=placed_at,
                signal=entry_sig,
                status="placed",
            ),
            fill_status=FillStatus.PLACED,
            entry_price=float(ord_data.get("px", 0)),
            entry_ts=int(placed_at * 1000),
        )

    def _cancel_dupes(self, orders: list[dict]) -> None:
        """Cancel all but the newest order."""
        sorted_orders = sorted(orders, key=lambda o: float(o.get("cTime", 0)), reverse=True)
        for o in sorted_orders[1:]:
            oid = o.get("ordId", "")
            try:
                if self._client:
                    TradingClient._retry(self._client.cancel, self._symbol, oid)
                    self._log("cancel", CancelPayload(reason="duplicate", ord_id=oid))
            except Exception:
                pass

    def _read_log_order(self) -> dict | None:
        """Read last uncancelled order from log file — pure query."""
        log_path = LOG_DIR / f"exec_{self._symbol.replace('-', '_')}_{self._tf}.jsonl"
        if not log_path.exists():
            return None
        try:
            lines = [json.loads(ln) for ln in log_path.read_text().splitlines() if ln.strip()]
        except Exception:
            return None

        cancelled_ids: set[str] = set()
        last_order: dict | None = None
        for entry in lines:
            etype = entry.get("event", "")
            p = entry.get("payload", {}) if isinstance(entry.get("payload"), dict) else {}
            if etype == "order":
                last_order = entry
            elif etype == "cancel":
                oid = p.get("ord_id", "")
                if oid:
                    cancelled_ids.add(oid)

        if last_order is None:
            return None
        p = last_order.get("payload", {})
        if not isinstance(p, dict):
            return None
        ord_id = p.get("ord_id", "")
        if not ord_id or ord_id in cancelled_ids:
            return None

        orig_placed_at = float(p.get("placed_at", 0))
        if orig_placed_at > 0 and (time.time() - orig_placed_at) > self._timeout:
            return None

        return {
            "ordId": ord_id,
            "side": p.get("side", "buy"),
            "sz": str(p.get("sz", 0)),
            "px": str(p.get("px", 0)),
            "signal": str(p.get("signal", 0)),
            "cTime": str(int(orig_placed_at * 1000)) if orig_placed_at > 0 else "",
        }

    # -- execution ------------------------------------------------------------

    def _apply_amend(self, decision: Decision, _signal: float) -> None:
        """Apply dynamic risk adjustments without changing state."""
        if not self._position:
            return
        # local adjustments (no API call)
        if decision.new_stop is not None:
            self._position.stop_price = decision.new_stop
        if decision.new_target is not None:
            self._position.target_price = decision.new_target
        # exchange adjustments (amend order)
        if (decision.amend_px is not None or decision.amend_sz is not None) and self._client:
            try:
                o = self._position.order
                self._client.amend(
                    o.inst_id,
                    o.ord_id,
                    new_sz=str(round(decision.amend_sz, 8))
                    if decision.amend_sz is not None
                    else None,
                    new_px=str(round(decision.amend_px, 1))
                    if decision.amend_px is not None
                    else None,
                )
                if decision.amend_sz is not None:
                    o.sz = decision.amend_sz
                if decision.amend_px is not None:
                    o.px = decision.amend_px
            except Exception as e:
                self._log("order_error", ErrorPayload(error=str(e)))
        # partial scale-out
        if decision.scale_out_pct > 0 and self._client:
            try:
                exit_sz = self._position.order.sz * decision.scale_out_pct
                self._client.place(
                    self._symbol,
                    "sell" if self._position.order.side == "buy" else "buy",
                    str(round(exit_sz, 8)),
                    ord_type="market",
                )
                self._position.order.sz *= 1 - decision.scale_out_pct
            except Exception as e:
                self._log("order_error", ErrorPayload(error=str(e)))
        self._log("skip", SkipPayload(reason=decision.detail))

    def _fetch_signal(self, signal_source: SignalSource | None) -> SignalResult | None:
        if signal_source:
            sr = signal_source(self._symbol, self._tf)
            if sr:
                (SIGNAL_DIR / f"{self._symbol.replace('-', '_')}_{self._tf}.json").write_text(
                    sr.model_dump_json(indent=2)
                )
            return sr
        sf = SIGNAL_DIR / f"{self._symbol.replace('-', '_')}_{self._tf}.json"
        return SignalResult.model_validate(json.loads(sf.read_text())) if sf.exists() else None

    def _compute_size(self, clipped: float, entry_px: float) -> float:
        peak = max(self._equity) if self._equity else self._capital
        dd = (peak - self._equity[-1]) / peak if peak > 0 else 0.0
        eff_lev = self._leverage * (0.5 if dd > 0.15 else 1.0)
        ml = self._risk.max_leverage
        notional = self._capital * self._max_pos * min(abs(clipped), ml) * eff_lev
        if notional < 10.0:
            return 0.0
        # Contracts = notional / (entry_px * ct_val), min 1 contract
        contract_notional = entry_px * self._ct_val
        contracts = max(1.0, notional / max(contract_notional, 1))
        return round(contracts)

    def check_fill(self) -> FillFacts:
        """Query OKX for fill status — pure query, no mutation."""
        if not self._position or not self._client or self._dry:
            return FillFacts()
        if self._position.fill_status not in (FillStatus.PLACED, FillStatus.PARTIAL):
            return FillFacts()
        try:
            pending = self._client.pending()
            for p in pending:
                if p.get("ordId") == self._position.order.ord_id:
                    filled_sz = float(p.get("fillSz", 0))
                    if filled_sz >= self._position.order.sz:
                        return FillFacts(filled=True, filled_px=float(p.get("fillPx", 0)),
                                         filled_sz=filled_sz)
                    if filled_sz > 0:
                        return FillFacts(partial=True)
                    return FillFacts()
            # Order not in pending + older than 5 min → missing
            age = time.time() - self._position.order.placed_at
            if age > 300:
                return FillFacts(missing=True)
        except Exception:
            pass
        return FillFacts()

    def status(self) -> dict:
        return {
            "symbol": self._symbol,
            "tf": self._tf,
            "capital": self._capital,
            "leverage": self._leverage,
            "dry": self._dry,
            "trades": self._trade_count,
            "state": self._state.value,
            "active": self._position.order.ord_id if self._position else None,
        }

    # -- internals ------------------------------------------------------------

    def _place(
        self,
        side: str,
        sz: float,
        px: float,
        signal: float,
        obi,
        flow: float,
        force_market: bool = False,
    ) -> OrderPayload | None:
        sz, px = round(sz, 8), round(px, 1)
        atr = px * 0.02
        parent = PositionState.enter_long if side == "buy" else PositionState.enter_short
        pos = parent(px, atr, self._risk, int(time.time() * 1000))
        pos.order = OrderPayload(
            inst_id=self._symbol,
            side=side,
            sz=sz,
            px=px,
            placed_at=time.time(),
            signal=signal,
            obi=getattr(obi, "imbalance_5", 0) if obi else 0,
            ofi_flow=flow,
            status="placed",
        )
        if self._dry:
            pos.order.ord_id = "dry_" + str(int(time.time()))
            pos.order.status = "simulated"
            pos.fill_status = FillStatus.SIMULATED
            self._position = pos
            self._trade_count += 1
            self._log_order()
            return pos.order
        if self._error_streak >= 3:
            self._log("order_error", ErrorPayload(error="circuit_breaker"))
            return None
        try:
            # Entries: limit orders (signal verification).  Exits: market (risk guarantee).
            otype = (
                "market"
                if force_market
                else (self._ord_type or ("post_only" if self._post_only else "limit"))
            )
            assert self._client is not None, "_place called without client"
            if otype == "market":
                resp = self._client.place(
                    self._symbol, side, str(sz), ord_type=otype, td_mode=self._td_mode
                )
            else:
                resp = self._client.place(
                    self._symbol, side, str(sz), ord_type=otype, px=str(px), td_mode=self._td_mode
                )
            pos.order.ord_id = resp.get("ordId", "")
            self._position = pos
            self._trade_count += 1
            self._error_streak = 0
            self._log_order()
            return pos.order
        except Exception as e:
            self._log("order_error", ErrorPayload(error=str(e)))
            self._error_streak += 1
            return None

    def _cancel(self, reason: str) -> None:
        """Cancel current order.  Does NOT change state — caller handles that."""
        if not self._position:
            return
        oid = self._position.order.ord_id
        self._position.order.status, self._position.order.reason = "cancelled", reason
        self._log("cancel", CancelPayload(reason=reason, ord_id=oid))
        if not self._dry and self._client and oid:
            try:
                self._client.cancel(self._symbol, oid)
            except Exception:
                pass
        self._position = None

    def _log_order(self) -> None:
        if self._position:
            self._log("order", self._position.order)

    def _log_skip(self, reason: str) -> None:
        self._log("skip", SkipPayload(reason=reason))
        return None

    def _log(self, event: str, payload: LogPayload = SkipPayload()) -> None:
        line = LogLine(
            ts=int(time.time() * 1000),
            event=event,
            symbol=self._symbol,
            tf=self._tf,
            payload=payload,
        )
        log_dir = LOG_DIR / f"exec_{self._symbol.replace('-', '_')}_{self._tf}.jsonl"
        log_dir.open("a").write(line.model_dump_json() + "\n")
        print(str(line))


# ============================================================================
# 5. Portfolio runner
# ============================================================================


class PortfolioConfig(BaseModel):
    pairs: list[dict]
    dry_run: bool = True
    post_only: bool = True
    limit_timeout_sec: int = 0  # 0 = derive from timeframe (2x bar duration)
    max_frozen_pct: float = 0.50  # skip entries when > this fraction of balance is frozen
    capital: float = 100
    risk_pct: float = 0.03
    leverage: float = 1.0


class PortfolioRunner:
    def __init__(self, config: PortfolioConfig) -> None:
        self.config = config  # public �?needed by Summary.from_runner
        self._executors = {}
        for p in config.pairs:
            sym = p.get("exec_symbol", p.get("sig_symbol", p.get("symbol", "")))
            self._executors[f"{sym}_{p.get('tf', '4h')}"] = LiveExecutor(
                symbol=sym,
                timeframe=p.get("tf", "4h"),
                initial_capital=p.get("capital", config.capital),
                max_position_pct=p.get("risk_pct", config.risk_pct),
                leverage=p.get("leverage", config.leverage),
                ct_val=p.get("ct_val", 1.0),
                td_mode=p.get("td_mode", "isolated"),
                post_only=p.get("post_only", config.post_only),
                ord_type=p.get("ord_type", ""),
                limit_timeout_sec=config.limit_timeout_sec,
                dry_run=config.dry_run,
            )

    def step(
        self,
        source: SignalSource | None = None,
        signals: list[SignalResult] | None = None,
        tc: TradingClient | None = None,
    ) -> dict:
        """Run all executors.  If ``signals`` is provided (from
        ``compute_all_signals``), passes the matching SignalResult to
        each executor to avoid re-computation.

        If ``tc`` is provided, checks frozen-capital gate before placing:
        skips all IDLE executors if >50% of balance is frozen in pending orders.
        """
        # --- C1: frozen-capital gate ---
        if tc:
            n_pending = 0
            frozen = 0.0
            avail = 0.0
            try:
                n_pending = len(tc.pending())
                for b in tc.balance():
                    avail += float(b.get("availBal", 0))
                    frozen += float(b.get("frozenBal", 0))
            except Exception:
                pass
            total = avail + frozen
            # Gate fires if >max_frozen_pct of balance is frozen AND there
            # are actual pending orders (not testnet display artifacts).
            if n_pending > 0 and total > 0 and frozen > self.config.max_frozen_pct * total:
                for key, exe in self._executors.items():
                    if exe._state == State.IDLE:
                        exe._log_skip("frozen_cap_exceeded")
                return {}

        signal_map: dict[str, SignalResult] = {}
        if signals:
            for s in signals:
                signal_map[f"{s.symbol}_{s.timeframe}"] = s
        results = {}
        for key, exe in self._executors.items():
            sr = signal_map.get(key)
            results[key] = exe.step(signal_source=source, signal=sr)
        return results

    def compute_all_signals(self, source: SignalSource | None = None) -> list[SignalResult]:
        out = []
        for p in self.config.pairs:
            th = p.get("sig_threshold", 0.35)
            src = default_signal_source(sig_threshold=th)
            # Use sig_symbol for signal (spot), exec_symbol for execution (swap)
            sig_sym = p.get("sig_symbol", p.get("symbol", ""))
            exec_sym = p.get("exec_symbol", p.get("symbol", ""))
            tf = p.get("tf", "4h")
            try:
                s = src(sig_sym, tf)
            except Exception as e:
                # API down — try cached signal
                sf = SIGNAL_DIR / f"{sig_sym.replace('-', '_')}_{tf}.json"
                if sf.exists():
                    try:
                        s = SignalResult.model_validate(json.loads(sf.read_text()))
                        s.symbol = exec_sym
                        print(f"  {exec_sym:20s} sig={s.signal:+.4f} th={s.threshold:.3f} (cached)")
                        out.append(s)
                    except Exception:
                        print(f"  {exec_sym:20s} ERROR: API down, no cache")
                else:
                    print(f"  {exec_sym:20s} ERROR: API down, no cache ({e})")
                continue
            if s:
                # Override symbol to exec_sym so signal_map lookup in step()
                # matches executor keys (which use exec_symbol).
                s.symbol = exec_sym
                out.append(s)
                print(f"  {exec_sym:20s} sig={s.signal:+.4f} th={s.threshold:.3f}")
        return out

    def status(self) -> dict:
        return {k: e.status() for k, e in self._executors.items()}

    def write_summary(self, tc: TradingClient, pre_usdt: float, pre_pos: int) -> Summary:
        summary = Summary.from_runner(self, tc, pre_usdt, pre_pos)
        summary.write_to(tc=tc)
        return summary


# ============================================================================
# 6. Reporting
# ============================================================================


def asset_report(symbol: str = "ETH-USDT", timeframe: str = "4h") -> AssetReport:
    import polars as pl

    from qooi.exchange.eval import compute_metrics

    fname = LOG_DIR / f"exec_{symbol.replace('-', '_')}_{timeframe}.jsonl"
    if not fname.exists():
        return AssetReport(error="no log file")
    try:
        raw = [json.loads(line) for line in fname.read_text().splitlines() if line.strip()]
    except Exception:
        return AssetReport(error="log file corrupted")
    orders = [LogLine.model_validate(o) for o in raw if o.get("event") == "order"]
    if not orders:
        return AssetReport(error="no trades yet")

    # Extract OrderPayloads in chronological order
    ops: list[OrderPayload] = []
    for o in orders:
        p = o.payload
        if isinstance(p, OrderPayload):
            ops.append(p)
    if not ops:
        return AssetReport(error="no trades yet")

    # Pair buys with sells for PnL.  Unpaired buys are open positions.
    # For spot: buy = entry, sell = exit.
    trades = []
    open_buys: list[OrderPayload] = []
    for p in ops:
        if p.side == "buy":
            open_buys.append(p)
        elif p.side == "sell" and open_buys:
            entry = open_buys.pop(0)
            pnl = (p.px / entry.px - 1) if entry.px > 0 else 0.0
            trades.append(
                {
                    "entry_time": 0,
                    "exit_time": 0,
                    "side": entry.side,
                    "entry_price": entry.px,
                    "exit_price": p.px,
                    "pnl": pnl,
                    "reason": "sold",
                }
            )

    n_closed = len(trades)
    n_open = len(open_buys)
    if n_closed == 0 and n_open == 0:
        return AssetReport(
            trades=0, win_rate_pct=0.0, sharpe=0.0, total_return_pct=0.0, max_drawdown_pct=0.0
        )

    if n_closed == 0:
        return AssetReport(
            trades=n_open,
            win_rate_pct=0.0,
            sharpe=0.0,
            total_return_pct=0.0,
            max_drawdown_pct=0.0,
            error=f"{n_open} open position{'s' if n_open > 1 else ''}, no closed trades",
        )

    trade_df = pl.DataFrame(trades)
    initial = 1000.0
    eq = pl.Series(
        [initial] + [initial * (1.0 + float(t["pnl"])) for t in trades], dtype=pl.Float64
    )
    eq_df = pl.DataFrame({"portfolio_value": eq, "returns": eq.pct_change().fill_null(0.0)})
    m = compute_metrics(eq_df, trades=trade_df)
    # Total trades = all orders placed (closed + open)
    return AssetReport(
        trades=n_closed + n_open,
        win_rate_pct=m.win_rate_pct,
        sharpe=round(m.sharpe_ratio, 2),
        total_return_pct=round(m.total_return_pct, 1),
        max_drawdown_pct=round(m.max_drawdown_pct, 1),
        error=f"closed={n_closed} open={n_open}" if n_open > 0 else None,
    )
