"""OKX trading — orders, signals, execution, portfolio management.

Typed data layer using pydantic — no raw dicts in public-facing API.
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

from qooi.exchange.backtest import RiskConfig

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
# 1. Trading client — thin SDK wrapper
# ============================================================================


class TradingClient:
    _RETRY_ATTEMPTS = 3
    _RETRY_DELAY = 1.0

    def __init__(self) -> None:
        from okx.Account import AccountAPI
        from okx.Trade import TradeAPI

        if os.getenv("OKX_ENV"):
            load_okx_env()
        elif not os.getenv("OKX_API_KEY") and not os.getenv("OKX_API_KEY_TEST"):
            load_okx_env()
        flag = os.getenv("OKX_FLAG", "1")
        k = os.getenv("OKX_API_KEY") or os.getenv("OKX_API_KEY_TEST", "")
        s = os.getenv("OKX_SECRET_KEY") or os.getenv("OKX_SECRET_KEY_TEST", "")
        p = os.getenv("OKX_PASSPHRASE") or os.getenv("OKX_PASSPHRASE_TEST", "")
        if not k:
            raise RuntimeError("Missing OKX_API_KEY — call load_okx_env() or set envvars")
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

    # -- single-asset snapshot (replaces 4 separate REST calls) -----------------

    def snapshot(self) -> ExchangeSnapshot:
        """Query all exchange state in one go."""

        def _list(fn, *args, **kwargs):
            resp = fn(*args, **kwargs)
            if int(resp.get("code", -1)) != 0:
                raise RuntimeError(
                    f"OKX error [{resp.get('code')}]: {resp.get('msg')} data={resp.get('data')}"
                )
            return resp.get("data", []) or []

        orders = self._retry(lambda: _list(self._trade.get_order_list))
        positions = self._retry(lambda: _list(self._account.get_positions))
        balance_data = self._retry(lambda: _list(self._account.get_account_balance))
        algos = self._retry(lambda: _list(self._trade.order_algos_list, ordType="conditional,oco"))

        details = balance_data[0].get("details", []) if balance_data else []

        usdt_avail = 0.0
        usdt_frozen = 0.0
        for d in details:
            if d.get("ccy") == "USDT":
                usdt_avail = float(d.get("availBal", 0))
                usdt_frozen = float(d.get("frozenBal", 0))
                break

        return ExchangeSnapshot(
            orders=orders,
            positions=positions or [],
            algo_orders=algos or [],
            usdt_balance=usdt_avail,
            usdt_frozen=usdt_frozen,
        )

    # -- order operations -------------------------------------------------------

    def place(
        self,
        inst_id: str,
        side: str,
        sz: str,
        ord_type: str = "post_only",
        px: str | None = None,
        td_mode: str = "isolated",
        cl_ord_id: str = "",
        attach_algo_ords: list[dict] | None = None,
    ) -> dict:
        params = {"instId": inst_id, "side": side, "ordType": ord_type, "sz": sz, "tdMode": td_mode}
        if px:
            params["px"] = px
        if cl_ord_id:
            params["clOrdId"] = cl_ord_id
        if attach_algo_ords:
            params["attachAlgoOrds"] = attach_algo_ords
        return self._okx(self._trade.place_order(**params))

    def cancel(self, inst_id: str, ord_id: str) -> dict:
        return self._okx(self._trade.cancel_order(instId=inst_id, ordId=ord_id))

    def amend(
        self, inst_id: str, ord_id: str, new_sz: str | None = None, new_px: str | None = None
    ) -> dict:
        params: dict = {"instId": inst_id, "ordId": ord_id}
        if new_sz is not None:
            params["newSz"] = new_sz
        if new_px is not None:
            params["newPx"] = new_px
        return self._okx(self._trade.amend_order(**params))

    def cancel_algo(self, inst_id, algo_id):
        return self._okx(self._trade.cancel_algo_order(instId=inst_id, algoId=algo_id))

    def amend_algo(
        self,
        inst_id: str,
        algo_id: str,
        new_sl_trigger_px: str = "",
        new_sl_ord_px: str = "",
        new_tp_trigger_px: str = "",
        new_tp_ord_px: str = "",
    ) -> dict:
        params: dict = {"instId": inst_id, "algoId": algo_id}
        if new_sl_trigger_px:
            params["newSlTriggerPx"] = new_sl_trigger_px
        if new_sl_ord_px:
            params["newSlOrdPx"] = new_sl_ord_px
        if new_tp_trigger_px:
            params["newTpTriggerPx"] = new_tp_trigger_px
        if new_tp_ord_px:
            params["newTpOrdPx"] = new_tp_ord_px
        return self._okx(self._trade.amend_algo_order(**params))

    # -- position / account -----------------------------------------------------

    def close_position(self, inst_id, mgn_mode="isolated"):
        return self._okx(self._trade.close_positions(instId=inst_id, mgnMode=mgn_mode))

    def set_leverage(self, inst_id, lever, mgn_mode="isolated"):
        return self._okx(
            self._account.set_leverage(instId=inst_id, lever=str(lever), mgnMode=mgn_mode)
        )

    def balance(self, ccy: str | None = None) -> list:
        p = {} if not ccy else {"ccy": ccy}
        return self._okx(self._account.get_account_balance(**p)).get("details", [])

    def positions(self) -> list:
        return self._okx(self._account.get_positions()).get("posData", [])


# ============================================================================
# 2. Exchange snapshot — full state from OKX (zero local data)
# ============================================================================


@dataclass
class ExchangeSnapshot:
    """Full exchange state queried at invocation start — single source of truth.

    Replaces 4 separate REST calls (pending + positions + balance + algos).
    No local persistence — everything comes from OKX.
    """

    orders: list[dict]  # GET /trade/orders-pending
    positions: list[dict]  # GET /account/positions
    algo_orders: list[dict]  # GET /trade/orders-algo-pending
    usdt_balance: float  # USDT availBal from GET /account/balance
    usdt_frozen: float = 0.0  # USDT frozenBal from GET /account/balance

    def order_for_symbol(self, inst_id: str) -> dict | None:
        for o in self.orders:
            if o.get("instId") == inst_id:
                return o
        return None

    def position_for_symbol(self, inst_id: str) -> dict | None:
        for p in self.positions:
            if p.get("instId") == inst_id:
                return p
        return None

    def algos_for_symbol(self, inst_id: str) -> list[dict]:
        return [a for a in self.algo_orders if a.get("instId") == inst_id]


# ============================================================================
# 3. Pure data models — all pydantic, zero side-effects
# ============================================================================

# -- signals ---------------------------------------------------------------


class SignalResult(BaseModel):
    symbol: str
    timeframe: str
    timestamp: int
    signal: float
    flow: float = 0.0
    threshold: float = 0.0
    atr: float = 0.0
    computed_at: int = 0

    @classmethod
    def from_dataframe(cls, symbol: str, timeframe: str, df) -> SignalResult:
        sv = float(df["signal"][-1] or 0.0)
        fv = float(df["ofi_flow_score"][-1] or 0.0) if "ofi_flow_score" in df.columns else 0.0
        atr_val = float(df["atr_14"][-1] or 0.0) if "atr_14" in df.columns else 0.0
        return cls(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=int(df["timestamp"][-1]),
            signal=round(sv, 4),
            flow=round(fv, 4),
            atr=round(atr_val, 2),
            computed_at=int(time.time()),
        )


# -- log payloads ----------------------------------------------------------


class SkipPayload(BaseModel):
    reason: str = ""


class SignalPayload(BaseModel):
    signal: float = 0.0
    obi_5: float = 0.0
    ofi_flow: float = 0.0
    cl_ord_id: str = ""


class CancelPayload(BaseModel):
    reason: str = ""
    ord_id: str = ""


class ErrorPayload(BaseModel):
    error: str = ""


class OrderPayload(BaseModel):
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
    cl_ord_id: str = ""


LogPayload = SignalPayload | OrderPayload | SkipPayload | CancelPayload | ErrorPayload


# -- enums -----------------------------------------------------------------


class FillStatus(enum.StrEnum):
    PLACED = "placed"
    PARTIAL = "partial_fill"
    FILLED = "filled"
    SIMULATED = "simulated"


class State(enum.StrEnum):
    IDLE = "idle"
    PENDING = "pending"
    ACTIVE = "active"


# -- position state (preserved for backtest.py compatibility) --------------


class PositionState(BaseModel):
    """Bundles active order + risk management into one state object.

    Used by both live executor and backtest engine."""

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


# -- decision (pure output of state machine) -------------------------------


@dataclass
class Decision:
    action: str  # "enter" | "exit" | "amend" | "skip"
    new_state: State = State.IDLE
    side: str = ""
    sz: float = 0.0
    entry_px: float = 0.0
    exit_px: float = 0.0
    detail: str = ""

    # risk parameters set at entry
    stop_px: float | None = None
    target_px: float | None = None

    # amendments
    new_stop: float | None = None
    new_target: float | None = None
    scale_out_pct: float = 0.0
    amend_px: float | None = None
    amend_sz: float | None = None

    # algo trailing
    amend_sl_trigger_px: str = ""
    amend_sl_ord_px: str = ""

    @classmethod
    def exit(cls, reason: str, px: float = 0.0, new_state: State = State.IDLE) -> Decision:
        return cls(action="exit", new_state=new_state, detail=reason, exit_px=px)

    @classmethod
    def enter(
        cls,
        side: str,
        sz: float,
        entry_px: float,
        *,
        stop_px: float | None = None,
        target_px: float | None = None,
    ) -> Decision:
        return cls(
            action="enter",
            new_state=State.PENDING,
            side=side,
            sz=sz,
            entry_px=entry_px,
            stop_px=stop_px,
            target_px=target_px,
        )

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
        amend_sl_trigger_px: str = "",
        amend_sl_ord_px: str = "",
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
            amend_sl_trigger_px=amend_sl_trigger_px,
            amend_sl_ord_px=amend_sl_ord_px,
        )


# -- log line --------------------------------------------------------------


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
                zz = getattr(p, "sz", 0)
                xx = getattr(p, "px", 0)
                oid = getattr(p, "ord_id", "?")
                return f"[{t}] {s} {tf} | ORDER {sd:4s} sz={zz:.6f} px={xx:.1f} id={oid}"
            case "cancel":
                oid = getattr(p, "ord_id", "")
                return f"[{t}] {s} {tf} | CANCEL {getattr(p, 'reason', '?')} id={oid}"
            case "order_error" | "error":
                return f"[{t}] {s} {tf} | ERROR {getattr(p, 'error', '?')}"
            case _:
                return p.model_dump_json()


# -- reporting -------------------------------------------------------------


class AssetReport(BaseModel):
    error: str | None = None
    trades: int = 0
    win_rate_pct: float = 0.0
    sharpe: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0


class Summary(BaseModel):
    pre_usdt: float = 0.0
    post_usdt: float = 0.0
    usdt_change: float = 0.0
    total_upl: float = 0.0
    total_margin: float = 0.0
    positions: list[dict] = []
    assets: dict[str, AssetReport] = {}

    @classmethod
    def from_snapshot(
        cls, snap: ExchangeSnapshot, pre_usdt: float, pairs: list[tuple[str, str]]
    ) -> Summary:
        positions: list[dict] = []
        total_upl = 0.0
        total_margin = 0.0
        for p in snap.positions:
            inst = p.get("instId", "?")
            pos_sz = float(p.get("pos", 0))
            upl = float(p.get("upl", 0))
            margin = float(p.get("margin", 0))
            avg_px = float(p.get("avgPx", 0))
            mark_px = float(p.get("markPx", 0))
            side = "long" if pos_sz > 0 else "short"
            positions.append(
                dict(
                    inst=inst,
                    side=side,
                    sz=abs(pos_sz),
                    avg_px=avg_px,
                    mark_px=mark_px,
                    upl=upl,
                    margin=margin,
                )
            )
            total_upl += upl
            total_margin += margin
        return cls(
            pre_usdt=pre_usdt,
            post_usdt=snap.usdt_balance,
            usdt_change=snap.usdt_balance - pre_usdt,
            total_upl=total_upl,
            total_margin=total_margin,
            positions=positions,
            assets={s: asset_report(s, tf) for s, tf in pairs},
        )

    def write_to(self, fh: io.TextIOBase | None = None) -> None:
        if fh is None:
            fh = sys.stdout
        fh.write("## qooi Portfolio\n\n")
        total_value = self.post_usdt + self.total_margin + self.total_upl
        yield_pct = (total_value / self.pre_usdt - 1) * 100 if self.pre_usdt > 0 else 0.0
        fh.write(f"Total: ${total_value:,.2f} ({yield_pct:+.2f}% since inception)\n")
        fh.write(f"  USDT free:   ${self.post_usdt:,.2f}\n")
        fh.write(f"  USDT margin: ${self.total_margin:,.2f}\n")
        if self.positions:
            fh.write("\n  Positions:\n")
            for p in self.positions:
                pnl_pct = (p["upl"] / p["margin"] * 100) if p["margin"] > 0 else 0.0
                fh.write(
                    f"    {p['inst']:20s} {p['side']:5s} "
                    f"{p['sz']}ct @ {p['avg_px']:,.1f} -> {p['mark_px']:,.1f}  "
                    f"upl=${p['upl']:+,.2f} ({pnl_pct:+.1f}%)\n"
                )
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
# 4. Signal source
# ============================================================================

SignalSource = Callable[[str, str], SignalResult | None]


def default_signal_source(sig_threshold: float = 0.35) -> SignalSource:
    from qooi.exchange.market import MarketData

    _md = MarketData("okx")

    def _src(symbol: str, timeframe: str) -> SignalResult | None:
        from qooi.exchange.indicator import add_indicators
        from qooi.strategies.flow_pipeline import (
            add_ofi_flow_columns,
            add_regime_features,
            apply_regime_gate,
        )

        df = _md.candles(symbol, timeframe=timeframe, limit=500, cache=True)
        if df.is_empty():
            return None
        df = add_indicators(df)
        df = add_regime_features(df)
        df = add_ofi_flow_columns(df)
        df = apply_regime_gate(df, signal_col="ofi_flow_score")  # zero signal in strong trends

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
            atr=round(float(df["atr_14"][-1] or 0.0), 2),
            computed_at=int(time.time()),
        )

    return _src


# ============================================================================
# 5. Stateless executor — single-step, zero local state
# ============================================================================


@dataclass
class AssetConfig:
    """Immutable per-asset configuration."""

    symbol: str  # exec symbol: "ETH-USDT-SWAP"
    sig_symbol: str = ""  # signal symbol: "ETH-USDT"
    timeframe: str = "4h"
    capital: float = 500.0
    max_risk_pct: float = 0.50
    leverage: float = 2.0
    ct_val: float = 0.1
    atr_stop_mult: float = 2.0
    atr_target_mult: float = 3.0
    trail_activation_mult: float = 2.0
    trail_distance_mult: float = 1.0
    max_bars_held: int = 0
    signal_threshold: float = 0.25
    ord_type: str = "post_only"
    td_mode: str = "isolated"
    limit_timeout_sec: int = 0


@dataclass
class ReconstructedState:
    """Full state reconstructed from ExchangeSnapshot + SignalResult.

    This is the ONLY state. Nothing persists across invocations.
    """

    state: State = State.IDLE
    symbol: str = ""

    # order
    ord_id: str = ""
    cl_ord_id: str = ""
    side: str = ""
    sz: str = ""
    acc_fill_sz: str = ""
    ord_px: str = ""
    ord_type: str = ""
    ord_state: str = ""
    ord_ctime: str = ""

    # position
    pos_id: str = ""
    pos_side: str = ""
    pos_sz: str = ""
    avg_px: str = ""
    mark_px: str = ""
    upl: str = ""
    margin: str = ""

    # attached algo orders
    algo_sl_id: str = ""
    algo_tp_id: str = ""
    sl_trigger_px: str = ""
    sl_ord_px: str = ""
    tp_trigger_px: str = ""
    tp_ord_px: str = ""

    # signal
    signal: float = 0.0
    flow: float = 0.0
    atr_estimate: float = 0.0

    age_sec: float = 0.0
    bars_held: int = 0

    @property
    def side_dir(self) -> int:
        if self.has_position:
            if self.pos_side == "net":
                return 1 if float(self.pos_sz or "0") > 0 else -1
            return 1 if self.pos_side == "long" else -1
        return 1 if self.side == "buy" else (-1 if self.side == "sell" else 0)

    @property
    def has_order(self) -> bool:
        return bool(self.ord_id and self.ord_state in ("live", "partially_filled"))

    @property
    def has_position(self) -> bool:
        return bool(self.pos_id and self.pos_sz and float(self.pos_sz) > 0)

    @property
    def is_filled(self) -> bool:
        return self.ord_state == "filled" or (
            self.has_position and self.acc_fill_sz and float(self.acc_fill_sz) >= float(self.sz) > 0
        )

    @property
    def entry_price(self) -> float:
        px = self.avg_px or self.ord_px
        return float(px) if px else 0.0


class StatelessExecutor:
    """Single-step decision engine. Reads exchange state, decides, executes.

    Invariants:
    1. One order per asset — state machine IDLE->PENDING->ACTIVE->IDLE
    2. Stop-loss + take-profit via OKX algo orders at placement time
    3. Trailing stop via algo order amendments
    4. Zero local state — everything from ExchangeSnapshot + SignalResult
    """

    def __init__(self, config: AssetConfig):
        self.cfg = config

    # -- public entry point -------------------------------------------------

    def step(
        self, snap: ExchangeSnapshot, signal: SignalResult, obi, tc: TradingClient | None
    ) -> tuple[Decision, ReconstructedState]:
        """Main entry point. Called once per GitHub Actions invocation."""
        state = self._reconstruct(snap, signal, obi)
        decision = self._decide(state, obi)
        self._execute(decision, state, obi, tc)
        self._log(state, decision)
        return decision, state

    # -- state reconstruction -----------------------------------------------

    def _reconstruct(self, snap: ExchangeSnapshot, signal: SignalResult, obi) -> ReconstructedState:
        order = snap.order_for_symbol(self.cfg.symbol)
        position = snap.position_for_symbol(self.cfg.symbol)
        algos = snap.algos_for_symbol(self.cfg.symbol)

        now_ms = int(time.time() * 1000)
        s = ReconstructedState(
            symbol=self.cfg.symbol,
            state=State.IDLE,
            signal=signal.signal,
            flow=signal.flow,
            atr_estimate=signal.atr if signal.atr > 0 else (obi.ask_price * 0.02 if obi else 0.0),
        )

        if order:
            s.state = State.PENDING
            s.ord_id = order.get("ordId", "")
            s.cl_ord_id = order.get("clOrdId", "")
            s.side = order.get("side", "")
            s.sz = order.get("sz", "0")
            s.acc_fill_sz = order.get("accFillSz", "0")
            s.ord_px = order.get("px", "0")
            s.ord_type = order.get("ordType", "")
            s.ord_state = order.get("state", "")
            s.ord_ctime = order.get("cTime", "")
            ctime = int(order.get("cTime", now_ms))
            s.age_sec = max(0, (now_ms - ctime) / 1000)

        if position:
            s.pos_id = position.get("posId", "")
            s.pos_side = position.get("posSide", "")
            s.pos_sz = position.get("pos", "0")
            s.avg_px = position.get("avgPx", "0")
            s.mark_px = position.get("markPx", "0")
            s.upl = position.get("upl", "0")
            s.margin = position.get("margin", "0")
            # Derive age from position creation time when no pending order
            if not order:
                p_ctime = int(position.get("cTime", now_ms))
                s.age_sec = max(0, (now_ms - p_ctime) / 1000)

        if s.has_position and not s.has_order:
            s.state = State.ACTIVE
            if not s.side and position:
                ps = position.get("posSide", "")
                if ps == "long":
                    s.side = "buy"
                elif ps == "short":
                    s.side = "sell"
                elif ps == "net":
                    s.side = "buy" if float(position.get("pos", "0")) > 0 else "sell"
                else:
                    s.side = "buy" if float(position.get("pos", "0")) > 0 else "sell"

        if s.ord_state == "filled" or s.is_filled:
            s.state = State.ACTIVE

        if s.ord_state == "partially_filled":
            s.state = State.PENDING

        for a in algos:
            sl = a.get("slTriggerPx", "")
            tp = a.get("tpTriggerPx", "")
            aid = a.get("algoId", "")
            if sl:
                s.algo_sl_id = aid
                s.sl_trigger_px = sl
                s.sl_ord_px = a.get("slOrdPx", "")
            if tp:
                s.algo_tp_id = aid
                s.tp_trigger_px = tp
                s.tp_ord_px = a.get("tpOrdPx", "")

        bar_sec = {"1h": 3600, "4h": 14400, "1d": 86400}.get(self.cfg.timeframe.lower(), 14400)
        if s.age_sec > 0 and s.has_position:
            s.bars_held = max(0, int(s.age_sec / bar_sec))

        return s

    # -- decision engine (pure function, no side effects) -------------------

    def _decide(self, s: ReconstructedState, obi) -> Decision:
        abs_sig = abs(s.signal)

        if s.state == State.IDLE:
            return self._decide_idle(s, obi, abs_sig)
        if s.state == State.PENDING:
            return self._decide_pending(s, obi)
        if s.state == State.ACTIVE:
            return self._decide_active(s, obi)
        return Decision.skip("unknown_state")

    def _decide_idle(self, s: ReconstructedState, obi, abs_sig: float) -> Decision:
        if abs_sig < self.cfg.signal_threshold:
            return Decision.skip("weak_signal")

        side = "buy" if s.signal > 0 else "sell"
        entry_px = obi.ask_price if side == "buy" else obi.bid_price if obi else 0.0
        d = 1 if side == "buy" else -1
        atr = s.atr_estimate

        stop_px = round(entry_px - d * self.cfg.atr_stop_mult * atr, 2)
        target_px = round(entry_px + d * self.cfg.atr_target_mult * atr, 2)

        risk_per_ct = abs(entry_px - stop_px) * self.cfg.ct_val
        if risk_per_ct <= 0:
            return Decision.skip("zero_risk")

        max_risk = self.cfg.capital * self.cfg.max_risk_pct
        sz = max(1, int(max_risk / risk_per_ct))

        # Margin constraint: ensure notional won't exceed available capital × leverage.
        # notional = sz × ct_val × entry_px / leverage
        notional_per_ct = self.cfg.ct_val * entry_px
        max_notional = self.cfg.capital * self.cfg.leverage
        max_sz = int(max_notional / max(notional_per_ct, 1e-9))
        if max_sz < 1:
            return Decision.skip("insufficient_margin")
        sz = min(sz, max_sz)
        sz = max(1, sz)

        return Decision.enter(
            side=side, sz=float(sz), entry_px=entry_px, stop_px=stop_px, target_px=target_px
        )

    def _decide_pending(self, s: ReconstructedState, obi) -> Decision:
        timeout = self.cfg.limit_timeout_sec or {
            "1h": 7200,
            "4h": 28800,
            "1d": 172800,
        }.get(self.cfg.timeframe.lower(), 28800)

        if s.is_filled:
            return Decision.skip("order_filled", new_state=State.ACTIVE)

        if s.age_sec > timeout:
            px = obi.bid_price if s.side_dir > 0 else obi.ask_price if obi else 0.0
            return Decision.exit("timeout", px=px)

        if s.signal * s.side_dir < 0:
            px = obi.bid_price if s.side_dir > 0 else obi.ask_price if obi else 0.0
            return Decision.exit("signal_flipped", px=px)

        if obi and s.ord_px:
            current = obi.ask_price if s.side_dir > 0 else obi.bid_price
            px = float(s.ord_px)
            if px > 0 and abs(current - px) / px > 0.005:
                new_px = round(px + s.side_dir * (current - px) * 0.2, 2)
                return Decision.amend("price_chase", amend_px=new_px, new_state=State.PENDING)

        return Decision.skip("order_outstanding", new_state=State.PENDING)

    def _decide_active(self, s: ReconstructedState, obi) -> Decision:
        if self.cfg.max_bars_held > 0 and s.bars_held >= self.cfg.max_bars_held:
            px = obi.bid_price if s.side_dir > 0 else obi.ask_price if obi else 0.0
            return Decision.exit("time", px=px)

        if s.signal * s.side_dir < 0:
            px = obi.bid_price if s.side_dir > 0 else obi.ask_price if obi else 0.0
            return Decision.exit("signal_flipped", px=px)

        # SL/TP are managed by OKX attached algo orders — no need to duplicate checks.
        # Only manage trailing stop via algo amendments.

        mark = float(s.mark_px) if s.mark_px else 0.0
        if mark > 0:
            sl = float(s.sl_trigger_px) if s.sl_trigger_px else 0.0
            d = s.side_dir
            atr = s.atr_estimate
            entry = s.entry_price

            if entry > 0 and atr > 0 and sl > 0:
                profit_atr = d * (mark - entry) / atr
                if profit_atr >= self.cfg.trail_activation_mult:
                    new_sl = round(mark - d * self.cfg.trail_distance_mult * atr, 2)
                    if d * (new_sl - sl) > 0:
                        new_sl_s = str(new_sl)
                        return Decision.amend(
                            "trail_update",
                            new_state=State.ACTIVE,
                            amend_sl_trigger_px=new_sl_s,
                            amend_sl_ord_px=new_sl_s,
                        )

        return Decision.skip("holding", new_state=State.ACTIVE)

    # -- execution (side effects only) --------------------------------------

    def _execute(self, d: Decision, s: ReconstructedState, obi, tc: TradingClient | None) -> None:
        if tc is None:
            return

        if d.action == "enter":
            self._execute_enter(d, tc)
        elif d.action == "exit":
            self._execute_exit(d, s, tc)
        elif d.action == "amend":
            self._execute_amend(d, s, tc)

    def _execute_enter(self, d: Decision, tc: TradingClient) -> None:
        sz = str(int(d.sz))
        px = str(d.entry_px) if self.cfg.ord_type != "market" else ""
        side = d.side

        attach = []
        if d.stop_px is not None and d.target_px is not None:
            attach = [
                {
                    "slTriggerPx": str(d.stop_px),
                    "slOrdPx": "-1",
                    "tpTriggerPx": str(d.target_px),
                    "tpOrdPx": "-1",
                    "cxlOnClosePos": "true",
                }
            ]

        try:
            resp = tc.place(
                inst_id=self.cfg.symbol,
                side=side,
                sz=sz,
                ord_type=self.cfg.ord_type,
                px=px,
                td_mode=self.cfg.td_mode,
                attach_algo_ords=attach if attach else None,
            )
            print(
                f"  ORDER {side} sz={sz} px={px} id={resp.get('ordId', '?')} "
                f"sl={d.stop_px} tp={d.target_px}"
            )
        except Exception as e:
            print(f"  ORDER FAILED: {e}")

    def _execute_exit(self, d: Decision, s: ReconstructedState, tc: TradingClient) -> None:
        if s.has_order:
            # PENDING exit: cancel limit order. Per OKX API, attached TP/SL are
            # auto-discarded when parent is cancelled before any fill.
            try:
                tc.cancel(self.cfg.symbol, s.ord_id)
                print(f"  CANCEL {s.ord_id} reason={d.detail}")
            except Exception:
                pass
            # Partial fill: cancel un-filled portion, then close the filled part.
            if not s.has_position:
                return

        # ACTIVE exit (or PENDING-exit after partial fill): close position
        # at market, then clean up any remaining algo orders.
        if s.has_position:
            try:
                tc.close_position(self.cfg.symbol)
                print(f"  CLOSE {s.pos_side} sz={s.pos_sz} reason={d.detail}")
            except Exception as e:
                print(f"  CLOSE FAILED: {e}")

        # Clean up algos — they should already be cxlOnClosePos=true, but be safe.
        if s.algo_sl_id and s.algo_sl_id != s.algo_tp_id:
            try:
                tc.cancel_algo(self.cfg.symbol, s.algo_sl_id)
            except Exception:
                pass
        if s.algo_tp_id:
            try:
                tc.cancel_algo(self.cfg.symbol, s.algo_tp_id)
            except Exception:
                pass

    def _execute_amend(self, d: Decision, s: ReconstructedState, tc: TradingClient) -> None:
        if (d.amend_px is not None or d.amend_sz is not None) and s.has_order:
            try:
                tc.amend(
                    self.cfg.symbol,
                    s.ord_id,
                    new_px=str(d.amend_px) if d.amend_px else "",
                    new_sz=str(int(d.amend_sz)) if d.amend_sz else "",
                )
                print(f"  AMEND ord={s.ord_id} px={d.amend_px} sz={d.amend_sz} reason={d.detail}")
            except Exception as e:
                print(f"  AMEND FAILED: {e}")

        if d.amend_sl_trigger_px and s.algo_sl_id:
            try:
                tc.amend_algo(
                    self.cfg.symbol,
                    s.algo_sl_id,
                    new_sl_trigger_px=d.amend_sl_trigger_px,
                    new_sl_ord_px=d.amend_sl_ord_px or d.amend_sl_trigger_px,
                )
                print(f"  TRAIL SL={d.amend_sl_trigger_px}")
            except Exception as e:
                print(f"  TRAIL FAILED: {e}")

        if d.new_stop is not None or d.new_target is not None:
            pass

    # -- logging ------------------------------------------------------------

    def _log(self, s: ReconstructedState, d: Decision) -> None:
        log_path = LOG_DIR / f"exec_{self.cfg.symbol.replace('-', '_')}_{self.cfg.timeframe}.jsonl"
        icon = {"enter": "ORDER", "exit": "EXIT", "amend": "AMEND", "skip": "skip"}
        params = [f"state={s.state.value}", f"act={icon.get(d.action, d.action)}"]
        if d.detail:
            params.append(f"({d.detail})")
        if s.has_position:
            params.append(f"upl={s.upl}")
        line = LogLine(
            ts=int(time.time() * 1000),
            event=d.action,
            symbol=self.cfg.symbol,
            tf=self.cfg.timeframe,
            payload=SkipPayload(reason=d.detail),
        )
        log_path.open("a").write(line.model_dump_json() + "\n")
        print(f"  {self.cfg.symbol:20s} {' '.join(params)}")


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

    ops: list[OrderPayload] = []
    for o in orders:
        p = o.payload
        if isinstance(p, OrderPayload):
            ops.append(p)
    if not ops:
        return AssetReport(error="no trades yet")

    trades = []
    open_buys: list[OrderPayload] = []
    for p in ops:
        if p.side == "buy":
            open_buys.append(p)
        elif p.side == "sell" and open_buys:
            entry = open_buys.pop(0)
            pnl = (p.px / entry.px - 1) if entry.px > 0 else 0.0
            trades.append(
                dict(
                    entry_time=0,
                    exit_time=0,
                    side=entry.side,
                    entry_price=entry.px,
                    exit_price=p.px,
                    pnl=pnl,
                    reason="sold",
                )
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
    return AssetReport(
        trades=n_closed + n_open,
        win_rate_pct=m.win_rate_pct,
        sharpe=round(m.sharpe_ratio, 2),
        total_return_pct=round(m.total_return_pct, 1),
        max_drawdown_pct=round(m.max_drawdown_pct, 1),
        error=f"closed={n_closed} open={n_open}" if n_open > 0 else None,
    )
