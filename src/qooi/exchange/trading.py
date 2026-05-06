"""OKX trading �?orders, signals, execution, portfolio management.

Typed data layer using pydantic �?no raw dicts in public-facing API.
Side-effectful operations (IO, network) are separated from pure data models.
"""

from __future__ import annotations

import enum
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
    load_dotenv(path) if path else load_dotenv()


LOG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
SIGNAL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "signals"
SIGNAL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 1. Trading client
# ============================================================================


class TradingClient:
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
            raise RuntimeError(f"OKX error: {resp.get('msg', resp)}")
        return resp.get(key, [{}])[0] if resp.get(key) else {}

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
        return self._okx(self._trade.place_order(**params))

    def cancel(self, inst_id: str, ord_id: str) -> dict:
        return self._okx(self._trade.cancel_order(instId=inst_id, ordId=ord_id))

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


@dataclass
class Decision:
    """Pure output of the state machine — no side effects."""

    action: str  # "exit" | "enter" | "skip"
    side: str = ""
    sz: float = 0.0
    entry_px: float = 0.0
    exit_px: float = 0.0
    detail: str = ""

    @classmethod
    def exit(cls, reason: str, px: float) -> Decision:
        return cls(action="exit", detail=reason, exit_px=px)

    @classmethod
    def enter(cls, side: str, sz: float, entry_px: float) -> Decision:
        return cls(action="enter", side=side, sz=sz, entry_px=entry_px)

    @classmethod
    def skip(cls, reason: str) -> Decision:
        return cls(action="skip", detail=reason)


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
                return f"[{t}] {s} {tf} | CANCEL {getattr(p, 'reason', '?')}"
            case "order_error" | "error":
                return f"[{t}] {s} {tf} | ERROR {getattr(p, 'error', '?')}"
            case _:
                return p.model_dump_json()


# -- position state --------------------------------------------------------


class FillStatus(enum.StrEnum):
    PLACED = "placed"
    PARTIAL = "partial_fill"
    FILLED = "filled"
    SIMULATED = "simulated"


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
    positions_pre: int = 0
    positions_post: int = 0
    assets: dict[str, AssetReport] = {}

    @classmethod
    def from_runner(cls, runner, tc: TradingClient, pre_usdt: float, pre_pos: int) -> Summary:
        post_usdt = float(tc.balance("USDT")[0].get("availBal", 0))
        post_pos = len(tc.positions())
        pairs = [(p["symbol"], p.get("tf", "4h")) for p in runner.config.pairs]
        return cls(
            pre_usdt=pre_usdt,
            post_usdt=post_usdt,
            usdt_change=post_usdt - pre_usdt,
            positions_pre=pre_pos,
            positions_post=post_pos,
            assets={s: asset_report(s, tf) for s, tf in pairs},
        )

    def write_to(self, fh: object = sys.stdout) -> None:
        fh.write("## qooi Portfolio\n\n")
        fh.write(f"USDT: {self.pre_usdt:.2f} �?{self.post_usdt:.2f} ({self.usdt_change:+.2f})\n")
        fh.write(f"Positions: {self.positions_pre} �?{self.positions_post}\n\n")
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


def default_signal_source() -> SignalSource:
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

        df = MarketData("okx").candles(symbol, timeframe=timeframe, limit=500)
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
# 4. Live executor
# ============================================================================


class LiveExecutor:
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
        self._client: TradingClient | None = None if dry_run else TradingClient()
        self._position: PositionState | None = None
        self._trade_count = 0
        self._equity = [initial_capital]

    # -- public ---------------------------------------------------------------

    def step(self, signal_source: SignalSource | None = None) -> OrderPayload | None:
        """Orchestrate one bar cycle: fetch → decide → execute."""
        self._log("cycle_start")
        sr = self._fetch_signal(signal_source)
        if not sr:
            return self._log_skip("no_signal")
        bar_ms = {"1h": 3600000, "4h": 14400000, "1d": 86400000}.get(self._tf, 14400000)
        if time.time() * 1000 - sr.timestamp > bar_ms * 1.5:
            return self._log_skip("stale_signal")

        obi = self._md.ob_snapshot(self._symbol)
        self._log(
            "signal",
            SignalPayload(signal=sr.signal, obi_5=round(obi.imbalance_5, 4), ofi_flow=sr.flow),
        )

        self._check_fill_status()
        decision = self._decide(sr, obi)
        if decision.action == "exit":
            if self._position and self._position.order.px > 0:
                d = 1 if self._position.order.side == "buy" else -1
                self._equity.append(
                    self._equity[-1] * (1 + d * (decision.exit_px / self._position.order.px - 1))
                )
            self._cancel(decision.detail)
        elif decision.action == "enter":
            return self._place(
                decision.side, decision.sz, decision.entry_px, sr.signal, obi, sr.flow
            )
        return None

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

    def _decide(self, sr: SignalResult, obi) -> Decision:
        """Pure state machine — no I/O, no network.  Returns a Decision."""
        cur_close = (
            obi.ask_price
            if self._position and self._position.order.side == "buy"
            else obi.bid_price
        )
        atr_est = obi.ask_price * 0.02

        if self._position:
            reason = self._position.check_exit(cur_close, atr_est, self._risk)
            if reason:
                return Decision.exit(reason, cur_close)
            if (
                self._position.order.status == "placed"
                and time.time() - self._position.order.placed_at > self._timeout
            ):
                return Decision.exit("timeout", cur_close)

        if abs(sr.signal) < 0.25:
            return Decision.skip("weak_signal")
        ml = self._risk.max_leverage
        clipped = max(-ml, min(ml, sr.signal)) if ml > 0 else 0.0
        if clipped == 0.0:
            return Decision.skip("clipped_to_zero")
        if self._position and self._position.fill_status.value in ("placed", "partial_fill"):
            return Decision.skip("order_outstanding")

        side = "buy" if clipped > 0 else "sell"
        entry_px = obi.ask_price if side == "buy" else obi.bid_price
        peak = max(self._equity) if self._equity else self._capital
        dd = (peak - self._equity[-1]) / peak if peak > 0 else 0.0
        eff_lev = self._leverage * (0.5 if dd > 0.15 else 1.0)
        sz = min(
            self._capital * self._max_pos * min(abs(clipped), ml) * eff_lev,
            self._capital * ml * eff_lev,
        ) / max(entry_px, 1)
        if sz < 0.00001:
            return Decision.skip("size_too_small")
        return Decision.enter(side, sz, entry_px)

    def _check_fill_status(self) -> None:
        """Query OKX for fill status of the active order."""
        if not self._position or not self._client or self._dry:
            return
        if self._position.fill_status not in (FillStatus.PLACED, FillStatus.PARTIAL):
            return
        try:
            pending = self._client.pending()
            for p in pending:
                if p.get("ordId") == self._position.order.ord_id:
                    filled_sz = float(p.get("fillSz", 0))
                    if filled_sz >= self._position.order.sz:
                        self._position.fill_status = FillStatus.FILLED
                        self._position.order.filled_sz = filled_sz
                        self._position.order.filled_px = float(p.get("fillPx", 0))
                    elif filled_sz > 0:
                        self._position.fill_status = FillStatus.PARTIAL
        except Exception:
            pass  # network flaky — don't crash on fill query

    def status(self) -> dict:
        return {
            "symbol": self._symbol,
            "tf": self._tf,
            "capital": self._capital,
            "leverage": self._leverage,
            "dry": self._dry,
            "trades": self._trade_count,
            "active": self._position.order.ord_id if self._position else None,
        }

    # -- internals ------------------------------------------------------------

    def _place(
        self, side: str, sz: float, px: float, signal: float, obi, flow: float
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
            obi=obi.imbalance_5,
            ofi_flow=flow,
            status="placed",
        )
        if self._dry:
            pos.order.ord_id = "dry_" + str(int(time.time()))
            pos.order.status = "simulated"
            pos.fill_status = FillStatus.SIMULATED
            self._position = pos
            self._log_order()
            self._trade_count += 1
            return pos.order
        try:
            otype = "post_only" if self._post_only else "limit"
            resp = self._client.place(self._symbol, side, str(sz), ord_type=otype, px=str(px))
            pos.order.ord_id = resp.get("ordId", "")
            self._position = pos
            self._log_order()
            self._trade_count += 1
            return pos.order
        except Exception as e:
            self._log("order_error", ErrorPayload(error=str(e)))
            return None

    def _cancel(self, reason: str) -> None:
        if not self._position:
            return
        self._position.order.status, self._position.order.reason = "cancelled", reason
        self._log("cancel", CancelPayload(reason=reason))
        if not self._dry and self._client:
            self._client.cancel(self._symbol, self._position.order.ord_id)
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
    limit_timeout_sec: int = 120
    capital: float = 100
    risk_pct: float = 0.03
    leverage: float = 1.0


class PortfolioRunner:
    def __init__(self, config: PortfolioConfig) -> None:
        self.config = config  # public �?needed by Summary.from_runner
        self._executors = {
            f"{p['symbol']}_{p.get('tf', '4h')}": LiveExecutor(
                symbol=p["symbol"],
                timeframe=p.get("tf", "4h"),
                initial_capital=p.get("capital", config.capital),
                max_position_pct=p.get("risk_pct", config.risk_pct),
                leverage=p.get("leverage", config.leverage),
                post_only=config.post_only,
                limit_timeout_sec=config.limit_timeout_sec,
                dry_run=config.dry_run,
            )
            for p in config.pairs
        }

    def step(self, source: SignalSource | None = None) -> dict:
        results = {}
        for key, exe in self._executors.items():
            results[key] = exe.step(signal_source=source)
        return results

    def compute_all_signals(self, source: SignalSource | None = None) -> list[SignalResult]:
        src = source or default_signal_source()
        out = []
        for p in self.config.pairs:
            s = src(p["symbol"], p.get("tf", "4h"))
            if s:
                out.append(s)
                print(f"  {p['symbol']:15s} sig={s.signal:+.4f}")
        return out

    def status(self) -> dict:
        return {k: e.status() for k, e in self._executors.items()}

    def write_summary(self, tc: TradingClient, pre_usdt: float, pre_pos: int) -> Summary:
        summary = Summary.from_runner(self, tc, pre_usdt, pre_pos)
        summary.write_to()
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
    trades = []
    entry: LogLine | None = None
    for o in orders:
        p = o.payload
        if not isinstance(p, OrderPayload):
            continue
        if not entry:
            entry = o
            continue
        ep: OrderPayload = entry.payload  # type: ignore[assignment]
        pnl = (p.px / ep.px - 1) * (1 if ep.side in ("long", "buy") else -1)
        trades.append(
            {
                "entry_time": entry.ts,
                "exit_time": o.ts,
                "side": ep.side,
                "entry_price": ep.px,
                "exit_price": p.px,
                "pnl": pnl,
                "reason": "signal",
            }
        )
        entry = o
    if not trades:
        return AssetReport(error="need >=2 trades", trades=0)
    trade_df = pl.DataFrame(trades)
    eq = pl.Series(
        [1000, *(1000 * (1 + (t["exit_price"] / t["entry_price"] - 1)) for t in trades)],
        dtype=pl.Float64,
    )
    eq_df = pl.DataFrame({"portfolio_value": eq, "returns": eq.pct_change().fill_null(0.0)})
    m = compute_metrics(eq_df, trades=trade_df)
    return AssetReport(
        trades=m.num_trades,
        win_rate_pct=m.win_rate_pct,
        sharpe=round(m.sharpe_ratio, 2),
        total_return_pct=round(m.total_return_pct, 1),
        max_drawdown_pct=round(m.max_drawdown_pct, 1),
    )
