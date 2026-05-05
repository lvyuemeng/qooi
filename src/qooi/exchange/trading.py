"""OKX trading — orders, signals, execution, portfolio management.

Layers:
  1. TradingClient — atomic order/account/position operations
  2. compute_signal — offline signal pipeline, writes JSON to data/signals/
  3. LiveExecutor — reads signal file → places limit orders → logs
  4. PortfolioRunner — multi-asset deployment from a config list

Usage::

    # Offline: compute signal (no API key needed)
    from qooi.exchange.trading import compute_signal
    compute_signal("ETH-USDT", "4h")

    # Online: execute from signal file
    from qooi.exchange.trading import LiveExecutor
    LiveExecutor(dry_run=False).step()

    # Portfolio: multi-asset
    from qooi.exchange.trading import PortfolioConfig, PortfolioRunner
    PortfolioRunner(PortfolioConfig([...], dry_run=False)).step()
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# === constants ===============================================================

LOG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

SIGNAL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "signals"
SIGNAL_DIR.mkdir(parents=True, exist_ok=True)


def _env(key: str) -> str:
    val = os.getenv(key)
    if not val:
        msg = f"Missing {key}. Set in .env or export {key}=..."
        raise RuntimeError(msg)
    return val


# ============================================================================
# 1. Atomic trading client (order / account / position)
# ============================================================================


class TradingClient:
    """Place/cancel orders, query balance and positions.

    flag="1" → testnet (demo). flag="0" → live.
    """

    def __init__(self, flag: str = "1") -> None:
        from okx.Account import AccountAPI
        from okx.Trade import TradeAPI

        k, s, p = _env("OKX_API_KEY"), _env("OKX_SECRET_KEY"), _env("OKX_PASSPHRASE")
        self._trade = TradeAPI(k, s, p, flag=flag, debug=False)
        self._account = AccountAPI(k, s, p, flag=flag, debug=False)

    # -- orders ---------------------------------------------------------------

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
        r = self._trade.place_order(**params)
        return _check(r)

    def cancel(self, inst_id: str, ord_id: str) -> dict:
        return _check(self._trade.cancel_order(instId=inst_id, ordId=ord_id))

    def pending(self) -> list:
        return self._trade.get_order_list().get("data", [])

    # -- account --------------------------------------------------------------

    def balance(self, ccy: str | None = None) -> list:
        params = {}
        if ccy:
            params["ccy"] = ccy
        resp = self._account.get_account_balance(**params)
        return _check(resp).get("details", [])

    def positions(self) -> list:
        return _check(self._account.get_positions()).get("data", [])


def _check(resp: dict) -> dict:
    if resp.get("code") != "0":
        raise RuntimeError(f"OKX error: {resp.get('msg', resp)}")
    return resp.get("data", [{}])[0] if resp.get("data") else resp


# ============================================================================
# 2. Order state tracker (lifecycle record)
# ============================================================================


@dataclass
class OrderState:
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


# ============================================================================
# 3. Signal pipeline — pluggable (any callable that takes a DataFrame)
# ============================================================================


def default_pipeline(df):
    """Multi-factor intraday ensemble — used when no custom pipeline given."""
    from qooi.exchange.indicator import add_indicators
    from qooi.strategies.flow_pipeline import (
        add_adaptive_threshold,
        add_ofi_flow_columns,
        add_regime_features,
        apply_adaptive_gate,
        apply_micro_confirmation,
    )
    from qooi.strategies.intraday import multi_factor_intraday_signal

    df = add_indicators(df)
    df = add_regime_features(df)
    df = multi_factor_intraday_signal(df)
    df = add_ofi_flow_columns(df)
    df = apply_micro_confirmation(df)
    df = add_adaptive_threshold(df)
    df = apply_adaptive_gate(df)
    return df


def compute_signal(
    symbol: str = "BTC-USDT",
    timeframe: str = "4h",
    pipeline=default_pipeline,
) -> dict | None:
    """Run pipeline on latest bar → write signal file → return dict.

    ``pipeline`` is any callable ``DataFrame -> DataFrame`` that
    produces a ``signal`` column.  Pass your own to swap strategies
    without touching executor code.

    Example::

        # Use trend-pullback instead of ensemble
        from qooi.strategies.trend_pullback import trend_pullback_signal
        def tp_pipeline(df):
            df = add_indicators(df)
            return trend_pullback_signal(df)
        compute_signal(\"BTC-USDT\", \"1D\", pipeline=tp_pipeline)
    """
    from qooi.exchange.market import MarketData

    fname = SIGNAL_DIR / f"{symbol.replace('-', '_')}_{timeframe}.json"
    md = MarketData()
    df = md.candles(symbol, timeframe=timeframe, limit=500)
    if df.is_empty():
        return None

    df = pipeline(df)

    signal_val = float(df["signal"][-1] or 0.0)
    flow_val = float(df["ofi_flow_score"][-1] or 0.0) if "ofi_flow_score" in df.columns else 0.0
    ts = int(df["timestamp"][-1])

    result = {
        "symbol": symbol,
        "timeframe": timeframe,
        "timestamp": ts,
        "signal": round(signal_val, 4),
        "flow": round(flow_val, 4),
        "computed_at": int(time.time()),
    }
    fname.write_text(json.dumps(result, indent=2))
    return result


# ============================================================================
# 4. Live executor (reads signal file → places limit orders → logs)
# ============================================================================


class LiveExecutor:
    """Read signal file, place/manage limit orders, log everything.

    dry_run=True → simulate only (no orders). dry_run=False → real testnet/live.
    """

    def __init__(
        self,
        symbol: str = "BTC-USDT",
        timeframe: str = "4h",
        initial_capital: float = 1000.0,
        max_position_pct: float = 0.05,
        post_only: bool = True,
        limit_timeout_sec: int = 120,
        dry_run: bool = True,
    ) -> None:
        self._symbol = symbol
        self._tf = timeframe
        self._capital = initial_capital
        self._max_pos = max_position_pct
        self._post_only = post_only
        self._timeout = limit_timeout_sec
        self._dry = dry_run

        from qooi.exchange.market import MarketData

        self._md = MarketData()
        self._client: TradingClient | None = None if dry_run else TradingClient()
        self._active: OrderState | None = None
        self._trade_count = 0

    def step(self) -> OrderState | None:
        self._log("cycle_start", {})
        sf = SIGNAL_DIR / f"{self._symbol.replace('-', '_')}_{self._tf}.json"
        if not sf.exists():
            return self._log("skip", {"reason": "no_signal_file"})

        sig_data = json.loads(sf.read_text())
        signal = sig_data.get("signal", 0.0)
        flow_val = sig_data.get("flow", 0.0)
        sig_ts = sig_data.get("timestamp", 0)

        bar_ms = {"1h": 3600000, "4h": 14400000, "1d": 86400000}.get(self._tf, 14400000)
        if time.time() * 1000 - sig_ts > bar_ms * 1.5:
            return self._log("skip", {"reason": "stale_signal"})

        obi = self._md.ob_snapshot(self._symbol)
        self._log(
            "signal",
            {
                "signal": round(signal, 4),
                "obi_5": round(obi.imbalance_5, 4),
                "ofi_flow": round(flow_val, 4),
            },
        )

        if self._active and self._active.status == "placed":
            if time.time() - self._active.placed_at > self._timeout:
                self._cancel("timeout")

        if abs(signal) < 0.25:
            return self._log("skip", {"reason": "weak_signal"})

        if self._active and self._active.status in ("placed", "partial_fill"):
            return self._log("skip", {"reason": "order_outstanding"})

        side = "buy" if signal > 0 else "sell"
        if not self._dry and self._client:
            pos = self._client.positions()
            if any(p.get("instId") == self._symbol and float(p.get("pos", "0")) != 0 for p in pos):
                if side == "buy":
                    return self._log("skip", {"reason": "already_long"})

        entry_px = obi.ask_price if side == "buy" else obi.bid_price
        sz = self._capital * self._max_pos * abs(signal) / (entry_px or 1)
        if sz < 0.00001:
            return self._log("skip", {"reason": "size_too_small"})

        return self._place(side, sz, entry_px, signal, obi, flow_val)

    def status(self) -> dict:
        return {
            "symbol": self._symbol,
            "tf": self._tf,
            "capital": self._capital,
            "dry": self._dry,
            "active": self._active.ord_id if self._active else None,
            "trades": self._trade_count,
        }

    # -- internals ------------------------------------------------------------

    def _place(self, side, sz, px, signal, obi, flow):
        sz, px = round(sz, 8), round(px, 1)
        if self._dry:
            self._active = OrderState(
                ord_id="dry_" + str(int(time.time())),
                inst_id=self._symbol,
                side=side,
                sz=sz,
                px=px,
                placed_at=time.time(),
                signal_value=signal,
                obi_value=obi.imbalance_5,
                flow_value=flow,
                status="simulated",
            )
            self._log("order", self._to_dict(self._active))
            self._trade_count += 1
            return self._active

        try:
            otype = "post_only" if self._post_only else "limit"
            resp = self._client.place(self._symbol, side, str(sz), ord_type=otype, px=str(px))
            self._active = OrderState(
                ord_id=resp.get("ordId", ""),
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
            self._log("order", self._to_dict(self._active))
            self._trade_count += 1
            return self._active
        except Exception as e:
            self._log("order_error", {"error": str(e)})
            return None

    def _cancel(self, reason):
        if not self._active:
            return
        self._active.status, self._active.reason = "cancelled", reason
        self._log("cancel", self._to_dict(self._active))
        if not self._dry and self._client:
            self._client.cancel(self._symbol, self._active.ord_id)
        self._active = None

    @staticmethod
    def _to_dict(o):
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

    def _log(self, event, data):
        record = {
            "ts": int(time.time() * 1000),
            "event": event,
            "symbol": self._symbol,
            "tf": self._tf,
            **data,
        }
        fname = LOG_DIR / f"exec_{self._symbol.replace('-', '_')}_{self._tf}.jsonl"
        with open(fname, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(_human(event, self._symbol, self._tf, data))
        return (
            None if event in ("skip", "error") else None
        )  # return None only when we shouldn't return order


# ============================================================================
# 5. Portfolio runner
# ============================================================================


@dataclass
class PortfolioConfig:
    pairs: list[dict]
    dry_run: bool = True
    post_only: bool = True
    limit_timeout_sec: int = 120


class PortfolioRunner:
    """Multi-asset deployment."""

    def __init__(self, config: PortfolioConfig) -> None:
        self._config = config
        self._executors: dict[str, LiveExecutor] = {}
        for p in config.pairs:
            key = f"{p['symbol']}_{p.get('tf', '4h')}"
            self._executors[key] = LiveExecutor(
                symbol=p["symbol"],
                timeframe=p.get("tf", "4h"),
                initial_capital=p.get("capital", 100),
                max_position_pct=p.get("risk_pct", 0.03),
                post_only=config.post_only,
                limit_timeout_sec=config.limit_timeout_sec,
                dry_run=config.dry_run,
            )

    def step(self) -> dict:
        results = {}
        t = time.strftime("%Y-%m-%d %H:%M:%S")
        hdr = f"\n{'=' * 60}\n  Portfolio {t}  Dry: {self._config.dry_run}\n{'=' * 60}"
        print(hdr)
        for key, exe in self._executors.items():
            results[key] = exe.step()
        self._summary(results)
        return results

    def compute_all_signals(self) -> list[dict]:
        out = []
        for p in self._config.pairs:
            s = compute_signal(p["symbol"], p.get("tf", "4h"))
            if s:
                out.append(s)
                print(f"  {p['symbol']:15s} sig={s['signal']:+.4f}")
        return out

    def status(self) -> dict:
        return {k: e.status() for k, e in self._executors.items()}

    def _summary(self, results):
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
                json.dumps({"ts": int(time.time() * 1000), "placed": placed, "total": len(results)})
                + "\n"
            )


# ============================================================================
# Helpers
# ============================================================================


def _human(event, symbol, tf, data):
    ts = time.strftime("%H:%M:%S")
    if event == "cycle_start":
        return f"[{ts}] {symbol} {tf} ── cycle start"
    if event == "skip":
        return f"[{ts}] {symbol} {tf} ⏭  skip ({data.get('reason', '?')})"
    if event == "signal":
        s, o, f = data.get("signal", 0), data.get("obi_5", 0), data.get("ofi_flow", 0)
        return f"[{ts}] {symbol} {tf} | sig={s:+.3f} obi={o:+.3f} flow={f:+.3f}"
    if event == "order":
        s, z, p, i = (
            data.get("side", "?"),
            data.get("sz", 0),
            data.get("px", 0),
            data.get("ord_id", "?"),
        )
        return f"[{ts}] {symbol} {tf} | ORDER {s:4s} sz={z:.6f} px={p:.1f} id={i}"
    if event in ("cancel",):
        return f"[{ts}] {symbol} {tf} | CANCEL {data.get('reason', '?')}"
    if event in ("order_error", "error"):
        return f"[{ts}] {symbol} {tf} | ERROR {data.get('error', data.get('msg', '?'))}"
    return json.dumps(data)[:120]
