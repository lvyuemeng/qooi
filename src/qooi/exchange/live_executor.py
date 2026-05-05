"""Live strategy executor — reads signal file → places orders → tracks lifecycle.

Separated from signal computation. To run:

1. ``compute_signal("BTC-USDT-SWAP", "4h")`` — writes ``data/signals/BTC_USDT_SWAP_4h.json``
2. ``LiveExecutor().step()`` — reads signal file, places limit orders, logs events

No internet connection needed for step 1 (uses cached data).
Step 2 auto-skips if signal file missing or stale (>1 bar old).
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

LOG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "logs"
SIGNAL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "signals"
SIGNAL_DIR.mkdir(parents=True, exist_ok=True)


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


# ---------------------------------------------------------------------------
# 1. Signal computation (offline / batch — no exchange connection needed)
# ---------------------------------------------------------------------------


def compute_signal(symbol: str = "BTC-USDT-SWAP", timeframe: str = "4h") -> dict | None:
    """Compute latest bar signal and write to ``data/signals/``.

    Uses cached OHLCV if available (no API key needed). Returns the
    computed signal dict, also writes JSON file for executor to consume.
    """
    fname = SIGNAL_DIR / f"{symbol.replace('-', '_')}_{timeframe}.json"

    md = MarketData()
    df = md.candles(symbol, timeframe=timeframe, limit=500)
    if df.is_empty():
        return None

    df = add_indicators(df)
    df = add_regime_features(df)
    df = multi_factor_intraday_signal(df)
    df = add_ofi_flow_columns(df)
    df = apply_micro_confirmation(df)
    df = add_adaptive_threshold(df)
    df = apply_adaptive_gate(df)

    signal_val = float(df["signal"][-1] or 0.0)
    flow_val = float(df["ofi_flow_score"][-1] or 0.0)
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


# ---------------------------------------------------------------------------
# 2. Live executor (reads signal file → places orders)
# ---------------------------------------------------------------------------


class LiveExecutor:
    """Read signal file, place/manage limit orders, log everything.

    No signal computation — reads from ``data/signals/`` (produced by
    ``compute_signal``). This decouples compute from execute.

    Use ``dry_run=True`` to verify without sending real orders.
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
        """Read signal file → place order if warranted → track lifecycle.

        Skips if signal file missing or older than 1 bar (signal stale).
        """
        self._log_event("cycle_start", {})

        # 1. Read signal file
        sf = SIGNAL_DIR / f"{self._symbol.replace('-', '_')}_{self._timeframe}.json"
        if not sf.exists():
            self._log_event("skip", {"reason": "no_signal_file"})
            return None

        sig_data = json.loads(sf.read_text())
        signal = sig_data.get("signal", 0.0)
        flow_val = sig_data.get("flow", 0.0)
        sig_ts = sig_data.get("timestamp", 0)

        # Check staleness: signal must be from current bar window
        bar_ms = _timeframe_ms(self._timeframe)
        if time.time() * 1000 - sig_ts > bar_ms * 1.5:
            self._log_event("skip", {"reason": "stale_signal", "signal_ts": sig_ts})
            return None

        # 2. Get live order book
        obi = self._md.ob_snapshot(self._symbol)

        self._log_event(
            "signal",
            {
                "signal": round(signal, 4),
                "obi_5": round(obi.imbalance_5, 4),
                "ofi_flow": round(flow_val, 4),
            },
        )

        # 3. Cancel stale orders
        if self._active and self._active.status == "placed":
            elapsed = time.time() - self._active.placed_at
            if elapsed > self._timeout:
                self._cancel_order("timeout")

        # 4. Entry gate
        if abs(signal) < 0.25:
            self._log_event("skip", {"reason": "weak_signal", "signal": round(signal, 4)})
            return None

        if self._active and self._active.status in ("placed", "partial_fill"):
            self._log_event("skip", {"reason": "order_outstanding", "ord_id": self._active.ord_id})
            return None

        # 5. Position check
        side = "buy" if signal > 0 else "sell"
        if not self._dry and self._trader:
            pos = self._trader.get_positions()
            has_pos = any(
                p.get("instId") == self._symbol and float(p.get("pos", "0")) != 0
                for p in pos.get("data", [])
            )
            if has_pos and side == "buy":
                self._log_event("skip", {"reason": "already_long"})
                return None

        # 6. Size
        entry_px = obi.ask_price if side == "buy" else obi.bid_price
        risk = self._capital * self._max_pos * abs(signal)
        sz = risk / entry_px if entry_px > 0 else 0.001

        if sz < 0.00001:
            self._log_event("skip", {"reason": "size_too_small", "sz": round(sz, 8)})
            return None

        order = self._place_limit(side, sz, entry_px, signal, obi, flow_val)
        return order

    def get_status(self) -> dict:
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

    def _place_limit(self, side, sz, px, signal, obi, flow):
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

    def _cancel_order(self, reason):
        if not self._active:
            return
        self._active.status = "cancelled"
        self._active.reason = reason
        self._log_event("cancel", self._order_dict(self._active))

        if not self._dry and self._trader:
            self._trader._api.cancel_order(instId=self._symbol, ordId=self._active.ord_id)

        self._active = None

    def _order_dict(self, o):
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

    def _log_event(self, event, data):
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
        # Human-readable echo
        msg = _human_line(event, self._symbol, self._timeframe, data)
        print(msg)


# ---------------------------------------------------------------------------
# Portfolio runner
# ---------------------------------------------------------------------------


@dataclass
class PortfolioConfig:
    """Single config for multi-asset deployment.

    Example::

        config = PortfolioConfig(
            pairs=[
                {"symbol": "ETH-USDT", "tf": "4h", "capital": 100, "risk_pct": 0.03},
                {"symbol": "SOL-USDT", "tf": "4h", "capital": 50,  "risk_pct": 0.05},
            ],
            dry_run=True,
        )
        PortfolioRunner(config).step()
    """

    pairs: list[dict]
    dry_run: bool = True
    post_only: bool = True
    limit_timeout_sec: int = 120


class PortfolioRunner:
    """Run multiple LiveExecutor instances from one config, with summary log."""

    def __init__(self, config: PortfolioConfig) -> None:
        self._config = config
        self._executors: dict[str, LiveExecutor] = {}
        for pair in config.pairs:
            sym = pair["symbol"]
            tf = pair.get("tf", "4h")
            self._executors[f"{sym}_{tf}"] = LiveExecutor(
                symbol=sym,
                timeframe=tf,
                initial_capital=pair.get("capital", 100),
                max_position_pct=pair.get("risk_pct", 0.03),
                post_only=config.post_only,
                limit_timeout_sec=config.limit_timeout_sec,
                dry_run=config.dry_run,
            )

    def step(self) -> dict[str, OrderState | None]:
        results = {}
        print(f"\n{'=' * 60}")
        print(f"  Portfolio step @ {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Dry: {self._config.dry_run}  Pairs: {len(self._config.pairs)}")
        print(f"{'=' * 60}")
        for key, exe in self._executors.items():
            result = exe.step()
            results[key] = result
        self._write_summary(results)
        return results

    def compute_all_signals(self) -> list[dict]:
        """Pre-compute signals for all pairs (offline step)."""
        computed = []
        print(f"\\n{'=' * 60}")
        print(f"  Computing signals for {len(self._config.pairs)} pairs")
        print(f"{'=' * 60}")
        for pair in self._config.pairs:
            s = compute_signal(pair["symbol"], pair.get("tf", "4h"))
            if s:
                computed.append(s)
                print(f"  {pair['symbol']:15s} sig={s['signal']:+.4f}  flow={s['flow']:+.4f}")
        return computed

    def _write_summary(self, results):
        """Human-readable summary to both JSONL and plain-text."""
        # Plain-text log
        txt_path = LOG_DIR / "portfolio_summary.txt"
        lines = [f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]"]
        placed = 0
        for key, result in results.items():
            if result and result.status == "placed":
                lines.append(
                    f"  {key:20s} {result.side.upper():4s}  sz={result.sz:.6f}  "
                    f"px={result.px:.1f}  sig={result.signal_value:+.3f}  "
                    f"obi={result.obi_value:+.3f}"
                )
                placed += 1
            elif result:
                lines.append(f"  {key:20s} {result.status}")
            else:
                lines.append(f"  {key:20s} no_signal")
        lines.append(f"  Placed: {placed}/{len(results)}")
        with open(txt_path, "a") as f:
            f.write("\n".join(lines) + "\n\n")

        # JSONL summary
        jl_path = LOG_DIR / "portfolio.jsonl"
        summary = {
            "ts": int(time.time() * 1000),
            "placed": placed,
            "total": len(results),
            "results": {key: (r._order_dict(r) if r else None) for key, r in results.items()},
        }
        with open(jl_path, "a") as f:
            f.write(json.dumps(summary) + "\n")

    def get_status(self) -> dict:
        return {key: exe.get_status() for key, exe in self._executors.items()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _human_line(event, symbol, tf, data):
    ts = time.strftime("%H:%M:%S")
    if event == "cycle_start":
        return f"[{ts}] {symbol} {tf} ── cycle start"
    if event == "skip":
        return f"[{ts}] {symbol} {tf} ⏭  skip ({data.get('reason', '?')})"
    if event == "signal":
        sig = data.get("signal", 0)
        obi = data.get("obi_5", 0)
        flow = data.get("ofi_flow", 0)
        return f"[{ts}] {symbol} {tf} | sig={sig:+.3f}  obi={obi:+.3f}  flow={flow:+.3f}"
    if event == "order":
        side = data.get("side", "?")
        sz = data.get("sz", 0)
        px = data.get("px", 0)
        oid = data.get("ord_id", "?")
        return f"[{ts}] {symbol} {tf} | ORDER {side:4s} sz={sz:.6f} px={px:.1f} id={oid}"
    if event == "cancel":
        return f"[{ts}] {symbol} {tf} | CANCEL {data.get('reason', '?')}"
    if event == "order_error":
        return f"[{ts}] {symbol} {tf} ❌ ERROR {data.get('error', '?')}"
    if event == "error":
        return f"[{ts}] {symbol} {tf} ❌ {data.get('msg', '?')}"
    return json.dumps(data)[:120]


def _timeframe_ms(tf: str) -> int:
    return {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}.get(tf, 14_400_000)
