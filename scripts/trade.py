"""Live trading entry point — GitHub Actions single invocation.

Uses shared signal pipeline (qooi.core.signal) and decision engine
(qooi.core.decide) so backtest and live trade identically.

Workflow:
  1. Query OKX signal/orders-algo-pending for existing bot configs
  2. Match to hardcoded PAIRS by inst_id
  3. For each pair: compute signal, query position state, decide, push

1H strategies:
  - momentum_1h → ETH (6-bar momentum burst + ADX + session filter)
  - rsi_reversion → SOL (oversold bounce in uptrend with RSI confirmation)

Position state: queried from OKX GET /signal/positions (server-side truth),
NOT from file-persisted _last_side (stateless GitHub Actions fix).

Usage::

    uv run python scripts/trade.py testnet
    uv run python scripts/trade.py live [dry]
"""

from __future__ import annotations

import os
import sys

PAIRS = [
    {
        "symbol": "ETH-USDT-SWAP",
        "sig_symbol": "ETH-USDT",
        "strategy": "momentum_1h",
        "tf": "1H",
        "capital": 500,
        "leverage": 2.0,
        "ct_val": 0.1,
        "sig_threshold": 0.01,
    },
    {
        "symbol": "SOL-USDT-SWAP",
        "sig_symbol": "SOL-USDT",
        "strategy": "rsi_reversion",
        "tf": "1H",
        "capital": 200,
        "leverage": 3.0,
        "ct_val": 1.0,
        "sig_threshold": 0.01,
    },
]


def _get_position_state(tc, algo_id: str, inst_id: str) -> tuple[bool, str]:
    """Query OKX for current position: (has_position, pos_side).

    Uses GET /signal/positions — the server-side source of truth.
    pos > 0 → long, pos < 0 → short, pos == "0" → flat.
    """
    try:
        resp = tc.signal_get_positions(algo_id)
        data = resp.get("data", []) if isinstance(resp, dict) else []
        for pos in data:
            if pos.get("instId") == inst_id:
                qty = str(pos.get("pos", "0"))
                if qty != "0" and qty not in ("", "nan", "None"):
                    p = float(qty)
                    if p > 0:
                        return (True, "buy")
                    elif p < 0:
                        return (True, "sell")
        return (False, "")
    except Exception:
        return (False, "")


def _chan_name(symbol: str) -> str:
    """Derive OKX signal channel name from instrument symbol.

    Must match the naming convention used by setup_signal.py::

        f"qooi-{sym.replace('-', '_')}"
    """
    return f"qooi-{symbol.replace('-', '_')}"


def _bot_configs(tc) -> dict[str, dict]:
    """Query OKX for active signal bots, keyed by signalChanName.

    Returns dict like::

        {"qooi-ETH_USDT_SWAP": {"algo_id": "...", "signal_chan_id": "..."}, ...}
    """
    result: dict[str, dict] = {}
    try:
        resp = tc.signal_get_pending()
        bots = resp.get("data", []) if isinstance(resp, dict) else []
        for bot in bots:
            name = bot.get("signalChanName", "")
            if name:
                result[name] = {
                    "algo_id": bot.get("algoId", ""),
                    "signal_chan_id": bot.get("signalChanId", ""),
                }
    except Exception as e:
        print(f"    WARNING: signal_get_pending failed: {e}")
    return result


def _run(dry_run: bool, env: str) -> None:
    os.environ["OKX_ENV"] = env
    from qooi.core.decide import AssetConfig, decide_active, decide_idle
    from qooi.core.signal import compute_momentum_1h, compute_rsi_reversion_1h
    from qooi.exchange.market import MarketData
    from qooi.exchange.trading import TradingClient, load_okx_env

    load_okx_env()
    tc = TradingClient()
    md = MarketData("okx")

    configs = _bot_configs(tc)

    for p in PAIRS:
        sym = p["symbol"]
        strategy = p["strategy"]
        cfg = configs.get(_chan_name(sym))
        if not cfg:
            print(f"  {sym:20s}  skip (no signal bot — run setup_signal.py)")
            continue

        # 1. Compute signal via strategy-specific function
        if strategy == "momentum_1h":
            signal = compute_momentum_1h(p["sig_symbol"])
        elif strategy == "rsi_reversion":
            signal = compute_rsi_reversion_1h(p["sig_symbol"])
        else:
            print(f"  {sym:20s}  skip (unknown strategy: {strategy})")
            continue

        if signal is None:
            print(f"  {sym:20s}  skip (no_signal)")
            continue

        # 2. Query OKX position state — server-side source of truth
        has_position, pos_side = _get_position_state(tc, cfg["algo_id"], sym)

        # 3. Decide (same functions as backtest)
        asset_cfg = AssetConfig(
            symbol=sym,
            sig_symbol=p["sig_symbol"],
            timeframe=p["tf"],
            capital=p["capital"],
            max_risk_pct=p.get("risk_pct", 0.50),
            leverage=p["leverage"],
            ct_val=p["ct_val"],
            signal_threshold=p["sig_threshold"],
            ord_type=p.get("ord_type", "limit"),
        )

        if not has_position:
            obi = md.ob_snapshot(sym, limit=1)
            entry_px = obi.ask_price if obi else 0
            side = "buy" if signal.signal > 0 else "sell"
            if side == "sell" and obi:
                entry_px = obi.bid_price
            d = decide_idle(signal, entry_px, side, asset_cfg)
        else:
            d = decide_active(signal, pos_side, asset_cfg)

        print(
            f"  {sym:20s} strategy={strategy} sig={signal.signal:+.0f} atr={signal.atr} "
            f"pos={pos_side if has_position else 'flat'} action={d.action.value} {d.detail}"
        )

        if dry_run:
            continue

        # 4. Execute
        if d.action.value == "enter":
            try:
                tc.signal_execute_enter(d, cfg["algo_id"], cfg["signal_chan_id"], sym)
                print(
                    f"    ORDER {d.side} sz={d.sz} px={d.entry_px} sl={d.stop_px} tp={d.target_px}"
                )
            except Exception as e:
                print(f"    ORDER FAILED: {e}")

        elif d.action.value == "exit":
            try:
                tc.signal_execute_exit(cfg["algo_id"], cfg["signal_chan_id"], sym)
                print(f"    CLOSE ({d.detail})")
            except Exception as e:
                print(f"    CLOSE FAILED: {e}")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "testnet"
    if cmd in ("testnet", "live"):
        dry = cmd == "live" and (len(sys.argv) <= 2 or sys.argv[2] != "live")
        _run(dry_run=dry, env="test" if cmd == "testnet" else "live")
    else:
        print("Usage: uv run python scripts/trade.py testnet|live [dry]")
        sys.exit(1)


if __name__ == "__main__":
    main()
