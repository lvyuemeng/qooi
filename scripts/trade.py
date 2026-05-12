"""Trading script — single invocation for GitHub Actions.

Sends one signal per invocation to OKX Signal Bot.  OKX manages
order lifecycle, TP/SL, position tracking — no Python state machine.

Usage::

    uv run python scripts/trade.py testnet
    uv run python scripts/trade.py live [dry]
    uv run python scripts/trade.py backtest

Pre-requisite: run scripts/setup_signal.py once per environment
to create signal channels and strategies on OKX.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import polars as pl

CONFIG_PATH = Path(__file__).resolve().parent.parent / "data" / "signal_bot_config.json"

PAIRS_CONFIG = [
    {
        "symbol": "ETH-USDT-SWAP",
        "sig_symbol": "ETH-USDT",
        "tf": "4h",
        "capital": 500,
        "leverage": 2.0,
        "ct_val": 0.1,
        "sig_threshold": 0.25,
    },
    {
        "symbol": "SOL-USDT-SWAP",
        "sig_symbol": "SOL-USDT",
        "tf": "4h",
        "capital": 200,
        "leverage": 3.0,
        "ct_val": 1.0,
        "sig_threshold": 0.35,
    },
    {
        "symbol": "BTC-USDT-SWAP",
        "sig_symbol": "BTC-USDT",
        "tf": "4h",
        "capital": 1000,
        "leverage": 2.0,
        "ct_val": 0.01,
        "sig_threshold": 0.25,
    },
]


def _compute_signal(symbol: str, timeframe: str, threshold: float):
    from qooi.exchange.indicator import add_indicators
    from qooi.exchange.market import MarketData
    from qooi.exchange.trading import SignalResult
    from qooi.strategies.flow_pipeline import (
        add_ofi_flow_columns,
        add_regime_features,
        apply_regime_gate,
    )

    md = MarketData("okx")
    df = md.candles(symbol, timeframe=timeframe, limit=500, cache=True)
    if df.is_empty():
        return None
    df = add_indicators(df)
    df = add_regime_features(df)
    df = add_ofi_flow_columns(df)
    df = apply_regime_gate(df, signal_col="ofi_flow_score")

    ofi = float(df["ofi_flow_score"][-1])
    sig = round(ofi, 4) if abs(ofi) >= threshold else 0.0
    atr = round(float(df["atr_14"][-1] or 0.0), 2)
    regime = round(float(df["regime_strength"][-1] or 0.0), 3)
    mom = round(float(df["regime_mom_fast"][-1] or 0.0), 3)
    vol = round(float(df["regime_vol_conf"][-1] or 0.5), 3)

    return SignalResult(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=int(df["timestamp"][-1]),
        signal=sig,
        flow=round(ofi, 4),
        threshold=threshold,
        atr=atr,
        regime_strength=regime,
        mom_fast=mom,
        vol_conf=vol,
    )


def _run(dry_run: bool, env: str) -> None:
    os.environ["OKX_ENV"] = env
    from qooi.exchange.trading import TradingClient, load_okx_env

    load_okx_env()
    tc = TradingClient()

    configs = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}

    for p in PAIRS_CONFIG:
        sym = p["symbol"]
        cfg = configs.get(sym)
        if not cfg:
            print(f"  {sym:20s}  skip (no signal bot config)")
            continue

        # 1. Compute signal
        sr = _compute_signal(p["sig_symbol"], p["tf"], p["sig_threshold"])
        if sr is None:
            print(f"  {sym:20s}  skip (no_signal)")
            continue
        print(f"  {sym:20s} signal={sr.signal:+.4f} atr={sr.atr} regime={sr.regime_strength}")

        # 2. Query strategy state from OKX signal bot
        details = tc.signal_get_details(cfg["algo_id"])
        bot_state = details if isinstance(details, dict) else {}
        if isinstance(bot_state.get("data"), list) and bot_state["data"]:
            bot_state = bot_state["data"][0]
        frozen = float(bot_state.get("frozenBal", "0"))
        float_pnl = float(bot_state.get("floatPnl", "0"))
        has_position = frozen > 0 or abs(float_pnl) > 1

        # 3. Decide
        abs_sig = abs(sr.signal)
        action = None
        detail = ""

        if not has_position:
            if abs_sig < p["sig_threshold"]:
                detail = "weak_signal"
            else:
                action = "enter"
        else:
            if sr.signal * (1 if float_pnl >= 0 else -1) < 0:
                action = "exit"
                detail = "signal_flipped"
            else:
                detail = "holding"

        if dry_run:
            label = f"[{action}] {detail}" if action else f"skip ({detail})"
            print(f"  {sym:20s} {label}")
            continue

        # 4. Execute via signal bot
        if action == "enter":
            side = "buy" if sr.signal > 0 else "sell"
            atr = sr.atr if sr.atr > 0 else 50.0
            risk_per_ct = atr * 2.0 * p["ct_val"]
            max_risk = p["capital"] * 0.50
            sz = max(1, int(max_risk / risk_per_ct))
            entry_px = 2300.0  # approximate; signal bot uses limit
            notional_per_ct = p["ct_val"] * entry_px
            max_sz = int(p["capital"] * p["leverage"] / max(notional_per_ct, 1))
            sz = max(1, min(sz, max_sz))

            d = 1 if side == "buy" else -1
            stop_px = round(entry_px - d * 2.0 * atr, 2)
            target_px = round(entry_px + d * 3.0 * atr, 2)

            try:
                tc.signal_push_sub_order(
                    algo_id=cfg["algo_id"],
                    signal_chan_id=cfg["signal_chan_id"],
                    inst_id=sym,
                    side=side,
                    sz=str(int(sz)),
                    ord_type="limit",
                    px=str(entry_px),
                    attach_algo_ords=[
                        {
                            "slTriggerPx": str(stop_px),
                            "slOrdPx": "-1",
                            "tpTriggerPx": str(target_px),
                            "tpOrdPx": "-1",
                            "cxlOnClosePos": "true",
                        }
                    ],
                )
                print(f"  {sym:20s} ORDER {side} sz={sz} sl={stop_px} tp={target_px}")
            except Exception as e:
                print(f"  {sym:20s} ORDER FAILED: {e}")

        elif action == "exit":
            try:
                tc.signal_close_position(
                    algo_id=cfg["algo_id"],
                    signal_chan_id=cfg["signal_chan_id"],
                    inst_id=sym,
                )
                print(f"  {sym:20s} CLOSE ({detail})")
            except Exception as e:
                print(f"  {sym:20s} CLOSE FAILED: {e}")

        else:
            print(f"  {sym:20s} skip ({detail})")


def _backtest() -> None:
    from qooi.exchange.backtest import Backtest, CostModel
    from qooi.exchange.backtest import RiskConfig as BtRisk
    from qooi.exchange.indicator import add_indicators
    from qooi.strategies.flow_pipeline import add_ofi_flow_columns, add_regime_features

    cost = CostModel(commission_pct=0.00005)

    for p in PAIRS_CONFIG:
        sig_sym = p["sig_symbol"]
        tf = p["tf"]
        th = p["sig_threshold"]
        lev = p["leverage"]
        cache_path = f"data/cache/{sig_sym.replace('-', '_')}_{tf.replace(' ', '').upper()}.parquet"
        try:
            df = pl.read_parquet(cache_path)
        except Exception:
            print(f"**{p['symbol']}**: no cache")
            continue

        df = add_indicators(df)
        df = add_regime_features(df)
        df = add_ofi_flow_columns(df)
        ofi = df["ofi_flow_score"]
        sig = pl.when(ofi.abs() >= th).then(ofi).otherwise(0.0)
        df = df.with_columns(sig.alias("signal"))

        risk = BtRisk(
            atr_stop_mult=2.0,
            atr_target_mult=3.0,
            max_leverage=min(lev, 0.4),
            trailing_activation_mult=2.0,
            trailing_distance_mult=1.0,
        )
        bt = Backtest(df, pl.col("signal"), cost=cost, risk=risk, threshold=th, ord_type="market")
        r = bt.run()
        m = r.metrics
        t = r.trades
        print(f"\n=== {p['symbol']} — {df.height} bars, th={th:.2f} ===")
        if t.height == 0:
            print("  No trades")
            continue
        print(f"  Trades: {t.height}  WR={m.win_rate_pct:.0f}%  Sharpe={m.sharpe_ratio:+.2f}")
        print(f"  Return: {m.total_return_pct:+.1f}%  DD: {m.max_drawdown_pct:.1f}%")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "testnet"
    if cmd == "testnet":
        _run(dry_run=False, env="test")
    elif cmd == "live":
        dry = sys.argv[2] != "live" if len(sys.argv) > 2 else True
        _run(dry_run=dry, env="live")
    elif cmd == "backtest":
        _backtest()
    else:
        print("Usage: uv run python scripts/trade.py testnet|live|backtest [dry]")
        sys.exit(1)


if __name__ == "__main__":
    main()
