"""Trading script — single invocation for GitHub Actions workflows.

Usage::

    uv run python scripts/trade.py testnet
    uv run python scripts/trade.py live [dry]
    uv run python scripts/trade.py backtest
"""

from __future__ import annotations

import os
import sys

# ==========================================================================
# Pair configs — single source of truth for all environments.
# ==========================================================================
TESTNET_PAIRS = [
    {
        "exec_symbol": "ETH-USDT-SWAP",
        "sig_symbol": "ETH-USDT",
        "tf": "4h",
        "capital": 500,
        "risk_pct": 0.50,
        "leverage": 2.0,
        "ct_val": 0.1,
        "td_mode": "isolated",
        "sig_threshold": 0.25,
        "ord_type": "limit",
    },
    {
        "exec_symbol": "SOL-USDT-SWAP",
        "sig_symbol": "SOL-USDT",
        "tf": "4h",
        "capital": 200,
        "risk_pct": 0.50,
        "leverage": 3.0,
        "ct_val": 1.0,
        "td_mode": "isolated",
        "sig_threshold": 0.35,
        "ord_type": "limit",
    },
    {
        "exec_symbol": "BTC-USDT-SWAP",
        "sig_symbol": "BTC-USDT",
        "tf": "4h",
        "capital": 1000,
        "risk_pct": 0.80,
        "leverage": 2.0,
        "ct_val": 0.01,
        "td_mode": "isolated",
        "sig_threshold": 0.25,
        "ord_type": "limit",
    },
]
LIVE_PAIRS = [
    {
        "exec_symbol": "ETH-USDT-SWAP",
        "sig_symbol": "ETH-USDT",
        "tf": "4h",
        "capital": 500,
        "risk_pct": 0.50,
        "leverage": 2.0,
        "ct_val": 0.1,
        "td_mode": "isolated",
        "sig_threshold": 0.25,
        "ord_type": "post_only",
    },
    {
        "exec_symbol": "SOL-USDT-SWAP",
        "sig_symbol": "SOL-USDT",
        "tf": "4h",
        "capital": 200,
        "risk_pct": 0.50,
        "leverage": 3.0,
        "ct_val": 1.0,
        "td_mode": "isolated",
        "sig_threshold": 0.35,
        "ord_type": "post_only",
    },
    {
        "exec_symbol": "BTC-USDT-SWAP",
        "sig_symbol": "BTC-USDT",
        "tf": "4h",
        "capital": 1000,
        "risk_pct": 0.80,
        "leverage": 2.0,
        "ct_val": 0.01,
        "td_mode": "isolated",
        "sig_threshold": 0.25,
        "ord_type": "post_only",
    },
]


def _run(pairs: list[dict], dry_run: bool, env: str) -> None:
    """Run portfolio step — single invocation, all state from OKX."""
    os.environ["OKX_ENV"] = env
    from qooi.exchange.trading import (
        AssetConfig,
        ExchangeSnapshot,
        State,
        StatelessExecutor,
        Summary,
        TradingClient,
        default_signal_source,
        load_okx_env,
    )

    load_okx_env()

    tc = TradingClient()

    # --- 0. Set leverage (sticky, call once) ---
    for p in pairs:
        try:
            tc.set_leverage(p["exec_symbol"], int(p["leverage"]))
        except Exception:
            pass

    # --- 1. Exchange snapshot ---
    snap = tc.snapshot()
    pre_usdt = snap.usdt_balance
    pre_pos = len(snap.positions)

    print("=== pre-flight ===")
    print(f"  USDT avail={snap.usdt_balance:.2f}  frozen={snap.usdt_frozen:.2f}")
    print(f"  positions: {pre_pos}  orders: {len(snap.orders)}  algos: {len(snap.algo_orders)}")
    print("==================\n")

    # --- 2. Signals ---
    signals: dict[str, SignalResult] = {}  # noqa: F821
    from qooi.exchange.trading import SignalResult

    for p in pairs:
        th = p["sig_threshold"]
        src = default_signal_source(sig_threshold=th)
        sig_sym = p["sig_symbol"]
        exec_sym = p["exec_symbol"]
        tf = p["tf"]
        try:
            s = src(sig_sym, tf)
            if s is None:
                raise RuntimeError("empty signal")
        except Exception:
            # API down — skip, no local fallback
            print(f"  {exec_sym:20s}  skip (signal_unavailable)")
            continue
        s.symbol = exec_sym
        signals[exec_sym] = s
        print(f"  {exec_sym:20s} sig={s.signal:+.4f} flow={s.flow:+.4f} th={s.threshold:.3f}")

    # --- 3. Frozen-capital gate ---
    total_balance = snap.usdt_frozen + snap.usdt_balance
    max_frozen_pct = 0.95 if env == "test" else 0.50
    skip_idle = (snap.usdt_frozen / total_balance) >= max_frozen_pct if total_balance > 0 else False

    # --- 4. Execute ---
    from qooi.exchange.market import MarketData

    md = MarketData("okx")

    print(f"\n{'=' * 70}")
    print(f"{'QOOI ' + env.upper():^70}")
    print(f"{'=' * 70}")
    print(f"  USDT: {snap.usdt_balance:>10.2f}  Positions: {pre_pos}  Orders: {len(snap.orders)}")
    print(f"{'-' * 70}")

    for p in pairs:
        exec_sym = p["exec_symbol"]
        config = AssetConfig(
            symbol=exec_sym,
            sig_symbol=p["sig_symbol"],
            timeframe=p["tf"],
            capital=p["capital"],
            max_risk_pct=p["risk_pct"],
            leverage=p["leverage"],
            ct_val=p["ct_val"],
            signal_threshold=p["sig_threshold"],
            td_mode=p["td_mode"],
            ord_type=p.get("ord_type", "limit"),
        )
        exe = StatelessExecutor(config)

        sr = signals.get(exec_sym)
        if sr is None:
            print(f"  {exec_sym:20s}  skip (no_signal)")
            continue

        dummy = exe._reconstruct(snap, sr, None)
        if skip_idle and dummy.state == State.IDLE:
            print(f"  {exec_sym:20s}  skip (frozen_gate)")
            continue

        obi = None
        if dummy.state in (State.IDLE, State.PENDING):
            try:
                obi = md.ob_snapshot(exec_sym, limit=5)
            except Exception:
                pass

        decision, state = exe.step(snap, sr, obi, None if dry_run else tc)

    # --- 5. Post-step summary ---
    post_positions = tc.positions()
    usdt_details = tc.balance("USDT")
    post_usdt = float(usdt_details[0].get("availBal", 0)) if usdt_details else snap.usdt_balance
    snap2 = ExchangeSnapshot(
        orders=[], positions=post_positions, algo_orders=[], usdt_balance=post_usdt
    )
    summary = Summary.from_snapshot(snap2, pre_usdt, [(p["exec_symbol"], p["tf"]) for p in pairs])
    print(f"{'-' * 70}")
    total = snap2.usdt_balance
    for pos in snap2.positions:
        total += float(pos.get("upl", 0))
    print(f"  Total: {total:>10.2f}  USDT free: {snap2.usdt_balance:>8.2f}")
    for pos in snap2.positions:
        print(
            f"  {pos['instId']:20s} {pos['posSide']:6s} sz={pos['pos']:>6s} "
            f"upl={float(pos.get('upl', 0)):>+8.2f}"
        )
    print(f"{'=' * 70}\n")
    summary.write_to()


def _backtest() -> None:
    """Run backtest on cached data."""
    import polars as pl

    from qooi.exchange.backtest import Backtest, CostModel
    from qooi.exchange.backtest import RiskConfig as BtRisk
    from qooi.exchange.indicator import add_indicators
    from qooi.strategies.flow_pipeline import add_ofi_flow_columns, add_regime_features

    cost = CostModel(commission_pct=0.00005)

    for p in TESTNET_PAIRS:
        exec_sym = p["exec_symbol"]
        sig_sym = p["sig_symbol"]
        tf = p["tf"]
        th = p["sig_threshold"]
        lev = p["leverage"]
        cache_path = f"data/cache/{sig_sym.replace('-', '_')}_{tf.replace(' ', '').upper()}.parquet"
        try:
            df = pl.read_parquet(cache_path)
        except Exception:
            print(f"**{exec_sym}**: no cache")
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
        print(f"\n=== {exec_sym} — {df.height} bars, th={th:.2f}, lev={lev:.1f} ===")
        if t.height == 0:
            print("  No trades")
            continue
        pnl = t["pnl"]
        print(f"  Trades: {t.height}  WR={m.win_rate_pct:.0f}%  Sharpe={m.sharpe_ratio:+.2f}")
        print(f"  Return: {m.total_return_pct:+.1f}%  DD: {m.max_drawdown_pct:.1f}%")
        print(
            f"  PnL: sum={pnl.sum():+.2f} avg={pnl.mean():+.3f} "
            f"best={pnl.max():+.2f} worst={pnl.min():+.2f}"
        )
        n_long = t.filter(pl.col("side") == "long").height
        print(f"  Sides: long={n_long} short={t.filter(pl.col('side') == 'short').height}")
        vc = t["reason"].value_counts()
        reasons = dict(zip(vc["reason"], vc["count"]))
        print(
            f"  Exits: stop={reasons.get('stop', 0)} "
            f"target={reasons.get('target', 0)} "
            f"trail={reasons.get('trailing_stop', 0)} "
            f"signal={reasons.get('signal', 0)}"
        )


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "testnet"
    if cmd == "testnet":
        _run(TESTNET_PAIRS, dry_run=False, env="test")
    elif cmd == "live":
        dry = sys.argv[2] != "live" if len(sys.argv) > 2 else True
        _run(LIVE_PAIRS, dry_run=dry, env="live")
    elif cmd == "backtest":
        _backtest()
    else:
        print("Usage: uv run python scripts/trade.py testnet|live|backtest [dry]")
        sys.exit(1)


if __name__ == "__main__":
    main()
