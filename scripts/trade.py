"""Trading script — single invocation for GitHub Actions workflows.

Usage::

    uv run python scripts/trade.py testnet
    uv run python scripts/trade.py live [dry]
"""

from __future__ import annotations

import os
import sys

from qooi.exchange.trading import (
    PortfolioConfig,
    PortfolioRunner,
    TradingClient,
)

# ==========================================================================
# Pair configs — single source of truth for all environments.
#
# Both testnet and live use SWAP (perpetual futures) with spot-sourced
# signals.  Requires "single-currency margin" or "multi-currency margin"
# account mode.  Signal from spot candles (clean volume data).
# Execution on swap instruments (99.9% price correlation).
# ==========================================================================
TESTNET_PAIRS = [
    {"exec_symbol": "ETH-USDT-SWAP", "sig_symbol": "ETH-USDT", "tf": "4h",
     "capital": 500, "risk_pct": 0.50, "leverage": 2.0,
     "ct_val": 0.1, "td_mode": "isolated", "sig_threshold": 0.25},
    {"exec_symbol": "SOL-USDT-SWAP", "sig_symbol": "SOL-USDT", "tf": "4h",
     "capital": 200, "risk_pct": 0.50, "leverage": 3.0,
     "ct_val": 1.0, "td_mode": "isolated", "sig_threshold": 0.35},
    {"exec_symbol": "BTC-USDT-SWAP", "sig_symbol": "BTC-USDT", "tf": "4h",
     "capital": 1000, "risk_pct": 0.80, "leverage": 2.0,
     "ct_val": 0.01, "td_mode": "isolated", "sig_threshold": 0.25},
]
LIVE_PAIRS = [
    {"exec_symbol": "ETH-USDT-SWAP", "sig_symbol": "ETH-USDT", "tf": "4h",
     "capital": 500, "risk_pct": 0.50, "leverage": 2.0,
     "ct_val": 0.1, "td_mode": "isolated", "sig_threshold": 0.25},
    {"exec_symbol": "SOL-USDT-SWAP", "sig_symbol": "SOL-USDT", "tf": "4h",
     "capital": 200, "risk_pct": 0.50, "leverage": 3.0,
     "ct_val": 1.0, "td_mode": "isolated", "sig_threshold": 0.35},
    {"exec_symbol": "BTC-USDT-SWAP", "sig_symbol": "BTC-USDT", "tf": "4h",
     "capital": 1000, "risk_pct": 0.80, "leverage": 2.0,
     "ct_val": 0.01, "td_mode": "isolated", "sig_threshold": 0.25},
]


def _run(pairs: list[dict], dry_run: bool, env: str) -> None:
    """Run portfolio step.

    ``env`` must be ``"test"`` (OKX demo) or ``"live"`` (OKX production).
    The env-var is set *before* TradingClient imports so the correct
    credentials are loaded.
    """
    os.environ["OKX_ENV"] = env
    # Force-load the correct .env.* file, overriding any pre-existing
    # env vars (e.g. from IDE auto-loading .env).  Without this,
    # TradingClient picks up the live key even when env="test".
    from qooi.exchange.trading import load_okx_env
    load_okx_env()
    tc = TradingClient()

    # --- pre-flight: query once, reuse (was 5 API calls, now 3) ---
    pre_balance: list[dict] = []
    pre_positions: list[dict] = []
    pre_pending: list[dict] = []
    try:
        pre_balance = tc.balance()
        pre_positions = tc.positions()
        pre_pending = tc.pending()
    except Exception:
        pass

    pre_usdt = 0.0
    for b in pre_balance:
        if b.get("ccy") == "USDT":
            pre_usdt = float(b.get("availBal", 0))
            break
    pre_pos = len(pre_positions)

    print("=== pre-flight ===")
    for b in pre_balance:
        print(
            f"  {b.get('ccy', '?'):6s} avail={b.get('availBal', '?')} frozen={b.get('frozenBal', '?')}"
        )
    print(f"  positions: {pre_pos} open")
    print(f"  pending orders: {len(pre_pending)}")
    print("==================\n")

    max_frozen = 0.95 if env == "test" else 0.50  # testnet has frozen display artifacts
    config = PortfolioConfig(pairs=pairs, dry_run=dry_run, max_frozen_pct=max_frozen)
    runner = PortfolioRunner(config)
    signals = runner.compute_all_signals()
    runner.step(signals=signals, tc=tc)

    runner.write_summary(tc, pre_usdt, pre_pos)


def _backtest() -> None:
    """Run backtest on cached swap data with current pair config thresholds.

    Self-contained — no API keys needed.  One execution shows the full
    lifecycle: enter → fill → manage → exit → PnL per symbol.
    """
    import polars as pl

    from qooi.exchange.backtest import Backtest, CostModel
    from qooi.exchange.backtest import RiskConfig as BtRisk
    from qooi.exchange.indicator import add_indicators
    from qooi.strategies.flow_pipeline import add_ofi_flow_columns, add_regime_features

    cost = CostModel(commission_pct=0.00005)

    for p in TESTNET_PAIRS:
        exec_sym = p.get("exec_symbol", p.get("sig_symbol", "?"))
        sig_sym = p.get("sig_symbol", p.get("exec_symbol", "?"))
        tf = p.get("tf", "4h")
        th = p.get("sig_threshold", 0.35)
        lev = p.get("leverage", 1.0)
        # Fetch signal from spot cache (where volume data is clean)
        cache_path = f"data/cache/{sig_sym.replace('-', '_')}_{tf.replace(' ', '').upper()}.parquet"
        try:
            df = pl.read_parquet(cache_path)
        except Exception:
            print(f"**{exec_sym}**: no spot cache — need ccxt deep fetch (see docs/testnet.md)")
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

        print(
            f"\n=== {exec_sym} (sig={sig_sym}) — {df.height} bars, th={th:.2f}, lev={lev:.1f} ==="
        )
        if t.height == 0:
            print("  No trades in period")
            continue

        # Individual trade PnL
        pnl = t["pnl"]
        print(f"  Trades: {t.height}  WR={m.win_rate_pct:.0f}%  Sharpe={m.sharpe_ratio:+.2f}")
        print(f"  Return: {m.total_return_pct:+.1f}%  DD: {m.max_drawdown_pct:.1f}%")
        print(
            f"  PnL:  sum={pnl.sum():+.2f}  avg={pnl.mean():+.3f}  best={pnl.max():+.2f}  worst={pnl.min():+.2f}"
        )
        print(
            f"  Sides: long={t.filter(pl.col('side') == 'long').height}  short={t.filter(pl.col('side') == 'short').height}"
        )
        reasons = dict(
            zip(t["reason"].value_counts()["reason"], t["reason"].value_counts()["count"])
        )
        print(
            f"  Exits: stop={reasons.get('stop', 0)}  target={reasons.get('target', 0)}  trail={reasons.get('trailing_stop', 0)}  signal={reasons.get('signal', 0)}"
        )

        # Last 5 trades
        print("  Last trades:")
        for row in t.tail(5).iter_rows(named=True):
            px_in = row["entry_price"]
            px_out = row["exit_price"]
            r_pnl = row["pnl"]
            r_side = row["side"]
            r_reason = row["reason"]
            print(
                f"    {r_side:5s}  in={px_in:>10.2f}  out={px_out:>10.2f}  pnl={r_pnl:+.2f}  {r_reason}"
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
