"""Full backtest — all assets with shared signal pipeline (same as live).
Uses proper position sizing: each trade's PnL in USD based on actual sz.
"""

from __future__ import annotations

from statistics import mean, stdev

import polars as pl

from qooi.core.decide import (
    AssetConfig,
    decide_active,
    decide_idle,
)
from qooi.core.signal import SignalResult, compute_dataframe

# ---------------------------------------------------------------------------
ASSETS = [
    # (cache_file, symbol, ct_val, capital, leverage, threshold, exec_symbol)
    ("ETH_USDT_4H.parquet", "ETH-USDT", 0.1, 500, 2.0, 0.25, "ETH-USDT-SWAP"),
    ("SOL_USDT_4H.parquet", "SOL-USDT", 1.0, 200, 3.0, 0.35, "SOL-USDT-SWAP"),
    ("BTC_USDT_4H.parquet", "BTC-USDT", 0.01, 1000, 2.0, 0.25, "BTC-USDT-SWAP"),
    ("XRP_USDT_4H.parquet", "XRP-USDT", 10.0, 200, 3.0, 0.30, "XRP-USDT-SWAP"),
    ("LTC_USDT_4H.parquet", "LTC-USDT", 1.0, 200, 3.0, 0.30, "LTC-USDT-SWAP"),
    ("DOGE_USDT_4H.parquet", "DOGE-USDT", 1000.0, 100, 3.0, 0.40, "DOGE-USDT-SWAP"),
    ("XAU_USDT_SWAP_4H.parquet", "XAU-USDT", 0.001, 500, 5.0, 0.25, "XAU-USDT-SWAP"),
    ("XAG_USDT_SWAP_4H.parquet", "XAG-USDT", 0.01, 300, 5.0, 0.25, "XAG-USDT-SWAP"),
]


def _run():
    for cache_file, symbol, ct_val, capital, leverage, threshold, exec_sym in ASSETS:
        path = f"data/cache/{cache_file}"
        try:
            df = pl.read_parquet(path)
        except Exception:
            print(f"**{exec_sym}**: no cache")
            continue

        df = compute_dataframe(df, threshold)
        n = df.height

        cfg = AssetConfig(
            symbol=exec_sym,
            sig_symbol=symbol,
            timeframe="4h",
            capital=capital,
            leverage=leverage,
            ct_val=ct_val,
            signal_threshold=threshold,
            ord_type="limit",
        )

        trades = []
        pos_side = ""
        equity = capital
        equity_curve = [equity]

        for i in range(n):
            row = df.row(i, named=True)
            sig_val = float(row["signal"] or 0)
            atr_val = float(row.get("atr_14", 0) or 0)
            regime = float(row.get("regime_strength", 0) or 0)
            mom = float(row.get("regime_mom_fast", 0) or 0)
            vol = float(row.get("regime_vol_conf", 0.5) or 0.5)

            signal = SignalResult(
                symbol=symbol,
                timeframe="4h",
                timestamp=int(row["timestamp"]),
                signal=sig_val,
                flow=sig_val,
                threshold=threshold,
                atr=atr_val,
                regime_strength=regime,
                mom_fast=mom,
                vol_conf=vol,
            )

            if not pos_side:
                close = float(row["close"])
                side = "buy" if sig_val > 0 else "sell"
                d = decide_idle(signal, close, side, cfg)
                if d.action.value == "enter":
                    pos_side = d.side
                    trades.append(
                        {
                            "entry_time": int(row["timestamp"]),
                            "exit_time": 0,
                            "side": pos_side,
                            "entry_price": d.entry_px,
                            "exit_price": 0.0,
                            "pnl_pct": 0.0,
                            "reason": "",
                            "sz": d.sz,
                        }
                    )
            else:
                d = decide_active(signal, pos_side, cfg)
                if d.action.value == "exit":
                    close = float(row["close"])
                    t = trades[-1]
                    e_px = t["entry_price"]
                    d_sign = 1 if pos_side == "buy" else -1
                    pnl_pct = d_sign * (close / e_px - 1) if e_px > 0 else 0.0
                    # PnL in USD: notional = sz * ct_val * entry_px
                    notional = t["sz"] * ct_val * e_px
                    pnl_usd = pnl_pct * notional
                    trades[-1]["exit_time"] = int(row["timestamp"])
                    trades[-1]["exit_price"] = close
                    trades[-1]["pnl_pct"] = pnl_pct
                    trades[-1]["reason"] = d.detail
                    trades[-1]["pnl_usd"] = pnl_usd
                    equity += pnl_usd
                    equity_curve.append(equity)
                    pos_side = ""

        n_trades = len(trades)
        closed = [t for t in trades if t.get("exit_price", 0) > 0]
        n_closed = len(closed)
        if n_closed < 3:
            print(
                f"{exec_sym:24s} {n:5d} bars  {n_trades:3d} trades  {n_closed:2d} closed  (insufficient data)"
            )
            continue

        wins = [t for t in closed if t["pnl_pct"] > 0]
        wr = len(wins) / n_closed * 100
        pnl_pcts = [t["pnl_pct"] for t in closed]
        avg_pnl = mean(pnl_pcts)
        total_ret = (equity / capital - 1) * 100

        # Annualized: bars per year at 4h = 365*6 = 2190
        years = n / (365 * 6)
        cagr = ((equity / capital) ** (1 / max(years, 1e-9)) - 1) * 100

        # Sharpe from trade-level returns (not bar-level because sparse)
        mu = mean(pnl_pcts)
        sig = stdev(pnl_pcts) if len(pnl_pcts) > 1 else 1e-9
        avg_intertrade_bars = n / n_trades if n_trades > 0 else n
        trades_per_year = 365 * 6 / avg_intertrade_bars
        sharpe = (mu / sig) * (trades_per_year**0.5) if sig > 0 else 0.0

        # Max DD
        peak = equity_curve[0]
        max_dd = 0.0
        for v in equity_curve:
            peak = max(peak, v)
            dd = (peak - v) / peak * 100
            max_dd = max(max_dd, dd)

        print(
            f"{exec_sym:24s} {n:5d}b  {n_trades:3d}t  {n_closed:2d}c  "
            f"WR={wr:.0f}%  Shp={sharpe:+.2f}  CAGR={cagr:+.1f}%  "
            f"Ret={total_ret:+.1f}%  DD={max_dd:.1f}%  years={years:.1f}"
        )


if __name__ == "__main__":
    _run()
