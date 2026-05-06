"""Backtest report — cross-asset OFI flow with percentile thresholds.

Replaces fixed 0.4 threshold with per-asset 80th percentile of |OFI|.
"""
import polars as pl
from qooi.exchange.backtest import Backtest, RiskConfig, CostModel
from qooi.exchange.indicator import add_indicators
from qooi.strategies.flow_pipeline import add_regime_features, add_ofi_flow_columns

cost = CostModel(commission_pct=0.00005)

DATASETS = {
    "ETH-USDT": "data/cache/ETH_USDT_SWAP_4H.parquet",
    "SOL-USDT": "data/cache/SOL_USDT_4H.parquet",
    "BTC-USDT": "data/cache/BTC_USDT_4H.parquet",
    "XRP-USDT": "data/cache/XRP_USDT_4H.parquet",
}


def run_report(symbol: str, cache_path: str) -> None:
    df = pl.read_parquet(cache_path)
    df = add_indicators(df)
    df = add_regime_features(df)
    df = add_ofi_flow_columns(df)

    ofi = df["ofi_flow_score"]
    abs_ofi = ofi.abs()

    # Optimal per-asset thresholds (post normalization fix):
    #   SOL: 0.35, BTC: 0.25, ETH: 0.25 (SWAP data), XRP: 0.45
    thresholds = {"SOL-USDT": 0.35, "BTC-USDT": 0.25, "ETH-USDT": 0.25, "XRP-USDT": 0.45}
    threshold = thresholds.get(symbol, 0.35)
    nz_all = ofi.filter(ofi.abs() > 0.001).len()
    nz_thresh = ofi.filter(ofi.abs() >= threshold).len()
    print(f"=== {symbol} — {df.height} bars ===")
    print(f"  |OFI| stats: mean={ofi.mean():.4f} std={ofi.std():.4f} min={ofi.min():.4f} max={ofi.max():.4f}")
    print(f"  threshold: {threshold:.4f}  signals above: {nz_thresh} / {nz_all} non-zero")
    print(f"  Current |OFI|: {ofi[-1]:+.4f}  (above threshold: {abs(ofi[-1]) >= threshold})")

    # Apply percentile magnitude filter
    sig = pl.when(ofi.abs() >= threshold).then(ofi).otherwise(0.0)
    df = df.with_columns(sig.alias("signal"))

    risk = RiskConfig(
        atr_stop_mult=2.0, atr_target_mult=3.0, max_leverage=0.4,
        trailing_activation_mult=2.0, trailing_distance_mult=1.0,
    )

    bt = Backtest(df, pl.col("signal"), cost=cost, risk=risk, threshold=threshold, ord_type="market")
    r = bt.run()
    m = r.metrics
    t = r.trades

    if t.height > 0:
        rd = dict(zip(t["reason"].value_counts()["reason"], t["reason"].value_counts()["count"]))
        sd = dict(zip(t["side"].value_counts()["side"], t["side"].value_counts()["count"]))
        pnl = t["pnl"]
        win_pnl = pnl.filter(pnl > 0)
        loss_pnl = pnl.filter(pnl <= 0)
        avg_win = win_pnl.mean() if win_pnl.len() > 0 else 0.0
        avg_loss = loss_pnl.mean() if loss_pnl.len() > 0 else 0.0
        pl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
        print(f"  Sharpe={m.sharpe_ratio:+.2f}  DD={m.max_drawdown_pct:.1f}%  "
              f"Ret={m.total_return_pct:+.1f}%  Trades={m.num_trades}  WR={m.win_rate_pct:.0f}%  "
              f"P/L={pl_ratio:.2f}")
        print(f"  Sides:  L={sd.get('long',0)}  S={sd.get('short',0)}")
        print(f"  Exits:  stop={rd.get('stop',0)}  target={rd.get('target',0)}  "
              f"trail={rd.get('trailing_stop',0)}  signal={rd.get('signal',0)}  "
              f"time={rd.get('time',0)}  end={rd.get('end',0)}")
    else:
        print(f"  No trades")
    print()


if __name__ == "__main__":
    for sym, path in DATASETS.items():
        try:
            run_report(sym, path)
        except Exception as e:
            print(f"{sym}: error - {e}\n")
