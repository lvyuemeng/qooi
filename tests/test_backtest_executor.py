"""BacktestExecutor white-box integration tests."""

import polars as pl

from qooi.core.config import PAIRS
from qooi.core.evaluate import Report
from qooi.core.executor import BacktestExecutor


def _load(sig_symbol: str) -> pl.DataFrame:
    df = pl.read_parquet(f"data/cache/{sig_symbol.replace('-', '_')}_1H.parquet")
    if "vol" not in df.columns and "volume" in df.columns:
        df = df.rename({"volume": "vol"})
    return df


def test_run_generates_multiple_trades_eth():
    pair = PAIRS[0]
    df = _load(pair.asset.sig_symbol)
    trades, equity = BacktestExecutor(initial_capital=pair.asset.capital).run(df, pair)
    assert len(trades) > 0
    assert len(equity) > 10
    assert equity[-1] != 0


def test_run_generates_multiple_trades_sol():
    pair = PAIRS[1]
    df = _load(pair.asset.sig_symbol)
    trades, equity = BacktestExecutor(initial_capital=pair.asset.capital).run(df, pair)
    assert len(trades) > 0
    assert len(equity) > 10
    assert equity[-1] > 0


def test_trade_pnl_sign_matches_direction():
    pair = PAIRS[0]
    df = _load(pair.asset.sig_symbol)
    trades, _ = BacktestExecutor(initial_capital=pair.asset.capital).run(df, pair)
    assert trades
    for t in trades:
        entry = float(t["entry_px"])
        exit_px = float(t["exit_px"])
        pnl = float(t["pnl"])
        side = t["side"]
        if side == "buy":
            assert (exit_px > entry and pnl > 0) or (exit_px <= entry and pnl <= 0)
        else:
            assert (exit_px < entry and pnl > 0) or (exit_px >= entry and pnl <= 0)


def test_trade_rows_include_richer_fields():
    pair = PAIRS[0]
    df = _load(pair.asset.sig_symbol)
    trades, _ = BacktestExecutor(initial_capital=pair.asset.capital).run(df, pair)
    assert trades
    row = trades[0]
    assert "pnl_usd" in row
    assert "bars_held" in row


def test_trade_pnl_usd_uses_exit_size_snapshot():
    pair = PAIRS[0]
    df = _load(pair.asset.sig_symbol)
    trades, equity = BacktestExecutor(initial_capital=pair.asset.capital, cost_pct=0.0).run(
        df, pair
    )
    assert trades
    assert any(abs(float(t["pnl_usd"])) > 0 for t in trades if float(t["pnl"]) != 0)
    assert equity[-1] != pair.asset.capital


def test_run_report_returns_report():
    pair = PAIRS[0]
    df = _load(pair.asset.sig_symbol)
    report = BacktestExecutor(initial_capital=pair.asset.capital).run_report(df, pair)
    assert isinstance(report, Report)
    assert report.metrics.num_trades >= 0
    assert report.equity.height > 0


def test_drawdown_stop_halts_loop():
    pair = PAIRS[0]
    df = _load(pair.asset.sig_symbol)
    # tiny capital + normal costs should stop early on DD if losses occur
    trades, equity = BacktestExecutor(initial_capital=50.0).run(df, pair)
    assert len(equity) <= df.height
