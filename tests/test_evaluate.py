"""Evaluation layer tests."""

from qooi.core.config import PAIRS
from qooi.core.evaluate import Report, compare


def test_report_uses_return_ratios_not_usd():
    pair = PAIRS[0]
    trades = [
        {"side": "buy", "entry_px": 100.0, "exit_px": 101.0, "pnl": 0.01, "reason": "x"},
        {"side": "buy", "entry_px": 100.0, "exit_px": 99.0, "pnl": -0.01, "reason": "y"},
    ]
    equity = [100.0, 101.0, 99.99]
    report = Report.from_raw(trades, equity, pair)
    assert abs(report.metrics.avg_win_pct - 1.0) < 1e-6
    assert abs(report.metrics.avg_loss_pct - 1.0) < 1e-6
    assert report.trade_expectancy_usd == 0.0


def test_compare_formats_multiple_reports():
    pair = PAIRS[0]
    r1 = Report.from_raw([], [100.0, 100.0], pair, label="A")
    r2 = Report.from_raw([], [100.0, 100.0], pair, label="B")
    table = compare(r1, r2)
    assert "Label" in table
    assert "PF" in table
    assert "PL" not in table
    assert "A" in table and "B" in table


def test_report_marks_unstable_annualization_for_sparse_equity():
    pair = PAIRS[0]
    trades = [
        {
            "side": "buy",
            "entry_px": 100.0,
            "exit_px": 101.0,
            "pnl": 0.01,
            "pnl_usd": 1.0,
            "reason": "x",
        }
    ]
    equity = [100.0] + [100.0] * 50
    report = Report.from_raw(trades, equity, pair)
    assert report.unstable_annualization is True


def test_trade_expectancy_fields_populated():
    pair = PAIRS[0]
    trades = [
        {
            "side": "buy",
            "entry_px": 100.0,
            "exit_px": 101.0,
            "pnl": 0.01,
            "pnl_usd": 1.0,
            "reason": "x",
        },
        {
            "side": "buy",
            "entry_px": 100.0,
            "exit_px": 99.0,
            "pnl": -0.005,
            "pnl_usd": -0.5,
            "reason": "y",
        },
    ]
    equity = [100.0, 101.0, 100.495]
    report = Report.from_raw(trades, equity, pair)
    assert report.trade_expectancy_pct > 0
    assert report.trade_expectancy_usd > 0
    assert report.trade_sharpe != 0


def test_active_bar_sharpe_uses_active_exposure():
    pair = PAIRS[0]
    trades = []
    equity = [100.0, 100.5, 100.5, 101.0, 101.0]
    active_exposure = [0.0, 1.0, 0.0, 2.0, 0.0]
    report = Report.from_raw(trades, equity, pair, active_exposure=active_exposure)
    assert report.active_bar_pct > 0
    assert report.active_bar_sharpe != 0 or report.active_bar_pct > 0


def test_report_table_distinguishes_pl_ratio_and_profit_factor():
    pair = PAIRS[0]
    trades = [
        {"side": "buy", "entry_px": 100.0, "exit_px": 102.0, "pnl": 0.02, "reason": "x"},
        {"side": "buy", "entry_px": 100.0, "exit_px": 99.0, "pnl": -0.01, "reason": "y"},
    ]
    report = Report.from_raw(trades, [100.0, 102.0, 100.98], pair)
    table = report.table()
    assert "P/L=" in table
    assert "PF=" in table
