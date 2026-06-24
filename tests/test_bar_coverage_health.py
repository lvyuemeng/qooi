"""Bar coverage health uses timeframe-aware targets."""

import polars as pl

from qooi.pipeline import now_ms
from qooi.pipeline.load import BarLoadRequest, _bars_product


def _bars(symbol: str, timeframe: str, rows: int) -> pl.DataFrame:
    latest = now_ms()
    step = 3_600_000
    return pl.DataFrame(
        {
            "symbol": [symbol] * rows,
            "timeframe": [timeframe] * rows,
            "timestamp": [latest - step * (rows - index) for index in range(rows)],
            "open": [1.0] * rows,
            "high": [1.0] * rows,
            "low": [1.0] * rows,
            "close": [1.0] * rows,
            "vol": [1.0] * rows,
        }
    )


def test_bar_load_request_target_rows_are_timeframe_aware() -> None:
    request = BarLoadRequest(
        symbols=("BTC", "ETH"),
        timeframes=("1H", "4H", "1D"),
        target_days=10,
        max_staleness_hours=24,
    )

    assert request.target_rows_for("1H") == 480
    assert request.target_rows_for("4H") == 120
    assert request.target_rows_for("1D") == 20
    assert request.target_rows_total() == 620


def test_bars_product_reports_timeframe_and_decision_coverage() -> None:
    request = BarLoadRequest(
        symbols=("BTC",),
        timeframes=("1H", "4H", "1D"),
        target_days=2,
        max_staleness_hours=24,
    )
    frames = {
        ("BTC", "1H"): _bars("BTC", "1H", 48),
        ("BTC", "4H"): _bars("BTC", "4H", 12),
        ("BTC", "1D"): _bars("BTC", "1D", 2),
    }

    result = _bars_product(frames, request)

    assert result.health.target_rows == 62
    assert result.health.actual_rows == 62
    assert result.health.coverage_pct == 100.0
    assert "bar_coverage:1H:actual=48:target=48:coverage=100.0" in result.health.notes
    assert "bar_coverage:4H:actual=12:target=12:coverage=100.0" in result.health.notes
    assert "bar_coverage:1D:actual=2:target=2:coverage=100.0" in result.health.notes
    assert "decision_timeframe:1H:actual=48:target=48:coverage=100.0" in result.health.notes
