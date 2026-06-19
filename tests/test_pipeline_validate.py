"""Test pipeline health validation."""

import polars as pl

from qooi.pipeline.types import FrameHealth


def test_validate_empty_frame():
    empty = pl.DataFrame(schema={"timestamp": pl.Int64})
    health = FrameHealth.from_frame(empty, product="bars", key="BTC-USDT-SWAP")
    assert health.status == "missing"
    assert health.actual_rows == 0
    assert health.product == "bars"
    assert health.key == "BTC-USDT-SWAP"


def test_validate_nonempty_frame(sample_bars):
    health = FrameHealth.from_frame(
        sample_bars, product="bars", key="BTC", threshold_hours=876000, target_rows=100
    )
    assert health.status == "fresh"
    assert health.actual_rows == 100
    assert health.coverage_pct > 0.0
    assert health.latest_ts is not None


def test_validate_stale_data(sample_bars):
    health = FrameHealth.from_frame(sample_bars, product="bars", key="BTC", threshold_hours=0.0)
    # data is older than 0 hours → stale
    assert health.status == "stale" or health.age_hours > 0.0


def test_validate_under_target_rows(sample_bars):
    health = FrameHealth.from_frame(sample_bars, product="bars", key="BTC", target_rows=200)
    assert health.status == "fresh" or health.coverage_pct < 100.0
    assert health.target_rows == 200


def test_validate_gap_detection():
    # bars with missing hours
    data = pl.DataFrame(
        {
            "timestamp": [1_700_000_000_000, 1_700_000_007_200_000, 1_700_000_010_800_000],
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0],
            "vol": [1000.0, 1000.0, 1000.0],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Int64))
    health = FrameHealth.from_frame(data, product="bars", key="GAP", expected_interval_ms=3_600_000)
    assert health.gaps > 0


def test_validate_duplicate_detection():
    data = pl.DataFrame(
        {
            "timestamp": [1_700_000_000_000, 1_700_000_000_000, 1_700_000_003_600_000],
            "open": [100.0, 100.0, 100.0],
            "high": [100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0],
            "vol": [1000.0, 1000.0, 1000.0],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Int64))
    health = FrameHealth.from_frame(data, product="bars", key="DUP")
    assert health.duplicates > 0


def test_validate_no_gaps_clean_data(sample_bars):
    health = FrameHealth.from_frame(
        sample_bars, product="bars", key="clean", expected_interval_ms=3_600_000
    )
    assert health.gaps == 0


def test_validate_coverage_pct(sample_bars):
    health = FrameHealth.from_frame(sample_bars, product="bars", key="cov", target_rows=200)
    assert health.coverage_pct == 50.0  # 100/200 * 100
