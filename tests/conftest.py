"""Shared test fixtures and markers."""

import os
from pathlib import Path

import polars as pl
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: tests requiring network connection")
    config.addinivalue_line("markers", "benchmark: performance timing tests")


def pytest_collection_modifyitems(config, items):
    if not os.environ.get("RUN_INTEGRATION_TESTS"):
        skip_integration = pytest.mark.skip(reason="set RUN_INTEGRATION_TESTS=1 to enable")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Path:
    d = tmp_path / "cache"
    d.mkdir()
    return d


@pytest.fixture
def sample_bars() -> pl.DataFrame:
    """100 rows of valid OHLCV data with sequential timestamps."""
    base = 1_700_000_000_000
    interval = 3_600_000  # 1 hour in ms
    return pl.DataFrame(
        {
            "timestamp": [base + i * interval for i in range(100)],
            "open": [100.0 + i * 0.1 for i in range(100)],
            "high": [101.0 + i * 0.1 for i in range(100)],
            "low": [99.0 + i * 0.1 for i in range(100)],
            "close": [100.5 + i * 0.1 for i in range(100)],
            "vol": [1000.0 for _ in range(100)],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Int64))


@pytest.fixture
def sample_bars_newer() -> pl.DataFrame:
    """10 newer bars, partial overlap with sample_bars last 5."""
    base = 1_700_000_000_000 + 95 * 3_600_000  # starts at row 95
    return pl.DataFrame(
        {
            "timestamp": [base + i * 3_600_000 for i in range(10)],
            "open": [110.0 + i * 0.1 for i in range(10)],
            "high": [111.0 + i * 0.1 for i in range(10)],
            "low": [109.0 + i * 0.1 for i in range(10)],
            "close": [110.5 + i * 0.1 for i in range(10)],
            "vol": [1000.0 for _ in range(10)],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Int64))


@pytest.fixture
def sample_instruments() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "inst_id": ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "DOGE-USDT-SWAP"],
            "inst_type": ["SWAP", "SWAP", "SWAP"],
            "state": ["live", "live", "live"],
            "ct_val": [0.01, 0.1, 100.0],
        }
    )


@pytest.fixture
def sample_tickers() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "inst_id": ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "DOGE-USDT-SWAP"],
            "last": [65000.0, 3500.0, 0.15],
            "bid_px": [64999.0, 3499.0, 0.149],
            "ask_px": [65001.0, 3501.0, 0.151],
            "quote_volume_24h": [5_000_000_000.0, 1_000_000_000.0, 100_000.0],
            "spread_bps": [3.0, 5.0, 60.0],
            "history_coverage_pct": [100.0, 95.0, 50.0],
        }
    )
