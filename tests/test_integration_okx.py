"""Integration tests — real OKX API connection via OkxClient.

Run with: RUN_INTEGRATION_TESTS=1 uv run pytest tests/test_integration_okx.py -v
"""

import asyncio
from datetime import UTC, datetime

import pytest

from qooi.transport.okx import OkxClient


def _sync(fn):
    return asyncio.run(fn())


# ═══════════════════════════════════════════════════════════════════════════
# bars / bars_since
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
def test_fetch_bars_returns_ohlcv():
    async def _run():
        async with OkxClient(timeout=30.0) as okx:
            return await okx.bars_since("BTC-USDT-SWAP", bar="1H", limit=10)

    frame = _sync(_run)
    assert not frame.is_empty()
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    assert required <= set(frame.columns)


@pytest.mark.integration
def test_fetch_bars_timestamps_sorted():
    async def _run():
        async with OkxClient(timeout=30.0) as okx:
            return await okx.bars_since("BTC-USDT-SWAP", bar="1H", limit=20)

    frame = _sync(_run)
    assert frame.get_column("timestamp").is_sorted()


@pytest.mark.integration
def test_fetch_bars_no_duplicates():
    async def _run():
        async with OkxClient(timeout=30.0) as okx:
            return await okx.bars_since("BTC-USDT-SWAP", bar="1H", limit=50)

    frame = _sync(_run)
    ts = frame.get_column("timestamp").to_list()
    assert len(ts) == len(set(ts))


@pytest.mark.integration
def test_fetch_bars_limit():
    async def _run(limit):
        async with OkxClient(timeout=30.0) as okx:
            return await okx.bars_since("BTC-USDT-SWAP", bar="1H", limit=limit)

    for limit in (5, 10, 100):
        frame = _sync(lambda: _run(limit))
        assert frame.height <= limit


@pytest.mark.integration
def test_fetch_bars_timeframes():
    async def _run(bar):
        async with OkxClient(timeout=30.0) as okx:
            return await okx.bars_since("BTC-USDT-SWAP", bar=bar, limit=5)

    for bar in ("1H", "4H", "1D"):
        frame = _sync(lambda: _run(bar))
        assert not frame.is_empty()


@pytest.mark.integration
def test_fetch_bars_since_filters():
    async def _run():
        async with OkxClient(timeout=30.0) as okx:
            return await okx.bars_since("BTC-USDT-SWAP", bar="1H", limit=100)

    frame_all = _sync(_run)
    all_ts = sorted(frame_all.get_column("timestamp").to_list())
    mid = all_ts[len(all_ts) // 2]

    async def _run_since():
        since_date = datetime.fromtimestamp(mid / 1000, tz=UTC).strftime("%Y-%m-%d")
        async with OkxClient(timeout=30.0) as okx:
            return await okx.bars_since("BTC-USDT-SWAP", bar="1H", since=since_date, limit=100)

    frame_since = _sync(_run_since)
    assert frame_since.height > 0
    assert frame_since.get_column("timestamp").min() >= mid - 86_400_000  # within a day


# ═══════════════════════════════════════════════════════════════════════════
# book_snapshot
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
def test_fetch_books_returns_bid_ask():
    async def _run():
        async with OkxClient(timeout=30.0) as okx:
            return await okx.book_snapshot("BTC-USDT-SWAP", limit=5)

    result = _sync(_run)
    assert not result.frame.is_empty()
    assert result.frame.height >= 2


# ═══════════════════════════════════════════════════════════════════════════
# funding_history
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
def test_fetch_funding_returns_columns():
    async def _run():
        async with OkxClient(timeout=30.0) as okx:
            return await okx.funding_history("BTC-USDT-SWAP", limit=10)

    result = _sync(_run)
    assert not result.frame.is_empty()
    funding_cols = {"funding_rate", "funding_time"}
    found = set(result.frame.columns) & funding_cols
    assert found


@pytest.mark.integration
def test_fetch_funding_sorted():
    async def _run():
        async with OkxClient(timeout=30.0) as okx:
            return await okx.funding_history("BTC-USDT-SWAP", limit=20)

    result = _sync(_run)
    if "funding_time" in result.frame.columns:
        assert result.frame.get_column("funding_time").is_sorted()


# ═══════════════════════════════════════════════════════════════════════════
# Discovery
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
def test_discover_instruments_returns_swaps():
    async def _run():
        async with OkxClient(timeout=30.0) as okx:
            return await okx.instruments()

    result = _sync(_run)
    assert not result.frame.is_empty()
    assert result.frame.height > 10


@pytest.mark.integration
def test_fetch_tickers_returns_prices():
    async def _run():
        async with OkxClient(timeout=30.0) as okx:
            return await okx.tickers()

    result = _sync(_run)
    assert not result.frame.is_empty()


@pytest.mark.integration
def test_instruments_tickers_joinable():
    async def _run():
        async with OkxClient(timeout=30.0) as okx:
            instruments = await okx.instruments()
            tickers = await okx.tickers()
            return instruments, tickers

    instruments, tickers = _sync(_run)
    joined = instruments.frame.join(tickers.frame, on="inst_id", how="inner")
    assert not joined.is_empty()
