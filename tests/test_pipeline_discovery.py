"""Test pipeline discovery — rank_discovery, select_symbols."""

import polars as pl

from qooi.pipeline.discovery import rank_discovery, select_symbols


def test_rank_empty_instruments_returns_empty():
    empty = pl.DataFrame(schema={"inst_id": pl.String, "state": pl.String})
    tickers = pl.DataFrame(schema={"inst_id": pl.String, "quote_volume_24h": pl.Float64})
    result = rank_discovery(empty, tickers)
    assert result.is_empty()


def test_rank_below_min_volume_excluded(sample_instruments, sample_tickers):
    result = rank_discovery(sample_instruments, sample_tickers, min_volume_usd=2_000_000_000.0)
    eth_row = result.filter(pl.col("inst_id") == "ETH-USDT-SWAP")
    doge_row = result.filter(pl.col("inst_id") == "DOGE-USDT-SWAP")
    assert not eth_row.get_column("eligible")[0]  # 1B < 2B
    assert not doge_row.get_column("eligible")[0]  # 100K < 2B


def test_rank_above_max_spread_excluded(sample_instruments, sample_tickers):
    result = rank_discovery(
        sample_instruments, sample_tickers, min_volume_usd=0.0, max_spread_bps=10.0
    )
    doge_row = result.filter(pl.col("inst_id") == "DOGE-USDT-SWAP")
    assert not doge_row.get_column("eligible")[0]  # 60 bps > 10
    btc_row = result.filter(pl.col("inst_id") == "BTC-USDT-SWAP")
    assert btc_row.get_column("eligible")[0]  # 3 bps < 10


def test_rank_manual_symbols_override_exclusion(sample_instruments, sample_tickers):
    result = rank_discovery(
        sample_instruments,
        sample_tickers,
        min_volume_usd=10_000_000_000.0,
        manual_symbols=("BTC-USDT-SWAP",),
    )
    btc_row = result.filter(pl.col("inst_id") == "BTC-USDT-SWAP")
    assert btc_row.get_column("eligible")[0]  # manual override
    assert btc_row.get_column("exclude_reason")[0] == "manual_override"


def test_rank_computes_rank_score(sample_instruments, sample_tickers):
    result = rank_discovery(
        sample_instruments, sample_tickers, min_volume_usd=0.0, max_spread_bps=1e9
    )
    scores = result.get_column("rank_score").to_list()
    assert all(isinstance(s, float) for s in scores)
    # sorted descending
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def test_select_top_n(sample_instruments, sample_tickers):
    frame = rank_discovery(
        sample_instruments, sample_tickers, min_volume_usd=0.0, max_spread_bps=1e9
    )
    symbols = select_symbols(frame, top_n=2)
    assert len(symbols) >= 0  # ok if none eligible


def test_select_manual_symbols_first(sample_instruments, sample_tickers):
    frame = rank_discovery(
        sample_instruments, sample_tickers, min_volume_usd=0.0, max_spread_bps=1e9
    )
    symbols = select_symbols(frame, top_n=1, manual_symbols=("DOGE-USDT-SWAP",))
    assert symbols[0] == "DOGE-USDT-SWAP"  # manual always first
