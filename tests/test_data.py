"""Data processing tests — indicators, cache, market data."""

import polars as pl

from qooi.core.config import PAIRS, RESEARCH_PAIRS
from qooi.exchange.market import _cache_path as market_cache_path
from qooi.exchange.store import CacheStore, plan_history, validate_history
from qooi.strategies.indicators import add_indicators, attach_order_book_features


class TestIndicators:
    def test_add_indicators_columns(self):
        df = pl.DataFrame(
            {
                "timestamp": list(
                    range(1_700_000_000_000, 1_700_000_000_000 + 50 * 14_400_000, 14_400_000)
                ),
                "open": [100.0] * 50,
                "high": [102.0] * 50,
                "low": [98.0] * 50,
                "close": [100.0] * 50,
                "vol": [1000.0] * 50,
            }
        )
        df = add_indicators(df)
        for col in ("atr_14", "ema_20", "ema_50", "ema_200"):
            assert col in df.columns, f"Missing {col}"
        # ATR should be positive for non-flat prices
        assert df["atr_14"].drop_nulls().sum() > 0

    def test_add_indicators_on_real_data(self):
        df = pl.read_parquet("data/cache/BTC_USDT_4H.parquet")
        df = add_indicators(df)
        assert df.height > 0
        assert df["atr_14"].drop_nulls().len() > df.height * 0.8, (
            "ATR should have values for 80%+ of rows"
        )


def test_cache_store_path_matches_market_cache_path():
    assert CacheStore._path("BTC-USDT", "1H") == market_cache_path("BTC-USDT", "1H")
    assert CacheStore._path("XAU/USDT", "1h") == market_cache_path("XAU/USDT", "1h")
    assert not hasattr(CacheStore, "validate_ohlcv")
    assert not hasattr(CacheStore, "describe_ohlcv")


def test_research_pairs_do_not_change_live_pairs():
    live_symbols = {pair.asset.symbol for pair in PAIRS}
    research_symbols = {pair.asset.symbol for pair in RESEARCH_PAIRS}

    assert live_symbols < research_symbols
    assert "XRP-USDT-SWAP" in research_symbols


def test_history_coverage_preserves_fetch_audit_notes():
    target = plan_history("ETH-USDT-SWAP", "1H", days=730, min_bars=12000)
    df = pl.DataFrame(
        {
            "timestamp": [target.target_since_ms, target.target_since_ms + 3_600_000],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "vol": [10.0, 11.0],
        }
    )

    coverage = validate_history(
        df,
        target,
        refreshed=True,
        extra_notes=("fetch_pages=3", "fetch_stop=short_page"),
    )

    assert coverage.target.target_days == 730
    assert coverage.target.target_bars == 12000
    assert "fetch_pages=3" in coverage.notes
    assert "fetch_stop=short_page" in coverage.notes


def test_order_book_features_are_strategy_pipe():
    bars = pl.DataFrame(
        {
            "timestamp": [1_000, 2_000],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "vol": [10.0, 11.0],
        }
    )
    snapshots = pl.DataFrame(
        {
            "timestamp": [1_100, 1_200],
            "ob_bid_price": [100.0, 100.1],
            "ob_ask_price": [100.2, 100.3],
            "ob_bid_vol_5": [5.0, 7.0],
            "ob_ask_vol_5": [4.0, 6.0],
            "ob_bid_vol_25": [15.0, 17.0],
            "ob_ask_vol_25": [14.0, 16.0],
            "ob_bid_vol": [20.0, 22.0],
            "ob_ask_vol": [18.0, 20.0],
            "ob_imbalance_5": [0.1, 0.2],
            "ob_imbalance_25": [0.05, 0.15],
        }
    )

    enriched = attach_order_book_features(bars, snapshots)

    assert "ob_imbalance_5" in enriched.columns
    assert enriched["ob_samples"][0] == 2
    assert not hasattr(CacheStore, "attach_order_book")
