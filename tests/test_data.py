"""Data processing tests — indicators, cache, market data."""

import asyncio

import polars as pl

import qooi.exchange.market as market
from qooi.core.config import CORE_UNIVERSE, RESEARCH_UNIVERSE
from qooi.exchange.market import _parse_bars, okx_index_inst_id
from qooi.exchange.store import (
    AsyncCacheStore,
    BooksRequest,
    CacheStore,
    FundingRequest,
    HistoryRefreshRequest,
    _resource_path,
    plan_history,
    validate_history,
)
from qooi.research.data import (
    _add_missing_context_columns,
    _compact_higher_timeframe_context,
    _context_min_bars,
    _mark_higher_context_available,
    add_mtf_state_keys,
    attach_higher_timeframe_context,
)
from qooi.strategies.indicators import add_indicators, attach_order_book_features
from qooi.strategies.structure import add_price_structure_stage_features


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


def test_cache_store_path_is_source_aware():
    assert _resource_path("BTC-USDT", bar="1H").name == "BTC_USDT_1H.parquet"
    assert _resource_path("XAU/USDT", bar="1h").name == "XAU_USDT_1H.parquet"
    assert _resource_path("BTC-USDT-SWAP", bar="1H", source="mark").name == (
        "BTC_USDT_SWAP_MARK_1H.parquet"
    )
    assert _resource_path("BTC-USD", bar="1H", source="index").name == "BTC_USD_INDEX_1H.parquet"
    assert not hasattr(CacheStore, "_path")
    assert not hasattr(CacheStore, "validate_ohlcv")
    assert not hasattr(CacheStore, "describe_ohlcv")
    assert not hasattr(CacheStore, "load")
    assert not hasattr(CacheStore, "load_history")
    assert not hasattr(CacheStore, "load_or_refresh")
    assert not hasattr(CacheStore, "intraday_frame")
    assert not hasattr(CacheStore, "refresh_sync")
    assert not hasattr(CacheStore, "refresh_funding")
    assert not hasattr(CacheStore, "load_funding")
    assert not hasattr(CacheStore, "cache_order_book")
    assert not hasattr(CacheStore, "record_order_book")
    assert not hasattr(CacheStore, "load_order_book")
    assert hasattr(CacheStore, "bars")
    assert hasattr(CacheStore, "funding")
    assert hasattr(CacheStore, "books")
    assert not hasattr(AsyncCacheStore, "load_history_async")
    assert not hasattr(AsyncCacheStore, "refresh_async")
    assert not hasattr(AsyncCacheStore, "refresh_many_async")


def test_market_exports_resource_first_exchange_api():
    for removed in (
        "MarketData",
        "ExchangeBackend",
        "OhlcvProvider",
        "AsyncOhlcvProvider",
        "OrderBookProvider",
        "StreamProvider",
        "FundingRateProvider",
        "CcxtBackend",
        "OkxSdkBackend",
        "OkxAsyncBackend",
        "CcxtProBackend",
        "ObSnapshot",
        "_parse_ohlcv",
    ):
        assert not hasattr(market, removed)
    for exposed in (
        "SyncExchange",
        "AsyncExchange",
        "OkxSyncExchange",
        "OkxAsyncExchange",
        "CcxtSyncExchange",
        "CcxtBooksStream",
        "BookSnapshot",
        "_parse_bars",
    ):
        assert hasattr(market, exposed)


def test_research_pairs_do_not_change_live_pairs():
    live_symbols = {pair.asset.symbol for pair in CORE_UNIVERSE}
    research_symbols = {pair.asset.symbol for pair in RESEARCH_UNIVERSE}

    assert live_symbols < research_symbols
    assert "XRP-USDT-SWAP" in research_symbols
    assert {
        "BTC-USDT-SWAP",
        "ETH-USDT-SWAP",
        "SOL-USDT-SWAP",
        "BNB-USDT-SWAP",
        "XRP-USDT-SWAP",
        "ADA-USDT-SWAP",
    } <= research_symbols


def test_xau_contract_value_matches_okx_metadata():
    xau = next(pair for pair in CORE_UNIVERSE if pair.asset.symbol == "XAU-USDT-SWAP")

    assert xau.asset.ct_val == 0.001


def test_core_swap_lot_sizes_match_okx_metadata():
    by_symbol = {pair.asset.symbol: pair.asset for pair in CORE_UNIVERSE}

    assert by_symbol["ETH-USDT-SWAP"].min_contracts == 0.01
    assert by_symbol["ETH-USDT-SWAP"].lot_size == 0.01
    assert by_symbol["SOL-USDT-SWAP"].min_contracts == 0.01
    assert by_symbol["SOL-USDT-SWAP"].lot_size == 0.01
    assert by_symbol["BTC-USDT-SWAP"].min_contracts == 0.01
    assert by_symbol["BTC-USDT-SWAP"].lot_size == 0.01
    assert by_symbol["BTC-USDT-SWAP"].tick_size == 0.1


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
    assert coverage.source == "trade"
    assert "fetch_pages=3" in coverage.notes
    assert "fetch_stop=short_page" in coverage.notes


def test_source_aware_index_mark_parser_uses_confirm_not_volume():
    parsed = _parse_bars(
        [
            [1_000, "100", "101", "99", "100.5", "1"],
            [2_000, "101", "102", "100", "101.5", "0"],
        ],
        source="index",
    )

    assert parsed.height == 1
    assert parsed["timestamp"].to_list() == [1_000]
    assert parsed["volume"].to_list() == [0.0]


def test_okx_index_inst_id_mapping_is_explicit():
    assert okx_index_inst_id("BTC-USDT-SWAP") == "BTC-USD"
    try:
        okx_index_inst_id("XAU-USDT-SWAP")
    except ValueError as exc:
        assert "No explicit OKX index instrument mapping" in str(exc)
    else:
        raise AssertionError("unsupported index mapping should fail clearly")


def test_structure_stage_unknown_semantics_are_split():
    timestamps = [idx * 3_600_000 for idx in range(80)]
    frame = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0 + (idx % 2) for idx in range(80)],
            "high": [110.0 + (idx % 3) for idx in range(80)],
            "low": [90.0 - (idx % 3) for idx in range(80)],
            "close": [100.0 + (idx % 2) for idx in range(80)],
            "vol": [10.0] * 80,
        }
    )
    out = add_price_structure_stage_features(range_width_atr_max=0.01)(add_indicators(frame))

    assert out["market_stage"][0] == "warmup"
    assert out["stage_unknown_reason"][0] == "warmup"
    assert "wide_range" in out["market_stage"].to_list()
    assert "wide_range" in out["stage_unknown_reason"].to_list()
    transition_conflicts = out.filter(
        (pl.col("stage_unknown_reason") == "transition") & (pl.col("market_stage") != "transition")
    )
    assert transition_conflicts.is_empty()


def test_raw_unknown_is_not_used_for_normal_warmup_or_wide_range():
    timestamps = [idx * 3_600_000 for idx in range(80)]
    frame = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0 + (idx % 2) for idx in range(80)],
            "high": [110.0 + (idx % 3) for idx in range(80)],
            "low": [90.0 - (idx % 3) for idx in range(80)],
            "close": [100.0 + (idx % 2) for idx in range(80)],
            "vol": [10.0] * 80,
        }
    )
    out = add_price_structure_stage_features(range_width_atr_max=0.01)(add_indicators(frame))

    normal_reasons = {"warmup", "wide_range"}
    normal = out.filter(pl.col("stage_unknown_reason").is_in(list(normal_reasons)))

    assert normal.height > 0
    assert "unknown" not in normal["market_stage"].to_list()


def test_async_refresh_skips_deep_history_for_complete_incremental_cache(tmp_path, monkeypatch):
    import qooi.exchange.store as store_module

    monkeypatch.setattr(store_module, "CACHE_DIR", tmp_path)
    request = HistoryRefreshRequest("BTC-USDT-SWAP", "1H", days=1, min_bars=3)
    target = plan_history(
        request.inst_id, request.bar, days=request.days, min_bars=request.min_bars
    )
    existing = pl.DataFrame(
        {
            "timestamp": [
                target.target_since_ms,
                target.target_since_ms + 3_600_000,
                target.target_since_ms + 2 * 3_600_000,
            ],
            "open": [1.0, 2.0, 3.0],
            "high": [1.0, 2.0, 3.0],
            "low": [1.0, 2.0, 3.0],
            "close": [1.0, 2.0, 3.0],
            "vol": [1.0, 1.0, 1.0],
        }
    )
    existing.write_parquet(_resource_path(request.inst_id, bar=request.bar))

    class FakeAsyncExchange:
        last_bars_audit = ("fetch_backend=fake",)

        def __init__(self):
            self.deep_calls = 0

        async def bars_since(self, *args, **kwargs):
            self.deep_calls += 1
            return pl.DataFrame()

        async def bars(self, *args, **kwargs):
            return pl.DataFrame()

    exchange = FakeAsyncExchange()
    result = asyncio.run(AsyncCacheStore(exchange).many((request,)))[0]

    assert result.error is None
    assert exchange.deep_calls == 0
    assert "refresh_skipped_history=yes" in result.coverage.notes


def test_async_refresh_many_dedupes_requests_and_honors_concurrency(tmp_path, monkeypatch):
    import qooi.exchange.store as store_module

    monkeypatch.setattr(store_module, "CACHE_DIR", tmp_path)

    active = 0
    max_active = 0

    class FakeAsyncExchange:
        last_bars_audit = ("fetch_backend=fake",)

        async def bars_since(self, symbol, bar, since, limit, source="trade"):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return pl.DataFrame(
                {
                    "timestamp": list(range(limit)),
                    "open": [1.0] * limit,
                    "high": [1.0] * limit,
                    "low": [1.0] * limit,
                    "close": [1.0] * limit,
                    "vol": [1.0] * limit,
                }
            )

        async def bars(self, *args, **kwargs):
            return pl.DataFrame()

    requests = (
        HistoryRefreshRequest("BTC-USDT-SWAP", "1H", min_bars=2),
        HistoryRefreshRequest("BTC-USDT-SWAP", "1H", min_bars=2),
        HistoryRefreshRequest("ETH-USDT-SWAP", "1H", min_bars=2),
    )

    results = asyncio.run(AsyncCacheStore(FakeAsyncExchange()).many(requests, concurrency=1))

    assert len(results) == 2
    assert max_active == 1


def test_sync_store_books_requires_async_for_ws():
    try:
        CacheStore().books(BooksRequest("BTC-USDT-SWAP", samples=1, transport="ws"))
    except RuntimeError as exc:
        assert "AsyncCacheStore.books" in str(exc)
    else:
        raise AssertionError("sync websocket books collection should fail clearly")


def test_sync_store_funding_reads_existing_cache(tmp_path, monkeypatch):
    import qooi.exchange.store as store_module

    monkeypatch.setattr(store_module, "CACHE_DIR", tmp_path)
    frame = pl.DataFrame({"timestamp": [1_000], "funding_rate": [0.01], "funding_time": [1_000]})
    frame.write_parquet(store_module._resource_path("BTC-USDT-SWAP", resource="funding"))

    out = CacheStore().funding(FundingRequest("BTC-USDT-SWAP"))

    assert out["funding_rate"].to_list() == [0.01]


def test_mtf_context_min_bars_are_timeframe_specific():
    assert _context_min_bars("4H", 730, role="higher_context") == 4_630
    assert _context_min_bars("1D", 730, role="higher_context") == 980


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


def test_higher_timeframe_context_uses_only_closed_bars_and_keeps_rows():
    base = pl.DataFrame(
        {
            "timestamp": [0, 3_600_000, 14_400_000, 18_000_000],
            "close": [100.0, 101.0, 102.0, 103.0],
        }
    )
    h4 = pl.DataFrame(
        {
            "timestamp": [0, 14_400_000],
            "close": [200.0, 300.0],
            "trend": ["old", "future"],
        }
    )

    out = attach_higher_timeframe_context(base, h4, prefix="h4")

    assert out.height == base.height
    assert out["h4_close"].to_list() == [None, None, 200.0, 200.0]
    assert out["h4_trend"].to_list() == [None, None, "old", "old"]


def test_higher_timeframe_structure_context_uses_closed_bars():
    base = pl.DataFrame({"timestamp": [0, 3_600_000, 14_400_000], "close": [1.0, 2.0, 3.0]})
    h4 = pl.DataFrame(
        {
            "timestamp": [0, 14_400_000],
            "structure_trend_state": ["uptrend", "downtrend"],
            "market_stage": ["markup", "markdown"],
        }
    )

    out = attach_higher_timeframe_context(base, h4, prefix="h4")

    assert out["h4_structure_trend_state"].to_list() == [None, None, "uptrend"]
    assert out["h4_market_stage"].to_list() == [None, None, "markup"]


def test_d1_context_does_not_use_future_daily_close():
    base = pl.DataFrame(
        {
            "timestamp": [0, 23 * 3_600_000, 24 * 3_600_000, 47 * 3_600_000],
            "close": [100.0, 101.0, 102.0, 103.0],
        }
    )
    d1 = pl.DataFrame(
        {
            "timestamp": [0, 86_400_000],
            "close": [200.0, 300.0],
            "trend": ["old", "future"],
        }
    )

    out = attach_higher_timeframe_context(base, d1, prefix="d1")

    assert out.height == base.height
    assert out["d1_close"].to_list() == [None, None, 200.0, 200.0]
    assert out["d1_trend"].to_list() == [None, None, "old", "old"]


def test_higher_timeframe_compact_context_marks_availability():
    timestamps = [idx * 14_400_000 for idx in range(220)]
    h4 = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0 + idx for idx in range(220)],
            "high": [101.0 + idx for idx in range(220)],
            "low": [99.0 + idx for idx in range(220)],
            "close": [100.5 + idx for idx in range(220)],
            "vol": [10.0] * 220,
        }
    )
    compact = _compact_higher_timeframe_context(h4)
    base = pl.DataFrame({"timestamp": [timestamps[-1] + 14_400_000], "close": [500.0]})

    out = _mark_higher_context_available(
        attach_higher_timeframe_context(base, compact, prefix="h4"), "h4"
    )

    assert {"h4_close", "h4_ema_20", "h4_trend_state", "h4_context_available"} <= set(out.columns)
    assert out["h4_context_available"].to_list() == [True]


def test_higher_timeframe_compact_context_adds_structure_columns():
    timestamps = [idx * 14_400_000 for idx in range(80)]
    h4 = pl.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0 + idx for idx in range(80)],
            "high": [101.0 + idx for idx in range(80)],
            "low": [99.0 + idx for idx in range(80)],
            "close": [100.5 + idx for idx in range(80)],
            "vol": [10.0] * 80,
        }
    )

    compact = _compact_higher_timeframe_context(h4)

    assert {"structure_trend_state", "market_stage", "range_compression"} <= set(compact.columns)


def test_mtf_state_key_normalizes_missing_context_as_data_error():
    frame = pl.DataFrame(
        {
            "timestamp": [1],
            "h4_market_stage": ["range"],
            "liquidity_event_type": ["failed_breakout_low"],
        }
    )

    out = add_mtf_state_keys(frame)

    assert out["mtf_state_key"].to_list() == ["data_error|range|data_error"]
    assert out["mtf_event_state_key"].to_list() == [
        "data_error|range|data_error|failed_breakout_low"
    ]


def test_state_keys_remain_human_readable():
    frame = pl.DataFrame(
        {
            "timestamp": [1],
            "d1_structure_trend_state": ["uptrend"],
            "h4_market_stage": ["wide_range"],
            "h1_market_stage": ["transition"],
        }
    )

    out = add_mtf_state_keys(frame)

    assert out["mtf_state_key"].to_list() == ["uptrend|wide_range|transition"]


def test_missing_higher_timeframe_context_marks_unavailable():
    base = pl.DataFrame({"timestamp": [0], "close": [100.0]})

    out = _add_missing_context_columns(base, "d1")

    assert out["d1_context_available"].to_list() == [False]


