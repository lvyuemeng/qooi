from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

import httpx
import polars as pl
import pytest

from qooi.scanner.config import PotentialConfig, SourceConfig, TransitionConfig
from qooi.sources.artifacts import coerce_frame, source_capability, source_manifest_family
from qooi.sources.bundle import (
    SourceBundle,
    latest_timestamp,
    missing_symbols,
    replace_symbol_rows,
    source_symbols,
)
from qooi.sources.coingecko import fetch_coingecko_trending
from qooi.sources.collect import (
    SourceCollectRequest,
    _collect_funding_source,
    _collect_rubik_source,
    _combine_source_results,
    _incremental_rubik_window,
)
from qooi.sources.context import (
    SourceContextRequest,
    manifest_latest_maps,
    merge_context_frames,
    source_availability,
)
from qooi.sources.coverage import (
    eligible_backfill_symbols,
    eligible_fetch_symbols,
    latest_manifest_status,
    stale_symbols,
)
from qooi.sources.http import SourceHttpError, request_json, request_json_sync, sanitize_error
from qooi.sources.manifest import manifest_frame, source_manifest_row
from qooi.sources.models import SourceResult
from qooi.sources.okx import (
    _fetch_okx_frame,
    _normalize_current_funding,
    _normalize_funding,
    _normalize_instruments,
    _normalize_open_interest_history,
    _normalize_ratio_rows,
    _normalize_taker_volume_contract,
    fetch_okx_funding_rate,
    normalize_okx_trades,
)
from qooi.sources.okx_ws import (
    collect_okx_ws_public,
    normalize_okx_ws_books,
    normalize_okx_ws_trades,
    okx_ws_subscribe_message,
)
from qooi.sources.schema import SOURCE_FUNDING_SCHEMA


def test_collect_module_exposes_demand_first_contracts() -> None:
    collect = importlib.import_module("qooi.sources.collect")

    need = collect.SourceNeed(
        family="funding",
        symbols=("BTC-USDT-SWAP",),
        start_ms=1_000,
        end_ms=2_000,
        min_rows=2,
        freshness_ms=3_600_000,
        mode="both",
    )
    plan = collect.SourceFetchPlan(
        family="funding",
        raw_source="funding_rate",
        symbol="BTC-USDT-SWAP",
        start_ms=None,
        end_ms=2_000,
        limit=1,
        reason="current_freshness",
    )

    assert need.mode == "both"
    assert plan.raw_source == "funding_rate"


def test_source_needs_from_request_derives_quantitative_family_demand() -> None:
    collect = importlib.import_module("qooi.sources.collect")
    request = SourceContextRequest(
        output_dir=Path("data/output/potential"),
        symbols=("BTC-USDT-SWAP",),
        context_symbols=("ETH-USDT-SWAP",),
        discovery=pl.DataFrame(),
        target_days=3,
        concurrency=3,
        refresh_mode="incremental",
        source=SourceConfig(
            max_staleness_hours=6,
            book_depth=20,
            trade_limit=50,
            funding_limit=400,
            rubik_limit=144,
            book_mode="snapshot",
            rubik_period="1H",
            disabled_sources=("trades",),
        ),
    )

    needs = collect.source_needs_from_request(request, start_ms=1_000, end_ms=2_000)
    by_family = {need.family: need for need in needs}

    assert "trades" not in by_family
    assert by_family["books"].mode == "snapshot"
    assert by_family["books"].min_rows == 1
    assert by_family["books"].freshness_ms == 6 * 60 * 60 * 1000
    assert by_family["funding"].mode == "both"
    assert by_family["funding"].symbols == ("ETH-USDT-SWAP",)
    assert by_family["funding"].min_rows == 9
    assert by_family["open_interest"].mode == "history"
    assert by_family["open_interest"].min_rows == 72


def test_source_context_request_resolves_refresh_without_scanner_root_config() -> None:
    workflow = importlib.import_module("qooi.scanner.workflow")
    config = PotentialConfig(
        output=Path("data/output/potential/report.md"),
        days=2,
        refresh_mode="force",
        fetch_concurrency=7,
        transition=TransitionConfig(history_days=5),
        source=SourceConfig(book_depth=20),
    )

    request = workflow.source_context_request(
        config,
        symbols=("BTC-USDT-SWAP",),
        context_symbols=("ETH-USDT-SWAP",),
        discovery=pl.DataFrame({"symbol": ["ETH-USDT-SWAP"]}),
    )

    assert isinstance(request, SourceContextRequest)
    assert request.output_dir == Path("data/output/potential")
    assert request.target_days == 5
    assert request.concurrency == 7
    assert request.refresh_mode == "force"
    assert request.source.book_depth == 20
    assert request.symbols == ("BTC-USDT-SWAP",)
    assert request.context_symbols == ("ETH-USDT-SWAP",)


def test_source_config_rejects_nested_refresh_mode() -> None:
    with pytest.raises(ValueError, match="refresh_mode"):
        SourceConfig.model_validate({"refresh_mode": "cache_only"})


def test_source_context_module_does_not_accept_potential_config_boundary() -> None:
    import inspect

    import qooi.sources.context as context

    signature = inspect.signature(context.load_source_context)

    assert tuple(signature.parameters) == ("request",)
    assert not hasattr(context, "PotentialSourceConfig")


def _source_request(
    *,
    symbols: tuple[str, ...] = ("BTC-USDT-SWAP",),
    source: SourceConfig | None = None,
    target_days: int = 60,
) -> SourceContextRequest:
    return SourceContextRequest(
        output_dir=Path("data/output/potential"),
        symbols=symbols,
        context_symbols=symbols,
        discovery=pl.DataFrame(),
        target_days=target_days,
        concurrency=1,
        refresh_mode="cache_only",
        source=source or SourceConfig(),
    )


def test_source_family_table_derives_manifest_aliases_and_merge_keys() -> None:
    from qooi.sources.artifacts import source_family, source_manifest_family

    funding = source_family("funding")

    assert source_manifest_family("funding_rate") == "funding"
    assert funding.artifact == "source_funding"
    assert funding.timestamp_col == "known_at_ms"
    assert funding.raw_sources == ("funding", "funding_rate")
    assert funding.row_kind_col == "funding_source_kind"
    assert funding.history_kind == "history"
    assert funding.merge_keys == (("symbol", "funding_time"), ("symbol", "timestamp"))


def test_source_capability_marks_rubik_contract_sources_provider_bounded() -> None:
    capability = source_capability("long_short_ratio_contract", period="1H")

    assert capability.family == "long_short_ratios"
    assert capability.scope == "instrument"
    assert capability.max_rows == 1440
    assert capability.supports_latest_refresh_int == 1
    assert capability.supports_backfill_int == 1
    assert capability.required_for_review_int == 1
    assert capability.required_for_evidence_int == 0
    assert capability.optional_int == 0


def test_source_availability_uses_frame_freshness_over_latest_missing_manifest() -> None:
    current_ms = 2_000_000
    request = _source_request(
        symbols=("ACT-USDT-SWAP",),
        target_days=730,
        source=SourceConfig(max_staleness_hours=24, rubik_period="1H"),
    )
    frames = {
        "long_short_ratios": pl.DataFrame(
            {
                "symbol": ["ACT-USDT-SWAP", "ACT-USDT-SWAP"],
                "timestamp": [current_ms - 3_600_000, current_ms - 1_800_000],
                "long_short_ratio": [1.1, 1.2],
            }
        )
    }
    manifest = manifest_frame(
        [
            source_manifest_row(
                symbol="ACT-USDT-SWAP",
                source="long_short_ratio_contract",
                phase="collect-source",
                status="missing",
                rows=0,
                warning="long_short_ratio_contract_missing",
            )
        ]
    )

    rows = source_availability(
        frames,
        manifest,
        request,
        current_ms=current_ms,
    )
    row = next(item for item in rows if item.family == "long_short_ratios")

    assert row.rows == 2
    assert row.latest_age_hours == 0.5
    assert row.latest_fetch_status == "missing"
    assert row.frame_fresh_int == 1
    assert row.fetch_failed_frame_fresh_int == 1
    assert row.frame_missing_int == 0
    assert row.usable_int == 1
    assert row.status == "provider_bounded"
    assert row.provider_bounded_int == 1
    assert row.coverage_capability_pct > row.coverage_target_pct


def test_source_availability_treats_messages_as_optional_absent_without_penalty() -> None:
    request = _source_request(source=SourceConfig(max_staleness_hours=24))

    rows = source_availability(
        {},
        pl.DataFrame(),
        request,
        current_ms=2_000_000,
    )
    row = next(item for item in rows if item.family == "messages")

    assert row.rows == 0
    assert row.status == "optional_absent"
    assert row.optional_absent_int == 1
    assert row.frame_missing_int == 0
    assert row.rank_penalty_weight == 0.0
    assert row.source_penalty_component == 0.0


def test_funding_history_normalizer_marks_history_rows() -> None:
    out = _normalize_funding([{"fundingTime": "1000", "fundingRate": "0.0001"}])

    assert out.to_dicts() == [
        {
            "timestamp": 1000,
            "funding_time": 1000,
            "funding_rate": 0.0001,
            "funding_source_kind": "history",
            "known_at_ms": 1000,
            "next_funding_rate": None,
            "next_funding_time": None,
        }
    ]


def test_current_funding_normalizer_preserves_known_at_and_next_fields() -> None:
    out = _normalize_current_funding(
        [
            {
                "instId": "BTC-USDT-SWAP",
                "ts": "2000",
                "fundingRate": "0.0002",
                "fundingTime": "3000",
                "nextFundingRate": "0.0003",
                "nextFundingTime": "4000",
            }
        ],
        symbol="BTC-USDT-SWAP",
    )

    assert out.to_dicts() == [
        {
            "symbol": "BTC-USDT-SWAP",
            "timestamp": 2000,
            "funding_rate": 0.0002,
            "next_funding_rate": 0.0003,
            "funding_time": 3000,
            "next_funding_time": 4000,
            "funding_source_kind": "current",
            "known_at_ms": 2000,
        }
    ]


def test_funding_schema_keeps_current_and_history_semantics() -> None:
    frame = pl.DataFrame(
        {
            "symbol": ["BTC-USDT-SWAP", "BTC-USDT-SWAP"],
            "timestamp": [1000, 2000],
            "funding_time": [1000, 3000],
            "funding_rate": [0.0001, 0.0002],
            "funding_source_kind": ["history", "current"],
            "known_at_ms": [1000, 2000],
            "next_funding_rate": [None, 0.0003],
            "next_funding_time": [None, 4000],
            "provider_extra": ["dropped", "dropped"],
        }
    )

    out = coerce_frame(frame, SOURCE_FUNDING_SCHEMA)

    assert out.columns == list(SOURCE_FUNDING_SCHEMA)
    assert out.select(
        "funding_source_kind", "known_at_ms", "next_funding_rate", "next_funding_time"
    ).to_dicts() == [
        {
            "funding_source_kind": "history",
            "known_at_ms": 1000,
            "next_funding_rate": None,
            "next_funding_time": None,
        },
        {
            "funding_source_kind": "current",
            "known_at_ms": 2000,
            "next_funding_rate": 0.0003,
            "next_funding_time": 4000,
        },
    ]


def test_swap_trade_notional_uses_contract_value() -> None:
    out = normalize_okx_trades(
        [{"ts": "1", "tradeId": "a", "px": "100", "sz": "2", "side": "buy"}],
        contract_value=0.01,
        contract_value_currency="USDT",
    )

    assert out["notional_usd"][0] == 2.0


def test_sources_package_does_not_import_scanner_or_trading_layers() -> None:
    import qooi.sources as sources

    source_root = sources.__path__[0]
    forbidden = (
        "qooi.scanner",
        "qooi.core.executor",
        "qooi.core.basket",
        "qooi.core.recovery",
        "qooi.exchange.trading",
    )
    for path in Path(source_root).glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden), path


def test_source_bundle_helpers_operate_on_loaded_frames() -> None:
    frame = pl.DataFrame(
        [
            {"symbol": "BTC-USDT-SWAP", "timestamp": 100},
            {"symbol": "ETH-USDT-SWAP", "timestamp": 200},
            {"symbol": "BTC-USDT-SWAP", "timestamp": 300},
        ]
    )

    assert source_symbols(frame) == {"BTC-USDT-SWAP", "ETH-USDT-SWAP"}
    assert missing_symbols(frame, ("SOL-USDT-SWAP", "BTC-USDT-SWAP")) == ("SOL-USDT-SWAP",)
    assert latest_timestamp(frame, symbol="BTC-USDT-SWAP") == 300
    assert latest_timestamp(frame, symbol="DOGE-USDT-SWAP") is None


def test_replace_symbol_rows_preserves_unrelated_symbols() -> None:
    existing = pl.DataFrame(
        [
            {"symbol": "BTC-USDT-SWAP", "timestamp": 100},
            {"symbol": "ETH-USDT-SWAP", "timestamp": 100},
        ]
    )
    incoming = pl.DataFrame([{"symbol": "BTC-USDT-SWAP", "timestamp": 200}])

    out = replace_symbol_rows(existing, incoming).sort(["symbol", "timestamp"])

    assert out.to_dicts() == [
        {"symbol": "BTC-USDT-SWAP", "timestamp": 200},
        {"symbol": "ETH-USDT-SWAP", "timestamp": 100},
    ]


def test_source_freshness_helpers_select_missing_stale_and_refresh_symbols() -> None:
    frame = pl.DataFrame(
        [
            {"symbol": "BTC-USDT-SWAP", "timestamp": 1_000},
            {"symbol": "ETH-USDT-SWAP", "timestamp": 9_000},
        ]
    )
    symbols = ("BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP")

    assert stale_symbols(frame, symbols, now_ms=10_000, max_age_ms=5_000) == (
        "BTC-USDT-SWAP",
        "SOL-USDT-SWAP",
    )
    assert eligible_fetch_symbols(
        frame, symbols, now_ms=10_000, max_age_ms=5_000, refresh=False
    ) == ("BTC-USDT-SWAP", "SOL-USDT-SWAP")
    assert (
        eligible_fetch_symbols(frame, symbols, now_ms=10_000, max_age_ms=5_000, refresh=True)
        == symbols
    )


def test_source_freshness_uses_configurable_timestamp_column() -> None:
    frame = pl.DataFrame([{"symbol": "BTC-USDT-SWAP", "funding_time": 1_000}])

    assert eligible_fetch_symbols(
        frame,
        ("BTC-USDT-SWAP",),
        now_ms=10_000,
        max_age_ms=5_000,
        refresh=False,
        timestamp_col="funding_time",
    ) == ("BTC-USDT-SWAP",)


def test_source_backfill_eligibility_fetches_shallow_but_fresh_history() -> None:
    frame = pl.DataFrame(
        {
            "symbol": ["BTC-USDT-SWAP"] * 50 + ["ETH-USDT-SWAP"] * 720,
            "timestamp": list(range(51_000, 101_000, 1_000))
            + list(range(-619_000, 101_000, 1_000)),
        }
    )

    assert eligible_backfill_symbols(
        frame,
        ("BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"),
        target_start_ms=-619_000,
        now_ms=101_000,
        max_age_ms=5_000,
        min_rows=720,
        refresh=False,
    ) == ("BTC-USDT-SWAP", "SOL-USDT-SWAP")


def test_rubik_incremental_window_backfills_before_existing_earliest_history() -> None:
    frame = pl.DataFrame(
        {
            "symbol": ["BTC-USDT-SWAP"] * 50,
            "timestamp": list(range(51_000, 101_000, 1_000)),
        }
    )
    request = SourceCollectRequest(
        output_dir=Path("data/output/test"),
        symbols=("BTC-USDT-SWAP",),
        discovery=pl.DataFrame(),
        concurrency=1,
        book_mode="off",
        book_depth=0,
        max_source_staleness_hours=24,
        trade_limit=50,
        funding_limit=50,
        rubik_period="1H",
        rubik_limit=50,
        rubik_taker_unit="2",
        disabled_sources=(),
        disabled_symbols=(),
        target_source_start_ms=-619_000,
        target_source_end_ms=101_000,
        rubik_min_rows=720,
    )

    assert _incremental_rubik_window(frame, "BTC-USDT-SWAP", request) == (
        None,
        "50999",
    )


def test_rubik_backfill_fetches_multiple_pages_until_depth() -> None:
    existing = pl.DataFrame(
        {
            "symbol": ["BTC-USDT-SWAP"] * 50,
            "timestamp": list(range(51_000, 101_000, 1_000)),
            "open_interest": [float(i) for i in range(50)],
        }
    )
    request = SourceCollectRequest(
        output_dir=Path("data/output/test"),
        symbols=("BTC-USDT-SWAP",),
        discovery=pl.DataFrame(),
        concurrency=1,
        book_mode="off",
        book_depth=0,
        max_source_staleness_hours=24,
        trade_limit=50,
        funding_limit=50,
        rubik_period="1H",
        rubik_limit=50,
        rubik_taker_unit="2",
        disabled_sources=(),
        disabled_symbols=(),
        target_source_start_ms=-49_000,
        target_source_end_ms=101_000,
        rubik_min_rows=150,
    )
    calls: list[tuple[str | None, str | None]] = []

    async def fetch(symbol: str, begin: str | None, end: str | None) -> SourceResult:
        assert symbol == "BTC-USDT-SWAP"
        calls.append((begin, end))
        if len(calls) == 1:
            timestamps = list(range(1_000, 51_000, 1_000))
        elif len(calls) == 2:
            timestamps = list(range(-49_000, 1_000, 1_000))
        else:
            timestamps = []
        return SourceResult(
            pl.DataFrame(
                {
                    "timestamp": timestamps,
                    "open_interest": [float(i) for i in range(len(timestamps))],
                }
            ),
            manifest_frame([]),
        )

    out = asyncio.run(
        _collect_rubik_source(
            httpx.AsyncClient(),
            request,
            existing,
            frame_source="open_interest_history",
            artifact_name="source_open_interest",
            request_budget=asyncio.Semaphore(1),
            fetch=fetch,
        )
    ).frame.sort("timestamp")

    assert calls == [(None, "50999"), (None, "999")]
    assert out.height == 150
    assert out.get_column("timestamp").n_unique() == 150
    assert out.get_column("timestamp").min() == -49_000
    assert out.get_column("timestamp").max() == 100_000


def test_rubik_backfill_stops_on_empty_page() -> None:
    existing = pl.DataFrame(
        {
            "symbol": ["BTC-USDT-SWAP"] * 50,
            "timestamp": list(range(51_000, 101_000, 1_000)),
        }
    )
    request = SourceCollectRequest(
        output_dir=Path("data/output/test"),
        symbols=("BTC-USDT-SWAP",),
        discovery=pl.DataFrame(),
        concurrency=1,
        book_mode="off",
        book_depth=0,
        max_source_staleness_hours=24,
        trade_limit=50,
        funding_limit=50,
        rubik_period="1H",
        rubik_limit=50,
        rubik_taker_unit="2",
        disabled_sources=(),
        disabled_symbols=(),
        target_source_start_ms=-49_000,
        target_source_end_ms=101_000,
        rubik_min_rows=150,
    )
    calls: list[tuple[str | None, str | None]] = []

    async def fetch(symbol: str, begin: str | None, end: str | None) -> SourceResult:
        calls.append((begin, end))
        return SourceResult(pl.DataFrame(), manifest_frame([]))

    out = asyncio.run(
        _collect_rubik_source(
            httpx.AsyncClient(),
            request,
            existing,
            frame_source="open_interest_history",
            artifact_name="source_open_interest",
            request_budget=asyncio.Semaphore(1),
            fetch=fetch,
        )
    ).frame

    assert calls == [(None, "50999")]
    assert out.height == 50


def test_rubik_backfill_stops_on_repeated_earliest_page() -> None:
    existing = pl.DataFrame(
        {
            "symbol": ["BTC-USDT-SWAP"] * 50,
            "timestamp": list(range(51_000, 101_000, 1_000)),
        }
    )
    request = SourceCollectRequest(
        output_dir=Path("data/output/test"),
        symbols=("BTC-USDT-SWAP",),
        discovery=pl.DataFrame(),
        concurrency=1,
        book_mode="off",
        book_depth=0,
        max_source_staleness_hours=24,
        trade_limit=50,
        funding_limit=50,
        rubik_period="1H",
        rubik_limit=50,
        rubik_taker_unit="2",
        disabled_sources=(),
        disabled_symbols=(),
        target_source_start_ms=-49_000,
        target_source_end_ms=101_000,
        rubik_min_rows=150,
    )
    calls: list[tuple[str | None, str | None]] = []

    async def fetch(symbol: str, begin: str | None, end: str | None) -> SourceResult:
        calls.append((begin, end))
        return SourceResult(
            pl.DataFrame(
                {
                    "timestamp": list(range(51_000, 101_000, 1_000)),
                    "open_interest": [1.0] * 50,
                }
            ),
            manifest_frame([]),
        )

    out = asyncio.run(
        _collect_rubik_source(
            httpx.AsyncClient(),
            request,
            existing,
            frame_source="open_interest_history",
            artifact_name="source_open_interest",
            request_budget=asyncio.Semaphore(1),
            fetch=fetch,
        )
    ).frame

    assert calls == [(None, "50999")]
    assert out.select("symbol", "timestamp").unique().height == 50


def test_funding_backfill_fetches_multiple_pages_until_depth() -> None:
    existing = pl.DataFrame(
        {
            "symbol": ["BTC-USDT-SWAP"] * 50,
            "funding_time": list(range(51_000, 101_000, 1_000)),
            "funding_rate": [float(i) for i in range(50)],
        }
    )
    request = SourceCollectRequest(
        output_dir=Path("data/output/test"),
        symbols=("BTC-USDT-SWAP",),
        discovery=pl.DataFrame(),
        concurrency=1,
        book_mode="off",
        book_depth=0,
        max_source_staleness_hours=24,
        trade_limit=50,
        funding_limit=50,
        rubik_period="1H",
        rubik_limit=50,
        rubik_taker_unit="2",
        disabled_sources=(),
        disabled_symbols=(),
        target_source_start_ms=-49_000,
        target_source_end_ms=101_000,
        funding_min_rows=150,
    )
    calls: list[tuple[str | None, str | None]] = []

    async def fetch(symbol: str, after: str | None, before: str | None) -> SourceResult:
        assert symbol == "BTC-USDT-SWAP"
        calls.append((after, before))
        if len(calls) == 1:
            timestamps = list(range(1_000, 51_000, 1_000))
        elif len(calls) == 2:
            timestamps = list(range(-49_000, 1_000, 1_000))
        else:
            timestamps = []
        return SourceResult(
            pl.DataFrame(
                {
                    "funding_time": timestamps,
                    "funding_rate": [float(i) for i in range(len(timestamps))],
                }
            ),
            manifest_frame([]),
        )

    out = asyncio.run(
        _collect_funding_source(
            httpx.AsyncClient(),
            request,
            existing,
            request_budget=asyncio.Semaphore(1),
            fetch=fetch,
        )
    ).frame.sort("funding_time")

    assert calls == [("51000", None), ("1000", None)]
    assert out.height == 150
    assert out.get_column("funding_time").n_unique() == 150
    assert out.get_column("funding_time").min() == -49_000
    assert out.get_column("funding_time").max() == 100_000


def test_funding_backfill_stops_on_empty_page() -> None:
    existing = pl.DataFrame(
        {
            "symbol": ["BTC-USDT-SWAP"] * 50,
            "funding_time": list(range(51_000, 101_000, 1_000)),
        }
    )
    request = SourceCollectRequest(
        output_dir=Path("data/output/test"),
        symbols=("BTC-USDT-SWAP",),
        discovery=pl.DataFrame(),
        concurrency=1,
        book_mode="off",
        book_depth=0,
        max_source_staleness_hours=24,
        trade_limit=50,
        funding_limit=50,
        rubik_period="1H",
        rubik_limit=50,
        rubik_taker_unit="2",
        disabled_sources=(),
        disabled_symbols=(),
        target_source_start_ms=-49_000,
        target_source_end_ms=101_000,
        funding_min_rows=150,
    )
    calls: list[tuple[str | None, str | None]] = []

    async def fetch(symbol: str, after: str | None, before: str | None) -> SourceResult:
        calls.append((after, before))
        return SourceResult(pl.DataFrame(), manifest_frame([]))

    out = asyncio.run(
        _collect_funding_source(
            httpx.AsyncClient(),
            request,
            existing,
            request_budget=asyncio.Semaphore(1),
            fetch=fetch,
        )
    ).frame

    assert calls == [("51000", None)]
    assert out.height == 50


def test_funding_backfill_stops_on_repeated_earliest_page() -> None:
    existing = pl.DataFrame(
        {
            "symbol": ["BTC-USDT-SWAP"] * 50,
            "funding_time": list(range(51_000, 101_000, 1_000)),
        }
    )
    request = SourceCollectRequest(
        output_dir=Path("data/output/test"),
        symbols=("BTC-USDT-SWAP",),
        discovery=pl.DataFrame(),
        concurrency=1,
        book_mode="off",
        book_depth=0,
        max_source_staleness_hours=24,
        trade_limit=50,
        funding_limit=50,
        rubik_period="1H",
        rubik_limit=50,
        rubik_taker_unit="2",
        disabled_sources=(),
        disabled_symbols=(),
        target_source_start_ms=-49_000,
        target_source_end_ms=101_000,
        funding_min_rows=150,
    )
    calls: list[tuple[str | None, str | None]] = []

    async def fetch(symbol: str, after: str | None, before: str | None) -> SourceResult:
        calls.append((after, before))
        return SourceResult(
            pl.DataFrame(
                {
                    "funding_time": list(range(51_000, 101_000, 1_000)),
                    "funding_rate": [1.0] * 50,
                }
            ),
            manifest_frame([]),
        )

    out = asyncio.run(
        _collect_funding_source(
            httpx.AsyncClient(),
            request,
            existing,
            request_budget=asyncio.Semaphore(1),
            fetch=fetch,
        )
    ).frame

    assert calls == [("51000", None)]
    assert out.select("symbol", "funding_time").unique().height == 50


def test_funding_source_fetches_current_rate_when_history_is_deep() -> None:
    existing = pl.DataFrame(
        {
            "symbol": ["BTC-USDT-SWAP"] * 150,
            "timestamp": list(range(1_000, 151_000, 1_000)),
            "funding_time": list(range(1_000, 151_000, 1_000)),
            "funding_rate": [0.01] * 150,
        }
    )
    request = SourceCollectRequest(
        output_dir=Path("data/output/test"),
        symbols=("BTC-USDT-SWAP",),
        discovery=pl.DataFrame(),
        concurrency=1,
        book_mode="off",
        book_depth=0,
        max_source_staleness_hours=24,
        trade_limit=50,
        funding_limit=50,
        rubik_period="1H",
        rubik_limit=50,
        rubik_taker_unit="2",
        disabled_sources=(),
        disabled_symbols=(),
        target_source_start_ms=1_000,
        target_source_end_ms=200_000,
        funding_min_rows=150,
    )
    history_calls: list[tuple[str | None, str | None]] = []
    current_calls: list[str] = []

    async def fetch_history(symbol: str, after: str | None, before: str | None) -> SourceResult:
        history_calls.append((after, before))
        return SourceResult(pl.DataFrame(), manifest_frame([]))

    async def fetch_current(symbol: str) -> SourceResult:
        current_calls.append(symbol)
        return SourceResult(
            pl.DataFrame(
                {
                    "symbol": [symbol],
                    "timestamp": [200_000],
                    "funding_time": [208_800_000],
                    "funding_rate": [0.02],
                }
            ),
            manifest_frame(
                [
                    source_manifest_row(
                        symbol=symbol,
                        source="funding_rate",
                        phase="collect-market",
                        status="ok",
                        rows=1,
                        range_start=200_000,
                        range_end=200_000,
                    )
                ]
            ),
        )

    result = asyncio.run(
        _collect_funding_source(
            httpx.AsyncClient(),
            request,
            existing,
            request_budget=asyncio.Semaphore(1),
            fetch=fetch_history,
            current_fetch=fetch_current,
        )
    )

    assert history_calls == []
    assert current_calls == ["BTC-USDT-SWAP"]
    assert result.frame.filter(pl.col("timestamp") == 200_000).height == 1
    assert result.frame.get_column("timestamp").max() == 200_000


def test_merge_context_frames_migrates_cached_funding_rows_to_history_kind() -> None:
    empty = pl.DataFrame()
    bundle = SourceBundle(
        discovery=empty,
        bars=empty,
        books=empty,
        trades=empty,
        funding=pl.DataFrame(
            {
                "symbol": ["ACH-USDT-SWAP"],
                "timestamp": [1_000],
                "funding_time": [1_000],
                "funding_rate": [0.01],
            }
        ),
        open_interest=empty,
        taker_volume=empty,
        long_short_ratios=empty,
        onchain_flows=empty,
        messages=empty,
        polymarket_events=empty,
        polymarket_markets=empty,
        message_classifications=empty,
        manifest=manifest_frame([]),
    )

    out = merge_context_frames(bundle, {})["funding"]

    assert out.select("funding_source_kind", "known_at_ms").to_dicts() == [
        {"funding_source_kind": "history", "known_at_ms": 1_000}
    ]


def test_funding_source_migrates_cached_rows_to_history_kind() -> None:
    existing = pl.DataFrame(
        {
            "symbol": ["BTC-USDT-SWAP"],
            "timestamp": [1_000],
            "funding_time": [1_000],
            "funding_rate": [0.01],
        }
    )
    request = SourceCollectRequest(
        output_dir=Path("data/output/test"),
        symbols=("BTC-USDT-SWAP",),
        discovery=pl.DataFrame(),
        concurrency=1,
        book_mode="off",
        book_depth=0,
        max_source_staleness_hours=24,
        trade_limit=50,
        funding_limit=50,
        rubik_period="1H",
        rubik_limit=50,
        rubik_taker_unit="2",
        disabled_sources=(),
        disabled_symbols=(),
        target_source_start_ms=1_000,
        target_source_end_ms=1_000,
        funding_min_rows=1,
    )

    async def fetch_history(symbol: str, after: str | None, before: str | None) -> SourceResult:
        raise AssertionError("cached historical row should satisfy historical funding depth")

    async def fetch_current(symbol: str) -> SourceResult:
        return SourceResult(pl.DataFrame(), manifest_frame([]))

    result = asyncio.run(
        _collect_funding_source(
            httpx.AsyncClient(),
            request,
            existing,
            request_budget=asyncio.Semaphore(1),
            fetch=fetch_history,
            current_fetch=fetch_current,
        )
    )

    assert result.frame.select(
        "funding_source_kind", "known_at_ms", "next_funding_rate", "next_funding_time"
    ).to_dicts() == [
        {
            "funding_source_kind": "history",
            "known_at_ms": 1_000,
            "next_funding_rate": None,
            "next_funding_time": None,
        }
    ]


def test_funding_source_fetches_history_when_only_current_rows_exist() -> None:
    existing = pl.DataFrame(
        {
            "symbol": ["BTC-USDT-SWAP"],
            "timestamp": [200_000],
            "funding_time": [208_800_000],
            "funding_rate": [0.02],
            "funding_source_kind": ["current"],
            "known_at_ms": [200_000],
        }
    )
    request = SourceCollectRequest(
        output_dir=Path("data/output/test"),
        symbols=("BTC-USDT-SWAP",),
        discovery=pl.DataFrame(),
        concurrency=1,
        book_mode="off",
        book_depth=0,
        max_source_staleness_hours=24,
        trade_limit=50,
        funding_limit=50,
        rubik_period="1H",
        rubik_limit=50,
        rubik_taker_unit="2",
        disabled_sources=(),
        disabled_symbols=(),
        target_source_start_ms=1_000,
        target_source_end_ms=200_000,
        funding_min_rows=150,
    )
    history_calls: list[tuple[str | None, str | None]] = []

    async def fetch_history(symbol: str, after: str | None, before: str | None) -> SourceResult:
        history_calls.append((after, before))
        return SourceResult(
            pl.DataFrame(
                {
                    "symbol": [symbol],
                    "timestamp": [1_000],
                    "funding_time": [1_000],
                    "funding_rate": [0.01],
                    "funding_source_kind": ["history"],
                    "known_at_ms": [1_000],
                }
            ),
            manifest_frame([]),
        )

    async def fetch_current(symbol: str) -> SourceResult:
        return SourceResult(pl.DataFrame(), manifest_frame([]))

    asyncio.run(
        _collect_funding_source(
            httpx.AsyncClient(),
            request,
            existing,
            request_budget=asyncio.Semaphore(1),
            fetch=fetch_history,
            current_fetch=fetch_current,
        )
    )

    assert history_calls
    assert history_calls[0] == (None, None)


def test_funding_rate_manifest_maps_to_funding_family() -> None:
    manifest = manifest_frame(
        [
            source_manifest_row(
                symbol="BTC-USDT-SWAP",
                source="funding_rate",
                phase="collect-market",
                status="ok",
                rows=1,
                timestamp=2,
            )
        ]
    )

    status, warning = manifest_latest_maps(manifest)

    assert source_manifest_family("funding_rate") == "funding"
    assert status[("funding", "BTC-USDT-SWAP")] == "ok"
    assert warning[("funding", "BTC-USDT-SWAP")] == ""


def test_long_short_manifest_sources_map_to_long_short_ratios_family() -> None:
    assert source_manifest_family("long_short_ratio_contract") == "long_short_ratios"
    assert (
        source_manifest_family("top_trader_long_short_account_ratio_contract")
        == "long_short_ratios"
    )
    assert (
        source_manifest_family("top_trader_long_short_position_ratio_contract")
        == "long_short_ratios"
    )


def test_funding_rate_manifest_status_overrides_older_funding_history_status() -> None:
    manifest = manifest_frame(
        [
            source_manifest_row(
                symbol="BTC-USDT-SWAP",
                source="funding_rate",
                phase="collect-market",
                status="ok",
                rows=1,
                timestamp=2,
            ),
            source_manifest_row(
                symbol="BTC-USDT-SWAP",
                source="funding",
                phase="collect-market",
                status="missing",
                rows=100,
                warning="funding_missing",
                timestamp=1,
            ),
        ]
    )

    status, warning = manifest_latest_maps(manifest)

    assert status[("funding", "BTC-USDT-SWAP")] == "ok"
    assert warning[("funding", "BTC-USDT-SWAP")] == ""


def test_combine_source_results_accepts_different_column_order() -> None:
    local_frame = pl.DataFrame(
        {
            "symbol": ["BTC-USDT-SWAP"],
            "timestamp": [100],
            "open_interest": [10.0],
        }
    )
    fetched_frame = pl.DataFrame(
        {
            "timestamp": [50],
            "open_interest": [8.0],
            "symbol": ["BTC-USDT-SWAP"],
        }
    )

    out = _combine_source_results(
        [SourceResult(frame=fetched_frame, manifest=manifest_frame([]))],
        local_frame=local_frame,
        local_manifest=manifest_frame([]),
    ).frame.sort("timestamp")

    assert out.select("symbol", "timestamp", "open_interest").to_dicts() == [
        {"symbol": "BTC-USDT-SWAP", "timestamp": 50, "open_interest": 8.0},
        {"symbol": "BTC-USDT-SWAP", "timestamp": 100, "open_interest": 10.0},
    ]


def test_latest_manifest_status_uses_latest_symbol_source_row() -> None:
    manifest = manifest_frame(
        [
            source_manifest_row(
                timestamp=100,
                symbol="BTC-USDT-SWAP",
                source="trades",
                phase="collect-market",
                status="missing",
            ),
            source_manifest_row(
                timestamp=200,
                symbol="BTC-USDT-SWAP",
                source="trades",
                phase="collect-market",
                status="ok",
            ),
        ]
    )

    assert latest_manifest_status(manifest, source="trades", symbol="BTC-USDT-SWAP") == "ok"


def test_base_currency_contract_value_is_supported_for_swap_notional() -> None:
    out = normalize_okx_trades(
        [{"ts": "1", "tradeId": "a", "px": "100", "sz": "2", "side": "buy"}],
        contract_value=0.01,
        contract_value_currency="BTC",
        contract_base_currency="BTC",
    )

    assert out["notional_usd"][0] == 2.0


def test_unsupported_contract_currency_yields_null_notional() -> None:
    out = normalize_okx_trades(
        [{"ts": "1", "tradeId": "a", "px": "100", "sz": "2", "side": "buy"}],
        contract_value=0.01,
        contract_value_currency="BTC",
        contract_base_currency="ETH",
    )

    assert out["notional_usd"][0] is None


def test_swap_instrument_base_quote_are_inferred_when_okx_omits_them() -> None:
    out = _normalize_instruments(
        [
            {
                "instId": "PEPE-USDT-SWAP",
                "instType": "SWAP",
                "state": "live",
                "baseCcy": "",
                "quoteCcy": "",
                "settleCcy": "USDT",
                "ctVal": "10000000",
                "ctValCcy": "PEPE",
            }
        ]
    )

    assert out["base_ccy"][0] == "PEPE"
    assert out["quote_ccy"][0] == "USDT"


def test_missing_contract_value_yields_null_notional() -> None:
    out = normalize_okx_trades(
        [{"ts": "1", "tradeId": "a", "px": "100", "sz": "2", "side": "sell"}],
        contract_value=None,
    )

    assert out["notional_usd"][0] is None


def test_sanitized_http_error_omits_query_string_and_keys() -> None:
    request = httpx.Request("GET", "https://www.okx.com/api/v5/x?apiKey=secret&instId=BTC")
    response = httpx.Response(500, request=request)
    exc = httpx.HTTPStatusError("boom", request=request, response=response)

    out = sanitize_error(exc)

    assert "apiKey" not in out
    assert "secret" not in out
    assert out.startswith("500")


def test_request_json_raises_sanitized_source_http_error(monkeypatch) -> None:
    def fake_get(url: str, **_kwargs) -> httpx.Response:
        request = httpx.Request("GET", f"{url}?apikey=secret")
        return httpx.Response(500, request=request)

    monkeypatch.setattr("qooi.sources.http.httpx.get", fake_get)

    with pytest.raises(SourceHttpError) as exc_info:
        request_json_sync("https://example.test/api", params={"apikey": "secret"})

    assert exc_info.value.category == "transport_error"
    assert exc_info.value.message == "500 Internal Server Error"
    assert exc_info.value.endpoint == "/api"
    assert "secret" not in exc_info.value.message


def test_request_json_uses_provider_classifier(monkeypatch) -> None:
    def fake_get(_url: str, **_kwargs) -> httpx.Response:
        return httpx.Response(
            200,
            request=httpx.Request("GET", "https://example.test/api"),
            json={"status": "0", "message": "NOTOK", "result": "Max rate limit reached"},
        )

    monkeypatch.setattr("qooi.sources.http.httpx.get", fake_get)

    with pytest.raises(SourceHttpError) as exc_info:
        request_json_sync(
            "https://example.test/api",
            error_classifier=lambda _payload: "rate_limited",
        )

    assert exc_info.value.category == "rate_limited"
    assert "Max rate limit reached" in exc_info.value.message


def test_request_json_allows_configured_empty_message(monkeypatch) -> None:
    def fake_get(_url: str, **_kwargs) -> httpx.Response:
        return httpx.Response(
            200,
            request=httpx.Request("GET", "https://example.test/api"),
            json={"status": "0", "message": "No transactions found", "result": []},
        )

    monkeypatch.setattr("qooi.sources.http.httpx.get", fake_get)

    payload = request_json_sync(
        "https://example.test/api",
        error_classifier=lambda _payload: "bad_request",
        allow_empty_message="No transactions found",
    )

    assert payload["result"] == []


@pytest.mark.asyncio
async def test_request_json_uses_existing_client() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api"
        return httpx.Response(200, json={"status": "1", "result": "ok"})

    async with httpx.AsyncClient(
        base_url="https://example.test", transport=httpx.MockTransport(handler)
    ) as client:
        payload = await request_json(client, "/api", params={"q": "x"})

    assert payload["result"] == "ok"


@pytest.mark.asyncio
async def test_okx_source_uses_retry_iteration() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": "0", "data": [{"ts": "1"}]})

    async with httpx.AsyncClient(
        base_url="https://www.okx.com", transport=httpx.MockTransport(handler)
    ) as client:
        result = await _fetch_okx_frame(
            client,
            endpoint="/test",
            params={},
            source="test",
            symbol="BTC-USDT-SWAP",
            normalizer=lambda rows: pl.DataFrame([{"timestamp": int(rows[0]["ts"])}]),
        )

    assert result.frame["timestamp"][0] == 1
    assert result.manifest["status"][0] == "ok"


@pytest.mark.asyncio
async def test_okx_current_funding_rate_normalizes_snapshot() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v5/public/funding-rate"
        assert request.url.params["instId"] == "BTC-USDT-SWAP"
        return httpx.Response(
            200,
            json={
                "code": "0",
                "data": [
                    {
                        "instId": "BTC-USDT-SWAP",
                        "ts": "1000",
                        "fundingRate": "-0.0002",
                        "nextFundingRate": "-0.0001",
                        "fundingTime": "2000",
                        "nextFundingTime": "3000",
                    }
                ],
            },
        )

    async with httpx.AsyncClient(
        base_url="https://www.okx.com", transport=httpx.MockTransport(handler)
    ) as client:
        result = await fetch_okx_funding_rate(client, "BTC-USDT-SWAP")

    assert result.frame.to_dicts() == [
        {
            "symbol": "BTC-USDT-SWAP",
            "timestamp": 1000,
            "funding_rate": -0.0002,
            "next_funding_rate": -0.0001,
            "funding_time": 2000,
            "next_funding_time": 3000,
            "funding_source_kind": "current",
            "known_at_ms": 1000,
        }
    ]
    assert result.manifest["source"][0] == "funding_rate"
    assert result.manifest["status"][0] == "ok"


@pytest.mark.asyncio
async def test_coingecko_trending_empty_response_writes_missing_manifest() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/search/trending")
        return httpx.Response(200, json={"coins": []})

    async with httpx.AsyncClient(
        base_url="https://api.coingecko.com/api/v3", transport=httpx.MockTransport(handler)
    ) as client:
        result = await fetch_coingecko_trending(client)

    assert result.frame.is_empty()
    assert result.manifest["source"][0] == "coingecko_trending"
    assert result.manifest["status"][0] == "missing"
    assert result.manifest["warning"][0] == "coingecko_trending_empty"


@pytest.mark.asyncio
async def test_coingecko_trending_failed_request_writes_failed_manifest() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    async with httpx.AsyncClient(
        base_url="https://api.coingecko.com/api/v3", transport=httpx.MockTransport(handler)
    ) as client:
        result = await fetch_coingecko_trending(client)

    assert result.frame.is_empty()
    assert result.manifest["source"][0] == "coingecko_trending"
    assert result.manifest["status"][0] == "failed"
    assert result.manifest["warning"][0] == "500 Internal Server Error"


def test_okx_rubik_open_interest_history_normalizer() -> None:
    out = _normalize_open_interest_history([["3600000", "100", "10", "2500000"]])

    assert out["timestamp"][0] == 3_600_000
    assert out["open_interest"][0] == 100.0
    assert out["open_interest_ccy"][0] == 10.0
    assert out["open_interest_usd"][0] == 2_500_000.0


def test_okx_rubik_taker_volume_contract_normalizer() -> None:
    out = _normalize_taker_volume_contract([["3600000", "40", "60"]], "2")

    assert out["timestamp"][0] == 3_600_000
    assert out["taker_sell_volume"][0] == 40.0
    assert out["taker_buy_volume"][0] == 60.0
    assert out["taker_volume_unit"][0] == "2"


def test_okx_rubik_ratio_normalizer() -> None:
    out = _normalize_ratio_rows([["3600000", "1.25"]], "long_short_account_ratio")

    assert out["timestamp"][0] == 3_600_000
    assert out["long_short_account_ratio"][0] == 1.25


def test_okx_ws_subscribe_message_builds_public_channel_args() -> None:
    message = okx_ws_subscribe_message(("BTC-USDT-SWAP",), channels=("trades", "books5"))

    assert message == {
        "op": "subscribe",
        "args": [
            {"channel": "trades", "instId": "BTC-USDT-SWAP"},
            {"channel": "books5", "instId": "BTC-USDT-SWAP"},
        ],
    }


def test_okx_ws_trade_normalizer_maps_public_trade_rows() -> None:
    frame = normalize_okx_ws_trades(
        {
            "arg": {"channel": "trades", "instId": "BTC-USDT-SWAP"},
            "data": [{"ts": "1000", "tradeId": "1", "px": "100", "sz": "2", "side": "buy"}],
        },
        contract_values={"BTC-USDT-SWAP": 0.01},
    )

    assert frame.to_dicts() == [
        {
            "symbol": "BTC-USDT-SWAP",
            "timestamp": 1000,
            "trade_id": "1",
            "price": 100.0,
            "size": 2.0,
            "side": "buy",
            "notional_usd": 2.0,
        }
    ]


def test_okx_ws_book_normalizer_maps_depth_distribution_rows() -> None:
    frame = normalize_okx_ws_books(
        {
            "arg": {"channel": "books5", "instId": "BTC-USDT-SWAP"},
            "data": [
                {
                    "ts": "1000",
                    "bids": [["99", "2"], ["98.9", "1"]],
                    "asks": [["100", "1"], ["100.1", "1"]],
                }
            ],
        },
        contract_values={"BTC-USDT-SWAP": 0.01},
    )

    row = frame.to_dicts()[0]
    assert row["symbol"] == "BTC-USDT-SWAP"
    assert row["timestamp"] == 1000
    assert row["ob_bid_price"] == 99.0
    assert row["ob_ask_price"] == 100.0
    assert row["ob_bid_vol_5"] > row["ob_ask_vol_5"]
    assert row["ob_imbalance_5"] > 0.0
    assert row["spread_bps"] > 0.0


@pytest.mark.asyncio
async def test_okx_ws_collector_uses_supplied_message_stream_and_manifest() -> None:
    async def messages():
        yield {
            "arg": {"channel": "trades", "instId": "BTC-USDT-SWAP"},
            "data": [{"ts": str(10**13), "tradeId": "1", "px": "100", "sz": "2", "side": "buy"}],
        }
        yield {
            "arg": {"channel": "books5", "instId": "BTC-USDT-SWAP"},
            "data": [
                {
                    "ts": str(10**13),
                    "bids": [["99", "2"]],
                    "asks": [["100", "1"]],
                }
            ],
        }

    result = await collect_okx_ws_public(
        ("BTC-USDT-SWAP",),
        message_source=messages(),
        contract_values={"BTC-USDT-SWAP": 0.01},
        stale_after_ms=10**15,
    )

    assert result.trades.height == 1
    assert result.books.height == 1
    statuses = result.manifest.select("source", "status").to_dicts()
    assert statuses == [
        {"source": "okx_ws_trades", "status": "ok"},
        {"source": "okx_ws_books", "status": "ok"},
    ]


@pytest.mark.asyncio
async def test_okx_ws_collector_reports_stale_and_errors() -> None:
    async def messages():
        yield {"event": "error", "msg": "channel rejected"}
        yield {
            "arg": {"channel": "trades", "instId": "BTC-USDT-SWAP"},
            "data": [{"ts": "1", "tradeId": "1", "px": "100", "sz": "2", "side": "buy"}],
        }

    result = await collect_okx_ws_public(
        ("BTC-USDT-SWAP",), message_source=messages(), stale_after_ms=1
    )

    trades = result.manifest.filter(pl.col("source") == "okx_ws_trades")
    books = result.manifest.filter(pl.col("source") == "okx_ws_books")
    assert trades["status"][0] == "partial"
    assert "channel rejected" in trades["warning"][0]
    assert "okx_ws_stale" in trades["warning"][0]
    assert books["status"][0] == "failed"
