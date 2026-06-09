from __future__ import annotations

from pathlib import Path

import httpx
import polars as pl
import pytest

from qooi.sources.bundle import (
    latest_timestamp,
    missing_symbols,
    replace_symbol_rows,
    source_symbols,
)
from qooi.sources.coingecko import fetch_coingecko_trending
from qooi.sources.coverage import eligible_fetch_symbols, latest_manifest_status, stale_symbols
from qooi.sources.http import SourceHttpError, request_json, request_json_sync, sanitize_error
from qooi.sources.manifest import manifest_frame, source_manifest_row
from qooi.sources.okx import (
    _fetch_okx_frame,
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
    assert eligible_fetch_symbols(
        frame, symbols, now_ms=10_000, max_age_ms=5_000, refresh=True
    ) == symbols


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


