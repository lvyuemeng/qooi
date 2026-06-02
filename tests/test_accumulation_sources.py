from __future__ import annotations

from pathlib import Path

import httpx
import polars as pl
import pytest

from qooi.sources.http import SourceHttpError, request_json, request_json_async, sanitize_error
from qooi.sources.okx import (
    _fetch_okx_frame,
    _normalize_instruments,
    normalize_okx_trades,
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
        "qooi.accumulation",
        "qooi.core.executor",
        "qooi.core.basket",
        "qooi.core.recovery",
        "qooi.exchange.trading",
    )
    for path in Path(source_root).glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden), path


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
        request_json("https://example.test/api", params={"apikey": "secret"})

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
        request_json(
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

    payload = request_json(
        "https://example.test/api",
        error_classifier=lambda _payload: "bad_request",
        allow_empty_message="No transactions found",
    )

    assert payload["result"] == []


@pytest.mark.asyncio
async def test_request_json_async_uses_existing_client() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api"
        return httpx.Response(200, json={"status": "1", "result": "ok"})

    async with httpx.AsyncClient(
        base_url="https://example.test", transport=httpx.MockTransport(handler)
    ) as client:
        payload = await request_json_async(client, "/api", params={"q": "x"})

    assert payload["result"] == "ok"


@pytest.mark.asyncio
async def test_async_okx_fetch_uses_async_retry_iteration() -> None:
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
