import asyncio

import pytest

from qooi.transport.okx import OkxClient


def test_instruments_empty_response_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    okx = OkxClient()

    async def empty_response(
        endpoint: str,
        params: dict[str, str] | None = None,
        *,
        error_classifier: object | None = None,
    ) -> dict[str, object]:
        return {"code": "0", "data": []}

    monkeypatch.setattr(okx, "request", empty_response)

    with pytest.raises(ValueError, match="required source discovery returned 0 rows"):
        asyncio.run(okx.instruments())


def test_tickers_empty_response_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    okx = OkxClient()

    async def empty_response(
        endpoint: str,
        params: dict[str, str] | None = None,
        *,
        error_classifier: object | None = None,
    ) -> dict[str, object]:
        return {"code": "0", "data": []}

    monkeypatch.setattr(okx, "request", empty_response)

    with pytest.raises(ValueError, match="required source discovery returned 0 rows"):
        asyncio.run(okx.tickers())
