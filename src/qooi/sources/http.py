"""Shared HTTP helpers for read-only source collectors."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx

from qooi.sources.models import SourceResult

SourceHttpStatus = Literal[
    "ok",
    "empty",
    "api_error",
    "bad_request",
    "rate_limited",
    "timeout_or_too_broad",
    "transport_error",
]


@dataclass(frozen=True)
class SourceHttpError(RuntimeError):
    category: SourceHttpStatus
    message: str
    http_status: int | None = None
    endpoint: str = ""

    def __post_init__(self) -> None:
        RuntimeError.__init__(self, self.message)


def sanitize_error(exc: BaseException) -> str:
    if isinstance(exc, SourceHttpError):
        return exc.message
    if isinstance(exc, httpx.HTTPStatusError):
        return f"{exc.response.status_code} {exc.response.reason_phrase}"
    return type(exc).__name__


def sanitized_provider_message(payload: dict[str, Any], *, limit: int = 160) -> str:
    parts = []
    for key in ("message", "result"):
        value = payload.get(key)
        if isinstance(value, (str, int, float, bool)):
            text = str(value).strip()
            if text:
                parts.append(text)
    out = ": ".join(parts) or "provider_error"
    return out[:limit]


def request_json_sync(
    url: str,
    *,
    params: dict[str, str] | None = None,
    timeout: float = 20.0,
    error_classifier: Callable[[dict[str, Any]], SourceHttpStatus | None] | None = None,
    allow_empty_message: str = "",
) -> dict[str, Any]:
    endpoint = _safe_endpoint(url)
    try:
        response = httpx.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as exc:
        raise SourceHttpError(
            _http_status_category(exc.response.status_code),
            sanitize_error(exc),
            http_status=exc.response.status_code,
            endpoint=endpoint,
        ) from None
    except httpx.TimeoutException as exc:
        raise SourceHttpError(
            "timeout_or_too_broad", sanitize_error(exc), endpoint=endpoint
        ) from None
    except httpx.TransportError as exc:
        raise SourceHttpError("transport_error", sanitize_error(exc), endpoint=endpoint) from None
    return _validate_json_payload(
        payload,
        endpoint=endpoint,
        error_classifier=error_classifier,
        allow_empty_message=allow_empty_message,
    )


async def request_json(
    client: httpx.AsyncClient,
    endpoint: str,
    *,
    params: dict[str, str] | None = None,
    error_classifier: Callable[[dict[str, Any]], SourceHttpStatus | None] | None = None,
    allow_empty_message: str = "",
) -> dict[str, Any]:
    try:
        response = await client.get(endpoint, params=params)
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as exc:
        raise SourceHttpError(
            _http_status_category(exc.response.status_code),
            sanitize_error(exc),
            http_status=exc.response.status_code,
            endpoint=endpoint,
        ) from None
    except httpx.TimeoutException as exc:
        raise SourceHttpError(
            "timeout_or_too_broad", sanitize_error(exc), endpoint=endpoint
        ) from None
    except httpx.TransportError as exc:
        raise SourceHttpError("transport_error", sanitize_error(exc), endpoint=endpoint) from None
    return _validate_json_payload(
        payload,
        endpoint=endpoint,
        error_classifier=error_classifier,
        allow_empty_message=allow_empty_message,
    )


async def request_json_value(
    client: httpx.AsyncClient,
    endpoint: str,
    *,
    params: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
) -> Any:
    try:
        response = await client.get(endpoint, params=params, headers=headers)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        raise SourceHttpError(
            _http_status_category(exc.response.status_code),
            sanitize_error(exc),
            http_status=exc.response.status_code,
            endpoint=endpoint,
        ) from None
    except httpx.TimeoutException as exc:
        raise SourceHttpError(
            "timeout_or_too_broad", sanitize_error(exc), endpoint=endpoint
        ) from None
    except httpx.TransportError as exc:
        raise SourceHttpError("transport_error", sanitize_error(exc), endpoint=endpoint) from None


async def gather_source_results(
    calls: list[Callable[[httpx.AsyncClient], Awaitable[SourceResult]]],
    *,
    base_url: str,
    concurrency: int = 3,
    timeout: float = 20.0,
) -> list[SourceResult]:
    semaphore = asyncio.Semaphore(max(1, concurrency))
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:

        async def run(call: Callable[[httpx.AsyncClient], Awaitable[SourceResult]]) -> SourceResult:
            async with semaphore:
                return await call(client)

        return await asyncio.gather(*(run(call) for call in calls))


def _validate_json_payload(
    payload: Any,
    *,
    endpoint: str,
    error_classifier: Callable[[dict[str, Any]], SourceHttpStatus | None] | None,
    allow_empty_message: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SourceHttpError("api_error", "JSON response was not an object", endpoint=endpoint)
    if allow_empty_message and str(payload.get("message", "")) == allow_empty_message:
        return payload
    if error_classifier is not None:
        category = error_classifier(payload)
        if category and category not in {"ok", "empty"}:
            raise SourceHttpError(
                category,
                sanitized_provider_message(payload),
                endpoint=endpoint,
            )
    return payload


def _http_status_category(status_code: int) -> SourceHttpStatus:
    if status_code == 429:
        return "rate_limited"
    if 400 <= status_code < 500:
        return "bad_request"
    return "transport_error"


def _safe_endpoint(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme and parsed.netloc:
        return parsed.path or "/"
    return url.split("?", 1)[0]

