"""Transport core — HTTP client, errors, retry, request gathering."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal, Self, TypeVar
from urllib.parse import urlsplit

import httpx
from tenacity import retry_if_exception, retry_if_exception_type, stop_after_attempt
from tenacity.wait import wait_exponential_jitter

# ── client ──


class BaseHttpClient:
    """Context-managed httpx.AsyncClient. trust_env=False by default."""

    def __init__(self, base_url: str, *, timeout: float = 20.0, proxy: str | None = None) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.proxy = proxy
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> Self:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            trust_env=self.proxy is None,
            proxy=self.proxy,
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("BaseHttpClient not entered")
        return self._client


# ── errors ──


HttpStatusCategory = Literal[
    "ok",
    "empty",
    "api_error",
    "bad_request",
    "rate_limited",
    "timeout_or_too_broad",
    "transport_error",
]


@dataclass(frozen=True)
class HttpError(RuntimeError):
    category: HttpStatusCategory
    message: str
    http_status: int | None = None
    endpoint: str = ""

    def __post_init__(self) -> None:
        RuntimeError.__init__(self, self.message)


def sanitize_error(exc: BaseException) -> str:
    if isinstance(exc, HttpError):
        return exc.message
    if isinstance(exc, httpx.HTTPStatusError):
        return f"{exc.response.status_code} {exc.response.reason_phrase}"
    return type(exc).__name__


def sanitized_provider_message(payload: dict[str, Any], *, limit: int = 160) -> str:
    parts = []
    for key in ("message", "msg", "result", "code"):
        value = payload.get(key)
        if isinstance(value, str | int | float | bool):
            text = str(value).strip()
            if text:
                parts.append(text)
    out = ": ".join(parts) or "provider_error"
    return out[:limit]


def _http_status_category(status_code: int) -> HttpStatusCategory:
    if status_code == 429:
        return "rate_limited"
    if 400 <= status_code < 500:
        return "bad_request"
    return "transport_error"


def _response_json(response: httpx.Response, _endpoint: str) -> Any:
    response.raise_for_status()
    return response.json()


def _http_error(exc: httpx.HTTPError, endpoint: str) -> HttpError:
    if isinstance(exc, httpx.HTTPStatusError):
        return HttpError(
            _http_status_category(exc.response.status_code),
            sanitize_error(exc),
            http_status=exc.response.status_code,
            endpoint=endpoint,
        )
    if isinstance(exc, httpx.TimeoutException):
        return HttpError("timeout_or_too_broad", sanitize_error(exc), endpoint=endpoint)
    return HttpError("transport_error", sanitize_error(exc), endpoint=endpoint)


def _validate_json_payload(
    payload: Any,
    *,
    endpoint: str,
    error_classifier: Callable[[dict[str, Any]], HttpStatusCategory | None] | None,
    allow_empty_message: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HttpError("api_error", "JSON response was not an object", endpoint=endpoint)
    if allow_empty_message and str(payload.get("message", "")) == allow_empty_message:
        return payload
    if error_classifier is not None:
        category = error_classifier(payload)
        if category and category not in {"ok", "empty"}:
            raise HttpError(category, sanitized_provider_message(payload), endpoint=endpoint)
    return payload


def _safe_endpoint(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme and parsed.netloc:
        return parsed.path or "/"
    return url.split("?", 1)[0]


def request_json_sync(
    url: str,
    *,
    params: dict[str, str] | None = None,
    timeout: float = 20.0,
    error_classifier: Callable[[dict[str, Any]], HttpStatusCategory | None] | None = None,
    allow_empty_message: str = "",
) -> dict[str, Any]:
    endpoint = _safe_endpoint(url)
    try:
        payload = _response_json(httpx.get(url, params=params, timeout=timeout), endpoint)
    except httpx.HTTPError as exc:
        raise _http_error(exc, endpoint) from None
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
    error_classifier: Callable[[dict[str, Any]], HttpStatusCategory | None] | None = None,
    allow_empty_message: str = "",
) -> dict[str, Any]:
    try:
        payload = _response_json(await client.get(endpoint, params=params), endpoint)
    except httpx.HTTPError as exc:
        raise _http_error(exc, endpoint) from None
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
        return _response_json(await client.get(endpoint, params=params, headers=headers), endpoint)
    except httpx.HTTPError as exc:
        raise _http_error(exc, endpoint) from None


# ── retry ──
# ── retry ──


@dataclass(frozen=True)
class RetryPolicy:
    stop_attempts: int = 5
    wait_initial: float = 0.5
    wait_max: float = 8.0
    retry_on: Callable[[BaseException], bool] | None = None
    reraise: bool = True

    def to_kwargs(self) -> dict[str, Any]:
        return {
            "stop": stop_after_attempt(self.stop_attempts),
            "wait": wait_exponential_jitter(initial=self.wait_initial, max=self.wait_max),
            "retry": retry_if_exception(self.retry_on)
            if self.retry_on is not None
            else retry_if_exception_type(
                (httpx.TimeoutException, httpx.TransportError, ConnectionError)
            ),
            "reraise": self.reraise,
        }


# ── gather ──


T = TypeVar("T")


async def gather_requests(
    calls: list[Callable[[httpx.AsyncClient], Awaitable[T]]],
    *,
    base_url: str,
    concurrency: int = 3,
    timeout: float = 20.0,
) -> list[T]:
    semaphore = asyncio.Semaphore(max(1, concurrency))
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout, trust_env=False) as client:

        async def run(call: Callable[[httpx.AsyncClient], Awaitable[T]]) -> T:
            async with semaphore:
                return await call(client)

        return await asyncio.gather(*(run(call) for call in calls))
