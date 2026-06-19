from __future__ import annotations

from tenacity import RetryCallState, Retrying

from qooi.transport.core import HttpError, RetryPolicy


def _state_for(exc: BaseException) -> RetryCallState:
    retrying = Retrying()
    state = RetryCallState(retrying, None, (), {})
    state.set_exception((type(exc), exc, exc.__traceback__))
    return state


def test_retry_policy_uses_custom_predicate_for_rate_limits() -> None:
    policy = RetryPolicy(
        retry_on=lambda exc: isinstance(exc, HttpError) and exc.category == "rate_limited"
    )

    retry = policy.to_kwargs()["retry"]

    assert retry(_state_for(HttpError("rate_limited", "429 Too Many Requests")))
    assert not retry(_state_for(HttpError("bad_request", "400 Bad Request")))
