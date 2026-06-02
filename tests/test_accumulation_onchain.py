from __future__ import annotations

from typing import Any

import polars as pl
import pytest
from pydantic import ValidationError

from qooi.accumulation.config import AccumulationConfig
from qooi.accumulation.onchain import (
    ETHERSCAN_API_KEY_ENV,
    ExplorerApiError,
    ExplorerClient,
    classify_explorer_error,
)


def test_ethereum_and_bsc_clients_read_same_etherscan_key(monkeypatch) -> None:
    monkeypatch.setenv(ETHERSCAN_API_KEY_ENV, "test-key")

    assert ExplorerClient("ethereum").api_key == "test-key"
    assert ExplorerClient("bsc").api_key == "test-key"


def test_missing_etherscan_key_is_sanitized_for_all_chains(monkeypatch) -> None:
    monkeypatch.delenv(ETHERSCAN_API_KEY_ENV, raising=False)

    for chain in ("ethereum", "bsc"):
        with pytest.raises(ExplorerApiError) as exc_info:
            _ = ExplorerClient(chain).api_key
        assert exc_info.value.chain == chain
        assert exc_info.value.category == "bad_request"
        assert ETHERSCAN_API_KEY_ENV in exc_info.value.message


def test_request_params_include_chainid_for_each_chain(monkeypatch) -> None:
    monkeypatch.setenv(ETHERSCAN_API_KEY_ENV, "test-key")
    seen: list[dict[str, str]] = []

    def fake_request_json(_url: str, **kwargs: Any) -> dict[str, Any]:
        seen.append(kwargs["params"])
        return {"status": "1", "result": "0x10"}

    monkeypatch.setattr("qooi.accumulation.onchain.request_json", fake_request_json)

    assert ExplorerClient("ethereum", requests_per_second=0).latest_block() == 16
    assert ExplorerClient("bsc", requests_per_second=0).latest_block() == 16

    assert seen[0]["chainid"] == "1"
    assert seen[1]["chainid"] == "56"
    assert seen[0]["apikey"] == "test-key"
    assert seen[1]["apikey"] == "test-key"


def test_latest_block_parses_hex_result(monkeypatch) -> None:
    monkeypatch.setenv(ETHERSCAN_API_KEY_ENV, "test-key")
    monkeypatch.setattr(
        "qooi.accumulation.onchain.request_json",
        lambda *_args, **_kwargs: {"status": "1", "result": "0x2a"},
    )

    assert ExplorerClient("ethereum", requests_per_second=0).latest_block() == 42


def test_block_number_by_time_parses_decimal_result(monkeypatch) -> None:
    monkeypatch.setenv(ETHERSCAN_API_KEY_ENV, "test-key")
    monkeypatch.setattr(
        "qooi.accumulation.onchain.request_json",
        lambda *_args, **_kwargs: {"status": "1", "result": "12345"},
    )

    assert (
        ExplorerClient("ethereum", requests_per_second=0).block_number_by_time(1_700_000_000)
        == 12345
    )


def test_token_transfers_allows_no_transactions_found(monkeypatch) -> None:
    monkeypatch.setenv(ETHERSCAN_API_KEY_ENV, "test-key")

    def fake_request_json(_url: str, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["allow_empty_message"] == "No transactions found"
        return {"status": "0", "message": "No transactions found", "result": []}

    monkeypatch.setattr("qooi.accumulation.onchain.request_json", fake_request_json)

    frame = ExplorerClient("ethereum", requests_per_second=0).token_transfers(
        "0xwallet", "0xtoken", 1, 2
    )

    assert isinstance(frame, pl.DataFrame)
    assert frame.is_empty()


@pytest.mark.parametrize(
    ("message", "result", "category"),
    [
        ("NOTOK", "Free API access is not supported for this chain", "unsupported_chain_plan"),
        ("NOTOK", "Invalid API Key", "invalid_key"),
        ("NOTOK", "Max rate limit reached", "rate_limited"),
        ("NOTOK", "Query Timeout occured", "timeout_or_too_broad"),
        ("NOTOK", "Missing or unsupported chainid", "bad_request"),
        ("NOTOK", "Missing Or invalid Action", "bad_request"),
        ("NOTOK", "Invalid address format", "bad_request"),
        ("No transactions found", "", "empty"),
    ],
)
def test_explorer_error_classifier_maps_documented_messages(
    message: str, result: str, category: str
) -> None:
    assert classify_explorer_error(message, result) == category


def test_config_rejects_bscscan_provider() -> None:
    with pytest.raises(ValidationError):
        AccumulationConfig.model_validate({"onchain": {"provider": "bscscan"}})
