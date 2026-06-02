"""Optional explorer clients and exchange-flow classification."""

from __future__ import annotations

import os
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import polars as pl

from qooi.sources.http import SourceHttpError, SourceHttpStatus, request_json

ChainName = Literal["ethereum", "bsc"]
ExplorerStatus = Literal[
    "ok",
    "empty",
    "unsupported_chain_plan",
    "invalid_key",
    "rate_limited",
    "timeout_or_too_broad",
    "bad_request",
    "transport_error",
]

ETHERSCAN_V2_BASE_URL = "https://api.etherscan.io/v2/api"
ETHERSCAN_API_KEY_ENV = "ETHERSCAN_API_KEY"

EXPLORER_CHAIN_ID: dict[ChainName, str] = {
    "ethereum": "1",
    "bsc": "56",
}


def explorer_base_url(chain: ChainName) -> str:
    if chain not in EXPLORER_CHAIN_ID:
        raise ValueError(f"unsupported chain: {chain}")
    return ETHERSCAN_V2_BASE_URL


@dataclass(frozen=True)
class ExplorerApiError(RuntimeError):
    chain: ChainName
    category: ExplorerStatus
    message: str
    http_status: int | None = None

    def __post_init__(self) -> None:
        RuntimeError.__init__(self, self.message)


@dataclass(frozen=True)
class ExplorerProbe:
    chain: ChainName
    chainid: str
    status: ExplorerStatus
    latest_block: int | None = None
    message: str = ""


@dataclass(frozen=True)
class ExplorerClient:
    chain: ChainName
    requests_per_second: float = 4.0
    base_url: str | None = None

    @property
    def api_key(self) -> str:
        value = os.environ.get(ETHERSCAN_API_KEY_ENV, "").strip()
        if not value:
            raise ExplorerApiError(
                self.chain,
                "bad_request",
                f"Missing explorer API key in environment variable {ETHERSCAN_API_KEY_ENV}",
            )
        return value

    def latest_block(self) -> int:
        payload = self._request({"module": "proxy", "action": "eth_blockNumber"})
        value = parse_hex_int(payload.get("result"))
        if value is None:
            raise ExplorerApiError(self.chain, "bad_request", "invalid latest block result")
        return value

    def block_number_by_time(
        self,
        timestamp_seconds: int,
        *,
        closest: Literal["before", "after"] = "before",
    ) -> int:
        payload = self._request(
            {
                "module": "block",
                "action": "getblocknobytime",
                "timestamp": str(timestamp_seconds),
                "closest": closest,
            }
        )
        value = parse_decimal_int(payload.get("result"))
        if value is None:
            raise ExplorerApiError(self.chain, "bad_request", "invalid block number result")
        return value

    def token_transfers(
        self,
        address: str,
        contract: str,
        start_block: int,
        end_block: int,
        *,
        page: int = 1,
        offset: int = 100,
        sort: Literal["asc", "desc"] = "asc",
    ) -> pl.DataFrame:
        payload = self._request(
            {
                "module": "account",
                "action": "tokentx",
                "address": address,
                "contractaddress": contract,
                "startblock": str(start_block),
                "endblock": str(end_block),
                "page": str(max(1, page)),
                "offset": str(max(1, min(offset, 1000))),
                "sort": sort,
            },
            allow_empty_message="No transactions found",
        )
        rows = payload.get("result", [])
        return pl.DataFrame(rows) if rows else pl.DataFrame()

    def token_balance(self, contract: str, address: str, *, tag: str = "latest") -> int | None:
        payload = self._request(
            {
                "module": "account",
                "action": "tokenbalance",
                "contractaddress": contract,
                "address": address,
                "tag": tag,
            }
        )
        return parse_decimal_int(payload.get("result"))

    def token_supply(self, contract: str) -> int | None:
        payload = self._request(
            {"module": "stats", "action": "tokensupply", "contractaddress": contract}
        )
        return parse_decimal_int(payload.get("result"))

    def event_logs(
        self,
        address: str,
        from_block: int,
        to_block: int,
        *,
        topics: tuple[str, ...] = (),
        page: int = 1,
        offset: int = 100,
    ) -> pl.DataFrame:
        params = {
            "module": "logs",
            "action": "getLogs",
            "address": address,
            "fromBlock": str(from_block),
            "toBlock": str(to_block),
            "page": str(max(1, page)),
            "offset": str(max(1, min(offset, 1000))),
        }
        for index, topic in enumerate(topics):
            params[f"topic{index}"] = topic
        payload = self._request(params, allow_empty_message="No records found")
        rows = payload.get("result", [])
        return pl.DataFrame(rows) if rows else pl.DataFrame()

    def probe(self) -> ExplorerProbe:
        try:
            latest_block = self.latest_block()
        except ExplorerApiError as exc:
            return ExplorerProbe(
                self.chain,
                EXPLORER_CHAIN_ID[self.chain],
                exc.category,
                message=exc.message,
            )
        return ExplorerProbe(
            self.chain,
            EXPLORER_CHAIN_ID[self.chain],
            "ok",
            latest_block=latest_block,
        )

    def _request(
        self, params: dict[str, str], *, allow_empty_message: str = ""
    ) -> dict[str, Any]:
        if self.requests_per_second > 0:
            time.sleep(1.0 / self.requests_per_second)
        request_params = {
            "chainid": EXPLORER_CHAIN_ID[self.chain],
            **params,
            "apikey": self.api_key,
        }
        try:
            return request_json(
                self.base_url or explorer_base_url(self.chain),
                params=request_params,
                timeout=20.0,
                error_classifier=classify_etherscan_payload,
                allow_empty_message=allow_empty_message,
            )
        except SourceHttpError as exc:
            raise ExplorerApiError(
                self.chain,
                _explorer_status(exc.category, exc.message),
                exc.message,
                http_status=exc.http_status,
            ) from None


def classify_etherscan_payload(payload: dict[str, Any]) -> SourceHttpStatus | None:
    if str(payload.get("status", "1")) != "0":
        return None
    status = classify_explorer_error(
        str(payload.get("message", "")), payload.get("result", "")
    )
    if status == "rate_limited":
        return "rate_limited"
    if status == "timeout_or_too_broad":
        return "timeout_or_too_broad"
    if status in {"unsupported_chain_plan", "invalid_key"}:
        return "api_error"
    return "bad_request"


def classify_explorer_error(message: str, result: object = "") -> ExplorerStatus:
    text = f"{message} {result}".lower()
    if "free api access is not supported for this chain" in text:
        return "unsupported_chain_plan"
    if "invalid api key" in text:
        return "invalid_key"
    if "max rate limit reached" in text:
        return "rate_limited"
    if (
        "query timeout" in text
        or "timeout occurred" in text
        or "smaller result dataset" in text
    ):
        return "timeout_or_too_broad"
    if (
        "missing or unsupported chainid" in text
        or "missing or invalid action" in text
        or "invalid address format" in text
    ):
        return "bad_request"
    if "no transactions found" in text or "no records found" in text:
        return "empty"
    return "bad_request"


def parse_hex_int(value: object) -> int | None:
    if not isinstance(value, str) or not value.startswith("0x"):
        return None
    try:
        return int(value, 16)
    except ValueError:
        return None


def parse_decimal_int(value: object) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def _explorer_status(category: SourceHttpStatus, message: str) -> ExplorerStatus:
    specific = classify_explorer_error(message)
    if specific != "bad_request":
        return specific
    if category in {"rate_limited", "timeout_or_too_broad", "transport_error", "bad_request"}:
        return category
    return "bad_request"


@dataclass(frozen=True)
class ExchangeAddressBook:
    labels: dict[str, str]

    @classmethod
    def from_toml(cls, path: Path) -> ExchangeAddressBook:
        data = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        labels: dict[str, str] = {}
        for entry in data.get("addresses", []):
            address = str(entry.get("address", "")).lower()
            label = str(entry.get("label", "exchange"))
            if address:
                labels[address] = label
        return cls(labels)

    def is_exchange(self, address: str | None) -> bool:
        return bool(address) and address.lower() in self.labels


def classify_exchange_flows(
    transfers: pl.DataFrame, address_book: ExchangeAddressBook
) -> pl.DataFrame:
    if transfers.is_empty():
        return pl.DataFrame(
            schema={
                "timestamp": pl.Int64,
                "from_address": pl.String,
                "to_address": pl.String,
                "amount": pl.Float64,
                "direction": pl.String,
                "exchange_label": pl.String,
            }
        )
    rows: list[dict[str, Any]] = []
    for row in transfers.to_dicts():
        from_addr = str(row.get("from", row.get("from_address", ""))).lower()
        to_addr = str(row.get("to", row.get("to_address", ""))).lower()
        decimals = int(row.get("tokenDecimal", row.get("decimals", 0)) or 0)
        raw_value = float(row.get("value", row.get("amount", 0)) or 0)
        amount = raw_value / (10**decimals) if decimals > 0 else raw_value
        from_exchange = address_book.is_exchange(from_addr)
        to_exchange = address_book.is_exchange(to_addr)
        if to_exchange and not from_exchange:
            direction = "inflow"
            label = address_book.labels[to_addr]
        elif from_exchange and not to_exchange:
            direction = "outflow"
            label = address_book.labels[from_addr]
        else:
            direction = "internal_or_unknown"
            label = ""
        rows.append(
            {
                "timestamp": int(row.get("timeStamp", row.get("timestamp", 0)) or 0) * 1000,
                "from_address": from_addr,
                "to_address": to_addr,
                "amount": amount,
                "direction": direction,
                "exchange_label": label,
            }
        )
    return pl.DataFrame(rows).sort("timestamp")
