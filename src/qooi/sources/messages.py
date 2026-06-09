"""Local message source normalization and deterministic classification."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import polars as pl


@dataclass(frozen=True)
class LocalMessageSettings:
    default_source: str = "local_csv"
    model_name: str = "keyword_rules"
    model_version: str = "v1"


MESSAGE_COLUMNS = {
    "symbol": pl.String,
    "timestamp": pl.Int64,
    "source": pl.String,
    "source_id": pl.String,
    "author_id_hash": pl.String,
    "text_hash": pl.String,
    "lang": pl.String,
    "text": pl.String,
    "url": pl.String,
    "engagement_count": pl.Int64,
    "reply_count": pl.Int64,
    "repost_count": pl.Int64,
}

CLASSIFICATION_COLUMNS = {
    "symbol": pl.String,
    "timestamp": pl.Int64,
    "message_id": pl.String,
    "message_type": pl.String,
    "message_type_confidence": pl.Float64,
    "stage_hint": pl.String,
    "model_name": pl.String,
    "model_version": pl.String,
    "data_quality_warning": pl.String,
}


def normalize_local_messages(
    frame: pl.DataFrame,
    *,
    settings: LocalMessageSettings = LocalMessageSettings(),
) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(schema=MESSAGE_COLUMNS)
    required = {"symbol", "timestamp", "text"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"local messages missing required columns: {sorted(missing)}")
    rows = []
    for row in frame.to_dicts():
        text = str(row.get("text") or "")
        text_hash = str(row.get("text_hash") or _hash_text(text))
        source = str(row.get("source") or settings.default_source)
        source_id = str(row.get("source_id") or text_hash)
        rows.append(
            {
                "symbol": str(row.get("symbol") or ""),
                "timestamp": int(row.get("timestamp") or 0),
                "source": source,
                "source_id": source_id,
                "author_id_hash": str(row.get("author_id_hash") or ""),
                "text_hash": text_hash,
                "lang": str(row.get("lang") or ""),
                "text": text,
                "url": str(row.get("url") or ""),
                "engagement_count": _int(row.get("engagement_count")),
                "reply_count": _int(row.get("reply_count")),
                "repost_count": _int(row.get("repost_count")),
            }
        )
    return _coerce(pl.DataFrame(rows), MESSAGE_COLUMNS)


def classify_message_rows(
    frame: pl.DataFrame,
    *,
    settings: LocalMessageSettings = LocalMessageSettings(),
) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(schema=CLASSIFICATION_COLUMNS)
    rows = []
    for row in frame.to_dicts():
        message_type = _message_type(str(row.get("text") or ""))
        rows.append(
            {
                "symbol": str(row.get("symbol") or ""),
                "timestamp": int(row.get("timestamp") or 0),
                "message_id": str(row.get("source_id") or row.get("text_hash") or ""),
                "message_type": message_type,
                "message_type_confidence": 0.8 if message_type != "unknown_or_noise" else 0.3,
                "stage_hint": "",
                "model_name": settings.model_name,
                "model_version": settings.model_version,
                "data_quality_warning": ""
                if message_type != "unknown_or_noise"
                else "message_low_confidence",
            }
        )
    return _coerce(pl.DataFrame(rows), CLASSIFICATION_COLUMNS)


def _message_type(text: str) -> str:
    lowered = text.lower()
    if _contains(
        lowered,
        (
            "mainnet",
            "upgrade",
            "partnership",
            "integration",
            "governance",
            "tokenomics",
            "ecosystem",
        ),
    ):
        return "fundamental"
    if _contains(
        lowered,
        (
            "listing",
            "transfer",
            "whale",
            "market maker",
            "funding",
            "oi",
            "open interest",
            "liquidation",
        ),
    ):
        return "trading_funds"
    if _contains(lowered, ("fomo", "fud", "panic", "anger", "meme", "shill", "hype")):
        return "community_emotion"
    return "unknown_or_noise"


def _contains(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _int(value: Any) -> int:
    return int(value or 0)


def _coerce(frame: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    for col, dtype in schema.items():
        if col not in frame.columns:
            frame = frame.with_columns(pl.lit(None).cast(dtype).alias(col))
        else:
            frame = frame.with_columns(pl.col(col).cast(dtype, strict=False).alias(col))
    return frame.select(schema.keys())

