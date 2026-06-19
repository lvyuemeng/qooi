"""Composable pipeline constants and time helpers."""

from __future__ import annotations

from datetime import UTC, datetime

HOUR_MS = 60 * 60 * 1000
DAY_MS = 24 * HOUR_MS


def now_ms() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)
