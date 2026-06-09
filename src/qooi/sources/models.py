"""Source collection result models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

SourceStatus = Literal["ok", "partial", "missing", "skipped", "failed"]


@dataclass(frozen=True)
class SourceResult:
    frame: pl.DataFrame
    manifest: pl.DataFrame

