"""Frame I/O — load, save, merge."""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl


def load_frame(path: Path, schema: dict[str, pl.DataType], *, fmt: str = "csv") -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame(schema=schema)
    if fmt == "parquet":
        return pl.read_parquet(path)
    return pl.read_csv(path)


def save_frame(
    path: Path, frame: pl.DataFrame, schema: dict[str, pl.DataType], *, fmt: str = "csv"
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if fmt == "parquet":
        frame.write_parquet(tmp)
    else:
        frame.write_csv(tmp)
    tmp.replace(path)


def merge_frames(
    existing: pl.DataFrame, fetched: pl.DataFrame, keys: tuple[str, ...], *, max_rows: int = 0
) -> pl.DataFrame:
    if fetched.is_empty():
        return existing
    if existing.is_empty():
        return fetched.head(max_rows) if max_rows else fetched
    combined = pl.concat([existing, fetched], how="diagonal_relaxed")
    combined = combined.unique(subset=list(keys), keep="last") if keys else combined
    combined = combined.sort(list(keys))
    return combined.tail(max_rows) if max_rows else combined
