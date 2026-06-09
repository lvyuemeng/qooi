"""Research artifact bundle and typed frame helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class ArtifactBundle:
    name: str
    tables: dict[str, pl.DataFrame]
    summary: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)

    def write(self, export_dir: str | Path) -> list[str]:
        root = Path(export_dir)
        root.mkdir(parents=True, exist_ok=True)
        written = []
        for name, table in self.tables.items():
            path = root / name
            table.write_csv(path)
            written.append(str(path))
        return written

    def non_empty(self, allow_empty: Iterable[str] = ()) -> ArtifactBundle:
        allowed = set(allow_empty)
        return ArtifactBundle(
            self.name,
            {
                name: frame
                for name, frame in self.tables.items()
                if not frame.is_empty() or name in allowed
            },
            summary=self.summary,
            warnings=self.warnings,
            metadata=self.metadata,
        )

    def pick(self, names: Iterable[str]) -> ArtifactBundle:
        wanted = set(names)
        return ArtifactBundle(
            self.name,
            {name: frame for name, frame in self.tables.items() if name in wanted},
            summary=self.summary,
            warnings=self.warnings,
            metadata=self.metadata,
        )


def ensure_columns(frame: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(schema=schema)
    additions = [
        pl.lit(None).cast(dtype).alias(column)
        for column, dtype in schema.items()
        if column not in frame.columns
    ]
    work = frame.with_columns(additions) if additions else frame
    casts = [pl.col(column).cast(dtype).alias(column) for column, dtype in schema.items()]
    extras = [pl.col(column) for column in work.columns if column not in schema]
    return work.select([*casts, *extras])


def empty_frame(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)
