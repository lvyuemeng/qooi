"""Optional SQLite adapter for accumulation scanner outputs."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import polars as pl

FEATURE_TABLE = "accumulation_features"
SCORE_TABLE = "accumulation_scores"
ALERT_TABLE = "accumulation_alerts"
TRANSFER_TABLE = "onchain_transfers"
TRADE_TABLE = "market_trades"
DISCOVERY_TABLE = "candidate_discovery"
SOURCE_MANIFEST_TABLE = "source_manifest"
CANDIDATE_SUMMARY_TABLE = "candidate_summary"

ALLOWED_TABLES = {
    FEATURE_TABLE,
    SCORE_TABLE,
    ALERT_TABLE,
    TRANSFER_TABLE,
    TRADE_TABLE,
    DISCOVERY_TABLE,
    SOURCE_MANIFEST_TABLE,
    CANDIDATE_SUMMARY_TABLE,
}


class AccumulationStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def init(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS accumulation_features (
                    timestamp INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY(timestamp, symbol)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS accumulation_scores (
                    timestamp INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    alert_level TEXT NOT NULL,
                    score_total INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY(timestamp, symbol)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS accumulation_alerts (
                    timestamp INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    alert_level TEXT NOT NULL,
                    score_total INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY(timestamp, symbol)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS onchain_transfers (
                    timestamp INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY(timestamp, symbol, payload)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS market_trades (
                    timestamp INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY(timestamp, symbol, payload)
                )
                """
            )
            for table in (DISCOVERY_TABLE, SOURCE_MANIFEST_TABLE, CANDIDATE_SUMMARY_TABLE):
                conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                        timestamp INTEGER NOT NULL,
                        symbol TEXT NOT NULL,
                        payload TEXT NOT NULL,
                        PRIMARY KEY(timestamp, symbol, payload)
                    )
                    """
                )

    def upsert_frame(self, table: str, frame: pl.DataFrame) -> None:
        if table not in ALLOWED_TABLES:
            raise ValueError(f"Unsupported accumulation table: {table}")
        if frame.is_empty():
            return
        self.init()
        with sqlite3.connect(self.path) as conn:
            score_rows = []
            generic_rows = []
            for row in frame.to_dicts():
                timestamp = int(row.get("timestamp", 0) or 0)
                symbol = str(row.get("symbol", row.get("inst_id", "")))
                payload = str(row)
                if table in {SCORE_TABLE, ALERT_TABLE}:
                    score_rows.append(
                        (
                            timestamp,
                            symbol,
                            str(row.get("alert_level", "")),
                            int(row.get("score_total", 0) or 0),
                            payload,
                        )
                    )
                else:
                    generic_rows.append((timestamp, symbol, payload))
            if score_rows:
                conn.executemany(
                    f"""
                    INSERT OR REPLACE INTO {table}
                        (timestamp, symbol, alert_level, score_total, payload)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    score_rows,
                )
            if generic_rows:
                conn.executemany(
                    f"""
                    INSERT OR REPLACE INTO {table} (timestamp, symbol, payload)
                    VALUES (?, ?, ?)
                    """,
                    generic_rows,
                )

    def read_table(self, table: str) -> pl.DataFrame:
        if table not in ALLOWED_TABLES:
            raise ValueError(f"Unsupported accumulation table: {table}")
        self.init()
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(f"SELECT * FROM {table} ORDER BY timestamp, symbol").fetchall()
            cols = [desc[0] for desc in conn.execute(f"SELECT * FROM {table} LIMIT 0").description]
        return (
            pl.DataFrame([dict(zip(cols, row)) for row in rows])
            if rows
            else pl.DataFrame(schema={col: pl.String for col in cols})
        )


def maybe_store_frame(
    output_dir: Path,
    *,
    enabled: bool,
    relative_path: str,
    table: str,
    frame: pl.DataFrame,
) -> None:
    if enabled:
        AccumulationStore(output_dir / relative_path).upsert_frame(table, frame)
