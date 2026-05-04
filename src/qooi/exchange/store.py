"""OHLCV data cache — fetch, store as Parquet, load locally.

No API key needed. Caches repeated API calls and enables fast local
backtesting by avoiding network I/O on every run.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from qooi.exchange.market import MarketData

CACHE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "cache"


class CacheStore:
    """OHLCV cache backed by Parquet files.

    Usage::

        cs = CacheStore()
        cs.refresh("BTC-USDT", bar="1H", days=90)   # fetch & save
        df = cs.load("BTC-USDT", bar="1H")           # load from cache
    """

    def __init__(self, md: MarketData | None = None) -> None:
        self._md = md or MarketData()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _path(inst_id: str, bar: str) -> Path:
        safe = inst_id.replace("-", "_")
        return CACHE_DIR / f"{safe}_{bar}.parquet"

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh(
        self,
        inst_id: str,
        bar: str = "1H",
        days: int = 30,
        overwrite: bool = False,
        min_bars: int = 400,
    ) -> pl.DataFrame:
        """Fetch OHLCV and cache as Parquet.

        Paginates ``candles_history`` (OKX) or uses ``since`` (CCXT)
        until ``min_bars`` are collected or the time boundary
        (``days`` ago) is reached, then merges with recent data.
        """
        if self._is_ccxt:
            since = (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%d")
            df = self._md.candles_range(inst_id, bar=bar, since=since, limit=min_bars)
            df.write_parquet(self._path(inst_id, bar))
            return self._normalize(df)

        now = datetime.now(UTC)
        earliest_ms = int((now - timedelta(days=days)).timestamp() * 1000)

        seen_ts: set[int] = set()
        chunks: list[pl.DataFrame] = []
        after: str | None = None

        for _ in range(200):
            chunk = self._md.candles_history(inst_id, bar=bar, after=after, limit=100)
            if chunk.is_empty():
                break
            new = chunk.filter(~pl.col("timestamp").is_in(seen_ts))
            if new.is_empty():
                break
            chunks.append(new)
            seen_ts.update(new["timestamp"].to_list())
            after = str(new["timestamp"].min())
            if int(after) < earliest_ms:
                break
            if sum(len(c) for c in chunks) >= min_bars:
                break

        df = (
            pl.concat(chunks).unique(subset=["timestamp"]).sort("timestamp")
            if chunks
            else pl.DataFrame()
        )

        recent = self._md.candles(inst_id, bar=bar, limit=300)
        if not recent.is_empty():
            new_recent = recent.filter(~pl.col("timestamp").is_in(seen_ts))
            if not new_recent.is_empty():
                df = pl.concat([df, new_recent]).unique(subset=["timestamp"]).sort("timestamp")

        df.write_parquet(self._path(inst_id, bar))
        return self._normalize(df)

    @property
    def _is_ccxt(self) -> bool:
        return self._md._is_ccxt

    # ------------------------------------------------------------------
    # Load / list / clear
    # ------------------------------------------------------------------

    def load(self, inst_id: str, bar: str = "1H") -> pl.DataFrame:
        path = self._path(inst_id, bar)
        if not path.exists():
            raise FileNotFoundError(f"No cache for {inst_id} ({bar}). Run .refresh() first.")
        return self._normalize(pl.read_parquet(path))

    def list_cached(self) -> list[dict]:
        results = []
        for f in sorted(CACHE_DIR.glob("*.parquet")):
            parts = f.stem.split("_")
            bar = (
                parts[-1]
                if parts[-1] in ("1m", "5m", "15m", "30m", "1H", "4H", "1D", "1W")
                else "unknown"
            )
            results.append(
                {
                    "inst_id": f.stem.replace(f"_{bar}", "").replace("_", "-"),
                    "bar": bar,
                    "size_kb": f"{f.stat().st_size / 1024:.0f}",
                }
            )
        return results

    def clear(self, inst_id: str | None = None, bar: str | None = None) -> int:
        removed = 0
        for f in CACHE_DIR.glob("*.parquet"):
            if inst_id and inst_id.replace("-", "_") not in f.stem:
                continue
            if bar and bar not in f.stem:
                continue
            f.unlink()
            removed += 1
        return removed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return df
        cols = {"timestamp", "datetime", "open", "high", "low", "close", "vol"}
        keep = [c for c in cols if c in df.columns]
        return df.select(keep).sort("timestamp")
