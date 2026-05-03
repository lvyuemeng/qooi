"""OKX OHLCV data cache — fetch, store as Parquet, load locally.

No API key needed for public market data. Caching avoids repeated API calls
and enables fast local backtesting.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from qooi.exchange.market import MarketData

CACHE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "cache"


class CacheStore:
    """OHLCV cache backed by Parquet files on disk.

    Usage::

        cs = CacheStore()
        cs.refresh("BTC-USDT", bar="1H", days=30)   # fetch & save
        df = cs.load("BTC-USDT", bar="1H")            # load from cache
    """

    def __init__(self, md: MarketData | None = None) -> None:
        self._md = md or MarketData()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _path(self, inst_id: str, bar: str) -> Path:
        safe = inst_id.replace("-", "_")
        return CACHE_DIR / f"{safe}_{bar}.parquet"

    # ------------------------------------------------------------------
    # Refresh (fetch from API and save)
    # ------------------------------------------------------------------

    def refresh(
        self,
        inst_id: str,
        bar: str = "1H",
        days: int = 30,
        overwrite: bool = False,
    ) -> pl.DataFrame:
        """Fetch OHLCV data from OKX and save to local Parquet cache.

        Uses ``candles_history`` (archived data, up to 3 months back)
        for older bars and ``candles`` for recent ones.
        """
        now = datetime.now(UTC)
        after_ms = int((now - timedelta(days=days)).timestamp() * 1000)
        limit = 300
        # OKX candles_history returns up to 3 months, max 100 per call
        # candles returns up to 300 per call

        all_rows: list[pl.DataFrame] = []

        # 1. Fetch historical (older data) - up to 100 per page
        seen_ts: set[int] = set()
        after = str(after_ms)
        for _ in range(50):  # max 50 pages
            df = self._md.candles_history(inst_id, bar=bar, after=after, limit=100)
            if df.is_empty():
                break
            new = df.filter(~pl.col("timestamp").is_in(seen_ts))
            if new.is_empty():
                break
            all_rows.append(new)
            seen_ts.update(new["timestamp"].to_list())
            after = str(new["timestamp"].min())
            if after is None or int(after) < after_ms:
                break

        if all_rows:
            df = pl.concat(all_rows).unique(subset=["timestamp"]).sort("timestamp")
            # Save historical
            df.write_parquet(self._path(inst_id, bar))
        else:
            df = pl.DataFrame()

        # 2. Merge with recent data from candles (up to 300)
        recent = self._md.candles(inst_id, bar=bar, limit=limit)
        if not recent.is_empty():
            recent = recent.filter(~pl.col("timestamp").is_in(seen_ts))
            if not recent.is_empty():
                df = pl.concat([df, recent]).unique(subset=["timestamp"]).sort("timestamp")
                df.write_parquet(self._path(inst_id, bar))

        return self._normalize(df)

    # ------------------------------------------------------------------
    # Load from cache
    # ------------------------------------------------------------------

    def load(self, inst_id: str, bar: str = "1H") -> pl.DataFrame:
        """Load cached OHLCV data. Raises if not found."""
        path = self._path(inst_id, bar)
        if not path.exists():
            raise FileNotFoundError(f"No cache for {inst_id} ({bar}). Run .refresh() first.")
        return self._normalize(pl.read_parquet(path))

    # ------------------------------------------------------------------
    # List what's cached
    # ------------------------------------------------------------------

    def list_cached(self) -> list[dict]:
        """List all cached datasets."""
        results = []
        for f in sorted(CACHE_DIR.glob("*.parquet")):
            parts = f.stem.split("_")
            bar = (
                parts[-1]
                if parts[-1] in ("1m", "5m", "15m", "30m", "1H", "4H", "1D", "1W")
                else "unknown"
            )
            inst_id = f.stem.replace(f"_{bar}", "").replace("_", "-")
            size_kb = f.stat().st_size / 1024
            results.append({"inst_id": inst_id, "bar": bar, "size_kb": f"{size_kb:.0f}"})
        return results

    def clear(self, inst_id: str | None = None, bar: str | None = None) -> int:
        """Remove cached files. Returns number of files deleted."""
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

    @staticmethod
    def _bar_to_ms(bar: str) -> int:
        mapping = {
            "1m": 60_000,
            "3m": 180_000,
            "5m": 300_000,
            "15m": 900_000,
            "30m": 1_800_000,
            "1H": 3_600_000,
            "2H": 7_200_000,
            "4H": 14_400_000,
            "6H": 21_600_000,
            "12H": 43_200_000,
            "1D": 86_400_000,
        }
        return mapping.get(bar, 3_600_000)
