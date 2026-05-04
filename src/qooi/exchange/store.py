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

        Uses ``candles_range`` for deep paginated history, then merges
        with recent data from ``candles`` (OKX only).  Works with both
        OKX SDK and CCXT backends.
        """
        since = (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%d")
        hist = self._md.candles_range(inst_id, bar=bar, since=since, limit=min_bars)
        seen_ts: set[int] = set(hist["timestamp"].to_list()) if not hist.is_empty() else set()

        recent = pl.DataFrame()
        try:
            recent = self._md.candles(inst_id, bar=bar, limit=300)
        except Exception:
            pass

        if not recent.is_empty():
            new_recent = recent.filter(~pl.col("timestamp").is_in(seen_ts))
            if not new_recent.is_empty():
                hist = pl.concat([hist, new_recent]).unique(subset=["timestamp"]).sort("timestamp")

        hist.write_parquet(self._path(inst_id, bar))
        return self._normalize(hist)

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
