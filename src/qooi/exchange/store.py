"""OHLCV data cache — fetch, store as Parquet, load locally.

No API key needed. Caches repeated API calls and enables fast local
backtesting by avoiding network I/O on every run.
"""

from __future__ import annotations

import asyncio
import time
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

    @staticmethod
    def _funding_path(inst_id: str) -> Path:
        safe = inst_id.replace("-", "_")
        return CACHE_DIR / f"{safe}_funding.parquet"

    @staticmethod
    def _order_book_path(inst_id: str) -> Path:
        safe = inst_id.replace("-", "_")
        return CACHE_DIR / f"{safe}_order_book.parquet"

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
        hist = self._md.candles_range(inst_id, timeframe=bar, since=since, limit=min_bars)
        seen_ts: set[int] = set(hist["timestamp"].to_list()) if not hist.is_empty() else set()

        recent = pl.DataFrame()
        try:
            recent = self._md.candles(inst_id, timeframe=bar, limit=300)
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

    def refresh_funding(self, inst_id: str, limit: int = 400) -> pl.DataFrame:
        """Fetch funding-rate history and cache it as Parquet."""
        funding = self._md.funding_rate_history(inst_id, limit=limit)
        funding = self._normalize_funding(funding)
        funding.write_parquet(self._funding_path(inst_id))
        return funding

    def load_funding(self, inst_id: str) -> pl.DataFrame:
        path = self._funding_path(inst_id)
        if not path.exists():
            raise FileNotFoundError(
                f"No funding cache for {inst_id}. Run .refresh_funding() first."
            )
        return self._normalize_funding(pl.read_parquet(path))

    def cache_order_book(
        self, inst_id: str, snapshots: pl.DataFrame, append: bool = True
    ) -> pl.DataFrame:
        """Persist order-book snapshots collected elsewhere."""
        path = self._order_book_path(inst_id)
        current = self._normalize_order_book(snapshots)
        if append and path.exists():
            existing = self._normalize_order_book(pl.read_parquet(path))
            current = (
                pl.concat([existing, current], how="vertical")
                .unique(subset=["timestamp"])
                .sort("timestamp")
            )
        current.write_parquet(path)
        return current

    def record_order_book_rest(
        self,
        inst_id: str,
        *,
        samples: int,
        every_seconds: float = 5.0,
        limit: int = 25,
        append: bool = True,
    ) -> pl.DataFrame:
        """Poll REST order-book snapshots.

        This is useful for ad-hoc diagnostics, but for OKX strategy data
        collection the documented path is the WebSocket depth feed.
        """
        if samples <= 0:
            if self._order_book_path(inst_id).exists():
                return self.load_order_book(inst_id)
            return pl.DataFrame()

        rows: list[dict] = []
        for idx in range(samples):
            snap = self._md.ob_snapshot(inst_id, limit=limit)
            row = snap.to_row()
            if row["timestamp"] <= 0:
                row["timestamp"] = int(datetime.now(UTC).timestamp() * 1000)
            rows.append(row)
            if idx + 1 < samples and every_seconds > 0:
                time.sleep(every_seconds)

        return self.cache_order_book(inst_id, pl.DataFrame(rows), append=append)

    def record_order_book(
        self,
        inst_id: str,
        *,
        samples: int,
        every_seconds: float = 5.0,
        limit: int = 25,
        params: dict | None = None,
        append: bool = True,
        transport: str = "ws",
    ) -> pl.DataFrame:
        """Record and cache order-book snapshots.

        Defaults to WebSocket depth, which is the correct collection path
        for OKX order-book research. Use ``transport="rest"`` only when a
        polling snapshot is explicitly desired.
        """
        if transport == "rest":
            return self.record_order_book_rest(
                inst_id,
                samples=samples,
                every_seconds=every_seconds,
                limit=limit,
                append=append,
            )
        if transport != "ws":
            raise ValueError(f"Unsupported order-book transport: {transport}")
        return self.record_order_book_ws(
            inst_id,
            samples=samples,
            limit=limit,
            params=params,
            append=append,
        )

    def load_order_book(self, inst_id: str) -> pl.DataFrame:
        path = self._order_book_path(inst_id)
        if not path.exists():
            raise FileNotFoundError(
                f"No order-book cache for {inst_id}. Run .record_order_book() first."
            )
        return self._normalize_order_book(pl.read_parquet(path))

    async def record_order_book_ws_async(
        self,
        inst_id: str,
        *,
        samples: int,
        limit: int = 25,
        params: dict | None = None,
        append: bool = True,
    ) -> pl.DataFrame:
        """Record order-book snapshots from the exchange WebSocket stream.

        For OKX, the documented public choices are typically:
        - default ``limit=25`` with no params, which CCXT Pro maps to ``books``
        - ``{"depth": "books"}`` for the incremental public depth feed
        - ``{"depth": "books5"}`` for top-5 public snapshots
        """
        if samples <= 0:
            if self._order_book_path(inst_id).exists():
                return self.load_order_book(inst_id)
            return pl.DataFrame()

        stream_md = await MarketData.async_(self._md.exchange_id, self._md.proxy)
        rows: list[dict] = []
        try:
            async for snap in stream_md.ob_stream(inst_id, limit=limit, params=params):
                row = snap.to_row()
                if row["timestamp"] <= 0:
                    row["timestamp"] = int(datetime.now(UTC).timestamp() * 1000)
                rows.append(row)
                if len(rows) >= samples:
                    break
        finally:
            await stream_md.close()

        return self.cache_order_book(inst_id, pl.DataFrame(rows), append=append)

    def record_order_book_ws(
        self,
        inst_id: str,
        *,
        samples: int,
        limit: int = 25,
        params: dict | None = None,
        append: bool = True,
    ) -> pl.DataFrame:
        """Synchronous wrapper around ``record_order_book_ws_async``."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.record_order_book_ws_async(
                    inst_id,
                    samples=samples,
                    limit=limit,
                    params=params,
                    append=append,
                )
            )
        msg = (
            "record_order_book_ws() cannot run inside an existing event loop; "
            "use the async variant."
        )
        raise RuntimeError(msg)

    def intraday_frame(
        self,
        inst_id: str,
        *,
        bar: str = "1H",
        funding_inst_id: str | None = None,
        order_book_inst_id: str | None = None,
    ) -> pl.DataFrame:
        """Load cached bars and align optional funding / order-book features."""
        df = self.load(inst_id, bar=bar)
        if funding_inst_id:
            df = self.attach_funding_rate(df, self.load_funding(funding_inst_id))
        if order_book_inst_id:
            df = self.attach_order_book(df, self.load_order_book(order_book_inst_id))
        return df

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

    @staticmethod
    def attach_funding_rate(df: pl.DataFrame, funding_df: pl.DataFrame) -> pl.DataFrame:
        """Point-in-time align funding history to each market-data bar."""
        if df.is_empty():
            return df.with_columns(
                [
                    pl.lit(None, dtype=pl.Float64).alias("funding_rate"),
                    pl.lit(None, dtype=pl.Int64).alias("funding_time"),
                ]
            )

        funding = CacheStore._normalize_funding(funding_df)
        if funding.is_empty():
            return df.with_columns(
                [
                    pl.lit(None, dtype=pl.Float64).alias("funding_rate"),
                    pl.lit(None, dtype=pl.Int64).alias("funding_time"),
                ]
            )

        return (
            df.sort("timestamp")
            .join_asof(funding, on="timestamp", strategy="backward")
            .with_columns(
                (
                    (pl.col("timestamp") - pl.col("funding_time")) / 3_600_000.0
                ).alias("funding_age_hours")
            )
        )

    @staticmethod
    def attach_order_book(df: pl.DataFrame, snapshot_df: pl.DataFrame) -> pl.DataFrame:
        """Aggregate recorded order-book snapshots into the OHLCV bar grid."""
        if df.is_empty():
            return df.with_columns(
                [
                    pl.lit(None, dtype=pl.Float64).alias("ob_bid_price"),
                    pl.lit(None, dtype=pl.Float64).alias("ob_ask_price"),
                    pl.lit(None, dtype=pl.Float64).alias("ob_bid_vol_5"),
                    pl.lit(None, dtype=pl.Float64).alias("ob_ask_vol_5"),
                    pl.lit(None, dtype=pl.Float64).alias("ob_bid_vol_25"),
                    pl.lit(None, dtype=pl.Float64).alias("ob_ask_vol_25"),
                    pl.lit(None, dtype=pl.Float64).alias("ob_bid_vol"),
                    pl.lit(None, dtype=pl.Float64).alias("ob_ask_vol"),
                    pl.lit(None, dtype=pl.Float64).alias("ob_imbalance_5"),
                    pl.lit(None, dtype=pl.Float64).alias("ob_imbalance_25"),
                    pl.lit(0, dtype=pl.Int64).alias("ob_samples"),
                ]
            )

        snapshots = CacheStore._normalize_order_book(snapshot_df)
        if snapshots.is_empty():
            return df.with_columns(
                [
                    pl.lit(None, dtype=pl.Float64).alias("ob_bid_price"),
                    pl.lit(None, dtype=pl.Float64).alias("ob_ask_price"),
                    pl.lit(None, dtype=pl.Float64).alias("ob_bid_vol_5"),
                    pl.lit(None, dtype=pl.Float64).alias("ob_ask_vol_5"),
                    pl.lit(None, dtype=pl.Float64).alias("ob_bid_vol_25"),
                    pl.lit(None, dtype=pl.Float64).alias("ob_ask_vol_25"),
                    pl.lit(None, dtype=pl.Float64).alias("ob_bid_vol"),
                    pl.lit(None, dtype=pl.Float64).alias("ob_ask_vol"),
                    pl.lit(None, dtype=pl.Float64).alias("ob_imbalance_5"),
                    pl.lit(None, dtype=pl.Float64).alias("ob_imbalance_25"),
                    pl.lit(0, dtype=pl.Int64).alias("ob_samples"),
                ]
            )

        bar_ms = CacheStore._bar_interval_ms(df)
        bars = df.select(pl.col("timestamp").alias("bar_timestamp")).sort("bar_timestamp")
        mapped = (
            snapshots.rename({"timestamp": "snapshot_timestamp"})
            .sort("snapshot_timestamp")
            .join_asof(
                bars,
                left_on="snapshot_timestamp",
                right_on="bar_timestamp",
                strategy="backward",
            )
            .drop_nulls(["bar_timestamp"])
            .filter((pl.col("snapshot_timestamp") - pl.col("bar_timestamp")) < bar_ms)
        )
        if mapped.is_empty():
            return df.with_columns(
                [
                    pl.lit(None, dtype=pl.Float64).alias("ob_bid_price"),
                    pl.lit(None, dtype=pl.Float64).alias("ob_ask_price"),
                    pl.lit(None, dtype=pl.Float64).alias("ob_bid_vol_5"),
                    pl.lit(None, dtype=pl.Float64).alias("ob_ask_vol_5"),
                    pl.lit(None, dtype=pl.Float64).alias("ob_bid_vol_25"),
                    pl.lit(None, dtype=pl.Float64).alias("ob_ask_vol_25"),
                    pl.lit(None, dtype=pl.Float64).alias("ob_bid_vol"),
                    pl.lit(None, dtype=pl.Float64).alias("ob_ask_vol"),
                    pl.lit(None, dtype=pl.Float64).alias("ob_imbalance_5"),
                    pl.lit(None, dtype=pl.Float64).alias("ob_imbalance_25"),
                    pl.lit(0, dtype=pl.Int64).alias("ob_samples"),
                ]
            )

        per_bar = (
            mapped.group_by("bar_timestamp")
            .agg(
                [
                    pl.col("ob_bid_price").last().alias("ob_bid_price"),
                    pl.col("ob_ask_price").last().alias("ob_ask_price"),
                    pl.col("ob_bid_vol_5").mean().alias("ob_bid_vol_5"),
                    pl.col("ob_ask_vol_5").mean().alias("ob_ask_vol_5"),
                    pl.col("ob_bid_vol_25").mean().alias("ob_bid_vol_25"),
                    pl.col("ob_ask_vol_25").mean().alias("ob_ask_vol_25"),
                    pl.col("ob_bid_vol").mean().alias("ob_bid_vol"),
                    pl.col("ob_ask_vol").mean().alias("ob_ask_vol"),
                    pl.col("ob_imbalance_5").mean().alias("ob_imbalance_5"),
                    pl.col("ob_imbalance_25").mean().alias("ob_imbalance_25"),
                    pl.len().alias("ob_samples"),
                ]
            )
            .rename({"bar_timestamp": "timestamp"})
            .sort("timestamp")
        )
        return df.sort("timestamp").join(per_bar, on="timestamp", how="left")

    @staticmethod
    def _normalize_funding(df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return pl.DataFrame(
                schema={
                    "timestamp": pl.Int64,
                    "funding_rate": pl.Float64,
                    "funding_time": pl.Int64,
                }
            )

        normalized = df
        if "funding_time" not in normalized.columns:
            normalized = normalized.with_columns(pl.col("timestamp").alias("funding_time"))
        return (
            normalized.select(["timestamp", "funding_rate", "funding_time"])
            .unique(subset=["timestamp"])
            .sort("timestamp")
        )

    @staticmethod
    def _normalize_order_book(df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return pl.DataFrame(
                schema={
                    "timestamp": pl.Int64,
                    "ob_bid_price": pl.Float64,
                    "ob_ask_price": pl.Float64,
                    "ob_bid_vol_5": pl.Float64,
                    "ob_ask_vol_5": pl.Float64,
                    "ob_bid_vol_25": pl.Float64,
                    "ob_ask_vol_25": pl.Float64,
                    "ob_bid_vol": pl.Float64,
                    "ob_ask_vol": pl.Float64,
                    "ob_imbalance_5": pl.Float64,
                    "ob_imbalance_25": pl.Float64,
                }
            )

        required = [
            "timestamp",
            "ob_bid_price",
            "ob_ask_price",
            "ob_bid_vol_5",
            "ob_ask_vol_5",
            "ob_bid_vol_25",
            "ob_ask_vol_25",
            "ob_bid_vol",
            "ob_ask_vol",
            "ob_imbalance_5",
            "ob_imbalance_25",
        ]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Order-book frame missing columns: {missing}")
        return df.select(required).unique(subset=["timestamp"]).sort("timestamp")

    @staticmethod
    def _bar_interval_ms(df: pl.DataFrame) -> int:
        if df.height < 2:
            return 3_600_000
        timestamps = df["timestamp"].to_list()
        deltas = [int(timestamps[i] - timestamps[i - 1]) for i in range(1, len(timestamps))]
        positive = [delta for delta in deltas if delta > 0]
        return min(positive) if positive else 3_600_000
