"""OHLCV data cache — fetch, store as Parquet, load locally.

No API key needed. Caches repeated API calls and enables fast local
backtesting by avoiding network I/O on every run.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from qooi.exchange.market import MarketData

CACHE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "cache"


@dataclass(frozen=True)
class HistoryRequest:
    inst_id: str
    bar: str = "1H"
    days: int = 30
    min_bars: int = 400
    refresh: bool = False


@dataclass(frozen=True)
class HistoryTarget:
    inst_id: str
    bar: str
    requested_days: int
    requested_min_bars: int
    target_days: int
    target_bars: int
    target_since_ms: int


@dataclass(frozen=True)
class HistoryCoverage:
    inst_id: str
    bar: str
    target: HistoryTarget
    actual_bars: int
    actual_start_ms: int | None
    actual_end_ms: int | None
    duplicate_timestamps: int
    gap_count: int
    newest_age_hours: float
    coverage_pct: float
    refreshed: bool = False
    notes: tuple[str, ...] = ()

    def note(self) -> str:
        return (
            f"{self.inst_id} {self.bar}: target_bars={self.target.target_bars} "
            f"target_days={self.target.target_days} actual_bars={self.actual_bars} "
            f"coverage={self.coverage_pct:.1f}% "
            f"start={self.actual_start_ms or 'n/a'} end={self.actual_end_ms or 'n/a'} "
            f"dups={self.duplicate_timestamps} gaps={self.gap_count} "
            f"age_h={self.newest_age_hours:.1f} refreshed={'yes' if self.refreshed else 'no'}"
        )


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
        safe = inst_id.replace("-", "_").replace("/", "_").upper()
        timeframe = bar.replace(" ", "").upper()
        return CACHE_DIR / f"{safe}_{timeframe}.parquet"

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
        target = plan_history(inst_id, bar, days=days, min_bars=min_bars)
        since = datetime.fromtimestamp(target.target_since_ms / 1000, tz=UTC).strftime("%Y-%m-%d")
        hist = self._md.candles_range(inst_id, timeframe=bar, since=since, limit=target.target_bars)
        path = self._path(inst_id, bar)
        if hist.is_empty() and path.exists():
            return self.load(inst_id, bar=bar)
        hist = self._normalize(hist)
        if path.exists():
            existing = self.load(inst_id, bar=bar)
            if not existing.is_empty():
                common_cols = [col for col in existing.columns if col in hist.columns]
                existing = existing.select(common_cols)
                hist = hist.select(common_cols)
                hist = pl.concat([existing, hist]).unique(subset=["timestamp"]).sort("timestamp")
                if hist.height > target.target_bars:
                    hist = hist.tail(target.target_bars)
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

        hist.write_parquet(path)
        return self._normalize(hist)

    def load_history(self, request: HistoryRequest) -> tuple[pl.DataFrame, HistoryCoverage]:
        """Load or refresh one cache and return data-layer coverage metadata."""
        target = plan_history(
            request.inst_id,
            request.bar,
            days=request.days,
            min_bars=request.min_bars,
        )
        refreshed = request.refresh or not self._path(request.inst_id, request.bar).exists()
        df = self.load_or_refresh(
            request.inst_id,
            bar=request.bar,
            days=request.days,
            min_bars=request.min_bars,
            refresh=request.refresh,
        )
        fetch_notes = self._md.last_ohlcv_audit if refreshed else ()
        return df, validate_history(df, target, refreshed=refreshed, extra_notes=fetch_notes)

    # ------------------------------------------------------------------
    # Load / list / clear
    # ------------------------------------------------------------------

    def load(self, inst_id: str, bar: str = "1H") -> pl.DataFrame:
        path = self._path(inst_id, bar)
        if not path.exists():
            raise FileNotFoundError(f"No cache for {inst_id} ({bar}). Run .refresh() first.")
        return self._normalize(pl.read_parquet(path))

    def load_or_refresh(
        self,
        inst_id: str,
        bar: str = "1H",
        *,
        days: int = 30,
        min_bars: int = 400,
        refresh: bool = False,
    ) -> pl.DataFrame:
        """Load one OHLCV cache, refreshing only when requested or missing."""
        if refresh or not self._path(inst_id, bar).exists():
            return self.refresh(inst_id, bar=bar, days=days, min_bars=min_bars)
        return self.load(inst_id, bar=bar)

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
        """Load cached bars and align optional funding data."""
        df = self.load(inst_id, bar=bar)
        if funding_inst_id:
            df = self.attach_funding_rate(df, self.load_funding(funding_inst_id))
        if order_book_inst_id:
            raise ValueError("Order-book enrichment belongs in qooi.strategies.indicators")
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
        cols = ("timestamp", "datetime", "open", "high", "low", "close", "vol")
        keep = [col for col in cols if col in df.columns]
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
                ((pl.col("timestamp") - pl.col("funding_time")) / 3_600_000.0).alias(
                    "funding_age_hours"
                )
            )
        )

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
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Order-book frame missing columns: {missing}")
        return df.select(required).unique(subset=["timestamp"]).sort("timestamp")


def plan_history(inst_id: str, bar: str, *, days: int, min_bars: int) -> HistoryTarget:
    interval_ms = _bar_interval_ms(bar) or 86_400_000
    days_from_bars = max(1, int((min_bars * interval_ms + 86_400_000 - 1) / 86_400_000))
    target_bars = max(1, min_bars)
    target_days = max(days, days_from_bars)
    target_since_ms = int((datetime.now(UTC) - timedelta(days=target_days)).timestamp() * 1000)
    return HistoryTarget(
        inst_id=inst_id,
        bar=bar,
        requested_days=days,
        requested_min_bars=min_bars,
        target_days=target_days,
        target_bars=target_bars,
        target_since_ms=target_since_ms,
    )


def validate_history(
    df: pl.DataFrame,
    target: HistoryTarget,
    *,
    refreshed: bool = False,
    extra_notes: tuple[str, ...] = (),
) -> HistoryCoverage:
    if df.is_empty() or "timestamp" not in df.columns:
        return HistoryCoverage(
            inst_id=target.inst_id,
            bar=target.bar,
            target=target,
            actual_bars=0,
            actual_start_ms=None,
            actual_end_ms=None,
            duplicate_timestamps=0,
            gap_count=0,
            newest_age_hours=0.0,
            coverage_pct=0.0,
            refreshed=refreshed,
            notes=("empty_or_missing_timestamp", *extra_notes),
        )
    ts = [int(v) for v in df["timestamp"].to_list() if v is not None]
    unique_ts = len(set(ts))
    duplicate_timestamps = len(ts) - unique_ts
    expected_ms = _bar_interval_ms(target.bar)
    gap_count = sum(
        1 for i in range(1, len(ts)) if expected_ms > 0 and ts[i] - ts[i - 1] != expected_ms
    )
    newest_age_hours = max(0.0, (datetime.now(UTC).timestamp() * 1000 - ts[-1]) / 3_600_000.0)
    coverage_pct = len(ts) / max(target.target_bars, 1) * 100.0
    notes = []
    if len(ts) < target.target_bars:
        notes.append("below_target_bars")
    if ts[0] > target.target_since_ms + expected_ms:
        notes.append("starts_after_target_since")
    if duplicate_timestamps:
        notes.append("duplicate_timestamps")
    if gap_count:
        notes.append("timeframe_gaps")
    return HistoryCoverage(
        inst_id=target.inst_id,
        bar=target.bar,
        target=target,
        actual_bars=df.height,
        actual_start_ms=ts[0] if ts else None,
        actual_end_ms=ts[-1] if ts else None,
        duplicate_timestamps=duplicate_timestamps,
        gap_count=gap_count,
        newest_age_hours=newest_age_hours,
        coverage_pct=coverage_pct,
        refreshed=refreshed,
        notes=(*notes, *extra_notes),
    )


def _bar_interval_ms(bar: str) -> int:
    normalized = bar.replace(" ", "")
    unit = normalized[-1:].upper()
    try:
        value = int(normalized[:-1])
    except ValueError:
        return 0
    if unit == "M":
        return value * 60_000
    if unit == "H":
        return value * 3_600_000
    if unit == "D":
        return value * 86_400_000
    if unit == "W":
        return value * 7 * 86_400_000
    return 0
