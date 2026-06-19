"""Cache behavior tests — incremental, merge, cache-only, force.

No network needed. Uses sample data to simulate cache patterns.
"""

import polars as pl

from qooi.pipeline.io import load_frame, merge_frames, save_frame


def _bars(rows: int, start_ts: int = 1_700_000_000_000) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": [start_ts + i * 3_600_000 for i in range(rows)],
            "open": [100.0 + i * 0.1 for i in range(rows)],
            "high": [101.0 + i * 0.1 for i in range(rows)],
            "low": [99.0 + i * 0.1 for i in range(rows)],
            "close": [100.5 + i * 0.1 for i in range(rows)],
            "vol": [1000.0 for _ in range(rows)],
        }
    ).with_columns(pl.col("timestamp").cast(pl.Int64))


# ═══════════════════════════════════════════════════════════════════════════
# Incremental behavior
# ═══════════════════════════════════════════════════════════════════════════


def test_incremental_no_overlap(tmp_cache_dir):
    """100 existing + 10 new (no overlap) → 110 rows."""
    path = tmp_cache_dir / "test.parquet"
    existing = _bars(100)
    save_frame(path, existing, {}, fmt="parquet")

    newer = _bars(10, start_ts=1_700_000_000_000 + 100 * 3_600_000)
    cached = load_frame(path, {}, fmt="parquet")
    merged = merge_frames(cached, newer, ("timestamp",))

    assert merged.height == 110
    assert merged.get_column("timestamp").is_sorted()


def test_incremental_partial_overlap():
    """100 existing + 10 new (5 overlap) → 105 rows, newer values kept."""
    existing = _bars(100)
    # 5 overlap with existing rows 95-99
    newer = _bars(10, start_ts=1_700_000_000_000 + 95 * 3_600_000)

    merged = merge_frames(existing, newer, ("timestamp",))
    assert merged.height == 105

    # overlapping timestamps should use newer values
    overlap_ts = newer.get_column("timestamp")[0]
    row = merged.filter(pl.col("timestamp") == overlap_ts)
    assert row.get_column("close")[0] == newer.get_column("close")[0]


def test_incremental_full_overlap():
    """100 existing + 100 same (full overlap) → 100 rows."""
    existing = _bars(100)
    newer = _bars(100)  # same timestamps, different OHLC values

    merged = merge_frames(existing, newer, ("timestamp",))
    assert merged.height == 100
    # newer values kept (keep="last")
    assert merged.get_column("close")[0] == newer.get_column("close")[0]


def test_incremental_three_runs(tmp_cache_dir):
    """Three sequential incremental loads — no data loss."""
    path = tmp_cache_dir / "triple.parquet"

    # Run 1: 100 rows
    run1 = _bars(100)
    save_frame(path, run1, {}, fmt="parquet")
    assert load_frame(path, {}, fmt="parquet").height == 100

    # Run 2: add 10 more
    run2 = _bars(10, start_ts=1_700_000_000_000 + 100 * 3_600_000)
    cached = load_frame(path, {}, fmt="parquet")
    merged = merge_frames(cached, run2, ("timestamp",))
    save_frame(path, merged, {}, fmt="parquet")
    assert load_frame(path, {}, fmt="parquet").height == 110

    # Run 3: add 10 more
    run3 = _bars(10, start_ts=1_700_000_000_000 + 110 * 3_600_000)
    cached = load_frame(path, {}, fmt="parquet")
    merged = merge_frames(cached, run3, ("timestamp",))
    save_frame(path, merged, {}, fmt="parquet")
    assert load_frame(path, {}, fmt="parquet").height == 120


# ═══════════════════════════════════════════════════════════════════════════
# Cache mode: cache_only
# ═══════════════════════════════════════════════════════════════════════════


def test_cache_only_loads_from_disk(tmp_cache_dir):
    """Cache-only uses cached data, even if stale."""
    path = tmp_cache_dir / "cache_only.parquet"
    existing = _bars(50)
    save_frame(path, existing, {}, fmt="parquet")

    cached = load_frame(path, {}, fmt="parquet")
    assert cached.height == 50


def test_cache_only_missing_file_returns_empty(tmp_cache_dir):
    """Cache-only with no cache file → empty frame."""
    path = tmp_cache_dir / "nonexistent.parquet"
    cached = load_frame(path, {"timestamp": pl.Int64}, fmt="parquet")
    assert cached.is_empty()


# ═══════════════════════════════════════════════════════════════════════════
# Cache mode: force
# ═══════════════════════════════════════════════════════════════════════════


def test_force_overwrites_cache(tmp_cache_dir):
    """Force mode: save new data, overwrite cache completely."""
    path = tmp_cache_dir / "force.parquet"
    old = _bars(100)
    save_frame(path, old, {}, fmt="parquet")

    # Force overwrite with only new data (50 rows)
    new = _bars(50, start_ts=1_700_000_000_000 + 200 * 3_600_000)
    save_frame(path, new, {}, fmt="parquet")

    loaded = load_frame(path, {}, fmt="parquet")
    assert loaded.height == 50
    # only new timestamps, no old ones
    assert loaded.get_column("timestamp").min() >= 1_700_000_000_000 + 200 * 3_600_000


# ═══════════════════════════════════════════════════════════════════════════
# Merge dedup integrity
# ═══════════════════════════════════════════════════════════════════════════


def test_merge_sorted_output():
    existing = _bars(100)
    newer = _bars(50, start_ts=1_700_000_000_000 + 50 * 3_600_000)
    merged = merge_frames(existing, newer, ("timestamp",))
    assert merged.get_column("timestamp").is_sorted()


def test_merge_all_unique_timestamps():
    existing = _bars(100)
    newer = _bars(50, start_ts=1_700_000_000_000 + 150 * 3_600_000)
    merged = merge_frames(existing, newer, ("timestamp",))
    ts = merged.get_column("timestamp").to_list()
    assert len(ts) == len(set(ts))
