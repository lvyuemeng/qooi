"""Test pipeline I/O primitives — merge_frames, load_frame, save_frame."""

import polars as pl

from qooi.pipeline.io import load_frame, merge_frames, save_frame

# ═══════════════════════════════════════════════════════════════════════════
# merge_frames
# ═══════════════════════════════════════════════════════════════════════════


def test_merge_empty_fetched_returns_existing(sample_bars):
    empty = pl.DataFrame(schema={"timestamp": pl.Int64})
    result = merge_frames(sample_bars, empty, ("timestamp",))
    assert result.height == 100


def test_merge_empty_existing_returns_fetched(sample_bars):
    empty = pl.DataFrame(schema={"timestamp": pl.Int64})
    result = merge_frames(empty, sample_bars, ("timestamp",))
    assert result.height == 100


def test_merge_both_nonempty_dedups_by_keys(sample_bars, sample_bars_newer):
    result = merge_frames(sample_bars, sample_bars_newer, ("timestamp",))
    # 100 existing + 10 fetched, 5 overlap → 105 unique timestamps
    assert result.height == 105
    # all timestamps sorted
    assert result.get_column("timestamp").is_sorted()


def test_merge_max_rows_truncates(sample_bars, sample_bars_newer):
    result = merge_frames(sample_bars, sample_bars_newer, ("timestamp",), max_rows=50)
    assert result.height == 50
    # should keep last 50 (most recent)
    max_ts = result.get_column("timestamp").max()
    assert max_ts == sample_bars_newer.get_column("timestamp").max()


def test_merge_preserves_dedup_keep_last(sample_bars, sample_bars_newer):
    # sample_bars_newer starts at row 95, overlapping 5 rows
    # merged result should use newer values for overlapping timestamps
    result = merge_frames(sample_bars, sample_bars_newer, ("timestamp",))
    overlap_ts = sample_bars_newer.get_column("timestamp")[0]  # first newer ts
    row = result.filter(pl.col("timestamp") == overlap_ts)
    assert row.get_column("close")[0] == sample_bars_newer.get_column("close")[0]


# ═══════════════════════════════════════════════════════════════════════════
# load_frame / save_frame
# ═══════════════════════════════════════════════════════════════════════════


def test_load_missing_file_returns_empty(tmp_cache_dir):
    path = tmp_cache_dir / "nonexistent.csv"
    frame = load_frame(path, {"timestamp": pl.Int64, "value": pl.Float64}, fmt="csv")
    assert frame.is_empty()
    assert "timestamp" in frame.columns
    assert "value" in frame.columns


def test_csv_roundtrip_preserves_types(tmp_cache_dir, sample_bars):
    path = tmp_cache_dir / "test.csv"
    save_frame(path, sample_bars, {}, fmt="csv")
    assert path.exists()

    loaded = load_frame(path, {}, fmt="csv")
    assert loaded.height == 100
    assert loaded.get_column("timestamp").dtype == pl.Int64
    assert loaded.get_column("close").dtype == pl.Float64


def test_parquet_roundtrip_preserves_types(tmp_cache_dir, sample_bars):
    path = tmp_cache_dir / "test.parquet"
    save_frame(path, sample_bars, {}, fmt="parquet")
    assert path.exists()

    loaded = load_frame(path, {}, fmt="parquet")
    assert loaded.height == 100
    assert loaded.get_column("timestamp").dtype == pl.Int64


def test_save_atomic_no_partial_file(tmp_cache_dir, sample_bars):
    path = tmp_cache_dir / "atomic.parquet"
    save_frame(path, sample_bars, {}, fmt="parquet")
    tmp_files = list(tmp_cache_dir.glob(".*"))
    assert len(tmp_files) == 0  # no tmp files left behind
    assert path.exists()
