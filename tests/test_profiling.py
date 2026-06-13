from __future__ import annotations

import polars as pl

from qooi.profiling import ProfileConfig, ProfileContext


def test_profile_context_records_stage_and_frame_artifacts(tmp_path) -> None:
    profile = ProfileContext.from_config(ProfileConfig(mode="stage"), tmp_path / "profile")

    with profile.stage("unit", "test", "work"):
        frame = pl.DataFrame(
            {
                "symbol": ["BTC", "ETH", "BTC"],
                "timeframe": ["1H", "1H", "4H"],
                "outcome_horizon": [12, 12, 24],
                "source_family": ["books", "books", "trades"],
                "decision_timeframe": ["1H", "1H", "1H"],
            }
        )
    profile.frame("unit", "test", "sample", frame)
    profile.write()

    stages = pl.read_csv(tmp_path / "profile" / "stages.csv")
    frames = pl.read_csv(tmp_path / "profile" / "frames.csv")
    summary = (tmp_path / "profile" / "summary.md").read_text(encoding="utf-8")

    assert stages.select("layer", "component", "stage", "status").to_dicts() == [
        {"layer": "unit", "component": "test", "stage": "work", "status": "ok"}
    ]
    assert stages.get_column("seconds").item() >= 0.0
    assert frames.row(0, named=True) == {
        "run_id": profile.run_id,
        "layer": "unit",
        "component": "test",
        "frame": "sample",
        "rows": 3,
        "cols": 5,
        "symbol_count": 2,
        "timeframe_count": 2,
        "horizon_count": 2,
        "source_family_count": 2,
        "decision_timeframe_count": 1,
    }
    assert "unit.test.work" in summary
    assert "unit.test.sample" in summary


def test_profile_context_hotpath_records_native_profile(tmp_path) -> None:
    profile = ProfileContext.from_config(
        ProfileConfig(mode="hotpath", top_n=5), tmp_path / "profile"
    )

    result = profile.native("unit.work", lambda: sum(range(10)))
    profile.write()

    native = pl.read_csv(tmp_path / "profile" / "native.csv")

    assert result == 45
    assert native.height > 0
    assert native.get_column("function").str.contains("unit.work").any()
    assert native.get_column("cumtime_s").max() >= 0.0


def test_profile_context_off_mode_writes_no_artifacts(tmp_path) -> None:
    profile = ProfileContext.from_config(ProfileConfig(mode="off"), tmp_path / "profile")

    with profile.stage("unit", "test", "work"):
        pass
    profile.frame("unit", "test", "empty", pl.DataFrame())
    profile.write()

    assert not (tmp_path / "profile").exists()
