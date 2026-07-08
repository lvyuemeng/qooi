from __future__ import annotations

from pathlib import Path

import polars as pl

from qooi.scanner.tailrun.features import FeatureSpec, ProposalFeatureManifest
from qooi.scanner.tailrun.research import ResearchSpec
from qooi.scanner.tailrun.review import (
    path_decile_monotonicity,
    path_feature_analysis,
    path_feature_analysis_report,
    path_robust_profit_metrics,
    path_source_feature_health,
)

HOUR_MS = 60 * 60 * 1000


def test_feature_spec_train_frame_includes_source_tsflex_without_source_fill() -> None:
    observations = pl.DataFrame(
        {
            "symbol": ["A"] * 8,
            "decision_bar_close_ms": [index * HOUR_MS for index in range(8)],
            "funding_rate_bps": [None, None, 1.0, 2.0, None, 4.0, 5.0, 6.0],
            "oi_change_pct": [0.1, None, 0.2, 0.3, None, 0.5, 0.8, 1.3],
            "taker_buy_pressure": [0.2, 0.1, None, 0.3, 0.5, None, 0.7, 0.9],
        }
    )
    labels = pl.DataFrame(
        {
            "symbol": ["A"],
            "decision_bar_close_ms": [7 * HOUR_MS],
            "horizon_hours": [4],
            "path_label": [1],
            "sample_weight": [1.0],
        }
    )

    matrix = FeatureSpec(
        horizons=(4,),
        windows_hours=(4,),
        source_tsfresh_value_columns=(
            "base__funding_rate_bps",
            "base__oi_change_pct",
            "base__taker_buy_pressure",
        ),
        source_tsfresh_calculators=("sample_count", "maximum", "mean_abs_change"),
    ).train_frame(observations, None, labels)

    assert "base__funding_rate_bps" in matrix.columns
    assert "tsfsrc__base_funding_rate_bps__w4h__sample_count" in matrix.columns
    assert "crosssrc__base_funding_rate_bps_base_oi_change_pct__w4h__corr" in matrix.columns
    assert matrix.select("tsfsrc__base_funding_rate_bps__w4h__sample_count").item() == 3.0
    assert not any(
        "zscore" in column.lower() or "z_score" in column.lower() for column in matrix.columns
    )


def test_source_tsflex_emits_temporal_and_sparse_aware_descriptors() -> None:
    observations = pl.DataFrame(
        {
            "symbol": ["A"] * 8,
            "decision_bar_close_ms": [index * HOUR_MS for index in range(8)],
            "funding_rate_bps": [None, None, 1.0, 2.0, None, 4.0, 5.0, 6.0],
            "oi_change_pct": [0.1, None, 0.2, 0.3, None, 0.5, 0.8, 1.3],
        }
    )
    labels = pl.DataFrame(
        {
            "symbol": ["A"],
            "decision_bar_close_ms": [7 * HOUR_MS],
            "horizon_hours": [4],
            "path_label": [1],
            "sample_weight": [1.0],
        }
    )

    matrix = FeatureSpec(
        horizons=(4,),
        windows_hours=(4,),
        source_tsfresh_value_columns=("base__funding_rate_bps", "base__oi_change_pct"),
        source_tsfresh_calculators=(
            "first_value",
            "last_value",
            "last_minus_first",
            "positive_change_rate",
            "valid_ratio",
            "max_valid_gap",
            "trend_slope",
        ),
    ).train_frame(observations, None, labels)

    assert "tsfsrc__base_funding_rate_bps__w4h__first_value" in matrix.columns
    assert "tsfsrc__base_funding_rate_bps__w4h__last_minus_first" in matrix.columns
    assert "tsfsrc__base_funding_rate_bps__w4h__valid_ratio" in matrix.columns
    assert "tsfsrc__base_funding_rate_bps__w4h__trend_slope" in matrix.columns
    assert "crosssrc__base_funding_rate_bps_base_oi_change_pct__w4h__lead_corr" in matrix.columns
    assert matrix.select("tsfsrc__base_funding_rate_bps__w4h__last_minus_first").item() == 3.0
    assert matrix.select("tsfsrc__base_funding_rate_bps__w4h__valid_ratio").item() == 0.75


def test_research_defaults_can_select_sparse_source_tsflex_prefixes() -> None:
    candidates = pl.DataFrame(
        {
            "symbol": ["A"] * 10,
            "decision_bar_close_ms": list(range(10)),
            "horizon_hours": [4] * 10,
            "path_label": [0, 1] * 5,
            "sample_weight": [1.0] * 10,
            "base__funding_rate_bps": [None] * 8 + [1.0, 2.0],
            "tsfsrc__base_funding_rate_bps__w4h__last_minus_first": [None] * 8 + [0.5, 1.0],
            "base__ret_4h": [None] * 8 + [1.0, 2.0],
        }
    )

    columns = ResearchSpec(min_non_null_rate=0.95, source_min_non_null_rate=0.10).feature_columns(
        candidates
    )

    assert "base__funding_rate_bps" in columns
    assert "tsfsrc__base_funding_rate_bps__w4h__last_minus_first" in columns
    assert "base__ret_4h" not in columns


def test_source_blended_manifest_includes_raw_and_generated_source_features() -> None:
    manifest = ProposalFeatureManifest(
        artifact_id="fixture",
        artifact_kind="feature_manifest.proposal",
        spec=FeatureSpec(horizons=(4,)),
        selected_columns=("base__ret_4h",),
        candidate_feature_columns=("base__ret_4h",),
        fold_ids=(0,),
        fit_row_count=2,
        validation_row_count=1,
        schema_hash="sha256:fixture",
        label_column="path_label",
    )
    matrix = pl.DataFrame(
        {
            "symbol": ["A"],
            "decision_bar_close_ms": [1],
            "horizon_hours": [4],
            "base__ret_4h": [0.1],
            "base__funding_rate_bps": [2.0],
            "tsfsrc__base_funding_rate_bps__w4h__last_minus_first": [1.0],
            "crosssrc__base_funding_rate_bps_base_oi_change_pct__w4h__lead_corr": [0.5],
        }
    )

    blended = manifest.source_blended(matrix)

    assert "base__funding_rate_bps" in blended.selected_columns
    assert "tsfsrc__base_funding_rate_bps__w4h__last_minus_first" in blended.selected_columns
    assert (
        "crosssrc__base_funding_rate_bps_base_oi_change_pct__w4h__lead_corr"
        in blended.selected_columns
    )


def test_source_health_extends_existing_feature_analysis_with_temporal_splits() -> None:
    manifest = ProposalFeatureManifest(
        artifact_id="fixture",
        artifact_kind="feature_manifest.proposal",
        spec=FeatureSpec(horizons=(4,)),
        selected_columns=("base__funding_rate_bps",),
        candidate_feature_columns=("base__funding_rate_bps",),
        fold_ids=(0,),
        fit_row_count=2,
        validation_row_count=1,
        schema_hash="sha256:fixture",
        label_column="path_label",
    )
    matrix = pl.DataFrame(
        {
            "symbol": ["A"] * 10,
            "decision_bar_close_ms": list(range(10)),
            "base__funding_rate_bps": [None] * 8 + [1.0, 2.0],
        }
    )

    source_health = path_source_feature_health(
        matrix, feature_sets={"source_blended_all": manifest}
    )
    analysis = path_feature_analysis(
        feature_set_counts={"source_blended_all": 1}, source_health=source_health
    )

    assert set(source_health.get_column("split").to_list()) == {"build", "train80", "blind20"}
    assert (
        analysis.filter(
            (pl.col("section") == "source_health")
            & (pl.col("split") == "blind20")
            & (pl.col("metric") == "min_finite_rate")
        )
        .select("value")
        .item()
        == 1.0
    )


def test_source_probe_extends_existing_feature_analysis() -> None:
    source_probe = pl.DataFrame(
        {
            "section": ["source_probe"],
            "feature_set": ["source_probe"],
            "split": ["probe_stratified"],
            "metric": ["selected_generated_count"],
            "k": [None],
            "value": [10.0],
            "sample_count": [30],
            "warning": [None],
            "action": ["use_selected_outputs"],
        }
    )

    analysis = path_feature_analysis(feature_set_counts={"base": 1}, source_probe=source_probe)

    assert (
        analysis.filter(
            (pl.col("section") == "source_probe") & (pl.col("metric") == "selected_generated_count")
        )
        .select("value")
        .item()
        == 10.0
    )


def test_feature_analysis_report_renders_source_probe_and_split_coverage() -> None:
    analysis = pl.DataFrame(
        {
            "section": ["selection", "source_probe", "source_health"],
            "feature_set": ["base", "source_probe", "source_blended_all"],
            "split": ["build", "probe_stratified", "train80"],
            "metric": ["selected_count", "selected_generated_count", "low_coverage_feature_count"],
            "k": [None, None, None],
            "value": [2.0, 25.0, 3.0],
            "sample_count": [2, 100, 50],
            "warning": [None, None, "source_coverage_warning"],
            "action": ["keep", "use_selected_outputs", "inspect"],
        }
    )

    report = path_feature_analysis_report(analysis)

    assert "## Source probe" in report
    assert "selected_generated_count=25.0" in report
    assert "source_blended_all/train80" in report


def test_workflow_import_does_not_depend_on_removed_output_module() -> None:
    import qooi.scanner.workflow as workflow

    assert hasattr(workflow, "MarketReadiness")
    assert hasattr(workflow, "prepare_potential_run")


def test_build_stage_removes_retired_output_dirs(tmp_path, monkeypatch) -> None:
    module = _load_build_features_module()
    legacy = tmp_path / "path-train"
    legacy.mkdir()
    (legacy / "feature-analysis.md").write_text("stale", encoding="utf-8")
    monkeypatch.setattr(module, "LEGACY_OUTPUT_DIRS", (legacy,))

    module.remove_legacy_outputs()

    assert not legacy.exists()


def test_build_source_defaults_explore_temporal_tsflex_combinations() -> None:
    import importlib.util

    path = Path("scripts/01_build_features.py")
    spec = importlib.util.spec_from_file_location("build_features_stage", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.MAX_STALENESS_HOURS == 2
    assert module.PREDICT_MATRIX_PATH.name == "predict_features.parquet"
    calculators = module.FEATURE_SPEC.source_tsfresh_calculators
    assert "first_value" not in calculators
    assert "last_value" not in calculators
    assert "mean_change" not in calculators
    assert "last_minus_first" in calculators
    assert "valid_ratio" in calculators
    assert "trend_slope" not in calculators
    assert "trend_r2" not in calculators
    assert "sample_count" in calculators


def test_source_tsflex_selected_outputs_limit_collection() -> None:
    observations = pl.DataFrame(
        {
            "symbol": ["A"] * 8,
            "decision_bar_close_ms": [index * HOUR_MS for index in range(8)],
            "funding_rate_bps": [None, None, 1.0, 2.0, None, 4.0, 5.0, 6.0],
            "oi_change_pct": [0.1, None, 0.2, 0.3, None, 0.5, 0.8, 1.3],
        }
    )
    labels = pl.DataFrame(
        {
            "symbol": ["A"],
            "decision_bar_close_ms": [7 * HOUR_MS],
            "horizon_hours": [4],
            "path_label": [1],
            "sample_weight": [1.0],
        }
    )

    matrix = FeatureSpec(
        horizons=(4,),
        windows_hours=(4, 12),
        source_tsfresh_value_columns=("base__funding_rate_bps", "base__oi_change_pct"),
        source_tsfresh_calculators=("sample_count", "last_minus_first", "trend_slope"),
        selected_generated_columns=(
            "tsfsrc__base_funding_rate_bps__w4h__last_minus_first",
            "crosssrc__base_funding_rate_bps_base_oi_change_pct__w4h__lead_corr",
        ),
    ).train_frame(observations, None, labels)

    source_columns = [
        column for column in matrix.columns if column.startswith(("tsfsrc__", "crosssrc__"))
    ]
    assert source_columns == [
        "tsfsrc__base_funding_rate_bps__w4h__last_minus_first",
        "crosssrc__base_funding_rate_bps_base_oi_change_pct__w4h__lead_corr",
    ]


def _load_build_features_module():
    import importlib.util

    path = Path("scripts/01_build_features.py")
    spec = importlib.util.spec_from_file_location("build_features_stage_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_probe_ndcg_prefers_high_gain_head() -> None:
    module = _load_build_features_module()
    aligned = pl.DataFrame(
        {"feature": [1.0, 2.0, 3.0, 4.0, 5.0], "target": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )
    inverted = pl.DataFrame(
        {"feature": [1.0, 2.0, 3.0, 4.0, 5.0], "target": [5.0, 4.0, 3.0, 2.0, 1.0]}
    )

    aligned_score, finite_count, finite_rate = module._source_probe_ndcg_score(
        aligned, "feature", "target"
    )
    inverted_score, _, _ = module._source_probe_ndcg_score(inverted, "feature", "target")

    assert aligned_score > inverted_score
    assert aligned_score == 1.0
    assert finite_count == 5
    assert finite_rate == 1.0


def test_source_probe_temporal_stability_uses_ndcg_threshold() -> None:
    module = _load_build_features_module()
    frame = pl.DataFrame(
        {
            "decision_bar_close_ms": list(range(20)),
            "feature": [float(x) for x in (list(range(1, 9)) + list(range(1, 9)) + [1, 2, 3, 4])],
            "target": [float(x) for x in (list(range(1, 9)) + ([0] * 8) + [1, 2, 3, 4])],
        }
    )

    stability = module._source_probe_temporal_stability(frame, "feature", "target")

    assert stability["temporal_stable"] is False
    assert stability["train_early_ndcg"] >= module._SOURCE_PROBE_MIN_NDCG
    assert stability["train_late_ndcg"] < stability["train_early_ndcg"]
    assert stability["selection_score"] == 0.0


def test_source_probe_ndcg_fails_below_threshold() -> None:
    module = _load_build_features_module()
    frame = pl.DataFrame(
        {
            "decision_bar_close_ms": list(range(8)),
            "feature": [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
            "target": [0.0] * 8,
        }
    )

    score, finite_count, finite_rate = module._source_probe_ndcg_score(frame, "feature", "target")

    assert score == 0.0
    assert finite_count == 8
    assert finite_rate == 1.0


def test_source_probe_consistency_rows_are_reported() -> None:
    source_probe = pl.DataFrame(
        {
            "section": ["source_probe"] * 3,
            "feature_set": ["source_probe"] * 3,
            "split": ["probe_stratified"] * 3,
            "metric": [
                "temporal_stability_ratio",
                "train_window_consistency_pass",
                "base_feature_coverage_pass",
            ],
            "k": [10, None, None],
            "value": [0.5, 1.0, 1.0],
            "sample_count": [30, 30, 30],
            "warning": [None, None, None],
            "action": ["compare", "keep", "keep"],
        }
    )

    report = path_feature_analysis_report(
        path_feature_analysis(feature_set_counts={"base": 1}, source_probe=source_probe)
    )

    assert "temporal_stability_ratio=0.5" in report
    assert "train_window_consistency_pass=1.0" in report
    assert "base_feature_coverage_pass=1.0" in report


def test_review_metrics_can_use_source_presence_calibrated_score() -> None:
    scored = pl.DataFrame(
        {
            "feature_set": ["x"] * 4,
            "split": ["all"] * 4,
            "trend_score": [0.9, 0.8, 0.2, 0.1],
            "source_presence_calibrated_score": [0.45, 0.8, 0.2, 0.1],
            "trend_excess_return": [-10.0, 3.0, 1.0, 0.0],
            "positive": [False, True, True, False],
        }
    )

    raw = path_robust_profit_metrics(scored, group_cols=("feature_set", "split"), k_values=(1,))
    calibrated = path_robust_profit_metrics(
        scored,
        group_cols=("feature_set", "split"),
        k_values=(1,),
        score_column="source_presence_calibrated_score",
    )
    deciles = path_decile_monotonicity(
        scored,
        group_cols=("feature_set", "split"),
        bucket_count=2,
        score_column="source_presence_calibrated_score",
    )

    assert raw.select("raw_mean_excess").item() == -10.0
    assert calibrated.select("raw_mean_excess").item() == 3.0
    assert deciles.select("bucket").to_series().to_list() == [0, 1]
