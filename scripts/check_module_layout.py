"""Ephemeral module-layout guards for reduction slices.

This script checks retired files/import paths that are not product behavior and
therefore should not be encoded as normal pytest tests.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTIVE_ROOTS = (Path("src"), Path("tests"), Path("scripts"), Path("configs"))
TEXT_SUFFIXES = {".py", ".toml", ".yaml", ".yml"}
SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "htmlcov",
}

RETIRED_PATHS = (
    Path("src/qooi/core/plot.py"),
    Path("src/qooi/core/executor.py"),
    Path("src/qooi/core/evaluate.py"),
    Path("src/qooi/core/metrics.py"),
    Path("src/qooi/strategies"),
    Path("src/qooi/scanner/ladder.py"),
    Path("src/qooi/scanner/transitions.py"),
    Path("src/qooi/scanner/tailtree"),
    Path("src/qooi/scanner/tailrun/comparison.py"),
    Path("src/qooi/scanner/tailrun/workflow.py"),
    Path("src/qooi/scanner/tailrun/legacy_features.py"),
    Path("scripts/scanner_backfill.py"),
    Path("tests/test_backtest_executor.py"),
    Path("tests/test_classifier.py"),
    Path("tests/test_flow_pipeline.py"),
    Path("tests/test_scanner_workflow_migration.py"),
    Path("tests/test_strategy_registry.py"),
    Path("tests/test_tailtree_actionability_audit.py"),
    Path("tests/test_tailtree_boundary_anatomy.py"),
    Path("tests/test_tailtree_candidate_promoter.py"),
    Path("tests/test_tailtree_error_anatomy.py"),
    Path("tests/test_tailtree_feature_pack_stability.py"),
    Path("tests/test_tailtree_guarded_anatomy.py"),
    Path("tests/test_tailtree_local_model_spec.py"),
    Path("tests/test_tailtree_opposite_guard.py"),
    Path("tests/test_tailtree_replay_comparison.py"),
    Path("tests/test_tailtree_path_config_reduction.py"),
    Path("tests/test_tailtree_stage_entries.py"),
    Path("tests/test_tailtree_weak_path_guard.py"),
)

FORBIDDEN_SNIPPETS = (
    "from qooi.core.plot",
    "import qooi.core.plot",
    "qooi.core.plot",
    "plot_market_state_modulation_heatmap",
    "plot_market_state_horizon_decay",
    "from qooi.core.executor",
    "import qooi.core.executor",
    "qooi.core.executor",
    "from qooi.core.evaluate",
    "import qooi.core.evaluate",
    "qooi.core.evaluate",
    "from qooi.core.metrics",
    "import qooi.core.metrics",
    "qooi.core.metrics",
    "from qooi.strategies",
    "import qooi.strategies",
    "qooi.strategies",
    "from qooi.scanner.tailrun.core import train_features",
    "def train_features",
    "TailtreeFeatureRole",
    "_BASE_TAILTREE_FEATURE_SET",
    "_PROMOTER_FEATURE_SET",
    "from qooi.scanner.ladder",
    "import qooi.scanner.ladder",
    "qooi.scanner.ladder",
    "from qooi.scanner.transitions",
    "import qooi.scanner.transitions",
    "qooi.scanner.transitions",
    "from qooi.scanner.tailtree",
    "import qooi.scanner.tailtree",
    "qooi.scanner.tailtree",
    "leaf_evidence_frame",
    "score_bucket_evidence_frame",
    "selected_evidence_level",
    "tail_evidence_score",
    "ladder_candidates",
    "rank_ladder_candidates",
    "from qooi.scanner.tailrun.comparison",
    "qooi.scanner.tailrun.comparison",
    "tailtree_replay_comparison_frame",
    "tailtree_trend_entry_frontier_frame",
    "tailtree_trend_setup_summary_frame",
    "tailtree_entry_candidates_frame",
    "tailtree_pipeline_quality_frame",
    "write_tailtree_replay_comparison",
    "write_tailtree_trend_entry_frontier",
    "write_tailtree_trend_setup_summary",
    "write_tailtree_entry_candidates",
    "write_tailtree_pipeline_quality",
    "candidate_conditional_promoter_efficiency_frame",
    "LocalModelSpec",
    "_local_model_ref",
    "TailtreeCandidateGateSpec",
    "TailtreeCandidateLocalModelRef",
    "candidate_gate_frame",
    "promoter_target_frame",
    "opposite_guard_target_frame",
    "weak_path_guard_target_frame",
    "selection_error_anatomy_frame",
    "guarded_selection_error_anatomy_frame",
    "dual_guard_boundary_anatomy_frame",
    "actionability_contradiction_audit_frame",
    "decision_key_action_surface_frame",
    "feature_pack_stability_frame",
    "frontier_benchmark_frame",
    "write_tailtree_source_timeseries_features",
    "write_tailtree_feature_pack_stability",
    "write_tailtree_frontier_benchmark",
    "baseline_selection",
    "def feature_manifest",
    "def make_feature_matrix",
    "TailtreeFeatureManifestName",
    "_FEATURE_MANIFESTS",
    "qooi.scanner.tailrun.legacy_features",
    "qooi.scanner.tailrun.workflow",
    "FeatureFactoryManifest",
    "FeatureMatrixArtifact",
    "ModelManifest",
    "FeatureCache",
    "matrix_path(",
    "write_matrix(",
    "read_matrix(",
    "model_arch",
    "scanner_backfill",
    "def _symbol_frame(",
)

ALLOWLIST_PATHS = {
    Path("scripts/check_module_layout.py"),
}


def _iter_text_files() -> list[Path]:
    files: list[Path] = []
    for root in ACTIVE_ROOTS:
        base = ROOT / root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            rel = path.relative_to(ROOT)
            if any(part in SKIP_DIRS for part in rel.parts):
                continue
            if path.is_file() and path.suffix in TEXT_SUFFIXES:
                files.append(path)
    return files


def main() -> int:
    failures: list[str] = []
    for rel in RETIRED_PATHS:
        if (ROOT / rel).exists():
            failures.append(f"retired path still exists: {rel}")

    for path in _iter_text_files():
        rel = path.relative_to(ROOT)
        if rel in ALLOWLIST_PATHS:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for snippet in FORBIDDEN_SNIPPETS:
            if snippet in text:
                failures.append(f"forbidden snippet {snippet!r} in {rel}")

    if failures:
        print("module layout check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("module layout check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
