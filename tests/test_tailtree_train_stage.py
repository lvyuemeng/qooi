from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl
import pytest

from qooi.scanner.path_model import (
    FixedSearchProvenance,
    GPDParams,
    OptunaSearchProvenance,
    TailTreeModel,
    TrainConfig,
    TreeMetadata,
)
from qooi.scanner.tailrun.features import (
    AcceptedFeatureManifest,
    FeatureSpec,
    ProposalFeatureManifest,
)


def _load_train_script():
    path = Path("scripts/02_train.py")
    spec = importlib.util.spec_from_file_location("tailtree_train_stage", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest() -> AcceptedFeatureManifest:
    return AcceptedFeatureManifest(
        artifact_id="accepted-test",
        spec=FeatureSpec(horizons=(4, 12, 24)),
        selected_columns=("ctx__horizon_hours", "base__momentum", "tsf__shape"),
        candidate_feature_columns=("ctx__horizon_hours", "base__momentum", "tsf__shape"),
        fold_ids=(0,),
        fit_row_count=20,
        validation_row_count=10,
        schema_hash="schema",
        label_column="path_label",
        label_contract_id="path_prototype",
        selection_metric="test",
    )


def _matrix(rows: int = 180) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"] * (rows // 3),
            "decision_bar_close_ms": [1_700_000_000_000 + i * 86_400_000 for i in range(rows)],
            "horizon_hours": [(4, 12, 24)[i % 3] for i in range(rows)],
            "path_label": [i % 5 for i in range(rows)],
            "sample_weight": [1.0 + (i % 5) * 0.1 for i in range(rows)],
            "final_return": [float((i % 7) - 3) for i in range(rows)],
            "ctx__horizon_hours": [float((4, 12, 24)[i % 3]) for i in range(rows)],
            "base__momentum": [float(i % 11) for i in range(rows)],
            "tsf__shape": [float((rows - i) % 13) for i in range(rows)],
        }
    )


def test_train_stage_reads_feature_matrix_not_live_workflow() -> None:
    train = _load_train_script()
    text = Path("scripts/02_train.py").read_text(encoding="utf-8")

    assert train.FEATURE_MATRIX_PATH.name == "features_full.parquet"
    assert "prepare_potential_run" not in text
    assert "qooi.scanner.workflow" not in text
    assert "argparse" not in text


def test_train_stage_removes_retired_output_dirs(tmp_path, monkeypatch) -> None:
    train = _load_train_script()
    legacy = tmp_path / "path-tailtree"
    legacy.mkdir()
    (legacy / "model-analysis.md").write_text("stale", encoding="utf-8")
    monkeypatch.setattr(train, "LEGACY_OUTPUT_DIRS", (legacy,))

    train.remove_legacy_outputs()

    assert not legacy.exists()


def test_train_feature_set_builds_walkforward_folds_and_label_distribution() -> None:
    train = _load_train_script()
    feature_set = train.TrainFeatureSet(_matrix(), _manifest(), train.EVALUATION)
    folds = feature_set.folds()
    labels = feature_set.label_distribution()

    assert folds
    assert set(labels.columns) == {"path_label", "row_count", "row_rate"}
    assert labels.select(pl.col("row_count").sum()).item() == 180


def test_analysis_report_is_csv_block_without_markdown_table_noise() -> None:
    train = _load_train_script()
    review = pl.DataFrame(
        {
            "trial_id": ["path-prototype-fixed-t0001"],
            "run_id": ["path-prototype-fixed-t0001"],
            "score": [0.75],
            "seconds": [1.2],
        }
    )
    analysis = train.model_analysis_frame(
        review,
        label_distribution=pl.DataFrame(
            {"path_label": [0, 1], "row_count": [10, 5], "row_rate": [0.67, 0.33]}
        ),
    )
    report = train.model_analysis_markdown(analysis)

    assert analysis.columns == [
        "section",
        "metric",
        "value",
        "trial_id",
        "run_id",
        "warning",
        "action",
    ]
    assert "```csv" in report
    assert "| metric |" not in report


def test_selected_model_copy_embeds_optuna_search_provenance(tmp_path: Path) -> None:
    train = _load_train_script()
    source = tmp_path / "source.json"
    target = tmp_path / "target.json"
    model = TailTreeModel(
        booster="fake booster",
        metadata=TreeMetadata(
            direction="path",
            num_leaves_actual=1,
            categorical_features=[],
            continuous_features=["ctx__horizon_hours"],
            global_baseline=GPDParams(xi=0.0, sigma=1.0, tail_rate=1.0),
            leaf_params={},
            feature_importance=[],
            train_config=TrainConfig(objective="path_prototype"),
            train_timestamp="2026-07-07T00:00:00+00:00",
            train_n_observations=10,
            train_n_exceedances=10,
            search=FixedSearchProvenance(),
        ),
    )
    model.to_json(source)

    train.write_selected_model(
        source,
        target,
        trial_number=7,
        score=1.25,
        seed=42,
        study_name="tailtree-path-walkforward",
    )

    loaded = TailTreeModel.from_json(target)
    assert isinstance(loaded.metadata.search, OptunaSearchProvenance)
    assert loaded.metadata.search.trial_number == 7
    assert loaded.metadata.search.score == 1.25
    assert loaded.metadata.search.study_name == "tailtree-path-walkforward"


def test_train_stage_config_uses_bounded_optuna_walkforward_feedback_paths() -> None:
    train = _load_train_script()

    assert train.FEATURE_DIR == Path("data/output/potential/path")
    assert train.REVIEW_DIR == train.FEATURE_DIR / "review"
    assert train.MODEL_DIR == train.FEATURE_DIR / "models"
    assert train.TRAINING.kind == "optuna"
    assert train.TRAINING.max_trials == 5
    assert train.TRAINING.num_leaves_range == (16, 96)
    assert train.TRAINING.min_data_in_leaf_range == (20, 120)
    assert train.EVALUATION.protocol == "walkforward"
    assert train.EVALUATION.max_folds == 4
    assert train.MODEL_ANALYSIS_PATH.name == "model-analysis.csv"
    assert train.MODEL_ANALYSIS_REPORT_PATH.name == "model-analysis.md"
    assert train.LABEL_DISTRIBUTION_PATH.name == "label-distribution.csv"


def test_board_utility_scores_use_calibrated_surface_and_ewma() -> None:
    train = _load_train_script()
    rows = [
        {
            "calibrated_decile_tau": 0.9,
            "calibrated_spread_at_10": 1.2,
            "calibrated_ndcg_at_10": 0.4,
            "top10_source_any_rate": 1.0,
        },
        {
            "calibrated_decile_tau": 0.2,
            "calibrated_spread_at_10": -0.5,
            "calibrated_ndcg_at_10": 0.1,
            "top10_source_any_rate": 0.0,
        },
    ]

    mean_score, ewma_score, min_score = train.add_board_utility_scores(rows)

    assert 0.0 <= mean_score <= 1.0
    assert 0.0 <= ewma_score <= 1.0
    assert min_score == min(row["fold_board_utility_score"] for row in rows)
    assert rows[1]["board_utility_penalty"] == pytest.approx(0.3)
    assert {"normalized_decile_tau", "normalized_spread_at_10", "normalized_ndcg_at_10"} <= set(
        rows[0]
    )


def test_board_utility_raw_metrics_penalizes_source_missing_head() -> None:
    train = _load_train_script()
    matrix = pl.DataFrame(
        {
            "symbol": [f"S{i}" for i in range(12)],
            "decision_bar_close_ms": [1] * 12,
            "horizon_hours": [24] * 12,
            "final_return": [float(i) for i in range(12)],
            "base__source_any_present": [0.0] * 10 + [1.0, 1.0],
        }
    )
    scored = pl.DataFrame(
        {
            "symbol": [f"S{i}" for i in range(12)],
            "decision_bar_close_ms": [1] * 12,
            "horizon_hours": [24] * 12,
            "path_prob_smooth_up": [0.9 - i * 0.01 for i in range(12)],
            "path_prob_smooth_down": [0.05] * 12,
            "path_prob_chop": [0.01] * 12,
            "path_prob_fake_breakout": [0.01] * 12,
        }
    )

    metrics = train.board_utility_raw_metrics(matrix, scored)
    rows = [{**metrics}]
    train.add_board_utility_scores(rows)

    assert metrics["top10_source_any_rate"] < 0.5
    assert rows[0]["board_utility_penalty"] >= 0.1
    assert "calibrated_ndcg_at_10" in metrics


def test_model_analysis_reports_board_utility_metrics() -> None:
    train = _load_train_script()
    review = pl.DataFrame(
        {
            "trial_id": ["path-prototype-fixed-t0001"],
            "run_id": ["path-prototype-fixed-t0001-f00"],
            "trial_score": [0.33],
            "score": [0.33],
            "seconds": [1.2],
            "ewma_board_utility_score": [0.72],
            "calibrated_spread_at_10": [0.8],
            "calibrated_decile_tau": [0.9],
        }
    )

    analysis = train.model_analysis_frame(
        review,
        label_distribution=pl.DataFrame(
            {"path_label": [0, 1], "row_count": [10, 5], "row_rate": [0.67, 0.33]}
        ),
    )

    metrics = set(analysis.get_column("metric"))
    assert "best_board_utility_score" in metrics
    assert "best_calibrated_spread_at_10" in metrics
    assert "best_calibrated_decile_tau" in metrics


def test_train_stage_uses_board_utility_as_active_optuna_objective() -> None:
    text = Path("scripts/02_train.py").read_text(encoding="utf-8")

    assert "return ewma_score, rows" in text
    assert '"best_metric": "ewma_board_utility_score"' in text
    assert '"legacy_ndcg_at_10": float(fold_row["score"])' in text


def test_train_stage_auto_accepts_latest_proposal_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = _load_train_script()
    proposal_path = tmp_path / "feature-manifest.proposal.json"
    accepted_path = tmp_path / "feature-manifest.accepted.json"
    proposal = ProposalFeatureManifest(
        **_manifest().model_dump(exclude={"artifact_kind", "acceptance"})
    )
    proposal.write(proposal_path)
    monkeypatch.setattr(train, "PROPOSAL_MANIFEST_PATH", proposal_path)
    monkeypatch.setattr(train, "ACCEPTED_MANIFEST_PATH", accepted_path)

    path = train.accepted_manifest_path(_matrix())

    accepted = AcceptedFeatureManifest.read(path)
    assert path == accepted_path
    assert accepted.acceptance.accepted_by == "scripts/02_train.py"
    assert accepted.acceptance.accepted_from_checksum == proposal.checksum()


def test_train_stage_refreshes_stale_accepted_manifest_when_columns_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = _load_train_script()
    proposal_path = tmp_path / "feature-manifest.proposal.json"
    accepted_path = tmp_path / "feature-manifest.accepted.json"
    stale = _manifest().model_copy(
        update={"selected_columns": ("ctx__horizon_hours", "missing_feature")}
    )
    proposal = ProposalFeatureManifest(
        **_manifest().model_dump(exclude={"artifact_kind", "acceptance"})
    )
    stale.write(accepted_path)
    proposal.write(proposal_path)
    monkeypatch.setattr(train, "PROPOSAL_MANIFEST_PATH", proposal_path)
    monkeypatch.setattr(train, "ACCEPTED_MANIFEST_PATH", accepted_path)

    path = train.accepted_manifest_path(_matrix())

    accepted = AcceptedFeatureManifest.read(path)
    assert accepted.selected_columns == proposal.selected_columns
    assert accepted.acceptance.accepted_from_checksum == proposal.checksum()
