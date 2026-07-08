from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from qooi.profiling import ProfileConfig, ProfileContext
from qooi.scanner.config import (
    Config,
    FixedTrainingConfig,
)
from qooi.scanner.path_model import GPDParams, TailTreeModel, TrainConfig, TreeMetadata
from qooi.scanner.tailrun import planning
from qooi.scanner.tailrun.core import run_tailtree_job
from qooi.scanner.tailrun.features import (
    AcceptedFeatureManifest,
    FeatureSpec,
    ProposalFeatureManifest,
    SelectSpec,
    select_manifest,
    train_candidates,
)
from qooi.scanner.tailrun.types import TailtreePreparedFrames


def _training() -> FixedTrainingConfig:
    return FixedTrainingConfig(
        num_leaves=16,
        min_data_in_leaf=10,
        learning_rate=0.1,
        num_iterations=12,
        early_stopping_rounds=5,
    )


def _observations(n: int = 60) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["BTC-USDT-SWAP"] * n,
            "decision_bar_close_ms": list(range(n)),
            "bar_return_1h_pct": [float(index % 7) for index in range(n)],
            "close": [100.0 + float(index % 17) for index in range(n)],
            "realized_volatility_6h": [float((index % 5) + 1) for index in range(n)],
            "path_pressure": [float((index % 11) - 5) for index in range(n)],
        }
    )


def _labels(n: int = 60) -> pl.DataFrame:
    labels = ([0, 1, 2, 3, 4] * (n // 5 + 1))[:n]
    return pl.DataFrame(
        {
            "symbol": ["BTC-USDT-SWAP"] * n,
            "decision_bar_close_ms": list(range(n)),
            "horizon_hours": [
                4 if index % 3 == 0 else 12 if index % 3 == 1 else 24 for index in range(n)
            ],
            "path_label": labels,
            "sample_weight": [1.0 + (labels[index] * 0.1) for index in range(n)],
            "path_reason": ["fixture"] * n,
        }
    )


def _prepared() -> TailtreePreparedFrames:
    observations = _observations()
    histories = observations.select(
        "symbol",
        pl.col("decision_bar_close_ms").alias("bar_close_ms"),
        "close",
    )
    return TailtreePreparedFrames(
        observations=observations,
        source_outcomes=pl.DataFrame(),
        realized=pl.DataFrame(),
        histories=histories,
        outcomes=pl.DataFrame(),
        labeled_outcomes=_labels(),
        categorical_features=[],
        continuous_features=["bar_return_1h_pct", "realized_volatility_6h", "path_pressure"],
    )


def test_short_feature_api_names_select_and_read_manifest(tmp_path: Path) -> None:
    spec = FeatureSpec(horizons=(4, 12, 24))
    candidates = train_candidates(
        _labels(),
        _observations().select(
            "symbol",
            "decision_bar_close_ms",
            pl.col("bar_return_1h_pct").alias("base__bar_return_1h_pct"),
            pl.col("path_pressure").alias("base__path_pressure"),
        ),
        spec=spec,
    )

    manifest = select_manifest(
        candidates,
        (),
        spec=SelectSpec(min_features=1, max_features=2),
        artifact_id="short-api",
    )
    matrix = manifest.select_matrix(candidates)
    path = tmp_path / "manifest.json"
    manifest.write(path)

    loaded = ProposalFeatureManifest.read(path)

    assert loaded == manifest
    assert set(manifest.selected_columns).issubset(matrix.columns)


def test_path_train_writes_proposal_manifest_boundary(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    tailtree = Config(
        model_dir=tmp_path / "models",
        feature_dir=feature_dir,
        outcome_horizon=(4, 12, 24),
        training=_training(),
    )
    run = planning.tailtree_fixed_run(tailtree)
    [job] = planning.tailtree_objective_jobs(run, fold_id=0, tailtree=tailtree)

    result = run_tailtree_job(
        job,
        _prepared(),
        tailtree=tailtree,
        profile=ProfileContext(ProfileConfig(), tmp_path),
    )

    manifest_path = feature_dir / "feature-manifest.proposal.json"
    review_path = feature_dir / "feature-review.csv"
    assert result.model is not None
    assert manifest_path.exists()
    assert review_path.exists()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["artifact_kind"] == "feature_manifest.proposal"
    assert data["selected_columns"] == result.model.metadata.selected_columns
    assert data["review_artifact_ids"] == ["feature-review.csv"]
    review = pl.read_csv(review_path)
    assert {
        "source_run_id",
        "feature",
        "importance_gain",
        "importance_rank",
        "selected_feature",
        "feature_manifest_id",
        "feature_schema_hash",
        "label_contract_id",
    }.issubset(review.columns)
    assert review.get_column("selected_feature").any()
    assert "action" not in result.scored_candidates.columns


def test_path_train_uses_accepted_manifest_when_configured(tmp_path: Path) -> None:
    spec = FeatureSpec(horizons=(4, 12, 24))
    candidates = train_candidates(
        _labels(),
        _observations().select(
            "symbol",
            "decision_bar_close_ms",
            pl.col("path_pressure").alias("base__path_pressure"),
        ),
        spec=spec,
    )
    manifest = select_manifest(
        candidates,
        (),
        spec=SelectSpec(min_features=1, max_features=1),
        artifact_id="accepted-fixture",
    ).accepted()
    manifest_path = tmp_path / "accepted.json"
    manifest.write(manifest_path)
    tailtree = Config(
        model_dir=tmp_path / "models",
        feature_dir=tmp_path / "features",
        manifest=manifest_path,
        outcome_horizon=(4, 12, 24),
        training=_training(),
    )
    run = planning.tailtree_fixed_run(tailtree)
    [job] = planning.tailtree_objective_jobs(run, fold_id=0, tailtree=tailtree)

    result = run_tailtree_job(
        job,
        _prepared(),
        tailtree=tailtree,
        profile=ProfileContext(ProfileConfig(), tmp_path),
    )

    assert result.model is not None
    assert result.model.metadata.selected_columns == list(manifest.selected_columns)
    assert not (tmp_path / "features" / "tailtree-path-f00_path_manifest.json").exists()


def test_path_train_reproduces_tsfresh_columns_from_accepted_manifest(tmp_path: Path) -> None:
    spec = FeatureSpec(
        horizons=(4, 12, 24),
        windows_hours=(1,),
        tsfresh_value_columns=("close",),
        tsfresh_calculators=("mean",),
    )
    manifest = AcceptedFeatureManifest(
        artifact_id="accepted-tsfresh",
        artifact_kind="feature_manifest.accepted",
        spec=spec,
        selected_columns=("tsf__close__w1h__mean",),
        candidate_feature_columns=("tsf__close__w1h__mean",),
        fold_ids=(0,),
        fit_row_count=60,
        validation_row_count=0,
        schema_hash="tsfresh-schema",
        label_column="path_label",
        label_contract_id="path_prototype",
        weight_column="sample_weight",
        selection_metric="fixture",
        created_at="2026-07-04T00:00:00+00:00",
    )
    manifest_path = tmp_path / "accepted-tsfresh.json"
    manifest.write(manifest_path)
    tailtree = Config(
        model_dir=tmp_path / "models",
        feature_dir=tmp_path / "features",
        manifest=manifest_path,
        outcome_horizon=(4, 12, 24),
        training=_training(),
    )
    run = planning.tailtree_fixed_run(tailtree)
    [job] = planning.tailtree_objective_jobs(run, fold_id=0, tailtree=tailtree)

    result = run_tailtree_job(
        job,
        _prepared(),
        tailtree=tailtree,
        profile=ProfileContext(ProfileConfig(), tmp_path),
    )

    assert result.model is not None
    assert result.model.metadata.selected_columns == ["tsf__close__w1h__mean"]


def test_path_load_predict_requires_manifest_for_tsfresh_model_metadata(tmp_path: Path) -> None:
    model_id = "tailtree-path_path"
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    TailTreeModel(
        booster="dummy",
        metadata=TreeMetadata(
            direction="path",
            num_leaves_actual=1,
            categorical_features=[],
            continuous_features=["tsf__close__w1h__mean"],
            global_baseline=GPDParams(xi=0.0, sigma=1.0, tail_rate=1.0),
            leaf_params={},
            feature_importance=[("tsf__close__w1h__mean", 1.0)],
            train_config=TrainConfig(objective="path_prototype"),
            train_timestamp="2026-07-04T00:00:00+00:00",
            train_n_observations=60,
            train_n_exceedances=60,
            num_class=5,
            selected_columns=["tsf__close__w1h__mean"],
            feature_schema_hash="schema-tsfresh",
            feature_manifest_id="accepted-tsfresh",
            label_contract_id="path_prototype",
            class_names=["calm", "smooth_up", "smooth_down", "chop", "fake_breakout"],
            class_counts={"0": 12, "1": 12, "2": 12, "3": 12, "4": 12},
            valid_n_observations=30,
        ),
    ).to_json(model_dir / f"{model_id}.json")
    tailtree = Config(
        lifecycle="load_predict",
        model_dir=model_dir,
        manifest=tmp_path / "feature-manifest.accepted.json",
        model_id=model_id,
    )
    run = planning.tailtree_predict_run(tailtree)
    [job] = planning.tailtree_objective_jobs(run, fold_id=0, tailtree=tailtree)

    with pytest.raises(FileNotFoundError, match="feature-manifest.accepted"):
        run_tailtree_job(
            job,
            _prepared(),
            tailtree=tailtree,
            profile=ProfileContext(ProfileConfig(), tmp_path),
        )
