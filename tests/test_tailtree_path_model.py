from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import pytest

from qooi.scanner.path_model import (
    TailTreeModel,
    TrainConfig,
)
from qooi.scanner.tailrun.features import AcceptedFeatureManifest, FeatureManifest, FeatureSpec


def _selected_manifest() -> FeatureManifest:
    return AcceptedFeatureManifest(
        artifact_id="selected-path",
        artifact_kind="feature_manifest.accepted",
        spec=FeatureSpec(horizons=(4, 12, 24)),
        selected_columns=("ctx__horizon_hours", "base__momentum", "tsf__shape"),
        candidate_feature_columns=(
            "ctx__horizon_hours",
            "base__momentum",
            "tsf__shape",
        ),
        fold_ids=(0,),
        fit_row_count=50,
        validation_row_count=25,
        schema_hash="schema-path-123",
        label_column="path_label",
        selection_metric="train_variance",
        created_at="2026-07-04T00:00:00+00:00",
    )


def _path_matrix(rows: int, *, offset: int = 0) -> pl.DataFrame:
    symbols = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "XRP-USDT-SWAP", "ADA-USDT-SWAP"]
    data = []
    for index in range(rows):
        label = (index + offset) % 5
        horizon = (4, 12, 24)[(index + offset) % 3]
        data.append(
            {
                "symbol": symbols[(index + offset) % len(symbols)],
                "decision_bar_close_ms": 1_000_000 + (index + offset) * 60_000,
                "horizon_hours": horizon,
                "path_label": label,
                "sample_weight": 1.0 + (0.25 * label),
                "ctx__horizon_hours": float(horizon),
                "base__momentum": float(label * 2 + ((index + offset) % 7) / 10),
                "tsf__shape": float((4 - label) * 3 + ((index + offset) % 5) / 10),
            }
        )
    return pl.DataFrame(data)


def test_train_path_model_reuses_tailtree_model_metadata_and_multiclass_objective() -> None:
    manifest = _selected_manifest()
    model = TailTreeModel.train_path(
        _path_matrix(60),
        _path_matrix(30, offset=60),
        config=TrainConfig(
            objective="path_prototype",
            num_leaves=16,
            min_data_in_leaf=10,
            learning_rate=0.05,
            num_iterations=20,
            early_stopping_rounds=5,
        ),
        selected_manifest=manifest,
        label_contract_id="path_prototype",
    )

    assert isinstance(model, TailTreeModel)
    assert model.metadata.direction == "path"
    assert model.metadata.train_config.objective == "path_prototype"
    assert model.metadata.num_class == 5
    assert model.metadata.selected_columns == list(manifest.selected_columns)
    assert model.metadata.feature_schema_hash == manifest.schema_hash
    assert model.metadata.feature_manifest_id == manifest.artifact_id
    assert model.metadata.label_contract_id == "path_prototype"
    assert model.metadata.class_names == [
        "calm",
        "smooth_up",
        "smooth_down",
        "chop",
        "fake_breakout",
    ]
    assert model.metadata.train_n_observations == 60
    assert model.metadata.valid_n_observations == 30
    assert model.metadata.class_counts == {"0": 12, "1": 12, "2": 12, "3": 12, "4": 12}


def test_score_path_model_emits_probability_surface_and_round_trips_json(tmp_path: Path) -> None:
    manifest = _selected_manifest()
    model = TailTreeModel.train_path(
        _path_matrix(60),
        _path_matrix(30, offset=60),
        config=TrainConfig(
            objective="path_prototype",
            num_leaves=16,
            min_data_in_leaf=10,
            learning_rate=0.05,
            num_iterations=20,
            early_stopping_rounds=5,
        ),
        selected_manifest=manifest,
        label_contract_id="path_prototype",
    )
    path = tmp_path / "tailtree-path_path.json"
    model.to_json(path)
    loaded = TailTreeModel.from_json(path)

    scored = loaded.score_path(
        _path_matrix(9, offset=100).drop("path_label", "sample_weight"),
    )

    expected = {
        "symbol",
        "decision_bar_close_ms",
        "horizon_hours",
        "path_prob_calm",
        "path_prob_smooth_up",
        "path_prob_smooth_down",
        "path_prob_chop",
        "path_prob_fake_breakout",
        "path_pred_label",
        "path_pred_label_name",
        "path_confidence",
    }
    assert expected.issubset(set(scored.columns))
    assert scored.height == 9
    for row in scored.select(
        "path_prob_calm",
        "path_prob_smooth_up",
        "path_prob_smooth_down",
        "path_prob_chop",
        "path_prob_fake_breakout",
        "path_confidence",
    ).iter_rows(named=True):
        total = sum(float(row[column]) for column in row if column.startswith("path_prob_"))
        assert math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6)
        assert 0.0 <= float(row["path_confidence"]) <= 1.0


def test_path_model_rejects_missing_weights_labels_and_selected_features() -> None:
    manifest = _selected_manifest()
    config = TrainConfig(objective="path_prototype")

    with pytest.raises(ValueError, match="sample_weight"):
        TailTreeModel.train_path(
            _path_matrix(60).drop("sample_weight"),
            _path_matrix(30, offset=60),
            config=config,
            selected_manifest=manifest,
            label_contract_id="path_prototype",
        )
    with pytest.raises(ValueError, match="path_label"):
        TailTreeModel.train_path(
            _path_matrix(60),
            _path_matrix(30, offset=60).drop("path_label"),
            config=config,
            selected_manifest=manifest,
            label_contract_id="path_prototype",
        )

    model = TailTreeModel.train_path(
        _path_matrix(60),
        _path_matrix(30, offset=60),
        config=TrainConfig(
            objective="path_prototype",
            num_leaves=16,
            min_data_in_leaf=10,
            learning_rate=0.05,
            num_iterations=20,
            early_stopping_rounds=5,
        ),
        selected_manifest=manifest,
        label_contract_id="path_prototype",
    )
    with pytest.raises(ValueError, match="missing selected path model features"):
        model.score_path(_path_matrix(4).drop("tsf__shape"))
