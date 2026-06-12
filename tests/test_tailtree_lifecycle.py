from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import qooi.scanner.diagnostics as diagnostics
import qooi.scanner.workflow as potential
from qooi.scanner import PotentialArtifacts
from qooi.scanner.config import EvidenceConfig, TailtreeConfig, TransitionConfig


class _TreeMetadata:
    categorical_features = ("source_family",)
    continuous_features = ("return_1bar",)


class _FakeTree:
    metadata = _TreeMetadata()

    def to_json(self, path: Path) -> None:
        path.write_text('{"fake": true}', encoding="utf-8")


class _Inputs:
    def __init__(self, config: potential.PotentialConfig, diagnostics_dir: Path) -> None:
        self.config = config
        self.artifacts = PotentialArtifacts(
            report=diagnostics_dir.parent / "report.md",
            diagnostics_dir=diagnostics_dir,
            states_dir=diagnostics_dir.parent / "states",
        )


def test_tailtree_lifecycle_config_loads_named_instance(tmp_path: Path) -> None:
    config_path = tmp_path / "potential.toml"
    config_path.write_text(
        """
[potential]
output = "data/output/potential/lifecycle/report.md"
bar = "1H"

[potential.transition]
mae_mfe_horizon = 12

[potential.evidence]
kind = "tailtree"

[potential.evidence.tailtree]
threshold_pct = 5.0
lifecycle = "load_predict"
model_dir = "data/output/potential/lifecycle/models"
model_tag = "tailtree-1h-12h-v1"
""",
        encoding="utf-8",
    )

    config = potential.load_config(config_path)

    assert config.evidence.kind == "tailtree"
    assert config.evidence.tailtree.lifecycle == "load_predict"
    assert config.evidence.tailtree.model_dir == Path("data/output/potential/lifecycle/models")
    assert config.evidence.tailtree.model_tag == "tailtree-1h-12h-v1"


def test_tailtree_load_predict_fails_without_frozen_artifacts(tmp_path: Path) -> None:
    config = potential.PotentialConfig(
        evidence=EvidenceConfig(
            kind="tailtree",
            tailtree=TailtreeConfig(
                lifecycle="load_predict",
                model_dir=tmp_path / "models",
                model_tag="missing-model",
            ),
        ),
    )
    observations = pl.DataFrame(
        {
            "symbol": ["BTC-USDT-SWAP"],
            "decision_bar_close_ms": [1],
            "source_family": ["market"],
        }
    )

    with pytest.raises(FileNotFoundError, match="tailtree load_predict artifact missing"):
        diagnostics._run_tailtree_pipeline(
            observations,
            pl.DataFrame(),
            pl.DataFrame(),
            _Inputs(config, tmp_path / "diagnostics"),
        )


def test_tailtree_train_lifecycle_writes_frozen_artifact_set(tmp_path: Path) -> None:
    config = potential.PotentialConfig(
        evidence=EvidenceConfig(
            kind="tailtree",
            tailtree=TailtreeConfig(
                threshold_pct=5.0,
                lifecycle="train",
                model_dir=tmp_path / "models",
                model_tag="tailtree-test-v1",
            ),
        ),
        bar="1H",
        transition=TransitionConfig(mae_mfe_horizon=12),
    )
    inputs = _Inputs(config, tmp_path / "diagnostics")
    evidence = pl.DataFrame(
        {
            "leaf_id": [1],
            "tree_direction": ["up"],
            "selected_evidence_level": [True],
        }
    )

    diagnostics._write_tailtree_artifacts(
        inputs,
        {"up": evidence},
        {"up": _FakeTree()},
    )

    root = tmp_path / "models" / "tailtree-test-v1"
    assert (root / "tail-tree-up.json").exists()
    assert (root / "potential-leaf-evidence-up.csv").exists()
    metadata = (root / "tailtree-artifact.json").read_text(encoding="utf-8")
    assert '"bar": "1H"' in metadata
    assert '"horizon_bars": 12' in metadata
    assert '"threshold_pct": 5.0' in metadata
    assert '"model_tag": "tailtree-test-v1"' in metadata
