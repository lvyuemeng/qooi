from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import qooi.scanner.diagnostics as diagnostics
import qooi.scanner.tailrun as tailrun
import qooi.scanner.tailtree as tailtree
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


class _FakeLoadTree:
    metadata = _TreeMetadata()

    def __init__(self, leaf_id: int) -> None:
        self.leaf_id = leaf_id

    def predict_leaf(self, features: pl.DataFrame) -> pl.DataFrame:
        return features.with_columns(pl.lit(self.leaf_id).cast(pl.Int32).alias("leaf_id"))


class _ScoreMetadata:
    direction = "up"
    global_baseline = type("Baseline", (), {"tail_rate": 0.5, "xi": 0.1, "sigma": 1.0})()
    leaf_params = {}


class _FakeScoreTree:
    metadata = _ScoreMetadata()

    def predict_score(self, features: pl.DataFrame) -> pl.DataFrame:
        return features.with_columns(
            pl.Series("tailtree_score", [0.10, 0.90, 0.95, 0.99][: len(features)])
        )

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
outcome_horizon = [6, 12, 24]
""",
        encoding="utf-8",
    )

    config = potential.load_config(config_path)

    assert config.evidence.kind == "tailtree"
    assert config.evidence.tailtree.lifecycle == "load_predict"
    assert config.evidence.tailtree.model_dir == Path("data/output/potential/lifecycle/models")
    assert config.evidence.tailtree.model_tag == "tailtree-1h-12h-v1"
    assert config.evidence.tailtree.outcome_horizon == (6, 12, 24)


def test_tailtree_config_normalizes_outcome_horizon() -> None:
    config = TailtreeConfig(outcome_horizon=(12, 6, 12, 0, -1))
    int_config = TailtreeConfig(outcome_horizon=12)

    assert config.outcome_horizon == (12, 6)
    assert int_config.outcome_horizon == (12,)


def test_tailtree_config_loads_tail_utility_objective(tmp_path: Path) -> None:
    config_path = tmp_path / "potential.toml"
    config_path.write_text(
        """
[potential.evidence]
kind = "tailtree"

[potential.evidence.tailtree]
objective = "tail_utility_quantile"
""",
        encoding="utf-8",
    )

    config = potential.load_config(config_path)

    assert config.evidence.tailtree.objective == "tail_utility_quantile"


def test_tailtree_labels_include_path_constrained_tail_utility() -> None:
    outcome = pl.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "decision_bar_close_ms": [1, 2],
            "forward_max_return_pct": [9.0, 4.0],
            "forward_min_return_pct": [-2.0, -10.0],
            "close_retention_ratio": [0.5, 0.25],
            "path_efficiency": [0.8, 0.4],
            "time_to_max_bar": [3, 1],
            "time_to_min_bar": [1, 3],
            "post_max_drawdown_pct": [1.0, 0.0],
            "post_min_rebound_pct": [0.0, 2.0],
        }
    )

    labeled = tailtree.label_tail_exceedances(outcome, threshold_pct=5.0)

    assert "tail_utility_up" in labeled.columns
    assert "tail_utility_down" in labeled.columns
    up_utility = labeled.get_column("tail_utility_up").to_list()[0]
    down_utility = labeled.get_column("tail_utility_down").to_list()[1]
    assert up_utility > 0.0
    assert down_utility > 0.0
    assert labeled.get_column("tail_utility_up").to_list()[1] == 0.0
    assert labeled.get_column("tail_utility_down").to_list()[0] == 0.0


def test_tailtree_leaf_id_vector_uses_last_tree_for_multitree_predictions() -> None:
    leaves = tailtree._leaf_id_vector([[1, 11, 111], [2, 22, 222]])

    assert leaves.tolist() == [111, 222]


def test_score_bucket_evidence_frame_uses_full_model_scores() -> None:
    observations = pl.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "decision_bar_close_ms": [1, 2, 3, 4],
        }
    )
    outcomes = pl.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "decision_bar_close_ms": [1, 2, 3, 4],
            "tail_up": [False, True, True, True],
            "tail_utility_up": [0.0, 0.4, 0.8, 1.6],
        }
    )

    evidence = tailtree.score_bucket_evidence_frame(
        _FakeScoreTree(), observations, outcomes, score_quantiles=(0.5, 0.75)
    )

    assert evidence.select("score_bucket", "N_total", "N_tail_exceedances").to_dicts() == [
        {"score_bucket": "top_50pct", "N_total": 2, "N_tail_exceedances": 2},
        {"score_bucket": "top_25pct", "N_total": 1, "N_tail_exceedances": 1},
    ]
    assert "leaf_id" not in evidence.columns
    assert evidence.get_column("tail_lift").to_list() == pytest.approx([2.0, 2.0])
    assert evidence.get_column("tail_utility_p90").to_list()[1] == pytest.approx(1.6)


def test_tailtree_training_frame_carries_utility_values() -> None:
    observations = pl.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "decision_bar_close_ms": [1, 2, 3],
            "return_1bar": [0.1, 0.2, 0.3],
        }
    )
    labeled = pl.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "decision_bar_close_ms": [1, 2, 3],
            "tail_up": [True, False, True],
            "tail_exceedance_value_up": [2.0, None, 4.0],
            "tail_utility_up": [1.5, 0.0, 3.5],
        }
    )

    training = tailtree.tailtree_training_frame(observations, labeled, direction="up")

    assert training.utility_values.tolist() == [1.5, 3.5]


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
            pl.DataFrame(),
            _Inputs(config, tmp_path / "diagnostics"),
        )


def test_tailtree_load_predict_loads_every_configured_horizon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = potential.PotentialConfig(
        evidence=EvidenceConfig(
            kind="tailtree",
            tailtree=TailtreeConfig(
                lifecycle="load_predict",
                model_dir=tmp_path / "models",
                model_tag="tailtree-load-mh-v1",
                outcome_horizon=(6, 12),
            ),
        ),
        bar="1H",
    )
    train_inputs = _Inputs(
        config.model_copy(
            update={
                "evidence": EvidenceConfig(
                    kind="tailtree",
                    tailtree=config.evidence.tailtree.model_copy(update={"lifecycle": "train"}),
                )
            }
        ),
        tmp_path / "train-diagnostics",
    )
    for horizon in (6, 12):
        evidence = pl.DataFrame(
            {
                "leaf_id": [horizon],
                "tree_direction": ["up"],
                "outcome_horizon": [horizon],
                "selected_evidence_level": [True],
            }
        )
        tailrun._write_tailtree_artifacts(
            train_inputs,
            {"up": evidence},
            {"up": _FakeTree()},
            outcome_horizon=horizon,
            cleanup=horizon == 6,
        )

    def _from_json(path: Path) -> _FakeLoadTree:
        horizon = int(path.stem.split("-")[2].removeprefix("h"))
        return _FakeLoadTree(horizon)

    import qooi.scanner.tailtree as tailtree

    monkeypatch.setattr(tailtree.TailTreeModel, "from_json", staticmethod(_from_json))
    result = tailrun.load_predict(
        pl.DataFrame(
            {
                "symbol": ["BTC-USDT-SWAP"],
                "decision_bar_close_ms": [1],
                "source_family": ["market"],
                "return_1bar": [0.1],
            }
        ),
        _Inputs(config, tmp_path / "diagnostics"),
    )

    assert set(result.models) == {(6, "up"), (12, "up")}
    assert set(result.evidence.get_column("outcome_horizon").unique().to_list()) == {6, 12}
    assert (tmp_path / "diagnostics" / "tailtree-run-summary.csv").exists()


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

    tailrun._write_tailtree_artifacts(
        inputs,
        {"up": evidence},
        {"up": _FakeTree()},
    )

    root = tmp_path / "models" / "tailtree-test-v1"
    assert (root / "tail-tree-h12-up.json").exists()
    assert (root / "potential-leaf-evidence-h12-up.csv").exists()
    metadata = (root / "tailtree-artifact-h12.json").read_text(encoding="utf-8")
    assert '"bar": "1H"' in metadata
    assert '"outcome_horizon": 12' in metadata
    assert '"horizon_bars"' not in metadata
    assert '"threshold_pct": 5.0' in metadata
    assert '"model_tag": "tailtree-test-v1"' in metadata


def test_tailtree_train_lifecycle_writes_one_artifact_set_per_horizon(
    tmp_path: Path,
) -> None:
    config = potential.PotentialConfig(
        evidence=EvidenceConfig(
            kind="tailtree",
            tailtree=TailtreeConfig(
                threshold_pct=5.0,
                lifecycle="train",
                model_dir=tmp_path / "models",
                model_tag="tailtree-mh-v1",
                outcome_horizon=(6, 12),
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
            "outcome_horizon": [6],
            "selected_evidence_level": [True],
        }
    )
    summary = pl.concat(
        [
            tailrun._tailtree_run_summary_frame(
                config,
                observations=pl.DataFrame({"symbol": ["BTC"], "decision_bar_close_ms": [1]}),
                source_event_row_count=0,
                source_outcomes=pl.DataFrame(),
                realized_transitions=pl.DataFrame(),
                outcome_frame=pl.DataFrame({"tail_up": [True], "tail_down": [False]}),
                categorical_features=[],
                continuous_features=[],
                train_counts={"up": (1, 1), "down": (1, 0)},
                selected_leaf_counts={"up": 1, "down": 0},
                trained_tree_count=1,
                outcome_horizon=horizon,
            )
            for horizon in (6, 12)
        ],
        how="diagonal_relaxed",
    )

    tailrun._write_tailtree_artifacts(
        inputs,
        {"up": evidence},
        {"up": _FakeTree()},
        summary=summary,
        outcome_horizon=6,
    )
    tailrun._write_tailtree_artifacts(
        inputs,
        {"up": evidence.with_columns(pl.lit(12).alias("outcome_horizon"))},
        {"up": _FakeTree()},
        summary=summary,
        outcome_horizon=12,
        cleanup=False,
    )

    root = tmp_path / "models" / "tailtree-mh-v1"
    assert (root / "tail-tree-h6-up.json").exists()
    assert (root / "tail-tree-h12-up.json").exists()
    assert (root / "potential-leaf-evidence-h6-up.csv").exists()
    assert (root / "potential-leaf-evidence-h12-up.csv").exists()
    assert (root / "tailtree-artifact-h6.json").exists()
    assert (root / "tailtree-artifact-h12.json").exists()
    persisted_summary = pl.read_csv(root / "tailtree-run-summary.csv")
    assert set(persisted_summary.get_column("outcome_horizon").unique().to_list()) == {
        6,
        12,
    }
    assert not (root / "tail-tree-up.json").exists()
    assert not (root / "potential-leaf-evidence-up.csv").exists()


def test_tailtree_train_lifecycle_writes_summary_rows_per_configured_horizon(
    tmp_path: Path,
) -> None:
    config = potential.PotentialConfig(
        evidence=EvidenceConfig(
            kind="tailtree",
            tailtree=TailtreeConfig(
                threshold_pct=5.0,
                lifecycle="train",
                model_dir=tmp_path / "models",
                model_tag="tailtree-summary-mh-v1",
                outcome_horizon=(6, 12),
            ),
        ),
        bar="1H",
        transition=TransitionConfig(mae_mfe_horizon=12),
    )
    inputs = _Inputs(config, tmp_path / "diagnostics")
    observations = pl.DataFrame(
        {
            "symbol": ["BTC", "BTC"],
            "decision_timeframe": ["1H", "1H"],
            "decision_bar_close_ms": [1, 2],
            "background_regime": ["trend", "trend"],
            "swing_core": ["bull", "bull"],
            "decision_core": ["bull", "bull"],
            "decision_transition": ["same", "same"],
            "decision_direction": ["up", "up"],
            "return_1bar": [0.1, 0.2],
        }
    )
    realized_transitions = pl.DataFrame(
        {
            "symbol": ["BTC", "BTC", "BTC", "BTC"],
            "timeframe": ["1H", "1H", "1H", "1H"],
            "bar_close_ms": [1, 1, 2, 2],
            "outcome_horizon": [6, 12, 6, 12],
            "terminal_core_context": ["bull", "bull", "bull", "bull"],
            "terminal_direction": ["up", "up", "up", "up"],
            "direction_changed": [False, False, False, False],
            "returned_to_origin": [False, False, False, False],
            "transition_count": [0, 0, 0, 0],
            "forward_return_pct": [1.0, 2.0, 3.0, 4.0],
            "forward_min_return_pct": [0.0, 0.0, 0.0, 0.0],
            "forward_max_return_pct": [6.0, 7.0, 8.0, 9.0],
            "path_range_pct": [6.0, 7.0, 8.0, 9.0],
            "time_to_max_bar": [1, 1, 1, 1],
            "time_to_min_bar": [1, 1, 1, 1],
            "close_retention_ratio": [0.1, 0.2, 0.3, 0.4],
            "post_max_drawdown_pct": [5.0, 5.0, 5.0, 5.0],
            "post_min_rebound_pct": [1.0, 2.0, 3.0, 4.0],
            "path_efficiency": [0.1, 0.2, 0.3, 0.4],
        }
    )

    tailrun.train_evaluate_predict(
        observations,
        pl.DataFrame(),
        realized_transitions,
        inputs,
        source_event_row_count=0,
    )

    summary = pl.read_csv(
        tmp_path / "models" / "tailtree-summary-mh-v1" / "tailtree-run-summary.csv"
    )
    assert set(summary.get_column("outcome_horizon").unique().to_list()) == {6, 12}
    assert summary.filter(pl.col("summary_scope") == "run").height == 2


def test_tailtree_train_lifecycle_removes_stale_direction_artifacts(tmp_path: Path) -> None:
    config = potential.PotentialConfig(
        evidence=EvidenceConfig(
            kind="tailtree",
            tailtree=TailtreeConfig(
                lifecycle="train",
                model_dir=tmp_path / "models",
                model_tag="tailtree-empty-v1",
            ),
        ),
    )
    inputs = _Inputs(config, tmp_path / "diagnostics")
    root = tmp_path / "models" / "tailtree-empty-v1"
    root.mkdir(parents=True)
    inputs.artifacts.diagnostics_dir.mkdir(parents=True)
    stale_paths = [
        root / "tail-tree-h12-up.json",
        root / "potential-leaf-evidence-h12-up.csv",
        root / "tailtree-artifact-h12.json",
        inputs.artifacts.diagnostics_dir / "tail-tree-h12-up.json",
        inputs.artifacts.diagnostics_dir / "potential-leaf-evidence-h12-up.csv",
        inputs.artifacts.diagnostics_dir / "potential-leaves-selected-h12-up.csv",
    ]
    for path in stale_paths:
        path.write_text("stale", encoding="utf-8")

    tailrun._write_tailtree_artifacts(inputs, {}, {})

    assert all(not path.exists() for path in stale_paths)
    summary = pl.read_csv(root / "tailtree-run-summary.csv")
    assert summary.get_column("removed_stale_file_count").max() == len(stale_paths)
    assert summary.get_column("written_model_file_count").max() == 0
    assert summary.get_column("written_evidence_file_count").max() == 0


def test_tailtree_run_summary_records_label_and_feature_availability(tmp_path: Path) -> None:
    config = potential.PotentialConfig(
        evidence=EvidenceConfig(kind="tailtree", tailtree=TailtreeConfig(threshold_pct=5.0)),
    )
    summary = tailrun._tailtree_run_summary_frame(
        config,
        observations=pl.DataFrame(
            {
                "symbol": ["BTC", "ETH"],
                "decision_bar_close_ms": [1, 2],
                "background_regime": ["trend", "range"],
                "return_1bar": [0.1, -0.2],
            }
        ),
        source_event_row_count=0,
        source_outcomes=pl.DataFrame(),
        realized_transitions=pl.DataFrame({"symbol": ["BTC", "ETH"]}),
        outcome_frame=pl.DataFrame(
            {
                "forward_return_pct": [None, None],
                "forward_min_return_pct": [None, None],
                "forward_max_return_pct": [None, None],
                "path_range_pct": [None, None],
                "time_to_max_bar": [None, None],
                "time_to_min_bar": [None, None],
                "close_retention_ratio": [None, None],
                "path_efficiency": [None, None],
                "tail_up": [False, False],
                "tail_down": [False, False],
            }
        ),
        categorical_features=["background_regime"],
        continuous_features=["return_1bar"],
        train_counts={"up": (2, 0), "down": (2, 0)},
        selected_leaf_counts={"up": 0, "down": 0},
        quality_by_direction={
            "up": tailrun.TailtreeDirectionQuality.zero("up"),
            "down": tailrun.TailtreeDirectionQuality.zero("down"),
        },
        trained_tree_count=0,
    )

    run = summary.filter(pl.col("summary_scope") == "run").row(0, named=True)
    assert run["observation_row_count"] == 2
    assert run["feature_count"] == 2
    assert run["forward_max_return_nonnull_count"] == 0
    assert run["time_to_max_nonnull_count"] == 0
    assert run["time_to_min_nonnull_count"] == 0
    assert run["retention_nonnull_count"] == 0
    assert run["path_efficiency_nonnull_count"] == 0
    assert run["objective"] == "tail_severity_gpd"
    assert run["tail_utility_mean"] == 0.0
    assert run["tail_utility_p90"] == 0.0
    assert run["valid_selected_utility_mean"] == 0.0
    assert run["valid_selected_utility_p90"] == 0.0
    assert run["train_tail_count"] == 0
    assert run["valid_observation_count"] == 0
    assert run["valid_tail_lift"] == 0.0
    assert run["tail_count"] == 0
    up = summary.filter(pl.col("summary_scope") == "up").row(0, named=True)
    assert up["train_observation_count"] == 2
    assert up["train_exceedance_count"] == 0
    assert up["train_tail_count"] == 0
    assert up["valid_observation_count"] == 0
    assert up["valid_selected_tail_rate"] == 0.0
    assert up["trainable_flag"] == 0


def test_tailtree_direction_quality_scores_selected_validation_lift() -> None:
    quality = tailrun.TailtreeDirectionQuality.from_labeled_leaf_frame(
        direction="up",
        train_tail_count=3,
        validation_leaf_frame=pl.DataFrame(
            {
                "leaf_id": [1, 1, 2, 2],
                "tail_up": [True, False, False, False],
            }
        ),
        selected_leaf_ids={1},
    )

    assert quality.direction == "up"
    assert quality.train_tail_count == 3
    assert quality.valid_observation_count == 4
    assert quality.valid_tail_count == 1
    assert quality.valid_tail_rate == 0.25
    assert quality.valid_selected_observation_count == 2
    assert quality.valid_selected_tail_count == 1
    assert quality.valid_selected_tail_rate == 0.5
    assert quality.valid_tail_lift == 2.0


def test_tailtree_training_features_exclude_ephemeral_current_review_columns() -> None:
    observations = pl.DataFrame(
        {
            "symbol": ["BTC"],
            "decision_bar_close_ms": [1],
            "background_regime": ["trend"],
            "source_family": ["books"],
            "source_state": ["imbalanced"],
            "return_1bar": [0.1],
            "funding_rate": [0.0001],
            "oi_delta": [1.2],
            "imbalance_value": [0.7],
            "spread_bps": [4.0],
            "buy_sell_ratio": [1.1],
            "book_age_ms": [100],
            "trade_age_ms": [100],
        }
    )

    categorical, continuous = tailrun._tailtree_training_features(observations)

    assert categorical == ["background_regime"]
    assert "return_1bar" in continuous
    assert "funding_rate" in continuous
    assert "oi_delta" in continuous
    assert "source_family" not in categorical
    assert "source_state" not in categorical
    assert "imbalance_value" not in continuous
    assert "spread_bps" not in continuous
    assert "buy_sell_ratio" not in continuous
    assert "book_age_ms" not in continuous
    assert "trade_age_ms" not in continuous
