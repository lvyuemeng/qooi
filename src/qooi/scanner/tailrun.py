"""Tailtree train/load_predict lifecycle boundary."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, TypedDict

import polars as pl

from qooi.scanner import frames as frames_eval

if TYPE_CHECKING:
    from qooi.scanner import ReportInputs


class TailtreeModelMetadata(Protocol):
    categorical_features: list[str]
    continuous_features: list[str]


class TailtreeArtifactTree(Protocol):
    metadata: TailtreeModelMetadata

    def predict_leaf(self, features: pl.DataFrame) -> pl.DataFrame: ...
    def to_json(self, path: Path) -> None: ...


class TailtreeArtifactMetadata(TypedDict):
    bar: str
    horizon_bars: int
    threshold_pct: float
    categorical_features: list[str]
    continuous_features: list[str]
    feature_schema_hash: str
    model_tag: str


TailtreeDirection = Literal["up", "down"]


@dataclass(frozen=True)
class TailtreeResult:
    """Tailtree path pipeline result. Every field has a concrete type."""

    evidence: pl.DataFrame
    candidates: pl.DataFrame
    ranked: pl.DataFrame
    tree_up: TailtreeArtifactTree | None
    tree_down: TailtreeArtifactTree | None
    sections: tuple


@dataclass(frozen=True)
class TailtreeEvidenceResult:
    """Tailtree evidence/model build result before candidate matching."""

    evidence: pl.DataFrame
    tree_up: TailtreeArtifactTree | None
    tree_down: TailtreeArtifactTree | None


def run(
    observations: pl.DataFrame,
    source_outcomes: pl.DataFrame,
    realized_transitions: pl.DataFrame,
    inputs: ReportInputs,
) -> TailtreeEvidenceResult:
    """Run the explicit tailtree lifecycle selected by config."""

    if inputs.config.evidence.tailtree.lifecycle == "load_predict":
        return load_predict(observations, inputs)
    return train_evaluate_predict(observations, source_outcomes, realized_transitions, inputs)


def train_evaluate_predict(
    observations: pl.DataFrame,
    source_outcomes: pl.DataFrame,
    realized_transitions: pl.DataFrame,
    inputs: ReportInputs,
) -> TailtreeEvidenceResult:
    """Train tailtree evidence, persist artifacts, and score observations."""

    return _build_tail_tree_evidence(observations, source_outcomes, realized_transitions, inputs)


def load_predict(
    observations: pl.DataFrame,
    inputs: ReportInputs,
) -> TailtreeEvidenceResult:
    """Load frozen tailtree artifacts and score current observations."""

    return _load_tail_tree_evidence(observations, inputs)


def _tailtree_model_root(inputs) -> Path:
    return (
        Path(inputs.config.evidence.tailtree.model_dir) / inputs.config.evidence.tailtree.model_tag
    )


def _tailtree_feature_schema_hash(
    categorical_features: list[str], continuous_features: list[str]
) -> str:
    payload = json.dumps(
        {
            "categorical_features": categorical_features,
            "continuous_features": continuous_features,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tailtree_artifact_metadata(
    inputs, tree_up: TailtreeArtifactTree | None, tree_down: TailtreeArtifactTree | None
) -> TailtreeArtifactMetadata:
    categorical_features: list[str] = []
    continuous_features: list[str] = []
    for tree in (tree_up, tree_down):
        if tree is None or not hasattr(tree, "metadata"):
            continue
        metadata = tree.metadata
        categorical_features = list(metadata.categorical_features)
        continuous_features = list(metadata.continuous_features)
        break
    return {
        "bar": inputs.config.bar,
        "horizon_bars": inputs.config.transition.mae_mfe_horizon,
        "threshold_pct": inputs.config.evidence.tailtree.threshold_pct,
        "categorical_features": categorical_features,
        "continuous_features": continuous_features,
        "feature_schema_hash": _tailtree_feature_schema_hash(
            categorical_features, continuous_features
        ),
        "model_tag": inputs.config.evidence.tailtree.model_tag,
    }


def _write_tailtree_artifacts(
    inputs,
    evidence_by_direction: dict[str, pl.DataFrame],
    trees: dict[TailtreeDirection, TailtreeArtifactTree],
) -> None:
    root = _tailtree_model_root(inputs)
    root.mkdir(parents=True, exist_ok=True)
    inputs.artifacts.diagnostics_dir.mkdir(parents=True, exist_ok=True)
    for direction, tree in trees.items():
        model_name = f"tail-tree-{direction}.json"
        tree.to_json(root / model_name)
        tree.to_json(inputs.artifacts.diagnostics_dir / model_name)
        evidence = evidence_by_direction.get(direction, pl.DataFrame())
        if not evidence.is_empty():
            evidence.write_csv(root / f"potential-leaf-evidence-{direction}.csv")
            evidence.write_csv(
                inputs.artifacts.diagnostics_dir / f"potential-leaf-evidence-{direction}.csv"
            )
    metadata = _tailtree_artifact_metadata(inputs, trees.get("up"), trees.get("down"))
    (root / "tailtree-artifact.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load_tail_tree_evidence(observations: pl.DataFrame, inputs) -> TailtreeEvidenceResult:
    from qooi.scanner.tailtree import TailTreeModel

    root = _tailtree_model_root(inputs)
    metadata_path = root / "tailtree-artifact.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"tailtree load_predict artifact missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "bar": inputs.config.bar,
        "horizon_bars": inputs.config.transition.mae_mfe_horizon,
        "threshold_pct": inputs.config.evidence.tailtree.threshold_pct,
        "model_tag": inputs.config.evidence.tailtree.model_tag,
    }
    mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
    if mismatches:
        details = ", ".join(
            f"{key}: artifact={metadata.get(key)!r} config={expected[key]!r}" for key in mismatches
        )
        raise ValueError(f"tailtree load_predict artifact mismatch: {details}")

    trees: dict[TailtreeDirection, TailtreeArtifactTree] = {}
    evidence_frames: list[pl.DataFrame] = []
    for direction in ("up", "down"):
        model_path = root / f"tail-tree-{direction}.json"
        evidence_path = root / f"potential-leaf-evidence-{direction}.csv"
        if not model_path.exists():
            raise FileNotFoundError(f"tailtree load_predict artifact missing: {model_path}")
        if not evidence_path.exists():
            raise FileNotFoundError(f"tailtree load_predict artifact missing: {evidence_path}")
        tree = TailTreeModel.from_json(model_path)
        missing_features = [
            column
            for column in (*tree.metadata.categorical_features, *tree.metadata.continuous_features)
            if column not in observations.columns
        ]
        if missing_features:
            raise ValueError(
                "tailtree load_predict artifact mismatch: missing observation features "
                + ", ".join(missing_features)
            )
        trees[direction] = tree
        evidence_frames.append(pl.read_csv(evidence_path))
    evidence = (
        pl.concat(evidence_frames, how="diagonal_relaxed") if evidence_frames else pl.DataFrame()
    )
    return TailtreeEvidenceResult(evidence, trees.get("up"), trees.get("down"))


def _build_tail_tree_evidence(
    observations,
    source_outcomes,
    realized_transitions,
    inputs,
) -> TailtreeEvidenceResult:
    from qooi.scanner.tailtree import (
        TailTreeModel,
        TrainConfig,
        label_tail_exceedances,
        leaf_context_frame,
        leaf_evidence_frame,
        select_tail_leaves,
        tailtree_training_frame,
    )

    logger = logging.getLogger("qooi.scanner")
    inputs.artifacts.diagnostics_dir.mkdir(parents=True, exist_ok=True)

    outcome_frame = frames_eval.potential_outcome_frame(
        observations,
        source_outcomes,
        realized_transitions,
        return_threshold_pct=inputs.config.transition.return_threshold_pct,
    )
    if outcome_frame.is_empty():
        return TailtreeEvidenceResult(pl.DataFrame(), None, None)

    outcome_frame = label_tail_exceedances(
        outcome_frame, threshold_pct=inputs.config.evidence.tailtree.threshold_pct
    )
    tail_up_count = (
        int(outcome_frame.get_column("tail_up").sum()) if "tail_up" in outcome_frame.columns else 0
    )
    tail_down_count = (
        int(outcome_frame.get_column("tail_down").sum())
        if "tail_down" in outcome_frame.columns
        else 0
    )
    logger.info(
        "outcome rows=%d tail_up=%d tail_down=%d",
        len(outcome_frame),
        tail_up_count,
        tail_down_count,
    )
    config = TrainConfig(
        num_leaves=inputs.config.evidence.tailtree.num_leaves,
        min_data_in_leaf=inputs.config.evidence.tailtree.min_data_in_leaf,
        learning_rate=inputs.config.evidence.tailtree.learning_rate,
        num_iterations=inputs.config.evidence.tailtree.num_iterations,
        early_stopping_rounds=inputs.config.evidence.tailtree.early_stopping_rounds,
    )
    cat = [
        "background_regime",
        "swing_core",
        "decision_core",
        "decision_transition",
        "decision_direction",
        "source_family",
        "source_state",
        "risk_context",
        "market_alignment",
    ]
    con = [
        "atr_percentile",
        "range_width_atr",
        "return_1bar",
        "return_4bar",
        "return_24bar",
        "vol_anomaly",
        "close_to_range_high_ratio",
        "imbalance_value",
        "spread_bps",
        "buy_sell_ratio",
        "funding_rate",
        "oi_delta",
        "taker_buy_sell_ratio",
        "long_short_ratio",
        "book_age_ms",
        "trade_age_ms",
        "funding_age_ms",
        "oi_age_ms",
        "taker_age_ms",
        "lsr_age_ms",
    ]
    cat = [c for c in cat if c in observations.columns]
    con = [c for c in con if c in observations.columns]

    all_evidence = []
    evidence_by_direction: dict[TailtreeDirection, pl.DataFrame] = {}
    trees: dict[TailtreeDirection, TailtreeArtifactTree] = {}
    for direction in ("up", "down"):
        training = tailtree_training_frame(observations, outcome_frame, direction=direction)
        if not training.has_min_exceedances(config.min_data_in_leaf):
            continue

        tree = TailTreeModel.train(
            training.tail_observations,
            training.exceedance_values,
            config=config,
            categorical_features=cat,
            continuous_features=con,
            direction=direction,
            global_tail_rate=training.global_tail_rate,
            train_n_observations=training.train_n_observations,
        )
        trees[direction] = tree

        lev = leaf_evidence_frame(tree, observations, outcome_frame)
        if lev.is_empty():
            continue
        lctx = leaf_context_frame(tree, observations, outcome_frame)
        merged = lev.join(lctx, on="leaf_id", how="left")
        selected = select_tail_leaves(merged)
        selected.write_csv(
            inputs.artifacts.diagnostics_dir / f"potential-leaves-selected-{direction}.csv"
        )
        evidence_by_direction[direction] = merged
        all_evidence.append(merged)

    _write_tailtree_artifacts(inputs, evidence_by_direction, trees)
    ev = pl.concat(all_evidence, how="diagonal_relaxed") if all_evidence else pl.DataFrame()
    return TailtreeEvidenceResult(ev, trees.get("up"), trees.get("down"))


__all__ = [
    "TailtreeEvidenceResult",
    "TailtreeResult",
    "load_predict",
    "run",
    "train_evaluate_predict",
]
