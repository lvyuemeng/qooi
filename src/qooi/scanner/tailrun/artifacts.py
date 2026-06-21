"""Tailtree artifact IO and load-predict validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

from qooi.scanner.tailrun.types import (
    TAILTREE_RUN_SUMMARY_SCHEMA,
    ReportInputs,
    TailtreeArtifactMetadata,
    TailtreeArtifactTree,
    TailtreeDirection,
    TailtreeEvidenceResult,
    TailtreeProfileFeedback,
)


@dataclass(frozen=True)
class TailtreeArtifactContext:
    """Resolved filesystem/config fields for one tailtree artifact set."""

    root: Path
    diagnostics_dir: Path
    bar: str
    threshold_pct: float
    model_tag: str
    horizons: tuple[int, ...]

    @classmethod
    def from_inputs(cls, inputs: ReportInputs) -> TailtreeArtifactContext:
        tailtree = inputs.config.evidence.tailtree
        bars = inputs.config.bars
        bar = bars.timeframes[0] if bars is not None and bars.timeframes else "1H"
        first_profile = tailtree.profiles[0]
        return cls(
            root=Path(tailtree.model_dir) / first_profile.model_tag,
            diagnostics_dir=inputs.artifacts.diagnostics_dir,
            bar=bar,
            threshold_pct=tailtree.threshold_pct,
            model_tag=first_profile.model_tag,
            horizons=tailtree.outcome_horizon,
        )

    @property
    def candidate_horizon(self) -> int:
        for horizon in self.horizons:
            return horizon
        msg = "tailtree outcome_horizon must contain at least one positive horizon"
        raise ValueError(msg)

    def suffix(self, outcome_horizon: int) -> str:
        return f"h{int(outcome_horizon)}"

    def ensure_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)


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


def write_tailtree_profile_runs(
    output_dir: Path, rows: tuple[TailtreeProfileFeedback, ...]
) -> None:
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([asdict(row) for row in rows]).write_csv(output_dir / "tailtree-profile-runs.csv")


def write_tailtree_selection_efficiency(
    output_dir: Path,
    model_dir: Path,
    frame: pl.DataFrame,
) -> None:
    if frame.is_empty():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    frame.write_csv(output_dir / "tailtree-selection-efficiency.csv")
    frame.write_csv(model_dir / "tailtree-selection-efficiency.csv")


def write_tailtree_action_surface(output_dir: Path, frame: pl.DataFrame) -> None:
    if frame.is_empty():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.write_csv(output_dir / "tailtree-action-surface.csv")


def write_tailtree_label_distribution(output_dir: Path, frame: pl.DataFrame) -> None:
    if frame.is_empty():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.write_csv(output_dir / "tailtree-label-distribution.csv")


def _tailtree_artifact_metadata(
    context: TailtreeArtifactContext,
    tree_up: TailtreeArtifactTree | None,
    tree_down: TailtreeArtifactTree | None,
    *,
    outcome_horizon: int | None = None,
    categorical_features: list[str] | None = None,
    continuous_features: list[str] | None = None,
) -> TailtreeArtifactMetadata:
    categorical_features = list(categorical_features or [])
    continuous_features = list(continuous_features or [])
    for tree in (tree_up, tree_down):
        if tree is None or not hasattr(tree, "metadata"):
            continue
        metadata = tree.metadata
        categorical_features = list(metadata.categorical_features)
        continuous_features = list(metadata.continuous_features)
        break
    return {
        "bar": context.bar,
        "outcome_horizon": int(outcome_horizon or context.candidate_horizon),
        "threshold_pct": context.threshold_pct,
        "categorical_features": categorical_features,
        "continuous_features": continuous_features,
        "feature_schema_hash": _tailtree_feature_schema_hash(
            categorical_features, continuous_features
        ),
        "model_tag": context.model_tag,
        "trained_tree_count": int(tree_up is not None) + int(tree_down is not None),
    }


def _tailtree_artifact_paths(context: TailtreeArtifactContext) -> list[Path]:
    paths: list[Path] = []
    for horizon in context.horizons:
        suffix = context.suffix(horizon)
        paths.append(context.root / f"tailtree-artifact-{suffix}.json")
        for direction in ("up", "down"):
            paths.extend(
                [
                    context.root / f"tail-tree-{suffix}-{direction}.json",
                    context.root / f"potential-leaf-evidence-{suffix}-{direction}.csv",
                    context.root / f"potential-score-bucket-evidence-{suffix}-{direction}.csv",
                    context.diagnostics_dir / f"tail-tree-{suffix}-{direction}.json",
                    context.diagnostics_dir / f"potential-leaf-evidence-{suffix}-{direction}.csv",
                    context.diagnostics_dir
                    / f"potential-score-bucket-evidence-{suffix}-{direction}.csv",
                    context.diagnostics_dir / f"potential-leaves-selected-{suffix}-{direction}.csv",
                ]
            )
    for direction in ("up", "down"):
        paths.extend(
            [
                context.root / f"tail-tree-{direction}.json",
                context.root / f"potential-leaf-evidence-{direction}.csv",
                context.root / f"potential-score-bucket-evidence-{direction}.csv",
                context.root / "tailtree-artifact.json",
                context.diagnostics_dir / f"tail-tree-{direction}.json",
                context.diagnostics_dir / f"potential-leaf-evidence-{direction}.csv",
                context.diagnostics_dir / f"potential-score-bucket-evidence-{direction}.csv",
                context.diagnostics_dir / f"potential-leaves-selected-{direction}.csv",
            ]
        )
    return paths


def _cleanup_tailtree_artifacts(inputs: ReportInputs) -> int:
    context = TailtreeArtifactContext.from_inputs(inputs)
    context.ensure_dirs()
    removed = 0
    for path in _tailtree_artifact_paths(context):
        if path.exists():
            path.unlink()
            removed += 1
    return removed


def _empty_tailtree_summary() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                name: "run"
                if name == "summary_scope"
                else ""
                if name in {"direction", "objective"}
                else 0.0
                if name in {"threshold_pct", "tail_rate"}
                else 0
                for name in TAILTREE_RUN_SUMMARY_SCHEMA
            }
        ],
        schema=TAILTREE_RUN_SUMMARY_SCHEMA,
    )


def _evidence_artifact_name(suffix: str, direction: str, evidence: pl.DataFrame) -> str:
    family = "score-bucket" if "score_bucket" in evidence.columns else "leaf"
    return f"potential-{family}-evidence-{suffix}-{direction}.csv"


def _write_tailtree_artifacts(
    inputs: ReportInputs,
    evidence_by_direction: dict[str, pl.DataFrame],
    trees: dict[TailtreeDirection, TailtreeArtifactTree],
    *,
    summary: pl.DataFrame | None = None,
    outcome_horizon: int | None = None,
    removed_stale_file_count: int | None = None,
    cleanup: bool = True,
    categorical_features: list[str] | None = None,
    continuous_features: list[str] | None = None,
) -> None:
    context = TailtreeArtifactContext.from_inputs(inputs)
    context.ensure_dirs()
    removed_count = (
        _cleanup_tailtree_artifacts(inputs) if cleanup else int(removed_stale_file_count or 0)
    )
    written_model_file_count = 0
    written_evidence_file_count = 0
    horizon = int(outcome_horizon or context.candidate_horizon)
    suffix = context.suffix(horizon)
    for direction, tree in trees.items():
        model_name = f"tail-tree-{suffix}-{direction}.json"
        tree.to_json(context.root / model_name)
        tree.to_json(context.diagnostics_dir / model_name)
        written_model_file_count += 2
        evidence = evidence_by_direction.get(direction, pl.DataFrame())
        if not evidence.is_empty():
            evidence_name = _evidence_artifact_name(suffix, direction, evidence)
            evidence.write_csv(context.root / evidence_name)
            evidence.write_csv(context.diagnostics_dir / evidence_name)
            written_evidence_file_count += 2
    metadata = _tailtree_artifact_metadata(
        context,
        trees.get("up"),
        trees.get("down"),
        outcome_horizon=horizon,
        categorical_features=categorical_features,
        continuous_features=continuous_features,
    )
    if trees:
        (context.root / f"tailtree-artifact-{suffix}.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    summary = summary if summary is not None else _empty_tailtree_summary()
    summary = summary.with_columns(
        pl.lit(written_model_file_count).alias("written_model_file_count"),
        pl.lit(written_evidence_file_count).alias("written_evidence_file_count"),
        pl.lit(removed_count).alias("removed_stale_file_count"),
    )
    summary.write_csv(context.root / "tailtree-run-summary.csv")
    summary.write_csv(context.diagnostics_dir / "tailtree-run-summary.csv")


def _load_tail_tree_evidence(
    observations: pl.DataFrame, inputs: ReportInputs
) -> TailtreeEvidenceResult:
    from qooi.scanner.tailtree import TailTreeModel

    context = TailtreeArtifactContext.from_inputs(inputs)
    models: dict[tuple[int, TailtreeDirection], TailtreeArtifactTree] = {}
    evidence_frames: list[pl.DataFrame] = []
    for horizon in context.horizons:
        suffix = context.suffix(horizon)
        metadata_path = context.root / f"tailtree-artifact-{suffix}.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"tailtree load_predict artifact missing: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "bar": context.bar,
            "outcome_horizon": horizon,
            "threshold_pct": context.threshold_pct,
            "model_tag": context.model_tag,
        }
        mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
        if mismatches:
            details = ", ".join(
                f"{key}: artifact={metadata.get(key)!r} config={expected[key]!r}"
                for key in mismatches
            )
            raise ValueError(f"tailtree load_predict artifact mismatch: {details}")

        loaded_horizon_model_count = 0
        for direction in ("up", "down"):
            model_path = context.root / f"tail-tree-{suffix}-{direction}.json"
            leaf_evidence_path = context.root / f"potential-leaf-evidence-{suffix}-{direction}.csv"
            score_evidence_path = (
                context.root / f"potential-score-bucket-evidence-{suffix}-{direction}.csv"
            )
            existing_evidence_paths = [
                path for path in (leaf_evidence_path, score_evidence_path) if path.exists()
            ]
            if model_path.exists() != bool(existing_evidence_paths):
                missing_path = leaf_evidence_path if model_path.exists() else model_path
                raise FileNotFoundError(f"tailtree load_predict artifact missing: {missing_path}")
            if len(existing_evidence_paths) > 1:
                raise ValueError(
                    "tailtree load_predict artifact mismatch: multiple evidence buckets "
                    f"for {suffix}-{direction}"
                )
            if not model_path.exists():
                continue
            evidence_path = existing_evidence_paths[0]
            tree = TailTreeModel.from_json(model_path)
            missing_features = [
                column
                for column in (
                    *tree.metadata.categorical_features,
                    *tree.metadata.continuous_features,
                )
                if column not in observations.columns
            ]
            if missing_features:
                raise ValueError(
                    "tailtree load_predict artifact mismatch: missing observation features "
                    + ", ".join(missing_features)
                )
            evidence = pl.read_csv(evidence_path)
            if "outcome_horizon" not in evidence.columns:
                raise ValueError(
                    "tailtree load_predict artifact mismatch: "
                    f"{evidence_path} missing outcome_horizon"
                )
            evidence_horizons = set(evidence.get_column("outcome_horizon").unique().to_list())
            if evidence_horizons != {horizon}:
                raise ValueError(
                    "tailtree load_predict artifact mismatch: "
                    f"{evidence_path} outcome_horizon={sorted(evidence_horizons)!r} "
                    f"config={horizon!r}"
                )
            models[(horizon, direction)] = tree
            evidence_frames.append(evidence)
            loaded_horizon_model_count += 1
        if loaded_horizon_model_count == 0:
            raise FileNotFoundError(
                f"tailtree load_predict artifact missing: no models for outcome_horizon={horizon}"
            )
    evidence = (
        pl.concat(evidence_frames, how="diagonal_relaxed") if evidence_frames else pl.DataFrame()
    )
    summary_path = context.root / "tailtree-run-summary.csv"
    if summary_path.exists():
        context.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        (context.diagnostics_dir / "tailtree-run-summary.csv").write_text(
            summary_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
    return TailtreeEvidenceResult(evidence, models)
