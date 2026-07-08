"""Offline path-feature research wiring for proposal manifests and review bundles."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from math import log

import polars as pl

from qooi.scanner.path_model import TailTreeModel, TrainConfig
from qooi.scanner.tailrun.features import (
    SOURCE_CONTEXT_TOKENS,
    FeatureManifest,
    FeatureSpec,
    ProposalFeatureManifest,
    SelectSpec,
    select_manifest,
)
from qooi.scanner.tailrun.review import (
    _weighted_spearman_groups,
    path_feature_blacklist,
    path_feature_importance,
    path_feature_matrix_review,
    path_feature_psi,
    path_shap_review,
)
from qooi.scanner.tailrun.types import TailtreeWalkforwardFold


class ResearchMethod(StrEnum):
    VARIANCE = "variance"
    RFECV_LGBM = "rfecv_lgbm"

    def propose(
        self,
        candidates: pl.DataFrame,
        folds: tuple[TailtreeWalkforwardFold, ...],
        *,
        spec: ResearchSpec,
        artifact_id: str,
        feature_spec: FeatureSpec | None,
    ) -> ProposalFeatureManifest:
        if self is ResearchMethod.VARIANCE:
            return select_manifest(
                candidates,
                folds,
                spec=spec.select_spec(),
                artifact_id=artifact_id,
                feature_spec=feature_spec or _feature_spec_from_candidates(candidates),
            )
        return select_rfecv_result(
            candidates,
            folds,
            spec=spec,
            artifact_id=artifact_id,
            feature_spec=feature_spec,
        ).manifest


class ResearchMetric(StrEnum):
    NEG_LOG_LOSS = "neg_log_loss"
    MACRO_F1 = "macro_f1"
    WEIGHTED_F1 = "weighted_f1"
    SPEARMAN_EXCESS = "spearman_excess"
    WEIGHTED_SPEARMAN_EXCESS = "weighted_spearman_excess"
    NDCG_EXCESS_AT_10 = "ndcg_excess_at_10"
    NDCG_EXCESS_AT_20 = "ndcg_excess_at_20"
    NDCG_EXCESS_AT_50 = "ndcg_excess_at_50"

    def ndcg_k(self) -> int | None:
        prefix = "ndcg_excess_at_"
        return int(self.value.removeprefix(prefix)) if self.value.startswith(prefix) else None

    def score(self, matrix: pl.DataFrame, scored: pl.DataFrame) -> float:
        ndcg_k = self.ndcg_k()
        if ndcg_k is not None:
            frame = _trend_excess_frame(matrix, scored, metric=self.value)
            return _ndcg_excess(frame, k=ndcg_k) if frame.height >= 2 else 0.0
        if self is ResearchMetric.WEIGHTED_SPEARMAN_EXCESS:
            frame = _trend_excess_frame(matrix, scored, metric=self.value)
            if frame.height < 2:
                return 0.0
            weighted = _weighted_spearman_groups(
                frame.select(
                    "decision_bar_close_ms",
                    "horizon_hours",
                    pl.col("trend_score").alias("score"),
                    pl.col("trend_excess_return").alias("excess_return"),
                )
            )
            return (
                float(weighted.select(pl.col("weighted_spearman").mean()).item() or 0.0)
                if not weighted.is_empty()
                else 0.0
            )
        if self is ResearchMetric.SPEARMAN_EXCESS:
            frame = _trend_excess_frame(matrix, scored, metric=self.value)
            if frame.height < 2:
                return 0.0
            value = frame.select(
                pl.corr("trend_score", "trend_excess_return", method="spearman")
            ).item()
            return float(value) if value is not None else 0.0
        return _label_metric_score(matrix, scored, metric=self)


@dataclass(frozen=True)
class ResearchSpec:
    """Offline feature-research method spec; not scanner runtime config."""

    method: ResearchMethod = ResearchMethod.RFECV_LGBM
    min_features: int = 20
    max_features: int = 80
    step: int = 5
    metric: ResearchMetric = ResearchMetric.NEG_LOG_LOSS
    label_column: str = "path_label"
    weight_column: str = "sample_weight"
    seed: int = 42
    feature_prefixes: tuple[str, ...] = (
        "ctx__",
        "base__",
        "tsf__",
        "cross__",
        "tsfsrc__",
        "crosssrc__",
    )
    excluded_feature_columns: tuple[str, ...] = ()
    mrmr_max_features: int = 60
    mrmr_corr_limit: float = 0.70
    min_non_null_rate: float = 0.95
    source_min_non_null_rate: float = 0.10
    min_unique_count: int = 3
    train_config: TrainConfig = field(
        default_factory=lambda: TrainConfig(
            objective="path_prototype",
            num_leaves=16,
            min_data_in_leaf=10,
            num_iterations=30,
            early_stopping_rounds=5,
            random_seed=42,
        )
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", ResearchMethod(self.method))
        object.__setattr__(self, "metric", ResearchMetric(self.metric))

    def feature_columns(self, candidates: pl.DataFrame) -> tuple[str, ...]:
        numeric_dtypes = {
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
            pl.Float32,
            pl.Float64,
        }
        excluded = set(self.excluded_feature_columns)
        columns = []
        for column, dtype in zip(candidates.columns, candidates.dtypes, strict=True):
            if column in excluded:
                continue
            if not column.startswith(self.feature_prefixes) or dtype not in numeric_dtypes:
                continue
            summary = candidates.select(
                pl.col(column).is_not_null().mean().alias("non_null_rate"),
                pl.col(column).n_unique().alias("unique_count"),
            ).row(0, named=True)
            threshold = (
                self.source_min_non_null_rate
                if self._is_source_feature(column)
                else self.min_non_null_rate
            )
            if (
                float(summary["non_null_rate"] or 0.0) >= threshold
                and int(summary["unique_count"] or 0) >= self.min_unique_count
            ):
                columns.append(column)
        return tuple(columns)

    def _is_source_feature(self, column: str) -> bool:
        return column.startswith(("tsfsrc__", "crosssrc__")) or (
            column.startswith(("base__", "ctx__"))
            and any(token in column for token in SOURCE_CONTEXT_TOKENS)
        )

    def select_spec(self) -> SelectSpec:
        return SelectSpec(
            min_features=self.min_features,
            max_features=self.max_features,
            label_column=self.label_column,
            weight_column=self.weight_column,
            feature_prefixes=self.feature_prefixes,
        )

    def selection_metric_name(self) -> str:
        return f"rfecv_lgbm_tailtree_path_{self.metric.value}"

    def ndcg_k(self) -> int | None:
        return self.metric.ndcg_k()


@dataclass(frozen=True)
class ReviewBundle:
    """Grouped offline review tables for a proposed or accepted manifest."""

    importance: pl.DataFrame
    shap: pl.DataFrame
    blacklist: pl.DataFrame
    psi: pl.DataFrame
    drift_alert: bool = False


@dataclass(frozen=True)
class FeatureSelectionResult:
    """Native RFECV result with the proposal manifest and per-round review."""

    manifest: ProposalFeatureManifest
    rfecv_review: pl.DataFrame
    model: TailTreeModel | None = None


@dataclass(frozen=True)
class FeatureResearchResult:
    """Complete script-01 research bundle; acceptance remains a manual step."""

    manifest: ProposalFeatureManifest
    candidate_matrix: pl.DataFrame
    feature_review: pl.DataFrame
    rfecv_review: pl.DataFrame
    review: ReviewBundle
    model: TailTreeModel


def _schema_hash(frame: pl.DataFrame) -> str:
    import hashlib

    parts = [f"{name}:{dtype}" for name, dtype in zip(frame.columns, frame.dtypes, strict=True)]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _fold_filter(
    frame: pl.DataFrame, fold: TailtreeWalkforwardFold, *, valid: bool
) -> pl.DataFrame:
    window = fold.valid_window if valid else fold.train_window
    return frame.filter(
        (pl.col("decision_bar_close_ms") >= int(window.start_ms))
        & (pl.col("decision_bar_close_ms") < int(window.end_ms))
    )


def _feature_spec_from_candidates(candidates: pl.DataFrame) -> FeatureSpec:
    if "horizon_hours" not in candidates.columns:
        return FeatureSpec(horizons=())
    return FeatureSpec(
        horizons=tuple(
            int(item) for item in candidates.get_column("horizon_hours").unique().sort().to_list()
        )
    )


def _manifest_for_columns(
    candidates: pl.DataFrame,
    folds: tuple[TailtreeWalkforwardFold, ...],
    columns: tuple[str, ...],
    *,
    spec: ResearchSpec,
    artifact_id: str,
    feature_spec: FeatureSpec | None,
    selection_metric: str,
) -> ProposalFeatureManifest:
    train_rows = pl.concat([_fold_filter(candidates, fold, valid=False) for fold in folds])
    valid_rows = pl.concat([_fold_filter(candidates, fold, valid=True) for fold in folds])
    selected_frame = candidates.select("symbol", "decision_bar_close_ms", "horizon_hours", *columns)
    return ProposalFeatureManifest(
        artifact_id=artifact_id,
        spec=feature_spec or _feature_spec_from_candidates(candidates),
        selected_columns=columns,
        candidate_feature_columns=spec.feature_columns(candidates),
        fold_ids=tuple(int(fold.fold_id) for fold in folds),
        fit_row_count=train_rows.height,
        validation_row_count=valid_rows.height,
        schema_hash=_schema_hash(selected_frame),
        label_column=spec.label_column,
        label_contract_id="path_prototype",
        weight_column=spec.weight_column,
        selection_metric=selection_metric,
        created_at=datetime.now(UTC).isoformat(),
    )


def research_candidates(
    observations: pl.DataFrame,
    histories: pl.DataFrame,
    labels: pl.DataFrame,
    *,
    spec: FeatureSpec,
) -> pl.DataFrame:
    """Build offline research candidates with financial base and real tsfresh features."""
    return spec.train_frame(observations, histories, labels)


def propose_manifest(
    candidates: pl.DataFrame,
    folds: tuple[TailtreeWalkforwardFold, ...],
    *,
    spec: ResearchSpec,
    artifact_id: str,
    feature_spec: FeatureSpec | None = None,
) -> ProposalFeatureManifest:
    """Propose a feature manifest using the requested offline research method."""
    return spec.method.propose(
        candidates,
        folds,
        spec=spec,
        artifact_id=artifact_id,
        feature_spec=feature_spec,
    )


def select_rfecv(
    candidates: pl.DataFrame,
    folds: tuple[TailtreeWalkforwardFold, ...],
    *,
    spec: ResearchSpec,
    artifact_id: str,
    feature_spec: FeatureSpec | None = None,
) -> ProposalFeatureManifest:
    """Select path features with architecture-native TailTreeModel RFECV."""
    return select_rfecv_result(
        candidates,
        folds,
        spec=spec,
        artifact_id=artifact_id,
        feature_spec=feature_spec,
    ).manifest


def _mrmr_prefilter(
    candidates: pl.DataFrame,
    folds: tuple[TailtreeWalkforwardFold, ...],
    columns: tuple[str, ...],
    *,
    spec: ResearchSpec,
) -> tuple[str, ...]:
    if spec.mrmr_max_features <= 0 or len(columns) <= spec.mrmr_max_features:
        return columns
    try:
        from sklearn.feature_selection import f_classif
    except ImportError as exc:  # pragma: no cover - feature-research env dependent
        raise RuntimeError("ANOVA prefilter requires `uv sync --group feature-research`") from exc

    train = pl.concat([_fold_filter(candidates, fold, valid=False) for fold in folds])
    if train.is_empty():
        raise ValueError("MRMR prefilter requires non-empty train-fold rows")
    import numpy as np

    x = train.select(*columns).to_numpy().astype(float)
    medians = np.nanmedian(np.where(np.isfinite(x), x, np.nan), axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    x = np.where(np.isfinite(x), x, medians)
    y = train.get_column(spec.label_column).cast(pl.Int64).to_numpy()
    scores, _pvalues = f_classif(x, y)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    ordered = [
        column
        for column, _score in sorted(
            zip(columns, scores, strict=True), key=lambda item: (-float(item[1]), item[0])
        )
    ]
    selected: list[str] = []
    for column in ordered:
        if len(selected) >= spec.mrmr_max_features:
            break
        redundant = False
        for kept in selected:
            corr = train.select(pl.corr(column, kept)).item()
            if corr is not None and abs(float(corr)) > spec.mrmr_corr_limit:
                redundant = True
                break
        if not redundant:
            selected.append(column)
    if len(selected) < spec.min_features:
        selected = ordered[: spec.min_features]
    return tuple(selected)


def select_rfecv_result(
    candidates: pl.DataFrame,
    folds: tuple[TailtreeWalkforwardFold, ...],
    *,
    spec: ResearchSpec,
    artifact_id: str,
    feature_spec: FeatureSpec | None = None,
) -> FeatureSelectionResult:
    """Recursive feature elimination using TailTreeModel.train_path for every fit."""
    if not folds:
        raise ValueError("rfecv_lgbm requires walk-forward folds")
    if spec.max_features < spec.min_features:
        raise ValueError("max_features must be >= min_features")
    for column in (
        "symbol",
        "decision_bar_close_ms",
        "horizon_hours",
        spec.label_column,
        spec.weight_column,
    ):
        if column not in candidates.columns:
            raise ValueError(f"candidates missing required column: {column}")
    candidate_columns = spec.feature_columns(candidates)
    if not candidate_columns:
        raise ValueError("candidates contain no selectable feature columns")

    current = _mrmr_prefilter(candidates, folds, candidate_columns, spec=spec)
    rounds: list[dict[str, object]] = []
    best: tuple[float, tuple[str, ...], TailTreeModel | None] | None = None
    round_index = 0
    while len(current) >= spec.min_features:
        manifest = _manifest_for_columns(
            candidates,
            folds,
            current,
            spec=spec,
            artifact_id=f"{artifact_id}-r{round_index:03d}",
            feature_spec=feature_spec,
            selection_metric=spec.selection_metric_name(),
        )
        fold_scores: list[float] = []
        importances: dict[str, float] = {column: 0.0 for column in current}
        model: TailTreeModel | None = None
        for fold in folds:
            train = _fold_filter(candidates, fold, valid=False)
            valid = _fold_filter(candidates, fold, valid=True)
            if train.is_empty() or valid.is_empty():
                continue
            train_matrix = manifest.select_matrix(train)
            valid_matrix = manifest.select_matrix(valid)
            model = TailTreeModel.train_path(
                train_matrix,
                valid_matrix,
                config=spec.train_config,
                selected_manifest=manifest,
                label_contract_id="path_prototype",
            )
            scored = model.score_path(valid_matrix)
            fold_scores.append(spec.metric.score(valid, scored))
            for feature, gain in model.metadata.feature_importance:
                importances[feature] = importances.get(feature, 0.0) + float(gain)
        if not fold_scores:
            raise ValueError(
                "rfecv_lgbm requires walk-forward folds with non-empty train/valid rows"
            )
        score = sum(fold_scores) / len(fold_scores)
        rounds.append(
            {
                "round": round_index,
                "feature_count": len(current),
                "score": score,
                "metric": spec.metric,
                "selected_feature_count": len(current),
                "fold_count": len(fold_scores),
                "features": ",".join(current),
            }
        )
        if len(current) <= spec.max_features and (best is None or score > best[0]):
            best = (score, current, model)
        if len(current) == spec.min_features:
            break
        drop_count = min(max(1, int(spec.step)), len(current) - spec.min_features)
        weakest = {
            feature
            for feature, _gain in sorted(importances.items(), key=lambda item: (item[1], item[0]))[
                :drop_count
            ]
        }
        current = tuple(feature for feature in current if feature not in weakest)
        round_index += 1

    if best is None:
        raise ValueError("rfecv_lgbm produced no selectable feature round")
    selected = best[1]
    manifest = _manifest_for_columns(
        candidates,
        folds,
        selected,
        spec=spec,
        artifact_id=artifact_id,
        feature_spec=feature_spec,
        selection_metric=spec.selection_metric_name(),
    )
    review = pl.DataFrame(rounds).sort("round") if rounds else pl.DataFrame()
    final_model = train_probe_model(candidates, folds, manifest=manifest, spec=spec)
    return FeatureSelectionResult(manifest=manifest, rfecv_review=review, model=final_model)


def train_probe_model(
    candidates: pl.DataFrame,
    folds: tuple[TailtreeWalkforwardFold, ...],
    *,
    manifest: FeatureManifest,
    spec: ResearchSpec,
) -> TailTreeModel:
    """Train one architecture-native review model for gain/SHAP artifacts."""
    train = pl.concat([_fold_filter(candidates, fold, valid=False) for fold in folds])
    valid = pl.concat([_fold_filter(candidates, fold, valid=True) for fold in folds])
    if train.is_empty():
        raise ValueError("probe model requires non-empty train rows")
    if valid.is_empty():
        valid = train
    return TailTreeModel.train_path(
        manifest.select_matrix(train),
        manifest.select_matrix(valid),
        config=spec.train_config,
        selected_manifest=manifest,
        label_contract_id="path_prototype",
    )


def _trend_excess_frame(matrix: pl.DataFrame, scored: pl.DataFrame, *, metric: str) -> pl.DataFrame:
    if "final_return" not in matrix.columns:
        raise ValueError(f"{metric} requires final_return in validation rows")
    return (
        matrix.select("symbol", "decision_bar_close_ms", "horizon_hours", "final_return")
        .join(
            scored.select(
                "symbol",
                "decision_bar_close_ms",
                "horizon_hours",
                "path_prob_smooth_up",
                "path_prob_smooth_down",
            ),
            on=("symbol", "decision_bar_close_ms", "horizon_hours"),
            how="inner",
        )
        .with_columns(
            pl.col("final_return")
            .median()
            .over("decision_bar_close_ms", "horizon_hours")
            .alias("market_return"),
            pl.max_horizontal("path_prob_smooth_up", "path_prob_smooth_down").alias("trend_score"),
        )
        .with_columns(
            pl.when(pl.col("path_prob_smooth_up") >= pl.col("path_prob_smooth_down"))
            .then(pl.col("final_return") - pl.col("market_return"))
            .otherwise(pl.col("market_return") - pl.col("final_return"))
            .alias("trend_excess_return")
        )
        .drop_nulls(["trend_score", "trend_excess_return"])
    )


def _ndcg_excess(frame: pl.DataFrame, *, k: int) -> float:
    gains = frame.with_columns(
        (pl.col("trend_excess_return").clip(0.0, None) + 1.0).log().alias("gain")
    )
    if gains.filter(pl.col("gain") > 0.0).is_empty():
        return 0.0
    discount = (pl.col("rank") + 2).cast(pl.Float64).log() / log(2.0)
    dcg = (
        gains.sort("trend_score", descending=True)
        .head(k)
        .with_row_index("rank")
        .select((pl.col("gain") / discount).sum())
        .item()
    )
    ideal = (
        gains.sort("gain", descending=True)
        .head(k)
        .with_row_index("rank")
        .select((pl.col("gain") / discount).sum())
        .item()
    )
    return float(dcg / ideal) if ideal else 0.0


def _label_metric_score(
    matrix: pl.DataFrame, scored: pl.DataFrame, *, metric: ResearchMetric
) -> float:
    prob_columns = (
        "path_prob_calm",
        "path_prob_smooth_up",
        "path_prob_smooth_down",
        "path_prob_chop",
        "path_prob_fake_breakout",
    )
    joined = matrix.select("symbol", "decision_bar_close_ms", "horizon_hours", "path_label").join(
        scored.select(
            "symbol",
            "decision_bar_close_ms",
            "horizon_hours",
            "path_pred_label",
            *prob_columns,
        ),
        on=("symbol", "decision_bar_close_ms", "horizon_hours"),
        how="inner",
    )
    if joined.is_empty():
        return 0.0
    if metric is ResearchMetric.NEG_LOG_LOSS:
        probability = (
            pl.when(pl.col("path_label") == 0)
            .then(pl.col("path_prob_calm"))
            .when(pl.col("path_label") == 1)
            .then(pl.col("path_prob_smooth_up"))
            .when(pl.col("path_label") == 2)
            .then(pl.col("path_prob_smooth_down"))
            .when(pl.col("path_label") == 3)
            .then(pl.col("path_prob_chop"))
            .otherwise(pl.col("path_prob_fake_breakout"))
        )
        return float(
            joined.select(pl.max_horizontal(probability, pl.lit(1e-15)).log().mean()).item()
        )
    classes = pl.DataFrame({"class_index": [0, 1, 2, 3, 4]})
    true_counts = joined.group_by(pl.col("path_label").alias("class_index")).agg(
        pl.len().alias("support")
    )
    pred_counts = joined.group_by(pl.col("path_pred_label").alias("class_index")).agg(
        pl.len().alias("predicted")
    )
    true_positive = (
        joined.filter(pl.col("path_label") == pl.col("path_pred_label"))
        .group_by(pl.col("path_label").alias("class_index"))
        .agg(pl.len().alias("tp"))
    )
    per_class = (
        classes.join(true_counts, on="class_index", how="left")
        .join(pred_counts, on="class_index", how="left")
        .join(true_positive, on="class_index", how="left")
        .with_columns(
            pl.col("support").fill_null(0),
            pl.col("predicted").fill_null(0),
            pl.col("tp").fill_null(0),
        )
        .with_columns(
            ((2 * pl.col("tp")) / (pl.col("support") + pl.col("predicted")))
            .fill_nan(0.0)
            .fill_null(0.0)
            .alias("f1")
        )
    )
    if metric is ResearchMetric.MACRO_F1:
        return float(per_class.select(pl.col("f1").mean()).item() or 0.0)
    if metric is ResearchMetric.WEIGHTED_F1:
        total = joined.height
        return float(
            per_class.select((pl.col("f1") * pl.col("support")).sum() / total).item() or 0.0
        )
    raise ValueError(f"unsupported RFECV metric: {metric.value}")


def run_feature_research(
    candidates: pl.DataFrame,
    folds: tuple[TailtreeWalkforwardFold, ...],
    *,
    spec: ResearchSpec,
    artifact_id: str,
    feature_spec: FeatureSpec | None = None,
    run_id: str = "feature-research",
    max_shap_rows: int = 1000,
) -> FeatureResearchResult:
    """Run native RFECV and model review for script-01 feature proposals."""
    selection = select_rfecv_result(
        candidates,
        folds,
        spec=spec,
        artifact_id=artifact_id,
        feature_spec=feature_spec,
    )
    feature_columns = spec.feature_columns(candidates)
    feature_review = path_feature_matrix_review(
        candidates,
        feature_columns=feature_columns,
        run_id=run_id,
    )
    train_matrix = selection.manifest.select_matrix(
        pl.concat([_fold_filter(candidates, fold, valid=False) for fold in folds])
    )
    valid_frame = pl.concat([_fold_filter(candidates, fold, valid=True) for fold in folds])
    recent_matrix = selection.manifest.select_matrix(
        valid_frame if not valid_frame.is_empty() else candidates
    )
    review = review_manifest(
        train_matrix,
        recent_matrix,
        selection.model
        if selection.model is not None
        else train_probe_model(candidates, folds, manifest=selection.manifest, spec=spec),
        manifest=selection.manifest,
        run_id=run_id,
        max_shap_rows=max_shap_rows,
    )
    model = (
        selection.model
        if selection.model is not None
        else train_probe_model(candidates, folds, manifest=selection.manifest, spec=spec)
    )
    return FeatureResearchResult(
        manifest=selection.manifest,
        candidate_matrix=candidates,
        feature_review=feature_review,
        rfecv_review=selection.rfecv_review,
        review=review,
        model=model,
    )


def review_manifest(
    train_matrix: pl.DataFrame,
    recent_matrix: pl.DataFrame,
    model: TailTreeModel,
    *,
    manifest: FeatureManifest,
    run_id: str,
    max_shap_rows: int = 1000,
) -> ReviewBundle:
    """Review a path model and manifest without accepting or mutating config."""
    importance = path_feature_importance(model, run_id=run_id)
    shap = pl.concat(
        (
            path_shap_review(
                model,
                train_matrix,
                selected_columns=manifest.selected_columns,
                run_id=run_id,
                sample_scope="train",
                max_rows=max_shap_rows,
            ),
            path_shap_review(
                model,
                recent_matrix,
                selected_columns=manifest.selected_columns,
                run_id=run_id,
                sample_scope="recent",
                max_rows=max_shap_rows,
            ),
        ),
        how="vertical",
    )
    psi = path_feature_psi(
        train_matrix,
        recent_matrix,
        selected_columns=manifest.selected_columns,
        run_id=run_id,
    )
    train_shap = shap.filter(pl.col("sample_scope") == "train")
    blacklist = path_feature_blacklist(importance, train_shap, run_id=run_id)
    drift_alert = bool(psi.filter(pl.col("psi") >= 0.2).height)
    return ReviewBundle(
        importance=importance,
        shap=shap,
        blacklist=blacklist,
        psi=psi,
        drift_alert=drift_alert,
    )


__all__ = [
    "FeatureResearchResult",
    "FeatureSelectionResult",
    "ResearchMethod",
    "ResearchMetric",
    "ResearchSpec",
    "ReviewBundle",
    "propose_manifest",
    "research_candidates",
    "review_manifest",
    "run_feature_research",
    "select_rfecv",
    "select_rfecv_result",
    "train_probe_model",
]
