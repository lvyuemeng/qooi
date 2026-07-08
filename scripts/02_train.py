"""Train the accepted path feature matrix into one prediction artifact."""

import asyncio
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import polars as pl

from qooi.scanner.config import OptunaTrainingConfig, RunContract, WalkforwardEvaluationConfig
from qooi.scanner.path_model import OptunaSearchProvenance, TailTreeModel, TrainConfig
from qooi.scanner.tailrun.features import AcceptedFeatureManifest, ProposalFeatureManifest
from qooi.scanner.tailrun.planning import TailtreeWalkforwardSpec, tailtree_fold_specs
from qooi.scanner.tailrun.research import ResearchMetric
from qooi.scanner.tailrun.types import TailtreeWalkforwardFold

OUTPUT_DIR = Path("data/output/potential/path")
FEATURE_DIR = OUTPUT_DIR
REVIEW_DIR = FEATURE_DIR / "review"
MODEL_DIR = FEATURE_DIR / "models"
FEATURE_MATRIX_PATH = FEATURE_DIR / "features_full.parquet"
ACCEPTED_MANIFEST_PATH = FEATURE_DIR / "feature-manifest.accepted.json"
PROPOSAL_MANIFEST_PATH = FEATURE_DIR / "feature-manifest.proposal.json"
MODEL_PATH = MODEL_DIR / "tailtree-path_path.json"
PROFILE_RUNS_PATH = FEATURE_DIR / "optuna-review.csv"
MODEL_REVIEW_PATH = FEATURE_DIR / "model-review.json"
MODEL_ANALYSIS_PATH = REVIEW_DIR / "model-analysis.csv"
MODEL_ANALYSIS_REPORT_PATH = REVIEW_DIR / "model-analysis.md"
LABEL_DISTRIBUTION_PATH = REVIEW_DIR / "label-distribution.csv"
LEGACY_OUTPUT_DIRS = (
    Path("data/output/potential/path-train"),
    Path("data/output/potential/path-predict"),
    Path("data/output/potential/path/tailtree"),
)
STUDY_NAME = "tailtree-path-walkforward"
CONTRACT = RunContract.profile("h24_swing", (4, 12, 24))
TRAINING = OptunaTrainingConfig(
    max_trials=5,
    seed=42,
    num_leaves=48,
    min_data_in_leaf=40,
    learning_rate=0.04,
    num_iterations=260,
    early_stopping_rounds=25,
    num_leaves_range=(16, 96),
    min_data_in_leaf_range=(20, 120),
    learning_rate_range=(0.015, 0.09),
    num_iterations_range=(160, 420),
    early_stopping_rounds_range=(10, 45),
)
EVALUATION = WalkforwardEvaluationConfig(
    train_days=90,
    valid_days=21,
    step_days=21,
    max_folds=4,
    embargo_bars=24,
)
BOARD_UTILITY_WEIGHTS = {"decile_tau": 0.50, "spread": 0.30, "ndcg": 0.20}
BOARD_UTILITY_HALF_LIFE_FOLDS = 2.0
BOARD_UTILITY_NDCG_TOLERANCE = 0.03
ANALYSIS_SCHEMA = {
    "section": pl.Utf8,
    "metric": pl.Utf8,
    "value": pl.Float64,
    "trial_id": pl.Utf8,
    "run_id": pl.Utf8,
    "warning": pl.Utf8,
    "action": pl.Utf8,
}


def _median(values: list[float]) -> float:
    clean = sorted(float(v) for v in values if v is not None)
    if not clean:
        return 0.0
    mid = len(clean) // 2
    return clean[mid] if len(clean) % 2 else (clean[mid - 1] + clean[mid]) / 2.0


def _iqr(values: list[float]) -> float:
    clean = sorted(float(v) for v in values if v is not None)
    if len(clean) < 4:
        return 0.0

    def quantile(q: float) -> float:
        pos = (len(clean) - 1) * q
        lower = int(pos)
        upper = min(lower + 1, len(clean) - 1)
        weight = pos - lower
        return clean[lower] * (1.0 - weight) + clean[upper] * weight

    return quantile(0.75) - quantile(0.25)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _normalise_metric(value: float, history: list[float], *, fallback: str) -> float:
    spread = _iqr(history)
    if spread > 1e-9:
        z_score = _clip((float(value) - _median(history)) / spread, -3.0, 3.0)
        return (z_score + 3.0) / 6.0
    if fallback == "tau":
        return _clip((float(value) + 1.0) / 2.0, 0.0, 1.0)
    if fallback == "spread":
        reference = max(1.0, _median([abs(v) for v in history]), abs(float(value)))
        return (_clip(float(value) / reference, -1.0, 1.0) + 1.0) / 2.0
    return _clip(float(value), 0.0, 1.0)


def _ndcg_for_score(frame: pl.DataFrame, *, score_column: str, k: int = 10) -> float:
    gains = frame.with_columns(
        (pl.col("trend_excess_return").clip(0.0, None) + 1.0).log().alias("gain")
    )
    if gains.filter(pl.col("gain") > 0.0).is_empty():
        return 0.0
    discount = (pl.col("rank") + 2).cast(pl.Float64).log() / 0.6931471805599453
    dcg = (
        gains.sort(score_column, descending=True)
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


def _calibrated_trend_frame(matrix: pl.DataFrame, scored: pl.DataFrame) -> pl.DataFrame:
    source_expr = (
        pl.col("base__source_any_present")
        if "base__source_any_present" in matrix.columns
        else pl.lit(1.0)
    )
    return (
        matrix.select(
            "symbol",
            "decision_bar_close_ms",
            "horizon_hours",
            "final_return",
            source_expr.alias("base__source_any_present"),
        )
        .join(
            scored.select(
                "symbol",
                "decision_bar_close_ms",
                "horizon_hours",
                "path_prob_smooth_up",
                "path_prob_smooth_down",
                "path_prob_chop",
                "path_prob_fake_breakout",
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
            (
                pl.col("trend_score")
                * (0.5 + 0.5 * pl.col("base__source_any_present").fill_null(0.0).clip(0.0, 1.0))
            ).alias("source_presence_calibrated_score"),
            pl.when(pl.col("path_prob_smooth_up") >= pl.col("path_prob_smooth_down"))
            .then(pl.col("final_return") - pl.col("market_return"))
            .otherwise(pl.col("market_return") - pl.col("final_return"))
            .alias("trend_excess_return"),
        )
        .with_columns(
            pl.col("trend_excess_return").abs().alias("abs_trend_excess_return"),
            (pl.col("trend_excess_return") > 0.0).alias("positive"),
        )
        .drop_nulls(["source_presence_calibrated_score", "trend_excess_return"])
    )


def board_utility_raw_metrics(matrix: pl.DataFrame, scored: pl.DataFrame) -> dict[str, float]:
    frame = _calibrated_trend_frame(matrix, scored)
    if frame.is_empty():
        return {
            "calibrated_spread_at_10": 0.0,
            "calibrated_precision_at_10": 0.0,
            "calibrated_decile_tau": 0.0,
            "calibrated_ndcg_at_10": 0.0,
            "top10_source_any_rate": 0.0,
        }
    score_column = "source_presence_calibrated_score"
    p05, p95 = frame.select(
        pl.col("trend_excess_return").quantile(0.05).alias("p05"),
        pl.col("trend_excess_return").quantile(0.95).alias("p95"),
    ).row(0)
    top = frame.sort(score_column, descending=True).head(min(10, frame.height))
    bottom = frame.sort(score_column).head(min(10, frame.height))
    top_w = top.select(pl.col("trend_excess_return").clip(p05, p95).mean()).item() or 0.0
    bottom_w = bottom.select(pl.col("trend_excess_return").clip(p05, p95).mean()).item() or 0.0
    bucketed = (
        frame.with_columns(
            ((pl.col(score_column).rank(method="ordinal", descending=True) - 1) * 10 / pl.len())
            .floor()
            .cast(pl.Int64)
            .clip(0, 9)
            .alias("bucket")
        )
        .group_by("bucket")
        .agg(pl.col("trend_excess_return").mean().alias("mean_excess_return"))
        .sort("bucket")
    )
    means = bucketed.get_column("mean_excess_return").to_list()
    signs = [
        1.0 if left > right else -1.0 if left < right else 0.0
        for i, left in enumerate(means)
        for right in means[i + 1 :]
    ]
    return {
        "calibrated_spread_at_10": float(top_w - bottom_w),
        "calibrated_precision_at_10": float(top.select(pl.col("positive").mean()).item() or 0.0),
        "calibrated_decile_tau": float(sum(signs) / len(signs)) if signs else 0.0,
        "calibrated_ndcg_at_10": _ndcg_for_score(frame, score_column=score_column, k=10),
        "top10_source_any_rate": float(
            top.select(pl.col("base__source_any_present").fill_null(0.0).mean()).item() or 0.0
        ),
    }


def add_board_utility_scores(rows: list[dict[str, object]]) -> tuple[float, float, float]:
    if not rows:
        return 0.0, 0.0, 0.0
    tau_history = [float(row["calibrated_decile_tau"]) for row in rows]
    spread_history = [float(row["calibrated_spread_at_10"]) for row in rows]
    ndcg_history = [float(row["calibrated_ndcg_at_10"]) for row in rows]
    fold_scores = []
    for row in rows:
        n_tau = _normalise_metric(float(row["calibrated_decile_tau"]), tau_history, fallback="tau")
        n_spread = _normalise_metric(
            float(row["calibrated_spread_at_10"]), spread_history, fallback="spread"
        )
        n_ndcg = _normalise_metric(
            float(row["calibrated_ndcg_at_10"]), ndcg_history, fallback="ndcg"
        )
        penalty = 0.0
        if float(row["top10_source_any_rate"]) < 0.50:
            penalty += 0.10
        if float(row["calibrated_spread_at_10"]) <= 0.0:
            penalty += 0.20
        score = (
            BOARD_UTILITY_WEIGHTS["decile_tau"] * n_tau
            + BOARD_UTILITY_WEIGHTS["spread"] * n_spread
            + BOARD_UTILITY_WEIGHTS["ndcg"] * n_ndcg
            - penalty
        )
        row["normalized_decile_tau"] = n_tau
        row["normalized_spread_at_10"] = n_spread
        row["normalized_ndcg_at_10"] = n_ndcg
        row["board_utility_penalty"] = penalty
        row["fold_board_utility_score"] = score
        fold_scores.append(score)
    mean_score = sum(fold_scores) / len(fold_scores)
    weights = [
        0.5 ** ((len(fold_scores) - 1 - idx) / BOARD_UTILITY_HALF_LIFE_FOLDS)
        for idx in range(len(fold_scores))
    ]
    ewma_score = sum(
        score * weight for score, weight in zip(fold_scores, weights, strict=True)
    ) / sum(weights)
    min_score = min(fold_scores)
    for row in rows:
        row["mean_board_utility_score"] = mean_score
        row["ewma_board_utility_score"] = ewma_score
        row["min_fold_board_utility_score"] = min_score
    return mean_score, ewma_score, min_score


@dataclass(frozen=True)
class TrainFeatureSet:
    matrix: pl.DataFrame
    manifest: AcceptedFeatureManifest
    evaluation: WalkforwardEvaluationConfig

    def folds(self) -> tuple[TailtreeWalkforwardFold, ...]:
        return tailtree_fold_specs(
            TailtreeWalkforwardSpec(
                train_days=self.evaluation.train_days,
                valid_days=self.evaluation.valid_days,
                step_days=self.evaluation.step_days,
                max_folds=self.evaluation.max_folds,
                embargo_bars=self.evaluation.embargo_bars,
            ),
            observations=self.matrix,
            bar=CONTRACT.decision_timeframe,
        )

    def train_valid(self, fold: TailtreeWalkforwardFold) -> tuple[pl.DataFrame, pl.DataFrame]:
        timestamp = pl.col("decision_bar_close_ms")
        train = self.matrix.filter(
            (timestamp >= fold.train_window.start_ms) & (timestamp < fold.train_window.end_ms)
        )
        valid = self.matrix.filter(
            (timestamp >= fold.valid_window.start_ms) & (timestamp < fold.valid_window.end_ms)
        )
        return self.manifest.select_matrix(train), self.manifest.select_matrix(valid)

    def final_train_valid(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        fold = self.folds()[-1]
        timestamp = pl.col("decision_bar_close_ms")
        train = self.matrix.filter(timestamp < fold.valid_window.start_ms)
        valid = self.matrix.filter(
            (timestamp >= fold.valid_window.start_ms) & (timestamp < fold.valid_window.end_ms)
        )
        return self.manifest.select_matrix(train), self.manifest.select_matrix(valid)

    def score(self, config: TrainConfig) -> tuple[float, list[dict[str, object]]]:
        rows: list[dict[str, object]] = []
        for fold in self.folds():
            started = perf_counter()
            train, valid = self.train_valid(fold)
            model = TailTreeModel.train_path(
                train,
                valid,
                config=config,
                selected_manifest=self.manifest,
                label_contract_id=self.manifest.label_contract_id,
            )
            scored = model.score_path(valid)
            score = ResearchMetric.NDCG_EXCESS_AT_10.score(valid, scored)
            metrics = board_utility_raw_metrics(valid, scored)
            rows.append(
                {
                    "fold_id": fold.fold_id,
                    "score": score,
                    **metrics,
                    "seconds": perf_counter() - started,
                    "train_rows": train.height,
                    "valid_rows": valid.height,
                }
            )
        _, ewma_score, _ = add_board_utility_scores(rows)
        return ewma_score, rows

    def label_distribution(self) -> pl.DataFrame:
        total = max(1, self.matrix.height)
        return (
            self.matrix.group_by("path_label")
            .len("row_count")
            .sort("path_label")
            .with_columns((pl.col("row_count") / total).alias("row_rate"))
        )

    def train_final(self, config: TrainConfig) -> TailTreeModel:
        train, valid = self.final_train_valid()
        return TailTreeModel.train_path(
            train,
            valid,
            config=config,
            selected_manifest=self.manifest,
            label_contract_id=self.manifest.label_contract_id,
        )


def train_config(params: object) -> TrainConfig:
    return TrainConfig(
        objective="path_prototype",
        num_leaves=params.num_leaves,
        min_data_in_leaf=params.min_data_in_leaf,
        learning_rate=params.learning_rate,
        num_iterations=params.num_iterations,
        early_stopping_rounds=params.early_stopping_rounds,
    )


def trial_number(trial_id: str) -> int:
    suffix = trial_id.rsplit("-t", 1)[-1]
    digits = "".join(char for char in suffix if char.isdigit())
    return int(digits or 0)


def optuna_review(feature_set: TrainFeatureSet) -> tuple[pl.DataFrame, TrainConfig, float, int]:
    optuna = TRAINING.module()
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=TRAINING.seed),
    )
    rows: list[dict[str, object]] = []

    def objective(trial: object) -> float:
        params = TRAINING.suggest(trial)
        config = train_config(params)
        score, fold_rows = feature_set.score(config)
        trial_id = f"path-prototype-fixed-t{trial.number:04d}"
        for fold_row in fold_rows:
            rows.append(
                {
                    "trial_id": trial_id,
                    "run_id": f"{trial_id}-f{int(fold_row['fold_id']):02d}",
                    "trial_score": score,
                    "score": float(fold_row["score"]),
                    "legacy_ndcg_at_10": float(fold_row["score"]),
                    "calibrated_spread_at_10": float(fold_row["calibrated_spread_at_10"]),
                    "calibrated_precision_at_10": float(fold_row["calibrated_precision_at_10"]),
                    "calibrated_decile_tau": float(fold_row["calibrated_decile_tau"]),
                    "calibrated_ndcg_at_10": float(fold_row["calibrated_ndcg_at_10"]),
                    "top10_source_any_rate": float(fold_row["top10_source_any_rate"]),
                    "normalized_decile_tau": float(fold_row["normalized_decile_tau"]),
                    "normalized_spread_at_10": float(fold_row["normalized_spread_at_10"]),
                    "normalized_ndcg_at_10": float(fold_row["normalized_ndcg_at_10"]),
                    "board_utility_penalty": float(fold_row["board_utility_penalty"]),
                    "fold_board_utility_score": float(fold_row["fold_board_utility_score"]),
                    "mean_board_utility_score": float(fold_row["mean_board_utility_score"]),
                    "ewma_board_utility_score": float(fold_row["ewma_board_utility_score"]),
                    "min_fold_board_utility_score": float(fold_row["min_fold_board_utility_score"]),
                    "seconds": float(fold_row["seconds"]),
                    "train_rows": int(fold_row["train_rows"]),
                    "valid_rows": int(fold_row["valid_rows"]),
                    "num_leaves": config.num_leaves,
                    "min_data_in_leaf": config.min_data_in_leaf,
                    "learning_rate": config.learning_rate,
                    "num_iterations": config.num_iterations,
                    "early_stopping_rounds": config.early_stopping_rounds,
                }
            )
        return score

    study.optimize(
        objective,
        n_trials=int(os.environ.get("QOOI_TRAIN_MAX_TRIALS", TRAINING.max_trials)),
        show_progress_bar=False,
    )
    best = study.best_trial
    best_params = TRAINING.suggest(best)
    return (
        pl.DataFrame(rows).sort(["trial_score", "score"], descending=True),
        train_config(best_params),
        float(best.value),
        int(best.number),
    )


def model_analysis_frame(review: pl.DataFrame, *, label_distribution: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    if not review.is_empty():
        best = review.row(0, named=True)
        rows.extend(
            [
                [
                    "best_run",
                    "best_score",
                    best["trial_score"] if "trial_score" in review.columns else best["score"],
                    best["trial_id"],
                    best["run_id"],
                    "",
                    "promote_selected_model",
                ],
                [
                    "best_run",
                    "best_board_utility_score",
                    best.get("ewma_board_utility_score"),
                    best["trial_id"],
                    best["run_id"],
                    "",
                    "compare_with_legacy_ndcg",
                ],
                [
                    "best_run",
                    "best_calibrated_spread_at_10",
                    best.get("calibrated_spread_at_10"),
                    best["trial_id"],
                    best["run_id"],
                    "negative_spread"
                    if float(best.get("calibrated_spread_at_10") or 0.0) <= 0.0
                    else "",
                    "reject_if_gate_fails"
                    if float(best.get("calibrated_spread_at_10") or 0.0) <= 0.0
                    else "keep",
                ],
                [
                    "best_run",
                    "best_calibrated_decile_tau",
                    best.get("calibrated_decile_tau"),
                    best["trial_id"],
                    best["run_id"],
                    "weak_monotonicity"
                    if float(best.get("calibrated_decile_tau") or 0.0) < 0.60
                    else "",
                    "reject_if_gate_fails"
                    if float(best.get("calibrated_decile_tau") or 0.0) < 0.60
                    else "keep",
                ],
                [
                    "best_run",
                    "best_seconds",
                    best["seconds"],
                    best["trial_id"],
                    best["run_id"],
                    "",
                    "watch_training_cost",
                ],
                ["trial_summary", "trial_fold_rows", review.height, "", "", "", "keep_if_stable"],
            ]
        )
    if not label_distribution.is_empty():
        rows.extend(
            [
                [
                    "label_distribution",
                    "label_count_total",
                    label_distribution.select(pl.col("row_count").sum()).item(),
                    "",
                    "",
                    "",
                    "coverage_recorded",
                ],
                [
                    "label_distribution",
                    "label_count_min_rate",
                    label_distribution.select(pl.col("row_rate").min()).item(),
                    "",
                    "",
                    "",
                    "inspect_sparse_labels",
                ],
            ]
        )
    return pl.DataFrame(rows, schema=ANALYSIS_SCHEMA, orient="row")


def model_analysis_markdown(analysis: pl.DataFrame) -> str:
    return "# Tailtree model analysis\n\n```csv\n" + analysis.write_csv() + "```\n"


def remove_legacy_outputs() -> None:
    for path in LEGACY_OUTPUT_DIRS:
        shutil.rmtree(path, ignore_errors=True)


def write_selected_model(
    source_model: Path,
    target_model: Path,
    *,
    trial_number: int,
    score: float,
    seed: int,
    study_name: str,
) -> None:
    model = TailTreeModel.from_json(source_model)
    target_model.parent.mkdir(parents=True, exist_ok=True)
    TailTreeModel(
        booster=model.booster,
        metadata=model.metadata.model_copy(
            update={
                "search": OptunaSearchProvenance(
                    study_name=study_name,
                    trial_number=trial_number,
                    score=float(score),
                    seed=int(seed),
                )
            }
        ),
    ).to_json(target_model)


def accepted_manifest_path(matrix: pl.DataFrame) -> Path:
    if ACCEPTED_MANIFEST_PATH.exists():
        manifest = AcceptedFeatureManifest.read(ACCEPTED_MANIFEST_PATH)
        missing = sorted(set(manifest.selected_columns) - set(matrix.columns))
        if not missing:
            return ACCEPTED_MANIFEST_PATH
    manifest = ProposalFeatureManifest.read(PROPOSAL_MANIFEST_PATH).accepted(
        accepted_by="scripts/02_train.py",
        note="auto-accepted from latest proposal",
    )
    ACCEPTED_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.write(ACCEPTED_MANIFEST_PATH)
    return ACCEPTED_MANIFEST_PATH


async def train_tailtree() -> Path:
    matrix = pl.read_parquet(FEATURE_MATRIX_PATH)
    manifest = AcceptedFeatureManifest.read(accepted_manifest_path(matrix))
    feature_set = TrainFeatureSet(matrix, manifest, EVALUATION)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    review, config, score, trial = optuna_review(feature_set)
    review.write_csv(PROFILE_RUNS_PATH)
    labels = feature_set.label_distribution()
    labels.write_csv(LABEL_DISTRIBUTION_PATH)
    analysis = model_analysis_frame(review, label_distribution=labels)
    analysis.write_csv(MODEL_ANALYSIS_PATH)
    MODEL_ANALYSIS_REPORT_PATH.write_text(model_analysis_markdown(analysis), encoding="utf-8")
    source_model = MODEL_DIR / f"tailtree-path-t{trial:04d}_path.json"
    feature_set.train_final(config).to_json(source_model)
    write_selected_model(
        source_model,
        MODEL_PATH,
        trial_number=trial,
        score=score,
        seed=TRAINING.seed,
        study_name=STUDY_NAME,
    )
    best_review = review.row(0, named=True) if not review.is_empty() else {}
    MODEL_REVIEW_PATH.write_text(
        json.dumps(
            {
                "model_path": str(MODEL_PATH),
                "source_model_path": str(source_model),
                "best_score": score,
                "best_metric": "ewma_board_utility_score",
                "best_board_utility_score": best_review.get("ewma_board_utility_score"),
                "best_mean_board_utility_score": best_review.get("mean_board_utility_score"),
                "best_min_fold_board_utility_score": best_review.get(
                    "min_fold_board_utility_score"
                ),
                "best_trial_number": trial,
                "evaluation": EVALUATION.model_dump(mode="json"),
                "training": TRAINING.model_dump(mode="json"),
                "feature_matrix_path": str(FEATURE_MATRIX_PATH),
                "accepted_manifest_path": str(ACCEPTED_MANIFEST_PATH),
                "analysis_path": str(MODEL_ANALYSIS_PATH),
                "analysis_report_path": str(MODEL_ANALYSIS_REPORT_PATH),
                "label_distribution_path": str(LABEL_DISTRIBUTION_PATH),
                "study_name": STUDY_NAME,
                "trained_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    remove_legacy_outputs()
    return MODEL_PATH


def main() -> None:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print("Train tailtree path model from accepted features.")
        return
    model_path = asyncio.run(train_tailtree())
    print(model_path)
    print(MODEL_REVIEW_PATH)
    print(MODEL_ANALYSIS_REPORT_PATH)


if __name__ == "__main__":
    main()
