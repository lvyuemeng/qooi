"""Stage A: prepare market data and materialize path feature-research artifacts."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from pathlib import Path

import polars as pl

from qooi.profiling import ProfileConfig, ProfileContext
from qooi.scanner.config import (
    BarsConfig,
    ExtremeTailConfig,
    RubikConfig,
    RunContract,
    SnapshotConfig,
    TransitionConfig,
)
from qooi.scanner.outcome import potential_outcome_frame
from qooi.scanner.path_labels import (
    PathLabelSpec,
    make_path_labels,
    path_label_outcome_frame,
)
from qooi.scanner.path_model import TailTreeModel, TrainConfig
from qooi.scanner.tailrun.features import FeatureSpec
from qooi.scanner.tailrun.planning import TailtreeWalkforwardSpec, tailtree_fold_specs
from qooi.scanner.tailrun.research import (
    ResearchMetric,
    ResearchSpec,
    research_candidates,
    run_feature_research,
)
from qooi.scanner.tailrun.review import (
    path_decile_monotonicity,
    path_extreme_events,
    path_feature_analysis,
    path_feature_analysis_report,
    path_profit_metrics,
    path_robust_profit_metrics,
    path_scored_trend_frame,
    path_source_feature_health,
)

OUTPUT_DIR = Path("data/output/potential/path")
FEATURE_DIR = OUTPUT_DIR
REVIEW_DIR = FEATURE_DIR / "review"
MATRIX_PATH = FEATURE_DIR / "features_full.parquet"
PREDICT_MATRIX_PATH = FEATURE_DIR / "predict_features.parquet"
RAW_MANIFEST_PATH = FEATURE_DIR / "manifest_raw.json"
PROPOSAL_PATH = FEATURE_DIR / "feature-manifest.proposal.json"
FEATURE_REVIEW_PATH = FEATURE_DIR / "feature-review.csv"
ANALYSIS_PATH = REVIEW_DIR / "feature-analysis.csv"
ANALYSIS_REPORT_PATH = REVIEW_DIR / "feature-analysis.md"
EXTREME_EVENTS_PATH = REVIEW_DIR / "feature-extreme-events.csv"
ITERATION_NOTES_PATH = REVIEW_DIR / "feature-iteration-notes.md"
LEGACY_OUTPUT_DIRS = (
    Path("data/output/potential/path-train"),
    Path("data/output/potential/path-predict"),
    Path("data/output/potential/path/tailtree"),
)

MAX_SYMBOLS = 10
MAX_STALENESS_HOURS = 2
FETCH_CONCURRENCY = 8
OUTCOME_HORIZONS = (4, 12, 24)
CONTRACT = RunContract.profile("h24_swing", OUTCOME_HORIZONS)
BARS = BarsConfig(timeframes=CONTRACT.required_timeframes, days=180, refresh_mode="incremental")
BOOKS = SnapshotConfig(limit=25, max_staleness_hours=1)
TRADES = SnapshotConfig(limit=100, max_staleness_hours=1)
FUNDING = SnapshotConfig(limit=500, max_staleness_hours=4)
OPEN_INTEREST = RubikConfig(period="1H", limit=100, unit="2", max_staleness_hours=1)
TAKER_VOLUME = RubikConfig(period="1H", limit=100, unit="2", max_staleness_hours=1)
LONG_SHORT = RubikConfig(period="1H", limit=100, unit="2", max_staleness_hours=1)
TRANSITION = TransitionConfig(
    scan_budget=80,
    history_days=180,
    context_scope="all_scanned",
    context_limit=80,
    ngram_length=4,
    horizon=24,
    mae_mfe_horizon=24,
    min_count=24,
    recent_window=720,
    long_window=4320,
    return_threshold_pct=0.0,
    min_probability=0.05,
    min_directional_probability=0.55,
    min_probability_delta=-0.1,
    min_reward_risk=1.0,
    max_tail_loss_pct=20.0,
    min_information_bits=0.001,
)
PROFILE = ProfileConfig(mode="hotpath", top_n=80)
EXTREME = ExtremeTailConfig(
    method="hybrid",
    material_floor_pct=20.0,
    quantile=0.95,
    min_event_rate=0.001,
    max_event_rate=0.10,
    reference_scope="universe_horizon",
)
FEATURE_SPEC = FeatureSpec(
    horizons=CONTRACT.outcome_horizons,
    windows_hours=(4, 12, 24, 48, 72),
    tsfresh_value_columns=(
        "open_rel_decision",
        "high_rel_decision",
        "low_rel_decision",
        "close_rel_decision",
        "volume",
    ),
    tsfresh_required_value_columns=(
        "open_rel_decision",
        "high_rel_decision",
        "low_rel_decision",
        "close_rel_decision",
        "volume",
    ),
    tsfresh_calculators=(
        "minimum",
        "maximum",
        "median",
        "skewness",
        "kurtosis",
        "abs_energy",
        "mean_abs_change",
    ),
    selected_generated_columns=(
        "cross__close_volume__w24h__corr",
        "tsf__low__w24h__minimum",
        "tsf__close__w12h__skewness",
        "tsf__low__w24h__kurtosis",
        "tsf__high__w24h__kurtosis",
        "tsf__high__w24h__mean_abs_change",
        "tsf__high__w12h__kurtosis",
        "tsf__close__w48h__mean_abs_change",
        "tsf__high__w72h__abs_energy",
        "tsf__range__w24h__kurtosis",
        "tsf__close__w24h__kurtosis",
        "tsf__high__w24h__maximum",
        "tsf__high__w4h__mean_abs_change",
        "cross__close_volume__w72h__corr",
        "tsf__range__w48h__kurtosis",
        "tsf__range__w12h__kurtosis",
        "tsf__volume__w48h__skewness",
        "tsf__close__w48h__kurtosis",
        "tsf__volume__w24h__skewness",
        "cross__range_volume__w48h__corr",
        "tsf__low__w4h__abs_energy",
        "tsf__high__w72h__skewness",
        "tsf__low__w72h__kurtosis",
        "tsf__close__w4h__median",
    ),
    source_tsfresh_value_columns=(
        "base__funding_rate_bps",
        "base__oi_change_pct",
        "base__taker_buy_pressure",
        "base__lsr_log_ratio",
        "base__funding_age_ms",
        "base__oi_age_ms",
        "base__taker_age_ms",
        "base__lsr_age_ms",
    ),
    source_tsfresh_calculators=(
        "last_minus_first",
        "minimum",
        "maximum",
        "median",
        "valid_ratio",
        "q90_q10_range",
        "sample_count",
    ),
)
RESEARCH_SPEC = ResearchSpec(
    method="rfecv_lgbm",
    min_features=20,
    max_features=80,
    step=10,
    mrmr_corr_limit=0.70,
    source_min_non_null_rate=0.10,
    metric=ResearchMetric.NDCG_EXCESS_AT_10,
    excluded_feature_columns=(
        "base__market_dispersion_24h",
        "tsf__high__w12h__skewness",
        "tsf__low__w12h__skewness",
        "tsf__volume__w12h__skewness",
        "tsf__high__w12h__maximum",
    ),
    train_config=TrainConfig(
        objective="path_prototype",
        num_leaves=32,
        min_data_in_leaf=20,
        learning_rate=0.05,
        num_iterations=60,
        early_stopping_rounds=10,
        random_seed=42,
    ),
)
FOLD_SPEC = TailtreeWalkforwardSpec(
    train_days=90,
    valid_days=21,
    step_days=21,
    max_folds=2,
    embargo_bars=24,
)
PROMOTION_FOLDS = TailtreeWalkforwardSpec(
    train_days=90,
    valid_days=7,
    step_days=7,
    max_folds=6,
    embargo_bars=24,
)


def _write_iteration_notes(shap_review, path: Path) -> None:
    top = (
        shap_review.filter(shap_review["feature"].str.starts_with("tsf__"))
        if not shap_review.is_empty() and "feature" in shap_review.columns
        else shap_review.head(0)
    )
    lines = [
        "# Feature iteration notes",
        "",
        "Top tsfresh features are research clues only. Promote new `base__` features "
        "only after a market-logic review.",
        "",
    ]
    if top.is_empty():
        lines.append("No tsfresh SHAP rows were available for iteration notes.")
    else:
        rows = top.sort("mean_abs_shap", descending=True).head(25).to_dicts()
        for row in rows:
            feature = str(row["feature"])
            parts = feature.split("__")
            value = parts[1] if len(parts) > 1 else "unknown"
            window = parts[2] if len(parts) > 2 else "unknown_window"
            calculator = parts[3] if len(parts) > 3 else "unknown_stat"
            lines.extend(
                [
                    f"## {feature}",
                    f"- class: {row.get('class_name', '')}",
                    f"- mean_abs_shap: {row.get('mean_abs_shap', 0.0)}",
                    f"- likely source: `{value}` over `{window}` with `{calculator}`",
                    "- candidate base-feature direction: express this as momentum, volatility, "
                    "position, trend-efficiency, or volume-pressure if the market rationale "
                    "is clear.",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def remove_legacy_outputs() -> None:
    for path in LEGACY_OUTPUT_DIRS:
        shutil.rmtree(path, ignore_errors=True)


def write_feature_selection(result) -> None:
    result.manifest.write(PROPOSAL_PATH)


def _blind_cutoff(matrix: pl.DataFrame) -> int:
    times = matrix.select(pl.col("decision_bar_close_ms").unique().sort()).to_series().to_list()
    if len(times) < 5:
        raise ValueError("feedback requires at least five decision timestamps")
    return int(times[int(len(times) * 0.8)])


def _time_filter(
    matrix: pl.DataFrame, *, start_ms: int | None = None, end_ms: int | None = None
) -> pl.DataFrame:
    expr = pl.lit(True)
    if start_ms is not None:
        expr = expr & (pl.col("decision_bar_close_ms") >= int(start_ms))
    if end_ms is not None:
        expr = expr & (pl.col("decision_bar_close_ms") < int(end_ms))
    return matrix.filter(expr)


_SOURCE_PROBE_CALCULATORS = ("sample_count", "valid_ratio", "last_minus_first", "median")
_SOURCE_PROBE_TOP_N = 24
_SOURCE_PROBE_MIN_NDCG = 0.01
_SOURCE_PROBE_NDCG_K = 10


def _source_probe_label_sample(labels: pl.DataFrame, *, per_slice: int = 24) -> pl.DataFrame:
    times = labels.select(pl.col("decision_bar_close_ms").unique().sort()).to_series().to_list()
    if len(times) < 5:
        return labels
    cutoff = int(len(times) * 0.8)
    train = times[:cutoff]
    blind = times[cutoff:]
    selected_times = set(train[:per_slice]) | set(train[-per_slice:]) | set(blind[:per_slice])
    return labels.filter(pl.col("decision_bar_close_ms").is_in(sorted(selected_times)))


def _expand_source_probe_output(column: str, windows: tuple[int, ...]) -> tuple[str, ...]:
    parts = column.split("__")
    if len(parts) != 4 or not parts[2].startswith("w") or not parts[2].endswith("h"):
        return (column,)
    return tuple(f"{parts[0]}__{parts[1]}__w{int(window)}h__{parts[3]}" for window in windows)


def _source_probe_ndcg_score(
    frame: pl.DataFrame, column: str, target: str, *, k: int = _SOURCE_PROBE_NDCG_K
) -> tuple[float, int, float]:
    clean = frame.select(pl.col(column).cast(pl.Float64), pl.col(target).cast(pl.Float64)).filter(
        pl.col(column).is_finite() & pl.col(target).is_finite()
    )
    finite_count = clean.height
    if finite_count < 4 or clean.select(pl.col(column).n_unique()).item() < 2:
        return 0.0, finite_count, 0.0
    gains = clean.with_columns((pl.col(target).clip(0.0, None) + 1.0).log().alias("gain"))
    if gains.filter(pl.col("gain") > 0.0).is_empty():
        return 0.0, finite_count, finite_count / max(frame.height, 1)
    discount = (pl.col("rank") + 2).cast(pl.Float64).log() / 0.6931471805599453
    dcg = (
        gains.sort(column, descending=True)
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
    return float(dcg / ideal) if ideal else 0.0, finite_count, finite_count / max(frame.height, 1)


def _source_probe_temporal_stability(
    frame: pl.DataFrame, column: str, target: str
) -> dict[str, float | bool]:
    times = frame.select(pl.col("decision_bar_close_ms").unique().sort()).to_series().to_list()
    if len(times) < 5:
        score, finite_count, finite_rate = _source_probe_ndcg_score(frame, column, target)
        return {
            "ndcg": score,
            "train_early_ndcg": score,
            "train_late_ndcg": score,
            "blind_ndcg": 0.0,
            "finite_count": float(finite_count),
            "finite_rate": finite_rate,
            "temporal_stable": score >= _SOURCE_PROBE_MIN_NDCG,
            "selection_score": score if score >= _SOURCE_PROBE_MIN_NDCG else 0.0,
        }
    train_cut = max(1, int(len(times) * 0.8))
    train_times = times[:train_cut]
    early_cut = max(1, len(train_times) // 2)
    train_early = frame.filter(pl.col("decision_bar_close_ms").is_in(train_times[:early_cut]))
    train_late = frame.filter(pl.col("decision_bar_close_ms").is_in(train_times[early_cut:]))
    blind = frame.filter(pl.col("decision_bar_close_ms").is_in(times[train_cut:]))
    score, finite_count, finite_rate = _source_probe_ndcg_score(frame, column, target)
    early_score, early_count, _ = _source_probe_ndcg_score(train_early, column, target)
    late_score, late_count, _ = _source_probe_ndcg_score(train_late, column, target)
    blind_score, blind_count, _ = _source_probe_ndcg_score(blind, column, target)
    temporal_stable = (
        early_count >= 4
        and late_count >= 4
        and early_score >= _SOURCE_PROBE_MIN_NDCG
        and late_score >= _SOURCE_PROBE_MIN_NDCG
        and (blind_count < 4 or blind_score >= _SOURCE_PROBE_MIN_NDCG)
    )
    return {
        "ndcg": score,
        "train_early_ndcg": early_score,
        "train_late_ndcg": late_score,
        "blind_ndcg": blind_score,
        "finite_count": float(finite_count),
        "finite_rate": finite_rate,
        "temporal_stable": temporal_stable,
        "selection_score": score if temporal_stable and score >= _SOURCE_PROBE_MIN_NDCG else 0.0,
    }


def _source_probe_outputs(
    observations: pl.DataFrame, labels: pl.DataFrame
) -> tuple[tuple[str, ...], pl.DataFrame]:
    probe_labels = _source_probe_label_sample(labels)
    probe_spec = FEATURE_SPEC.model_copy(
        update={
            "windows_hours": (4,),
            "source_tsfresh_calculators": _SOURCE_PROBE_CALCULATORS,
            "selected_generated_columns": (),
        }
    )
    probe = research_candidates(observations, None, probe_labels, spec=probe_spec)
    target = "final_return" if "final_return" in probe.columns else "path_label"
    source_columns = [
        column for column in probe.columns if column.startswith(("tsfsrc__", "crosssrc__"))
    ]
    scored = []
    for column in source_columns:
        stability = _source_probe_temporal_stability(probe, column, target)
        scored.append({"column": column, **stability})
    selected = [
        row
        for row in sorted(scored, key=lambda item: item["selection_score"], reverse=True)
        if row["finite_rate"] >= RESEARCH_SPEC.source_min_non_null_rate
        and row["selection_score"] >= _SOURCE_PROBE_MIN_NDCG
    ][:_SOURCE_PROBE_TOP_N]
    expanded = tuple(
        dict.fromkeys(
            output
            for row in selected
            for output in _expand_source_probe_output(
                str(row["column"]), FEATURE_SPEC.windows_hours
            )
        )
    )
    stable_count = sum(1 for row in scored if row["temporal_stable"])
    selected_min_finite_rate = min((float(row["finite_rate"]) for row in selected), default=0.0)
    metrics = {
        "candidate_generated_count": float(len(scored)),
        "ndcg_passing_count": float(
            sum(1 for row in scored if row["ndcg"] >= _SOURCE_PROBE_MIN_NDCG)
        ),
        "stable_ndcg_count": float(stable_count),
        "temporal_stability_ratio": stable_count / max(len(scored), 1),
        "selected_generated_count": float(len(expanded)),
        "max_ndcg": max((float(row["ndcg"]) for row in scored), default=0.0),
        "min_ndcg": min((float(row["ndcg"]) for row in scored), default=0.0),
        "sparse_split_finite_rate_min": selected_min_finite_rate,
        "train_window_consistency_pass": 1.0
        if selected and all(row["temporal_stable"] for row in selected)
        else 0.0,
        "base_feature_coverage_pass": 1.0 if probe.height > 0 and source_columns else 0.0,
    }
    rows = [
        {
            "section": "source_probe",
            "feature_set": "source_probe",
            "split": "probe_stratified",
            "metric": metric,
            "k": _SOURCE_PROBE_NDCG_K if "ndcg" in metric else None,
            "value": value,
            "sample_count": probe.height,
            "warning": None
            if metric != "selected_generated_count" or expanded
            else "no_temporally_stable_source_probe_selection",
            "action": "use_selected_outputs"
            if metric == "selected_generated_count" and expanded
            else "compare",
        }
        for metric, value in metrics.items()
    ]
    return expanded, pl.DataFrame(rows)


def _score_feedback_feature_set(
    matrix: pl.DataFrame,
    manifest,
    *,
    feature_set: str,
    cutoff: int,
    keep_profit: bool,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    pre_blind = _time_filter(matrix, end_ms=cutoff)
    blind = _time_filter(matrix, start_ms=cutoff)
    scored_frames = []
    profit_frames = []
    folds = tailtree_fold_specs(
        PROMOTION_FOLDS, observations=pre_blind, bar=CONTRACT.decision_timeframe
    )
    for fold in folds:
        train = _time_filter(
            pre_blind, start_ms=fold.train_window.start_ms, end_ms=fold.train_window.end_ms
        )
        valid = _time_filter(
            pre_blind, start_ms=fold.valid_window.start_ms, end_ms=fold.valid_window.end_ms
        )
        if train.is_empty() or valid.is_empty():
            continue
        split = f"promotion_fold_{fold.fold_id}"
        model = TailTreeModel.train_path(
            manifest.select_matrix(train),
            manifest.select_matrix(valid),
            config=RESEARCH_SPEC.train_config,
            selected_manifest=manifest,
            label_contract_id="path_prototype",
        )
        selected_valid = manifest.select_matrix(valid)
        scored_frames.append(
            path_scored_trend_frame(
                model,
                selected_valid,
                source_run_id=split,
                scope="promotion_fold_valid",
                feature_set=feature_set,
                split=split,
            )
        )
        if keep_profit:
            profit_frames.append(
                path_profit_metrics(
                    model,
                    selected_valid,
                    run_id=split,
                    scope="promotion_fold_valid",
                    k_values=(5, 10, 20, 50),
                )
            )
    model = TailTreeModel.train_path(
        manifest.select_matrix(pre_blind),
        manifest.select_matrix(blind),
        config=RESEARCH_SPEC.train_config,
        selected_manifest=manifest,
        label_contract_id="path_prototype",
    )
    selected_blind = manifest.select_matrix(blind)
    scored_frames.append(
        path_scored_trend_frame(
            model,
            selected_blind,
            source_run_id="promotion_blind",
            scope="promotion_blind",
            feature_set=feature_set,
            split="promotion_blind",
        )
    )
    if keep_profit:
        profit_frames.append(
            path_profit_metrics(
                model,
                selected_blind,
                run_id="promotion_blind",
                scope="promotion_blind",
                k_values=(5, 10, 20, 50),
            )
        )
    return (
        pl.concat(scored_frames, how="diagonal_relaxed"),
        pl.concat(profit_frames, how="vertical") if profit_frames else pl.DataFrame(),
    )


def _write_feedback_artifacts(
    matrix: pl.DataFrame, manifest, *, feedback: str, source_probe: pl.DataFrame | None = None
) -> tuple[Path, ...]:
    if feedback == "off":
        return ()
    cutoff = _blind_cutoff(matrix)
    feature_sets = {"base_ndcg_current": manifest}
    if feedback in {"robust", "full"}:
        feature_sets["source_blended_all"] = manifest.source_blended(matrix)
    scored = []
    base_profit = pl.DataFrame()
    for feature_set, feature_manifest in feature_sets.items():
        frame, profit = _score_feedback_feature_set(
            matrix,
            feature_manifest,
            feature_set=feature_set,
            cutoff=cutoff,
            keep_profit=feature_set == "base_ndcg_current",
        )
        scored.append(frame)
        if feature_set == "base_ndcg_current":
            base_profit = profit
    scored_frame = pl.concat(scored, how="diagonal_relaxed")
    robust = pl.DataFrame()
    deciles = pl.DataFrame()
    source_health = pl.DataFrame()
    extreme_events = pl.DataFrame()
    written: list[Path] = []
    if feedback in {"robust", "full"}:
        robust = path_robust_profit_metrics(scored_frame, group_cols=("feature_set", "split"))
        deciles = path_decile_monotonicity(scored_frame, group_cols=("feature_set", "split"))
        source_health = path_source_feature_health(matrix, feature_sets=feature_sets)
        extreme_events = path_extreme_events(
            scored_frame,
            group_cols=("feature_set", "split"),
            tail_count=20,
        )
        if not extreme_events.is_empty():
            extreme_events.write_csv(EXTREME_EVENTS_PATH)
            written.append(EXTREME_EVENTS_PATH)
    analysis = path_feature_analysis(
        feature_set_counts={
            name: len(item.selected_columns) for name, item in feature_sets.items()
        },
        profit=base_profit,
        robust=robust,
        deciles=deciles,
        source_health=source_health,
        extreme_events=extreme_events,
        source_probe=source_probe,
    )
    analysis.write_csv(ANALYSIS_PATH)
    ANALYSIS_REPORT_PATH.write_text(path_feature_analysis_report(analysis), encoding="utf-8")
    return (ANALYSIS_PATH, ANALYSIS_REPORT_PATH, *written)


async def build_features(feedback: str = "robust") -> tuple[Path, Path]:
    from qooi.scanner.workflow import prepare_potential_run

    profile = ProfileContext.from_config(PROFILE, OUTPUT_DIR / "profile")
    try:
        prepared = await prepare_potential_run(
            contract=CONTRACT,
            profile=profile,
            bars=BARS,
            transition=TRANSITION,
            max_symbols=MAX_SYMBOLS,
            max_staleness_hours=MAX_STALENESS_HOURS,
            fetch_concurrency=FETCH_CONCURRENCY,
            books=BOOKS,
            trades=TRADES,
            funding=FUNDING,
            open_interest=OPEN_INTEREST,
            taker_volume=TAKER_VOLUME,
            long_short=LONG_SHORT,
        )
        with profile.stage("scanner", "feature", "path_outcomes"):
            outcomes = potential_outcome_frame(
                prepared.base.observations,
                prepared.base.source_outcomes,
                prepared.base.realized,
                return_threshold_pct=TRANSITION.return_threshold_pct,
            )
        profile.frame("scanner", "feature", "path_outcomes", outcomes)
        with profile.stage("scanner", "feature", "path_labels"):
            label_outcomes = path_label_outcome_frame(outcomes, CONTRACT.outcome_horizons)
            labels = make_path_labels(label_outcomes, CONTRACT.outcome_horizons, PathLabelSpec())
        profile.frame("scanner", "feature", "path_labels", labels)
        with profile.stage("scanner", "feature", "source_probe"):
            selected_source_outputs, source_probe = _source_probe_outputs(
                prepared.base.observations, labels
            )
            feature_spec = FEATURE_SPEC.model_copy(
                update={
                    "selected_generated_columns": (
                        *FEATURE_SPEC.selected_generated_columns,
                        *selected_source_outputs,
                    )
                }
            )
        with profile.stage("scanner", "feature", "feature_matrix"):
            matrix = research_candidates(
                prepared.base.observations,
                prepared.base.histories,
                labels,
                spec=feature_spec,
            )
            latest_decision_ms = prepared.base.observations.select(
                pl.col("decision_bar_close_ms").max()
            ).item()
            recent_observations = prepared.base.observations.filter(
                pl.col("decision_bar_close_ms")
                >= int(latest_decision_ms - MAX_STALENESS_HOURS * 60 * 60 * 1000)
            )
            predict_matrix = feature_spec.predict_frame(
                recent_observations,
                prepared.base.histories,
            )
        folds = tailtree_fold_specs(
            FOLD_SPEC, observations=prepared.base.observations, bar=CONTRACT.decision_timeframe
        )
        with profile.stage("scanner", "feature", "feature_research"):
            result = run_feature_research(
                matrix,
                folds,
                spec=RESEARCH_SPEC,
                artifact_id="path-features-production-proposal",
                feature_spec=feature_spec,
                run_id="build_features",
            )
        profile.frame("scanner", "feature", "features_full", matrix)
        FEATURE_DIR.mkdir(parents=True, exist_ok=True)
        REVIEW_DIR.mkdir(parents=True, exist_ok=True)
        matrix.write_parquet(MATRIX_PATH)
        predict_matrix.write_parquet(PREDICT_MATRIX_PATH)
        result.feature_review.write_csv(FEATURE_REVIEW_PATH)
        feedback_artifacts = _write_feedback_artifacts(
            matrix, result.manifest, feedback=feedback, source_probe=source_probe
        )
        _write_iteration_notes(result.review.shap, ITERATION_NOTES_PATH)
        result.manifest.model_copy(
            update={
                "review_artifact_ids": (
                    str(FEATURE_REVIEW_PATH),
                    *(str(path) for path in feedback_artifacts),
                    str(ITERATION_NOTES_PATH),
                )
            }
        ).write(PROPOSAL_PATH)
        RAW_MANIFEST_PATH.write_text(
            json.dumps(
                {
                    "artifact_kind": "raw_feature_manifest",
                    "matrix_path": str(MATRIX_PATH),
                    "predict_matrix_path": str(PREDICT_MATRIX_PATH),
                    "review_path": str(FEATURE_REVIEW_PATH),
                    "feature_manifest_proposal_path": str(PROPOSAL_PATH),
                    "feature_analysis_path": str(ANALYSIS_PATH)
                    if ANALYSIS_PATH in feedback_artifacts
                    else "",
                    "feature_analysis_report_path": str(ANALYSIS_REPORT_PATH)
                    if ANALYSIS_REPORT_PATH in feedback_artifacts
                    else "",
                    "feature_extreme_events_path": str(EXTREME_EVENTS_PATH)
                    if EXTREME_EVENTS_PATH in feedback_artifacts
                    else "",
                    "feature_iteration_notes_path": str(ITERATION_NOTES_PATH),
                    "row_count": matrix.height,
                    "feature_columns": [
                        column
                        for column in matrix.columns
                        if column.startswith(
                            ("ctx__", "base__", "tsf__", "cross__", "tsfsrc__", "crosssrc__")
                        )
                    ],
                    "label_columns": [
                        column
                        for column in (
                            "path_label",
                            "path_label_name",
                            "sample_weight",
                            "trend_cleanliness",
                            "risk_adjusted_path_weight",
                            "first_touch_hours",
                            "final_return",
                            "path_reason",
                        )
                        if column in matrix.columns
                    ],
                    "spec": feature_spec.model_dump(mode="json"),
                    "research": {
                        "method": RESEARCH_SPEC.method,
                        "min_features": RESEARCH_SPEC.min_features,
                        "max_features": RESEARCH_SPEC.max_features,
                        "step": RESEARCH_SPEC.step,
                        "metric": RESEARCH_SPEC.metric,
                        "excluded_feature_columns": RESEARCH_SPEC.excluded_feature_columns,
                        "train_config": RESEARCH_SPEC.train_config.model_dump(mode="json"),
                    },
                    "extreme": EXTREME.model_dump(mode="json"),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        remove_legacy_outputs()
        return MATRIX_PATH, RAW_MANIFEST_PATH
    finally:
        profile.write()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tailtree features and review feedback.")
    parser.add_argument(
        "--feedback",
        choices=("off", "core", "robust", "full"),
        default="robust",
        help="feature-build feedback artifacts to write",
    )
    args = parser.parse_args()
    matrix, manifest = asyncio.run(build_features(feedback=args.feedback))
    print(f"features={matrix}")
    print(f"predict_features={PREDICT_MATRIX_PATH}")
    print(f"manifest={manifest}")
    print(f"proposal={PROPOSAL_PATH}")
    print(f"feature_report={ANALYSIS_REPORT_PATH}")


if __name__ == "__main__":
    main()
