"""Path-prototype review artifacts: importance, blacklist proposals, PSI."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil
from typing import Literal

import numpy as np
import polars as pl

from qooi.scanner.path_model import TailTreeModel

PolarsDtype = type[pl.DataType] | pl.DataType


@dataclass(frozen=True)
class ProfitSideSpec:
    side: str
    score_col: str
    return_col: str
    excess_col: str
    benchmark_col: str
    positive_labels: tuple[int, ...]

    def frame(self, source: pl.DataFrame) -> pl.DataFrame:
        return source.select(
            "decision_bar_close_ms",
            "horizon_hours",
            pl.col("path_label").is_in(list(self.positive_labels)).alias("positive"),
            pl.col(self.score_col).cast(pl.Float64).alias("score"),
            pl.col(self.return_col).cast(pl.Float64).alias("realized_return"),
            pl.col(self.excess_col).cast(pl.Float64).alias("excess_return"),
            pl.col(self.benchmark_col).cast(pl.Float64).alias("benchmark_return"),
        ).drop_nulls(["score", "realized_return", "excess_return", "benchmark_return"])


PROFIT_SIDES = (
    ProfitSideSpec(
        "up", "path_prob_smooth_up", "final_return", "up_excess_return", "market_return", (1,)
    ),
    ProfitSideSpec(
        "down",
        "path_prob_smooth_down",
        "down_realized_return",
        "down_excess_return",
        "market_return",
        (2,),
    ),
    ProfitSideSpec(
        "trend",
        "trend_score",
        "trend_realized_return",
        "trend_excess_return",
        "trend_market_return",
        (1, 2),
    ),
)

PATH_FEATURE_IMPORTANCE_SCHEMA: dict[str, PolarsDtype] = {
    "source_run_id": pl.String,
    "feature": pl.String,
    "importance_gain": pl.Float64,
    "importance_rank": pl.Int64,
    "selected_feature": pl.Boolean,
    "feature_manifest_id": pl.String,
    "feature_schema_hash": pl.String,
    "label_contract_id": pl.String,
}

PATH_BLACKLIST_PROPOSAL_SCHEMA: dict[str, PolarsDtype] = {
    "source_run_id": pl.String,
    "feature": pl.String,
    "importance_gain": pl.Float64,
    "min_gain": pl.Float64,
    "proposal_action": pl.String,
    "proposal_reason": pl.String,
    "feature_manifest_id": pl.String,
    "label_contract_id": pl.String,
}

PATH_FEATURE_PSI_SCHEMA: dict[str, PolarsDtype] = {
    "source_run_id": pl.String,
    "feature": pl.String,
    "psi": pl.Float64,
    "drift_status": pl.String,
    "train_null_rate": pl.Float64,
    "recent_null_rate": pl.Float64,
    "bin_count": pl.Int64,
}

PATH_SHAP_REVIEW_SCHEMA: dict[str, PolarsDtype] = {
    "source_run_id": pl.String,
    "sample_scope": pl.String,
    "feature_manifest_id": pl.String,
    "feature_schema_hash": pl.String,
    "label_contract_id": pl.String,
    "output_space": pl.String,
    "class_name": pl.String,
    "class_index": pl.Int64,
    "feature": pl.String,
    "mean_abs_shap": pl.Float64,
    "mean_shap": pl.Float64,
    "positive_share": pl.Float64,
    "row_count": pl.Int64,
    "importance_rank": pl.Int64,
}

PATH_FEATURE_MATRIX_REVIEW_SCHEMA: dict[str, PolarsDtype] = {
    "source_run_id": pl.String,
    "feature": pl.String,
    "row_count": pl.Int64,
    "non_null_rate": pl.Float64,
    "zero_rate": pl.Float64,
    "unique_count": pl.Int64,
    "mean": pl.Float64,
    "std": pl.Float64,
    "min": pl.Float64,
    "max": pl.Float64,
}

PATH_PREDICTION_METRICS_SCHEMA: dict[str, PolarsDtype] = {
    "source_run_id": pl.String,
    "scope": pl.String,
    "class_index": pl.Int64,
    "class_name": pl.String,
    "support": pl.Int64,
    "precision": pl.Float64,
    "recall": pl.Float64,
    "f1": pl.Float64,
}


PATH_FEATURE_ANALYSIS_SCHEMA: dict[str, PolarsDtype] = {
    "section": pl.String,
    "feature_set": pl.String,
    "split": pl.String,
    "metric": pl.String,
    "k": pl.Int64,
    "value": pl.Float64,
    "sample_count": pl.Int64,
    "warning": pl.String,
    "action": pl.String,
}

PATH_PROFIT_METRICS_SCHEMA: dict[str, PolarsDtype] = {
    "source_run_id": pl.String,
    "scope": pl.String,
    "side": pl.String,
    "metric": pl.String,
    "k": pl.Int64,
    "value": pl.Float64,
    "sample_count": pl.Int64,
    "positive_count": pl.Int64,
    "mean_realized_return": pl.Float64,
    "mean_excess_return": pl.Float64,
    "profit_factor_proxy": pl.Float64,
    "top_bottom_spread": pl.Float64,
    "sortino_ratio": pl.Float64,
    "topk_max_drawdown": pl.Float64,
    "average_win": pl.Float64,
    "average_loss": pl.Float64,
    "win_loss_ratio": pl.Float64,
    "calmar_proxy": pl.Float64,
    "upside_capture": pl.Float64,
    "downside_capture": pl.Float64,
}


def _select_schema(frame: pl.DataFrame, schema: dict[str, PolarsDtype]) -> pl.DataFrame:
    if frame.is_empty() and not frame.columns:
        return pl.DataFrame(schema=schema)
    filled = frame.with_columns(
        *[
            pl.lit(None, dtype=dtype).alias(column)
            for column, dtype in schema.items()
            if column not in frame.columns
        ]
    )
    return filled.select(*(pl.col(column).cast(dtype) for column, dtype in schema.items()))


def path_feature_importance(model: TailTreeModel, *, run_id: str) -> pl.DataFrame:
    """Project path model feature importance metadata into a review artifact."""
    if (
        model.metadata.direction != "path"
        or model.metadata.train_config.objective != "path_prototype"
    ):
        raise ValueError("path_feature_importance requires a path_prototype TailTreeModel")
    selected = set(model.metadata.selected_columns or model.metadata.continuous_features)
    importance = sorted(
        model.metadata.feature_importance,
        key=lambda item: float(item[1]),
        reverse=True,
    )
    rows = [
        {
            "source_run_id": run_id,
            "feature": feature,
            "importance_gain": float(gain),
            "importance_rank": rank,
            "selected_feature": feature in selected,
            "feature_manifest_id": model.metadata.feature_manifest_id,
            "feature_schema_hash": model.metadata.feature_schema_hash,
            "label_contract_id": model.metadata.label_contract_id,
        }
        for rank, (feature, gain) in enumerate(importance, start=1)
    ]
    if not rows:
        rows = [
            {
                "source_run_id": run_id,
                "feature": feature,
                "importance_gain": 0.0,
                "importance_rank": rank,
                "selected_feature": True,
                "feature_manifest_id": model.metadata.feature_manifest_id,
                "feature_schema_hash": model.metadata.feature_schema_hash,
                "label_contract_id": model.metadata.label_contract_id,
            }
            for rank, feature in enumerate(model.metadata.selected_columns, start=1)
        ]
    return _select_schema(pl.DataFrame(rows), PATH_FEATURE_IMPORTANCE_SCHEMA).sort(
        "importance_rank"
    )


def path_feature_matrix_review(
    matrix: pl.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    run_id: str,
) -> pl.DataFrame:
    """Summarize model-free feature matrix health for feature-build review."""
    row_count = matrix.height
    if row_count <= 0 or not feature_columns:
        return pl.DataFrame(schema=PATH_FEATURE_MATRIX_REVIEW_SCHEMA)
    rows = []
    for feature in feature_columns:
        if feature not in matrix.columns:
            continue
        values = matrix.select(pl.col(feature).cast(pl.Float64, strict=False).alias(feature))
        non_null = values.select(pl.col(feature).is_not_null().sum()).item()
        zero_count = values.select((pl.col(feature).fill_null(float("nan")) == 0.0).sum()).item()
        summary = values.select(
            pl.col(feature).n_unique().alias("unique_count"),
            pl.col(feature).mean().alias("mean"),
            pl.col(feature).std().alias("std"),
            pl.col(feature).min().alias("min"),
            pl.col(feature).max().alias("max"),
        ).row(0, named=True)
        rows.append(
            {
                "source_run_id": run_id,
                "feature": feature,
                "row_count": row_count,
                "non_null_rate": float(non_null or 0) / row_count,
                "zero_rate": float(zero_count or 0) / row_count,
                **summary,
            }
        )
    return _select_schema(pl.DataFrame(rows), PATH_FEATURE_MATRIX_REVIEW_SCHEMA).sort("feature")


def path_prediction_metrics(
    model: TailTreeModel, matrix: pl.DataFrame, *, run_id: str
) -> pl.DataFrame:
    """Summarize path-model precision/recall/F1 via sklearn metrics on LightGBM predictions."""
    if matrix.is_empty():
        return pl.DataFrame(schema=PATH_PREDICTION_METRICS_SCHEMA)
    from sklearn.metrics import precision_recall_fscore_support

    scored = model.score_path(matrix)
    joined = matrix.select("symbol", "decision_bar_close_ms", "horizon_hours", "path_label").join(
        scored.select("symbol", "decision_bar_close_ms", "horizon_hours", "path_pred_label"),
        on=("symbol", "decision_bar_close_ms", "horizon_hours"),
        how="inner",
    )
    labels = list(range(5))
    class_names = tuple(
        model.metadata.class_names or ["calm", "smooth_up", "smooth_down", "chop", "fake_breakout"]
    )
    y_true = joined.get_column("path_label").cast(pl.Int64).to_list()
    y_pred = joined.get_column("path_pred_label").cast(pl.Int64).to_list()
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0.0,
    )
    rows = [
        {
            "source_run_id": run_id,
            "scope": "class",
            "class_index": class_index,
            "class_name": class_names[class_index],
            "support": int(support[class_index]),
            "precision": float(precision[class_index]),
            "recall": float(recall[class_index]),
            "f1": float(f1[class_index]),
        }
        for class_index in labels
    ]
    for average in ("macro", "weighted"):
        p_avg, r_avg, f_avg, _support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average=average,
            zero_division=0.0,
        )
        rows.append(
            {
                "source_run_id": run_id,
                "scope": average,
                "class_index": -1,
                "class_name": average,
                "support": len(y_true),
                "precision": float(p_avg),
                "recall": float(r_avg),
                "f1": float(f_avg),
            }
        )
    return _select_schema(pl.DataFrame(rows), PATH_PREDICTION_METRICS_SCHEMA)


def _weighted_spearman_groups(side_frame: pl.DataFrame, *, alpha: float = 2.0) -> pl.DataFrame:
    ranked = side_frame.with_columns(
        pl.col("score")
        .rank(method="average")
        .over("decision_bar_close_ms", "horizon_hours")
        .alias("_score_rank"),
        pl.col("excess_return")
        .rank(method="average")
        .over("decision_bar_close_ms", "horizon_hours")
        .alias("_excess_rank"),
        (pl.max_horizontal(pl.lit(0.0), pl.col("excess_return")) ** float(alpha)).alias("_weight"),
    )
    grouped = ranked.group_by("decision_bar_close_ms", "horizon_hours").agg(
        pl.len().alias("n"),
        (pl.col("_weight") > 0.0).sum().cast(pl.Int64).alias("positive_weight_count"),
        pl.col("_weight").sum().alias("_w"),
        (pl.col("_weight") * pl.col("_score_rank")).sum().alias("_wx"),
        (pl.col("_weight") * pl.col("_excess_rank")).sum().alias("_wy"),
        (pl.col("_weight") * pl.col("_score_rank") * pl.col("_score_rank")).sum().alias("_wx2"),
        (pl.col("_weight") * pl.col("_excess_rank") * pl.col("_excess_rank")).sum().alias("_wy2"),
        (pl.col("_weight") * pl.col("_score_rank") * pl.col("_excess_rank")).sum().alias("_wxy"),
    )
    return (
        grouped.with_columns(
            (pl.col("_wxy") - (pl.col("_wx") * pl.col("_wy") / pl.col("_w"))).alias("_cov"),
            (pl.col("_wx2") - (pl.col("_wx") * pl.col("_wx") / pl.col("_w"))).alias("_varx"),
            (pl.col("_wy2") - (pl.col("_wy") * pl.col("_wy") / pl.col("_w"))).alias("_vary"),
        )
        .filter(
            (pl.col("n") >= 2)
            & (pl.col("positive_weight_count") >= 2)
            & (pl.col("_w") > 0.0)
            & (pl.col("_varx") > 0.0)
            & (pl.col("_vary") > 0.0)
        )
        .with_columns(
            (pl.col("_cov") / (pl.col("_varx") * pl.col("_vary")).sqrt()).alias("weighted_spearman")
        )
        .filter(pl.col("weighted_spearman").is_finite())
        .select(
            "decision_bar_close_ms",
            "horizon_hours",
            "n",
            "positive_weight_count",
            "weighted_spearman",
        )
        .sort("decision_bar_close_ms", "horizon_hours")
    )


def path_profit_metrics(
    model: TailTreeModel,
    matrix: pl.DataFrame,
    *,
    run_id: str,
    scope: str = "probe_full",
    k_values: tuple[int, ...] = (5, 10, 50),
) -> pl.DataFrame:
    """Summarize native Polars alpha, rank, and risk proxies for path proposals."""
    if matrix.is_empty() or not {"path_label", "final_return"} <= set(matrix.columns):
        return pl.DataFrame(schema=PATH_PROFIT_METRICS_SCHEMA)
    scored = model.score_path(matrix)
    frame = (
        matrix.select(
            "symbol", "decision_bar_close_ms", "horizon_hours", "path_label", "final_return"
        )
        .join(
            scored.select(
                "symbol",
                "decision_bar_close_ms",
                "horizon_hours",
                "path_prob_smooth_up",
                "path_prob_smooth_down",
                "path_prob_chop",
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
            (
                pl.col("final_return")
                - pl.col("final_return").median().over("decision_bar_close_ms", "horizon_hours")
            ).alias("up_excess_return"),
            (
                pl.col("final_return").median().over("decision_bar_close_ms", "horizon_hours")
                - pl.col("final_return")
            ).alias("down_excess_return"),
        )
        .with_columns(
            (-pl.col("final_return")).alias("down_realized_return"),
            pl.when(pl.col("path_prob_smooth_up") >= pl.col("path_prob_smooth_down"))
            .then(pl.col("final_return"))
            .otherwise(-pl.col("final_return"))
            .alias("trend_realized_return"),
            pl.when(pl.col("path_prob_smooth_up") >= pl.col("path_prob_smooth_down"))
            .then(pl.col("market_return"))
            .otherwise(-pl.col("market_return"))
            .alias("trend_market_return"),
            pl.when(pl.col("path_prob_smooth_up") >= pl.col("path_prob_smooth_down"))
            .then(pl.col("up_excess_return"))
            .otherwise(pl.col("down_excess_return"))
            .alias("trend_excess_return"),
        )
    )
    rows: list[dict[str, object]] = []
    for side in PROFIT_SIDES:
        side_frame = side.frame(frame)
        if side_frame.is_empty():
            continue
        for k in k_values:
            top = side_frame.sort("score", descending=True).head(min(int(k), side_frame.height))
            bottom = side_frame.sort("score").head(min(int(k), side_frame.height))
            summary = top.select(
                pl.len().alias("sample_count"),
                pl.col("positive").sum().cast(pl.Int64).alias("positive_count"),
                pl.col("realized_return").mean().alias("mean_return"),
                pl.col("excess_return").mean().alias("mean_excess_return"),
                pl.col("excess_return").sum().alias("sum_excess_return"),
                pl.col("excess_return").filter(pl.col("excess_return") > 0.0).sum().alias("gain"),
                (-pl.col("excess_return"))
                .filter(pl.col("excess_return") < 0.0)
                .sum()
                .alias("loss"),
                (pl.col("excess_return") < 0.0).sum().cast(pl.Int64).alias("loss_count"),
                pl.col("excess_return")
                .filter(pl.col("excess_return") > 0.0)
                .mean()
                .alias("average_win"),
                (-pl.col("excess_return"))
                .filter(pl.col("excess_return") < 0.0)
                .mean()
                .alias("average_loss"),
                pl.col("excess_return")
                .filter(pl.col("excess_return") < 0.0)
                .std()
                .alias("downside_std"),
                pl.col("excess_return").min().alias("min_excess_return"),
                pl.col("realized_return")
                .filter(pl.col("benchmark_return") > 0.0)
                .mean()
                .alias("upside_return"),
                pl.col("benchmark_return")
                .filter(pl.col("benchmark_return") > 0.0)
                .mean()
                .alias("upside_benchmark"),
                pl.col("realized_return")
                .filter(pl.col("benchmark_return") < 0.0)
                .mean()
                .alias("downside_return"),
                pl.col("benchmark_return")
                .filter(pl.col("benchmark_return") < 0.0)
                .mean()
                .alias("downside_benchmark"),
            ).row(0, named=True)
            samples = int(summary["sample_count"] or 0)
            positives = int(summary["positive_count"] or 0)
            mean_return = float(summary["mean_return"] or 0.0)
            mean_excess = float(summary["mean_excess_return"] or 0.0)
            gain = float(summary["gain"] or 0.0)
            loss = float(summary["loss"] or 0.0)
            loss_count = int(summary["loss_count"] or 0)
            average_win = float(summary["average_win"] or 0.0)
            average_loss = float(summary["average_loss"] or 0.0)
            downside_std = float(summary["downside_std"] or 0.0)
            topk_max_drawdown = abs(float(summary["min_excess_return"] or 0.0))
            profit_factor = (gain / loss) if loss_count >= 3 and loss > 0.0 else None
            win_loss_ratio = average_win / average_loss if average_loss > 0.0 else 0.0
            sortino = mean_excess / downside_std if loss_count >= 3 and downside_std > 0.0 else None
            calmar = (
                float(summary["sum_excess_return"] or 0.0) / topk_max_drawdown
                if topk_max_drawdown > 0.0
                else 0.0
            )
            upside_benchmark = float(summary["upside_benchmark"] or 0.0)
            downside_benchmark = float(summary["downside_benchmark"] or 0.0)
            upside_capture = (
                float(summary["upside_return"] or 0.0) / upside_benchmark
                if upside_benchmark
                else 0.0
            )
            downside_capture = (
                float(summary["downside_return"] or 0.0) / downside_benchmark
                if downside_benchmark
                else 0.0
            )
            top_bottom_spread = mean_excess - float(
                bottom.select(pl.col("excess_return").mean()).item() or 0.0
            )
            row_base = {
                "source_run_id": run_id,
                "scope": scope,
                "side": side.side,
                "k": int(k),
                "sample_count": samples,
                "positive_count": positives,
                "mean_realized_return": mean_return,
                "mean_excess_return": mean_excess,
                "profit_factor_proxy": profit_factor,
                "top_bottom_spread": top_bottom_spread,
                "sortino_ratio": sortino,
                "topk_max_drawdown": topk_max_drawdown,
                "average_win": average_win,
                "average_loss": average_loss,
                "win_loss_ratio": win_loss_ratio,
                "calmar_proxy": calmar,
                "upside_capture": upside_capture,
                "downside_capture": downside_capture,
            }
            for metric, value in (
                ("precision_at_k", positives / samples if samples else 0.0),
                ("topk_mean_return", mean_return),
                ("mean_excess_return", mean_excess),
                ("top_bottom_spread", top_bottom_spread),
                ("profit_factor_proxy", profit_factor),
                ("sortino_ratio", sortino),
                ("loss_count", float(loss_count)),
                ("negative_excess_count", float(loss_count)),
                ("downside_deviation", downside_std),
                ("profit_factor_is_reliable", float(loss_count >= 3 and loss > 0.0)),
                ("sortino_is_reliable", float(loss_count >= 3 and downside_std > 0.0)),
                ("topk_max_drawdown", topk_max_drawdown),
                ("average_win", average_win),
                ("average_loss", average_loss),
                ("win_loss_ratio", win_loss_ratio),
                ("calmar_proxy", calmar),
                ("upside_capture", upside_capture),
                ("downside_capture", downside_capture),
            ):
                rows.append({**row_base, "metric": metric, "value": value})
        spearman = (
            side_frame.group_by("decision_bar_close_ms", "horizon_hours")
            .agg(
                pl.len().alias("n"),
                pl.corr("score", "excess_return", method="spearman").alias("spearman"),
            )
            .filter((pl.col("n") >= 2) & pl.col("spearman").is_finite())
            .sort("decision_bar_close_ms", "horizon_hours")
        )
        if spearman.is_empty():
            continue
        summary = spearman.select(
            pl.len().alias("sample_count"),
            pl.col("spearman").mean().alias("spearman_mean"),
            pl.col("spearman").median().alias("spearman_median"),
            (pl.col("spearman") > 0.0).mean().alias("spearman_positive_share"),
            (pl.col("spearman") > 0.0).sum().cast(pl.Int64).alias("positive_count"),
        ).row(0, named=True)
        decay = (
            spearman.with_row_index("period_index")
            .with_columns(
                ((pl.col("period_index") * 3) // pl.len()).clip(upper_bound=2).alias("period")
            )
            .group_by("period")
            .agg(pl.col("spearman").mean().alias("period_spearman"))
        )
        first = float(decay.filter(pl.col("period") == 0).select("period_spearman").item() or 0.0)
        middle = (
            float(decay.filter(pl.col("period") == 1).select("period_spearman").item() or 0.0)
            if 1 in set(decay.get_column("period"))
            else 0.0
        )
        last = (
            float(decay.filter(pl.col("period") == 2).select("period_spearman").item() or 0.0)
            if 2 in set(decay.get_column("period"))
            else 0.0
        )
        for metric, value in (
            ("spearman_mean", float(summary["spearman_mean"] or 0.0)),
            ("spearman_median", float(summary["spearman_median"] or 0.0)),
            ("spearman_positive_share", float(summary["spearman_positive_share"] or 0.0)),
            ("rank_ic_first_third", first),
            ("rank_ic_middle_third", middle),
            ("rank_ic_last_third", last),
            ("ic_decay_rate", ((first - last) / abs(first)) if first else 0.0),
        ):
            rows.append(
                {
                    "source_run_id": run_id,
                    "scope": f"{scope}:decision_time_horizon",
                    "side": side.side,
                    "metric": metric,
                    "k": None,
                    "value": value,
                    "sample_count": int(summary["sample_count"] or 0),
                    "positive_count": int(summary["positive_count"] or 0),
                }
            )
        weighted_spearman = _weighted_spearman_groups(side_frame)
        if not weighted_spearman.is_empty():
            weighted_summary = weighted_spearman.select(
                pl.len().alias("sample_count"),
                pl.col("weighted_spearman").mean().alias("weighted_spearman_mean"),
                pl.col("weighted_spearman").median().alias("weighted_spearman_median"),
                (pl.col("weighted_spearman") > 0.0)
                .mean()
                .alias("weighted_spearman_positive_share"),
                (pl.col("weighted_spearman") > 0.0).sum().cast(pl.Int64).alias("positive_count"),
            ).row(0, named=True)
            for metric, value in (
                (
                    "weighted_spearman_mean",
                    float(weighted_summary["weighted_spearman_mean"] or 0.0),
                ),
                (
                    "weighted_spearman_median",
                    float(weighted_summary["weighted_spearman_median"] or 0.0),
                ),
                (
                    "weighted_spearman_positive_share",
                    float(weighted_summary["weighted_spearman_positive_share"] or 0.0),
                ),
            ):
                rows.append(
                    {
                        "source_run_id": run_id,
                        "scope": f"{scope}:decision_time_horizon",
                        "side": side.side,
                        "metric": metric,
                        "k": None,
                        "value": value,
                        "sample_count": int(weighted_summary["sample_count"] or 0),
                        "positive_count": int(weighted_summary["positive_count"] or 0),
                    }
                )
    chop = frame.filter(pl.col("path_prob_chop") > 0.5).select(
        pl.col("trend_excess_return").cast(pl.Float64).alias("excess_return")
    )
    if not chop.is_empty():
        summary = chop.select(
            pl.len().alias("sample_count"),
            pl.col("excess_return").mean().alias("mean_excess_return"),
            (pl.col("excess_return") < 0.0).mean().alias("negative_share"),
        ).row(0, named=True)
        for metric, value in (
            ("chop_gate_mean_excess_return", float(summary["mean_excess_return"] or 0.0)),
            ("chop_gate_negative_share", float(summary["negative_share"] or 0.0)),
            ("chop_gate_sample_count", float(summary["sample_count"] or 0.0)),
        ):
            rows.append(
                {
                    "source_run_id": run_id,
                    "scope": f"{scope}_chop_gate",
                    "side": "trend",
                    "metric": metric,
                    "k": None,
                    "value": value,
                    "sample_count": int(summary["sample_count"] or 0),
                    "positive_count": None,
                    "mean_excess_return": float(summary["mean_excess_return"] or 0.0),
                }
            )
    return _select_schema(pl.from_dicts(rows, infer_schema_length=None), PATH_PROFIT_METRICS_SCHEMA)


def path_shap_review(
    model: TailTreeModel,
    matrix: pl.DataFrame,
    *,
    selected_columns: tuple[str, ...],
    run_id: str,
    sample_scope: Literal["train", "recent"],
    max_rows: int = 1000,
) -> pl.DataFrame:
    """Summarize native LightGBM SHAP contributions in raw-margin space."""
    if (
        model.metadata.direction != "path"
        or model.metadata.train_config.objective != "path_prototype"
    ):
        raise ValueError("path_shap_review requires a path_prototype TailTreeModel")
    missing = [column for column in selected_columns if column not in matrix.columns]
    if missing:
        raise ValueError(f"missing selected SHAP feature columns: {', '.join(missing)}")
    if not selected_columns:
        return pl.DataFrame(schema=PATH_SHAP_REVIEW_SCHEMA)
    sample = matrix.head(max(1, int(max_rows)))
    if sample.is_empty():
        return pl.DataFrame(schema=PATH_SHAP_REVIEW_SCHEMA)
    features = sample.select(
        *(pl.col(column).cast(pl.Float64).fill_null(0.0) for column in selected_columns)
    ).to_numpy()
    contributions = np.asarray(model._booster.predict(features, pred_contrib=True)).astype(float)
    class_names = tuple(
        model.metadata.class_names or ["calm", "smooth_up", "smooth_down", "chop", "fake_breakout"]
    )
    class_count = len(class_names)
    expected_width = (len(selected_columns) + 1) * class_count
    if contributions.ndim != 2 or contributions.shape[1] != expected_width:
        raise ValueError(
            "unexpected SHAP contribution shape: "
            f"{contributions.shape}, expected width {expected_width}"
        )
    values = contributions.reshape((features.shape[0], class_count, len(selected_columns) + 1))
    rows = []
    for class_index, class_name in enumerate(class_names):
        class_rows = []
        for feature_index, feature in enumerate(selected_columns):
            feature_values = values[:, class_index, feature_index]
            class_rows.append(
                {
                    "source_run_id": run_id,
                    "sample_scope": sample_scope,
                    "feature_manifest_id": model.metadata.feature_manifest_id,
                    "feature_schema_hash": model.metadata.feature_schema_hash,
                    "label_contract_id": model.metadata.label_contract_id,
                    "output_space": "raw_margin",
                    "class_name": class_name,
                    "class_index": class_index,
                    "feature": feature,
                    "mean_abs_shap": float(np.mean(np.abs(feature_values))),
                    "mean_shap": float(np.mean(feature_values)),
                    "positive_share": float(np.mean(feature_values > 0.0)),
                    "row_count": int(features.shape[0]),
                }
            )
        class_rows = sorted(class_rows, key=lambda row: row["mean_abs_shap"], reverse=True)
        for rank, row in enumerate(class_rows, start=1):
            row["importance_rank"] = rank
            rows.append(row)
    return _select_schema(pl.DataFrame(rows), PATH_SHAP_REVIEW_SCHEMA).sort(
        "sample_scope", "class_index", "importance_rank"
    )


def path_feature_blacklist(
    importance: pl.DataFrame,
    shap: pl.DataFrame,
    *,
    bottom_gain_fraction: float = 0.20,
    max_shap_fraction: float = 0.10,
    run_id: str,
) -> pl.DataFrame:
    """Propose feature removal only when both gain and raw-margin SHAP are weak."""
    if importance.is_empty() or shap.is_empty():
        return pl.DataFrame(schema=PATH_BLACKLIST_PROPOSAL_SCHEMA)
    missing_importance = [
        column for column in PATH_FEATURE_IMPORTANCE_SCHEMA if column not in importance.columns
    ]
    if missing_importance:
        raise ValueError(f"importance frame missing columns: {', '.join(missing_importance)}")
    missing_shap = [column for column in PATH_SHAP_REVIEW_SCHEMA if column not in shap.columns]
    if missing_shap:
        raise ValueError(f"SHAP frame missing columns: {', '.join(missing_shap)}")
    selected = importance.filter(pl.col("selected_feature"))
    if selected.is_empty():
        return pl.DataFrame(schema=PATH_BLACKLIST_PROPOSAL_SCHEMA)
    feature_count = selected.height
    bottom_rank_start = max(1, ceil(feature_count * (1.0 - float(bottom_gain_fraction))))
    global_shap_mean = float(shap.get_column("mean_abs_shap").mean() or 0.0)
    shap_threshold = global_shap_mean * float(max_shap_fraction)
    shap_by_feature = shap.group_by("feature").agg(
        pl.col("mean_abs_shap").max().alias("max_mean_abs_shap")
    )
    proposals = (
        selected.join(shap_by_feature, on="feature", how="left")
        .filter(
            (pl.col("importance_rank") >= bottom_rank_start)
            & (pl.col("max_mean_abs_shap").fill_null(0.0) < shap_threshold)
        )
        .with_columns(
            pl.lit(run_id).alias("source_run_id"),
            pl.lit(float(shap_threshold)).alias("min_gain"),
            pl.lit("review_blacklist").alias("proposal_action"),
            pl.lit("low_gain_and_low_raw_margin_shap").alias("proposal_reason"),
        )
    )
    return _select_schema(proposals, PATH_BLACKLIST_PROPOSAL_SCHEMA).sort("feature")


def _non_null_values(frame: pl.DataFrame, column: str) -> np.ndarray:
    return (
        frame.select(pl.col(column).cast(pl.Float64).drop_nulls())
        .to_series()
        .to_numpy()
        .astype(float)
    )


def _psi_for_column(training: np.ndarray, recent: np.ndarray, bins: int) -> float:
    if len(training) == 0 or len(recent) == 0:
        return 0.0
    low = float(np.nanmin(training))
    high = float(np.nanmax(training))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return 0.0
    edges = np.linspace(low, high, max(2, int(bins)) + 1)
    train_counts, _ = np.histogram(np.clip(training, low, high), bins=edges)
    recent_counts, _ = np.histogram(np.clip(recent, low, high), bins=edges)
    epsilon = 1e-6
    train_pct = np.maximum(train_counts / max(1, train_counts.sum()), epsilon)
    recent_pct = np.maximum(recent_counts / max(1, recent_counts.sum()), epsilon)
    return float(np.sum((recent_pct - train_pct) * np.log(recent_pct / train_pct)))


def path_feature_psi(
    training_matrix: pl.DataFrame,
    recent_matrix: pl.DataFrame,
    *,
    selected_columns: tuple[str, ...],
    run_id: str,
    bins: int = 10,
) -> pl.DataFrame:
    """Compare selected feature distributions; PSI >= 0.2 is a review warning."""
    missing = [
        column
        for column in selected_columns
        if column not in training_matrix.columns or column not in recent_matrix.columns
    ]
    if missing:
        raise ValueError(f"missing selected PSI feature columns: {', '.join(missing)}")
    rows = []
    train_height = max(1, training_matrix.height)
    recent_height = max(1, recent_matrix.height)
    for feature in selected_columns:
        train_values = _non_null_values(training_matrix, feature)
        recent_values = _non_null_values(recent_matrix, feature)
        psi = _psi_for_column(train_values, recent_values, bins)
        rows.append(
            {
                "source_run_id": run_id,
                "feature": feature,
                "psi": psi,
                "drift_status": "drift_warning" if psi >= 0.2 else "stable",
                "train_null_rate": 1.0 - (len(train_values) / train_height),
                "recent_null_rate": 1.0 - (len(recent_values) / recent_height),
                "bin_count": max(2, int(bins)),
            }
        )
    return _select_schema(pl.DataFrame(rows), PATH_FEATURE_PSI_SCHEMA).sort("feature")


SOURCE_REVIEW_TOKENS = ("source_", "funding", "oi_", "taker", "lsr", "market_")


def path_scored_trend_frame(
    model: TailTreeModel,
    matrix: pl.DataFrame,
    *,
    source_run_id: str | None = None,
    scope: str | None = None,
    feature_set: str | None = None,
    split: str | None = None,
) -> pl.DataFrame:
    """Score a path model and project side-normalized trend utility columns."""
    scored = model.score_path(matrix)
    frame = matrix.join(
        scored.select(
            "symbol",
            "decision_bar_close_ms",
            "horizon_hours",
            "path_prob_smooth_up",
            "path_prob_smooth_down",
            "path_prob_chop",
        ),
        on=("symbol", "decision_bar_close_ms", "horizon_hours"),
        how="inner",
    ).with_columns(
        pl.col("final_return")
        .median()
        .over("decision_bar_close_ms", "horizon_hours")
        .alias("market_return"),
        pl.max_horizontal("path_prob_smooth_up", "path_prob_smooth_down").alias("trend_score"),
    )
    optional = []
    if source_run_id is not None:
        optional.append(pl.lit(source_run_id).alias("source_run_id"))
    if scope is not None:
        optional.append(pl.lit(scope).alias("scope"))
    if feature_set is not None:
        optional.append(pl.lit(feature_set).alias("feature_set"))
    if split is not None:
        optional.append(pl.lit(split).alias("split"))
    if optional:
        frame = frame.with_columns(*optional)
    if "base__source_any_present" not in frame.columns:
        frame = frame.with_columns(pl.lit(1.0).alias("base__source_any_present"))
    return frame.with_columns(
        (
            pl.col("trend_score")
            * (0.5 + 0.5 * pl.col("base__source_any_present").fill_null(0.0).clip(0.0, 1.0))
        ).alias("source_presence_calibrated_score"),
        pl.when(pl.col("path_prob_smooth_up") >= pl.col("path_prob_smooth_down"))
        .then(pl.lit("up"))
        .otherwise(pl.lit("down"))
        .alias("predicted_side"),
        pl.when(pl.col("path_prob_smooth_up") >= pl.col("path_prob_smooth_down"))
        .then(pl.col("final_return") - pl.col("market_return"))
        .otherwise(pl.col("market_return") - pl.col("final_return"))
        .alias("trend_excess_return"),
    ).with_columns(
        pl.col("trend_excess_return").abs().alias("abs_trend_excess_return"),
        (pl.col("trend_excess_return") > 0.0).alias("positive"),
    )


def path_rank_buckets(
    scored: pl.DataFrame,
    *,
    group_cols: Sequence[str],
    buckets: int = 10,
    score_column: str = "trend_score",
) -> pl.DataFrame:
    """Summarize score deciles/buckets for a scored path-trend frame."""
    groups = list(group_cols)
    ranked = scored.with_columns(
        (
            (pl.col(score_column).rank(method="ordinal", descending=True).over(*groups) - 1)
            * int(buckets)
            / pl.len().over(*groups)
        )
        .floor()
        .cast(pl.Int64)
        .clip(0, int(buckets) - 1)
        .alias("bucket")
    )
    return (
        ranked.group_by(*groups, "bucket")
        .agg(
            pl.col(score_column).min().alias("score_min"),
            pl.col(score_column).max().alias("score_max"),
            pl.len().alias("sample_count"),
            pl.col("trend_excess_return").mean().alias("mean_excess_return"),
            pl.col("trend_excess_return").median().alias("median_excess_return"),
            pl.col("abs_trend_excess_return").mean().alias("mean_abs_excess_return"),
            pl.col("positive").mean().alias("positive_rate"),
            pl.col("trend_excess_return").quantile(0.9).alias("p90_excess_return"),
            pl.col("trend_excess_return").quantile(0.1).alias("p10_excess_return"),
            pl.col("final_return").mean().alias("mean_final_return"),
            pl.col("market_return").mean().alias("mean_market_return"),
        )
        .sort(*groups, "bucket")
    )


def path_robust_profit_metrics(
    scored: pl.DataFrame,
    *,
    group_cols: Sequence[str],
    k_values: tuple[int, ...] = (5, 10, 20, 50),
    score_column: str = "trend_score",
) -> pl.DataFrame:
    """Compute winsorized Top-K trend utility while retaining raw outlier sensitivity."""
    rows = []
    for key, frame in scored.partition_by(*group_cols, as_dict=True).items():
        key = key if isinstance(key, tuple) else (key,)
        p05, p95 = frame.select(
            pl.col("trend_excess_return").quantile(0.05).alias("p05"),
            pl.col("trend_excess_return").quantile(0.95).alias("p95"),
        ).row(0)
        ranked_desc = frame.sort(score_column, descending=True)
        ranked_asc = frame.sort(score_column)
        for k in k_values:
            top = ranked_desc.head(min(k, frame.height))
            bottom = ranked_asc.head(min(k, frame.height))
            top_w = top.select(pl.col("trend_excess_return").clip(p05, p95).mean()).item()
            bottom_w = bottom.select(pl.col("trend_excess_return").clip(p05, p95).mean()).item()
            raw_top = top.select(pl.col("trend_excess_return").mean()).item()
            raw_bottom = bottom.select(pl.col("trend_excess_return").mean()).item()
            rows.append(
                {
                    **dict(zip(group_cols, key, strict=True)),
                    "k": k,
                    "sample_count": top.height,
                    "raw_mean_excess": raw_top,
                    "winsorized_mean_excess": top_w,
                    "raw_top_bottom_spread": raw_top - raw_bottom,
                    "winsorized_top_bottom_spread": top_w - bottom_w,
                    "raw_minus_winsorized_spread": (raw_top - raw_bottom) - (top_w - bottom_w),
                    "precision_at_k": top.select(pl.col("positive").mean()).item(),
                }
            )
    return pl.from_dicts(rows, infer_schema_length=None).sort(*group_cols, "k")


def path_decile_monotonicity(
    scored: pl.DataFrame,
    *,
    group_cols: Sequence[str],
    bucket_count: int = 10,
    score_column: str = "trend_score",
) -> pl.DataFrame:
    """Measure full-score-curve monotonicity with pairwise decile mean signs."""
    rows = []
    groups = list(group_cols)
    for key, frame in scored.partition_by(*groups, as_dict=True).items():
        key = key if isinstance(key, tuple) else (key,)
        bucketed = (
            frame.with_columns(
                (
                    (pl.col(score_column).rank(method="ordinal", descending=True).over(*groups) - 1)
                    * int(bucket_count)
                    / pl.len().over(*groups)
                )
                .floor()
                .cast(pl.Int64)
                .clip(0, int(bucket_count) - 1)
                .alias("bucket")
            )
            .group_by("bucket")
            .agg(
                pl.len().alias("sample_count"),
                pl.col("trend_excess_return").mean().alias("mean_excess_return"),
                pl.col("trend_excess_return").median().alias("median_excess_return"),
                pl.col("positive").mean().alias("positive_rate"),
            )
            .sort("bucket")
        )
        means = bucketed.get_column("mean_excess_return").to_list()
        signs = [
            1.0 if left > right else -1.0 if left < right else 0.0
            for i, left in enumerate(means)
            for right in means[i + 1 :]
        ]
        tau = sum(signs) / len(signs) if signs else 0.0
        for row in bucketed.to_dicts():
            rows.append({**dict(zip(groups, key, strict=True)), "decile_tau": tau, **row})
    return pl.from_dicts(rows, infer_schema_length=None).sort(*groups, "bucket")


def path_extreme_events(
    scored: pl.DataFrame,
    *,
    group_cols: Sequence[str],
    tail_count: int = 50,
    material_floor: float = 10.0,
    source_tokens: tuple[str, ...] = SOURCE_REVIEW_TOKENS,
) -> pl.DataFrame:
    """List material extreme rows in top/bottom score tails without filtering them out."""
    rows = []
    groups = list(group_cols)
    source_cols = [
        column
        for column in scored.columns
        if column.startswith(("base__", "ctx__"))
        and any(token in column for token in source_tokens)
    ]
    keep_cols = [
        *groups,
        "symbol",
        "decision_bar_close_ms",
        "horizon_hours",
        "trend_score",
        "predicted_side",
        "path_prob_smooth_up",
        "path_prob_smooth_down",
        "final_return",
        "market_return",
        "trend_excess_return",
        "abs_trend_excess_return",
        *source_cols,
    ]
    for _key, frame in scored.partition_by(*groups, as_dict=True).items():
        threshold = max(
            float(frame.select(pl.col("abs_trend_excess_return").quantile(0.99)).item() or 0.0),
            float(material_floor),
        )
        marked = (
            frame.sort("trend_score", descending=True)
            .with_row_index("top_rank", offset=1)
            .sort("trend_score")
            .with_row_index("bottom_rank", offset=1)
            .filter(
                (pl.col("abs_trend_excess_return") >= threshold)
                & ((pl.col("top_rank") <= tail_count) | (pl.col("bottom_rank") <= tail_count))
            )
            .select("top_rank", "bottom_rank", *keep_cols)
        )
        if not marked.is_empty():
            rows.extend(marked.to_dicts())
    return pl.from_dicts(rows, infer_schema_length=None) if rows else pl.DataFrame()


def path_source_feature_health(
    matrix: pl.DataFrame,
    *,
    feature_sets: Mapping[str, object],
    source_tokens: tuple[str, ...] = SOURCE_REVIEW_TOKENS,
) -> pl.DataFrame:
    """Report source feature coverage for each manifest; does not gate rows or columns."""
    rows = []
    splits = {"build": matrix}
    if "decision_bar_close_ms" in matrix.columns and matrix.height:
        times = matrix.select(pl.col("decision_bar_close_ms").unique().sort()).to_series().to_list()
        if len(times) >= 5:
            cutoff = int(times[int(len(times) * 0.8)])
            splits["train80"] = matrix.filter(pl.col("decision_bar_close_ms") < cutoff)
            splits["blind20"] = matrix.filter(pl.col("decision_bar_close_ms") >= cutoff)
    for feature_set, manifest in feature_sets.items():
        for column in [c for c in manifest.selected_columns if any(t in c for t in source_tokens)]:
            for split, frame in splits.items():
                values = frame.select(pl.col(column).cast(pl.Float64, strict=False).alias("value"))
                rows.append(
                    {
                        "feature_set": feature_set,
                        "split": split,
                        "feature": column,
                        "row_count": frame.height,
                        "non_null_rate": values.select(pl.col("value").is_not_null().mean()).item(),
                        "finite_rate": values.select(
                            pl.col("value").is_finite().fill_null(False).mean()
                        ).item(),
                        "unique_count": values.select(pl.col("value").n_unique()).item(),
                        "mean": values.select(pl.col("value").mean()).item(),
                    }
                )
    return pl.from_dicts(rows, infer_schema_length=None).sort("feature_set", "split", "feature")


def _analysis_rows_from_profit(
    profit: pl.DataFrame, *, feature_set: str
) -> list[dict[str, object]]:
    if profit.is_empty():
        return []
    wanted = {
        "weighted_spearman_mean",
        "precision_at_k",
        "mean_excess_return",
        "top_bottom_spread",
        "profit_factor_is_reliable",
        "sortino_is_reliable",
    }
    frame = profit.filter(
        (pl.col("side") == "trend")
        & pl.col("metric").is_in(wanted)
        & ((pl.col("k") == 10) | pl.col("k").is_null())
    )
    rows = []
    for row in frame.to_dicts():
        metric = str(row["metric"])
        value = row.get("value")
        warning = None
        action = "keep"
        if metric == "top_bottom_spread" and value is not None and float(value) < 0.0:
            warning = "negative_raw_spread"
            action = "inspect_robust"
        elif metric.endswith("is_reliable") and not bool(value):
            warning = "insufficient_loss_samples"
            action = "ignore_risk_proxy"
        rows.append(
            {
                "section": "promotion",
                "feature_set": feature_set,
                "split": row["source_run_id"],
                "metric": metric,
                "k": row.get("k"),
                "value": value,
                "sample_count": row.get("sample_count"),
                "warning": warning,
                "action": action,
            }
        )
    return rows


def _analysis_rows_from_robust(robust: pl.DataFrame) -> list[dict[str, object]]:
    if robust.is_empty():
        return []
    metrics = (
        "raw_top_bottom_spread",
        "winsorized_top_bottom_spread",
        "raw_minus_winsorized_spread",
        "precision_at_k",
    )
    rows = []
    for row in robust.to_dicts():
        for metric in metrics:
            value = row.get(metric)
            warning = None
            action = "compare"
            if (
                metric == "raw_minus_winsorized_spread"
                and value is not None
                and abs(float(value)) > 1.0
            ):
                warning = "outlier_sensitive_raw_spread"
                action = "prefer_winsorized"
            elif (
                metric == "winsorized_top_bottom_spread"
                and value is not None
                and float(value) < 0.0
            ):
                warning = "negative_robust_spread"
                action = "do_not_promote"
            rows.append(
                {
                    "section": "robust",
                    "feature_set": row["feature_set"],
                    "split": row["split"],
                    "metric": metric,
                    "k": row["k"],
                    "value": value,
                    "sample_count": row.get("sample_count"),
                    "warning": warning,
                    "action": action,
                }
            )
    return rows


def _analysis_rows_from_deciles(deciles: pl.DataFrame) -> list[dict[str, object]]:
    if deciles.is_empty():
        return []
    frame = deciles.group_by("feature_set", "split").agg(
        pl.col("decile_tau").first().alias("decile_tau"),
        pl.col("sample_count").sum().alias("sample_count"),
    )
    return [
        {
            "section": "monotonicity",
            "feature_set": row["feature_set"],
            "split": row["split"],
            "metric": "decile_tau",
            "k": None,
            "value": row["decile_tau"],
            "sample_count": row["sample_count"],
            "warning": "weak_monotonicity" if float(row["decile_tau"] or 0.0) < 0.0 else None,
            "action": "inspect" if float(row["decile_tau"] or 0.0) < 0.0 else "keep",
        }
        for row in frame.to_dicts()
    ]


def _analysis_rows_from_source_health(source_health: pl.DataFrame) -> list[dict[str, object]]:
    if source_health.is_empty():
        return []
    frame = source_health.group_by("feature_set", "split").agg(
        pl.len().alias("source_feature_count"),
        ((pl.col("non_null_rate") < 0.10) | (pl.col("finite_rate") < 0.10))
        .sum()
        .cast(pl.Int64)
        .alias("low_coverage_count"),
        pl.col("non_null_rate").min().alias("min_non_null_rate"),
        pl.col("finite_rate").min().alias("min_finite_rate"),
    )
    rows = []
    for row in frame.to_dicts():
        warning = "source_coverage_warning" if int(row["low_coverage_count"] or 0) else None
        rows.extend(
            [
                {
                    "section": "source_health",
                    "feature_set": row["feature_set"],
                    "split": row["split"],
                    "metric": "source_feature_count",
                    "k": None,
                    "value": row["source_feature_count"],
                    "sample_count": row["source_feature_count"],
                    "warning": None,
                    "action": "keep",
                },
                {
                    "section": "source_health",
                    "feature_set": row["feature_set"],
                    "split": row["split"],
                    "metric": "low_coverage_feature_count",
                    "k": None,
                    "value": row["low_coverage_count"],
                    "sample_count": row["source_feature_count"],
                    "warning": warning,
                    "action": "inspect" if warning else "keep",
                },
                {
                    "section": "source_health",
                    "feature_set": row["feature_set"],
                    "split": row["split"],
                    "metric": "min_finite_rate",
                    "k": None,
                    "value": row["min_finite_rate"],
                    "sample_count": row["source_feature_count"],
                    "warning": warning,
                    "action": "inspect" if warning else "keep",
                },
            ]
        )
    return rows


def _analysis_rows_from_extremes(extremes: pl.DataFrame) -> list[dict[str, object]]:
    if extremes.is_empty():
        return []
    frame = extremes.group_by("feature_set", "split").agg(
        pl.len().alias("extreme_event_count"),
        pl.col("abs_trend_excess_return").max().alias("max_abs_excess_return"),
    )
    rows = []
    for row in frame.to_dicts():
        rows.extend(
            [
                {
                    "section": "extreme_events",
                    "feature_set": row["feature_set"],
                    "split": row["split"],
                    "metric": "extreme_event_count",
                    "k": None,
                    "value": row["extreme_event_count"],
                    "sample_count": row["extreme_event_count"],
                    "warning": "extreme_events_present"
                    if int(row["extreme_event_count"] or 0)
                    else None,
                    "action": "inspect_rows",
                },
                {
                    "section": "extreme_events",
                    "feature_set": row["feature_set"],
                    "split": row["split"],
                    "metric": "max_abs_excess_return",
                    "k": None,
                    "value": row["max_abs_excess_return"],
                    "sample_count": row["extreme_event_count"],
                    "warning": "extreme_events_present"
                    if int(row["extreme_event_count"] or 0)
                    else None,
                    "action": "inspect_rows",
                },
            ]
        )
    return rows


def path_feature_analysis(
    *,
    feature_set_counts: Mapping[str, int],
    profit: pl.DataFrame | None = None,
    robust: pl.DataFrame | None = None,
    deciles: pl.DataFrame | None = None,
    source_health: pl.DataFrame | None = None,
    extreme_events: pl.DataFrame | None = None,
    source_probe: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Compress feature-build review frames into one compact analysis table."""
    rows: list[dict[str, object]] = [
        {
            "section": "selection",
            "feature_set": feature_set,
            "split": "build",
            "metric": "selected_count",
            "k": None,
            "value": count,
            "sample_count": count,
            "warning": None,
            "action": "keep",
        }
        for feature_set, count in feature_set_counts.items()
    ]
    rows.extend(
        _analysis_rows_from_profit(
            profit if profit is not None else pl.DataFrame(), feature_set="base_ndcg_current"
        )
    )
    rows.extend(_analysis_rows_from_robust(robust if robust is not None else pl.DataFrame()))
    rows.extend(_analysis_rows_from_deciles(deciles if deciles is not None else pl.DataFrame()))
    rows.extend(
        _analysis_rows_from_source_health(
            source_health if source_health is not None else pl.DataFrame()
        )
    )
    rows.extend(
        _analysis_rows_from_extremes(
            extreme_events if extreme_events is not None else pl.DataFrame()
        )
    )
    if source_probe is not None and not source_probe.is_empty():
        rows.extend(source_probe.to_dicts())
    return _select_schema(
        pl.from_dicts(rows, infer_schema_length=None), PATH_FEATURE_ANALYSIS_SCHEMA
    )


def _analysis_value(
    analysis: pl.DataFrame, *, feature_set: str, split: str, metric: str, k: int | None = None
) -> float | None:
    frame = analysis.filter(
        (pl.col("feature_set") == feature_set)
        & (pl.col("split") == split)
        & (pl.col("metric") == metric)
        & (pl.col("k").is_null() if k is None else pl.col("k") == k)
    )
    if frame.is_empty():
        return None
    value = frame.select("value").item()
    return None if value is None else float(value)


def path_feature_analysis_report(analysis: pl.DataFrame) -> str:
    """Render compact feature-build analysis as a small human report."""
    warnings = analysis.filter(pl.col("warning").is_not_null())
    feature_sets = ", ".join(analysis.get_column("feature_set").unique().sort().to_list())
    base_spread = _analysis_value(
        analysis,
        feature_set="base_ndcg_current",
        split="promotion_blind",
        metric="winsorized_top_bottom_spread",
        k=10,
    )
    source_spread = _analysis_value(
        analysis,
        feature_set="source_blended_all",
        split="promotion_blind",
        metric="winsorized_top_bottom_spread",
        k=10,
    )
    source_tau = _analysis_value(
        analysis,
        feature_set="source_blended_all",
        split="promotion_blind",
        metric="decile_tau",
    )
    decision = "keep_reviewing"
    if (
        source_spread is not None
        and source_spread > 0.0
        and (source_tau is None or source_tau >= 0.0)
    ):
        decision = "source_context_candidate"
    if base_spread is not None and base_spread < 0.0:
        decision = "do_not_promote_base"
    lines = [
        "# Tailtree Feature Analysis",
        "",
        "## Build inputs",
        "",
        f"- feature_sets: {feature_sets}",
        f"- warning_rows: {warnings.height}",
        "",
        "## Selected feature set",
        "",
    ]
    selected = analysis.filter(
        (pl.col("section") == "selection") & (pl.col("metric") == "selected_count")
    )
    for row in selected.to_dicts():
        lines.append(f"- {row['feature_set']}: {int(row['value'])} selected features")
    lines.extend(["", "## Promotion feedback", ""])
    for feature_set in ("base_ndcg_current", "source_blended_all"):
        spread = _analysis_value(
            analysis,
            feature_set=feature_set,
            split="promotion_blind",
            metric="winsorized_top_bottom_spread",
            k=10,
        )
        tau = _analysis_value(
            analysis,
            feature_set=feature_set,
            split="promotion_blind",
            metric="decile_tau",
        )
        if spread is not None or tau is not None:
            lines.append(f"- {feature_set}: winsorized_spread@10={spread}, decile_tau={tau}")
    lines.extend(["", "## Source probe", ""])
    source_probe = analysis.filter(pl.col("section") == "source_probe")
    if source_probe.is_empty():
        lines.append("- no source-probe rows")
    else:
        for row in source_probe.sort("metric").to_dicts():
            lines.append(
                f"- {row['metric']}={row['value']} "
                f"sample_count={row.get('sample_count')} action={row.get('action')}"
            )
    lines.extend(["", "## Source coverage", ""])
    source_rows = analysis.filter(pl.col("section") == "source_health")
    if source_rows.is_empty():
        lines.append("- no source-health rows")
    else:
        for row in source_rows.filter(pl.col("metric") == "low_coverage_feature_count").to_dicts():
            lines.append(
                f"- {row['feature_set']}/{row['split']}: "
                f"low_coverage_features={int(row['value'] or 0)}"
            )
    lines.extend(["", "## Decision / next action", "", f"- decision: {decision}"])
    if not warnings.is_empty():
        lines.append("- inspect warning rows in `feature-analysis.csv`")
    return "\n".join(lines)


__all__ = [
    "PATH_BLACKLIST_PROPOSAL_SCHEMA",
    "PATH_FEATURE_ANALYSIS_SCHEMA",
    "PATH_FEATURE_IMPORTANCE_SCHEMA",
    "PATH_FEATURE_MATRIX_REVIEW_SCHEMA",
    "PATH_FEATURE_PSI_SCHEMA",
    "PATH_SHAP_REVIEW_SCHEMA",
    "path_feature_blacklist",
    "path_source_feature_health",
    "path_scored_trend_frame",
    "path_robust_profit_metrics",
    "path_rank_buckets",
    "path_feature_analysis_report",
    "path_feature_analysis",
    "path_extreme_events",
    "path_decile_monotonicity",
    "path_feature_importance",
    "path_feature_matrix_review",
    "path_feature_psi",
    "path_shap_review",
]
