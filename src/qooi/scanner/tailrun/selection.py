"""Tailtree scored-candidate replay and selection metric products."""

from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING

import polars as pl
from pydantic import BaseModel, ConfigDict, computed_field

from qooi.scanner.tailrun.types import (
    TailtreeArtifactTree,
    TailtreeCandidateGateSpec,
    TailtreeDirection,
    TailtreePreparedFrames,
    TailtreeSelectionEfficiencyRow,
)

if TYPE_CHECKING:
    from qooi.scanner.tailrun.planning import TailtreeProfileRun



def candidate_gate_frame(
    scored_population: pl.DataFrame,
    gates: tuple[TailtreeCandidateGateSpec, ...],
) -> pl.DataFrame:
    """Mark opportunity rows that fall inside candidate gates."""
    if scored_population.is_empty() or not gates:
        return pl.DataFrame()
    base = scored_population.filter(
        (pl.col("objective") == "tail_event_lift") & (pl.col("direction") == "up")
    ).unique(subset=["symbol", "decision_bar_close_ms", "outcome_horizon"], maintain_order=True)
    if base.is_empty():
        return pl.DataFrame()
    ranked = base.sort("tailtree_score", descending=True).with_row_index("candidate_rank")
    total = len(ranked)
    frames: list[pl.DataFrame] = []
    for gate in gates:
        cutoff = (
            max(1, ceil(total * gate.value / 100.0))
            if gate.family == "score_pct"
            else min(max(1, int(gate.value)), total)
        )
        frames.append(
            ranked.with_columns(
                pl.lit(gate.gate_id).alias("candidate_gate_id"),
                pl.lit(gate.family).alias("candidate_gate_family"),
                pl.lit(float(gate.value)).alias("candidate_gate_value"),
                (pl.col("candidate_rank") < cutoff).alias("in_candidate_gate"),
            ).drop("candidate_rank")
        )
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def promoter_target_frame(candidate_gates: pl.DataFrame) -> pl.DataFrame:
    """Attach candidate-conditional promoter labels with a gray abstain class."""
    if candidate_gates.is_empty():
        return candidate_gates
    promotable = (
        pl.col("selected_behavior_actionable").fill_null(False).cast(pl.Boolean)
        & ~pl.col("selected_behavior_false_direction").fill_null(False).cast(pl.Boolean)
        & (pl.col("selected_behavior_blocker").fill_null("") == "")
        & (pl.col("selected_behavior_utility_margin").fill_null(0.0).cast(pl.Float64) > 0.0)
        & (pl.col("selected_behavior_actionability").fill_null("") == "tradable_up")
    )
    reject = (
        pl.col("selected_behavior_false_direction").fill_null(False).cast(pl.Boolean)
        | pl.col("selected_behavior_blocker")
        .fill_null("")
        .is_in(["opposite_clean_path", "opposite_tail_dominates", "both_or_mixed_path"])
        | (pl.col("selected_behavior_utility_margin").fill_null(0.0).cast(pl.Float64) <= 0.0)
    )
    return candidate_gates.with_columns(
        promotable.alias("promotable_up"),
        (~promotable & reject).alias("reject_up"),
        (~promotable & ~reject).alias("gray_up"),
        pl.when(promotable).then(1).when(reject).then(0).otherwise(None).alias("promoter_label"),
        pl.when(promotable | reject)
        .then(
            1.0
            + pl.col("selected_behavior_utility_margin").fill_null(0.0).cast(pl.Float64).abs()
        )
        .otherwise(0.0)
        .alias("promoter_weight"),
    )


def opposite_guard_target_frame(candidate_gates: pl.DataFrame) -> pl.DataFrame:
    """Attach candidate-local direct opposite-path guard labels."""
    if candidate_gates.is_empty():
        return candidate_gates
    opposite = (
        (pl.col("selected_behavior_path_state").fill_null("") == "clean_down")
        | (pl.col("selected_behavior_actionability").fill_null("") == "tradable_down")
        | (pl.col("selected_behavior_blocker").fill_null("") == "opposite_clean_path")
    )
    return candidate_gates.with_columns(
        opposite.cast(pl.Int8).alias("opposite_guard_label"),
        pl.lit(1.0).alias("opposite_guard_weight"),
    )


def weak_path_guard_target_frame(candidate_gates: pl.DataFrame) -> pl.DataFrame:
    """Attach candidate-local weak/no-tail abstain labels."""
    if candidate_gates.is_empty():
        return candidate_gates
    weak = (
        (pl.col("selected_behavior_blocker").fill_null("") == "no_tail_touch")
        | (pl.col("selected_behavior_path_state").fill_null("") == "none")
        | (pl.col("selected_behavior_utility_margin").fill_null(0.0).cast(pl.Float64) <= 0.0)
    )
    return candidate_gates.with_columns(
        weak.cast(pl.Int8).alias("weak_path_guard_label"),
        pl.lit(1.0).alias("weak_path_guard_weight"),
    )


def actionability_contradiction_audit_frame(scored_promotions: pl.DataFrame) -> pl.DataFrame:
    """Audit rows where actionability conflicts with path/blocker labels."""
    required = {
        "candidate_gate_id",
        "promotion_score",
        "opposite_guard_score",
        "weak_path_guard_score",
        "selected_behavior_actionable",
        "selected_behavior_false_direction",
        "selected_behavior_utility_margin",
        "selected_behavior_path_state",
        "selected_behavior_actionability",
        "selected_behavior_blocker",
    }
    if scored_promotions.is_empty() or not required.issubset(scored_promotions.columns):
        return pl.DataFrame()
    frame = scored_promotions.with_columns(
        pl.col("selected_behavior_actionable").fill_null(False).alias("actionable"),
        pl.col("selected_behavior_false_direction").fill_null(False).alias("false_direction"),
        pl.col("selected_behavior_path_state").fill_null("none").alias("path_state"),
        pl.col("selected_behavior_actionability").fill_null("no_action").alias("actionability"),
        pl.col("selected_behavior_blocker").fill_null("").alias("blocker"),
    )
    no_tail = (pl.col("path_state") == "none") | (pl.col("blocker") == "no_tail_touch")
    clean_actionable = (
        pl.col("actionable")
        & (pl.col("path_state") == "clean_up")
        & (pl.col("actionability") == "tradable_up")
        & (pl.col("blocker") == "")
    )
    opposite_risk = (
        pl.col("false_direction")
        | (pl.col("path_state") == "clean_down")
        | (pl.col("actionability") == "tradable_down")
        | (pl.col("blocker") == "opposite_clean_path")
    )
    bucketed = frame.with_columns(
        pl.when(pl.col("actionable") & no_tail)
        .then(pl.lit("contradictory_actionable_no_tail"))
        .when(clean_actionable)
        .then(pl.lit("clean_actionable"))
        .when(opposite_risk)
        .then(pl.lit("opposite_risk"))
        .when(~pl.col("actionable") & no_tail)
        .then(pl.lit("blocked_negative"))
        .otherwise(pl.lit("other"))
        .alias("audit_bucket")
    )
    return bucketed.group_by(
        "candidate_gate_id",
        "audit_bucket",
        "path_state",
        "actionability",
        "blocker",
    ).agg(
        pl.len().alias("row_count"),
        pl.col("actionable").sum().alias("actionable_count"),
        pl.col("false_direction").sum().alias("false_direction_count"),
        pl.col("selected_behavior_utility_margin").mean().alias("utility_margin_mean"),
        pl.col("selected_behavior_utility_margin").median().alias("utility_margin_median"),
        pl.col("selected_behavior_utility_margin").min().alias("utility_margin_min"),
        pl.col("selected_behavior_utility_margin").max().alias("utility_margin_max"),
        pl.col("promotion_score").mean().alias("promotion_score_mean"),
        pl.col("opposite_guard_score").mean().alias("opposite_guard_score_mean"),
        pl.col("weak_path_guard_score").mean().alias("weak_path_guard_score_mean"),
    )


def dual_guard_boundary_anatomy_frame(
    scored_promotions: pl.DataFrame,
    *,
    opposite_keep_pct: float = 50.0,
    weak_keep_pct: float = 90.0,
    selected_count: int = 50,
    high_confidence_count: int = 25,
) -> pl.DataFrame:
    """Compare high-confidence rows, expansion rows, and missed actionable positives."""
    required = {
        "candidate_gate_id",
        "promotion_score",
        "opposite_guard_score",
        "weak_path_guard_score",
        "selected_behavior_actionable",
        "selected_behavior_false_direction",
        "selected_behavior_utility_margin",
        "selected_behavior_path_state",
        "selected_behavior_actionability",
        "selected_behavior_blocker",
    }
    if scored_promotions.is_empty() or not required.issubset(scored_promotions.columns):
        return pl.DataFrame()
    frames: list[pl.DataFrame] = []
    for gate_id in scored_promotions.get_column("candidate_gate_id").unique().to_list():
        gate = scored_promotions.filter(pl.col("candidate_gate_id") == gate_id)
        if gate.is_empty():
            continue
        opposite_count = (
            len(gate)
            if opposite_keep_pct >= 100.0
            else max(1, ceil(len(gate) * opposite_keep_pct / 100.0))
        )
        opposite_kept = gate.sort("opposite_guard_score").head(opposite_count)
        weak_count = (
            len(opposite_kept)
            if weak_keep_pct >= 100.0
            else max(1, ceil(len(opposite_kept) * weak_keep_pct / 100.0))
        )
        ranked = (
            opposite_kept.sort("weak_path_guard_score")
            .head(weak_count)
            .sort("promotion_score", descending=True)
            .with_row_index("final_rank", offset=1)
        )
        bucketed = ranked.with_columns(
            pl.when(pl.col("final_rank") <= high_confidence_count)
            .then(pl.lit("high_confidence_selected"))
            .when(pl.col("final_rank") <= selected_count)
            .then(pl.lit("expansion_selected"))
            .when(pl.col("selected_behavior_actionable").fill_null(False))
            .then(pl.lit("missed_actionable"))
            .otherwise(pl.lit("unselected_negative"))
            .alias("boundary_bucket"),
            pl.col("selected_behavior_path_state").fill_null("none").alias("path_state"),
            pl.col("selected_behavior_actionability").fill_null("no_action").alias("actionability"),
            pl.col("selected_behavior_blocker").fill_null("").alias("blocker"),
        )
        frames.append(
            bucketed.group_by(
                "candidate_gate_id",
                "boundary_bucket",
                "path_state",
                "actionability",
                "blocker",
            ).agg(
                pl.len().alias("row_count"),
                pl.col("selected_behavior_actionable").fill_null(False).sum().alias("actionable_count"),
                pl.col("selected_behavior_false_direction")
                .fill_null(False)
                .sum()
                .alias("false_direction_count"),
                pl.col("selected_behavior_utility_margin").mean().alias("utility_margin_mean"),
                pl.col("promotion_score").mean().alias("promotion_score_mean"),
                pl.col("opposite_guard_score").mean().alias("opposite_guard_score_mean"),
                pl.col("weak_path_guard_score").mean().alias("weak_path_guard_score_mean"),
                pl.col("final_rank").min().alias("rank_min"),
                pl.col("final_rank").max().alias("rank_max"),
            ).with_columns(
                pl.lit(float(opposite_keep_pct)).alias("opposite_keep_pct"),
                pl.lit(float(weak_keep_pct)).alias("weak_keep_pct"),
                pl.lit(int(selected_count)).alias("selected_count"),
                pl.lit(int(high_confidence_count)).alias("high_confidence_count"),
            )
        )
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def dual_guarded_promotion_selection_metrics_frame(
    scored_promotions: pl.DataFrame,
    *,
    opposite_keep_pcts: tuple[float, ...] = (50.0, 75.0, 90.0, 100.0),
    weak_keep_pcts: tuple[float, ...] = (50.0, 75.0, 90.0, 100.0),
    top_k_buckets: tuple[int, ...] = (50, 100, 200),
    pct_buckets: tuple[float, ...] = (10.0, 25.0, 50.0),
) -> pl.DataFrame:
    """Selection-efficiency rows after opposite and weak-path guard filtering."""
    required = {
        "candidate_gate_id",
        "promotion_score",
        "opposite_guard_score",
        "weak_path_guard_score",
    }
    if scored_promotions.is_empty() or not required.issubset(scored_promotions.columns):
        return pl.DataFrame()
    rows: list[dict[str, object]] = []
    for gate_id in scored_promotions.get_column("candidate_gate_id").unique().to_list():
        gate = scored_promotions.filter(pl.col("candidate_gate_id") == gate_id)
        total = len(gate)
        for opposite_keep_pct in opposite_keep_pcts:
            opposite_count = (
                total
                if opposite_keep_pct >= 100.0
                else max(1, ceil(total * opposite_keep_pct / 100.0))
            )
            opposite_kept = gate.sort("opposite_guard_score").head(opposite_count)
            for weak_keep_pct in weak_keep_pcts:
                weak_count = (
                    len(opposite_kept)
                    if weak_keep_pct >= 100.0
                    else max(1, ceil(len(opposite_kept) * weak_keep_pct / 100.0))
                )
                kept = opposite_kept.sort("weak_path_guard_score").head(weak_count)
                for top_k in top_k_buckets:
                    row = _promotion_selection_row(
                        kept.sort("promotion_score", descending=True),
                        selected_count=min(int(top_k), len(kept)),
                        budget_family="top_k",
                        budget_value=float(top_k),
                    )
                    if row:
                        labels = DualGuardRowLabels(
                            gate_id=str(gate_id),
                            opposite_keep_pct=float(opposite_keep_pct),
                            weak_keep_pct=float(weak_keep_pct),
                            budget_family="top_k",
                            budget_value=float(top_k),
                        )
                        rows.append({**row, **labels.model_dump()})
                for pct in pct_buckets:
                    row = _promotion_selection_row(
                        kept.sort("promotion_score", descending=True),
                        selected_count=max(1, ceil(len(kept) * float(pct) / 100.0)),
                        budget_family="gate_pct",
                        budget_value=float(pct),
                    )
                    if row:
                        labels = DualGuardRowLabels(
                            gate_id=str(gate_id),
                            opposite_keep_pct=float(opposite_keep_pct),
                            weak_keep_pct=float(weak_keep_pct),
                            budget_family="gate_pct",
                            budget_value=float(pct),
                        )
                        rows.append({**row, **labels.model_dump()})
    return pl.DataFrame(rows) if rows else pl.DataFrame()


class DualGuardRowLabels(BaseModel):
    model_config = ConfigDict(frozen=True)

    gate_id: str
    opposite_keep_pct: float
    weak_keep_pct: float
    budget_family: str
    budget_value: float
    model_tag: str = "tailtree-candidate-dual-guard"
    objective: str = "candidate_dual_guard"
    training_profile: str = "tail-event-lift-candidate-dual-guard"
    trial_id: str = "candidate-dual-guard"
    trial_source: str = "derived"

    @computed_field
    @property
    def score_bucket(self) -> str:
        return (
            f"{self.gate_id}_opp_keep_{self.opposite_keep_pct:g}_"
            f"weak_keep_{self.weak_keep_pct:g}_{self.budget_family}_{self.budget_value:g}"
        )


class PromotionSelectionMetricRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    universe_snapshot_id: str = "active"
    model_tag: str = "tailtree-candidate-conditional-promoter"
    objective: str = "candidate_conditional_promoter"
    training_profile: str = "tail-event-lift-candidate-promoter"
    trial_id: str = "candidate-conditional-promoter"
    trial_source: str = "derived"
    outcome_horizon: int = 24
    tree_direction: str = "up"
    candidate_gate_id: str
    candidate_gate_family: str
    candidate_gate_value: float
    budget_family: str
    budget_value: float
    score_bucket: str
    selected_observation_count: float
    candidate_pair_count: float
    selected_tail_count: float
    valid_tail_count: float
    selected_tail_rate: float
    base_hpo_score: float
    hpo_score: float
    objective_hpo_score: float
    side_hpo_score: float
    behavior_hpo_score: float
    paired_behavior_false_direction_rate: float
    paired_behavior_utility_margin_mean: float
    behavior_tp_count: float
    behavior_fp_count: float
    behavior_fn_count: float
    behavior_tn_count: float
    behavior_precision: float
    behavior_recall: float
    behavior_specificity: float
    behavior_false_positive_rate: float
    behavior_false_negative_rate: float
    behavior_accuracy: float


def guarded_selection_error_anatomy_frame(
    scored_promotions: pl.DataFrame,
    *,
    guard_keep_pcts: tuple[float, ...] = (50.0, 75.0, 90.0, 100.0),
    top_k_buckets: tuple[int, ...] = (50, 100, 200),
    pct_buckets: tuple[float, ...] = (10.0, 25.0, 50.0),
) -> pl.DataFrame:
    """Reason distribution after candidate-local opposite-guard filtering."""
    required = {"candidate_gate_id", "promotion_score", "opposite_guard_score"}
    if scored_promotions.is_empty() or not required.issubset(scored_promotions.columns):
        return pl.DataFrame()
    rows: list[pl.DataFrame] = []
    for gate_id in scored_promotions.get_column("candidate_gate_id").unique().to_list():
        gate = scored_promotions.filter(pl.col("candidate_gate_id") == gate_id)
        total = len(gate)
        for keep_pct in guard_keep_pcts:
            keep_count = total if keep_pct >= 100.0 else max(1, ceil(total * keep_pct / 100.0))
            kept = gate.sort("opposite_guard_score").head(keep_count).sort(
                "promotion_score", descending=True
            )
            for top_k in top_k_buckets:
                score_bucket = f"{gate_id}_guard_keep_{keep_pct:g}_top_k_{float(top_k):g}"
                rows.append(
                    _selection_error_anatomy_bucket(
                        kept,
                        selected_count=min(int(top_k), len(kept)),
                        score_bucket=score_bucket,
                    ).with_columns(
                        pl.lit("candidate_opposite_guard").alias("objective"),
                        pl.lit(float(keep_pct)).alias("guard_keep_pct"),
                    )
                )
            for pct in pct_buckets:
                score_bucket = f"{gate_id}_guard_keep_{keep_pct:g}_gate_pct_{float(pct):g}"
                rows.append(
                    _selection_error_anatomy_bucket(
                        kept,
                        selected_count=max(1, ceil(len(kept) * float(pct) / 100.0)),
                        score_bucket=score_bucket,
                    ).with_columns(
                        pl.lit("candidate_opposite_guard").alias("objective"),
                        pl.lit(float(keep_pct)).alias("guard_keep_pct"),
                    )
                )
    frames = [frame for frame in rows if not frame.is_empty()]
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def _selection_metric_defaults(frame: pl.DataFrame) -> pl.DataFrame:
    columns = frame.columns
    expressions = []
    if "candidate_gate_family" not in columns:
        expressions.append(pl.lit("test").alias("candidate_gate_family"))
    if "candidate_gate_value" not in columns:
        expressions.append(pl.lit(0.0).alias("candidate_gate_value"))
    if "selected_tail" not in columns:
        expressions.append(pl.lit(False).alias("selected_tail"))
    if expressions:
        return frame.with_columns(*expressions)
    return frame


def selection_error_anatomy_frame(
    scored_promotions: pl.DataFrame,
    *,
    top_k_buckets: tuple[int, ...] = (50, 100, 200),
    pct_buckets: tuple[float, ...] = (10.0, 25.0, 50.0),
) -> pl.DataFrame:
    """Reason distribution for selected candidate-promoter TP/FP/false-direction rows."""
    schema = {
        "candidate_gate_id": pl.String,
        "score_bucket": pl.String,
        "error_family": pl.String,
        "path_state": pl.String,
        "actionability": pl.String,
        "blocker": pl.String,
        "false_direction": pl.Boolean,
        "row_count": pl.Float64,
        "tp_count": pl.Float64,
        "fp_count": pl.Float64,
        "false_direction_count": pl.Float64,
        "precision": pl.Float64,
        "false_direction_rate": pl.Float64,
        "utility_margin_mean": pl.Float64,
    }
    required = {
        "candidate_gate_id",
        "promotion_score",
        "selected_behavior_actionable",
        "selected_behavior_false_direction",
        "selected_behavior_path_state",
        "selected_behavior_actionability",
        "selected_behavior_blocker",
        "selected_behavior_utility_margin",
    }
    if scored_promotions.is_empty() or not required.issubset(scored_promotions.columns):
        return pl.DataFrame(schema=schema)

    rows: list[pl.DataFrame] = []
    for gate_id in scored_promotions.get_column("candidate_gate_id").unique().to_list():
        gate = scored_promotions.filter(pl.col("candidate_gate_id") == gate_id).sort(
            "promotion_score", descending=True
        )
        total = len(gate)
        for top_k in top_k_buckets:
            rows.append(
                _selection_error_anatomy_bucket(
                    gate,
                    selected_count=min(int(top_k), total),
                    score_bucket=f"{gate_id}_top_k_{top_k:g}",
                )
            )
        for pct in pct_buckets:
            rows.append(
                _selection_error_anatomy_bucket(
                    gate,
                    selected_count=max(1, ceil(total * float(pct) / 100.0)),
                    score_bucket=f"{gate_id}_gate_pct_{pct:g}",
                )
            )
    frames = [frame for frame in rows if not frame.is_empty()]
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame(schema=schema)


def _selection_error_anatomy_bucket(
    gate: pl.DataFrame,
    *,
    selected_count: int,
    score_bucket: str,
) -> pl.DataFrame:
    if gate.is_empty() or selected_count <= 0:
        return pl.DataFrame()
    selected = pl.col("error_selected")
    tp = pl.col("error_tp")
    false_direction = pl.col("error_false_direction")
    frame = (
        gate.with_row_index("promotion_rank")
        .with_columns(
            (pl.col("promotion_rank") < int(selected_count)).alias("error_selected"),
            pl.col("selected_behavior_actionable").fill_null(False).cast(pl.Boolean).alias("error_tp"),
            pl.col("selected_behavior_false_direction")
            .fill_null(False)
            .cast(pl.Boolean)
            .alias("error_false_direction"),
        )
        .with_columns(
            pl.when(selected & false_direction)
            .then(pl.lit("false_direction"))
            .when(selected & tp)
            .then(pl.lit("selected_tp"))
            .when(selected)
            .then(pl.lit("selected_fp"))
            .otherwise(pl.lit("unselected_candidate"))
            .alias("error_family"),
            pl.lit(score_bucket).alias("score_bucket"),
            pl.col("selected_behavior_path_state").fill_null("missing").alias("path_state"),
            pl.col("selected_behavior_actionability").fill_null("missing").alias("actionability"),
            pl.col("selected_behavior_blocker").fill_null("").alias("blocker"),
        )
    )
    return (
        frame.group_by(
            "candidate_gate_id",
            "score_bucket",
            "error_family",
            "path_state",
            "actionability",
            "blocker",
            "error_false_direction",
        )
        .agg(
            pl.len().cast(pl.Float64).alias("row_count"),
            pl.col("error_tp").cast(pl.Float64).sum().alias("tp_count"),
            (~pl.col("error_tp")).cast(pl.Float64).sum().alias("fp_count"),
            pl.col("error_false_direction").cast(pl.Float64).sum().alias("false_direction_count"),
            pl.col("selected_behavior_utility_margin")
            .fill_null(0.0)
            .cast(pl.Float64)
            .mean()
            .alias("utility_margin_mean"),
        )
        .rename({"error_false_direction": "false_direction"})
        .with_columns(
            (pl.col("tp_count") / pl.col("row_count")).alias("precision"),
            (pl.col("false_direction_count") / pl.col("row_count")).alias("false_direction_rate"),
        )
        .sort(
            ["candidate_gate_id", "score_bucket", "error_family", "row_count"],
            descending=[False, False, False, True],
        )
    )


def _promotion_selection_row(
    gate: pl.DataFrame,
    *,
    selected_count: int,
    budget_family: str,
    budget_value: float,
) -> dict[str, object]:
    if gate.is_empty() or selected_count <= 0:
        return {}
    selected_flag = "promotion_selected"
    frame = gate.with_row_index("promotion_rank").with_columns(
        (pl.col("promotion_rank") < int(selected_count)).alias(selected_flag)
    )
    confusion = _confusion_from_population(frame, selected_flag=selected_flag)
    selected = frame.filter(pl.col(selected_flag))
    false_direction_rate = float(
        selected.select(
            pl.col("selected_behavior_false_direction").fill_null(False).cast(pl.Float64).mean()
        ).item()
        or 0.0
    )
    utility_mean = float(
        selected.select(
            pl.col("selected_behavior_utility_margin").fill_null(0.0).cast(pl.Float64).mean()
        ).item()
        or 0.0
    )
    selected_tail_count = float(
        selected.select(pl.col("selected_tail").fill_null(False).cast(pl.Float64).sum()).item()
        or 0.0
    )
    gate_id = str(gate.get_column("candidate_gate_id")[0])
    base_hpo_score = (
        confusion.get("behavior_precision", 0.0)
        + utility_mean
        - 10.0 * confusion.get("behavior_false_positive_rate", 0.0)
        - false_direction_rate
    )
    return PromotionSelectionMetricRow(
        candidate_gate_id=gate_id,
        candidate_gate_family=str(gate.get_column("candidate_gate_family")[0]),
        candidate_gate_value=float(gate.get_column("candidate_gate_value")[0]),
        budget_family=budget_family,
        budget_value=budget_value,
        score_bucket=f"{gate_id}_{budget_family}_{budget_value:g}",
        selected_observation_count=float(selected_count),
        candidate_pair_count=float(selected_count),
        selected_tail_count=selected_tail_count,
        valid_tail_count=selected_tail_count,
        selected_tail_rate=selected_tail_count / max(float(selected_count), 1.0),
        base_hpo_score=base_hpo_score,
        hpo_score=base_hpo_score,
        objective_hpo_score=base_hpo_score,
        side_hpo_score=base_hpo_score,
        behavior_hpo_score=base_hpo_score,
        paired_behavior_false_direction_rate=false_direction_rate,
        paired_behavior_utility_margin_mean=utility_mean,
        behavior_tp_count=confusion.get("behavior_tp_count", 0.0),
        behavior_fp_count=confusion.get("behavior_fp_count", 0.0),
        behavior_fn_count=confusion.get("behavior_fn_count", 0.0),
        behavior_tn_count=confusion.get("behavior_tn_count", 0.0),
        behavior_precision=confusion.get("behavior_precision", 0.0),
        behavior_recall=confusion.get("behavior_recall", 0.0),
        behavior_specificity=confusion.get("behavior_specificity", 0.0),
        behavior_false_positive_rate=confusion.get("behavior_false_positive_rate", 0.0),
        behavior_false_negative_rate=confusion.get("behavior_false_negative_rate", 0.0),
        behavior_accuracy=confusion.get("behavior_accuracy", 0.0),
    ).model_dump()


class TailtreeReplayMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidate_pair_count: float = 0.0
    paired_selected_tail_rate: float = 0.0
    paired_opposite_rate: float = 0.0
    paired_gray_zone_rate: float = 0.0
    paired_false_direction_rate: float = 0.0
    paired_false_direction_cost_mean: float = 0.0
    paired_directional_margin_mean: float = 0.0
    paired_side_only_rate: float = 0.0
    paired_tail_both_rate: float = 0.0
    paired_selected_utility_margin_mean: float = 0.0
    paired_calibrated_directional_margin_mean: float = 0.0
    paired_calibrated_side_margin_mean: float = 0.0
    paired_behavior_actionable_rate: float = 0.0
    paired_behavior_false_positive_proxy_rate: float = 0.0
    paired_behavior_false_direction_rate: float = 0.0
    paired_behavior_utility_margin_mean: float = 0.0

    @computed_field
    @property
    def paired_behavior_tp_proxy_count(self) -> float:
        return self.candidate_pair_count * self.paired_behavior_actionable_rate

    @computed_field
    @property
    def paired_behavior_fp_proxy_count(self) -> float:
        return self.candidate_pair_count * self.paired_behavior_false_positive_proxy_rate

    @computed_field
    @property
    def paired_behavior_false_direction_count(self) -> float:
        return self.candidate_pair_count * self.paired_behavior_false_direction_rate

    @computed_field
    @property
    def paired_selected_tail_count(self) -> float:
        return self.candidate_pair_count * self.paired_selected_tail_rate

    def directional_objective_score(
        self,
        base_score: float,
        *,
        false_rate_weight: float = 10.0,
        false_cost_weight: float = 1.0,
        margin_weight: float = 5.0,
        gray_weight: float = 0.0,
    ) -> float:
        return (
            base_score
            + margin_weight * self.paired_directional_margin_mean
            - false_rate_weight * self.paired_false_direction_rate
            - false_cost_weight * self.paired_false_direction_cost_mean
            - gray_weight * self.paired_gray_zone_rate
        )

    def side_objective_score(
        self,
        base_score: float,
        *,
        false_rate_weight: float = 10.0,
        false_cost_weight: float = 1.0,
        margin_weight: float = 5.0,
        side_only_weight: float = 5.0,
        tail_both_weight: float = 5.0,
    ) -> float:
        return (
            base_score
            + margin_weight * self.paired_directional_margin_mean
            + side_only_weight * self.paired_side_only_rate
            - false_rate_weight * self.paired_false_direction_rate
            - false_cost_weight * self.paired_false_direction_cost_mean
            - tail_both_weight * self.paired_tail_both_rate
        )

    def behavior_objective_score(self, base_score: float) -> float:
        return (
            base_score
            + 20.0 * self.paired_behavior_actionable_rate
            + 5.0 * self.paired_behavior_utility_margin_mean
            - 10.0 * self.paired_behavior_false_positive_proxy_rate
            - 10.0 * self.paired_behavior_false_direction_rate
        )



def score_bucket_candidate_frame(
    tree: TailtreeArtifactTree,
    observations: pl.DataFrame,
    outcomes: pl.DataFrame,
    outcome_horizon: int,
    *,
    behavior_targets: pl.DataFrame | None = None,
    objective: str = "tail_event_lift",
    score_quantiles: tuple[float, ...] = (0.99, 0.98, 0.95, 0.90),
) -> pl.DataFrame:
    direction = tree.metadata.direction
    tail_col = f"tail_{direction}"
    utility_col = f"tail_utility_{direction}"
    margin_col = f"tail_utility_margin_{direction}"
    if observations.is_empty() or outcomes.is_empty() or tail_col not in outcomes.columns:
        return pl.DataFrame()

    outcome_aggs = [pl.col(tail_col).fill_null(False).cast(pl.Boolean).max().alias(tail_col)]
    outcome_aggs.append(
        pl.col(utility_col).fill_null(0.0).cast(pl.Float64).max().alias(utility_col)
        if utility_col in outcomes.columns
        else pl.lit(0.0).alias(utility_col)
    )
    state_col = "path_state" if "path_state" in outcomes.columns else "tail_state"
    outcome_aggs.append(
        pl.col(state_col).drop_nulls().first().alias("tail_state")
        if state_col in outcomes.columns
        else pl.lit(None).alias("tail_state")
    )
    outcome_aggs.append(
        pl.col("tail_both").fill_null(False).cast(pl.Boolean).max().alias("tail_both")
        if "tail_both" in outcomes.columns
        else pl.lit(False).alias("tail_both")
    )
    outcome_aggs.append(
        pl.col(margin_col).fill_null(0.0).cast(pl.Float64).max().alias(margin_col)
        if margin_col in outcomes.columns
        else pl.lit(0.0).alias(margin_col)
    )
    outcome_by_decision = outcomes.group_by("symbol", "decision_bar_close_ms").agg(*outcome_aggs)
    behavior_by_decision = pl.DataFrame()
    if behavior_targets is not None and not behavior_targets.is_empty():
        behavior_by_decision = behavior_targets.group_by(
            "symbol", "decision_bar_close_ms"
        ).agg(
            pl.col("behavior_actionable")
            .fill_null(False)
            .cast(pl.Boolean)
            .max()
            .alias("behavior_actionable"),
            pl.col("behavior_false_direction")
            .fill_null(False)
            .cast(pl.Boolean)
            .max()
            .alias("behavior_false_direction"),
            pl.col("behavior_utility_margin")
            .fill_null(0.0)
            .cast(pl.Float64)
            .max()
            .alias("behavior_utility_margin"),
            pl.col("behavior_path_state").drop_nulls().first().alias("behavior_path_state"),
            pl.col("behavior_actionability")
            .drop_nulls()
            .first()
            .alias("behavior_actionability"),
            pl.col("behavior_blocker").drop_nulls().first().alias("behavior_blocker"),
        )
    scored = tree.predict_score(observations).join(
        outcome_by_decision,
        on=["symbol", "decision_bar_close_ms"],
        how="left",
    )
    if not behavior_by_decision.is_empty():
        scored = scored.join(
            behavior_by_decision,
            on=["symbol", "decision_bar_close_ms"],
            how="left",
        )
    else:
        scored = scored.with_columns(
            pl.lit(False).alias("behavior_actionable"),
            pl.lit(False).alias("behavior_false_direction"),
            pl.lit(0.0).alias("behavior_utility_margin"),
            pl.lit("none").alias("behavior_path_state"),
            pl.lit("no_action").alias("behavior_actionability"),
            pl.lit("").alias("behavior_blocker"),
        )
    joined = (
        scored
        .with_columns(
            pl.col(tail_col).fill_null(False).cast(pl.Boolean).alias("selected_tail"),
            pl.col(utility_col).fill_null(0.0).cast(pl.Float64).alias("selected_utility"),
            (
                pl.col("tail_state").is_in([direction, f"clean_{direction}"])
            ).alias("selected_side_only"),
            pl.col("tail_both").fill_null(False).cast(pl.Boolean).alias("selected_tail_both"),
            pl.col("tail_state").alias("selected_tail_state"),
            pl.col(margin_col).fill_null(0.0).cast(pl.Float64).alias("selected_utility_margin"),
            pl.col("behavior_actionable")
            .fill_null(False)
            .cast(pl.Boolean)
            .alias("selected_behavior_actionable"),
            pl.col("behavior_false_direction")
            .fill_null(False)
            .cast(pl.Boolean)
            .alias("selected_behavior_false_direction"),
            pl.col("behavior_utility_margin")
            .fill_null(0.0)
            .cast(pl.Float64)
            .alias("selected_behavior_utility_margin"),
            pl.col("behavior_path_state")
            .fill_null("none")
            .alias("selected_behavior_path_state"),
            pl.col("behavior_actionability")
            .fill_null("no_action")
            .alias("selected_behavior_actionability"),
            pl.col("behavior_blocker")
            .fill_null("")
            .alias("selected_behavior_blocker"),
            pl.lit(int(outcome_horizon)).alias("outcome_horizon"),
            pl.lit(direction).alias("direction"),
        )
    )
    if joined.is_empty():
        return joined.head(0)

    sorted_scores = joined.sort("tailtree_score", descending=True)
    frames: list[pl.DataFrame] = []
    for quantile in score_quantiles:
        bucket_value = round((1.0 - quantile) * 100)
        bucket = f"top_{int(bucket_value)}pct"
        frames.append(
            sorted_scores.head(max(1, ceil(len(sorted_scores) * (1.0 - quantile)))).select(
                "symbol",
                "decision_bar_close_ms",
                "outcome_horizon",
                pl.lit(objective).alias("objective"),
                "direction",
                "tailtree_score",
                "selected_tail",
                "selected_utility",
                "selected_side_only",
                "selected_tail_both",
                "selected_tail_state",
                "selected_utility_margin",
                "selected_behavior_actionable",
                "selected_behavior_false_direction",
                "selected_behavior_utility_margin",
                "selected_behavior_path_state",
                "selected_behavior_actionability",
                "selected_behavior_blocker",
                pl.lit(bucket).alias("score_bucket"),
                pl.lit(float(bucket_value)).alias("budget_value"),
            )
        )
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def score_bucket_population_frame(
    tree: TailtreeArtifactTree,
    observations: pl.DataFrame,
    outcomes: pl.DataFrame,
    outcome_horizon: int,
    *,
    behavior_targets: pl.DataFrame | None = None,
    objective: str = "tail_event_lift",
    score_quantiles: tuple[float, ...] = (0.99, 0.98, 0.95, 0.90),
) -> pl.DataFrame:
    scored = score_bucket_candidate_frame(
        tree,
        observations,
        outcomes,
        outcome_horizon,
        behavior_targets=behavior_targets,
        objective=objective,
        score_quantiles=(0.0,),
    )
    if scored.is_empty():
        return scored
    scored = scored.drop("score_bucket", "budget_value").sort("tailtree_score", descending=True)
    total = len(scored)
    ranked = scored.with_row_index("score_rank")
    frames: list[pl.DataFrame] = []
    for quantile in score_quantiles:
        bucket_value = round((1.0 - quantile) * 100)
        cutoff = max(1, ceil(total * (1.0 - quantile)))
        frames.append(
            ranked.with_columns(
                (pl.col("score_rank") < cutoff).alias("in_score_bucket"),
                pl.lit(f"top_{int(bucket_value)}pct").alias("score_bucket"),
                pl.lit(float(bucket_value)).alias("budget_value"),
            ).drop("score_rank")
        )
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def paired_candidate_replay_frame(scored_candidates: pl.DataFrame) -> pl.DataFrame:
    if scored_candidates.is_empty():
        return scored_candidates

    keys = ["symbol", "decision_bar_close_ms", "outcome_horizon", "objective", "score_bucket"]
    up = scored_candidates.filter(pl.col("direction") == "up").rename(
        {
            "tailtree_score": "score_up",
            "selected_tail": "tail_up",
            "selected_utility": "utility_up",
            "selected_side_only": "side_only_up",
            "selected_tail_both": "tail_both_up",
            "selected_tail_state": "tail_state_up",
            "selected_utility_margin": "utility_margin_up",
            "selected_behavior_actionable": "behavior_actionable_up",
            "selected_behavior_false_direction": "behavior_false_direction_up",
            "selected_behavior_utility_margin": "behavior_utility_margin_up",
            "selected_behavior_path_state": "behavior_path_state_up",
            "selected_behavior_actionability": "behavior_actionability_up",
            "selected_behavior_blocker": "behavior_blocker_up",
        }
    )
    down = scored_candidates.filter(pl.col("direction") == "down").rename(
        {
            "tailtree_score": "score_down",
            "selected_tail": "tail_down",
            "selected_utility": "utility_down",
            "selected_side_only": "side_only_down",
            "selected_tail_both": "tail_both_down",
            "selected_tail_state": "tail_state_down",
            "selected_utility_margin": "utility_margin_down",
            "selected_behavior_actionable": "behavior_actionable_down",
            "selected_behavior_false_direction": "behavior_false_direction_down",
            "selected_behavior_utility_margin": "behavior_utility_margin_down",
            "selected_behavior_path_state": "behavior_path_state_down",
            "selected_behavior_actionability": "behavior_actionability_down",
            "selected_behavior_blocker": "behavior_blocker_down",
        }
    )
    selected_up = up.join(
        down.select(
            *keys,
            "score_down",
            "tail_down",
            "utility_down",
            "side_only_down",
            "tail_both_down",
            "tail_state_down",
            "utility_margin_down",
            "behavior_actionable_down",
            "behavior_false_direction_down",
            "behavior_utility_margin_down",
            "behavior_path_state_down",
            "behavior_actionability_down",
            "behavior_blocker_down",
        ),
        on=keys,
        how="left",
    ).with_columns(pl.lit("up").alias("selected_direction"))
    selected_down = down.join(
        up.select(
            *keys,
            "score_up",
            "tail_up",
            "utility_up",
            "side_only_up",
            "tail_both_up",
            "tail_state_up",
            "utility_margin_up",
            "behavior_actionable_up",
            "behavior_false_direction_up",
            "behavior_utility_margin_up",
            "behavior_path_state_up",
            "behavior_actionability_up",
            "behavior_blocker_up",
        ),
        on=keys,
        how="left",
    ).with_columns(pl.lit("down").alias("selected_direction"))

    replay = pl.concat(
        [
            selected_up.select(
                *keys,
                "selected_direction",
                pl.col("score_up").alias("selected_score"),
                pl.col("score_down").alias("opposite_score"),
                pl.col("tail_up").alias("selected_tail"),
                pl.col("tail_down").fill_null(False).alias("opposite_tail"),
                pl.col("utility_up").alias("selected_utility"),
                pl.col("utility_down").fill_null(0.0).alias("opposite_utility"),
                pl.col("side_only_up").alias("selected_side_only"),
                pl.col("side_only_down").fill_null(False).alias("opposite_side_only"),
                pl.col("tail_both_up").alias("selected_tail_both"),
                pl.col("tail_both_down").fill_null(False).alias("opposite_tail_both"),
                pl.col("tail_state_up").alias("selected_tail_state"),
                pl.col("tail_state_down").fill_null("none").alias("opposite_tail_state"),
                pl.col("utility_margin_up").alias("selected_utility_margin"),
                pl.col("utility_margin_down").fill_null(0.0).alias("opposite_utility_margin"),
                pl.col("behavior_actionable_up").alias("selected_behavior_actionable"),
                pl.col("behavior_false_direction_up").alias(
                    "selected_behavior_false_direction"
                ),
                pl.col("behavior_utility_margin_up").alias(
                    "selected_behavior_utility_margin"
                ),
                pl.col("behavior_path_state_up").alias("selected_behavior_path_state"),
                pl.col("behavior_actionability_up").alias(
                    "selected_behavior_actionability"
                ),
                pl.col("behavior_blocker_up").alias("selected_behavior_blocker"),
            ),
            selected_down.select(
                *keys,
                "selected_direction",
                pl.col("score_down").alias("selected_score"),
                pl.col("score_up").alias("opposite_score"),
                pl.col("tail_down").alias("selected_tail"),
                pl.col("tail_up").fill_null(False).alias("opposite_tail"),
                pl.col("utility_down").alias("selected_utility"),
                pl.col("utility_up").fill_null(0.0).alias("opposite_utility"),
                pl.col("side_only_down").alias("selected_side_only"),
                pl.col("side_only_up").fill_null(False).alias("opposite_side_only"),
                pl.col("tail_both_down").alias("selected_tail_both"),
                pl.col("tail_both_up").fill_null(False).alias("opposite_tail_both"),
                pl.col("tail_state_down").alias("selected_tail_state"),
                pl.col("tail_state_up").fill_null("none").alias("opposite_tail_state"),
                pl.col("utility_margin_down").alias("selected_utility_margin"),
                pl.col("utility_margin_up").fill_null(0.0).alias("opposite_utility_margin"),
                pl.col("behavior_actionable_down").alias("selected_behavior_actionable"),
                pl.col("behavior_false_direction_down").alias(
                    "selected_behavior_false_direction"
                ),
                pl.col("behavior_utility_margin_down").alias(
                    "selected_behavior_utility_margin"
                ),
                pl.col("behavior_path_state_down").alias("selected_behavior_path_state"),
                pl.col("behavior_actionability_down").alias(
                    "selected_behavior_actionability"
                ),
                pl.col("behavior_blocker_down").alias("selected_behavior_blocker"),
            ),
        ],
        how="diagonal_relaxed",
    )
    return replay.with_columns(
        (pl.col("selected_score") - pl.col("opposite_score").fill_null(0.0)).alias(
            "directional_score_margin"
        ),
        (pl.col("selected_tail") & pl.col("opposite_tail")).cast(pl.Int64).alias("gray_zone_int"),
        ((~pl.col("selected_tail")) & pl.col("opposite_tail"))
        .cast(pl.Int64)
        .alias("false_direction_int"),
        (
            pl.col("selected_side_only")
            & ~pl.col("opposite_side_only")
            & ~pl.col("selected_tail_both")
        )
        .cast(pl.Int64)
        .alias("side_only_int"),
        (
            pl.col("selected_tail_both")
            | pl.col("opposite_tail_both")
            | (pl.col("selected_tail") & pl.col("opposite_tail"))
        )
        .cast(pl.Int64)
        .alias("tail_both_int"),
    )


def calibrated_candidate_replay_frame(replay: pl.DataFrame) -> pl.DataFrame:
    if replay.is_empty():
        return replay
    keys = ["outcome_horizon", "score_bucket", "selected_direction"]
    calibration = replay.group_by(keys).agg(
        pl.col("selected_tail").fill_null(False).mean().alias("selected_bucket_tail_rate"),
        pl.col("opposite_tail").fill_null(False).mean().alias("opposite_bucket_tail_rate"),
        pl.col("selected_side_only")
        .fill_null(False)
        .mean()
        .alias("selected_bucket_side_only_rate"),
        pl.col("opposite_side_only")
        .fill_null(False)
        .mean()
        .alias("opposite_bucket_side_only_rate"),
        pl.col("selected_tail_both")
        .fill_null(False)
        .mean()
        .alias("selected_bucket_tail_both_rate"),
    )
    return replay.join(calibration, on=keys, how="left").with_columns(
        (pl.col("selected_bucket_tail_rate") - pl.col("opposite_bucket_tail_rate")).alias(
            "calibrated_directional_margin"
        ),
        (
            pl.col("selected_bucket_side_only_rate")
            - pl.col("opposite_bucket_side_only_rate")
        ).alias("calibrated_side_margin"),
    )


def candidate_population_confusion_metrics(
    population: pl.DataFrame,
    *,
    outcome_horizon: int,
    direction: TailtreeDirection | str,
    objective: str,
    score_bucket: str,
) -> dict[str, float]:
    if population.is_empty():
        return {}
    selected = population.filter(
        (pl.col("objective") == str(objective))
        & (pl.col("direction") == str(direction))
        & (pl.col("outcome_horizon") == int(outcome_horizon))
        & (pl.col("score_bucket") == str(score_bucket))
    )
    return _confusion_from_population(selected, selected_flag="in_score_bucket")


def _confusion_from_population(
    population: pl.DataFrame,
    *,
    selected_flag: str,
) -> dict[str, float]:
    if population.is_empty() or selected_flag not in population.columns:
        return {}
    actual = pl.col("selected_behavior_actionable").fill_null(False).cast(pl.Boolean)
    predicted = pl.col(selected_flag).fill_null(False).cast(pl.Boolean)
    metrics = population.select(
        (predicted & actual).sum().alias("behavior_tp_count"),
        (predicted & ~actual).sum().alias("behavior_fp_count"),
        (~predicted & actual).sum().alias("behavior_fn_count"),
        (~predicted & ~actual).sum().alias("behavior_tn_count"),
    ).row(0, named=True)
    tp = float(metrics["behavior_tp_count"] or 0.0)
    fp = float(metrics["behavior_fp_count"] or 0.0)
    fn = float(metrics["behavior_fn_count"] or 0.0)
    tn = float(metrics["behavior_tn_count"] or 0.0)
    total = max(tp + fp + fn + tn, 1.0)
    return {
        "behavior_tp_count": tp,
        "behavior_fp_count": fp,
        "behavior_fn_count": fn,
        "behavior_tn_count": tn,
        "behavior_precision": tp / max(tp + fp, 1.0),
        "behavior_recall": tp / max(tp + fn, 1.0),
        "behavior_specificity": tn / max(tn + fp, 1.0),
        "behavior_false_positive_rate": fp / max(fp + tn, 1.0),
        "behavior_false_negative_rate": fn / max(fn + tp, 1.0),
        "behavior_accuracy": (tp + tn) / total,
    }


def tailtree_action_surface_frame(replay: pl.DataFrame) -> pl.DataFrame:
    if replay.is_empty():
        return replay
    work = replay.with_columns(
        pl.col("selected_direction").alias("action_side"),
        pl.col("outcome_horizon").cast(pl.Int64).alias("entry_horizon"),
        pl.col("outcome_horizon").cast(pl.Int64).alias("max_valid_horizon"),
        pl.col("selected_tail_state").fill_null("none").alias("best_path_state"),
        pl.col("selected_tail_state").fill_null("none").alias("path_state_profile"),
        pl.col("selected_utility_margin").fill_null(0.0).alias("best_utility_margin"),
    )
    clean_side = (
        pl.col("selected_side_only").fill_null(False)
        & ~pl.col("selected_tail_both").fill_null(False)
        & (pl.col("calibrated_side_margin").fill_null(0.0) > 0.0)
        & (pl.col("selected_utility_margin").fill_null(0.0) > 0.0)
    )
    mixed_path = (
        pl.col("selected_tail_both").fill_null(False)
        | pl.col("opposite_tail_both").fill_null(False)
        | pl.col("best_path_state").is_in(["up_first_both", "down_first_both", "chop_both"])
    )
    opposite_dominates = pl.col("false_direction_int").fill_null(0).cast(pl.Int64) == 1
    actionability = (
        pl.when(clean_side)
        .then(pl.lit("trade_candidate"))
        .when(opposite_dominates)
        .then(pl.lit("reversal_watch"))
        .when(mixed_path)
        .then(pl.lit("gray_zone"))
        .otherwise(pl.lit("no_action"))
    )
    blocker = (
        pl.when(clean_side)
        .then(pl.lit(""))
        .when(opposite_dominates)
        .then(pl.lit("opposite_tail_dominates"))
        .when(mixed_path)
        .then(pl.lit("both_or_mixed_path"))
        .otherwise(pl.lit("no_clean_side"))
    )
    return work.with_columns(
        actionability.alias("actionability"),
        blocker.alias("blocker_reason"),
        pl.when(clean_side).then(pl.lit(1)).otherwise(pl.lit(0)).alias("clean_horizon_count"),
        pl.when(mixed_path).then(pl.lit(1)).otherwise(pl.lit(0)).alias("chop_horizon_count"),
        pl.when(opposite_dominates)
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("contradicting_horizon_count"),
        pl.when(actionability == "reversal_watch")
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("reversal_horizon_count"),
    ).select(
        "symbol",
        "decision_bar_close_ms",
        "action_side",
        "entry_horizon",
        "max_valid_horizon",
        "actionability",
        "path_state_profile",
        "best_path_state",
        "best_utility_margin",
        "clean_horizon_count",
        "chop_horizon_count",
        "reversal_horizon_count",
        "contradicting_horizon_count",
        "calibrated_side_margin",
        "false_direction_int",
        "blocker_reason",
        "score_bucket",
    )


def candidate_replay_metrics(
    replay: pl.DataFrame,
    *,
    outcome_horizon: int,
    direction: TailtreeDirection | str,
    score_bucket: str,
) -> TailtreeReplayMetrics:
    empty = TailtreeReplayMetrics()
    if replay.is_empty():
        return empty

    selected = replay.filter(
        (pl.col("selected_direction") == str(direction))
        & (pl.col("outcome_horizon") == int(outcome_horizon))
        & (pl.col("score_bucket") == str(score_bucket))
    )
    if selected.is_empty():
        return empty

    false_cost = selected.filter(pl.col("false_direction_int") == 1).get_column("opposite_utility")

    def mean(values: pl.Series) -> float:
        if values.is_empty():
            return 0.0
        value = values.to_frame().select(pl.col(values.name).cast(pl.Float64).mean()).item()
        return float(value) if value is not None else 0.0

    behavior_actionable_rate = (
        mean(selected.get_column("selected_behavior_actionable"))
        if "selected_behavior_actionable" in selected.columns
        else 0.0
    )
    metrics = {
        "candidate_pair_count": float(selected.height),
        "paired_selected_tail_rate": mean(selected.get_column("selected_tail")),
        "paired_opposite_rate": float(
            selected.select(pl.col("opposite_score").is_not_null().mean()).item() or 0.0
        ),
        "paired_gray_zone_rate": mean(selected.get_column("gray_zone_int")),
        "paired_false_direction_rate": mean(selected.get_column("false_direction_int")),
        "paired_false_direction_cost_mean": mean(false_cost),
        "paired_directional_margin_mean": mean(selected.get_column("directional_score_margin")),
        "paired_side_only_rate": mean(selected.get_column("side_only_int")),
        "paired_tail_both_rate": mean(selected.get_column("tail_both_int")),
        "paired_selected_utility_margin_mean": mean(
            selected.get_column("selected_utility_margin")
        ),
        "paired_behavior_actionable_rate": behavior_actionable_rate,
        "paired_behavior_false_positive_proxy_rate": 1.0 - behavior_actionable_rate,
        "paired_behavior_false_direction_rate": mean(
            selected.get_column("selected_behavior_false_direction")
        )
        if "selected_behavior_false_direction" in selected.columns
        else 0.0,
        "paired_behavior_utility_margin_mean": mean(
            selected.get_column("selected_behavior_utility_margin")
        )
        if "selected_behavior_utility_margin" in selected.columns
        else 0.0,
    }
    if "calibrated_directional_margin" in selected.columns:
        metrics["paired_calibrated_directional_margin_mean"] = mean(
            selected.get_column("calibrated_directional_margin")
        )
    if "calibrated_side_margin" in selected.columns:
        metrics["paired_calibrated_side_margin_mean"] = mean(
            selected.get_column("calibrated_side_margin")
        )
    return TailtreeReplayMetrics(**metrics)


def tailtree_selection_metrics_frame(
    run: TailtreeProfileRun,
    evidence: pl.DataFrame,
    prepared: TailtreePreparedFrames,
    models: dict[tuple[int, TailtreeDirection], TailtreeArtifactTree],
    seconds: float,
    candidate_replay: pl.DataFrame,
    candidate_population: pl.DataFrame | None = None,
) -> pl.DataFrame:
    if evidence.is_empty():
        return pl.DataFrame()
    eligible_symbol_count = prepared.observations.get_column("symbol").n_unique()
    observation_row_count = prepared.observations.height
    feature_count = len(prepared.categorical_features) + len(prepared.continuous_features)
    rows: list[dict[str, object]] = []
    for row in evidence.to_dicts():
        selected_observations = int(row.get("N_total") or 0)
        selected_tails = int(row.get("N_tail_exceedances") or 0)
        valid_tail_lift = float(row.get("tail_lift") or 0.0)
        selected_tail_rate = (
            selected_tails / selected_observations if selected_observations else 0.0
        )
        valid_tail_rate = selected_tail_rate / valid_tail_lift if valid_tail_lift else 0.0
        selected_symbols = int(row.get("symbol_count") or eligible_symbol_count)
        utility_mean = float(row.get("tail_utility_mean") or 0.0)
        utility_p90 = float(row.get("tail_utility_p90") or 0.0)
        profit_per_1k = (utility_mean * 1000.0) / max(float(selected_observations), 1.0)
        base_hpo_score = valid_tail_lift + utility_mean + (selected_tails + 1.0) ** 0.5 / 10.0
        replay_metrics = candidate_replay_metrics(
            candidate_replay,
            outcome_horizon=int(row.get("outcome_horizon") or 0),
            direction=str(row.get("tree_direction") or ""),
            score_bucket=str(row.get("score_bucket") or ""),
        )
        confusion_metrics = candidate_population_confusion_metrics(
            candidate_population if candidate_population is not None else pl.DataFrame(),
            outcome_horizon=int(row.get("outcome_horizon") or 0),
            direction=str(row.get("tree_direction") or ""),
            objective=run.objective,
            score_bucket=str(row.get("score_bucket") or ""),
        )
        row_values = {
            **TailtreeSelectionEfficiencyRow(
            universe_snapshot_id="active",
            model_tag=run.model_tag,
            objective=run.objective,
            training_profile=run.profile_id,
            trial_id=run.run_id.rsplit("-t", 1)[0] if run.run_source == "optuna" else run.run_id,
            trial_source=run.run_source,
            outcome_horizon=int(row.get("outcome_horizon") or 0),
            tree_direction=str(row.get("tree_direction") or ""),
            budget_family="score_bucket" if row.get("score_bucket") is not None else "leaf",
            budget_value=float(
                str(row.get("score_bucket") or row.get("leaf_id") or 0)
                .replace("top_", "")
                .replace("pct", "")
                or 0.0
            ),
            eligible_symbol_count=int(eligible_symbol_count),
            selected_symbol_count=selected_symbols,
            observation_row_count=int(observation_row_count),
            feature_count=int(feature_count),
            train_exceedance_count=selected_tails,
            valid_observation_count=int(observation_row_count),
            valid_tail_count=selected_tails,
            valid_tail_rate=float(valid_tail_rate),
            selected_observation_count=selected_observations,
            selected_observation_rate=selected_observations / observation_row_count
            if observation_row_count
            else 0.0,
            selected_tail_count=selected_tails,
            selected_tail_rate=float(selected_tail_rate),
            selected_tail_per_1k_obs=(selected_tails * 1000.0) / selected_observations
            if selected_observations
            else 0.0,
            valid_tail_lift=valid_tail_lift,
            selected_profit_proxy_mean=utility_mean,
            selected_profit_proxy_p90=utility_p90,
            selected_utility_mean=utility_mean,
            selected_utility_p90=utility_p90,
            profit_proxy_per_selected_obs=utility_mean,
            profit_proxy_per_1k_observed=profit_per_1k,
            hpo_score=base_hpo_score,
            promotion_threshold_pass_int=int(
                selected_observations >= 500 and valid_tail_lift >= 3.0
            ),
            trained_tree_count=len(models),
            selected_bucket_or_leaf_count=1,
            fit_seconds=seconds,
            score_seconds=0.0,
            ).model_dump(),
            "base_hpo_score": base_hpo_score,
            "objective_hpo_score": replay_metrics.directional_objective_score(base_hpo_score),
            "side_hpo_score": replay_metrics.side_objective_score(base_hpo_score),
            "behavior_hpo_score": replay_metrics.behavior_objective_score(base_hpo_score),
            **replay_metrics.model_dump(),
            **confusion_metrics,
        }
        rows.append(row_values)
    return pl.DataFrame(rows)


def decision_key_action_surface_frame(action_surface: pl.DataFrame) -> pl.DataFrame:
    """Collapse action-surface rows to one row per symbol/decision timestamp."""
    required = {
        "symbol",
        "decision_bar_close_ms",
        "actionability",
        "false_direction_int",
        "best_utility_margin",
    }
    if action_surface.is_empty() or not required.issubset(action_surface.columns):
        return pl.DataFrame()
    return (
        action_surface.with_columns(
            (pl.col("actionability") == "trade_candidate").alias("_trade_candidate"),
            pl.col("false_direction_int")
            .fill_null(0)
            .cast(pl.Int64, strict=False)
            .alias("_false_direction"),
            pl.col("best_utility_margin")
            .fill_null(0.0)
            .cast(pl.Float64, strict=False)
            .alias("_utility"),
        )
        .group_by("symbol", "decision_bar_close_ms")
        .agg(
            pl.col("_trade_candidate").max().alias("any_candidate"),
            pl.col("_false_direction").max().alias("any_false_direction"),
            pl.col("_utility").max().alias("best_utility"),
            pl.len().alias("surface_rows"),
        )
        .sort("symbol", "decision_bar_close_ms")
    )


def feature_pack_stability_frame(
    source_features: pl.DataFrame,
    action_surface: pl.DataFrame,
    *,
    feature_pack: str = "source_timeseries_context",
    feature_columns: tuple[str, ...] = (
        "funding_level_state",
        "funding_level_transition",
        "funding_price_divergence_24h",
        "lsr_level_state",
        "lsr_level_transition",
        "lsr_price_divergence_24h",
        "oi_flow_state",
        "oi_flow_transition",
        "taker_pressure_state",
        "taker_pressure_transition",
    ),
    min_support: int = 50,
) -> pl.DataFrame:
    """Profile source feature-pack states with explicit improvement actions."""
    required_source = {"symbol", "timestamp"}
    required_surface = {"symbol", "decision_bar_close_ms"}
    if (
        source_features.is_empty()
        or action_surface.is_empty()
        or not required_source.issubset(source_features.columns)
        or not required_surface.issubset(action_surface.columns)
    ):
        return pl.DataFrame()
    available_features = tuple(
        column for column in feature_columns if column in source_features.columns
    )
    if not available_features:
        return pl.DataFrame()
    decision_surface = (
        action_surface
        if {"any_candidate", "any_false_direction", "best_utility", "surface_rows"}.issubset(
            action_surface.columns
        )
        else decision_key_action_surface_frame(action_surface)
    )
    if decision_surface.is_empty():
        return pl.DataFrame()
    joined = source_features.join(
        decision_surface.select(
            "symbol",
            pl.col("decision_bar_close_ms").alias("timestamp"),
            "any_candidate",
            "any_false_direction",
            "best_utility",
            "surface_rows",
        ),
        on=["symbol", "timestamp"],
        how="inner",
    ).with_columns(
        pl.col("any_candidate").fill_null(False).alias("_trade_candidate"),
        pl.col("any_false_direction")
        .fill_null(0)
        .cast(pl.Float64, strict=False)
        .alias("_false_direction"),
        pl.col("best_utility").fill_null(0.0).cast(pl.Float64, strict=False).alias("_utility"),
    )
    if joined.is_empty():
        return pl.DataFrame()
    total_rows = joined.height
    frames: list[pl.DataFrame] = []
    for feature_name in available_features:
        frame = (
            joined.filter(pl.col(feature_name).is_not_null())
            .with_columns(pl.col(feature_name).cast(pl.Utf8).alias("feature_value"))
            .group_by("feature_value")
            .agg(
                pl.len().alias("support"),
                pl.col("_trade_candidate").sum().alias("candidate_count"),
                pl.col("_false_direction").sum().alias("false_direction_count"),
                pl.col("_utility").mean().alias("utility_mean"),
                pl.col("_utility").median().alias("utility_median"),
                pl.col("surface_rows").sum().alias("surface_rows"),
            )
            .with_columns(
                pl.lit(feature_pack).alias("feature_pack"),
                pl.lit(feature_name).alias("feature_name"),
                (pl.col("support") / pl.lit(float(total_rows))).alias("coverage_rate"),
                (pl.col("candidate_count") / pl.col("support")).alias("candidate_rate"),
                (pl.col("candidate_count") / pl.col("support")).alias("precision"),
                (pl.col("false_direction_count") / pl.col("support")).alias(
                    "false_direction_rate"
                ),
            )
            .with_columns(
                pl.when(pl.col("support") < int(min_support))
                .then(pl.lit("reject_low_support"))
                .when(pl.col("coverage_rate") < 0.25)
                .then(pl.lit("improve_source_coverage"))
                .when(
                    (pl.col("precision") >= 0.50)
                    & (pl.col("false_direction_rate") <= 0.20)
                    & (pl.col("utility_mean") > 0.0)
                )
                .then(pl.lit("promote_candidate_modifier"))
                .when(
                    (pl.col("precision") >= 0.50)
                    & (pl.col("false_direction_rate") > 0.20)
                    & (pl.col("utility_mean") > 0.0)
                )
                .then(pl.lit("investigate_high_risk_opportunity"))
                .when(pl.col("utility_mean") <= 0.0)
                .then(pl.lit("reject_negative_utility"))
                .otherwise(pl.lit("keep_diagnostic_only"))
                .alias("improvement_action")
            )
            .with_columns(
                pl.when(pl.col("improvement_action") == "reject_low_support")
                .then(pl.lit("state support is below the minimum decision threshold"))
                .when(pl.col("improvement_action") == "improve_source_coverage")
                .then(pl.lit("state coverage is too sparse for model promotion"))
                .when(pl.col("improvement_action") == "promote_candidate_modifier")
                .then(pl.lit("state meets precision, false-direction, and utility thresholds"))
                .when(pl.col("improvement_action") == "investigate_high_risk_opportunity")
                .then(pl.lit("state has opportunity but false-direction is too high"))
                .when(pl.col("improvement_action") == "reject_negative_utility")
                .then(pl.lit("state has non-positive mean utility"))
                .otherwise(pl.lit("state is useful for diagnostics but not promotion"))
                .alias("improvement_reason")
            )
            .select(
                "feature_pack",
                "feature_name",
                "feature_value",
                "support",
                "coverage_rate",
                "candidate_count",
                "candidate_rate",
                "precision",
                "false_direction_count",
                "false_direction_rate",
                "utility_mean",
                "utility_median",
                "surface_rows",
                "improvement_action",
                "improvement_reason",
            )
        )
        frames.append(frame)
    return pl.concat(frames, how="diagonal_relaxed").sort(
        ["feature_name", "support"], descending=[False, True]
    )



def frontier_benchmark_frame(
    selection_efficiency: pl.DataFrame, *, min_selected: int = 50
) -> pl.DataFrame:
    """Compact active candidate-dual-guard frontier rows."""
    required = {
        "objective",
        "feature_set",
        "candidate_gate_id",
        "selected_observation_count",
        "behavior_precision",
        "paired_behavior_false_direction_rate",
        "paired_behavior_utility_margin_mean",
    }
    if selection_efficiency.is_empty() or not required.issubset(selection_efficiency.columns):
        return pl.DataFrame()
    floor = selection_efficiency.filter(
        (pl.col("objective") == "candidate_dual_guard")
        & (pl.col("selected_observation_count") >= int(min_selected))
    )
    if floor.is_empty():
        return pl.DataFrame()
    dedup_columns = [
        "feature_set",
        "objective",
        "candidate_gate_id",
        "selected_observation_count",
        "behavior_precision",
        "paired_behavior_false_direction_rate",
        "paired_behavior_utility_margin_mean",
    ]
    return (
        floor.sort(
            [
                "behavior_precision",
                "paired_behavior_false_direction_rate",
                "paired_behavior_utility_margin_mean",
            ],
            descending=[True, False, True],
        )
        .unique(subset=dedup_columns, maintain_order=True)
        .with_columns(
            pl.lit("active_candidate_dual_guard").alias("control_objective"),
            pl.lit("active").alias("control_feature_set"),
            pl.col("selected_observation_count").alias("control_selected_count"),
            pl.col("behavior_precision").alias("control_precision"),
            pl.col("paired_behavior_false_direction_rate").alias("control_false_direction_rate"),
            pl.col("paired_behavior_utility_margin_mean").alias("control_utility_mean"),
            pl.lit(1).cast(pl.Int8).alias("beats_control_int"),
            pl.lit("promote_candidate_frontier").alias("improvement_action"),
        )
    )



def _row_float(row: dict[str, object], key: str) -> float:
    value = row.get(key)
    return float(value) if isinstance(value, int | float | str) else 0.0


def _row_int(row: dict[str, object], key: str) -> int:
    value = row.get(key)
    return int(value) if isinstance(value, int | float | str) else 0


__all__ = [
    "candidate_replay_metrics",
    "candidate_gate_frame",
    "candidate_population_confusion_metrics",
    "actionability_contradiction_audit_frame",
    "calibrated_candidate_replay_frame",
    "decision_key_action_surface_frame",
    "dual_guard_boundary_anatomy_frame",
    "dual_guarded_promotion_selection_metrics_frame",
    "guarded_selection_error_anatomy_frame",
    "frontier_benchmark_frame",
    "feature_pack_stability_frame",
    "paired_candidate_replay_frame",
    "opposite_guard_target_frame",
    "promoter_target_frame",
    "selection_error_anatomy_frame",
    "weak_path_guard_target_frame",
    "score_bucket_candidate_frame",
    "score_bucket_population_frame",
    "tailtree_action_surface_frame",
    "TailtreeReplayMetrics",
    "tailtree_selection_metrics_frame",
]
