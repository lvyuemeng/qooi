"""Score recent accepted path features and write a user-readable prediction report."""

from __future__ import annotations

import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from qooi.scanner.path_model import TailTreeModel
from qooi.scanner.tailrun.features import AcceptedFeatureManifest

OUTPUT_DIR = Path("data/output/potential/path")
FEATURE_DIR = OUTPUT_DIR
MODEL_DIR = FEATURE_DIR / "models"
FEATURE_MATRIX_PATH = FEATURE_DIR / "predict_features.parquet"
MANIFEST_PATH = FEATURE_DIR / "feature-manifest.accepted.json"
MODEL_PATH = MODEL_DIR / "tailtree-path_path.json"
BOARD_PATH = OUTPUT_DIR / "path_probability_board.parquet"
REPORT_PATH = OUTPUT_DIR / "prediction-report.md"
MODEL_ID = "tailtree-path_path"
RECENT_DECISION_MAX_AGE_HOURS = 2.0
LEGACY_OUTPUT_DIRS = (
    Path("data/output/potential/path-train"),
    Path("data/output/potential/path-predict"),
    Path("data/output/potential/path/tailtree"),
)


def now_ms() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)


def recent_latest_feature_rows(
    matrix: pl.DataFrame, *, max_age_hours: float = RECENT_DECISION_MAX_AGE_HOURS
) -> pl.DataFrame:
    """Return latest symbol × horizon rows inside the matrix's most recent window."""
    if matrix.is_empty():
        return matrix
    max_decision_ms = matrix.get_column("decision_bar_close_ms").max()
    cutoff_ms = int(max_decision_ms - max_age_hours * 60 * 60 * 1000)
    recent = matrix.filter(pl.col("decision_bar_close_ms") >= cutoff_ms)
    return recent.join(
        recent.group_by("symbol", "horizon_hours").agg(
            pl.col("decision_bar_close_ms").max().alias("decision_bar_close_ms")
        ),
        on=["symbol", "horizon_hours", "decision_bar_close_ms"],
        how="inner",
    )


def latest_decision_age_hours(matrix: pl.DataFrame, *, scored_at_ms: int | None = None) -> float:
    if matrix.is_empty():
        return float("inf")
    scored_at_ms = now_ms() if scored_at_ms is None else int(scored_at_ms)
    latest_ms = matrix.get_column("decision_bar_close_ms").max()
    return float((scored_at_ms - int(latest_ms)) / 3_600_000.0)


def require_recent_input(matrix: pl.DataFrame) -> None:
    age_hours = latest_decision_age_hours(matrix)
    if age_hours > RECENT_DECISION_MAX_AGE_HOURS:
        raise RuntimeError(
            "predict_features.parquet is stale: "
            f"latest decision age is {age_hours:.2f}h, "
            f"validity window is {RECENT_DECISION_MAX_AGE_HOURS:.1f}h. "
            "Run `uv run python scripts/01_build_features.py` before `03_predict.py`."
        )


def prediction_board(
    scored: pl.DataFrame,
    *,
    scored_at_ms: int | None = None,
    valid_for_hours: float = RECENT_DECISION_MAX_AGE_HOURS,
) -> pl.DataFrame:
    """Rank scored path probabilities by calibrated participation value and freshness."""
    scored_at_ms = now_ms() if scored_at_ms is None else int(scored_at_ms)
    valid_for_ms = int(valid_for_hours * 60 * 60 * 1000)
    if "base__source_any_present" not in scored.columns:
        scored = scored.with_columns(pl.lit(1.0).alias("base__source_any_present"))
    direction = (
        pl.when(pl.col("path_prob_smooth_up") >= pl.col("path_prob_smooth_down"))
        .then(pl.lit("long"))
        .otherwise(pl.lit("short"))
    )
    trend_probability = pl.max_horizontal("path_prob_smooth_up", "path_prob_smooth_down")
    risk_probability = pl.max_horizontal("path_prob_chop", "path_prob_fake_breakout")
    return (
        scored.with_columns(
            direction.alias("direction"),
            trend_probability.alias("trend_probability"),
            risk_probability.alias("risk_probability"),
            pl.col("path_prob_calm").alias("calm_probability"),
            pl.lit(scored_at_ms).alias("scored_at_ms"),
            pl.lit(valid_for_hours).alias("valid_for_hours"),
        )
        .with_columns(
            pl.col("base__source_any_present")
            .fill_null(0.0)
            .clip(0.0, 1.0)
            .alias("source_any_present"),
            (pl.col("trend_probability") - pl.col("risk_probability")).alias("participation_score"),
            ((pl.lit(scored_at_ms) - pl.col("decision_bar_close_ms")) / 3_600_000.0).alias(
                "decision_age_hours"
            ),
            (pl.col("decision_bar_close_ms") + pl.lit(valid_for_ms)).alias(
                "prediction_valid_until_ms"
            ),
        )
        .with_columns(
            (pl.col("participation_score") * (0.5 + 0.5 * pl.col("source_any_present"))).alias(
                "source_presence_calibrated_score"
            ),
            pl.when(pl.col("decision_age_hours") <= pl.lit(valid_for_hours))
            .then(pl.lit("valid"))
            .otherwise(pl.lit("stale"))
            .alias("prediction_validity"),
            pl.from_epoch("decision_bar_close_ms", time_unit="ms")
            .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            .alias("decision_time_utc"),
            pl.from_epoch("prediction_valid_until_ms", time_unit="ms")
            .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            .alias("prediction_valid_until_utc"),
            pl.from_epoch("scored_at_ms", time_unit="ms")
            .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            .alias("scored_at_utc"),
        )
        .with_columns(
            pl.col("source_presence_calibrated_score").alias("promotion_score"),
            pl.lit(True).alias("side_gate_pass"),
            pl.lit("model side probabilities used directly").alias("side_gate_reason"),
            pl.lit(None, dtype=pl.Int64).alias("side_trade_count"),
            pl.lit(None, dtype=pl.Float64).alias("side_mean_margin"),
            pl.lit(None, dtype=pl.Float64).alias("side_false_direction_rate"),
        )
        .with_columns(
            pl.when(pl.col("promotion_score") > 0.0)
            .then(pl.lit("promote"))
            .otherwise(pl.lit("watch"))
            .alias("promotion_action"),
            pl.when(pl.col("promotion_score") > 0.0)
            .then(pl.lit("promote"))
            .otherwise(pl.lit("non-positive promotion score"))
            .alias("promotion_reason"),
        )
        .with_columns(
            (
                pl.lit("direction=")
                + pl.col("direction")
                + pl.lit(" trend=")
                + pl.col("trend_probability").round(3).cast(pl.Utf8)
                + pl.lit(" risk=")
                + pl.col("risk_probability").round(3).cast(pl.Utf8)
                + pl.lit(" calm=")
                + pl.col("calm_probability").round(3).cast(pl.Utf8)
                + pl.lit(" action=")
                + pl.col("promotion_action")
                + pl.lit(" validity=")
                + pl.col("prediction_validity")
            ).alias("reason"),
        )
        .select(
            "symbol",
            "decision_bar_close_ms",
            "decision_time_utc",
            "scored_at_utc",
            "decision_age_hours",
            "prediction_validity",
            "prediction_valid_until_utc",
            "valid_for_hours",
            "horizon_hours",
            "direction",
            "promotion_action",
            "promotion_reason",
            "promotion_score",
            "participation_score",
            "source_presence_calibrated_score",
            "source_any_present",
            "side_gate_pass",
            "side_gate_reason",
            "side_trade_count",
            "side_mean_margin",
            "side_false_direction_rate",
            "trend_probability",
            "risk_probability",
            "calm_probability",
            "path_confidence",
            "path_pred_label_name",
            "reason",
            "path_prob_smooth_up",
            "path_prob_smooth_down",
            "path_prob_chop",
            "path_prob_fake_breakout",
            "path_prob_calm",
        )
        .sort(["promotion_action", "promotion_score"], descending=[False, True])
    )


def best_per_symbol(board: pl.DataFrame) -> pl.DataFrame:
    return (
        board.with_columns(
            pl.when(pl.col("promotion_action") == "promote")
            .then(pl.lit(0))
            .otherwise(pl.lit(1))
            .alias("_action_rank")
        )
        .sort(["_action_rank", "promotion_score"], descending=[False, True])
        .unique("symbol", keep="first", maintain_order=True)
        .drop("_action_rank")
    )


def _freshness_summary(board: pl.DataFrame) -> dict[str, object]:
    if board.is_empty():
        return {
            "scored_at_utc": "",
            "valid_for_hours": RECENT_DECISION_MAX_AGE_HOURS,
            "fresh_rows": 0,
            "stale_rows": 0,
            "oldest_decision_age_hours": 0.0,
            "newest_decision_age_hours": 0.0,
        }
    return {
        "scored_at_utc": str(board.get_column("scored_at_utc")[0]),
        "valid_for_hours": float(board.get_column("valid_for_hours")[0]),
        "fresh_rows": board.filter(pl.col("prediction_validity") == "valid").height,
        "stale_rows": board.filter(pl.col("prediction_validity") == "stale").height,
        "oldest_decision_age_hours": float(board.get_column("decision_age_hours").max() or 0.0),
        "newest_decision_age_hours": float(board.get_column("decision_age_hours").min() or 0.0),
    }


def prediction_report(
    board: pl.DataFrame, *, model_id: str, selected_feature_count: int, top_n: int = 15
) -> str:
    best = best_per_symbol(board).head(top_n)
    freshness = _freshness_summary(board)
    lines = [
        "# Tailtree prediction report",
        "",
        "This is model evidence, not financial advice.",
        "",
        "## Summary",
        "",
        f"- model_id: `{model_id}`",
        f"- selected_features: {selected_feature_count}",
        f"- scored_rows: {board.height}",
        f"- ranked_symbols: {best.height}",
        f"- scored_at_utc: {freshness['scored_at_utc']}",
        f"- valid_for_hours: {freshness['valid_for_hours']:.1f}",
        f"- fresh_rows: {freshness['fresh_rows']}",
        f"- stale_rows: {freshness['stale_rows']}",
        f"- oldest_decision_age_hours: {freshness['oldest_decision_age_hours']:.2f}",
        f"- newest_decision_age_hours: {freshness['newest_decision_age_hours']:.2f}",
        "- participation_score = max(smooth_up, smooth_down) - max(chop, fake_breakout)",
        "- source_presence_calibrated_score = participation_score "
        "* (0.5 + 0.5 * source_any_present)",
        "- promotion_score = source_presence_calibrated_score",
        "- side selection uses model probabilities directly for long and short rows",
        "",
        "## Most worth participating",
        "",
        "| rank | symbol | decision_time_utc | decision_age_h | validity | horizon_h | side | "
        "promotion_action | promotion_score | participation_score | source_any_present | "
        "trend_probability | risk_probability | confidence | reason |",
        "|---:|---|---|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(best.to_dicts(), start=1):
        lines.append(
            (
                "| {rank} | {symbol} | {decision_time} | {age:.2f} | {validity} | "
                "{horizon} | {side} | {action} | {promotion:.3f} | {score:.3f} | "
                "{source_any:.0f} | {trend:.3f} | {risk:.3f} | {conf:.3f} | {reason} |"
            ).format(
                rank=rank,
                symbol=row["symbol"],
                decision_time=row["decision_time_utc"],
                age=float(row["decision_age_hours"]),
                validity=row["prediction_validity"],
                horizon=int(row["horizon_hours"]),
                side=row["direction"],
                action=row["promotion_action"],
                promotion=float(row["promotion_score"]),
                score=float(row["participation_score"]),
                source_any=float(row["source_any_present"]),
                trend=float(row["trend_probability"]),
                risk=float(row["risk_probability"]),
                conf=float(row["path_confidence"]),
                reason=row["reason"],
            )
        )
    ranked_symbols = [str(row["symbol"]) for row in best.select("symbol").to_dicts()]
    horizon_rows = (
        board.filter(pl.col("symbol").is_in(ranked_symbols))
        .sort(["symbol", "promotion_score"], descending=[False, True])
        .to_dicts()
    )
    lines.extend(
        [
            "",
            "## Per-symbol horizon ranking",
            "",
            "| symbol | decision_age_h | validity | horizon_h | side | promotion_action | "
            "promotion_score | participation_score | source_any_present | "
            "trend_probability | risk_probability | confidence | "
            "horizon_direction_conflict_count | reason |",
            "|---|---:|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    directions_by_symbol: dict[str, set[str]] = {}
    for row in horizon_rows:
        directions_by_symbol.setdefault(str(row["symbol"]), set()).add(str(row["direction"]))
    for row in horizon_rows:
        symbol = str(row["symbol"])
        conflict_count = max(0, len(directions_by_symbol.get(symbol, set())) - 1)
        note = " conflict/watch" if conflict_count else ""
        lines.append(
            (
                "| {symbol} | {age:.2f} | {validity} | {horizon} | {side} | {action} | "
                "{promotion:.3f} | {score:.3f} | {source_any:.0f} | {trend:.3f} | "
                "{risk:.3f} | {conf:.3f} | {conflicts} | {reason}{note} |"
            ).format(
                symbol=symbol,
                age=float(row["decision_age_hours"]),
                validity=row["prediction_validity"],
                horizon=int(row["horizon_hours"]),
                side=row["direction"],
                action=row["promotion_action"],
                promotion=float(row["promotion_score"]),
                score=float(row["participation_score"]),
                source_any=float(row["source_any_present"]),
                trend=float(row["trend_probability"]),
                risk=float(row["risk_probability"]),
                conf=float(row["path_confidence"]),
                conflicts=conflict_count,
                reason=row["reason"],
                note=note,
            )
        )
    watch = board.filter(pl.col("participation_score") > 0.20).height
    risk = board.filter(pl.col("risk_probability") >= pl.col("trend_probability")).height
    lines.extend(
        [
            "",
            "## Watchlist metrics",
            "",
            f"- rows with participation_score > 0.20: {watch}",
            f"- rows where risk_probability >= trend_probability: {risk}",
            "- Prefer high score with low risk_probability; stale rows need fresh data before use.",
            "",
            "## Model context",
            "",
            "- Path classes: calm, smooth_up, smooth_down, chop, fake_breakout.",
            "- Side is derived from smooth_up vs smooth_down probability.",
            "- Report uses recent latest feature rows per symbol × horizon from the "
            "accepted feature matrix.",
            "",
        ]
    )
    return "\n".join(lines)


def remove_legacy_outputs() -> None:
    for path in LEGACY_OUTPUT_DIRS:
        shutil.rmtree(path, ignore_errors=True)


def predict_scores() -> Path:
    manifest = AcceptedFeatureManifest.read(MANIFEST_PATH)
    model = TailTreeModel.from_json(MODEL_PATH)
    matrix = pl.read_parquet(FEATURE_MATRIX_PATH)
    require_recent_input(matrix)
    feature_rows = manifest.select_matrix(recent_latest_feature_rows(matrix))
    scored = model.score_path(feature_rows)
    if "base__source_any_present" in feature_rows.columns:
        scored = scored.join(
            feature_rows.select(
                "symbol",
                "decision_bar_close_ms",
                "horizon_hours",
                "base__source_any_present",
            ),
            on=["symbol", "decision_bar_close_ms", "horizon_hours"],
            how="left",
        )
    board = prediction_board(scored)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    board.write_parquet(BOARD_PATH)
    REPORT_PATH.write_text(
        prediction_report(
            board,
            model_id=MODEL_ID,
            selected_feature_count=len(manifest.selected_columns),
        ),
        encoding="utf-8",
    )
    remove_legacy_outputs()
    return BOARD_PATH


def main() -> None:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print("Score recent accepted path features and write prediction-report.md.")
        return
    board_path = predict_scores()
    print(board_path)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
