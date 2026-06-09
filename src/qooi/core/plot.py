"""Dataframe-driven plotting helpers for diagnostics artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def plot_market_state_modulation_heatmap(
    frame: pl.DataFrame,
    *,
    output_path: str | Path,
    limit: int = 30,
) -> Path:
    path = Path(output_path)
    plot_frame = _top_modulation_rows(frame, limit=limit)
    if plot_frame.is_empty():
        return _empty_plot(path, "No market-state modulation rows")
    row_records = list(plot_frame.iter_rows(named=True))
    rows = [f"{row['base_feature']}={row['base_value']}" for row in row_records]
    cols = [f"{row['modulator']}={row['modulator_value']}" for row in row_records]
    values = np.asarray(
        [
            float(row.get("delta_cohens_d") or row.get("delta_return_pct") or 0.0)
            for row in row_records
        ],
        dtype=float,
    )
    matrix = np.diag(values)
    height = max(4.0, min(14.0, len(rows) * 0.35))
    width = max(6.0, min(16.0, len(cols) * 0.35))
    fig, ax = plt.subplots(figsize=(width, height))
    vmax = max(float(np.max(np.abs(values))), 1e-9)
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(np.arange(len(rows)), labels=rows, fontsize=7)
    ax.set_xticks(np.arange(len(cols)), labels=cols, rotation=60, ha="right", fontsize=7)
    ax.set_title("Market-state modulation effect size")
    ax.set_xlabel("Modulator value")
    ax.set_ylabel("Base state")
    fig.colorbar(image, ax=ax, label="Cohen's d / delta")
    fig.tight_layout()
    return _save_plot(fig, path)


def plot_market_state_horizon_decay(
    frame: pl.DataFrame,
    *,
    output_path: str | Path,
    limit: int = 8,
) -> Path:
    path = Path(output_path)
    if frame.is_empty() or "horizon" not in frame.columns:
        return _empty_plot(path, "No horizon decay rows")
    effect_col = "delta_cohens_d" if "delta_cohens_d" in frame.columns else "delta_return_pct"
    key_expr = pl.concat_str(
        [
            pl.col("base_feature"),
            pl.lit("="),
            pl.col("base_value"),
            pl.lit(" x "),
            pl.col("modulator"),
            pl.lit("="),
            pl.col("modulator_value"),
            pl.lit(" / "),
            pl.col("outcome_kind").fill_null("return_pct"),
        ]
    )
    work = frame.with_columns(
        key_expr.alias("_plot_key"),
        pl.col(effect_col).abs().alias("_abs_effect"),
    )
    keys = (
        work.group_by("_plot_key")
        .agg(pl.col("_abs_effect").max().alias("max_abs_effect"))
        .sort("max_abs_effect", descending=True)
        .head(limit)["_plot_key"]
        .to_list()
    )
    work = work.filter(pl.col("_plot_key").is_in(keys)).sort(["_plot_key", "horizon"])
    if work.is_empty():
        return _empty_plot(path, "No horizon decay rows")
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, group in work.group_by("_plot_key", maintain_order=True):
        label = str(key[0] if isinstance(key, tuple) else key)
        ax.plot(
            group["horizon"].to_list(),
            group[effect_col].to_list(),
            marker="o",
            linewidth=1.2,
            label=label,
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Market-state modulation horizon decay")
    ax.set_xlabel("Forward horizon")
    ax.set_ylabel(effect_col)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    return _save_plot(fig, path)


def _top_modulation_rows(frame: pl.DataFrame, *, limit: int) -> pl.DataFrame:
    if frame.is_empty() or "delta_return_pct" not in frame.columns:
        return pl.DataFrame()
    effect_col = "delta_cohens_d" if "delta_cohens_d" in frame.columns else "delta_return_pct"
    return (
        frame.with_columns(pl.col(effect_col).abs().alias("_abs_effect"))
        .sort(["_abs_effect", "conditional_rows"], descending=[True, True])
        .head(limit)
    )


def _empty_plot(path: Path, message: str) -> Path:
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.axis("off")
    return _save_plot(fig, path)


def _save_plot(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path

