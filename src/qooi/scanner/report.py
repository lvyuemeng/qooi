"""Potential scanner Markdown report rendering.

Composable section-based design. Dispatch happens ONCE in render_report().
No evidence-path branching inside any section.
"""

from __future__ import annotations

from typing import Protocol

import polars as pl

from qooi.scanner import ReportInputs

# ── Protocol ─────────────────────────────────────────────────────────────────


class ReportSection(Protocol):
    """A composable section of the scanner report."""

    def render(self, inputs: ReportInputs) -> str: ...


# ── Orchestrator ─────────────────────────────────────────────────────────────


def render_report(inputs: ReportInputs) -> str:
    """Compose report from pre-built sections. No dispatch. No branching."""
    sections = (
        [_ScanScopeSection(), _SourceFreshnessSection(), _CoverageSection()]
        + list(inputs.report_sections)
        + [_CaveatsSection()]
    )
    parts = [s.render(inputs) for s in sections]
    return "\n\n".join(p for p in parts if p) + "\n"


def report_sections_for(evidence: str) -> tuple:
    """Return path-specific report sections. One dispatch point for the report."""
    if evidence == "tailtree":
        return (
            _TreeSummarySection(),
            _TreeImportanceSection(),
            _TreeLeafSection(),
            _TreeReviewSection(),
            _TreeGateSection(),
        )
    return (
        _LadderEvidenceSection(),
        _LadderReviewSection(),
        _LadderGateSection(),
    )


# ── Common sections ──────────────────────────────────────────────────────────


class _ScanScopeSection:
    def render(self, inputs: ReportInputs) -> str:
        c = inputs.config
        watch = sum(1 for d in inputs.decisions if d.group == "watch")
        blocked = sum(1 for d in inputs.decisions if d.group == "blocked")
        evidence_path = (
            "tailtree (LightGBM + GPD)"
            if c.evidence.kind == "tailtree"
            else "ladder (fixed 5-level)"
        )
        return "\n".join(
            [
                "# Potential Altcoin Diagnostics Report",
                "",
                "This is a research-only evidence report. It does not authorize live trading, "
                "place orders, mutate baskets, or bypass executor risk and sizing controls.",
                "",
                "## Scan Scope",
                "",
                f"- Universe: `{c.universe}`",
                f"- Bar: `{c.bar}`",
                f"- Days: `{c.days}`",
                f"- Transition history days: `{max(c.days, c.transition.history_days)}`",
                f"- Transition n-gram length: `{c.transition.ngram_length}`",
                f"- Evidence path: **{evidence_path}**",
                f"- Selected symbols scanned: `{len(inputs.universe.symbols)}` of "
                f"`{inputs.universe.eligible_count}` eligible symbols",
                f"- Transition scan budget: `{c.transition.scan_budget}`",
                f"- Refresh mode: `{c.refresh_mode}`",
                f"- Source context scope: `{c.transition.context_scope}`",
                f"- Transition watch rows: `{watch}`; blocked rows: `{blocked}`",
                f"- Unsupported current transition paths: `{len(inputs.transitions.unsupported)}` "
                f"in `{inputs.artifacts.diagnostics_dir / 'unsupported-current-paths.csv'}`",
                f"- Diagnostics directory: `{inputs.artifacts.diagnostics_dir}`",
            ]
        )


class _SourceFreshnessSection:
    def render(self, inputs: ReportInputs) -> str:
        path = inputs.artifacts.diagnostics_dir / "potential-observation-summary.csv"
        if not path.exists():
            return f"## Source Freshness\n\n- Missing observation summary: `{path}`"
        obs = pl.read_csv(path)
        if obs.is_empty():
            return f"## Source Freshness\n\n- Observation summary empty: `{path}`"
        total = int(obs.get_column("row_count").sum())
        return "\n".join(
            [
                "## Source Freshness",
                "",
                f"- Summary: `{path}`",
                f"- Summarized observation rows: `{total}`; groups: `{obs.height}`",
                "- Source freshness: " + _counts(obs, "source_freshness"),
                "- Source families: " + _counts(obs, "source_family"),
                "- Market alignment: " + _counts(obs, "market_alignment"),
            ]
        )


class _CoverageSection:
    def render(self, inputs: ReportInputs) -> str:
        d = inputs.artifacts.diagnostics_dir
        hist_path = d / "history-feasibility.csv"
        watch_path = d / "watchlist-feasibility.csv"
        lines = [
            "## Data Coverage And Feasibility",
            "",
            f"- History feasibility: `{hist_path}`",
            f"- Watchlist feasibility: `{watch_path}`",
        ]
        if hist_path.exists():
            hist = pl.read_csv(hist_path)
            if not hist.is_empty():
                lines.append("- History feasibility: " + _value_counts(hist, "feasibility_status"))
            else:
                lines.append("- History feasibility: no rows produced.")
        else:
            lines.append("- History feasibility artifact is missing.")

        if watch_path.exists():
            w = pl.read_csv(watch_path)
            if not w.is_empty():
                lines.append(
                    "- Watchlist feasibility: " + _value_counts(w, "watchlist_feasibility")
                )
                limited = w.filter(
                    pl.col("watchlist_feasibility").is_in(
                        ["blocked_by_history", "coverage_limited_review", "source_blind_review"]
                    )
                )
                if not limited.is_empty():
                    lines.extend(
                        [
                            "",
                            "| Symbol | Group | Feasibility | History | Sources | Reason |",
                            "|---|---|---|---|---|---|",
                        ]
                    )
                    for row in limited.head(8).iter_rows(named=True):
                        lines.append(
                            f"| `{row['symbol']}` | {row.get('group', '—')} | "
                            f"{row['watchlist_feasibility']} | {row.get('history_status', '—')} | "
                            f"{row.get('source_status', '—')} | {row.get('history_reason', '—')} |"
                        )
            else:
                lines.append("- Watchlist feasibility: no decision rows produced.")
        else:
            lines.append("- Watchlist feasibility artifact is missing.")
        return "\n".join(lines)


class _CaveatsSection:
    def render(self, inputs: ReportInputs) -> str:
        use_tree = inputs.config.evidence.kind == "tailtree"
        base = [
            "## Baseline Caveats",
            "",
            "- **Coverage gaps on short-lived symbols**: new or delisted contracts have shallow "
            "history, so 730-day targets cannot be met. This is data reality, not a scanner bug.",
            "- **Context-blind by design**: we lack a full-fledged message/social source; "
            "ephemeral signals (news, sentiment, events) are hard to transform into "
            "decisive quantitative data. Missing context is surfaced explicitly rather "
            "than imputed.",
        ]
        if use_tree:
            base.extend(
                [
                    "- **GPD fits on 30+ exceedances have wider confidence intervals** "
                    "than entropy on 400+ observations. This is the trade-off for "
                    "targeting tail extremes rather than average returns. Leaves with "
                    "N_tail < 30 are excluded by the tail gate.",
                    "- **Two trees, directional separation**: Tree_UP identifies "
                    "conditions for extreme-up moves; Tree_DOWN identifies conditions "
                    "for extreme-down moves. A single leaf can only predict one tail "
                    "direction. Leaves from both trees may coexist for the same observation.",
                    "- **Feature importance is gain-based** from LightGBM leaf-wise "
                    "splits. High-gain features drive tail-heaviness separation. "
                    "Low-gain features may still provide context but don't materially "
                    "shift the GPD shape parameter.",
                ]
            )
        else:
            base.extend(
                [
                    "- **Evidence insufficiency is expected**: most state-vector "
                    "observations do not co-occur with enough outcome history to "
                    "produce statistically stable evidence. The scanner measures how "
                    "much signal exists, not claiming signal exists everywhere.",
                    "- **Baseline hit rates near 50-57%** reflect inherent difficulty "
                    "of predicting short-horizon crypto returns. Higher-tail candidates "
                    "(`market_swing`, `market_decision`) tend to have cleaner tail vs "
                    "adverse gaps.",
                ]
            )
        return "\n".join(base)


# ── Ladder path sections ────────────────────────────────────────────────────


class _LadderEvidenceSection:
    def render(self, inputs: ReportInputs) -> str:
        d = inputs.artifacts.diagnostics_dir
        summary_path = d / "potential-evidence-summary.csv"
        selected_path = d / "potential-evidence-selected.csv"
        if not summary_path.exists():
            return f"## Unified Evidence Surface\n\n- Missing: `{summary_path}`"

        evidence = pl.read_csv(summary_path)
        if evidence.is_empty():
            return "## Unified Evidence Surface\n\n- No evidence rows produced."

        total = int(evidence.get_column("row_count").sum())
        selected_rows = int(
            evidence.filter(pl.col("selected_evidence_level")).get_column("row_count").sum()
        )
        lines = [
            "## Unified Evidence Surface",
            "",
            f"- Summarized evidence rows: `{total}`; groups: `{evidence.height}`",
            "- Evidence levels: " + _counts(evidence, "evidence_level"),
            "- Evidence status: " + _counts(evidence, "evidence_status"),
            "- Transition status: " + _counts(evidence, "transition_status"),
            "- Statistical direction: " + _counts(evidence, "statistical_direction"),
            "- Research suggestions: " + _counts(evidence, "research_suggestion"),
            f"- Selected evidence rows: `{selected_rows}`",
        ]
        if selected_rows == 0 or not selected_path.exists():
            lines.append("- No evidence level passed gates.")
            return "\n".join(lines)

        selected = pl.read_csv(selected_path)
        if selected.is_empty():
            return "\n".join(lines)

        lines.extend(
            [
                "",
                "| Level | Horizon | Suggestion | Direction | Obs | Symbols | Info | "
                "Trans | TailUp | TailDown | MaxPath | MinPath | OriginRate | Skew |",
                "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        top = selected.sort(
            [
                "transition_information_gain_bits",
                "information_gain_bits",
                "conditioned_observations",
            ],
            descending=[True, True, True],
        ).head(12)
        for row in top.iter_rows(named=True):
            lines.append(
                f"| `{row['evidence_level']}` | {row['outcome_horizon']} | "
                f"{row['research_suggestion']} | {row['statistical_direction']} | "
                f"{row['conditioned_observations']} | {row['symbol_count']} | "
                f"{_fmt(row.get('information_gain_bits'))} | "
                f"{_fmt(row.get('transition_information_gain_bits'))} | "
                f"{_fmt(row.get('tail_up_rate'))} | {_fmt(row.get('tail_down_rate'))} | "
                f"{_fmt(row.get('avg_forward_max_return_pct'))} | "
                f"{_fmt(row.get('avg_forward_min_return_pct'))} | "
                f"{_fmt(row.get('returned_to_origin_rate'))} | "
                f"{_fmt(row.get('path_skew'))} |"
            )
        return "\n".join(lines)


class _LadderReviewSection:
    def render(self, inputs: ReportInputs) -> str:
        decisions = [d for d in inputs.decisions if d.group == "watch"]
        lines = [
            "## Review Rows",
            "",
            "Tiers: 1=top-decile Info,Sym≥15  |  2=top-quartile Info  | 3=ranked  |  —=no evidence",
            "",
            "< evidence (cross-coin) | decision (coin-specific) >",
        ]
        if not decisions:
            lines.append("\n- No watchlist candidates.")
            return "\n".join(lines)

        rank_path = inputs.artifacts.diagnostics_dir / "candidate-rank.csv"
        rank_data = _read_rank_data(rank_path)
        lines.extend(
            [
                "| T | Symbol | Info | Rank | Direction | Suggestion | Caveat |",
                "|---|---:|---:|---:|---:|---:|---:|",
                "| | | bits | score | | | |",
            ]
        )
        for d in decisions[:15]:
            rd = rank_data.get(d.symbol, {})
            tier = rd.get("tier", "—") if rd else "—"
            info = _fmt(rd.get("transition_information_gain_bits"))
            rank = _fmt(rd.get("rank_score"))
            suggestion = _suggestion_from_decision(d)
            lines.append(
                f"| {tier} | `{d.symbol}` | {info} | {rank} | "
                f"{d.direction} | {suggestion} | {d.review_caveat} |"
            )
        if len(decisions) > 15:
            lines.append(f"- {len(decisions) - 15} additional watch rows omitted.")
        return "\n".join(lines)


class _LadderGateSection:
    def render(self, inputs: ReportInputs) -> str:
        path = inputs.artifacts.diagnostics_dir / "potential-evidence-summary.csv"
        if not path.exists():
            return "## Evidence Gate Summary\n\n- Evidence summary artifact missing."
        evidence = pl.read_csv(path)
        if evidence.is_empty():
            return "## Evidence Gate Summary\n\n- No evidence rows."
        selected = int(
            evidence.filter(pl.col("selected_evidence_level")).get_column("row_count").sum()
        )
        if selected:
            return (
                "## Evidence Gate Summary\n\n"
                "- At least one parent-gated evidence row is reviewable."
            )
        return "\n".join(
            [
                "## Evidence Gate Summary",
                "- No parent-gated evidence row is currently reviewable.",
                f"- Evidence status: {_counts(evidence, 'evidence_status')}",
                f"- Transition status: {_counts(evidence, 'transition_status')}",
                "- This is an evidence-quality result, not a neutral trading signal.",
            ]
        )


# ── Tree path sections ──────────────────────────────────────────────────────


class _TreeSummarySection:
    def render(self, inputs: ReportInputs) -> str:
        d = inputs.artifacts.diagnostics_dir
        lines = ["## Tail Tree Evidence", "", "Path: LightGBM + GPD (tail exceedances only)", ""]

        for direction in ("up", "down"):
            tree_path = d / f"tail-tree-{direction}.json"
            leaf_path = d / "potential-leaf-evidence.csv"
            sel_path = d / f"potential-leaves-selected-{direction}.csv"

            if tree_path.exists():
                lines.append(f"- Tree_{direction.upper()}: model saved to `{tree_path.name}`")
            else:
                lines.append(
                    f"- Tree_{direction.upper()}: not trained (insufficient tail exceedances)"
                )

        if leaf_path.exists():
            evidence = pl.read_csv(leaf_path)
            if not evidence.is_empty():
                total_leaves = evidence.get_column("leaf_id").n_unique()
                lines.append(f"- Total leaves: `{total_leaves}` across both trees")

            for direction in ("up", "down"):
                if sel_path.exists():
                    sel = pl.read_csv(sel_path) if sel_path.exists() else pl.DataFrame()
                    if not sel.is_empty():
                        dir_sel = (
                            sel.filter(pl.col("tree_direction") == direction)
                            if "tree_direction" in sel.columns
                            else sel
                        )
                        lines.append(
                            f"- Tree_{direction.upper()} leaves passing tail gate: "
                            f"`{dir_sel.height}`"
                        )

        # Global baseline from any tree JSON
        for direction in ("up", "down"):
            tree_path = d / f"tail-tree-{direction}.json"
            if tree_path.exists():
                import json

                with open(tree_path) as f:
                    data = json.load(f)
                gb = data.get("metadata", {}).get("global_baseline", {})
                if gb:
                    lines.append(
                        f"- Global baseline (from Tree_{direction.upper()}): "
                        f"ξ={gb.get('xi', '?'):.3f}, σ={gb.get('sigma', '?'):.2f}, "
                        f"tail_rate={gb.get('tail_rate', 0) * 100:.1f}%"
                    )
                    break

        return "\n".join(lines)


class _TreeImportanceSection:
    def render(self, inputs: ReportInputs) -> str:
        d = inputs.artifacts.diagnostics_dir
        imp_path = d / "tail-tree-feature-importance.csv"
        if not imp_path.exists():
            # Try to extract from tree JSON
            import json

            rows = []
            for direction in ("up", "down"):
                tree_path = d / f"tail-tree-{direction}.json"
                if tree_path.exists():
                    with open(tree_path) as f:
                        data = json.load(f)
                    fi = data.get("metadata", {}).get("feature_importance", [])
                    for feature, gain in fi:
                        rows.append({"feature": feature, f"gain_{direction}": float(gain)})
            if not rows:
                return ""

            imp = pl.DataFrame(rows)
            if "gain_up" not in imp.columns:
                imp = imp.with_columns(pl.lit(None, dtype=pl.Float64).alias("gain_up"))
            if "gain_down" not in imp.columns:
                imp = imp.with_columns(pl.lit(None, dtype=pl.Float64).alias("gain_down"))
        else:
            imp = pl.read_csv(imp_path)

        if imp.is_empty():
            return ""

        imp = (
            imp.sort("gain_up", descending=True, nulls_last=True)
            if "gain_up" in imp.columns
            else imp
        )
        lines = [
            "## Feature Importance",
            "",
            "Gain-based importance from LightGBM leaf-wise splits. Higher gain = feature "
            "drives tail-heaviness separation more.",
            "",
            "| Feature | Gain UP | Gain DOWN |",
            "|---|---:|---:|",
        ]
        for row in imp.head(12).iter_rows(named=True):
            gu = _fmt(row.get("gain_up"))
            gd = _fmt(row.get("gain_down"))
            lines.append(f"| `{row['feature']}` | {gu} | {gd} |")
        return "\n".join(lines)


class _TreeLeafSection:
    def render(self, inputs: ReportInputs) -> str:
        d = inputs.artifacts.diagnostics_dir
        leaf_path = d / "potential-leaf-evidence.csv"
        if not leaf_path.exists():
            return ""

        evidence = pl.read_csv(leaf_path)
        if evidence.is_empty():
            return ""

        top = evidence.sort("tail_lift", descending=True, nulls_last=True).head(10)
        lines = [
            "## Top Tail Leaves",
            "",
            "Leaves ranked by tail_lift (multiplicative increase in extreme-move probability "
            "vs global baseline).",
            "",
            "| Dir | Leaf | Lift | ξ | σ | N_tail | Path |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in top.iter_rows(named=True):
            direction = row.get("tree_direction", "—")
            leaf_id = row.get("leaf_id", "—")
            lift = _fmt(row.get("tail_lift"))
            xi = _fmt(row.get("gpd_shape_xi"))
            sigma = _fmt(row.get("gpd_scale_sigma"))
            n_tail = row.get("N_tail_exceedances", "—")
            lpath = row.get("leaf_path", "—")
            if lpath and len(str(lpath)) > 80:
                lpath = str(lpath)[:77] + "..."
            lines.append(
                f"| {direction} | {leaf_id} | {lift} | {xi} | {sigma} | {n_tail} | {lpath} |"
            )
        return "\n".join(lines)


class _TreeReviewSection:
    def render(self, inputs: ReportInputs) -> str:
        decisions = [d for d in inputs.decisions if d.group == "watch"]
        if not decisions:
            return "## Review Rows\n\n- No watchlist candidates."

        rank_path = inputs.artifacts.diagnostics_dir / "candidate-rank.csv"
        rank_data = _read_rank_data(rank_path)
        lines = [
            "## Review Rows",
            "",
            "Tiers: 1=tail_lift≥2.0,ξ>0.15,N≥50  |  2=lift≥1.5,N≥30  | 3=lift≥1.0  |  —=below gate",
            "",
            "< evidence (cross-coin) | decision (coin-specific) >",
            "| T | Symbol | TailLift | ξ | Depth | Direction | Leaf | Caveat |",
            "|---|---:|---:|---:|---:|---:|---:|",
            "| | | × | | | | | |",
        ]
        for d in decisions[:15]:
            rd = rank_data.get(d.symbol, {})
            tier = rd.get("tier", "—") if rd else "—"
            lift = _fmt(rd.get("tail_lift"))
            xi = _fmt(rd.get("gpd_shape_xi"))
            depth = str(rd.get("leaf_depth", "—")) if rd.get("leaf_depth") is not None else "—"
            lpath = str(rd.get("leaf_path", "—")) if rd.get("leaf_path") else "—"
            if len(lpath) > 60:
                lpath = lpath[:57] + "..."
            lines.append(
                f"| {tier} | `{d.symbol}` | {lift} | {xi} | {depth} | "
                f"{d.direction} | {lpath} | {d.review_caveat} |"
            )
        if len(decisions) > 15:
            lines.append(f"- {len(decisions) - 15} additional watch rows omitted.")
        return "\n".join(lines)


class _TreeGateSection:
    def render(self, inputs: ReportInputs) -> str:
        d = inputs.artifacts.diagnostics_dir
        lines = ["## Tail Gate Summary", ""]

        total_passing = 0
        for direction in ("up", "down"):
            sel_path = d / f"potential-leaves-selected-{direction}.csv"
            if sel_path.exists():
                sel = pl.read_csv(sel_path)
                if not sel.is_empty():
                    lines.append(
                        f"- Tree_{direction.upper()}: `{sel.height}` leaves passed "
                        f"(≥30 exceedances, lift≥1.5×, ξ stable)"
                    )
                    total_passing += sel.height
            else:
                lines.append(f"- Tree_{direction.upper()}: no selected leaves artifact.")

        if total_passing == 0:
            lines.append(
                "- No leaves passed the tail gate. Review feature importance and "
                "consider lowering tail_threshold_pct or increasing history."
            )

        return "\n".join(lines)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _read_rank_data(rank_path) -> dict[str, dict]:
    if not rank_path.exists():
        return {}
    df = pl.read_csv(rank_path).sort("rank_score", descending=True)

    has_tail = "tail_lift" in df.columns and df["tail_lift"].drop_nulls().len() > 0
    if has_tail:
        df = df.with_columns(
            pl.when(
                (pl.col("tail_lift") >= 2.0)
                & (pl.col("N_tail_exceedances").fill_null(0) >= 50)
                & (pl.col("gpd_shape_xi").fill_null(0) > 0.15)
            )
            .then(pl.lit("1"))
            .when((pl.col("tail_lift") >= 1.5) & (pl.col("N_tail_exceedances").fill_null(0) >= 30))
            .then(pl.lit("2"))
            .when(pl.col("tail_lift") >= 1.0)
            .then(pl.lit("3"))
            .otherwise(pl.lit("—"))
            .alias("tier"),
        )
    else:
        decile = df.get_column("rank_score").quantile(0.9)
        quartile = df.get_column("rank_score").quantile(0.75)
        df = df.with_columns(
            pl.when(
                (pl.col("transition_information_gain_bits").fill_null(0) >= decile)
                & (pl.col("symbol_count").fill_null(0) >= 15)
            )
            .then(pl.lit("1"))
            .when(pl.col("transition_information_gain_bits").fill_null(0) >= quartile)
            .then(pl.lit("2"))
            .when(pl.col("rank_score") > 0)
            .then(pl.lit("3"))
            .otherwise(pl.lit("—"))
            .alias("tier"),
        )

    result = {}
    for row in df.iter_rows(named=True):
        result[row["symbol"]] = row
    return result


def _suggestion_from_decision(decision) -> str:
    if decision.block_reason:
        return f"watch: {decision.block_reason}"
    if decision.transition_evidence:
        for part in decision.transition_evidence.split("; "):
            if part.startswith("suggestion="):
                return part.removeprefix("suggestion=")
    return "research review only"


def _counts(frame: pl.DataFrame, column: str) -> str:
    if column not in frame.columns:
        return "missing"
    counts = (
        frame.group_by(column, maintain_order=True)
        .agg(pl.col("row_count").sum())
        .sort("row_count", descending=True)
    )
    return (
        ", ".join(
            f"{row[column]}={row['row_count']}" for row in counts.head(8).iter_rows(named=True)
        )
        or "none"
    )


def _value_counts(frame: pl.DataFrame, column: str) -> str:
    if column not in frame.columns:
        return "missing"
    counts = (
        frame.get_column(column).fill_null("missing").value_counts().sort("count", descending=True)
    )
    return (
        ", ".join(f"{row[column]}={row['count']}" for row in counts.head(8).iter_rows(named=True))
        or "none"
    )


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"
