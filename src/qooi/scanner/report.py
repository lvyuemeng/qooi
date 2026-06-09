"""Potential scanner Markdown report rendering."""

from __future__ import annotations

import polars as pl

from qooi.scanner.contracts import ReportInputs, ScanDecision


def render_report(inputs: ReportInputs) -> str:
    config = inputs.config
    watch = sum(1 for decision in inputs.decisions if decision.group == "watch")
    blocked = sum(1 for decision in inputs.decisions if decision.group == "blocked")
    lines = [
        "# Potential Altcoin Diagnostics Report",
        "",
        "This is a research-only evidence report. It does not authorize live trading, place "
        "orders, mutate baskets, or bypass executor risk and sizing controls.",
        "",
        "## Scan Scope",
        "",
        f"- Universe: `{config.universe}`",
        f"- Bar: `{config.bar}`",
        f"- Days: `{config.days}`",
        f"- Transition history days: `{max(config.days, config.transition_history_days)}`",
        f"- Transition n-gram length: `{config.transition_ngram_length}`",
        f"- Selected symbols scanned: `{len(inputs.universe.symbols)}` of "
        f"`{inputs.universe.eligible_count}` eligible symbols",
        f"- Transition scan budget: `{config.transition_scan_budget}`",
        f"- Refresh mode: `{config.refresh_mode}`",
        f"- Source context scope: `{config.transition_context_scope}`",
        f"- Context source limit: `{config.transition_context_limit}`",
        f"- Transition watch rows: `{watch}`; blocked rows: `{blocked}`",
        f"- Unsupported current transition paths: `{len(inputs.transitions.unsupported)}` "
        f"in `{inputs.artifacts.diagnostics_dir / 'unsupported-current-paths.csv'}`",
        f"- Diagnostics directory: `{inputs.artifacts.diagnostics_dir}`",
        "",
        "## Unified Evidence Surface",
        "",
        *_potential_evidence_report_lines(inputs),
        "",
        "## Source Freshness",
        "",
        *_potential_observation_report_lines(inputs),
        "",
        "## Data Coverage And Feasibility",
        "",
        *_feasibility_report_lines(inputs),
        "",
        "## Method-Grouped Review Rows",
        "",
        "These rows are grouped deterministic review methods, not compatibility "
        "or trading surfaces.",
        "",
        *_candidate_lines(inputs.decisions, "watch", limit=15),
        "",
        "## Evidence Gate Summary",
        "",
        *_evidence_gate_lines(inputs),
        "",
        "## Interpretation",
        "",
        "- `potential-observation.csv` is the known-at-close state vector surface.",
        "- `potential-evidence.csv` is the primary parent-gated evidence surface.",
        "- Future returns and transitions are outcome columns only; they do not feed current "
        "state construction.",
        "- Suggestions are neutral research labels; `statistical_direction` carries empirical "
        "direction separately.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _potential_evidence_report_lines(inputs: ReportInputs) -> list[str]:
    path = inputs.artifacts.diagnostics_dir / "potential-evidence.csv"
    if not path.exists():
        return [f"- Missing primary evidence artifact: `{path}`"]
    evidence = pl.read_csv(path)
    if evidence.is_empty():
        return [
            f"- Primary evidence artifact exists but has no rows: `{path}`",
            "- No current state vector has enough outcome history for evidence scoring.",
        ]
    lines = [
        f"- Artifact: `{path}`",
        f"- Rows: `{evidence.height}`; columns: `{len(evidence.columns)}`",
        "- Evidence levels: " + _value_count_text(evidence, "evidence_level"),
        "- Evidence status: " + _value_count_text(evidence, "evidence_status"),
        "- Transition status: " + _value_count_text(evidence, "transition_status"),
        "- Statistical direction: " + _value_count_text(evidence, "statistical_direction"),
        "- Research suggestions: " + _value_count_text(evidence, "research_suggestion"),
    ]
    selected = evidence.filter(pl.col("selected_evidence_level"))
    if selected.is_empty():
        lines.append("- Selected parent-gated evidence rows: `0`")
        lines.append(
            "- Result: no evidence level passed sample, symbol, information, and stability gates."
        )
        return lines
    lines.append(f"- Selected parent-gated evidence rows: `{selected.height}`")
    lines.extend([
        "",
        "| Level | Horizon | Suggestion | Direction | Observations | Symbols | Info Bits | "
        "Transition Bits | Tail Up | Tail Down | Max Path | Min Path | Origin Rate | Path Skew |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    top = selected.sort(
        ["transition_information_gain_bits", "information_gain_bits", "conditioned_observations"],
        descending=[True, True, True],
    ).head(12)
    for row in top.iter_rows(named=True):
        lines.append(
            f"| `{row['evidence_level']}` | {row['outcome_horizon']} | "
            f"{row['research_suggestion']} | {row['statistical_direction']} | "
            f"{row['conditioned_observations']} | {row['symbol_count']} | "
            f"{_format_float(row['information_gain_bits'])} | "
            f"{_format_float(row['transition_information_gain_bits'])} | "
            f"{_format_float(row['tail_up_rate'])} | "
            f"{_format_float(row['tail_down_rate'])} | "
            f"{_format_float(row['avg_forward_max_return_pct'])} | "
            f"{_format_float(row['avg_forward_min_return_pct'])} | "
            f"{_format_float(row['returned_to_origin_rate'])} | "
            f"{_format_float(row['path_skew'])} |"
        )
    return lines


def _potential_observation_report_lines(inputs: ReportInputs) -> list[str]:
    path = inputs.artifacts.diagnostics_dir / "potential-observation.csv"
    if not path.exists():
        return [f"- Missing observation artifact: `{path}`"]
    observations = pl.read_csv(path)
    if observations.is_empty():
        return [f"- Observation artifact exists but has no rows: `{path}`"]
    return [
        f"- Artifact: `{path}`",
        f"- Rows: `{observations.height}`; columns: `{len(observations.columns)}`",
        "- Source freshness: " + _value_count_text(observations, "source_freshness"),
        "- Source families: " + _value_count_text(observations, "source_family"),
        "- Market alignment: " + _value_count_text(observations, "market_alignment"),
    ]


def _evidence_gate_lines(inputs: ReportInputs) -> list[str]:
    evidence_path = inputs.artifacts.diagnostics_dir / "potential-evidence.csv"
    if not evidence_path.exists():
        return ["- Evidence gates could not run because the primary evidence artifact is missing."]
    evidence = pl.read_csv(evidence_path)
    if evidence.is_empty():
        return [
            "- Evidence gates produced no rows; inspect coverage and realized-transition artifacts."
        ]
    selected = evidence.filter(pl.col("selected_evidence_level"))
    if not selected.is_empty():
        return ["- At least one parent-gated evidence row is reviewable for research follow-up."]
    status = _value_count_text(evidence, "evidence_status")
    transition_status = _value_count_text(evidence, "transition_status")
    return [
        "- No parent-gated evidence row is currently reviewable.",
        f"- Evidence status distribution: {status}",
        f"- Transition status distribution: {transition_status}",
        "- This is an evidence-quality result, not a neutral trading signal.",
    ]


def _feasibility_report_lines(inputs: ReportInputs) -> list[str]:
    history_path = inputs.artifacts.diagnostics_dir / "history-feasibility.csv"
    watchlist_path = inputs.artifacts.diagnostics_dir / "watchlist-feasibility.csv"
    lines = [
        f"- History feasibility artifact: `{history_path}`",
        f"- Watchlist feasibility artifact: `{watchlist_path}`",
    ]
    if history_path.exists():
        history = pl.read_csv(history_path)
        if history.is_empty():
            lines.append("- History feasibility: no coverage rows were produced.")
        else:
            lines.append(
                "- History feasibility: " + _value_count_text(history, "feasibility_status")
            )
    else:
        lines.append("- History feasibility artifact is missing.")
    if watchlist_path.exists():
        watchlist = pl.read_csv(watchlist_path)
        if watchlist.is_empty():
            lines.append("- Watchlist feasibility: no decision rows were produced.")
        else:
            lines.append(
                "- Watchlist feasibility: "
                + _value_count_text(watchlist, "watchlist_feasibility")
            )
            limited = watchlist.filter(
                pl.col("watchlist_feasibility").is_in(
                    ["blocked_by_history", "coverage_limited_review", "source_blind_review"]
                )
            )
            if not limited.is_empty():
                lines.extend([
                    "",
                    "| Symbol | Decision Group | Feasibility | History | Sources | Reason |",
                    "|---|---|---|---|---|---|",
                ])
                for row in limited.head(8).iter_rows(named=True):
                    lines.append(
                        f"| `{row['symbol']}` | {row['group']} | "
                        f"{row['watchlist_feasibility']} | {row['history_status']} | "
                        f"{row['source_status']} | {row['history_reason']} |"
                    )
    else:
        lines.append("- Watchlist feasibility artifact is missing.")
    return lines


def _value_count_text(frame: pl.DataFrame, column: str) -> str:
    if column not in frame.columns:
        return "missing"
    counts = frame.get_column(column).fill_null("missing").value_counts().sort(
        "count", descending=True
    )
    return ", ".join(
        f"{row[column]}={row['count']}" for row in counts.head(8).iter_rows(named=True)
    ) or "none"


def _format_float(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _candidate_lines(decisions: tuple[ScanDecision, ...], group: str, *, limit: int) -> list[str]:
    selected = [decision for decision in decisions if decision.group == group]
    if not selected:
        return ["- None."]
    lines = [
        "| Symbol | Direction | Confidence | Transition Evidence | Outcome Metrics | Suggestion | "
        "Confirmations | Contradictions | Missing Critical Evidence | Review Caveat | Reason |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for decision in selected[:limit]:
        evidence_parts = (
            decision.transition_evidence.split("; ") if decision.transition_evidence else ()
        )
        outcome_metrics = "; ".join(
            part
            for part in evidence_parts
            if part.startswith(
                (
                    "p_up=",
                    "p_down=",
                    "p_recent=",
                    "p_long=",
                    "p_delta=",
                    "expected_return=",
                    "loss_stop=",
                    "profit_stop=",
                    "rr=",
                )
            )
        )
        suggestion = "research review only"
        for part in evidence_parts:
            if part.startswith("suggestion="):
                suggestion = part.removeprefix("suggestion=")
        if decision.block_reason:
            suggestion = f"watch: {decision.block_reason}"
        lines.append(
            f"| `{decision.symbol}` | {decision.direction} | {decision.confidence} | "
            f"{_compact_evidence(decision.transition_evidence)} | "
            f"{outcome_metrics or 'n/a'} | "
            f"{suggestion} | {_confirmations_text(decision)} | "
            f"{_joined_or_none(decision.contradictory_evidence)} | "
            f"{_joined_or_none(decision.missing_evidence)} | "
            f"{decision.review_caveat} | {decision.block_reason or 'reviewable'} |"
        )
    if len(selected) > limit:
        lines.append(
            f"- {len(selected) - limit} additional `{group}` rows omitted; "
            "see diagnostics artifacts."
        )
    return lines


def _confirmations_text(decision: ScanDecision) -> str:
    confirmations = []
    if _evidence_confirms(decision.flow_evidence, decision.direction):
        confirmations.append("trades")
    if _evidence_confirms(decision.liquidity_evidence, decision.direction):
        confirmations.append("books")
    if _evidence_confirms(decision.derivatives_evidence, decision.direction):
        confirmations.append("derivatives")
    if _evidence_confirms(decision.context_evidence, decision.direction):
        confirmations.append("context")
    return ", ".join(confirmations) if confirmations else "none"


def _evidence_confirms(evidence: str, direction: str) -> bool:
    if direction not in {"bullish", "bearish"} or "missing" in evidence or "disabled" in evidence:
        return False
    terms = (
        ("buy", "bid", "bull", "constructive")
        if direction == "bullish"
        else ("sell", "ask", "bear")
    )
    return any(term in evidence.lower() for term in terms)


def _compact_evidence(evidence: str) -> str:
    if not evidence:
        return "none"
    parts = evidence.split("; ")
    selected = [
        part
        for part in parts
        if part.startswith(("timeframe=", "path=", "event=", "p="))
    ]
    return "; ".join(selected[:5]) if selected else evidence[:160]


def _joined_or_none(values: tuple[str, ...]) -> str:
    return ", ".join(values) if values else "none"
