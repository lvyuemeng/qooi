# Scanner Evidence Module Graph (Ladder Path)

Fixed 5-level evidence ladder. Active when `config.evidence = "ladder"`.
Consumed by `docs/graph/scanner.md`. Replaced by `docs/graph/tailtree.md` when
`config.evidence = "tailtree"`.

Design doc: `docs/architecture/scanner.md`.

---

## Purpose

Group observations by fixed state-feature levels, compute entropy-based information gain versus parent context, gate on stability and parent improvement. Answers: "does conditioning on this state variable reduce outcome uncertainty?"

## Public API

### `qooi.scanner.evidence.potential_evidence_frame`

```text
qooi.scanner.evidence.potential_evidence_frame(
    observations: pl.DataFrame,           # from potential_observation_frame
    source_outcomes: pl.DataFrame,        # from source_events.source_outcomes_frame
    realized_transitions: pl.DataFrame,   # from history.realized_transition_frame
    *,
    return_threshold_pct: float,
) -> pl.DataFrame
```

**Evidence ladder (5 hardcoded levels):**

| Level | Group columns |
|---|---|
| `market_background` | `background_regime` |
| `market_swing` | `background_regime`, `swing_core` |
| `market_decision` | `background_regime`, `swing_core`, `decision_core`, `decision_transition` |
| `market_decision_source` | above + `source_family`, `source_state` |
| `market_decision_source_risk` | above + `risk_context` |

**Computation per level:**
1. Join observations to outcomes (market-level: `realized_transitions`; source-level: `source_outcomes` then `realized_transitions`).
2. Compute baseline: unconditional `p_up, p_down, p_flat`, `entropy_bits`, direction/core change rates — pooled across all symbols.
3. Compute conditioned: same metrics per group columns.
4. Compute `information_gain_bits = baseline_entropy − conditioned_entropy`.
5. Compute tail metrics: `tail_up_rate, tail_down_rate, avg_forward_max_return_pct, avg_path_range_pct, path_skew, returned_to_origin_rate`.
6. Compute recent-window (30-day) metrics for stability.
7. Assign `evidence_status`:
   - `usable_stable_information`: obs ≥ 100, symbols ≥ 20, info > 0, recent obs ≥ 30, recent info > 0
   - `usable_unstable_information`: obs ≥ 100, symbols ≥ 20, info > 0
   - `exploratory_information`: obs ≥ 50, symbols ≥ 12, info > 0
   - `insufficient_information`: otherwise
8. Assign `transition_status`: same thresholds on `transition_information_gain_bits`.
9. Compute `statistical_direction`: "up" / "down" / "flat" from max of conditioned_p_*.

**Output schema:** `POTENTIAL_EVIDENCE_SCHEMA` — ~45 columns including:
`evidence_level, outcome_horizon, background_regime, swing_core, decision_core,
decision_transition, source_family, source_state, risk_context,
baseline_p_up, baseline_p_down, baseline_p_flat,
conditioned_p_up, conditioned_p_down, conditioned_p_flat,
lift_up, lift_down, lift_flat,
baseline_entropy_bits, conditioned_entropy_bits, information_gain_bits,
transition_information_gain_bits,
tail_up_rate, tail_down_rate,
avg_forward_max_return_pct, avg_path_range_pct, path_skew,
returned_to_origin_rate, information_stability, transition_information_stability,
evidence_status, transition_status, statistical_direction, research_suggestion`.

Artifact: `potential-evidence-summary.csv`.

---

### `qooi.scanner.evidence.add_potential_parent_gain`

```text
qooi.scanner.evidence.add_potential_parent_gain(
    evidence: pl.DataFrame,
) -> pl.DataFrame
```

**Parent chain:**
- `market_background` → parent = null (root)
- `market_swing` → parent = `market_background` (join on `background_regime`)
- `market_decision` → parent = `market_swing` (join on `background_regime, swing_core`)
- `market_decision_source` → parent = `market_decision` (join on above + `decision_core, decision_transition`)
- `market_decision_source_risk` → parent = `market_decision_source` (join on above + `source_family, source_state`)

Adds columns:
- `parent_evidence_level`: parent level name or null
- `parent_information_gain_bits`: parent's info_gain
- `information_gain_over_parent`: `info_gain − parent_info_gain`
- `parent_transition_information_gain_bits`
- `transition_information_gain_over_parent`

Positive `information_gain_over_parent` = child level adds information beyond parent.

---

### `qooi.scanner.evidence.select_potential_evidence_level`

```text
qooi.scanner.evidence.select_potential_evidence_level(
    evidence: pl.DataFrame,
) -> pl.DataFrame
```

**Gate (two-tier: status rank + level rank):**

1. Compute `selection_status_rank`:
   - 3: `usable_stable_information` or `usable_stable_transition_information`
   - 2: `usable_unstable_information` or `usable_unstable_transition_information`
   - 1: `exploratory_information` or `exploratory_transition_information`
   - 0: otherwise

2. Compute `selection_level_rank`:
   - 0: `market_background` (excluded — always rank 0)
   - 1: `market_swing`
   - 2: `market_decision`
   - 3: `market_decision_source`
   - 4: `market_decision_source_risk`

3. Gate:
   ```
   passes = (selection_status_rank >= 2)
            & (selection_level_rank >= 1)
            & (information_gain_bits > 0)
            & (parent_evidence_level is null | information_gain_over_parent > 0
               | transition_information_gain_over_parent > 0)
   ```
   Plus absolute-source bypass:
   ```
   evidence_level contains "_source"
   & information_gain_bits > 0.1
   & symbol_count >= 5
   → passes regardless of parent improvement
   ```

4. Select best evidence: per `(outcome_horizon, statistical_direction)`, pick highest `selection_status_rank` + lowest `selection_level_rank`.

Adds column: `selected_evidence_level` (bool).

Artifact: `potential-evidence-selected.csv`.

---

### `qooi.scanner.evidence._potential_research_suggestion_expr`

```text
qooi.scanner.evidence._potential_research_suggestion_expr() -> pl.Expr
```

Internal Polars expression. Produces `research_suggestion` column:

| Condition | Label |
|---|---|
| selected & `returned_to_origin_rate ≥ 0.25` & `abs(path_skew) ≤ 0.10` | `chop_avoid` |
| selected & conditioned_direction_change > baseline & `avg_path_range_pct > 0` | `volatility_expansion_watch` |
| selected & conditioned_core_change > baseline | `rapid_trend_watch` |
| selected & returned_to_origin > baseline_returned_to_origin | `mean_reversion_watch` |
| else | `insufficient_evidence` |

---

## Internal helpers (private)

```text
qooi.scanner.evidence._potential_level_metrics(
    frame, evidence_level, group_columns, *, baseline=None,
) -> pl.DataFrame
    # Aggregates conditioned + baseline metrics per group.

qooi.scanner.evidence._outcome_baseline(frame) -> pl.DataFrame
    # Global unconditional baseline per outcome_horizon.

qooi.scanner.evidence._potential_evidence_identity_columns() -> tuple[str, ...]
    # evidence_level, outcome_horizon, + role columns.

qooi.scanner.evidence._potential_evidence_role_columns() -> tuple[str, ...]
    # background_regime, swing_core, decision_core, decision_transition,
    # source_family, source_state, risk_context.

qooi.scanner.evidence._binary_entropy_expr(probability_col) -> pl.Expr
    # −p·log₂(p) − (1−p)·log₂(1−p), safe for p ∈ (0,1).
```

---

## Consumers

```text
qooi.scanner.diagnostics.write_diagnostics(inputs)
  -> when config.evidence == "ladder":
       evidence = qooi.scanner.evidence.potential_evidence_frame(...)
       evidence = qooi.scanner.evidence.add_potential_parent_gain(evidence)
       evidence = qooi.scanner.evidence.select_potential_evidence_level(evidence)
       -> standardized evidence frame
       -> qooi.scanner.candidates.candidate_evidence_frame(...)
```

---

## Artifacts

```text
potential-evidence-summary.csv     # all evidence rows (all levels, all statuses)
potential-evidence-selected.csv    # gated subset (selected_evidence_level=True)
```

---

## Status

**Production.** All functions implemented. Known limitation: baseline entropy often remains near-random from coarse categorical features. This is a data property, not a code bug. The ladder path is retained as the production default while the tree path is developed.

## Boundary rules

- Must not import from `qooi.scanner.tailtree`.
- Must not import from `qooi.core`, `qooi.dynamic`.
- Future returns are outcome columns only — never features.
