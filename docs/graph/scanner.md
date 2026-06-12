# Scanner Module Graph

Shared scanner skeleton: CLI orchestration, package-root scanner contracts, known-at-close feature/observation products, one evidence dispatch, candidate/rank/report. Current path internals live in `frames`, `ladder`, `tailrun`, and `rank`; removed compatibility modules such as `evidence.py`, `candidates.py`, and `contracts.py` are not public surfaces.

Design doc: `docs/architecture/scanner.md`.

---

## CLI entry

```text
scripts/potential_scan.py
  main()
    → qooi.scanner.workflow.run(Path(args.config))
```

---

## Orchestrator: `qooi.scanner.workflow`

```text
qooi.scanner.workflow.run(config_path: Path) -> None

  1. qooi.scanner.workflow.load_config(config_path) -> PotentialConfig
  2. qooi.scanner.workflow.resolve_universe(config) -> PotentialUniverse
  3. qooi.scanner.workflow.load_bars(config, symbols) -> BarFetchResult
  4. qooi.scanner.transitions.compute_transition_insights(...)
  5. qooi.sources.context.load_source_context(...)
  6. qooi.scanner.decisions.compute_source_states(...)
  7. qooi.scanner.decisions.scan_review_decisions(...)       # decision lens
  8. qooi.scanner.diagnostics.write_diagnostics(inputs)       # evidence lens
  9. qooi.scanner.report.render_report(inputs) -> str
```

`workflow.py` passes named data products between modules. It must not grow a global materialized-data bag that every downstream consumer accepts.

---

## Shared data pipeflow

```text
bars/state_frames/source_frames
  → qooi.scanner.features.extract_continuous_features(...)
  → continuous_features

kline_history/source_events/continuous_features
  → qooi.scanner.frames.potential_observation_frame(...)
  → observations

observations/source_outcomes/realized_transitions
  → qooi.scanner.frames.potential_outcome_frame(...)
  → outcomes

observations/outcomes
  → qooi.scanner.diagnostics._run_pipeline(...)
       ├─ evidence="ladder"   → qooi.scanner.ladder.evaluate(...)
       └─ evidence="tailtree" → qooi.scanner.tailrun.run(...)

result.evidence/latest_observations
  → qooi.scanner.rank.candidate_evidence_frame(...)
  → qooi.scanner.rank.rank_candidate_evidence(...)
  → candidate-inspection.csv + candidate-rank.csv + report
```

`qooi.scanner.__init__` owns scanner-local contracts/protocols and shared expression
helpers. There are no `qooi.scanner.contracts`, `qooi.scanner.evidence`, or
`qooi.scanner.candidates` compatibility modules in the resolved graph.

---

## Source context + feasibility inputs

```text
qooi.sources.context.load_source_context(...)
  -> SourceContextResult(
       manifest,
       frames,
       availability,
     )
```

`availability` is the scanner's source-feasibility input. Scanner modules consume numeric availability fields; they do not reinterpret provider manifests directly.

Current implemented availability columns:

```text
source_family
symbol
rows
latest_timestamp
status
warning
```

Target availability columns for the source-availability fix:

```text
symbol
source_family
rows
latest_timestamp
latest_age_hours
freshness_threshold_hours
frame_fresh_int
frame_missing_int
usable_int
latest_fetch_status
latest_fetch_warning
latest_fetch_status_code
latest_fetch_provider_code
```

History feasibility artifacts:

```text
diagnostics/history-feasibility.csv
  key: symbol, bar
  current columns:
    target_rows
    actual_rows
    coverage_pct
    range_start
    range_end
    newest_age_hours
    gap_count
    duplicate_timestamps
    refreshed
    feasibility_status
    feasibility_reason
    notes
  target columns after review-window split:
    observed_rows
    target_rows
    history_target_coverage_pct
    review_window_rows
    review_window_coverage_pct
    fetch_limited_int
    history_start_limited_int
    reviewable_history_int
    fetch_stop
    fetch_status_code
    fetch_provider_code
```

Source freshness artifacts:

```text
diagnostics/source-freshness.csv
  key: symbol, source_family
  columns from SourceAvailability
```

These artifacts preserve two separate questions:

```text
Can evidence train/evaluate on the available cross-coin history?
Can this current symbol row be reviewed with enough local history and fresh source context?
```

---

## Continuous features: `qooi.scanner.features`

```text
qooi.scanner.features.extract_continuous_features(
    bars: dict[tuple[str, str], pl.DataFrame],
    state_frames: dict[tuple[str, str], pl.DataFrame],
    source_frames: dict[str, pl.DataFrame],
    *,
    decision_timeframe: str,
) -> pl.DataFrame
```

Output key:

```text
(symbol, timestamp)
```

Required invariant:

```text
one row per (symbol, timestamp)
```

Source reducer rule:

```text
GOOD: group_by(["symbol", "timestamp"]), join(on=["symbol", "timestamp"])
BAD:  group_by("timestamp"), join(on="timestamp")
```

Important columns:

| Column | Type | Source |
|---|---|---|
| `symbol` | String | bars/source frame |
| `timestamp` | Int64 | decision/source timestamp |
| `atr_percentile` | Float64 | kline state frame |
| `range_width_atr` | Float64 | kline state frame |
| `return_1bar` | Float64 | bar close returns |
| `return_4bar` | Float64 | bar close returns |
| `return_24bar` | Float64 | bar close returns |
| `vol_anomaly` | Float64 | bar volume / rolling mean |
| `close_to_range_high_ratio` | Float64 | close position in state range |
| `imbalance_value` | Float64 | order-book imbalance |
| `spread_bps` | Float64 | order-book spread |
| `buy_sell_ratio` | Float64 | trade notional imbalance |
| `funding_rate` | Float64 | funding source |
| `oi_delta` | Float64 | open-interest change |
| `taker_buy_sell_ratio` | Float64 | taker volume source |
| `long_short_ratio` | Float64 | long/short account ratio |
| `book_age_ms` | Int64 | order-book source age at decision close |
| `trade_age_ms` | Int64 | trade source age at decision close |
| `funding_age_ms` | Int64 | funding source age at decision close |
| `oi_age_ms` | Int64 | open-interest source age at decision close |
| `taker_age_ms` | Int64 | taker source age at decision close |
| `lsr_age_ms` | Int64 | long/short ratio source age at decision close |

Planned cost/freshness extensions for promoted candidate review:

| Column | Type | Purpose |
|---|---|---|
| `source_age_hours` | Float64 | numeric freshness for sorting/gating |
| `bar_age_bars` | Float64 | current bar recency check |
| `spread_percentile_30d` | Float64 | symbol-relative spread condition |
| `depth_percentile_30d` | Float64 | symbol-relative book depth condition |
| `estimated_slippage_bps_for_size` | Float64 | size-aware cost estimate |
| `expected_edge_bps` | Float64 | evidence-implied move budget |
| `cost_adjusted_score` | Float64 | rank score after liquidity/cost penalty |

---

## Observation + outcome: `qooi.scanner.frames`

```text
qooi.scanner.frames.potential_observation_frame(
    kline_history: pl.DataFrame,
    source_events: pl.DataFrame,
    continuous_features: pl.DataFrame | None,
    *,
    decision_timeframe: str,
    max_source_staleness_hours: int,  # from PotentialConfig.source.max_staleness_hours
) -> pl.DataFrame
```

Observation contract:

```text
key: symbol, decision_bar_close_ms, source_family
continuous join: (symbol, decision_bar_close_ms) = (symbol, timestamp)
```

Continuous features are joined on every path, including no-source/stale-source paths.

```text
qooi.scanner.frames.potential_outcome_frame(
    observations: pl.DataFrame,
    source_outcomes: pl.DataFrame,
    realized_transitions: pl.DataFrame,
    *,
    return_threshold_pct: float,
) -> pl.DataFrame
```

Outcome contract:

```text
key: symbol, decision_bar_close_ms, outcome_horizon
contains future-return/path diagnostics only
```

Implemented outcome columns:

| Column | Meaning |
|---|---|
| `forward_return_pct` | close-to-close return at `outcome_horizon` |
| `forward_max_return_pct` | maximum high excursion inside horizon |
| `forward_min_return_pct` | minimum low excursion inside horizon |
| `path_range_pct` | high/low excursion width |
| `tail_up`, `tail_down` | excursion labels after thresholding |

Planned path-shape columns:

| Column | Meaning |
|---|---|
| `time_to_max_bar` | first bar offset where forward max was reached |
| `time_to_min_bar` | first bar offset where forward min was reached |
| `close_retention_ratio` | terminal close return divided by max favorable excursion |
| `post_peak_drawdown_pct` | amount unwound after favorable peak |
| `path_efficiency` | terminal move divided by path range |
| `entry_delay_bars` | realistic delay between signal close and executable entry |

These planned columns distinguish continuation from burst-then-fade while preserving the current excursion label.

---

## Evidence dispatch: `qooi.scanner.diagnostics`

```text
qooi.scanner.diagnostics._run_pipeline(
    observations,
    source_outcomes,
    realized_transitions,
    inputs,
) -> LadderResult | TailtreeResult
```

One dispatch point:

```text
if inputs.config.evidence.kind == "tailtree":
    return _run_tailtree_pipeline(...)
return _run_ladder_pipeline(...)
```

Implemented tailtree lifecycle dispatch is nested inside the tailtree path, not spread across consumers:

```text
if inputs.config.evidence.tailtree.lifecycle == "load_predict":
    result = _load_tail_tree_evidence(...)
else:
    result = _build_tail_tree_evidence(...)
```

Downstream code consumes concrete result fields:

```text
result.evidence
result.candidates
result.ranked
```

No variant tuple, no `tree_up: None`, no `if config.evidence` in report sections.

---

## Package contracts: `qooi.scanner`

Scanner-local contracts and small shared expressions are exported from the package root:

```text
qooi.scanner.PotentialScanConfig
qooi.scanner.PotentialUniverse
qooi.scanner.PotentialArtifacts
qooi.scanner.BarFetchResult
qooi.scanner.SourceStateRow
qooi.scanner.TransitionPattern
qooi.scanner.TransitionInsight
qooi.scanner.TransitionAnalysis
qooi.scanner.SymbolStateBundle
qooi.scanner.ScanDecision
qooi.scanner.ReportInputs
qooi.scanner.missing_state(...)
qooi.scanner.context_symbols(...)
qooi.scanner.pct_change_expr(...)
qooi.scanner.outcome_bucket_expr(...)
qooi.scanner.entropy_expr(...)
```

Import rule:

```text
GOOD: from qooi.scanner import ReportInputs, SourceStateRow
BAD:  from qooi.scanner.contracts import ReportInputs
```

---

## Result contracts

```text
qooi.scanner.ladder.LadderResult
├── evidence: pl.DataFrame
├── candidates: pl.DataFrame
└── ranked: pl.DataFrame

qooi.scanner.tailrun.TailtreeResult
├── evidence: pl.DataFrame
├── candidates: pl.DataFrame
├── ranked: pl.DataFrame
├── tree_up: TailTreeModel
└── tree_down: TailTreeModel

qooi.scanner.tailrun.TailtreeEvidenceResult  # lifecycle evidence/model result
```

---

## Candidate inspection + ranking graph

Implementation module:

```text
qooi.scanner.rank.candidate_evidence_frame(
    latest_observations: pl.DataFrame,
    evidence: pl.DataFrame,
    *,
    tree_up: TailTreeModel | None = None,
    tree_down: TailTreeModel | None = None,
) -> pl.DataFrame

qooi.scanner.rank.rank_candidate_evidence(candidates: pl.DataFrame) -> pl.DataFrame
qooi.scanner.rank.rank_candidates(candidates: pl.DataFrame) -> pl.DataFrame
```

The only caller that supplies tree models is the tailtree pipeline. Candidate internals prefer path-specific helpers rather than spreading config checks. Candidate inspection and ranking ownership is `rank.py`; there is no `candidates.py` compatibility shim.

Current output:

```text
candidate-inspection.csv # latest rows matched to evidence/leaf metrics
candidate-rank.csv       # ranked current candidates
```

`candidate-inspection.csv` is the diagnostic surface; `candidate-rank.csv` is the promoted review surface. Do not add promotion-only fields to inspection rows; add those to `rank.py`/`candidate-rank.csv`.

Ranking uses numeric columns that are present:

- ladder: information gain, stability, support, path quality, data quality;
- tailtree: tail lift, tail stability, support, path quality, data quality.

Planned promoted-rank scoring should be explicit and cost-aware:

```text
promoted = selected_evidence_level
           and freshness gate
           and bar recency gate
           and liquidity sanity gate

cost_adjusted_score = raw_tail_score - cost_penalty
cost_penalty        = estimated_roundtrip_cost_bps / expected_edge_bps
```

Manual slippage thresholds are allowed only as extreme sanity guards. Normal review should use symbol-relative, size-aware columns such as `spread_percentile_30d`, `depth_percentile_30d`, and `estimated_slippage_bps_for_size`.

---

## Diagnostics and report

```text
qooi.scanner.diagnostics.write_diagnostics(inputs: ReportInputs) -> DiagnosticFrames
qooi.scanner.report.render_report(inputs: ReportInputs) -> str
```

Diagnostics writes path-specific artifacts after the one dispatch. Report renders the frames it receives; path-specific sections are composed before rendering, not by branching inside every section.

---

## Cross-package import boundaries

Observed package dependency shape:

```text
scanner  → core, exchange, sources, strategies
research → core, dynamic, exchange, scanner, strategies
sources  → exchange
exchange → core, sources
```

Resolved scanner boundary:

| Edge | Status | Rule |
|---|---|---|
| `scanner.workflow → exchange` | allowed | only workflow fetches/cache/discovers; evidence/model modules consume DataFrames |
| `scanner → sources.context` | allowed | scanner consumes source context; `sources` owns acquisition/schema |
| `scanner → strategies` | overlap to remove | scanner classifier vocabulary should be scanner-owned |
| `research → scanner` | allowed at artifact boundary | research should inspect scanner artifacts, not internal classifier helpers |
| `scanner → executor/basket/live` | forbidden | scanner emits diagnostics/review candidates only |

Target direction for strategy integration:

```text
scanner outputs → strategies → basket/executor
```

Forbidden direction:

```text
strategies semantics → scanner classifiers
```

---

## Removed/stale API references

These must not appear in current scanner docs/code contracts:

```text
use_tail_tree                 # replaced by [potential.evidence] kind = "ladder" | "tailtree"
leaf_path / leaf_paths        # removed; leaf_id + numeric stats are enough
to_pandas training path       # optional pyarrow/pandas path is not required
fobj custom-objective arg     # LightGBM 4 custom objective is params["objective"]
```
