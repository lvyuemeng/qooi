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
qooi.scanner.workflow.run(config_path: Path) -> Path

  1. qooi.scanner.workflow.load_config(config_path) -> PotentialConfig
  2. qooi.scanner.workflow.resolve_universe(config) -> PotentialUniverse
  3. qooi.scanner.workflow.load_bars(config, symbols) -> BarFetchResult
       # internally builds exchange.store.HistoryRefreshRequest
  4. qooi.scanner.transitions.compute_transition_insights(...)
  5. qooi.scanner.workflow.source_context_request(config, ...) -> SourceContextRequest
       # request.refresh_mode = config.refresh_mode; no nested source refresh override
  6. qooi.sources.context.load_source_context(request) -> SourceContextResult
  7. qooi.scanner.decisions.compute_source_states(...)
  8. qooi.scanner.decisions.scan_review_decisions(...)       # audit lens/artifact
  9. qooi.scanner.diagnostics.build_diagnostic_frames(inputs) -> DiagnosticFrames
 10. qooi.scanner.diagnostics.write_diagnostic_frames(frames, artifacts) -> None
 11. qooi.scanner.report.render_report(inputs, frames) -> str
```

No separate `ScannerWorkflowPlan` is required unless a future caller needs to
inspect a complete dry-run plan. The lean boundary is a small request object at
the package boundary, not another scanner-wide abstraction.

```text
PotentialConfig                      # root scanner config, owned by workflow
  -> HistoryRefreshRequest           # exchange-owned cache request using config.refresh_mode
  -> SourceContextRequest            # sources-owned demand request using config.refresh_mode
  -> evidence/transition/review calls # scanner-local section consumers
```

`workflow.py` passes named data products between modules. It must not grow a global materialized-data bag that every downstream consumer accepts. It also must not pass the whole root config into packages that only need a source/exchange request.

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
  → qooi.scanner.diagnostics.candidate_feasibility_frame(...)
  → qooi.scanner.diagnostics.data_health_frame(...)
  → candidate-inspection.csv + candidate-rank.csv + candidate-feasibility.csv + report
```

`qooi.scanner.__init__` owns scanner-local contracts/protocols and shared expression
helpers. There are no `qooi.scanner.contracts`, `qooi.scanner.evidence`, or
`qooi.scanner.candidates` compatibility modules in the resolved graph.

---

## Refresh and source context boundary

One scan refresh field feeds both materialized input families:

```text
PotentialConfig.refresh_mode: "incremental" | "cache_only" | "force"
  -> qooi.exchange.store.HistoryRefreshRequest(refresh, incremental, cache_only)
  -> qooi.sources.context.SourceContextRequest(refresh_mode)
```

Forbidden target APIs:

```text
SourceConfig.refresh_mode
SourceRefreshMode
resolve_source_refresh_mode(...)
```

If bars and sources need independent cadence, add explicit workflow commands/config
profiles rather than another nested `refresh_mode` field.

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

Target availability columns for the capability-aware source fix:

```text
symbol
source_family
rows
latest_timestamp
latest_age_hours
freshness_threshold_hours
provider_cap_rows
provider_cap_lookback_days
coverage_target_pct
coverage_capability_pct
frame_fresh_int
frame_stale_int
frame_missing_int
provider_bounded_int
optional_absent_int
fetch_failed_frame_fresh_int
usable_int
required_for_review_int
required_for_evidence_int
rank_penalty_weight
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

Capability summary artifacts:

```text
diagnostics/source-capability.csv
  key: source_family, raw_source, period
  columns:
    scope
    max_rows
    max_lookback_days
    earliest_provider_ms
    supports_latest_refresh_int
    supports_backfill_int
    required_for_review_int
    required_for_evidence_int
    optional_int
    rank_penalty_weight

diagnostics/source-availability.csv
  key: symbol, source_family
  columns:
    rows
    latest_age_hours
    coverage_target_pct
    coverage_capability_pct
    frame_fresh_int
    frame_stale_int
    frame_missing_int
    provider_bounded_int
    optional_absent_int
    usable_int
    source_penalty_component
```

These artifacts preserve three separate questions:

```text
Can evidence train/evaluate on the available cross-coin history?
Which source gaps are true missing/stale rows versus provider-bounded capability limits?
Can this current symbol row be reviewed with enough local history and fresh required source context?
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
candidate-inspection.csv  # latest rows matched to evidence/leaf metrics
candidate-rank.csv        # symbol × selected direction ranked evidence matches
candidate-feasibility.csv # one best ranked row per symbol joined to review feasibility
```

`candidate-inspection.csv` is the diagnostic surface. `candidate-rank.csv` is rank detail and may contain multiple rows per symbol. `candidate-feasibility.csv` is the report-facing candidate-selection surface.

Ranking uses numeric columns that are present:

- ladder: information gain, stability, support, path quality, data quality;
- tailtree: tail lift, tail stability, support, path quality, data quality.

Candidate selection projection currently belongs to `diagnostics.py`, not `report.py`:

```text
qooi.scanner.diagnostics.candidate_feasibility_frame(
    candidate_rank: pl.DataFrame,
    watchlist_feasibility: pl.DataFrame,
) -> pl.DataFrame
```

Split to `selection.py` only if it removes real module pressure after CSV read-back and default audit rendering are removed; do not create a forwarding wrapper.

Output contract:

```text
key: symbol
selection: highest rank_score per symbol
columns:
  symbol: String
  watchlist_feasibility: String
  rank_score: Float64
  rank_tier: String
  source_penalty_score: Float64
  required_missing_source_count: Int64
  required_stale_source_count: Int64
  provider_bounded_source_count: Int64
  optional_absent_source_count: Int64
  min_history_coverage_pct: Float64
  min_source_capability_coverage_pct: Float64
  tree_direction: String
  matched_evidence_level: String
  tail_lift: Float64
  gpd_shape_xi: Float64
  N_tail_exceedances: Int64
  source_status: String
  history_status: String
  candidate_reason: String
```

This projection is semantic, not presentational. It may derive `rank_tier`, select the best row per symbol, and derive stable blocker/reason codes. It must not emit display-formatted strings for numeric values.

Promoted-rank scoring should be explicit, capability-aware, and cost-aware:

```text
source_gate = required_missing_source_count == 0
              and required_stale_source_count <= stale_budget
              and coverage_capability_pct_min >= capability_threshold

promoted = selected_evidence_level
           and source_gate
           and bar recency gate
           and liquidity sanity gate

source_penalty = missing_required_penalty
               + stale_required_penalty
               + provider_bounded_penalty
               + optional_absent_penalty

provider_bounded_penalty = 0 or low when frame_fresh_int=1 and coverage_capability_pct is high
optional_absent_penalty  = 0 for messages until a real provider is enabled

cost_adjusted_score = raw_tail_score - source_penalty - cost_penalty
cost_penalty        = estimated_roundtrip_cost_bps / expected_edge_bps
```

Manual slippage thresholds are allowed only as extreme sanity guards. Normal review should use symbol-relative, size-aware columns such as `spread_percentile_30d`, `depth_percentile_30d`, and `estimated_slippage_bps_for_size`.

---

## Selection and health projections

### Candidate-selection projection

Canonical candidate-selection projection:

```text
qooi.scanner.diagnostics.candidate_feasibility_frame(
    candidate_rank: pl.DataFrame,
    watchlist_feasibility: pl.DataFrame,
) -> pl.DataFrame
```

This function may later move to `qooi.scanner.selection.candidate_selection_frame(...)` only if that removes real module pressure without becoming a wrapper alias.

Schema:

```text
CANDIDATE_SELECTION_SCHEMA
  symbol: String
  feasibility: String
  rank_score: Float64
  rank_tier: String
  source_penalty_score: Float64
  required_missing_source_count: Int64
  required_stale_source_count: Int64
  provider_bounded_source_count: Int64
  optional_absent_source_count: Int64
  min_history_coverage_pct: Float64
  min_source_capability_coverage_pct: Float64
  tree_direction: String
  matched_evidence_level: String
  tail_lift: Float64
  gpd_shape_xi: Float64
  n_tail_exceedances: Int64
  source_status: String
  history_status: String
  candidate_reason: String
```

Rules:

```text
key: symbol
selection: highest rank_score per symbol after deterministic tie sort
candidate_reason: semantic blocker code, not display prose
no Markdown strings or report formatting
```

### Data-health projection

Aggregate report-health projection:

```text
qooi.scanner.diagnostics.data_health_frame(
    history_feasibility: pl.DataFrame,
    source_availability: pl.DataFrame,
    candidate_selection: pl.DataFrame,
) -> pl.DataFrame
```

This function can stay in `diagnostics.py` while it only feeds diagnostics/report. Split to `health.py` only if it gains independent callers or materially reduces `diagnostics.py` after CSV read-back is removed.

Schema:

```text
DATA_HEALTH_SCHEMA
  scope: String
  row_count: Int64
  required_missing_source_count: Int64
  required_stale_source_count: Int64
  provider_bounded_source_count: Int64
  optional_absent_source_count: Int64
  reviewable_count: Int64
  limited_count: Int64
```

Rules:

```text
source/history/candidate health are aggregate counts
no symbol-level candidate rows in data-health report section
no candidate ranking or source collection
```

## Diagnostics build/write and report render

```text
qooi.scanner.diagnostics.build_diagnostic_frames(inputs: ReportInputs) -> DiagnosticFrames
qooi.scanner.diagnostics.write_diagnostic_frames(
    frames: DiagnosticFrames,
    artifacts: PotentialArtifacts,
) -> None
qooi.scanner.report.render_report(
    inputs: ReportInputs,
    frames: DiagnosticFrames,
) -> str
```

Current implementation has the public build/write boundary, while report rendering still reads some CSV artifacts during the same run:

```text
build_diagnostic_frames(inputs) -> DiagnosticFrames
write_diagnostic_frames(frames, artifacts) -> None
render_report(inputs) -> str  # still reads some CSV artifacts during same run
```

Target API removes same-run CSV read-back and separates expensive compute from cheap IO.
Measured cache-only hotpath:

```text
qooi.scanner.history.realized_transition_frame(...)        ~4.9s
qooi.scanner.history.kline_path_history_frame(...)         ~1.8s
qooi.scanner.diagnostics._run_pipeline(...)                ~0.7s
qooi.scanner.frames.potential_observation_frame(...)       ~0.5s
qooi.scanner.diagnostics.write_diagnostic_frames(...)      ~0.1s
qooi.scanner.report.render_report(...)                     ~0.1s
```

Optimization order:

```text
1. realized_transition_frame
2. kline_path_history_frame
3. tailtree/evidence pipeline
4. potential_observation_frame
5. in-memory report frames to remove CSV type recovery
```

Do not optimize Markdown table formatting before these frame builders.

Report rendering graph:

```text
DiagnosticFrames.candidate_selection
  → qooi.scanner.report.CandidateSelectionSection.render(frame) -> str

DiagnosticFrames.data_health
  → qooi.scanner.report.DataHealthSection.render(frame) -> str

DiagnosticFrames.path_evidence
  → report_sections_for(evidence)
  → ReportSection.render(inputs, frames)
```

Target frame contracts:

```text
qooi.scanner.diagnostics.CANDIDATE_SELECTION_SCHEMA
qooi.scanner.diagnostics.DATA_HEALTH_SCHEMA
```

`report.py` should not read `diagnostics/*.csv` during the same run. CSV artifacts are an output boundary for users/tests, not an internal type transport.

`report.py` is a renderer, not a schema inference engine. Forbidden report code patterns:

```text
dict[str, object] row contracts
row.get(...) as normal data access
getattr(...) probing
Any/object plus float(str(value)) recovery
business-rule joins between rank and feasibility
```

Path-specific report sections are composed before rendering, not by branching inside every section. The renderer formats typed frame columns; it does not decide source, history, or candidate semantics. Default report excludes `Decision Rule Audit`; audit rows remain diagnostics unless an explicit appendix mode is added.

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
