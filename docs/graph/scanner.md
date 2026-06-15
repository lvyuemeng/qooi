# Scanner Module Graph

Shared scanner skeleton: CLI orchestration, package-root scanner contracts, product-shaped data pipe, evidence dispatch, rank/feasibility/report outputs. The active graph is `state -> outcome -> evidence -> rank -> feasibility -> diagnostics/report`. State owns former classifier/feature internals; outcome owns former source-event/history internals. Removed compatibility/transitional modules such as `classifiers.py`, `features.py`, `source_events.py`, `history.py`, `decisions.py`, `frames.py`, `evidence.py`, `candidates.py`, and `contracts.py` are not public surfaces.

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

Implemented flow with injected profiling context:

```text
qooi.scanner.workflow.run(config_path: Path) -> Path

  1. qooi.scanner.workflow.load_config(config_path) -> PotentialConfig
  2. qooi.profiling.ProfileContext.from_config(config.profile, artifacts.profile_dir)
  3. profile.stage("scanner", "workflow", "resolve_universe")
       -> qooi.scanner.workflow.resolve_universe(config) -> PotentialUniverse
  4. profile.stage("scanner", "workflow", "load_bars")
       -> qooi.scanner.workflow.load_bars(config, symbols) -> BarFetchResult
       # internally builds exchange.store.HistoryRefreshRequest
  5. qooi.scanner.transitions.compute_transition_insights(...)
  6. qooi.scanner.workflow.source_context_request(config, ...) -> SourceContextRequest
       # request.refresh_mode = config.refresh_mode; no nested source refresh override
  7. profile.stage("scanner", "workflow", "load_source_context")
       -> qooi.sources.context.load_source_context(request) -> SourceContextResult
  8. qooi.scanner.workflow.compute_source_states(...)
  9. qooi.scanner.workflow.scan_review_decisions(...)        # audit lens/artifact
 10. qooi.scanner.diagnostics.write_diagnostics(inputs, profile) -> DiagnosticFrames
     # internally builds frames, writes CSV diagnostics, and writes state frames
 11. qooi.scanner.report.render_report(inputs, frames) -> str
     # common report sections consume in-memory DiagnosticFrames
 12. profile.write()
```

No separate `ScannerWorkflowPlan` is required unless a future caller needs to
inspect a complete dry-run plan. The lean boundary is a small request object at
the package boundary, not another scanner-wide abstraction.

```text
PotentialConfig                      # single scanner config entry, not runtime god-object
  -> HistoryRefreshRequest           # exchange-owned cache request using config.refresh_mode
  -> SourceContextRequest            # sources-owned demand request using config.refresh_mode
  -> ProfileContext                  # qooi.profiling-owned diagnostics context
  -> evidence/transition/review calls # scanner-local section consumers
```

`workflow.py` passes named data products between modules. It must not grow a global materialized-data bag that every downstream consumer accepts. It also must not pass the whole root config into packages that only need a source/exchange request.

---

## Shared data pipeflow

The urgent API graph is the data pipe, not another orchestration module. Do not add
`pipeline.py` unless a typed product object exists and has multiple real consumers. The
current implementation still runs much of this through `diagnostics.build_diagnostic_frames`,
but the target ownership is:

```text
workflow.run(config_path)
  → load config/universe/bars/source context/decisions/transitions
  → ReportInputs
  → diagnostics.write_diagnostics(inputs, profile) -> DiagnosticFrames
  → report.render_report(inputs, frames)
```

Inside the scanner product pipe:

```text
state product
  current calls:
    qooi.scanner.state.KlineClassifier.classify(...)
    qooi.scanner.state.extract_continuous_features(...)
    qooi.scanner.state.potential_observation_frame(...)
  target owner:
    qooi.scanner.state
  output:
    potential_observations

outcome product
  current calls:
    qooi.scanner.outcome.kline_path_history_frame(...)
    qooi.scanner.outcome.realized_transition_frame(...)
    qooi.scanner.outcome.source_events_frame(...)
    qooi.scanner.outcome.source_outcomes_frame(...)
    qooi.scanner.outcome.source_timeliness_frame(...)
    qooi.scanner.outcome.source_state_predictability_frame(...)
    qooi.scanner.outcome.potential_outcome_frame(...)
  target owner:
    qooi.scanner.outcome
  outputs:
    kline_path_history, realized_transitions, source_events, source_outcomes, potential_outcomes

evidence product
  current dispatch:
    evidence="ladder"
      → qooi.scanner.ladder.potential_evidence_frame(...)
    evidence="tailtree"
      → qooi.scanner.tailrun.run(...)
  outputs:
    potential_evidence plus tailtree model/evidence artifacts
  tailtree model-selection output:
    tailtree-selection-efficiency.csv is the canonical objective/HPO feedback artifact;
    tailtree-run-summary.csv is structural run health only

rank product
  current calls:
    qooi.scanner.rank.candidate_evidence_frame(...)
    qooi.scanner.rank.rank_candidate_evidence(...)
    qooi.scanner.rank.candidate_horizon_consistency_frame(...)
  outputs:
    candidate-inspection.csv, candidate-rank.csv, candidate-horizon-consistency.csv
  candidate-rank contract:
    rank_score is evidence-inspection score
    profit_proxy_score is current tail_utility proxy minus scanner data/source penalties
    promotion_score is the candidate-selection score consumed by feasibility/report
  invariant:
    candidate evidence/rank pipe carries `outcome_horizon`; consumers do not posterior-check it

feasibility product
  current transitional owner:
    qooi.scanner.feasibility.candidate_feasibility_frame(...)
    qooi.scanner.feasibility.join_candidate_source_constraints(...)
  target owner:
    qooi.scanner.feasibility
  outputs:
    candidate-feasibility.csv plus source/history/watchlist feasibility projections
  candidate-feasibility contract:
    one best row per symbol by promotion_score, then source/data penalties
    carries promotion_score, profit_proxy_score, tail_utility_mean, and normalized profit_proxy columns
    report renderer consumes this prepared frame only; no rank/HPO CSV read-back

diagnostics/report outputs
  current calls:
    qooi.scanner.diagnostics.write_diagnostic_frames(...)
    qooi.scanner.report.render_report(...)
  target:
    diagnostics writes artifacts; report renders prepared frames only
```

`qooi.scanner.__init__` owns scanner-local contracts/protocols and shared expression
helpers. There are no `qooi.scanner.contracts`, `qooi.scanner.evidence`, or
`qooi.scanner.candidates` compatibility modules in the resolved graph.

Current transitional debt:

```text
qooi.scanner.feasibility          # source/history/candidate feasibility product
qooi.scanner.diagnostics.py      # still builds too much of the data pipe
qooi.scanner.report.py           # still owns some typed projection/render helper logic
```

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

## Continuous features: `qooi.scanner.state`

```text
qooi.scanner.state.extract_continuous_features(
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

## Observation + outcome products: `qooi.scanner.state` / `qooi.scanner.outcome`

```text
qooi.scanner.state.potential_observation_frame(
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
qooi.scanner.outcome.potential_outcome_frame(
    observations: pl.DataFrame,
    source_outcomes: pl.DataFrame,
    realized_transitions: pl.DataFrame,
    *,
    return_threshold_pct: float,
) -> pl.DataFrame
```

Outcome contract:

```text
key: symbol, decision_timeframe, decision_bar_close_ms, outcome_horizon
contains future-return/path diagnostics only
```

Market outcome source:

```text
qooi.scanner.outcome.kline_path_history_frame(config, state_frames, bar_frames)
  -> known-at-close categorical state rows + raw close/high/low at the same bar close

qooi.scanner.outcome.realized_transition_frame(kline_history, horizons)
  -> terminal categorical transition columns
  -> forward_return_pct / forward_min_return_pct / forward_max_return_pct / path_range_pct
  -> time_to_max_bar / time_to_min_bar / close_retention_ratio / path_efficiency
```

`potential_outcome_frame(...)` preserves realized-transition excursion columns for market
rows and joins source-event excursion columns for source rows. It does not recompute
future labels.

Implemented outcome columns:

| Column | Meaning |
|---|---|
| `forward_return_pct` | close-to-close return at `outcome_horizon` |
| `forward_max_return_pct` | maximum high excursion inside horizon |
| `forward_min_return_pct` | minimum low excursion inside horizon |
| `path_range_pct` | high/low excursion width |
| `time_to_max_bar` | first forward bar offset where max high excursion was reached |
| `time_to_min_bar` | first forward bar offset where min low excursion was reached |
| `close_retention_ratio` | terminal return divided by same-direction favorable excursion |
| `post_max_drawdown_pct` | max-up excursion unwound by terminal close |
| `post_min_rebound_pct` | min-down excursion rebounded by terminal close |
| `path_efficiency` | absolute terminal return divided by path range |
| `tail_up`, `tail_down` | excursion labels after thresholding |

Path-shape columns classify fixed-horizon quality without replacing excursion labels:

| Column | Meaning |
|---|---|
| touch | threshold crossed intrahorizon |
| continuation | favorable excursion retained into terminal close |
| exhaustion | favorable excursion mostly unwound by terminal close |
| two-sided volatility | both max and min excursions are large relative to terminal move |

Multi-horizon is represented by additional `outcome_horizon` rows. Tailtree trains
and writes horizon-suffixed artifacts per configured `outcome_horizon` value.

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

Tailtree train integrity artifacts:

```text
qooi.scanner.tailrun._build_tail_tree_evidence(...)
  → qooi.scanner.outcome.potential_outcome_frame(...)
  → qooi.scanner.tailtree.label_tail_exceedances(...)
  → qooi.scanner.tailrun._tailtree_run_summary_frame(...)
  → qooi.scanner.tailrun._write_tailtree_artifacts(...)

outputs:
  diagnostics/tailtree-run-summary.csv
  model_dir/model_tag/tailtree-run-summary.csv
  model_dir/model_tag/tailtree-artifact.json
  model_dir/model_tag/tail-tree-{up,down}.json              # only current trained dirs
  model_dir/model_tag/potential-leaf-evidence-{up,down}.csv # only current evidence dirs
```

Run-complete artifact rule:

```text
before writing a train run, tailrun removes stale direction artifacts for the same tag;
current metadata, summary, tree files, and evidence CSVs must describe the same run.
```

Tailtree summary schema target:

```text
summary_scope: "run" | "up" | "down"
direction: "" | "up" | "down"
observation_row_count: int
outcome_row_count: int
source_event_row_count: int
source_outcome_row_count: int
realized_transition_row_count: int
feature_count: int
categorical_feature_count: int
continuous_feature_count: int
forward_return_nonnull_count: int
forward_min_return_nonnull_count: int
forward_max_return_nonnull_count: int
path_range_nonnull_count: int
threshold_pct: float
tail_count: int
tail_rate: float
train_observation_count: int
train_exceedance_count: int
min_exceedance_required: int
trainable_flag: int
trained_tree_count: int
selected_leaf_count: int
written_model_file_count: int
written_evidence_file_count: int
removed_stale_file_count: int
```

HPO target edge is intentionally absent from implemented flow until walk-forward validation has nonzero labels and validation tail counts. Random-split HPO is not a scanner API.

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
```

The only caller that supplies tree models is the tailtree pipeline. Candidate internals prefer path-specific helpers rather than spreading config checks. Candidate inspection and ranking ownership is `rank.py`; there is no `candidates.py` compatibility shim.

Current output:

```text
candidate-inspection.csv  # latest rows matched to evidence/leaf metrics; one row per symbol × horizon × direction
candidate-rank.csv        # ranked candidate-horizon-direction evidence rows
candidate-feasibility.csv # best ranked horizon row per symbol joined to review feasibility
```

`candidate-inspection.csv` is the diagnostic surface. `candidate-rank.csv` is rank detail and may contain multiple rows per symbol. `candidate-feasibility.csv` is the report-facing candidate-selection surface.

Ranking uses numeric columns that are present:

- ladder: information gain, stability, support, path quality, data quality;
- tailtree: tail lift, tail stability, support, path quality, data quality.

Candidate feasibility projection belongs to `qooi.scanner.feasibility`, not `report.py`:

```text
qooi.scanner.feasibility.candidate_feasibility_frame(
    candidate_rank: pl.DataFrame,
    watchlist_feasibility: pl.DataFrame,
) -> pl.DataFrame
```

Keep this product in `feasibility.py`; do not create `selection.py` as a forwarding wrapper.

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
qooi.scanner.feasibility.candidate_feasibility_frame(
    candidate_rank: pl.DataFrame,
    watchlist_feasibility: pl.DataFrame,
) -> pl.DataFrame
```

This function lives in `qooi.scanner.feasibility`. Do not reintroduce the removed plural compatibility package `qooi.scanner.candidates`, and do not create a candidate-specific package for feasibility.

Schema:

```text
CANDIDATE_FEASIBILITY_SCHEMA
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

Current implementation has the public build/write boundary. Common report sections consume
`DiagnosticFrames` in memory; path-specific tailtree artifact sections still inspect tree
JSON/CSV artifacts written by `tailrun`.

```text
build_diagnostic_frames(inputs) -> DiagnosticFrames
write_diagnostic_frames(frames, artifacts) -> None
write_diagnostics(inputs, profile) -> DiagnosticFrames
render_report(inputs, frames) -> str
```

Target API removes same-run CSV read-back and separates expensive compute from cheap IO.
Measured cache-only hotpath:

```text
qooi.scanner.outcome.realized_transition_frame(...)        ~4.9s
qooi.scanner.outcome.kline_path_history_frame(...)         ~1.8s
qooi.scanner.diagnostics._run_pipeline(...)                ~0.7s
qooi.scanner.state.potential_observation_frame(...)       ~0.5s
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

Tailtree optimization precondition:

```text
do not tune LightGBM parameters while tailtree-run-summary shows
forward_min/max non-null count = 0 or train_exceedance_count = 0
```

Do not optimize Markdown table formatting before these frame builders.

Report rendering graph:

```text
DiagnosticFrames.candidate_feasibility
  → qooi.scanner.report.CandidateSelectionSection.render(frame) -> str

DiagnosticFrames.history_feasibility/source_freshness/candidate_feasibility
  → qooi.scanner.report.DataHealthSection.render(...) -> str
  # current implementation derives aggregate data-health in the report section.

DiagnosticFrames.potential_evidence
  → report_sections_for(evidence)

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

## Migration verification

The workflow-first migration is complete when these static/fast checks pass without
running scanner scripts:

```bash
uv run ruff check src tests
uv run ty check
uv run pytest tests/ -q
git diff --check HEAD
```

Boundary checks must also show that removed compatibility/transitional modules are
absent:

```text
qooi.scanner.classifiers
qooi.scanner.features
qooi.scanner.history
qooi.scanner.source_events
qooi.scanner.decisions
qooi.scanner.frames
qooi.scanner.events
```

The old one-file `tailtree.py` and `tailrun.py` modules are also removed; their public
module names now resolve to real packages with product-owned files.

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
