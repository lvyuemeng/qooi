# Scanner Module Graph

This is the applied architecture-to-module-graph contract for scanner research. It maps the scanner architecture objects (`U`, `K`, `S`, `O`, `Y`, `E`, `E*`, `C`, `R`, `B`) to concrete current public calls, artifacts, and public calls, artifacts, and implementation status.

The scanner module surface stays thin and research-only. `workflow.py` orchestrates; scanner submodules expose only functions needed by workflow, diagnostics, report rendering, tests, or research artifact consumers. Helper functions remain private implementation details.

## Implementation status

| Architecture object | Current graph/API surface | Artifact surface | Status |
|---|---|---|---|
| `U` universe | `workflow.load_config()`, `workflow.resolve_universe()` | config/discovery result only | implemented |
| `K` kline state | `workflow.load_bars()`, `KlineClassifier.classify()`, `decisions.compute_kline_states()`, `history.kline_path_history_frame()` | state frames | implemented |
| `S` source state/event | `sources.context.load_source_context()`, `decisions.compute_source_states()`, `source_events.source_events_frame()` | source frames | implemented |
| `O` observation vector | `evidence.potential_observation_frame()` | `potential-observation-summary.csv` | implemented |
| `Y` future outcome | `history.realized_transition_frame()`, `source_events.source_outcomes_frame()`, `evidence.potential_outcome_frame()` | outcome frames | implemented |
| `E` evidence | `evidence.potential_evidence_frame()`, `evidence.add_potential_parent_gain()` | `potential-evidence-summary.csv` | implemented |
| `E*` selected evidence | `evidence.select_potential_evidence_level()` | `potential-evidence-selected.csv` | implemented — gate has known bug (market_background still selected via join path) |
| `C` candidate evidence row | `candidates.candidate_evidence_frame(...)` | `candidate-evidence.csv` | implemented |
| `R` ranked candidate | `candidates.rank_candidate_evidence(...)` | `candidate-rank.csv` | implemented |
| `B` evidence backtest row | removed | removed | removed — >97% holdout mismatch rate |

## Architecture: two-lens design

The scanner has two independent evaluation paths that converge in the report:

**Decision lens** — coin-specific, current-state, all source families

```
workflow.run()
  → compute_kline_states()         # latest kline state per symbol
  → compute_transition_insights()  # n-gram patterns from coin's own history
  → load_source_context()          # books, trades, funding, oi, taker, ratios
  → compute_source_states()        # per-source direction + contradictions
  → scan_review_decisions()        # 9-rule table → ScanDecision per symbol
```

Answers: "What does THIS coin's current state look like across all data sources right now?"

**Evidence lens** — cross-coin, historical, kline-primary state vector

```
write_diagnostics()
  → source_events/outcomes         # source forward returns
  → kline_path_history → realized_transitions  # kline forward paths
  → potential_observation_frame()  # state vectors at every bar close
  → potential_evidence_frame()     # group by state → entropy → info gain
  → select_potential_evidence()    # gate: stable, improving on parent, info > 0
  → candidate_evidence_frame()     # match latest observations to evidence pool
  → rank_candidate_evidence()      # composite score: info + tail + stability + quality
```

Answers: "When this state pattern appeared historically across any coin, what happened next — and how strong is the statistical signal?"

The two lenses use overlapping data (kline + sources) with different methodology. The report presents both in one merged row with provenance labels. Design doc: `docs/architecture/scanner-plan.md`.

Anything not listed here is not a supported scanner research surface unless another graph section explicitly names it.

## CLI entry

```text
scripts/potential_scan.py        # potential scanner CLI
  parse_args()
  main()
    -> qooi.scanner.workflow.run(Path(args.config))
```

## Orchestrator flow

```text
qooi.scanner.workflow
  PotentialConfig                 # TOML-backed scanner config model
  DiscoveryWorkflowConfig         # adapter for exchange discovery

  run(config_path)
    -> load_config(config_path)
    -> resolve_universe(config)
    -> load_bars(config, universe.symbols)
       -> _classify_kline_frames(...)
          -> qooi.scanner.classifiers.KlineClassifier(scale).classify(frame)
    -> qooi.scanner.decisions.compute_kline_states(...)
    -> qooi.scanner.transitions.compute_transition_insights(...)
    -> qooi.scanner.contracts.context_symbols(...)
    -> qooi.sources.context.load_source_context(...)
    -> qooi.scanner.decisions.compute_source_states(...)
    -> qooi.scanner.decisions.scan_review_decisions(...)
    -> qooi.scanner.diagnostics.write_diagnostics(...)
    -> qooi.scanner.report.render_report(...)
    -> writes report Markdown and diagnostics/state artifacts

  load_config(config_path)
  target_min_bars(days, timeframe)
  resolve_universe(config)
  load_bars(config, symbols)
```

## Computable object module graph

The scanner graph transforms the architecture object contracts into concrete module calls. Implemented edges are listed as direct calls. Backtest replay uses a private chronological workflow split, writes holdout artifacts, and keeps the public API limited to the pure candidate functions below.

```text
U: Universe
  scripts/potential_scan.py
    -> workflow.run(config_path)
       -> workflow.load_config(config_path) -> PotentialConfig
       -> workflow.resolve_universe(config) -> PotentialUniverse

K: Known-at-close kline state
  PotentialUniverse.symbols + PotentialConfig.timeframes
    -> workflow.load_bars(config, symbols) -> BarFetchResult
       -> AsyncCacheStore.bars(HistoryRefreshRequest) -> OHLCV frames + coverage
       -> KlineClassifier(timeframe).classify(frame) -> state_frames[(symbol,timeframe)]
    -> decisions.compute_kline_states(...) -> latest SourceStateRow rows
    -> history.kline_path_history_frame(config, state_frames) -> kline-path-history.csv

S: Known-at-close source state/event
  transitions + source scope
    -> contracts.context_symbols(config, symbols_with_decision_bars, transitions.insights)
    -> sources.context.load_source_context(...) -> source context frames + availability
    -> decisions.compute_source_states(...) -> latest source bundle rows
    -> source_events.source_events_frame(context.frames, bars, config.bar) -> source-events.csv

O: Observation vector
  kline-path-history.csv + source-events.csv
    -> evidence.potential_observation_frame(
         kline_history,
         source_events,
         decision_timeframe=config.bar,
         max_source_staleness_hours=config.max_source_staleness_hours,
       ) -> potential-observation-summary.csv

Y: Future outcome
  kline-path-history.csv
    -> history.realized_transition_frame(kline_history, horizons) -> realized-transition.csv
  source-events.csv + decision-clock OHLCV
    -> source_events.source_outcomes_frame(source_events, bars) -> source-outcomes.csv
  O + source outcomes + realized transitions
    -> evidence.potential_outcome_frame(...) -> outcome join frame

E: Parent-gated evidence
  O + Y
    -> evidence.potential_evidence_frame(
         observations,
         source_outcomes,
         realized_transitions,
         return_threshold_pct=config.transition_return_threshold_pct,
       )
       -> evidence.add_potential_parent_gain(...)
       -> evidence.select_potential_evidence_level(...)
       -> potential-evidence-summary.csv
       -> potential-evidence-selected.csv

C: Candidate evidence row
  latest O_t + selected E*(O,h) + coverage/freshness diagnostics
    -> candidates.candidate_evidence_frame(...) -> candidate-evidence.csv

R: Ranked candidate
  candidate-evidence.csv
    -> candidates.rank_candidate_evidence(...) -> candidate-rank.csv
```

Current implemented graph reaches `C` and `R` diagnostics. `B` was removed; the replay path had a >97% holdout-match failure rate and added no research signal.

Primary evidence diagnostics are real columns, including:

```text
baseline_p_up / conditioned_p_up
baseline_p_down / conditioned_p_down
lift_up / lift_down / lift_flat
information_gain_bits
transition_information_gain_bits
tail_up_rate / tail_down_rate
avg_forward_max_return_pct / avg_forward_min_return_pct / avg_path_range_pct
path_skew
returned_to_origin_rate
information_stability / transition_information_stability
statistical_direction
research_suggestion
```

These columns support extreme-behavior review. They should not be collapsed into one opaque score or treated as execution signals.

## Public scanner module surface

```text
qooi.scanner.classifiers
  ClassifierHealthResult
  KlineClassifier.classify()
  classifier_health()
  validate_state_frame()
```

Detailed classifier graph: `docs/graph/classifier.md`.

```text
qooi.scanner.contracts
  PotentialScanConfig
  PotentialUniverse
  PotentialArtifacts
  BarFetchResult
  SourceStateRow
  TransitionPattern
  TransitionInsight
  TransitionEdge
  UnsupportedTransitionPath
  TransitionAnalysis
  SymbolStateBundle
  ScanDecision
  ReportInputs

  missing_state()
  float_value()
  float_or_none()
  fmt()
  max_timestamp()
  best_transition_pattern()
  transition_consensus_passes()
  context_symbols()
```

```text
qooi.scanner.decisions
  compute_kline_states()
  compute_source_states()
  scan_review_decisions()
```

```text
qooi.scanner.transitions
  compute_transition_insights()
  transition_edges()
  transition_insight()
```

```text
qooi.scanner.candidates
  candidate_evidence_frame()
  rank_candidate_evidence()
```

```text
qooi.scanner.diagnostics
  DiagnosticFrames
  StateFrames
  write_diagnostics()
  coverage_frame()
```

```text
qooi.scanner.evidence
  potential_observation_frame()
  potential_evidence_frame()
  potential_outcome_frame()
  add_potential_parent_gain()
  select_potential_evidence_level()
```

```text
qooi.scanner.history
  kline_path_history_frame()
  kline_path_rows()
  realized_transition_frame()
```

```text
qooi.scanner.source_events
  source_events_frame()
  source_outcomes_frame()
  source_timeliness_frame()
  source_state_predictability_frame()
```

```text
qooi.scanner (package)
  pct_change_expr()
  outcome_bucket_expr()
  entropy_expr()
  entropy_term()
```

```text
qooi.scanner.report
  render_report()
```

## Artifact ownership

Implemented artifacts:

```text
workflow.run(...)
  -> PotentialArtifacts.report        # Markdown report
  -> PotentialArtifacts.diagnostics_dir
  -> PotentialArtifacts.states_dir

diagnostics.write_diagnostics(inputs)
  -> compact CSV diagnostic files (summaries, not full raw frames)
  -> potential-observation-summary.csv    # O_t grouped summary
  -> potential-evidence-summary.csv       # E grouped by level/status
  -> potential-evidence-selected.csv      # E* selected rows only
  -> candidate-evidence.csv               # C_t = latest O_t + selected E*
  -> candidate-rank.csv                   # R_t with explicit score components

report.render_report(inputs)
  -> report text with quantitative review rows; no writes
```

## Forbidden scanner edges

- No executor, live-trading, order, wallet, or basket mutation ownership.
- No dynamic/AI state input to scanner decisions unless a future architecture rewrite explicitly promotes it.
- No future returns/transitions in current-state classification; outcomes are diagnostics/evidence only.
- No secret, API key, or exchange-wallet label hardcoding.

## Boundary tests

`tests/test_module_boundaries.py` guards scanner dependencies from execution/trading/AI boundaries. Add to it before expanding scanner dependencies.
