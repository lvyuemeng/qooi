# Research Module Graph

Research should expose table-level contracts, not every helper used to assemble those tables. Private helpers stay private so scanner, strategy, and execution code cannot depend on incidental research internals.

```text
scripts/classifier_states.py
  -> qooi.research.config.load_research_command_config()
  -> qooi.research.reports.classifier_state_research()
```

## Public research module surface

```text
qooi.research.config
  load_research_command_config()
  resolve_research_outputs()
  apply_sizing_overrides()
  risk_gate_metadata()
  ResearchCommandConfig and nested config models

qooi.research.data
  FrameRequest
  CacheAuditRequest
  BacktestFrameOptions
  FrameResult
  PreparedBacktestFrame
  source_inst_ids()
  load_frame()
  load_frame_with_raw_rows()
  provenance_row()
  attach_higher_timeframe_context()
  add_mtf_state_keys()
  coverage_metadata()
  prepare_classifier_frame()
  prepare_signal_frame()
  build_history_refresh_requests()

qooi.research.artifacts
  ArtifactBundle
  ensure_columns()
  empty_frame()

qooi.research.patterns
  normalize_research_frame()
  materialize_transition_patterns()
  materialize_state_patterns()
  attach_forward_outcomes()
  filter_evaluation_outcomes()
  summarize_returns()
  summarize_transition_information()
  summarize_state_info()
  with_transition_path_scores()
  project_transition_paths()
  apply_candidate_gate()
  project_transition_graph()
  project_pattern_quality()
  build_transition_bundle()

qooi.research.behavior_tables
  summarize_state_diagnostics()
  build_state_transition_chains()
  summarize_state_chain_information()
  classify_state_taxonomy()

qooi.research.candidates
  build_candidate_nonoverlap_trades()
  bootstrap_candidate_trades()
  summarize_candidate_direction_asymmetry()
  summarize_candidate_alpha_beta()
  summarize_candidate_regime_segments()

qooi.research.rule_primitives
  RulePrimitiveConfig
  build_rule_primitive_signals()
  build_rule_primitive_trades()
  summarize_rule_primitives()
  build_rule_primitive_baselines()

qooi.research.reports
  classifier_state_research()
```

## Research pipe

```text
prepared frames
  -> ResearchFrame
  -> PatternTable
  -> OutcomeTable
  -> MetricTable
  -> ScoredPatternTable
  -> ArtifactBundle
```

## Reduction notes

- Backtest/report orchestration helpers inside `reports.py` are private implementation details.
- Cache/filter/context helpers inside `data.py` are private unless current callers need them.
- Promotion projection helpers remain private until promotion policy is stable.
- Do not add wrappers for old research table APIs; update current callers or remove stale tests.
