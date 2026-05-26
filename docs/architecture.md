# Behavior-Driven State Research Architecture

Date: 2026-05-22

## Purpose

This document defines the target research architecture for behavior-driven market-state discovery in `qooi` after the reduced static/joint bucket run failed strict promotion gates.

The architecture does not preserve prior helper-module boundaries as compatibility constraints. Prior `joint-forward-quality` work remains empirical evidence, but the new implementation should use a smaller, common workflow that can score deterministic transitions, learned states, and future policy contexts consistently.

## Motivation

The canonical reduced run produced valid artifacts but no promotion candidates:

| Artifact | Result |
|---|---:|
| `joint-forward-quality.csv` | 166,151 rows |
| Diagnostic candidates | 363 rows |
| Aggregate `ALL` diagnostic candidates | 47 rows |
| `joint-promotion-candidates.csv` | 0 rows |

Conclusion:

- Static/reduced manually configured buckets are not robust enough for strategy conversion.
- Thresholds should not be relaxed to manufacture survivors.
- The next research unit should be state change and transition behavior.
- The implementation should not be organized around legacy `joint_quality` helpers.

## Architecture Principles

Research principles:

- Keep diagnostics deterministic before adding model complexity.
- Treat empty promotion exports as valid rejection evidence.
- Prefer Polars/DataFrame contracts over object-heavy helper layers.
- Keep the public API graph small.
- Add dependencies only when a named module requires them.
- Favor one shared pattern-scoring workflow over output-specific code paths.

Layering principles:

- Data layer does not know strategies, baskets, execution, or reports.
- Strategy layer emits explicit signal columns and does not know fills, fees, equity, or recovery.
- `process_bar()` returns proposals only.
- `BasketBook` owns lifecycle mutation.
- Executor owns fills, fees, trade rows, and equity accounting.
- Research models must not own basket lifecycle, fills, fees, cash, exchange calls, or recovery.

Promotion principles:

- No diagnostic artifact becomes strategy logic directly.
- A candidate must have a market-behavior rationale before implementation.
- Deployable candidates must become normal signal columns.
- Execution-aware backtests remain the promotion boundary.

## API Graph

Diagnostics modes remain:

```text
DiagnosticMode = "backtest" | "research-evaluation"
```

Target retained output names:

```python
ResearchOutputName = Literal[
    "timeframe-classifier",
    "dynamic-transition-discovery",
    "pattern-quality",
    "trade-record-modulation",
]
```

`joint-forward-quality` is not a privileged target module, permanent architecture boundary, or compatibility output. The target concept is scored patterns, not a joint-quality subsystem.

Target graph:

```text
prepared market/classifier frames
  -> timeframe-classifier
  -> dynamic-transition-discovery
       -> state-transition-graph.csv
       -> transition-information.csv
       -> transition-ngram-quality.csv
       -> none-event-context-quality.csv
  -> pattern-quality
       -> scored-patterns.csv
       -> promotion-candidates.csv
optional backtest branch
  -> trade-record-modulation
```

Rules:

- Do not add new diagnostics modes.
- Do not restore `market-state-forward`, `timeframe-forward-quality`, `resonance-candidates`, `tradability`, `state-*`, or `modulation-effect`.
- Do not create compatibility aliases for removed outputs.
- Keep Stage 1 transition CSVs under `dynamic-transition-discovery`.
- Use `pattern-quality` as the shared scoring surface across static, transition, learned-state, and policy-context patterns.

## Module Dependency Graph

The target architecture is terse composition plus table-owned transformations. Scripts are single-role config entry points, not generic command dispatchers.

## Current Implementation Snapshot

The current code has removed the prior generic package modules:

- `src/qooi/research/run.py` is deleted.
- `src/qooi/research/workflows.py` is deleted.
- `src/qooi/research/runner.py` is deleted.
- `src/qooi/research/pipeline.py` remains deleted and must not return.

Current research modules:

```text
src/qooi/research/config.py                # command config, output names, stage config
src/qooi/research/instruments.py           # research/core universes
src/qooi/research/data.py                  # cache-backed request planning and prepared frames
src/qooi/research/tables.py                # ResearchFrame -> patterns/outcomes/metrics/artifacts
src/qooi/research/states.py                # classifier-health and learned-state contracts
src/qooi/research/reports.py               # backtest/cache-audit/research report helpers
```

Current classifier-state graph:

```text
scripts/classifier_states.py
  -> config.load_research_command_config(...)
  -> reports.classifier_state_research(...)
       -> data.prepare_classifier_frame(...)
       -> states.classifier_health(...)
       -> tables.build_transition_bundle(...)
       -> tables.ArtifactBundle.write(...)
```

Current assessment:

- `scripts/classifier_states.py` is the single-role entry point for classifier-state research evaluation.
- `scripts/learned_states.py` is the single-role entry point for learned behavior-state discovery.
- `data.py` owns research-specific cache requests plus classifier/context/signal frame preparation.
- `tables.py` owns the canonical research table pipe and artifact projections.
- `states.py` owns classifier-health state diagnostics and learned-state contracts.
- `reports.py` is the remaining command-facing report helper for backtest, cache-audit, and research-evaluation text.
- `joint_quality.py` is not preserved as an architecture module.
- Schemas describe pipe contracts or artifact projections, never unrelated artifact unions.

Compatibility stance:

- Do not keep `joint_quality.py` as a permanent module.
- Do not design new modules around old artifact families.
- Keep return metrics, gate logic, and grouping logic in `tables.py` unless a second cohesive owner emerges.
- Do not reintroduce `pipeline.py`, `run.py`, `workflows.py`, or `runner.py`.
- Do not let `reports.py`, `data.py`, or `tables.py` grow unrelated responsibilities.
- Keep scripts single-role; do not reintroduce a generic research command dispatcher.

Dependency direction:

| Module | Imports | Responsibility |
|---|---|---|
| `scripts/classifier_states.py` | argparse, config, reports | Classifier-state research entry point |
| `scripts/learned_states.py` | argparse, config, states, tables | Learned behavior-state research entry point |
| `config.py` | Pydantic, typing | Command config models and output names |
| `data.py` | cache store, strategy feature prep | Cache-backed request planning and classifier/context/signal frame preparation |
| `reports.py` | executor/evaluate/data/tables | Backtest, cache-audit text, and research-evaluation text |
| `states.py` | Polars, AI contracts | Classifier health and learned `behavior_state_id` labels through common contracts |
| `tables.py` | dataclasses, pathlib, `math`, Polars | Normalize frames, materialize/scored patterns, and project/write artifacts |
| `policy_lab.py` | Optional RL imports inside guarded functions | Evaluate simulated policy contexts through common contracts |

Forbidden dependencies:

- `tables.py` must not import `BacktestExecutor`, `BasketBook`, `TradingClient`, `StrategyBehavior`, or `compute_signal_frame`.
- `states.py` must not call exchange clients, mutate caches, or import Torch.
- `policy_lab.py` must not submit orders or mutate real baskets.
- Optional ML/RL packages must not be imported at package import time by core/runtime modules.

Layout guidance:

- Prefer table-oriented functions that accept and return `pl.DataFrame`.
- Avoid manager/helper classes unless they own durable state or a protocol boundary.
- Keep composition code short, linear, and boring; if it needs a graph class, the API graph is too bloated.
- Do not add package modules whose only job is command dispatch by mode.
- Keep metric kernels, pattern construction, export concerns, and transition bundle construction in `tables.py` while they remain one cohesive table pipe.

Monolith reduction rules:

- No function both materializes patterns and scores returns.
- No function both scores metrics and applies promotion gates.
- No function both projects artifacts and writes files unless explicitly named `write_*`.
- No package module should exist only to dispatch commands by mode.
- No `run.py`, `workflows.py`, `runner.py`, or generic orchestration monolith is an architectural target.
- No schema constant contains columns for unrelated artifact families.
- No core API name contains `joint` unless it is a temporary migration projection.
- No Stage 2 or Stage 3 API requires changing Stage 1 scoring and gating functions.
- Backtest and research-evaluation branches do not share one orchestration class.
- Trade-record modulation remains a post-trade branch, not part of pattern scoring.

Line targets:

- `scripts/classifier_states.py` and `scripts/learned_states.py`: each under `150` lines.
- `src/qooi/research/reports.py`: split by actual ownership if report/backtest/cache-audit responsibilities keep growing.
- `src/qooi/research/data.py`: split only if classifier/context frame prep and signal frame prep grow independently.
- `src/qooi/research/tables.py`: keep cohesive table-pipe ownership; split only by durable table owner.

## Current Redundancy Audit

The remaining redundancy is acceptable only as a short migration state:

- `reports.py` still contains more than one command-facing concern: backtest execution helpers, cache-audit rendering, research-evaluation composition, export text, and post-trade modulation formatting.
- `data.py` combines classifier/context frame preparation and signal-frame preparation; keep it only while that coupling remains practical.
- `tables.build_transition_bundle()` is acceptable because it is a small table-pipe assembly over local table operations.
- `ArtifactBundle.write()` is the target writer boundary; duplicated per-table export writers should not return.
- `MetricTable` and `ScoredPatternTable` intentionally repeat identity columns for traceability, but derived columns must have one owner.
- Strict promotion support is incomplete in the current Stage 1 bundle: `tables.apply_promotion_gate()` exists but is not yet wired into `build_transition_bundle()`.
- `DynamicTransitionDiscoveryConfig.information_min_rows` exists, but transition-information sufficiency currently uses the hard-coded `100` row threshold inside `tables.summarize_transition_information()`.

Derived-column ownership:

| Derived Column Family | Owner |
|---|---|
| `pattern_value`, `ngram_length`, `invalid_state_present` | `tables.py` |
| `forward_return_pct`, `side`, `side_return_pct`, `forward_direction` | `tables.py` |
| `omega_ratio`, `pwpr`, information metrics, sufficiency flags | `tables.py` |
| `passes_candidate_gate`, `passes_promotion_gate`, support/agreement fields, failure reasons | `tables.py` |
| `artifact`, CSV-specific column ordering/projections | `tables.py` |

Consolidation backlog:

1. Split `reports.py` by actual ownership if it remains large after Stage 1 empirical runs.
2. Split `data.py` only if classifier/context frame prep and signal frame prep keep growing independently.
3. Wire strict promotion support into `tables.build_transition_bundle()`.
4. Pass `information_min_rows` into transition information sufficiency instead of relying on the hard-coded `100` row threshold.
5. Expand classifier quality feedback with state persistence, unknown/warmup rates, state balance, and timeframe agreement metrics if Stage 1 artifacts show classifier instability.
6. Keep `promotion-candidates.csv` interpretation conservative until strict promotion support is wired.

If Stage 2 or Stage 3 makes `MetricTable` null-heavy, introduce a long metric observation table instead of widening the schema further:

```text
pattern_id | metric_family | metric_name | metric_value | rows | sufficient | bias_warning
```

## Canonical Data Contracts

The redesign uses common table contracts so Stage 1 deterministic states, Stage 2 learned states, and Stage 3 policy research share scoring and promotion logic.

Pipe shape:

```text
source wide frames
  -> ResearchFrame
  -> PatternTable
  -> OutcomeTable
  -> MetricTable
  -> ScoredPatternTable
  -> ArtifactBundle
```

### `ResearchFrame`

A normalized known-at-close state/event table.

```text
symbol
timeframe
timestamp
open
high
low
close
state_source
state_column
state_value
event_column
event_value
context_columns...
```

Rules:

- One normalized long table is preferred for state/event analysis.
- Wide prepared frames may exist internally for efficient feature creation.
- Scoring functions consume normalized state/event rows.
- `state_source` identifies deterministic classifier, reduced classifier, learned encoder, policy context, or another source.
- Forward labels must not exist in `ResearchFrame`.

### `PatternTable`

A pattern table represents candidate grouping units before outcomes are attached.

```text
pattern_id
pattern_family
pattern_source
symbol
timeframe
timestamp
state_source
state_column
pattern_value
event_value
side
ngram_length
invalid_state_present
```

Rules:

- Static buckets, transitions, transition n-grams, learned-state paths, and future policy contexts all become patterns.
- `pattern_family` examples are `static_state`, `transition`, `transition_ngram`, `learned_state`, `policy_context`, and `none_event_context`.
- `pattern_id` is deterministic from source columns and values, not from row order.
- Pattern construction cannot use future return labels.

### `OutcomeTable`

An outcome table is the first contract where future labels may appear.

```text
pattern_id
pattern_family
pattern_source
symbol
timeframe
timestamp
horizon
side
forward_return_pct
side_return_pct
forward_direction
```

Rules:

- `OutcomeTable` is derived from `PatternTable` plus market OHLCV after pattern construction.
- It must retain `pattern_id` so metrics and promotion can trace back to pattern definitions.
- It must not be fed back into state, event, or pattern construction.

### `MetricTable`

A metric table contains aggregate measurements before gates are applied.

```text
pattern_id
pattern_family
pattern_source
symbol
horizon
side
rows
positive_rate
negative_rate
positive_mean
negative_mean_abs
omega_ratio
pwpr
sortino_zero
mean_side_return_pct
transition_information
conditional_transition_information
normalized_transition_information
normalized_conditional_transition_information
bias_warning
sufficient
```

Rules:

- `MetricTable` contains aggregate measurements only.
- It has no candidate or promotion booleans.
- It can contain null information fields for return-only rows and null return fields for information-only rows.

### `ScoredPatternTable`

A scored pattern table contains information metrics, return-quality metrics, and promotion fields.

```text
pattern_id
pattern_family
pattern_source
symbol
horizon
side
rows
positive_rate
negative_rate
positive_mean
negative_mean_abs
omega_ratio
pwpr
sortino_zero
mean_side_return_pct
transition_information
conditional_transition_information
invalid_state_present
passes_candidate_gate
passes_promotion_gate
promotion_failure_reasons
```

Rules:

- Return-quality and information metrics share one scored table where possible.
- Artifact-specific CSVs are projections of this table, not independent bespoke implementations.
- Promotion gates operate on `ScoredPatternTable` regardless of how the pattern was generated.

## Orthogonal API Set

The target API is a small set of nouns and verbs. Core APIs are pipe steps, not outputs.

### `frames.py`

```python
normalize_research_frame(frame, *, symbol, timeframe, state_columns, event_column, context_columns) -> pl.DataFrame
concat_research_frames(frames) -> pl.DataFrame
validate_research_frame(frame) -> pl.DataFrame
```

Rules:

- Converts wide prepared frames to long state rows.
- Does not compute forward labels.
- Does not build transition strings.
- Does not score metrics or apply gates.

### `patterns.py`

```python
materialize_static_patterns(research_frame, spec) -> pl.DataFrame
materialize_transition_patterns(research_frame, spec) -> pl.DataFrame
materialize_none_event_context_patterns(research_frame, spec) -> pl.DataFrame
concat_patterns(patterns) -> pl.DataFrame
```

Rules:

- Consumes only `ResearchFrame`.
- Emits only `PatternTable` rows.
- Does not attach forward labels or compute return metrics.
- Does not project CSV artifacts; transition graph projection lives in `artifacts.py`.

### `outcomes.py`

```python
attach_forward_outcomes(patterns, market_frame, horizons) -> pl.DataFrame
side_from_event(event_value) -> pl.Expr
attach_side_returns(outcomes) -> pl.DataFrame
```

Rules:

- Future labels appear for the first time here.
- Event-to-side mapping lives here or in a tiny constants module.
- Does not aggregate metrics.

### `metrics.py`

```python
entropy(counts_or_frame, column) -> pl.DataFrame | float
mutual_information(frame, x, y) -> pl.DataFrame
conditional_mutual_information(frame, x, y, z) -> pl.DataFrame
summarize_returns(outcomes, group_cols) -> pl.DataFrame
summarize_transition_information(patterns_or_research_frame, spec) -> pl.DataFrame
```

Rules:

- Does not know promotion thresholds.
- Does not write exports.
- Emits `MetricTable`-compatible rows.

### `promotion.py`

```python
apply_candidate_gate(metrics, thresholds) -> pl.DataFrame
symbol_support(metrics, keys, min_symbol_rows) -> pl.DataFrame
time_split_support(outcomes_or_metrics, keys, config) -> pl.DataFrame
apply_promotion_gate(scored, thresholds) -> pl.DataFrame
```

Rules:

- Consumes metric or scored tables only.
- Does not compute base return metrics.
- Is shared by static, transition, none-context, and learned-state pattern families.

### `artifacts.py`

```python
ArtifactBundle(name, tables, summary, warnings, metadata)
ArtifactBundle.write(export_dir) -> list[str]
project_transition_graph(patterns) -> pl.DataFrame
project_transition_information(scored) -> pl.DataFrame
project_pattern_quality(scored, families) -> pl.DataFrame
project_promotion_candidates(scored) -> pl.DataFrame
```

Rules:

- Artifact projection is a view over shared contracts.
- CSV schemas are owned here, not in metric functions.
- `project_transition_graph()` owns state-transition graph projection from transition patterns.
- Empty promotion exports are allowed and explicit.

### `transition_discovery.py`

```python
build_dynamic_transition_bundle(prepared_frames, frame_specs, horizons, thresholds) -> ArtifactBundle
```

Rules:

- Contains one small Stage 1 bundle builder.
- Does not dispatch commands, load caches, render text, or write CSVs.
- Returns bundles; command-facing composition converts bundles to text or writes them.

### `ArtifactBundle`

A runner output bundle groups tables and summary metadata.

```text
name
tables: dict[str, pl.DataFrame]
summary: list[str]
warnings: list[str]
metadata: dict[str, str]
```

Rules:

- `transition_discovery.py` returns artifact bundles.
- `artifacts.py` writes CSVs and summaries from bundles.
- Export concerns do not leak into metric or pattern generation functions.

## Shared Workflow

Replace output-specific workflows with one stage pipeline:

```text
load/prep frames
  -> normalize state/event rows             # frames.py
  -> materialize patterns                   # patterns.py
  -> attach forward outcomes                # outcomes.py
  -> compute metrics                        # metrics.py
  -> apply promotion gates                  # promotion.py
  -> project artifact tables and summaries  # artifacts.py
  -> write exports                          # artifacts.py, called by command-facing composition
```

Stage mapping:

| Stage | Pattern Source | Shared Workflow Reuse |
|---|---|---|
| Stage 1 dynamic transitions | Deterministic classifier state paths | Uses all common steps with no optional dependencies |
| Stage 2 endogenous states | Learned `behavior_state_id` labels | Extends `ResearchFrame`, then reuses patterns/outcomes/metrics/promotion |
| Stage 3 policy lab | Simulated policy context/action proposals | Uses `PatternTable`/promotion-style evaluation before any signal adapter |

Output projections:

| CSV | Projection Source |
|---|---|
| `state-transition-graph.csv` | `PatternTable` transition edge counts |
| `transition-information.csv` | Current: `MetricTable` rows from `metrics.summarize_transition_information(research_frame)`; target: artifact-owned projection after information rows are integrated into scored tables |
| `transition-ngram-quality.csv` | `ScoredPatternTable` where `pattern_family=transition_ngram` |
| `none-event-context-quality.csv` | `ScoredPatternTable` where `pattern_family=none_event_context` |
| `scored-patterns.csv` | Full or configured projection of `ScoredPatternTable` |
| `promotion-candidates.csv` | `ScoredPatternTable` where `passes_promotion_gate=true` |

## Stage 1: Dynamic Transition Discovery

Stage 1 is deterministic and diagnostic-only.

Inputs:

- Known-at-close classifier labels.
- Known-at-close liquidity event labels.
- Current and previous deterministic context columns.
- Forward returns as labels only after pattern construction.

Default state columns:

```text
market_stage_reduced
d1_market_stage_reduced
h4_market_stage_reduced
structure_trend_state
```

Default event column:

```text
liquidity_event_type
```

Default n-gram lengths:

```text
2, 3
```

### Metrics

`metrics.py` owns empirical count-table and return-quality metrics:

```text
H(X) = -sum_x p(x) log2 p(x)
I(X; Y) = H(X) - H(X | Y)
I(X; Y | Z) = H(X | Z) - H(X | Y, Z)
```

Terms:

- `transition information` means `I(S_t; S_t-1)`.
- `conditional transition information` means `I(S_t; S_t-1 | E)`.
- These are not transfer entropy.

Metric implementation rules:

- Use Polars group-by counts and standard-library math.
- Filter null and invalid state labels explicitly.
- Emit sufficiency and sparsity warnings instead of silently trusting high-cardinality estimates.
- Reuse return-quality metrics across static, transition, none-context, and learned-state patterns.

### Current Stage 1 Readiness

Stage 1 can be applied now to the handcrafted classifier. The current implementation supports one command that produces both classifier-quality feedback and transition-property feedback:

```bash
uv run python scripts/classifier_states.py --config configs/research/dyn-trans.toml
```

The configured export directory is:

```text
F:\Stratum\TEMP\kilo\qooi-dyn-trans
```

Feedback families:

- Handcrafted classifier quality comes from `timeframe-classifier` via `diagnostics.classifier_health()`.
- Transition properties come from `dynamic-transition-discovery` and `pattern-quality` via the shared Stage 1 table pipe.

Current artifacts and the questions they answer:

| Artifact | Answers |
|---|---|
| `timeframe-classifier.csv` | Are handcrafted classifier columns present and cardinality sane by symbol/timeframe? |
| `state-transition-graph.csv` | Which deterministic states persist or transition, and with what empirical probabilities? |
| `transition-information.csv` | How much current state is explained by previous state, optionally conditioned on event labels? |
| `transition-ngram-quality.csv` | Do state transition paths have side-normalized forward-return signal? |
| `none-event-context-quality.csv` | Do non-event contexts carry useful conditional return behavior? |
| `scored-patterns.csv` | Full candidate-gated pattern metric surface. |
| `promotion-candidates.csv` | Strict promotion export; currently expected to be empty or unscored until symbol/time-split support is wired. |

Evaluation checklist after a Stage 1 run:

1. Review `timeframe-classifier.csv` for required-column failures and cardinality warnings.
2. Review `state-transition-graph.csv` for dominant self-transition probabilities and plausible non-self transitions by `state_column`.
3. Review `transition-information.csv` for `transition_information`, `normalized_transition_information`, and conditional information changes with `liquidity_event_type`.
4. Review `transition-ngram-quality.csv` and `scored-patterns.csv` for candidate counts, sparse high-ratio rows, and invalid-state contamination.
5. Treat `promotion-candidates.csv` as a placeholder until strict promotion support is wired.

### Stage 1 Interpretation Limits

Current Stage 1 is valid as a diagnostic read on deterministic handcrafted states, but it is not sufficient for strategy promotion.

Limits:

- `classifier_health()` is a structural health check, not a complete classifier-quality model.
- Current classifier feedback confirms required columns and basic cardinality sanity, but does not yet evaluate state persistence, unknown/warmup rates, state balance, or timeframe agreement.
- `transition-information.csv` currently comes directly from `metrics.summarize_transition_information(research_frame)` rather than a fully projected scored-pattern table.
- Transition information sufficiency currently uses a hard-coded `100` row threshold instead of config `information_min_rows`.
- Strict promotion fields are not yet meaningful because symbol/time-split support is not wired into `transition_discovery.py`.
- `promotion-candidates.csv` should not be used as evidence of no robust patterns until strict promotion wiring is complete.
- Stage 1 outputs can prioritize classifier revisions and transition hypotheses, but cannot authorize strategy rules.

### Pattern Materialization

`patterns.py` owns deterministic pattern construction.

Core functions:

```text
materialize_static_patterns(frame, state_columns)
materialize_transition_patterns(frame, state_columns, ngram_lengths)
materialize_none_event_context_patterns(frame, context_columns)
```

It returns `PatternTable` rows rather than output-specific dataclasses.

### Outcome Attachment

`outcomes.py` owns forward label construction.

Core functions:

```text
attach_forward_outcomes(patterns, market_frame, horizons)
attach_side_normalized_returns(patterns_with_outcomes)
```

Forward labels are attached after pattern construction so future data cannot leak into state or transition definitions.

### Artifact Projections

`artifacts.py` projects Stage 1 CSVs from common tables.

`state-transition-graph.csv` columns:

```text
artifact
symbol
timeframe
state_column
source_state
target_state
rows
source_rows
transition_probability
invalid_state_present
```

`transition-ngram-quality.csv` and `none-event-context-quality.csv` use the shared return-quality and promotion fields from `ScoredPatternTable`.

No NetworkX dependency is needed for Stage 1.

## Stage 2: Endogenous State Discovery

Stage 2 discovers endogenous market-behavior states from known-at-close OHLCV shape windows, then evaluates those labels through the same research pipe as deterministic Stage 1 states. The learned-state path supports multi-asset training through a shared codebook and optional per-symbol causal volatility scaling.

Implemented flow:

- `qooi.research.states.LearnedStateConfig.prepare()` builds causal relative OHLCV features, chronological splits, `PreparedWindows`, and separate `WindowProvenance`.
- `qooi.research.states.LearnedStateConfig.prepare_many()` applies the same preparation per symbol, assigns splits per symbol, and merges windows for one shared VQ-RSSM dataset.
- `qooi.ai.contracts.WindowDataset` is the only numeric model input and carries no symbol or timestamp provenance.
- `qooi.ai.vq_rssm` owns the fixed VQ-RSSM model, `train()`, `save_checkpoint()`, `load_checkpoint()`, and `predict_codes()`.
- `CodeSequence` is mapped back to research-owned `StateSequence` through `WindowProvenance` before `behavior_state_id` is attached.
- Learned labels enter the shared pipe through `tables.normalize_research_frame(..., state_source="vq_rssm")`, then reuse `tables.py` pattern, outcome, metric, gate, and artifact operations.

Constraints:

- Encoder inputs must end at or before the decision bar close.
- Future returns can train supervised heads but must not enter encoder input.
- Chronological train/validation/test splits are mandatory.
- Multi-asset splits are chronological per asset before windows are merged.
- VQ-RSSM training updates parameters only on `split == "train"`; validation rows are metrics-only and test rows are count/evaluation-only.
- `behavior_state_id` is a research label until promoted.
- `states.py` and `qooi.ai.contracts` import without Torch; `qooi.ai.vq_rssm` requires `uv sync --group ml`.
- Symbol IDs and asset embeddings must not enter encoder input unless a later ADR explicitly approves asset-conditioned learned states.

Dependency group if approved:

```toml
[dependency-groups]
ml = [
    "torch>=2.3",
]
```

TensorFlow/Keras is not a default dependency. Add it only for a concrete required model stack.

## Stage 3: Policy Learning And World Models

Stage 3 can start only after Stage 2 produces stable learned states.

Target implementation:

- Add `policy_lab.py` with simulation interfaces first.
- Define observation schema, action schema, reward function, and train/eval splits before choosing an RL library.
- Represent policy contexts or action proposals as `PatternTable`-compatible research rows where possible.
- Use policy learners as research proposal generators, not executors.

Constraints:

- No exchange calls.
- No basket mutation.
- No fill, fee, cash, or recovery ownership.
- Any deployable result must adapt into normal signal columns.

Optional dependency group if approved:

```toml
[dependency-groups]
rl = [
    "gymnasium>=0.29",
    "stable-baselines3>=2.0",
]
```

Dreamer/world-model dependencies are deferred until there is a concrete environment contract and evidence that simpler baselines are inadequate.

## Promotion And Backtest Contract

A transition pattern or learned state can become a strategy hypothesis only if all gates pass:

- Classifier or encoder health passes.
- State construction is no-lookahead.
- Row count, symbol support, and time-split support pass strict gates.
- Omega/PWPR and mean side return are material.
- Invalid states are excluded.
- The pattern has an ex-ante market-behavior rationale.
- The candidate is converted into explicit signal columns.
- Execution-aware backtests pass with costs, slippage, stops, targets, sizing, basket caps, and comparability checks.

Deployable signal columns:

```text
raw_entry_signal
entry_signal
position_signal
exit_signal
signal_strength
signal_id
```

## Dependency Policy

Stage 1 adds no new dependency.

Dependency decisions:

| Dependency | Decision |
|---|---|
| NumPy | Avoid in Stage 1 unless numeric kernels become clearer or faster |
| SciPy | Avoid in Stage 1; consider for later statistical validation |
| NetworkX | Avoid in Stage 1; use CSV/DataFrame graph artifacts |
| scikit-learn | Optional Stage 2 baseline/utility dependency |
| PyTorch | Preferred optional Stage 2/3 deep learning framework |
| TensorFlow/Keras | Not default |
| Gym | Do not use for new work |
| Gymnasium | Optional Stage 3 environment API |
| Stable-Baselines3 | Optional Stage 3 baseline policy toolkit |
| Dreamer/world-model stacks | Defer |

Acceptance gates for new dependencies:

- The dependency has a named module owner.
- The dependency solves a concrete problem not clear with existing Polars/standard-library code.
- It is optional unless required by the core runtime.
- Import errors mention the uv group required to install it.

## Refactor Procedure

Phase 0 documentation rewrite:

1. Rewrite `docs/research-evaluation-report.md` around static-bucket rejection and the architecture pivot.
2. Rewrite `docs/research-evaluation-api-reference.md` around the current reduced graph and future extension graph.
3. Update `docs/context.md` with behavior-driven state research boundaries and glossary terms.
4. Add this architecture document.

Phase 1 workflow redesign:

1. Add common contracts: `ResearchFrame`, `PatternTable`, `OutcomeTable`, `MetricTable`, `ScoredPatternTable`, and `ArtifactBundle`.
2. Add `frames.py`, `patterns.py`, `outcomes.py`, `metrics.py`, `promotion.py`, `artifacts.py`, and `transition_discovery.py`.
3. Migrate reusable logic out of `joint_quality.py`; do not preserve it as a target module.
4. Keep `diagnostics.py` small and outside Stage 1 scoring; do not preserve `run.py`, `workflows.py`, or `runner.py` facades.
5. Add tests for pattern materialization, forward outcome attachment, entropy/MI/CMI, return-quality metrics, promotion gates, and artifact projections.

Phase 2 API graph modification:

1. Add `dynamic-transition-discovery` to `ResearchOutputName`.
2. Add `pattern-quality` if a shared scored-pattern export is needed.
3. Add `DynamicTransitionDiscoveryConfig`.
4. Orchestrate dynamic transition discovery through the common workflow.
5. Project the four Stage 1 CSV artifacts from common tables.
6. Add config parse tests.

Phase 3 empirical run:

1. Create `configs/research/dyn-trans.toml`.
2. Run `uv run python scripts/classifier_states.py --config configs/research/dyn-trans.toml`.
3. Review all Stage 1 artifacts.
4. Apply strict promotion gates before strategy work.

## Validation Plan

Docs-only validation:

```bash
uv run ruff check docs
```

Stage 1 implementation validation:

```bash
uv run ruff check src/qooi/research scripts tests
uv run pytest tests/test_research_data.py tests/test_research_diagnostics.py tests/test_research_backtest.py tests/test_classifier_diagnostics.py
uv run python -c "from pathlib import Path; from qooi.research.config import load_research_command_config; [load_research_command_config(p) for p in sorted(Path('configs/research').glob('*.toml'))]"
```

## Non-Goals

- Do not authorize live trading.
- Do not add Stage 1 dependencies.
- Do not add new diagnostics modes.
- Do not resurrect removed diagnostic modes or outputs.
- Do not preserve `joint_quality.py` as a target architecture module.
- Do not weaken promotion thresholds.
- Do not modify execution, basket, recovery, or exchange layers for Stage 1.
- Do not implement ML/RL before deterministic transition evidence justifies it.
