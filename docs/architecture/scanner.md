# Scanner Architecture

## Purpose

The scanner is a deterministic research workflow for finding symbols whose known-at-close state vectors materially change the probability or severity of future extreme behavior. It emits research diagnostics and review candidates only; it does not authorize live trading, allocation, executor actions, or wallet operations.

## Module layout

Scanner modules should be organized by durable data-product ownership, not by every
intermediate helper computation. OHLCV labels, continuous features, source rows, kline
path rows, observations, outcomes, evidence, ranks, and feasibility are necessary
computations, but they do not all need peer public APIs.

Target product-shaped layout:

```text
src/qooi/scanner/
├── workflow.py       # outer scan lifecycle: config, universe, fetch/cache, final write
├── __init__.py       # scanner-local dataclasses/protocols + shared Polars expr helpers
├── config.py         # PotentialConfig and section config models
├── state.py          # known-at-close observation/state/feature product facade
├── outcome.py        # future/path/source outcome product facade
├── ladder.py         # fixed categorical evidence path
├── tailtree/         # LightGBM + GPD model/evidence package; optional deps
├── tailrun/          # tailtree lifecycle/artifacts package
├── rank.py           # candidate evidence, rank, and horizon-consistency products
├── feasibility.py    # source/history/candidate feasibility projections
├── diagnostics.py    # diagnostic artifact writer/assembler only
└── report.py         # markdown renderer from prepared frames
```

Removed transitional modules now live under their product owners:

| Removed module | Current owner | Migration note |
|---|---|---|
| `classifiers.py` | `state.py` | OHLCV categorical labels are state internals. |
| `features.py` | `state.py` | Continuous features are state internals. |
| `source_events.py` | `outcome.py` | Source event materialization/outcomes route through outcome for the current pipe. |
| `history.py` | `outcome.py` | Kline path history and realized transition rows are outcome products, not workflow APIs. |
| `decisions.py` | `workflow.py` | Current review decision rows are workflow audit output, not a peer product module. |
| former `frames.py` | removed | Observation and outcome implementations now live in state.py/outcome.py. |
| feasibility projection helpers | `feasibility.py` | Feasibility owns candidate/source/history feasibility projections. |

`pipeline.py` is intentionally not a target module yet. Add it only if a concrete typed
`ScanProducts`/`ScannerProducts` object is introduced and consumed by more than one
boundary. Moving `_build_diagnostic_frames` into `pipeline.py` without a stable product
contract would only rename the diagnostics monolith.

`contracts.py` is intentionally absent. Scanner-local contracts such as `ReportInputs`,
`PotentialArtifacts`, `SourceStateRow`, and `TransitionPattern` live in the package root
(`qooi.scanner`) together with shared expression helpers. Avoid adding `_utils.py`,
`common.py`, or another contracts module unless a future public surface genuinely spans
multiple packages.

## Product data pipe

The scanner pipe is:

```text
workflow.run(config)
  → load bars/source context/state bundles/transitions
  → ReportInputs
  → state product
  → outcome product
  → evidence product
  → rank product
  → feasibility product
  → diagnostics/report outputs
```

Detailed product flow:

```text
state product
  owns known-at-close rows only
  current sources: state.KlineClassifier, state.extract_continuous_features,
                   source known-at-close rows, state.potential_observation_frame
  target output: potential_observations

outcome product
  owns future/path labels and source outcomes
  current sources: outcome.kline_path_history_frame, outcome.source_events_frame,
                   outcome source-outcome rows, outcome.potential_outcome_frame
  target output: potential_outcomes / realized_transitions / kline_path_history

evidence product
  ladder.py or tailrun/ + tailtree/
  input: observations + outcomes
  output: potential_evidence

rank product
  rank.candidate_evidence_frame
  rank.rank_candidate_evidence
  rank.candidate_horizon_consistency_frame
  output: candidate_evidence, candidate_rank, candidate_horizon_consistency

feasibility product
  owner: feasibility.py
  current owner: feasibility.py
  output: candidate_feasibility, source/history/watchlist feasibility rows

diagnostics/report outputs
  diagnostics.py writes CSV artifacts and handles stale diagnostic cleanup
  report.py renders prepared frames; it should not recover schema or perform business joins
```

Workflow should not import or call `outcome.py`, `rank.py`, `tailrun/`,
`ladder.py`, or feasibility internals directly. Workflow owns outer lifecycle only; the
scanner data pipe remains below the workflow boundary.

## Dependency direction

```text
workflow
  → config, exchange/source requests, review audit/transitions, diagnostics, report
  → not state/outcome/rank/tailrun/ladder/feasibility internals

state target
  → known-at-close OHLCV/source features only
  → no outcomes, no future returns

outcome target
  → kline path/source outcome/realized transition rows
  → may use future/path data
  → no evidence-path internals

ladder
  → fixed categorical evidence path
  → does not import tailtree

tailtree
  → LightGBM + scipy only inside the tailtree path
  → model/statistical code only

tailrun
  → tailtree lifecycle + artifact persistence + run summaries

rank
  → candidate evidence, candidate-rank rows, horizon-consistency panel
  → enforces candidate evidence pipe invariants such as `outcome_horizon`

feasibility target
  → source/history/watchlist/candidate feasibility projections
  → consumes ranked candidates; does not posterior-check rank schema columns

diagnostics
  → artifact write/read orchestration and diagnostic frame assembly
  → not the long-term owner of state/outcome/evidence/rank/feasibility computation
```

Forbidden:

- `qooi.scanner` must not import executor/basket/recovery/live-trading modules.
- `qooi.scanner` should not import `qooi.strategies`; scanner emits research/review
  data products and strategies may consume promoted signals later.
- Known-at-close state code must not use future return/outcome columns.
- `ladder.py` and `tailtree/` must not cross-import each other as evidence paths.
- Timestamp-only joins are forbidden in source-derived feature construction.
- Do not introduce `pipeline.py` or one-file directory packages unless the graph has a
  real typed product boundary that needs them.
- Do not introduce a candidate-specific package as the target shape for feasibility; `feasibility.py` owns it.

## Scanner config workflow

`PotentialConfig` is the potential workflow's single config entry: one TOML parse
object for one scanner run. It is allowed to embed every section needed by this
workflow, but it must not become a runtime god-object. Section semantics belong
to the behavior/module that interprets them; `workflow.py` composes sections into
small request/context objects at call boundaries.

```text
PotentialConfig                                  # scanner config entry
  ├─ output/universe/bar/timeframes/days/refresh_mode/fetch_concurrency
  │    -> workflow/load_bars and exchange.store requests
  ├─ source: SourceConfig                         # source-owned shape target
  │    -> workflow builds sources.context.SourceContextRequest using root refresh_mode
  ├─ transition: TransitionConfig                 # transition-owned behavior
  │    -> transitions/history/evidence target windows
  ├─ evidence: EvidenceConfig                     # evidence dispatch behavior
  │    -> diagnostics evidence dispatch
  │       └─ tailtree: TailtreeConfig -> tailrun lifecycle
  ├─ review: ReviewConfig                         # decision-review behavior
  │    -> decisions only
  └─ profile: ProfileConfig                       # qooi.profiling context
       -> injected diagnostics context
```

Refresh-mode ownership is singular:

```text
config.refresh_mode  # bars and source context materialization for this scan
```

`SourceConfig` must not define another `refresh_mode`. A repeated nested refresh field is
ergonomic debt because `[potential.source].refresh_mode = "cache_only"` reads like a
whole-scan cache-only request while bars still refresh from the root mode. If bars and
source context need different cadences, split workflow commands or use distinct config
files rather than repeating the same field name under multiple sections.

No module should infer refresh behavior by checking several locations. No compatibility
aliases should preserve old config shapes once callers are updated.

Boundary target:

```text
workflow.py owns config composition, not section semantics.
sources.context owns SourceContextRequest.
exchange.store owns HistoryRefreshRequest.
qooi.profiling owns ProfileContext.
```

This keeps the ergonomic monolithic config entry while avoiding full-root config
leakage into packages.

## Tailtree training integrity before HPO

Tailtree improvement starts with label, feature, and artifact integrity, not parameter search. HPO is valid only after the current run has numeric tail labels, nonempty features, and time-block validation counts. Random row splits are forbidden for scanner tailtree tuning.

Tailtree label source is explicit:

```text
tail labels require forward_min_return_pct / forward_max_return_pct
source-event outcomes may supply those columns when source events exist
market realized-transition rows currently supply categorical transition outcomes only
```

Historical source-feature scope is narrower than the source-family inventory. The only
source families with consistent history suitable for tailtree training are the
funding-like derivative families:

```text
funding
open_interest
taker_volume
long_short_ratios
```

Books/trades are current-review/liquidity context unless a separate consistent history
contract is implemented. Messages remain optional context until provider-backed history
exists. Tailtree training must not treat inconsistent current-only source context as a
historical feature panel.

Tailtree IO contract:

```text
input feature row  = known-at-close market state + consistent derivative-source state at decision_bar_close_ms
label row          = future excursion over the configured horizon, keyed by symbol + decision_bar_close_ms
model output       = per-direction leaf assignment and per-leaf tail evidence, not a trade signal
candidate output   = inspection/rank evidence rows after matching current observations to historical leaves
```

Market excursion labels belong to the realized-transition history product. The same
market row that carries terminal categorical transition state should also carry numeric
future path metrics when OHLCV bar columns are available:

```text
outcome.realized_transition_frame(...)
  -> terminal state/context columns
  -> forward_return_pct / forward_min_return_pct / forward_max_return_pct / path_range_pct
  -> time_to_max_bar / time_to_min_bar / close_retention_ratio / path_efficiency
```

`outcome.potential_outcome_frame(...)` composes those market labels with source-event
outcomes. It must not synthesize null forward-return columns when realized transitions
already provide measured excursion labels.

Tailtree training features are selected by data contract, not by an extra manifest or
condition-check layer. A feature family is trainable only when its persisted artifact
contract supplies a consistent historical panel at the decision-bar grain. Ephemeral
families stay out of model training even if their current values are present in the
observation frame.

Extendability comes from the source/artifact boundary:

```text
new provider/API
  -> persisted artifact with event-time/known-at-time semantics
  -> historical panel aligned to decision_bar_close_ms
  -> source family is classified as persistent by data contract
  -> tailtree training feature list may include its columns
```

If a family only has latest/current snapshots, sparse one-off events, or inconsistent
lookback, it remains an ephemeral review/cost/feasibility input and is not a tailtree
training feature.

Fixed-horizon labels are the first implemented data product, not the permanent output
limit. Before adding more horizons, the fixed-horizon row must expose path-shape
diagnostics that distinguish touch, continuation, exhaustion, and two-sided volatility:

```text
outcome_horizon
forward_return_pct
forward_min_return_pct
forward_max_return_pct
path_range_pct
time_to_max_bar
time_to_min_bar
close_retention_ratio
post_max_drawdown_pct
post_min_rebound_pct
path_efficiency
```

The base `tail_up` / `tail_down` labels remain thresholded excursion labels. Path-shape
columns are numeric diagnostics beside those labels, not replacement qualitative labels.

Tailtree objective policy is severity/utility-first, not whole-market probability
classification. Market return mass is lumpy and profit concentrates in extreme behavior,
so ordinary all-row binary probability training is not the default model target.

Tailtree code is directory-owned rather than prefix-owned. The public API remains
`qooi.scanner.tailtree`, but implementation is split by data-product boundary:

```text
qooi/scanner/tailtree/
  __init__.py   # public exports only
  model.py      # labels, training frame, LightGBM/GPD model
  evidence.py   # leaf and score-bucket evidence products
```

Do not add more `_tailtree_*` helper sprawl to scanner-level modules when behavior belongs
to one of these products. Prefer typed products and direct field access over ad-hoc
`row.get`, `getattr`, or column-existence inference outside the product boundary.

```text
training population   = tail rows only
validation population = all rows with known outcomes
selection target      = concentrated extreme-event utility, not smooth average accuracy
```

Objective strategies are named instances, not boolean flags:

```text
tail_severity_gpd      # baseline: current GPD xi surrogate over raw exceedance
tail_utility_quantile  # tail-only utility target; LightGBM quantile loss over log1p(utility)
```

Objective strategies dispatch both the training target and the evidence bucket. Do not
train a boosted ensemble and then report only one residual tree leaf as if it were the
whole model state.

```text
tail_severity_gpd
  target_basis    = raw_exceedance
  model_family    = shallow/interpretable leaf evidence
  evidence_bucket = leaf_id
  distribution    = raw exceedance GPD

tail_utility_quantile
  target_basis    = path-constrained tail utility
  model_family    = boosted ensemble score
  evidence_bucket = score_bucket
  distribution    = raw exceedance GPD + utility moments
```

The `score_bucket` artifact is separate from leaf evidence. It is keyed by horizon,
direction, and score bucket, not by `leaf_id`, so artifact schemas remain explicit and
old leaf evidence is not reinterpreted.

`tail_utility_quantile` constrains raw excursion severity with path quality:

```text
up utility   = max(forward_max_return_pct - threshold_pct, 0)
             × retention_score × path_efficiency_score × speed_score
             - post_max_drawdown_penalty

down utility = max(abs(forward_min_return_pct) - threshold_pct, 0)
             × retention_score × path_efficiency_score × speed_score
             - post_min_rebound_penalty
```

The trained tree may use utility targets, while GPD `xi`/`sigma` remain leaf-level
extreme-value evidence. GPD parameters are descriptive tail/utility descriptors under
market-regime dependence, not a complete iid market probability model.

Raw exceedance and engineered utility remain separate distribution bases. `X ~ GPD`
is theoretically cleaner when `X` is raw threshold exceedance. Utility can be audited
with mean/p90 and may later get its own utility-exceedance GPD, but the current utility
objective does not overload raw `gpd_shape_xi` / `gpd_scale_sigma` as utility-GPD truth.

### Tailtree selection-efficiency architecture

Tailtree architecture optimizes **selection ability per unit of data/model cost**, not a
single global classifier score. Selection ability means the model can concentrate scarce
future tail utility into a smaller current candidate set while exposing how much breadth,
stability, and cost the concentration required.

The objective comparison from the 2026-06-15 bounded universe benchmark is a design
driver, not an architecture authority. On the same universe path (`symbols == ()`, OKX
discovery, `scan_budget=20`), `tail_utility_quantile` selected more realized tail rows
and much higher utility, while `tail_severity_gpd` kept higher sparse lift. The design
therefore keeps both as objective instances with different selection roles:

| Role | Preferred objective | Selection ability measured by | Efficiency pressure |
|---|---|---|---|
| Sparse extreme concentration | `tail_severity_gpd` | lift, selected tail rate, max lift | small selected set, interpretable leaf evidence |
| Utility-ranked breadth | `tail_utility_quantile` | selected utility mean/p90, selected tail count | more candidates accepted, score-bucket evidence |
| Horizon dispatch | per `(horizon, direction)` winner | horizon-local lift and utility | avoid averaging incompatible horizons |
| Tail-shape audit | GPD descriptors | `xi`, `sigma`, exceedance count | descriptive only, not a trade trigger |

Selection surfaces must report both **ability** and **efficiency** columns:

```text
ability columns:
  valid_tail_lift
  valid_selected_tail_rate
  valid_selected_tail_count
  valid_selected_utility_mean
  valid_selected_utility_p90
  horizon_count
  strong_horizon_count
  direction_consistency_score

efficiency columns:
  selected_observation_rate        # selected_obs / valid_observation_count
  selected_tail_per_1k_obs         # selected tails per 1000 selected observations
  utility_per_selected_obs         # selected utility sum / selected_obs
  train_exceedance_per_feature     # train_exceedance_count / feature_count
  trained_tree_count
  selected_leaf_or_bucket_count
  fit_seconds / score_seconds      # profiler-owned, when available
```

The candidate selector should evaluate objectives on normalized budgets before promotion:

```text
universe snapshot -> observation/outcome frames
  -> objective × horizon × direction training
  -> evidence rows
  -> candidate score rows
  -> budget replay: top_k, top_pct, min_score_gate
  -> selection-efficiency summary
  -> canonical candidate table
```

Budget replay is the key design correction. A wide quantile gate and a sparse GPD gate
cannot be compared only by raw selected counts. Every objective/horizon/direction row must
be replayable under the same budget families:

```text
top_k      = 1, 3, 5, 10
top_pct    = 1%, 5%, 10%, 20%
score_gate = objective-native threshold after calibration
```

Promotion should prefer rows that survive more than one budget family. A candidate that
only looks good because its objective admitted a very wide set is inspection evidence, not
a promoted high-confidence selection.

HPO is a deterministic search over typed objective instances, not an unbounded optimizer
over the whole workflow:

```text
HPO grain = universe_snapshot × objective × outcome_horizon × direction × budget_family

typed instance fields:
  target_basis              # raw_exceedance | path_utility
  lightgbm_objective        # custom GPD surrogate | quantile
  evidence_bucket           # leaf_id | score_bucket
  selection_budget_family   # top_k | top_pct | score_gate
  num_leaves
  min_data_in_leaf
  learning_rate
  num_iterations
  early_stopping_rounds
```

Required HPO artifact:

```text
tailtree-selection-efficiency.csv
```

Required row grain:

```text
model_tag × objective × outcome_horizon × tree_direction × budget_family × budget_value
```

Required summary columns:

```text
universe_snapshot_id
eligible_symbol_count
selected_symbol_count
observation_row_count
feature_count
train_exceedance_count
valid_observation_count
valid_tail_count
valid_tail_rate
selected_observation_count
selected_observation_rate
selected_tail_count
selected_tail_rate
selected_tail_per_1k_obs
valid_tail_lift
selected_utility_mean
selected_utility_p90
utility_per_selected_obs
trained_tree_count
selected_bucket_or_leaf_count
fit_seconds
score_seconds
```

Acceptance rule:

```text
promotion_score =
  lift_component
  + utility_component
  + horizon_consistency_component
  - breadth_penalty
  - conflict_penalty
  - data_cost_penalty
```

Where each component is numeric and auditable. Do not encode promotion as qualitative
labels such as "good" or "fresh"; use thresholds inline in the report legend.

Universe reproducibility is part of model efficiency. Current config has
`potential.universe = "research"`, but effective universe selection is the branch where
`symbols == ()` and `resolve_universe(config)` calls OKX discovery, then applies
`transition.scan_budget`. Durable HPO must persist either:

```text
universe_snapshot_id + discovery frame
```

or a typed universe provider object. Otherwise objective comparisons mix model changes
with live listing/ticker drift.

Feature efficiency is evaluated separately from objective efficiency:

```text
state/path only
state/path + persistent derivative source features
derivative source features only
```

Do not tune source-family HPO from a run where source freshness is structurally missing.
That run is a valid kline/state/path benchmark, not a source-feature benchmark.

The model may train separate artifacts per horizon/model_tag first. Multi-horizon shared
models are a later optimization after the per-horizon label quality is measurable.

Candidate-level multi-horizon use is a calibrated consistency panel, not raw score
averaging. Raw LightGBM scores and leaf ids are horizon-local; averaging them would mix
different label distributions and can cancel high-confidence long/short evidence. The
supported panel consumes calibrated candidate rows after per-horizon evidence matching:

```text
input grain   = symbol × outcome_horizon × tree_direction candidate rows
panel grain   = symbol × tree_direction
signal        = horizon agreement/counts and max-strength consistency
non-goal      = mean(raw_score) or mean(rank_score) across opposite directions
```

The panel reports counts and extrema rather than mean scores:

```text
horizon_count
strong_horizon_count
best_rank_score
best_tail_lift
best_tail_utility_score
direction_consistency_score
opposite_direction_count
opposite_direction_best_rank_score
conflict_penalty_score
```

This keeps independent horizon specialists while making h6/h12/h24 agreement visible
without pretending their raw model outputs live on a shared scale.

Therefore a tailtree run with zero non-null forward excursion columns is not a model failure and not an HPO target. It is an outcome-availability state that must be surfaced.

`tailrun/` owns a run-complete artifact contract:

```text
one model_dir/model_tag directory represents exactly the latest train run for that tag
directional tree/evidence artifacts from previous runs must not survive a zero-train run
metadata and evidence files must not describe different runs
```

Every train run writes a structural summary, even when no tree trains:

```text
tailtree-run-summary.csv
```

Summary grain:

```text
run       # data, feature, label, and artifact counts
up        # direction-level trainability and evidence counts
down      # direction-level trainability and evidence counts
```

Required numeric columns include:

```text
observation_row_count
outcome_row_count
source_event_row_count
source_outcome_row_count
realized_transition_row_count
feature_count
categorical_feature_count
continuous_feature_count
forward_return_nonnull_count
forward_min_return_nonnull_count
forward_max_return_nonnull_count
path_range_nonnull_count
time_to_max_nonnull_count
time_to_min_nonnull_count
retention_nonnull_count
path_efficiency_nonnull_count
tail_utility_mean
tail_utility_p90
valid_selected_utility_mean
valid_selected_utility_p90
train_tail_count
valid_observation_count
valid_tail_count
valid_tail_rate
valid_selected_observation_count
valid_selected_tail_count
valid_selected_tail_rate
valid_tail_lift
tail_count
tail_rate
train_observation_count
train_exceedance_count
min_exceedance_required
trainable_flag
trained_tree_count
written_model_file_count
written_evidence_file_count
removed_stale_file_count
```

Tailtree code layout stays lean:

```text
tailrun/ = tailtree lifecycle boundary, artifact names, summary rows, validation quality
tailtree/ = model/training/evidence math split by data product
outcome.py = market path/outcome row construction
```

Do not add a generic `_utils` module for this path. If a cross-scanner IO primitive
already exists in `src/qooi/core`, use it directly; otherwise keep tailtree artifact
file names local to `tailrun/` because they are part of the tailtree artifact
contract, not a generic IO abstraction. Typing imports are regular imports, not
`TYPE_CHECKING`-guarded aliases.

Multi-horizon starts as an explicit row contract:

```text
[potential.evidence.tailtree]
outcome_horizon = [6, 12, 24]
```

`outcome_horizon` accepts either one integer or a list of integers. Realized-transition
and tailtree summary rows carry `outcome_horizon`. Tailtree training writes one
horizon-suffixed artifact set per configured horizon so labels are never mixed
across horizons.

Multi-horizon is not a training-only feature. It changes the tailtree evidence and
candidate grain:

```text
symbol × decision_bar_close_ms × outcome_horizon × direction
```

Each configured horizon trains its own up/down model pair first. Current observations
are scored against every trained `(outcome_horizon, direction)` model, and ranking
receives horizon-dimensional candidate rows. `train` and `load_predict` must have the
same candidate/rank/report grain; a frozen multi-horizon model tag that scores only
one horizon is invalid architecture noise.

Per-horizon artifact metadata describes one artifact, not the config set:

```text
outcome_horizon = 6        # artifact identity, singular
```

Do not use `horizon_bars` in tailtree artifact metadata; `outcome_horizon` is the
label/model/report horizon measured in bars.

Load-predict feedback contract:

```text
configured horizons
  == loaded metadata horizons
  == loaded model horizons
  == loaded evidence horizons
```

For each configured horizon, load-predict validates the metadata, every available
up/down model file, the matching evidence CSV, and evidence row `outcome_horizon`.
Missing or mismatched horizon artifacts fail loudly before candidate ranking; there is
no single-horizon fallback and no synthetic horizon such as `0`.

HPO baseline uses no new dependency first: deterministic small grids over named objective
and training profiles, written as CSV feedback. `optuna` may be added later as an optional
`hpo` dependency only after the CSV validation score is stable.

HPO may be added as deterministic blocked/walk-forward search only when:

```text
forward_min/max non-null count > 0
tail_count >= min_exceedance_required
feature_count > 0
validation tail count > 0
```

The HPO score optimizes extreme utility concentration:

```text
hpo_score = lift + count + utility + concentration/stability penalties
```

Until then, the correct output is an honest baseline summary, not tuned parameters.

## Lean reduction target

The current scanner still has large report and diagnostics modules. The reduction target is
not more files by default; it is fewer public peer APIs and clearer product ownership.

| Product surface | Target owner | Does not own |
|---|---|---|
| outer run lifecycle | `workflow.py` | state/outcome/evidence/rank internals |
| known-at-close state/observation | `state.py` target | future returns, labels, evidence dispatch |
| future/path/source outcomes | `outcome.py` target | known-at-close feature extraction, rank/report rendering |
| categorical evidence | `ladder.py` | tailtree internals, feasibility, report joins |
| tree evidence lifecycle | `tailrun/` + `tailtree/` | source refresh, report rendering, random-split HPO |
| candidate evidence/rank/consistency | `rank.py` | source/history feasibility projection |
| source/history/candidate feasibility | `feasibility.py` target | candidate rank scoring, renderer formatting |
| diagnostics artifact IO | `diagnostics.py` | long-term state/outcome/evidence/rank/feasibility computation |
| markdown rendering | `report.py` | CSV read-back, type recovery, business joins |

Current transitional debt to remove deliberately:

```text
state.KlineClassifier + state.extract_continuous_features + state.potential_observation_frame
  -> state.py

outcome.kline_path_history_frame + outcome.realized_transition_frame + outcome.source_events_frame
  + outcome.source_outcomes_frame + outcome.potential_outcome_frame -> outcome.py

candidate/source/history feasibility helpers
  -> feasibility.py

diagnostics._build_diagnostic_frames / _run_pipeline
  -> stays in diagnostics until a real typed ScanProducts object has multiple consumers
```

Do not create `pipeline.py`, `selection.py`, `_utils.py`, or one-file directory packages
as line-count relief. Split only when the destination is a named product with a stable
grain, schema, owner, and tests.

Default report surface:

```text
Scan Scope
Data Health Summary
Candidate Selection
Path-specific Evidence Summary
Caveats
```

`Decision Rule Audit` is not a default peer section. It may remain as a CSV artifact or
explicit appendix mode, because it is a rule-order trace rather than the canonical ranked
candidate answer.

## Cross-package boundaries

The scanner intentionally consumes lower-level data products but must not become a
strategy or executor layer.

| Boundary | Allowed direction | Rule |
|---|---|---|
| `exchange` ↔ `scanner` | `scanner.workflow → exchange` | workflow may fetch/cache/discover; model/evidence modules consume DataFrames only |
| `sources` ↔ `scanner` | `scanner → sources.context` | source package owns acquisition/schema; scanner owns known-at-close event products |
| `scanner` ↔ `strategies` | `strategies → scanner outputs` | scanner should not import strategy semantics or allocation logic |
| `research` ↔ `scanner` | `research → scanner artifacts` | research may inspect outputs; it should not depend on scanner internals |
| `executor/basket` ↔ `scanner` | no scanner import | scanner diagnostics do not authorize live trading |

Resolved overlap: scanner classifier vocabulary is scanner-owned in `scanner.state`, with
strategies consuming final signal columns rather than supplying scanner state labels.

Source event materialization is scanner-side outcome construction in `scanner.outcome`,
while `qooi.sources` remains the acquisition/source-context package.

## Composable pipeflow, not monolithic data bag

The scanner shares by named product grains, not by passing a global
`MaterializedScannerFrames` object and letting downstream functions probe fields.

Target pipe:

```text
workflow.run(...)
  -> ReportInputs
  -> state product
  -> outcome product
  -> evidence product
  -> rank product
  -> feasibility product
  -> diagnostics/report outputs
```

Current implementation path:

```text
bars + state_frames + source context
  -> state.KlineClassifier / state.extract_continuous_features / state.potential_observation_frame
  -> potential_observations

kline path history + source outcomes + realized transitions
  -> outcome.py internals / outcome.potential_outcome_frame
  -> potential_outcomes

potential_observations + potential_outcomes
  -> ladder.py or tailrun/
  -> potential_evidence

potential_evidence + latest observations
  -> rank.candidate_evidence_frame
  -> rank.rank_candidate_evidence
  -> rank.candidate_horizon_consistency_frame

ranked candidates + source/history/watchlist feasibility
  -> feasibility.py target
  -> candidate_feasibility

prepared frames
  -> diagnostics.write_diagnostic_frames
  -> report.render_report
```

Each stage consumes the smallest named product it needs. Workflow remains high-level and
should not call history/frames/rank/tailrun/ladder/feasibility internals directly.

## Data products and invariants

### `source_availability`

Owner: `qooi.sources.context.load_source_context(...)`; consumed by `qooi.scanner.workflow`, `qooi.scanner.diagnostics`, and `qooi.scanner.report`.

Key:

```text
(symbol, source_family)
```

Invariants:

- source acquisition owns raw-source provenance, artifact schemas, provider capabilities, and frame availability;
- scanner may consume availability but must not infer provider fetch success from source feature nulls;
- frame availability is numeric: rows, latest timestamp, latest age, freshness threshold, provider cap rows, target coverage, capability coverage, fresh bit, stale bit, missing bit, provider-bounded bit, optional-absent bit, usable bit;
- latest fetch status is provenance only and must not overwrite observed frame rows;
- provider-bounded Rubik windows are acceptable for current review when recent rows are fresh and `coverage_capability_pct` is high;
- messages/context absence is optional until a real provider is enabled and must not count as required market-data failure.

### `history_feasibility`

Owner: scanner diagnostics/report from OHLCV cache coverage and configured review thresholds.

Key:

```text
(symbol, bar)
```

Invariants:

- evidence training coverage and current-review coverage are separate lenses;
- a deep evidence horizon may use a long history target, while current review may use a shorter explicit review window;
- report columns should expose both `history_target_coverage_pct` and `review_window_coverage_pct` when the windows differ;
- provider fetch-stop diagnostics remain visible rather than collapsed into a qualitative label.

### `continuous_features`

Owner: `qooi.scanner.state.extract_continuous_features(...)`

Key:

```text
(symbol, timestamp)
```

Invariants:

- one row per `(symbol, timestamp)`;
- all feature values are known at or before `timestamp`;
- cached OHLCV bars use canonical `volume`; old `vol` cache columns are migrated at read time;
- source frames preserve their native event/snapshot timestamp, then materialize known-at-close feature rows with `join_asof(..., by="symbol")`;
- source feature rows include `*_age_ms` columns so ephemeral values are not treated as fresh forever;
- source reducers group and join by `("symbol", "timestamp")` after known-at-close materialization;
- missing source families produce null numeric columns, not dropped decision rows;
- no outcome/future-return columns are allowed.

### `observations`

Owner: `qooi.scanner.state.potential_observation_frame(...)`

Key:

```text
(symbol, decision_bar_close_ms, source_family)
```

Invariants:

- categorical kline/source state columns describe known-at-close context;
- continuous features are joined by `(symbol, decision_bar_close_ms) = (symbol, timestamp)`;
- continuous feature availability is independent of whether source events are present;
- source freshness and source alignment are explicit columns.

Source event timestamps are **not rewritten** into bar-close timestamps. The raw source event/snapshot time remains source-native. The materialized feature row timestamp is the decision close at which that source value is known; source age columns quantify staleness.

### `outcomes`

Owner: `qooi.scanner.outcome.potential_outcome_frame(...)`

Key:

```text
(symbol, decision_bar_close_ms, outcome_horizon)
```

Invariants:

- outcome columns are future diagnostics only;
- missing horizons remain explicit through coverage/artifact diagnostics;
- outcomes do not feed feature construction.

Current tail labels use path extremes over the configured horizon:

```text
up tail:   max(high over next N bars) / close_now - 1 > threshold_pct
down tail: min(low  over next N bars) / close_now - 1 < -threshold_pct
```

This is an **excursion/touch** label. It detects whether price touched an extreme inside the horizon. It does not by itself prove trend persistence, tradability after latency, or close-to-close continuation. Architecture therefore distinguishes three outcome families:

| Family | Columns | Meaning | Status |
|---|---|---|---|
| excursion | `forward_max_return_pct`, `forward_min_return_pct`, `tail_*` | touched upside/downside extreme inside horizon | implemented |
| terminal | `forward_return_pct` | close-to-close result at horizon | implemented |
| path-shape | `time_to_max_bar`, `time_to_min_bar`, `close_retention_ratio`, `post_peak_drawdown_pct`, `path_efficiency` | separates continuation from burst-then-fade/exhaustion | planned |

Promotion-quality evidence should not rely on excursion alone. It should report whether the tail was retained, unwound, or only a transient wick.

## Evidence path contracts

The scanner has one dispatch point, keyed by the nested evidence section:

```toml
[potential.evidence]
kind = "ladder" # or "tailtree"
```

### `LadderResult`

```text
LadderResult
├── evidence: pl.DataFrame
├── candidates: pl.DataFrame
└── ranked: pl.DataFrame
```

The ladder path groups observations by fixed categorical evidence levels and computes probability/information diagnostics.

### `TailtreeResult`

```text
TailtreeResult
├── evidence: pl.DataFrame
├── candidates: pl.DataFrame
├── ranked: pl.DataFrame
├── tree_up: TailTreeModel
└── tree_down: TailTreeModel
```

The tailtree path trains direction-specific LightGBM trees on extreme exceedances and evaluates per-leaf tail evidence against the full observation population.

Leaf selection is auditable rather than silent:

- `selection_mode = "hard_gate"` when leaves pass the configured count/lift/stability gates;
- if no leaf passes, `selection_mode = "best_available"` writes top-ranked leaves with `selected_evidence_level = false` so review can inspect why the gate failed without promoting weak evidence.

Result types are decisive: no `Any | None` tree fields, no variant tuples, and no report/candidate branching on config outside the evidence dispatcher.

## Tailtree / GPD model interface

The tailtree path is an extreme-value evidence path. It does not model the full return distribution and does not replace broad up/down/flat probability diagnostics.

For each direction (`up`, `down`):

1. Label all aligned observation/outcome rows with a tail flag and exceedance magnitude.
2. Train the LightGBM tree on rows where that direction tail occurred.
3. Use a GPD negative-log-likelihood objective over positive exceedance magnitudes.
4. Project all aligned observations through the trained tree to compute per-leaf denominators.
5. Fit/report per-leaf GPD parameters from tail rows only.
6. Compute `tail_rate` and `tail_lift` from all rows assigned to each leaf.

Thus:

| Quantity | Population | Meaning |
|---|---|---|
| `tail_rate` | all rows assigned to the leaf | probability of entering the tail |
| `tail_lift` | all rows assigned to the leaf vs global baseline | concentration of tail frequency |
| `gpd_shape_xi` | tail exceedances in the leaf | tail heaviness/severity shape |
| `gpd_scale_sigma` | tail exceedances in the leaf | exceedance dispersion |

GPD is only for threshold exceedance severity:

```text
up_excess   = forward_max_return_pct - threshold_pct      where tail_up
down_excess = abs(forward_min_return_pct) - threshold_pct where tail_down
```

It is not a whole-return distribution model, not a classifier, and not a candidate ranking substitute by itself.

### Horizon semantics

The configured bar/horizon pair defines what the tree can learn. With `bar = "1H"` and `transition.mae_mfe_horizon = 12`, the implemented tail label means a 12-hour path-extreme touch. Changing the horizon changes the research question:

| Horizon pattern | Interpretation risk |
|---|---|
| short-horizon lift only | burst/scalp behavior; may decay before review |
| long-horizon lift only | slow setup; early rank may look quiet |
| both up/down lift high | volatility regime, not directional conviction |
| high excursion + poor retention | wick/trap/exhaustion rather than continuation |

The durable model contract should therefore support multi-horizon evidence and path-shape diagnostics before treating a candidate as promotion-quality.

### Train/load/predict lifecycle

Tailtree has two distinct lifecycle modes and they should be explicit in config:

| Mode | Reads outcomes? | Trains model? | Writes model? | Candidate purpose |
|---|---:|---:|---:|---|
| `train` | yes | yes | yes | research/evidence refresh |
| `load_predict` | no, except optional diagnostics | no | no | current candidate review from a frozen model |

`[potential.evidence] kind = "tailtree"` chooses the evidence path. A nested tailtree lifecycle section chooses whether this run trains or loads a frozen model:

```toml
[potential.evidence]
kind = "tailtree"

[potential.evidence.tailtree]
lifecycle = "train"          # "train" or "load_predict"
model_dir = "data/output/potential/daily-deep/models"
model_tag = "tailtree-1h-12h-v1"
```

This avoids mixing model fitting, validation, and current prediction in one scan. `load_predict` must fail clearly if the model artifact, feature schema, horizon, threshold, or training metadata does not match the current config.

### Validation semantics

Tailtree training must be evaluated by time, not random row shuffling. The planned validation surface is rolling by `decision_bar_close_ms`:

```text
train_window_1 → validation_window_1
train_window_2 → validation_window_2
...
```

Validation metrics should be numeric and leaf/candidate aligned:

```text
valid_tail_rate
valid_tail_lift
valid_N_tail_exceedances
valid_close_retention_ratio
valid_post_peak_drawdown_pct
leaf_selected_window_count
median_valid_tail_lift
min_valid_tail_lift
valid_lift_decay
```

A leaf should be promoted only when its lift, count, and path-shape quality survive rolling validation windows.

## Candidate review semantics

Candidate output has three different grains. The report must not present all three as competing candidate lists:

| Artifact | Grain | Meaning | Report role |
|---|---:|---|---|
| `candidate-inspection.csv` | symbol × evidence row | every latest symbol assigned to evidence/leaf metrics | debugging/research appendix only |
| `candidate-rank.csv` | symbol × selected direction | ranked evidence matches with source penalties | machine-readable rank detail |
| `candidate-feasibility.csv` | one best row per symbol | rank joined to current review feasibility | canonical report candidate-selection table |
| `watchlist-feasibility.csv` | one row per symbol | history/source reviewability diagnostics | data-health input, not a candidate table |

The top-level report should have one candidate-selection table sourced from `candidate-feasibility.csv` or its in-memory frame. `Review Rows`/decision-rule output is an audit lens and should be demoted to artifact/appendix, not a rank-ordered candidate list. `Data Coverage And Feasibility` should become an aggregate data-health summary, not another symbol table competing with candidate selection.

Candidate feasibility semantics belong to the feasibility product: best-row selection, promotion gates, source/history blocker codes, and stable numeric schema. Report rendering belongs to `report.py`; artifact writing belongs to `diagnostics.py`. Keep candidate/source/history feasibility in `feasibility.py`; do not create `selection.py` as a forwarding layer.

Freshness, capability, and tradability are numeric inputs, not manual labels:

```text
source_age_ms
source_age_hours
bar_age_bars
required_fresh_source_count
required_stale_source_count
required_missing_source_count
provider_bounded_source_count
optional_absent_source_count
coverage_target_pct_min
coverage_capability_pct_min
market_data_fresh_rate
source_penalty_score
spread_bps
spread_percentile_30d
depth_percentile_30d
estimated_slippage_bps_for_size
expected_edge_bps
cost_adjusted_score
```

Source penalty policy:

```text
missing required market family -> high penalty
stale required market family -> medium penalty
provider-bounded but fresh -> zero or low penalty
optional absent context family -> zero main-rank penalty
fetch failed but frame fresh -> zero or low penalty
```

Static slippage thresholds are acceptable only as hard sanity guards. Promotion should prefer data-derived, symbol-relative, size-aware cost features and penalize cost against expected edge.

## Diagnostics and report runtime boundary

Measured cache-only deep scan shows that Markdown rendering is not the slow stage:

```text
build_diagnostic_frames      ~8.2s
write_diagnostic_frames      ~0.1s
render_report                ~0.1s
```

Therefore "report construction" means diagnostic-frame construction, not Markdown
rendering. The dominant current hotpath is:

```text
outcome.realized_transition_frame      ~4.9s
history.kline_path_history_frame       ~1.8s
tailtree/evidence pipeline             ~0.7s
state.potential_observation_frame     ~0.5s
```

Optimization must target these frame builders before Markdown table helpers or CSV
writers. `write_diagnostics(inputs)` should be treated as two products:

```text
build_diagnostic_frames(inputs) -> DiagnosticFrames  # expensive computation
write_diagnostic_frames(frames, artifacts) -> None   # cheap artifact IO
```

## Report projection boundary

`diagnostics.py` owns semantic projection. `report.py` owns presentation only.

Allowed in `diagnostics.py`:

```text
rank + feasibility joins
best row per symbol selection
candidate blocker/reason derivation
numeric schema validation
empty-frame schema construction
```

Allowed in `report.py`:

```text
section composition
column ordering
formatting float | int | str | None values from typed rows
Markdown table rendering
```

Forbidden in `report.py`:

```text
row.get(...) over dict[str, object]
getattr(...) probing
Any/object recovery paths
float(str(value)) as type inference
business rules for source/history blocker semantics
rank/feasibility joins
```

Report sections should receive decisive frame schemas from `DiagnosticFrames`. The renderer should fail fast when projection schemas are wrong; it should not infer missing semantics with attribute checks, opaque row dictionaries, or CSV read-back conversions inside the same run.

## Artifact boundaries

Materialization artifacts:

```text
diagnostics/continuous-features.csv
diagnostics/potential-observations.csv
diagnostics/potential-outcomes.csv
```

Evidence artifacts:

```text
ladder:   potential-evidence-summary.csv, potential-evidence-selected.csv
tailtree: tail-tree-up.json, tail-tree-down.json,
          potential-leaf-evidence-*.csv, potential-leaves-selected-*.csv
```

Review artifacts:

```text
candidate-inspection.csv    # all latest evidence/leaf assignments, diagnostic surface
candidate-rank.csv          # symbol × selected direction rank detail
candidate-feasibility.csv   # one best ranked row per symbol joined to feasibility
report.md                   # candidate selection + data health + evidence/model diagnostics
```

`candidate-feasibility.csv` is the only top-level candidate table source for the report. `candidate-rank.csv` remains available for per-direction detail, and `watchlist-feasibility.csv` remains available for source/history audit joins.

A model iteration should be able to reuse materialized artifacts; code should not skip stages ad hoc to finish a scan. Reduce config when necessary, or split materialization from evidence/review in a later workflow command.

## Verification boundary

For scanner migration/refactor slices, verify in this order and do not run slow scanner
script execution unless explicitly requested:

```bash
uv run ruff check src tests
uv run ty check
uv run pytest tests/ -q
git diff --check HEAD
```

The module-boundary tests own removal checks for legacy/transitional scanner modules.
Architecture and graph docs must not document removed modules as active APIs.
