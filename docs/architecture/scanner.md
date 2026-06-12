# Scanner Architecture

## Purpose

The scanner is a deterministic research workflow for finding symbols whose known-at-close state vectors materially change the probability or severity of future extreme behavior. It emits research diagnostics and review candidates only; it does not authorize live trading, allocation, executor actions, or wallet operations.

## Module layout

```text
src/qooi/scanner/
├── workflow.py       # CLI orchestration: config, universe, fetch/cache, staged calls
├── __init__.py       # scanner-local dataclasses/protocols + shared Polars expr helpers
├── classifiers.py    # OHLCV → categorical kline state labels
├── features.py       # bars/source frames → continuous features keyed by symbol,timestamp
├── transitions.py    # n-gram transition/path discovery for the decision lens
├── decisions.py      # current-state/source-state review lens
├── history.py        # kline path history and realized transition frames
├── source_events.py  # source event/outcome frames from source context
├── frames.py         # shared observations/outcomes for evidence paths
├── ladder.py         # fixed categorical evidence ladder
├── tailtree.py       # LightGBM + GPD tail evidence path; optional deps
├── tailrun.py        # tailtree train/load_predict lifecycle and model artifacts
├── rank.py           # candidate-inspection and candidate-rank rows
├── diagnostics.py    # artifact writing and one evidence-path dispatch
└── report.py         # markdown renderer from computed artifacts
```

`contracts.py` is intentionally absent. Scanner-local contracts such as `ReportInputs`,
`PotentialArtifacts`, `SourceStateRow`, and `TransitionPattern` live in the package root
(`qooi.scanner`) together with shared expression helpers. Avoid adding `_utils.py`,
`common.py`, or another contracts module unless a future public surface genuinely spans
multiple packages.

### Current resolved layout

The compatibility shims `contracts.py`, `evidence.py`, and `candidates.py` are removed. The resolved graph is:

```text
src/qooi/scanner/
├── __init__.py     # shared scanner contracts + small expression helpers
├── workflow.py     # config + top-level run only
├── diagnostics.py  # diagnostic artifact assembly/writes only
├── frames.py       # observation/outcome frames shared by evidence paths
├── features.py     # known-at-close continuous features
├── events.py       # scanner event/timeliness frames derived from source context
├── ladder.py       # fixed categorical evidence ladder
├── tailtree.py     # LightGBM/GPD model, labels, leaf evidence, selection
├── tailrun.py      # tailtree train/load_predict lifecycle and model artifacts
├── rank.py         # candidate-inspection and promoted candidate-rank rows
├── costs.py        # spread/depth/slippage/cost-adjusted score
├── validation.py   # rolling time-block leaf/candidate validation
├── decisions.py    # current per-symbol review decision lens
├── transitions.py  # transition patterns/insights
├── history.py      # kline path history/realized transitions
├── classifiers.py  # scanner-owned deterministic state classifiers
└── report.py       # report sections/rendering
```

This layout keeps each module aligned to one data product or behavior. Planned modules such as `costs.py` and `validation.py` remain future work; removed compatibility modules must not be reintroduced.

## Dependency direction

```text
workflow
  → features, frames, events, decisions, transitions, diagnostics, report

diagnostics
  → ladder | tailrun
  → rank
  → costs, validation when those products exist
  → owns the single evidence dispatch point

features
  → no outcomes, no future returns
  → emits known-at-close continuous features only

frames
  → shared observation/outcome frames
  → no evidence-path internals
  → no model/lifecycle code

ladder
  → fixed categorical evidence path
  → consumes frames only
  → does not import tailtree

tailtree
  → LightGBM + scipy only inside the tailtree path
  → model/statistical code only

tailrun
  → tailtree + artifact persistence + lifecycle validation

rank
  → candidate-inspection rows
  → candidate-rank rows from numeric evidence columns
```

Forbidden:

- `qooi.scanner` must not import executor/basket/recovery/live-trading modules.
- `qooi.scanner` should not import `qooi.strategies`; scanner emits research/review
  data products and strategies may consume promoted signals later.
- `features.py` must not use future return/outcome columns.
- `ladder.py` and `tailtree.py` must not cross-import each other as evidence paths.
- Timestamp-only joins are forbidden in source-derived feature construction.

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

Current overlap to remove: `scanner.classifiers` depends on strategy semantics. The
scanner classifier vocabulary should become scanner-owned, with strategies consuming
final signal columns rather than supplying scanner state labels.

Naming cleanup: `scanner.source_events` is scanner-side event materialization, not
external source acquisition. A later one-word rename to `scanner.events` would make the
boundary clearer while `qooi.sources` remains the acquisition/source-context package.

## Composable pipeflow, not monolithic data bag

The scanner shares by **behavior and data product**, not by passing a global `MaterializedScannerFrames` object.

```text
bars + state_frames
  → qooi.scanner.features.extract_continuous_features
  → continuous_features(symbol,timestamp,...)

kline_history + source_events + continuous_features
  → qooi.scanner.frames.potential_observation_frame
  → observations(symbol,decision_bar_close_ms,...)

observations + source_outcomes + realized_transitions
  → qooi.scanner.frames.potential_outcome_frame
  → outcomes(symbol,decision_bar_close_ms,horizon,...)

observations + outcomes
  → one evidence dispatch
  → LadderResult | TailtreeResult

result.evidence + latest observations
  → rank.candidate_evidence_frame + rank.rank_candidate_evidence + report
```

Each stage consumes the smallest named data products it needs. No downstream function should accept the whole workflow state and select fields opportunistically.

## Data products and invariants

### `continuous_features`

Owner: `qooi.scanner.features.extract_continuous_features(...)`

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

Owner: `qooi.scanner.frames.potential_observation_frame(...)`

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

Owner: `qooi.scanner.frames.potential_outcome_frame(...)`

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

The scanner has one dispatch point, keyed by:

```toml
evidence = "ladder" | "tailtree"
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

Candidate output has two different purposes and should remain separated:

| Artifact | Meaning | Promotion role |
|---|---|---|
| `candidate-inspection.csv` | every latest symbol assigned to a leaf with evidence metrics | debugging/research surface |
| `candidate-rank.csv` | selected-leaf-only rows passing evidence/freshness/cost gates | promoted review list |

Current implementation ranks leaf-matched candidates. The desired architecture is stricter: promoted rank rows come only from selected evidence leaves, while all assignments remain available for inspection.

Freshness and tradability should be numeric inputs, not manual labels only:

```text
source_age_ms
source_age_hours
bar_age_bars
fresh_source_count
stale_source_count
spread_bps
spread_percentile_30d
depth_percentile_30d
estimated_slippage_bps_for_size
expected_edge_bps
cost_adjusted_score
```

Static slippage thresholds are acceptable only as hard sanity guards. Promotion should prefer data-derived, symbol-relative, size-aware cost features and penalize cost against expected edge.

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
candidate-inspection.csv   # all latest evidence/leaf assignments, diagnostic surface
candidate-rank.csv         # one row per promoted symbol-direction after gates
report.md
```

A model iteration should be able to reuse materialized artifacts; code should not skip stages ad hoc to finish a scan. Reduce config when necessary, or split materialization from evidence/review in a later workflow command.
