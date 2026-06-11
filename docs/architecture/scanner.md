# Scanner Architecture

## Purpose

The scanner is a deterministic research workflow for finding symbols whose known-at-close state vectors materially change the probability or severity of future extreme behavior. It emits research diagnostics and review candidates only; it does not authorize live trading, allocation, executor actions, or wallet operations.

## Module layout

```text
src/qooi/scanner/
├── workflow.py       # CLI orchestration: config, universe, fetch/cache, staged calls
├── contracts.py      # dataclasses/protocols shared by scanner modules
├── classifiers.py    # OHLCV → categorical kline state labels
├── features.py       # bars/source frames → continuous features keyed by symbol,timestamp
├── transitions.py    # n-gram transition/path discovery for the decision lens
├── decisions.py      # current-state/source-state review lens
├── history.py        # kline path history and realized transition frames
├── source_events.py  # source event/outcome frames from source context
├── evidence.py       # shared observations/outcomes + fixed ladder evidence path
├── tailtree.py       # LightGBM + GPD tail evidence path; optional deps
├── candidates.py     # candidate matching and ranking
├── diagnostics.py    # artifact writing and one evidence-path dispatch
├── report.py         # markdown renderer from computed artifacts
└── __init__.py       # shared scalar expressions/utilities
```

## Dependency direction

```text
workflow
  → classifiers, features, decisions, transitions, source_events,
    evidence, diagnostics, report

diagnostics
  → evidence, tailtree, candidates
  → owns the single evidence dispatch point

features
  → no outcomes, no future returns
  → emits known-at-close continuous features only

evidence
  → shared observation/outcome frames
  → fixed ladder evidence path
  → does not import tailtree

tailtree
  → LightGBM + scipy only inside the tailtree path
  → does not import evidence path internals except shared frames by schema

candidates
  → matches latest observations to path result data
  → ranks candidates from numeric evidence columns
```

Forbidden:

- `qooi.scanner` must not import executor/basket/recovery/live-trading modules.
- `features.py` must not use future return/outcome columns.
- `evidence.py` and `tailtree.py` must not cross-import each other as evidence paths.
- Timestamp-only joins are forbidden in source-derived feature construction.

## Composable pipeflow, not monolithic data bag

The scanner shares by **behavior and data product**, not by passing a global `MaterializedScannerFrames` object.

```text
bars + state_frames
  → qooi.scanner.features.extract_continuous_features
  → continuous_features(symbol,timestamp,...)

kline_history + source_events + continuous_features
  → qooi.scanner.evidence.potential_observation_frame
  → observations(symbol,decision_bar_close_ms,...)

observations + source_outcomes + realized_transitions
  → qooi.scanner.evidence.potential_outcome_frame
  → outcomes(symbol,decision_bar_close_ms,horizon,...)

observations + outcomes
  → one evidence dispatch
  → LadderResult | TailtreeResult

result.evidence + latest observations
  → candidates + ranked + report
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

Owner: `qooi.scanner.evidence.potential_observation_frame(...)`

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

Owner: `qooi.scanner.evidence.potential_outcome_frame(...)`

Key:

```text
(symbol, decision_bar_close_ms, outcome_horizon)
```

Invariants:

- outcome columns are future diagnostics only;
- missing horizons remain explicit through coverage/artifact diagnostics;
- outcomes do not feed feature construction.

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
candidate-evidence.csv
candidate-rank.csv
report.md
```

A model iteration should be able to reuse materialized artifacts; code should not skip stages ad hoc to finish a scan. Reduce config when necessary, or split materialization from evidence/review in a later workflow command.
