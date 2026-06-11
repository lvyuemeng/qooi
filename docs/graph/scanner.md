# Scanner Module Graph

Shared scanner skeleton: CLI orchestration, known-at-close feature/observation products, one evidence dispatch, candidate/rank/report. Path internals live in `docs/graph/evidence.md` and `docs/graph/tailtree.md`.

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
  → qooi.scanner.evidence.potential_observation_frame(...)
  → observations

observations/source_outcomes/realized_transitions
  → qooi.scanner.evidence.potential_outcome_frame(...)
  → outcomes

observations/outcomes
  → qooi.scanner.diagnostics._run_pipeline(...)
  → LadderResult | TailtreeResult

result.evidence/latest_observations
  → qooi.scanner.candidates.candidate_evidence_frame(...)
  → qooi.scanner.candidates.rank_candidate_evidence(...)
  → candidate artifacts + report
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

---

## Observation + outcome: `qooi.scanner.evidence`

```text
qooi.scanner.evidence.potential_observation_frame(
    kline_history: pl.DataFrame,
    source_events: pl.DataFrame,
    continuous_features: pl.DataFrame | None,
    *,
    decision_timeframe: str,
    max_source_staleness_hours: int,
) -> pl.DataFrame
```

Observation contract:

```text
key: symbol, decision_bar_close_ms, source_family
continuous join: (symbol, decision_bar_close_ms) = (symbol, timestamp)
```

Continuous features are joined on every path, including no-source/stale-source paths.

```text
qooi.scanner.evidence.potential_outcome_frame(
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
if inputs.config.evidence == "tailtree":
    return _run_tailtree_pipeline(...)
return _run_ladder_pipeline(...)
```

Downstream code consumes concrete result fields:

```text
result.evidence
result.candidates
result.ranked
```

No variant tuple, no `tree_up: None`, no `if config.evidence` in report sections.

---

## Result contracts

```text
qooi.scanner.contracts.LadderResult
├── evidence: pl.DataFrame
├── candidates: pl.DataFrame
└── ranked: pl.DataFrame

qooi.scanner.contracts.TailtreeResult
├── evidence: pl.DataFrame
├── candidates: pl.DataFrame
├── ranked: pl.DataFrame
├── tree_up: TailTreeModel
└── tree_down: TailTreeModel
```

---

## Candidate + ranking graph: `qooi.scanner.candidates`

```text
qooi.scanner.candidates.candidate_evidence_frame(
    latest_observations: pl.DataFrame,
    evidence: pl.DataFrame,
    *,
    coverage_frame: pl.DataFrame,
    freshness_frame: pl.DataFrame,
    tree_up: TailTreeModel | None = None,
    tree_down: TailTreeModel | None = None,
) -> pl.DataFrame
```

The only caller that supplies tree models is the tailtree pipeline. Candidate internals should prefer path-specific helpers rather than spreading config checks.

```text
qooi.scanner.candidates.rank_candidate_evidence(candidates: pl.DataFrame) -> pl.DataFrame
```

Ranking uses numeric columns that are present:

- ladder: information gain, stability, support, path quality, data quality;
- tailtree: tail lift, tail stability, support, path quality, data quality.

---

## Diagnostics and report

```text
qooi.scanner.diagnostics.write_diagnostics(inputs: ReportInputs) -> DiagnosticFrames
qooi.scanner.report.render_report(inputs: ReportInputs) -> str
```

Diagnostics writes path-specific artifacts after the one dispatch. Report renders the frames it receives; path-specific sections are composed before rendering, not by branching inside every section.

---

## Removed/stale API references

These must not appear in current scanner docs/code contracts:

```text
use_tail_tree                 # replaced by evidence = "ladder" | "tailtree"
leaf_path / leaf_paths        # removed; leaf_id + numeric stats are enough
to_pandas training path       # optional pyarrow/pandas path is not required
fobj custom-objective arg     # LightGBM 4 custom objective is params["objective"]
```
