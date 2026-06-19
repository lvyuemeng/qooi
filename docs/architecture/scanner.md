# Scanner Architecture

## Purpose

The scanner finds symbols whose known-at-close market/source state is associated with concentrated future tail behavior. It writes research artifacts and review candidates only.

Non-goals:

- no live trading;
- no order sizing;
- no execution/cost/slippage model inside scanner promotion;
- no wallet or exchange-account decisions.

## Current module ownership

```text
src/qooi/scanner/
├── config.py       # PotentialConfig and scanner section config models
├── workflow.py     # outer lifecycle: config, universe, market load, pipe composition, final write
├── state.py        # known-at-close state/features/observation rows
├── outcome.py      # future/path/source outcome rows
├── ladder.py       # fixed ladder evidence path
├── rank.py         # candidate matching, comparable surface, ranking
├── output.py       # market readiness, review decisions, markdown report
├── transitions.py  # transition-pattern analysis
├── tailtree/       # LightGBM/GPD model and evidence products
└── tailrun/        # tailtree lifecycle, profiles, Optuna, artifacts, typed run records
```

Do not document removed transitional scanner modules as current APIs. The list above is the current source-of-truth layout.

## Workflow pipe

```text
scripts/potential_scan.py
  -> qooi.scanner.workflow.run(config_path)
     -> load_config
     -> resolve universe
     -> pipeline.load_market via scanner_market_request/scanner_market_policy
     -> state.classify_states
     -> state.extract_continuous_features
     -> state.potential_observation_frame
     -> outcome.realized_transition_frame
     -> outcome.source_outcomes_frame
     -> outcome.potential_outcome_frame
     -> evidence dispatch
        ladder: ladder.evidence
        tailtree: tailrun.core.run_tailtree
     -> rank.candidate_metric_surface
     -> rank.rank_candidates
     -> output.review_decisions
     -> output.render_report
     -> profile artifacts
```

`workflow.py` is allowed to compose the scanner pipe. Tailtree training, Optuna sampling, fold construction, model persistence, and selection-efficiency rows belong under `tailrun/` and `tailtree/`, not workflow helpers.

## Config profiles

Daily scanner profile:

```text
configs/potential-daily-tailtree.toml
```

Role:

```text
fixed h24 tail_utility_quantile profile
bounded symbols/depth
fast daily review surface
```

Advanced scanner profile:

```text
configs/potential-advanced-tailtree.toml
```

Role:

```text
h24 tail_utility_quantile
Optuna training
walkforward evaluation
larger 80-symbol research surface
selection-efficiency feedback
```

The scanner currently prefers one horizon, `h24`, because daily prediction freshness often approaches the 24h boundary and h24 has stronger tail-label support than shorter horizons.

## Data and freshness boundaries

Known-at-close state only:

```text
state.py -> observations keyed by symbol/timeframe/bar close
```

Future/path labels only:

```text
outcome.py -> outcome rows keyed by symbol/bar close/horizon
```

Persistent derivative-source families may train tailtree when aligned historically:

```text
funding
open_interest
taker_volume
long_short_ratios
```

Books/trades are current-review context unless a consistent historical artifact contract exists.

Reports must expose:

```text
missing data
stale data
provider-bounded history
current-only sources
coin_too_new symbols
deferred_by_budget rows
```

## Tailtree boundary

Tailtree input:

```text
TailtreeInputFrames(observations, source_outcomes, realized, histories)
```

Tailtree output:

```text
TailtreeRunOutput(
  evidence,
  models,
  profile_runs,
  selection_efficiency,
)
```

Artifacts:

```text
report.md
tailtree-profile-runs.csv
tailtree-selection-efficiency.csv
models/*.json
profile/*.csv
```

Model persistence is JSON. The scanner does not pickle models.

## Promotion semantics

`rank.py` builds the comparable candidate surface. `output.review_decisions` applies review gates:

```text
prediction freshness
missing/stale source blockers
support threshold
tail_lift threshold
opposite-direction conflict watch
top-N promote cap
```

Promote/watch/skip rows are research review decisions, not execution instructions.

## Forbidden dependencies

- scanner -> executor/core basket/wallet modules;
- tailtree -> ladder cross-imports;
- state -> future outcome columns;
- outcome -> model/evidence internals;
- report/output code reading CSV artifacts back as an internal transport;
- opaque row probing in model/report boundaries when a typed frame or dataclass exists.
