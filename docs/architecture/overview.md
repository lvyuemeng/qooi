# Architecture Overview

## Purpose

`qooi` is a research codebase. The active scanner path is deterministic research infrastructure; it produces review artifacts, not trading authorization.

## Current package families

```text
qooi.pipeline      # market data load/cache/coverage composition
qooi.transport     # HTTP/client boundary; OKX transport lives here
qooi.scanner       # potential scanner workflow, state/outcome/evidence/rank/report
qooi.profiling     # cross-cutting profile context and artifacts
qooi.strategies    # strategy research boundary; consumes promoted signals only after explicit contracts
qooi.core          # basket/executor/evaluation boundary; not reached by scanner output
qooi.dynamic       # isolated learned-state/AI experiments
```

## Dependency direction

```text
transport  -> external network only
pipeline   -> transport + local storage/coverage contracts
scanner    -> pipeline + transport client + profiling
strategies -> scanner artifacts only through explicit signal contracts
core       -> strategies/evaluation/execution; scanner does not call it
dynamic    -> isolated research sandbox
```

Forbidden:

- scanner importing executor, basket, wallet, or live-trading modules;
- scanner output authorizing orders/allocation;
- source/API keys or exchange-wallet labels hardcoded in docs or config;
- hidden data gaps: missing, stale, provider-bounded, and current-only data must stay visible.

## Scanner stage status

The current scanner stage is:

```text
config -> load market data -> state -> outcome -> tailtree evidence -> rank -> review/report
```

Current preferred scanner profiles:

```text
configs/potential-tailtree-train.toml     # train + score h24 Optuna/walkforward current frontier
configs/potential-tailtree-predict.toml   # load existing model JSONs by model_id and score/report
```

## Current tailtree example

The current advanced tailtree choice is:

```text
known-at-close observation/source state
  -> h24 tail_event_lift stage-1 LightGBM evidence
  -> candidate-local promoter model
  -> opposite-direction guard model
  -> weak/no-tail path guard model
  -> candidate_dual_guard final selection-efficiency/frontier rows
```

Parameter/objective choice:

```text
horizon: h24
model target: tail_event_lift
validation: walkforward
search: Optuna
source input: 17 persistent funding/LSR/OI/taker state, transition, run-length, and divergence columns
final objective surface: candidate_dual_guard
```

Reason:

```text
tail_event_lift keeps the broad extreme-opportunity evidence signal;
candidate_dual_guard adds promoter + opposite guard + weak/no-tail guard safety;
source-context inputs are folded into the single active feature set because the best observed source-feature run improved selected count, precision, false-direction, and utility;
lower-performance competing objective rows were removed instead of kept as active alternatives.
```

Recent advanced smoke surface:

```text
tailtree-selection-efficiency.csv: 3504 rows = candidate_dual_guard 3456 + tail_event_lift 48
tailtree-frontier-benchmark.csv: 2107 rows = candidate_dual_guard only
predict-only selection-efficiency.csv: 8 rows = loaded tail_event_lift only
forbidden objective rows: 0 for source_blended, candidate_conditional_promoter, candidate_opposite_guard, continuous_guard_curve, two_model_guard
best inspected frontier rows: precision about 0.56-0.60, false-direction about 0.12-0.36, utility about 3.55-5.30
```

Empirical scanner reports live under `docs/report/`. Architecture docs should only summarize durable current choices and point to reports for run-level evidence.
