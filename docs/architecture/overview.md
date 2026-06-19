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
configs/potential-daily-tailtree.toml      # fast fixed h24 daily scan
configs/potential-advanced-tailtree.toml   # h24 Optuna + walkforward research scan
```

Empirical scanner reports live under `docs/report/`. Architecture docs should not duplicate run results.
