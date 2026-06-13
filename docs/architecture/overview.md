# System Architecture Overview

## Purpose

This document is the canonical architecture split for `qooi`. It defines module-family ownership and global dependency direction. Detailed implementation-facing function/module graphs live in `docs/graph/`.

## Current architecture decision

The active path is deterministic and manual-classifier-first. The potential scanner searches for symbols whose known-at-close state vectors materially change future transition/path distributions, especially tail behavior. Scanner and research outputs are information aids only; they do not authorize trading or allocation.

AI/learned-state work is isolated in `qooi.dynamic` because it is not the active promotion-ready path and must not contaminate scanner, strategy, or execution modules.

## Canonical module layout

| Architecture part | Package/path | Role |
|---|---|---|
| Exchange | `qooi.exchange` | Exchange adapters, OHLCV/history cache, market discovery, data coverage, thin trading IO wrapper. |
| Sources | `qooi.sources` | Provider/source collectors, source normalization, manifests, bundles, freshness, missing-data diagnostics. |
| Strategy / Signal | `qooi.strategies` | Indicators, market-structure semantics, strategy specs, portfolio qualification, explicit signal columns. |
| Core / Execution | `qooi.core` | Basket lifecycle, executor/backtests, recovery policies, evaluation, metrics, soft/live state. |
| Research | `qooi.research` | Deterministic known-at-close research tables, pattern/outcome/metric/promotion artifacts. |
| Scanner | `qooi.scanner` | Potential trading-change scanner, observation/outcome/evidence computation, research-review reports. |
| Dynamic | `qooi.dynamic` | Isolated optional learned-state / AI research sandbox. |
| Scripts | `scripts/` | Thin command entrypoints only; orchestration lives in packages. |

## Global dependency direction

```text
scripts
  -> scanner / research / dynamic / core orchestration

scanner
  -> exchange + sources + scanner-local Polars transforms
  -> no core executor/basket/recovery
  -> no dynamic

research
  -> research.data + strategies + deterministic table transforms
  -> may use core only in explicit report/backtest orchestration

core
  -> strategies for signal computation in executor convenience paths
  -> exchange.trading/state only at live/trading IO boundaries

strategies
  -> pure indicators/semantics/specs
  -> no exchange/sources/scanner/core execution ownership

exchange
  -> market/exchange/cache/trading IO
  -> may dispatch source-family context where currently implemented
  -> no strategy/scanner/research policy

sources
  -> provider APIs, source normalization, manifests, artifact bundles
  -> no strategy/scanner/research/core policy

dynamic
  -> prepared frames / sequence contracts / optional ML
  -> no exchange, sources, scanner, executor, basket, or strategy ownership
```

## Forbidden crossovers

- Data/exchange must not import strategies, baskets, executor, scanner decisions, research policy, or reports.
- Sources must not import scanner, research, strategies, execution, recovery, dynamic, or trading clients.
- Scanner must not import executor, basket lifecycle, recovery, live trading clients, or dynamic learned-state modules.
- Dynamic must not import scanner, exchange, sources, strategies, basket lifecycle, executor, or recovery modules.
- Strategies must not fetch data, collect sources, rank scanner evidence, size baskets, account fills, or mutate recovery state.
- Evaluation must not mutate execution state.

## Current default workflow

```text
configs/potential*.toml
  -> scripts/potential_scan.py       # potential scanner CLI
  -> qooi.scanner.workflow.run()
  -> OHLCV/source cache and diagnostics
  -> deterministic known-at-close state classification
  -> observation/outcome/evidence surfaces
  -> Markdown research report and CSV diagnostics
```

## Composable configuration policy

`PotentialConfig` remains the single scanner TOML/root parse boundary. Do not split
the user-facing config into independent top-level files or duplicate package config
trees. Composition happens at module call boundaries: `workflow.py` passes the
small request/config view that a module needs, not the whole root config.

Do not preserve backward-compatible aliases when current callers can be updated.

| Config section | Owner | Decision owned | Forbidden repetition |
|---|---|---|---|
| `[potential]` | `qooi.scanner.workflow` | scanner run identity, output, universe, one scan refresh mode for materialized inputs, concurrency | source-provider knobs, evidence lifecycle, report audit flags |
| `[potential.source]` | source request constructed by `workflow.py` | source limits, source staleness, disabled source/symbol demand | refresh mode, OHLCV bar refresh, candidate/rank/report policy |
| `[potential.transition]` | `qooi.scanner.transitions` | transition/history windows and probability thresholds | source freshness, model lifecycle |
| `[potential.evidence]` | `qooi.scanner.diagnostics` dispatch | evidence path name | booleans such as `use_tail_tree`, provider menus |
| `[potential.evidence.tailtree]` | `qooi.scanner.tailrun` | train/load-predict lifecycle and model artifact identity | source refresh, report rendering |
| `[potential.review]` | `qooi.scanner.decisions` | current review requirements only | candidate ranking, source acquisition |

Refresh semantics must be singular:

```text
PotentialConfig.refresh_mode -> materialized input refresh for bars and source context
```

Nested source refresh overrides are forbidden by default. The old two-field shape made
`[potential.source].refresh_mode` look like it controlled the whole scan while bar
refresh still used `[potential].refresh_mode`; that violates scanner config ergonomics.
If source and bar materialization ever need different cadence, use separate workflow
commands/config files such as materialize-bars, materialize-sources, and evaluate-report
rather than repeating a `refresh_mode` field in nested sections. If a config shape changes,
update callers directly and remove old aliases rather than accepting both shapes.

Boundary rule:

```text
GOOD: workflow builds SourceContextRequest(output_dir, target_days, concurrency,
      refresh_mode=PotentialConfig.refresh_mode, source=PotentialConfig.source, ...)
BAD:  qooi.sources.context.load_source_context(PotentialConfig, ...)
```

The source package is indifferent to scanner config shape; it needs a demand
request. It may define the request contract, but it should not know about the
whole `PotentialConfig` object or scanner transition/evidence/review sections.

## Lean-module policy

Large modules are not automatically wrong, but repeated helper surfaces and read-back/conversion layers violate the project context. Reduction should follow ownership, not create `_utils.py` grab bags.

Current high-pressure modules to reduce:

| Package | Module | Current pressure | Reduction direction |
|---|---|---:|---|
| scanner | `diagnostics.py` | artifact assembly, evidence dispatch, report projections, frame writers | first extract pure functions inside the module; split only when a product has independent callers/tests |
| scanner | `report.py` | section logic plus CSV read-back plus type recovery | render in-memory report frames; remove default decision audit; delete conversion helpers |
| sources | `collect.py` | demand planning plus provider execution | keep pure demand planning separate from side-effect collection within named APIs; no compatibility wrappers |
| exchange | `store.py` | cache request, refresh, validation, coverage rows | keep cache/coverage contracts decisive; no source-context policy |
| research | `reports.py` / `data.py` | wide report/data transforms | keep research artifacts consuming package outputs, not scanner internals |

Reduction rule:

```text
If two helpers differ only by family/source/report table, replace them with a
small typed table or dataclass method owned by the module's data product.
If a helper exists only to recover type after CSV read-back, remove the read-back.
If a module only forwards old names, delete it and update callers.
```

## Promotion boundary

Scanner and research outputs are review surfaces. They become strategy candidates only after they are converted into explicit strategy signal columns and pass execution-aware backtests through `qooi.core`.

## Documentation boundary

- `docs/architecture/*.md` defines durable ownership, responsibilities, allowed dependencies, and forbidden dependencies.
- `docs/graph/*.md` maps concrete modules, functions, classes, artifacts, and call flow.
- `docs/report/*.md` preserves empirical run summaries, historical notes, and decisions.
