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
| Profiling | `qooi.profiling` | Cross-cutting native profiling context, stage/frame records, and profile artifacts. |
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

profiling
  -> stdlib profiling + Polars profile adapters + profile artifact records
  -> no domain workflow ownership

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

`PotentialConfig` is the single potential-workflow config entry. That root
object is reasonable: one TOML document validates one run. The boundary is
semantic, not syntactic: nested sections are embedded component configs, while
`workflow.py` composes them into package-owned requests/contexts before calling
other modules.

Do not split the user surface into many independent files just to avoid a root
object. Avoid repetition by reusing component config types and by preventing the
root config from leaking past workflow composition.

Do not preserve backward-compatible aliases when current callers can be updated.

| Config section | Type owner | Workflow role | Forbidden repetition |
|---|---|---|---|
| `[potential]` | `qooi.scanner.workflow` | config entry, run identity, output, universe, bar, scan refresh, concurrency | provider/model/report internals |
| `[potential.source]` | `qooi.sources` request/config type | source demand and freshness input | refresh mode, rank/report policy |
| `[potential.transition]` | `qooi.scanner.transitions` | transition windows and thresholds | source/provider fields |
| `[potential.evidence]` | `qooi.scanner.diagnostics` dispatch type | evidence path selection | booleans such as `use_tail_tree` |
| `[potential.evidence.tailtree]` | `qooi.scanner.tailrun` | model lifecycle, artifact identity, train-summary integrity | source refresh, report rendering, random-split HPO |
| `[potential.review]` | `qooi.scanner.workflow` | current review audit requirements | candidate ranking, source acquisition |
| `[potential.profile]` | `qooi.profiling` | injected profile context mode | scanner/source-specific behavior |

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

Same rule for profiling:

```text
GOOD: workflow embeds ProfileConfig and injects ProfileContext
BAD:  scanner/source modules implement their own profile sinks
```

## Lean-module policy

Large modules are not automatically wrong, but repeated helper surfaces and read-back/conversion layers violate the project context. Reduction should follow ownership, not create `_utils.py` grab bags.

Current high-pressure modules to reduce:

| Package | Module | Current pressure | Reduction direction |
|---|---|---:|---|
| scanner | `diagnostics.py`, `feasibility.py` | artifact assembly/write orchestration plus source/history/candidate feasibility projections | clarify pipe/product grains first; split only when a product has an owner, schema, and tests |
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
