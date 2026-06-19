# qooi Documentation

Documentation follows:

```text
context -> architecture -> graph -> report
```

## Context

| Document | Purpose |
|---|---|
| `context.md` | Project rules, repository boundaries, and development constraints. |

## Architecture

Durable boundaries live in `docs/architecture/`.

| Document | Purpose |
|---|---|
| `architecture/overview.md` | Current package-family boundaries and dependency direction. |
| `architecture/pipeline.md` | Market data load/cache pipeline boundaries. |
| `architecture/transport.md` | Transport/client ownership. |
| `architecture/scanner.md` | Potential scanner architecture. |
| `architecture/profiling.md` | Cross-cutting profile context and artifacts. |
| `architecture/strategy.md` | Strategy layer boundary. |
| `architecture/core.md` | Core basket/executor/evaluation boundary. |
| `architecture/dynamic.md` | Isolated learned-state/AI research boundary. |

## Graphs

Implementation-facing API graphs live in `docs/graph/`.

| Document | Purpose |
|---|---|
| `graph/overview.md` | Whole-system module graph. |
| `graph/pipeline.md` | Market data load/cache graph. |
| `graph/transport.md` | Transport/client graph. |
| `graph/scanner.md` | Scanner workflow graph. |
| `graph/tailtree.md` | Tailtree/tailrun lifecycle graph. |
| `graph/profiling.md` | Profile context graph. |
| `graph/strategy.md` | Strategy graph. |
| `graph/core.md` | Core graph. |
| `graph/dynamic.md` | Dynamic/AI graph. |

## Reports

`docs/report/` preserves empirical runs, migration decisions, and stage-close notes. Reports are historical evidence, not parallel architecture authority.

## Rules

- Root context stays managerial.
- Architecture docs state ownership and forbidden dependencies.
- Graph docs name concrete modules/functions/artifacts.
- Reports carry run results, plans, and decisions.
