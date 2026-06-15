# qooi Documentation

Documentation follows a three-layer workflow:

```text
context -> architecture -> module graph
```

## 1. Context

| Document | Purpose |
|---|---|
| `context.md` | Root project-management context: current goal, `uv` management, promotion policy, and code principles. |

`context.md` should stay short and managerial. It should not accumulate module layouts, call graphs, or implementation details.

## 2. Architecture

Durable design boundaries live under `docs/architecture/`.

| Document | Purpose |
|---|---|
| `architecture/overview.md` | Canonical module-family layout and system-wide dependency direction. |
| `architecture/exchange.md` | Exchange adapters, cache, universe, coverage, and trading-IO boundaries. |
| `architecture/sources.md` | Source collectors, manifests, source bundles, freshness, and missing-data behavior. |
| `architecture/strategy.md` | Strategy/signal layer and required signal-column contract. |
| `architecture/core.md` | Basket lifecycle, recovery, executor, accounting, and evaluation boundaries. |
| `architecture/research.md` | Deterministic manual-classifier research and shared table pipe. |
| `architecture/scanner.md` | Potential trading-change scanner architecture. |
| `architecture/profiling.md` | Cross-cutting native profiling context and profile artifact contracts. |
| `architecture/dynamic.md` | Isolated AI/learned-state research sandbox. |
| `architecture/testing.md` | Test ownership map and reduction rules. |

## 3. Module graphs

Concrete implementation-facing module graphs live under `docs/graph/`.

| Document | Purpose |
|---|---|
| `graph/overview.md` | Whole-system graph and default active workflow. |
| `graph/exchange.md` | Exchange/cache/universe/context module graph. |
| `graph/sources.md` | Source collector and artifact module graph. |
| `graph/strategy.md` | Strategy/signal module graph. |
| `graph/core.md` | Core basket/executor/evaluation module graph. |
| `graph/research.md` | Research table/report module graph. |
| `graph/scanner.md` | Potential scanner module graph. |
| `graph/profiling.md` | Cross-cutting profile context API and artifact graph. |
| `graph/dynamic.md` | Dynamic/AI module graph and forbidden edges. |

Each isolated module family owns its own graph. Do not keep cross-cutting graph appendices when the implementation has no matching package boundary; fold the API surface into the owning family graph instead.

## Reports

| Document | Purpose |
|---|---|
| `report/research.md` | Empirical research summaries and decisions. |

## Agent metadata

| Directory | Purpose |
|---|---|
| `agents/` | Local agent-skill context, issue-tracker vocabulary, and domain-doc conventions. |

## Naming rules

- Root context is managerial only.
- Durable designs go in `docs/architecture/`.
- Module graphs go in `docs/graph/`.
- Empirical summaries, historical notes, and decisions go in `docs/report/`.
- Legacy topic vocabulary may be preserved in reports, but not as parallel architecture authority.
