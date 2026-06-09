# qooi — project context

This file is the root project-management context. It should stay durable and short. It records the project goal, development management, and code principles only. Module layout is ephemeral and belongs in architecture or graph docs when it is useful.

## Current goal

`qooi` is an under-development quantitative research workspace for OKX-style crypto perpetual swap data.

The active goal is to tidy and develop the potential altcoin scanner around a deterministic diagnostics-probability framework. The scanner should find symbols whose known-at-close state vectors materially change future transition/path distributions, especially extreme behavior, and provide information aid for trading research.

Scanner and research artifacts are hypotheses only; they do not authorize live trading, allocation, or automatic strategy promotion. AI/learned-state work remains optional and isolated until it produces promotion-ready evidence. It must not block deterministic classifier and scanner cleanup.

## Development management

Use `uv` for environment and command management:

```bash
uv sync
uv run ruff check src tests scripts
uv run pytest
```

Use narrower checks when iterating, but finish code changes with the relevant lint and tests. Missing data, shallow data, stale data, and unavailable optional dependencies must be explicit in diagnostics or test output.

Commit messages use short bracket-prefix form:

```text
[prefix]: content
```

Examples: `[docs]: update scanner graph`, `[feat]: add candidate ranking`, `[fix]: handle missing coverage`.

## Documentation workflow

Documentation follows:

```text
context -> architecture -> module graph
```

- `docs/context.md`: project goal, management rules, and code principles only.
- `docs/architecture/`: durable design boundaries.
- `docs/graph/`: implementation-facing module graphs for code that is stable enough to map.
- `docs/report/`: empirical run summaries and decisions.

Do not use this file for module layouts, detailed call graphs, run reports, or temporary plans.

## Code principles

1. Avoid spaghetti dependencies. A module should have a narrow reason to change.
2. Avoid redundant helper functions. Prefer one direct, named implementation over parallel compatibility wrappers.
3. Separate side effects from computation. Fetching, cache writes, file writes, report rendering, and trading IO should stay at orchestration boundaries.
4. Keep research tables known-at-close. Future returns may be outcome columns only, never inputs to state construction or current decision labels.
5. Prefer Polars-native DataFrame operations. Avoid row loops and pandas-style detours unless there is a measured reason.
6. Make missing coverage explicit instead of silently dropping symbols, sources, or windows.
7. Do not hardcode API keys, provider secrets, exchange-wallet labels, or environment-specific paths.
8. Remove stale APIs instead of adding compatibility layers when current callers can be updated directly.
9. Keep optional AI/ML dependencies lazy and isolated so normal research and scanner imports remain lightweight.
10. Treat `strategies/` as an unstable, vague area for now: computation is distributed across several parts, so strategy design should not be considered ideal or stable until promoted through explicit signal contracts and tests.

## Promotion policy

A research finding can become a strategy hypothesis only after:

1. No-lookahead construction passes.
2. Count, symbol, and time-split gates pass.
3. Missing/stale data is explicit and acceptable.
4. There is an ex-ante market-behavior rationale.
5. The idea is expressed as normal signal columns.
6. Execution-aware backtests pass fees, sizing, basket caps, stop/target behavior, recovery policy, and comparability gates.

Current recommendation: no scanner or research artifact is allocation-ready by itself.
