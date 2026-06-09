# qooi

`qooi` is an under-development research workspace for crypto market-structure experiments on OKX-style perpetual swap data.

The current project direction is **research only**: deterministic, known-at-close classifiers and scanner artifacts may generate hypotheses, but they do not authorize live trading or allocation.

## Current focus

- Keep research, scanner, data, execution, and experimental learned-state work separated.
- Prefer deterministic manual-classifier research before optional AI/learned-state experiments.
- Treat strategy work as hypothesis development until it is promoted through explicit signal columns and execution-aware evaluation.
- Keep missing or shallow data visible in diagnostics and coverage artifacts.

## Development

This project uses `uv` for environment and command management.

```bash
uv sync
uv run ruff check src tests scripts
uv run pytest
```

Useful research entry points live under `scripts/`; durable project guidance lives under `docs/`.

## Documentation

Start here:

- `docs/context.md` — project goal, development rules, and code principles.
- `docs/README.md` — documentation index.
- `docs/architecture/` — durable architecture boundaries.
- `docs/graph/` — implementation-facing module graphs.
- `docs/report/` — empirical reports and decisions.

## Status

The repository is intentionally under-developed and changing. Avoid treating current module layout, public helper functions, or strategy names as stable API unless they are documented and covered by tests.
