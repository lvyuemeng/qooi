# qooi Agent Context

Use `docs/context.md` as the canonical AGENTS/domain context for this repository.

Before changing code or docs, read and preserve the boundaries in `docs/context.md`:

- Data, strategy, basket, executor, evaluation, research, and scanner layers stay separated.
- Research artifacts do not authorize live trading.
- API keys and exchange-wallet labels must not be hardcoded.
- Missing data must be explicit in diagnostics and coverage artifacts.

Documentation categories:

- `docs/graph/` for implementation-facing module graphs and public module surfaces.
- `docs/architecture/` for durable implementation boundaries and module designs.
- `docs/report/` for empirical run summaries and decisions.
