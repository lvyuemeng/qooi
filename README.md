# qooi

`qooi` is a research workspace for deterministic crypto market-structure scanning on OKX-style perpetual swap data.

The active product is the **potential scanner**: it builds known-at-close market/source state rows, trains/evaluates tail-event evidence, ranks review candidates, and writes research artifacts. Scanner output is research-only; it does not authorize live trading, allocation, order execution, or wallet operations.

## Scanner entry points

Daily fast scan:

```bash
uv run python scripts/scanner_potential.py --config configs/potential-daily-tailtree.toml
```

Advanced Optuna/walkforward scan:

```bash
uv run python scripts/scanner_potential.py --config configs/potential-advanced-tailtree.toml
```

Primary outputs:

```text
data/output/potential/<run>/report.md
data/output/potential/<run>/tailtree-profile-runs.csv
data/output/potential/<run>/tailtree-selection-efficiency.csv
data/output/potential/<run>/profile/
data/output/potential/<run>/models/
```

## Current scanner shape

```text
config -> load market data -> state -> outcome -> tailtree/ladder evidence -> rank -> review/report
```

Operational defaults:

- one preferred scanner horizon: `h24` on `1H` bars;
- daily config is fixed/bounded for speed;
- advanced config uses Optuna + walkforward for model-quality research;
- promotion gates are conservative and report watch/skip reasons explicitly;
- missing, stale, provider-bounded, and current-only data remain visible in reports.

## Development

```bash
uv sync
uv run ruff check src tests scripts/scanner_backfill.py
uv run ty check
uv run pytest tests/ -q -m "not integration"
```

## Documentation

Start here:

- `docs/context.md` — project rules and durable constraints.
- `docs/architecture/scanner.md` — scanner ownership and boundaries.
- `docs/graph/scanner.md` — current scanner call graph.
- `docs/graph/tailtree.md` — tailtree/tailrun model lifecycle graph.
- `docs/report/` — empirical run reports and migration decisions.

## Status

The scanner is still research infrastructure. Treat candidates as review hypotheses, not trading signals.
