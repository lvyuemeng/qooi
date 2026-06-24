# qooi

`qooi` is an under-development quantitative research workspace for OKX-style crypto perpetual swap data.

The active product is the **potential scanner**. It builds known-at-close market/source state rows, trains or loads tail-event evidence models, ranks review candidates, and writes diagnostics artifacts. Scanner output is research evidence only; it is not a trade signal, allocation instruction, execution system, or wallet operation.

## Current scan

There are two public scanner configs.

### Train + score current frontier

Use this when changing the current tailtree model surface or refreshing the research frontier:

```bash
uv run python scripts/scanner_potential.py --config configs/potential-tailtree-train.toml
```

This workflow:

```text
load market/source data
-> build known-at-close state rows
-> build future outcome labels for training/evaluation only
-> train h24 tail_event_lift models with Optuna + walkforward
-> train candidate-local promoter/opposite/weak guards
-> emit candidate_dual_guard frontier diagnostics
```

Expected report root:

```text
data/output/potential/tailtree-train/report.md
```

Key train artifacts:

```text
data/output/potential/tailtree-train/tailtree-profile-runs.csv
data/output/potential/tailtree-train/tailtree-selection-efficiency.csv
data/output/potential/tailtree-train/tailtree-frontier-benchmark.csv
data/output/potential/tailtree-train/tailtree-action-surface.csv
data/output/potential/tailtree/models/*.json
```

Current train objective surface:

```text
stage-1 evidence: tail_event_lift
final selection:  candidate_dual_guard
horizon:          h24
source inputs:    persistent funding/LSR/OI/taker context features
```

### Predict-only from model ids

Use this when you want to score/report with existing model JSONs and avoid training during the scan:

```bash
uv run python scripts/scanner_potential.py --config configs/potential-tailtree-predict.toml
```

This workflow:

```text
load market/source data
-> build known-at-close state rows
-> load configured model JSONs by model_id
-> emit loaded tail_event_lift evidence/report artifacts
```

Expected report root:

```text
data/output/potential/tailtree-predict/report.md
```

Predict-only intentionally does **not** train candidate-local promoter/opposite/weak guard models. Therefore it does not emit `candidate_dual_guard` frontier rows unless those guard models are made loadable in a future design slice.

## Current workflow shape

```text
config
-> load market/source data
-> state: known-at-close observations and features
-> outcome: future labels/evaluation rows
-> tailtree evidence
-> rank/review diagnostics
-> report/profile artifacts
```

Operational defaults:

- public configs are limited to `potential-tailtree-train.toml` and `potential-tailtree-predict.toml`;
- preferred scanner horizon is `h24` on `1H` bars;
- train mode uses Optuna + walkforward for model-quality research;
- predict mode resolves model JSONs by `model_id` and carries no fixed training profile;
- current best-performance train path is `tail_event_lift` stage-1 evidence plus source-context `candidate_dual_guard` final selection;
- low-performance or stale objective surfaces are removed instead of preserved as compatibility branches.

## Development

```bash
uv sync
uv run ruff check src tests scripts/scanner_backfill.py
uv run ty check
uv run pytest tests/ -q -m "not integration"
```

For scanner/tailtree changes, finish with the relevant real smoke command from [Current scan](#current-scan), not only unit tests.

## Documentation

Start here:

- `docs/context.md` — project rules and durable constraints.
- `docs/architecture/overview.md` — package boundaries and current scanner/tailtree example.
- `docs/architecture/scanner.md` — scanner ownership, tailtree objective choice, and artifact semantics.
- `docs/graph/scanner.md` — current scanner call graph.
- `docs/graph/tailtree.md` — tailtree/tailrun model lifecycle graph.
- `docs/report/` — empirical run reports, plans, and migration decisions.

## Status

The scanner is research infrastructure. Treat every candidate as an information-bearing review row, not an executable trading instruction.
