# Overview Graph

Current package-level dependency graph.

```text
scripts/scanner_potential.py
  -> qooi.scanner.workflow.run(config_path)

qooi.scanner
  -> qooi.pipeline
  -> qooi.transport.OkxClient
  -> qooi.profiling

qooi.pipeline
  -> qooi.transport client methods supplied by caller

qooi.transport
  -> external provider APIs

qooi.strategies / qooi.core / qooi.dynamic
  -> separate boundaries; scanner output does not authorize them
```

Active scanner path:

```text
configs/potential-daily-tailtree.toml
configs/potential-advanced-tailtree.toml
  -> scripts/scanner_potential.py
  -> workflow.run
  -> pipeline.load_market
  -> state/outcome/evidence/rank/output
  -> data/output/potential/<run>/report.md
```

Docs by implementation family:

```text
docs/graph/pipeline.md
docs/graph/transport.md
docs/graph/scanner.md
docs/graph/tailtree.md
docs/graph/profiling.md
```
