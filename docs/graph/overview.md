# Overview Module Graph

```text
scripts/
  accumulation_scan.py
    -> qooi.scanner.workflow.run(config_path)
  classifier_states.py
    -> qooi.research.reports.run_reports(...)
  learned_states.py
    -> qooi.dynamic + qooi.research prepared frames

qooi.exchange
  -> market/cache/universe/context data
qooi.sources
  -> provider/source artifacts
qooi.scanner
  -> deterministic potential scanner reports
qooi.research
  -> deterministic research artifacts
qooi.strategies
  -> signal columns
qooi.core
  -> basket proposals, backtest/live execution, evaluation
qooi.dynamic
  -> isolated AI research labels and diagnostics
```

Default active path:

```text
configs/potential*.toml
  -> scripts/accumulation_scan.py
  -> qooi.scanner.workflow.run()
  -> docs/report-style Markdown + parquet diagnostics
```
