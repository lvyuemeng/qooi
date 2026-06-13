# Overview Module Graph

```text
scripts/
  potential_scan.py
    -> qooi.scanner.workflow.run(config_path)
  classifier_states.py
    -> qooi.research.reports.run_reports(...)
  learned_states.py
    -> qooi.dynamic + qooi.research prepared frames

qooi.exchange
  -> market/cache/universe/trading IO
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
  -> scripts/potential_scan.py
  -> qooi.scanner.workflow.run()
  -> docs/report-style Markdown + CSV diagnostics
```

Composable config graph:

```text
PotentialConfig.refresh_mode
  -> scanner.workflow.load_bars
  -> exchange.store.HistoryRefreshRequest
  -> scanner.workflow.source_context_request
  -> sources.context.SourceContextRequest
  -> sources.collect.SourceCollectRequest

PotentialConfig.source
  -> sources provider limits, staleness, disabled demand

PotentialConfig.evidence.kind
  -> scanner.diagnostics evidence dispatch

PotentialConfig.evidence.tailtree.lifecycle
  -> scanner.tailrun train/load_predict lifecycle
```

No compatibility graph:

```text
removed aliases/wrappers -> update callers -> delete old names
```
