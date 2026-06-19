# Pipeline Graph

Current market-data load/cache graph used by the scanner.

## Public modules

```text
qooi.pipeline
qooi.pipeline.types
qooi.pipeline.io
qooi.pipeline.coverage
qooi.pipeline.discovery
qooi.pipeline.load
```

## Time helper

```text
qooi.pipeline.now_ms() -> int
```

## IO helpers

```text
qooi.pipeline.io.load_frame(path, schema=None) -> pl.DataFrame
qooi.pipeline.io.save_frame(path, frame) -> None
qooi.pipeline.io.merge_frames(existing, incoming, keys, max_rows=None) -> pl.DataFrame
```

Scanner convention:

```text
source/cache data -> parquet
user-facing reports/diagnostics -> csv/markdown
```

## Coverage planning

```text
qooi.pipeline.coverage.CoverageRunPolicy
qooi.pipeline.coverage.ProductCoverageSpec
qooi.pipeline.coverage.CoverageState
qooi.pipeline.coverage.CoverageJob
qooi.pipeline.coverage.CoveragePlan

qooi.pipeline.coverage.bar_spec(...)
qooi.pipeline.coverage.source_spec(...)
qooi.pipeline.coverage.coverage_spec(...)
qooi.pipeline.coverage.plan_product_coverage(...)
qooi.pipeline.coverage.allocate_coverage(...)
qooi.pipeline.coverage.coverage_state(...)
qooi.pipeline.coverage.coverage_summary(...)
```

Coverage decides what is complete, stale, provider-bounded, deferred, or too young. Scanner reports expose those states instead of hiding missing data.

## Discovery

```text
qooi.pipeline.discovery.rank_discovery(client, universe, limit) -> DiscoveryResult
qooi.pipeline.discovery.select_symbols(discovery, max_symbols) -> tuple[str, ...]
```

Scanner uses this before market load to choose the research universe.

## Market load

Request/config types:

```text
BarLoadRequest
SourceProductLoadRequest
SourceLoadRequest
MarketLoadRequest
MarketLoadPolicy
LoadStats
LoadedMarketFrames
```

Primary call:

```text
qooi.pipeline.load.load_market(request, policy, client) -> LoadedMarketFrames
```

Scanner boundary:

```text
qooi.scanner.workflow.scanner_market_request(config, symbols) -> MarketLoadRequest
qooi.scanner.workflow.scanner_market_policy(config) -> MarketLoadPolicy
qooi.scanner.workflow.run(...)
  -> load_market(request, policy, OkxClient)
```

Pipeline owns load/cache composition. `OkxClient` owns transport. Scanner owns how loaded frames become state/outcome/evidence products.
