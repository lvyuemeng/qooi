# Source Collector Module Graph

```text
qooi.sources.context
  load_source_context(config, symbols, context_symbols, discovery)
    -> read_source_bundle()
    -> optional provider fetch through exchange.context / sources.*
    -> merge_context_frames()
    -> write_context_frames()
    -> source_availability()

qooi.sources.bundle
  read_source_bundle()
  source_frame()
  source_symbols()
  missing_symbols()
  replace_symbol_rows()
  write_source_bundle()
  merge_source_artifact()
  merge_source_frames()

qooi.sources.artifacts
  artifact_path()
  read_frame_artifact()
  write_frame_artifact()
  write_text_artifact()

qooi.sources.coverage
  manifest_row_from_history_coverage()
  compute_source_coverage_score()
  missing_evidence_for_symbol()
  stale_symbols()
  eligible_fetch_symbols()

provider helpers
  coingecko.fetch_*()      -> normalize_*()
  coinpaprika.fetch_*()    -> normalize_*()
  cryptopanic.fetch_*()    -> normalize_*()
  defillama.fetch_*()      -> normalize_*()
  okx.fetch_*()
  okx_ws.collect_okx_ws_public()
  polymarket.fetch_*()     -> normalize_*()
  messages.normalize_local_messages()
```

Output contract:

```text
SourceResult(frame, manifest, warnings/errors)
  -> source bundle artifacts
  -> scanner source-state and freshness diagnostics
```
