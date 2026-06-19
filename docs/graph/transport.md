# Transport Graph

Current implementation-facing graph for `qooi.transport`.

## Generic HTTP core

```text
qooi.transport.core.BaseHttpClient
qooi.transport.core.HttpError
qooi.transport.core.RetryPolicy
qooi.transport.core.request_json(...)
qooi.transport.core.request_json_value(...)
qooi.transport.core.request_json_sync(...)
qooi.transport.core.gather_requests(...)
qooi.transport.core.sanitize_error(...)
qooi.transport.core.sanitized_provider_message(...)
```

## OKX client

```text
qooi.transport.okx.OkxClient
qooi.transport.okx.OkxWsClient
qooi.transport.okx.okx_retry_policy()
qooi.transport.okx.collect_okx_ws_books(...)
```

Provider result records:

```text
SourceManifestRow
SourceResult
Manifest
```

Utility:

```text
qooi.transport.okx.now_ms() -> int
```

## Scanner/pipeline edge

```text
qooi.scanner.workflow.run(...)
  -> OkxClient()
  -> qooi.pipeline.discovery.rank_discovery(client, ...)
  -> qooi.pipeline.load.load_market(request, policy, client)
```

`OkxClient` owns exchange connection and provider request methods. Pipeline owns cache/load policy. Scanner owns state/outcome/evidence/rank/report semantics.

## Forbidden edges

```text
qooi.transport -> qooi.scanner
qooi.transport -> qooi.pipeline
qooi.transport -> qooi.strategies
qooi.transport -> qooi.core
```
