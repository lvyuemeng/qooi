# Transport Architecture

## Purpose

`qooi.transport` owns network I/O and provider response shaping. It does not own scanner state, model evidence, ranking, caching policy, or trading decisions.

## Owned modules

```text
src/qooi/transport/
├── core.py   # generic HTTP helpers, retry policy, sanitized errors, gather_requests
└── okx.py    # OKX REST/WS client, source manifest/result rows, OKX source helpers
```

## Responsibilities

- wrap HTTP requests and retries;
- sanitize provider/API errors before they reach reports/logs;
- expose OKX client methods used by pipeline loading/discovery;
- preserve provider manifest rows for source provenance;
- provide websocket book collection when explicitly requested.

## Non-responsibilities

- no cache ownership;
- no scanner evidence/rank/report logic;
- no source-family feasibility scoring;
- no hardcoded API keys, wallet labels, or account actions.

## Dependency direction

```text
pipeline/scanner -> qooi.transport
qooi.transport   -> httpx/tenacity/polars/stdlib
qooi.transport   -> no scanner/pipeline imports
```

Transport is a lower-level boundary. Callers decide what to fetch and how to interpret returned frames.
